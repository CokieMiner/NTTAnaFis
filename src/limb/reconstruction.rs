//! Native-limb output blocks from exact coefficients.
//!
//! Cofactor reconstruction supplies each nonnegative convolution coefficient
//! `c_i`. For output stride w, the result is `sum(c_i*2^(i*w))`, so each
//! coefficient is added once at its bit position. With capacity `c`, the
//! partial sum through `c_i` is below `2^(i*w+c+1)`. Every catalog capacity
//! satisfies `c < 64*words`, where `words = ceil(c/64)`, so the shifted
//! coefficient and the carry of its sum stay in `words+1` words. An integer
//! product uses the chunk width as stride. A direct convolution uses a whole
//! number of limbs at least as wide as every coefficient, so its coefficients
//! occupy disjoint blocks that are stored without addition. A signed
//! convolution, of coefficients in `[-2^s, 2^s)`, reconstructs each coefficient
//! plus `2^s`, added to the residues beforehand, and subtracts `2^s` inside
//! its block, which leaves it in two's complement.

#![expect(
    unsafe_code,
    reason = "Admitted geometry proves residue indices, accumulator widths, and output window spans"
)]

use core::{mem::take, num::NonZeroUsize};

use crate::arch::{CoefficientKernel, Kernels};

use super::{
    CRT_LANES, Cofactors, Executor, Geometry, LIMB_BITS, Limb, MAX_PRIMES, PART_BITS, Partition,
    WORDS,
};

/// Coefficients of the smallest parallel reconstruction task.
const PARALLEL_GRAIN: usize = 256;

/// Coefficients between task cuts: a multiple of them occupies a whole native limb
/// count on every pointer width.
const PARALLEL_UNIT: usize = 64;

/// Native limbs per coefficient word.
const WORD_LIMBS: usize = 64 >> LIMB_BITS.trailing_zeros();

/// Largest window of a shifted coefficient, in limbs.
const WINDOW_LIMBS: usize = (WORDS + 1) * WORD_LIMBS;

/// `2^49`, below every catalog prime.
const PART_POWER: u64 = 1 << PART_BITS;

/// Exact reconstruction for any admitted prefix of the prime catalog.
#[derive(Debug)]
pub struct Crt;

struct Reconstruction<'residues> {
    residues: &'residues [&'residues [f64]; MAX_PRIMES],
    count: usize,
    width: NonZeroUsize,
    disjoint: bool,
    carry_words: usize,
    kernel: Option<CoefficientKernel>,
}

impl Crt {
    /// Reconstructs cofactor-weighted residues into the admitted native product span.
    ///
    /// The coefficients of a signed geometry, with sign bit `s`, lie in
    /// `[-2^s, 2^s)`. Adding the weighted residue of `2^s` to every residue
    /// first makes each an unsigned coefficient below `2^(s+1)`, which fits its
    /// block; subtracting `2^s` inside the block afterwards leaves it in two's
    /// complement.
    ///
    /// # Safety
    /// `geometry` is admitted by an immutable native plan. `rows` holds one row
    /// of `transform_len` slots per prime, whose first `conv_len` slots contain
    /// canonical u64 residue bit patterns. Each residue is multiplied by
    /// `(P/p_i)^-1 mod p_i`, where P is the product of the selected primes. The
    /// unweighted values encode the convolution of the admitted chunks; their
    /// storage is no longer interpreted as floating-point values.
    /// `dst` is exactly the native product span of that plan.
    /// The kernel table admits the current CPU.
    pub unsafe fn reconstruct<E: Executor>(
        dst: &mut [Limb],
        rows: &mut [f64],
        geometry: &Geometry,
        kernels: &Kernels,
        executor: &E,
    ) {
        let width = geometry.output_stride();
        let prefix = geometry.prefix();
        let length = geometry.transform_len().get();
        let inverses = &Cofactors::of(prefix).inverses;
        let mut residues: [&[f64]; MAX_PRIMES] = [&[]; MAX_PRIMES];
        let mut rest = rows;
        for ((slot, prime), &inverse) in residues.iter_mut().zip(prefix.primes()).zip(inverses) {
            // SAFETY: `rows` holds one row of `length` slots per prime of the prefix.
            let (row, tail) = unsafe { take(&mut rest).split_at_mut_unchecked(length) };
            rest = tail;
            if let Some(sign_bit) = geometry.sign_bit() {
                let modulus = prime.modulus();
                // SAFETY: 2^s is 2^(s mod 49) times s/49 factors 2^49, each below
                // every catalog prime, and the table's `inverse` is canonical in
                // `0..p`, so every product has canonical factors.
                let offset = unsafe {
                    let mut power =
                        1_u64.unchecked_shl(u32::try_from(sign_bit % PART_BITS).unwrap_unchecked());
                    for _ in 0..sign_bit.div_euclid(PART_BITS) {
                        power = prime.mul_mod(power, PART_POWER);
                    }
                    prime.mul_mod(power, inverse)
                };
                for value in row.iter_mut().take(geometry.conv_len()) {
                    // Both terms lie below p < 2^50; the wrapped difference
                    // exceeds the sum exactly when the sum is below p.
                    let sum = value.to_bits().wrapping_add(offset);
                    *value = f64::from_bits(sum.min(sum.wrapping_sub(modulus)));
                }
            }
            *slot = row;
        }
        // SAFETY: admission gives positive operand lengths and bounds the stream bits,
        // and the output bit count fits usize. All arrays cover the derived word counts.
        // A signed geometry is disjoint, so its output holds one block of the stride
        // per coefficient, and its sign bit lies below the stride, a positive whole
        // limb count, so its limb lies in every block and its shift below LIMB_BITS.
        unsafe {
            let terms = geometry.left().digits.min(geometry.right().digits);
            let term_bits = usize::BITS.unchecked_sub(terms.get().unchecked_sub(1).leading_zeros());
            let carry_words = width
                .get()
                .unchecked_add(usize::try_from(term_bits).unwrap_unchecked())
                .div_ceil(64);
            let coefficients = geometry
                .conv_len()
                .min(dst.len().unchecked_mul(LIMB_BITS).div_ceil(width.get()));
            let context = Reconstruction {
                residues: &residues,
                count: prefix.count(),
                width,
                disjoint: geometry.disjoint(),
                carry_words,
                kernel: kernels
                    .coefficients
                    .as_ref()
                    .map(|table| *prefix.select(table)),
            };
            let carry =
                reconstruct_blocks(dst, 0, coefficients, &context, executor.budget(), executor);
            debug_assert!(
                carry.iter().all(|&word| word == 0),
                "the product fits its native span"
            );
            if let Some(sign_bit) = geometry.sign_bit() {
                let block = width.get() >> LIMB_BITS.trailing_zeros();
                let (limb, bit) = (sign_bit.div_euclid(LIMB_BITS), sign_bit % LIMB_BITS);
                let mut start = 0_usize;
                for _ in 0..coefficients {
                    let mut borrow = Limb::unchecked_shl(1, u32::try_from(bit).unwrap_unchecked());
                    for slot in
                        dst.get_unchecked_mut(start.unchecked_add(limb)..start.unchecked_add(block))
                    {
                        let (difference, wrapped) = slot.overflowing_sub(borrow);
                        *slot = difference;
                        borrow = Limb::from(wrapped);
                    }
                    start = start.unchecked_add(block);
                }
            }
        }
    }
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Each shifted u64 supplies its low native limb on 16-, 32-, and 64-bit targets"
)]
fn reconstruct_blocks<E: Executor>(
    dst: &mut [Limb],
    first: usize,
    coefficients: usize,
    context: &Reconstruction<'_>,
    budget: NonZeroUsize,
    executor: &E,
) -> [u64; WORDS] {
    if let Some(split) = Partition::range::<PARALLEL_UNIT, PARALLEL_GRAIN>(0, coefficients, budget)
    {
        // Each child keeps at least a grain of coefficients, so its output also
        // contains the complete carry.
        let middle = split.middle;
        // SAFETY: the cut is a whole limb count inside the coefficient span, and
        // admission bounds its bits.
        let (left, right) = unsafe {
            dst.split_at_mut_unchecked(
                middle
                    .unchecked_mul(context.width.get())
                    .div_euclid(LIMB_BITS),
            )
        };
        let (incoming, mut outgoing) = executor.join(
            || reconstruct_blocks(left, first, middle, context, split.left, executor),
            || {
                // SAFETY: middle lies inside this admitted coefficient span.
                unsafe {
                    reconstruct_blocks(
                        right,
                        first.unchecked_add(middle),
                        coefficients.unchecked_sub(middle),
                        context,
                        split.right,
                        executor,
                    )
                }
            },
        );
        // The local polynomial has value low + 2^output_bits*outgoing.
        // Adding the preceding block's carry changes low and increments
        // outgoing exactly when that addition crosses the output boundary.
        let mut carry = false;
        // SAFETY: each right child has at least 191*width output bits. Its
        // incoming carry uses at most width + usize::BITS bits, rounded to u64
        // words, and 191*width >= 64*ceil((width + 64)/64) for every positive
        // width; this prefix therefore fits the right output slice.
        unsafe {
            let native_words = context.carry_words.unchecked_mul(WORD_LIMBS);
            for (index, slot) in right
                .get_unchecked_mut(..native_words)
                .iter_mut()
                .enumerate()
            {
                let position = index.unchecked_mul(LIMB_BITS);
                let word = *incoming.get_unchecked(position >> 6) >> (position % 64);
                (*slot, carry) = slot.carrying_add(word as Limb, carry);
            }
            for slot in right.get_unchecked_mut(native_words..) {
                if !carry {
                    break;
                }
                (*slot, carry) = slot.overflowing_add(1);
            }
            for word in &mut outgoing {
                if !carry {
                    break;
                }
                (*word, carry) = word.overflowing_add(1);
            }
            debug_assert!(
                !carry,
                "the admitted carry bound fits the fixed accumulator"
            );
        }
        outgoing
    } else {
        const {
            assert!(
                MAX_PRIMES == 8,
                "the reconstruction dispatch covers every catalog prefix"
            );
        };
        if context.count == 1 && context.disjoint {
            // SAFETY: a disjoint output holds one block per coefficient, and
            // the residue row covers every index. A block holds a coefficient
            // below 2^49, so each shifted limb starts below bit 64.
            unsafe { store_residues(dst, first, coefficients, context) };
            return [0; WORDS];
        }
        match context.count {
            1 => reconstruct_range::<1>(dst, first, coefficients, context),
            2 => reconstruct_range::<2>(dst, first, coefficients, context),
            3 => reconstruct_range::<3>(dst, first, coefficients, context),
            4 => reconstruct_range::<4>(dst, first, coefficients, context),
            5 => reconstruct_range::<5>(dst, first, coefficients, context),
            6 => reconstruct_range::<6>(dst, first, coefficients, context),
            7 => reconstruct_range::<7>(dst, first, coefficients, context),
            _ => reconstruct_range::<8>(dst, first, coefficients, context),
        }
    }
}

/// Adds the coefficients of one block at their bit positions and returns the
/// value above the block's limbs.
fn reconstruct_range<const COUNT: usize>(
    dst: &mut [Limb],
    first: usize,
    coefficients: usize,
    context: &Reconstruction<'_>,
) -> [u64; WORDS] {
    let window = const {
        let capacity = Cofactors::of_count(COUNT).capacity;
        assert!(
            !capacity.is_multiple_of(64),
            "a partial sum stays below its coefficient window"
        );
        (capacity.div_ceil(64) + 1) * WORD_LIMBS
    };
    let width = context.width.get();
    let length = dst.len();
    dst.fill(0);
    // SAFETY: the driver partitions admitted coefficient and output spans. A
    // coefficient starting at word k occupies limbs k*WORD_LIMBS..+window. It
    // lies in dst while k*WORD_LIMBS+window <= length, which the direct count
    // establishes. The tail buffer holds the remaining dst limbs, shorter than
    // one window, and each later coefficient starts below the end of dst, so
    // its window fits 2*WINDOW_LIMBS limbs. Packet kernels write words+1
    // columns of complete packets inside the coefficient range, and scalar
    // coefficients write words entries of residue rows covering every index.
    unsafe {
        let direct = if length >= window {
            length
                .unchecked_sub(window)
                .div_euclid(WORD_LIMBS)
                .unchecked_add(1)
                .unchecked_mul(64)
                .div_ceil(width)
                .min(coefficients)
        } else {
            0
        };
        let output = dst.as_mut_ptr();
        let mut index = 0;
        if let Some(kernel) = context.kernel {
            let mut columns = [[0_u64; CRT_LANES]; WORDS + 1];
            while direct.unchecked_sub(index) >= CRT_LANES {
                let bit = index.unchecked_mul(width);
                kernel(
                    context.residues,
                    first.unchecked_add(index),
                    bit,
                    width,
                    &mut columns,
                );
                for lane in 0..CRT_LANES {
                    let position = bit.unchecked_add(lane.unchecked_mul(width)) >> 6;
                    add_words::<COUNT>(output.add(position.unchecked_mul(WORD_LIMBS)), |word| {
                        *columns.get_unchecked(word).get_unchecked(lane)
                    });
                }
                index = index.unchecked_add(CRT_LANES);
            }
        }
        let mut value = [0_u64; WORDS + 1];
        while index < direct {
            Cofactors::bounded::<COUNT>(context.residues, first.unchecked_add(index), &mut value);
            add_shifted::<COUNT>(output, index.unchecked_mul(width), &value);
            index = index.unchecked_add(1);
        }
        let base = if index < coefficients {
            (index.unchecked_mul(width) >> 6).unchecked_mul(WORD_LIMBS)
        } else {
            length
        };
        let mut tail = [0; 2 * WINDOW_LIMBS];
        let kept = length.unchecked_sub(base);
        tail.get_unchecked_mut(..kept)
            .copy_from_slice(dst.get_unchecked(base..));
        while index < coefficients {
            Cofactors::bounded::<COUNT>(context.residues, first.unchecked_add(index), &mut value);
            add_shifted::<COUNT>(
                tail.as_mut_ptr(),
                index
                    .unchecked_mul(width)
                    .unchecked_sub(base.unchecked_mul(LIMB_BITS)),
                &value,
            );
            index = index.unchecked_add(1);
        }
        dst.get_unchecked_mut(base..)
            .copy_from_slice(tail.get_unchecked(..kept));
        let mut outgoing = [0; WORDS];
        for (place, &limb) in tail.get_unchecked(kept..).iter().enumerate() {
            let position = place.unchecked_mul(LIMB_BITS);
            if let Some(word) = outgoing.get_mut(position >> 6) {
                #[expect(
                    clippy::as_conversions,
                    reason = "Native limbs have at most 64 bits, so widening to u64 is exact"
                )]
                {
                    *word |= (limb as u64) << (position % 64);
                }
            }
        }
        outgoing
    }
}

/// Writes coefficients of one prime to their disjoint blocks and zeros the rest.
///
/// With one prime `P/p_1 = 1`, so each coefficient is its canonical residue, and
/// a coefficient that fits its block needs neither a quotient nor an addition.
///
/// # Safety
/// `dst` holds a block of `width/LIMB_BITS` limbs per coefficient, the first
/// residue row covers `first..first+coefficients`, and each shift amount
/// `part*LIMB_BITS` for a limb of a block is below 64.
#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Each shifted u64 supplies its low native limb on 16-, 32-, and 64-bit targets"
)]
unsafe fn store_residues(
    dst: &mut [Limb],
    first: usize,
    coefficients: usize,
    context: &Reconstruction<'_>,
) {
    // SAFETY: the caller bounds the block span, residue indices, and shifts. A
    // disjoint stride is a positive whole limb count.
    unsafe {
        let block = NonZeroUsize::new_unchecked(context.width.get() >> LIMB_BITS.trailing_zeros());
        let (blocks, rest) = dst.split_at_mut_unchecked(coefficients.unchecked_mul(block.get()));
        rest.fill(0);
        let row = context
            .residues
            .get_unchecked(0)
            .get_unchecked(first..first.unchecked_add(coefficients));
        if block == NonZeroUsize::MIN {
            for (slot, value) in blocks.iter_mut().zip(row) {
                *slot = value.to_bits() as Limb;
            }
            return;
        }
        for (target, value) in blocks.chunks_exact_mut(block.get()).zip(row) {
            let residue = value.to_bits();
            for (part, slot) in target.iter_mut().enumerate() {
                let shift = part.unchecked_mul(LIMB_BITS);
                *slot = residue.unchecked_shr(u32::try_from(shift).unwrap_unchecked()) as Limb;
            }
        }
    }
}

/// Adds one coefficient of the prefix of `COUNT` primes, `words = ceil(c/64)`
/// words for its capacity `c`, to the output at bit `bit`.
///
/// # Safety
/// The window of `words+1` shifted words lies in the output, and the sum fits
/// that window.
#[inline]
unsafe fn add_shifted<const COUNT: usize>(output: *mut Limb, bit: usize, value: &[u64; WORDS + 1]) {
    let words = const { Cofactors::of_count(COUNT).capacity.div_ceil(64) };
    // SAFETY: bit%64 is below 64, so both shift amounts are below 64, and a
    // zero offset shifts the previous word out completely. The caller bounds
    // the window, and word indices stay in `words+1 <= WORDS+1`.
    unsafe {
        let offset = u32::try_from(bit & 63).unwrap_unchecked();
        let inverse = 63_u32.unchecked_sub(offset);
        let mut shifted = [0_u64; WORDS + 1];
        let mut previous = 0_u64;
        for word in 0..words {
            let current = *value.get_unchecked(word);
            *shifted.get_unchecked_mut(word) =
                current.unchecked_shl(offset) | (previous >> 1).unchecked_shr(inverse);
            previous = current;
        }
        *shifted.get_unchecked_mut(words) = (previous >> 1).unchecked_shr(inverse);
        add_words::<COUNT>(output.add((bit >> 6).unchecked_mul(WORD_LIMBS)), |word| {
            *shifted.get_unchecked(word)
        });
    }
}

/// Adds the shifted words of a coefficient of the prefix of `COUNT` primes.
///
/// # Safety
/// The output covers those words as limbs, and the sum fits them.
#[inline]
#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Each shifted u64 supplies its low native limb on 16-, 32-, and 64-bit targets"
)]
unsafe fn add_words<const COUNT: usize>(output: *mut Limb, word: impl Fn(usize) -> u64) {
    let words = const { Cofactors::of_count(COUNT).capacity.div_ceil(64) + 1 };
    let mut carry = false;
    // SAFETY: the caller bounds every limb. Shift amounts stay below 64.
    unsafe {
        for index in 0..words {
            let current = word(index);
            for part in 0..WORD_LIMBS {
                let slot = output.add(index.unchecked_mul(WORD_LIMBS).unchecked_add(part));
                let limb = current
                    .unchecked_shr(u32::try_from(part.unchecked_mul(LIMB_BITS)).unwrap_unchecked());
                (*slot, carry) = (*slot).carrying_add(limb as Limb, carry);
            }
        }
    }
    debug_assert!(!carry, "partial sums stay below their coefficient window");
}
