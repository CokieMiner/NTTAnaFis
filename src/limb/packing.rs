//! Shared binary-digit extraction and SIMD reduction across prime fields.
//!
//! Every coefficient splits into parts of at most 49 bits. A part at bit
//! position `b` lies in the 64-bit window starting at byte `floor(b/8)`,
//! because `b mod 8 + 49 <= 56`, so little-endian targets extract each part
//! with one unaligned load, one shift, and one mask.

#![expect(
    unsafe_code,
    reason = "Admitted geometry proves native reads and disjoint coefficient ranges in each output row"
)]

use core::{num::NonZeroUsize, ops::Div};

use crate::arch::{BIAS, COLUMN_ALIGN, Columns, Kernels, PackingKernel};

use super::{
    Executor, Geometry, LIMB_BITS, Limb, MAX_PRIMES, MODULUS_BITS, Operand, PRIME_CATALOG,
    Partition, PrimePrefix,
};

/// Bits per binary part; every part stays below each catalog prime.
pub const PART_BITS: usize = MODULUS_BITS - 1;
const PART_MASK: u64 = (1_u64 << PART_BITS) - 1;

/// Most parts of one coefficient. A capacity below `MODULUS_BITS*np` admits
/// coefficients of half its bits.
///
/// A digit and three centered products stay below `4*p`, so the packing kernels
/// reduce every coefficient once.
pub const MAX_PARTS: usize = const {
    assert!(
        MAX_PRIMES.div_ceil(2) <= 4,
        "packing sums at most four parts"
    );
    MAX_PRIMES.div_ceil(2)
};

/// Lane-padded length of the prime parameters.
const COLUMNS: usize = MAX_PRIMES.next_multiple_of(COLUMN_ALIGN);

/// Binary powers and prime parameters that every packing kernel reads.
pub static PACKING_DATA: PackingData = packing_data();

/// Coefficients in one stack-resident binary-digit tile, so tile starts are
/// column-aligned.
pub const PACKING_TILE: usize = const {
    assert!(
        32_usize.is_multiple_of(COLUMN_ALIGN),
        "a tile holds complete vectors"
    );
    32
};

/// One row of [`PACKING_TILE`] digits per part.
pub type DigitTile = [f64; MAX_PARTS * PACKING_TILE];

/// Coefficients of the smallest parallel packing task, in whole tiles.
const PARALLEL_GRAIN: usize = 8 * PACKING_TILE;

/// Immutable binary powers and prime parameters, padded to SIMD lane widths.
#[derive(Debug)]
pub struct PackingData {
    /// Exact floating-point prime moduli.
    pub moduli: [f64; COLUMNS],
    /// Rounded reciprocals of the prime moduli.
    pub reciprocals: [f64; COLUMNS],
    /// Centered residues of successive powers of `2^49`.
    pub powers: [[f64; COLUMNS]; MAX_PARTS],
}

/// Shared input and disjoint output rows for one admitted packing operation.
#[derive(Debug)]
pub struct Packing<'input> {
    /// First writable coefficient of the first prime row.
    pub output: *mut f64,
    /// Distance between consecutive prime rows.
    pub stride: usize,
    /// The prime rows.
    pub prefix: PrimePrefix,
    input: &'input [Limb],
    width: usize,
    parts: usize,
    high_bits: usize,
    windowed: usize,
    kernel: PackingKernel,
}

// SAFETY: packing tasks read immutable input and tables. They write disjoint
// coefficient intervals in every prime row. Synchronous executor joins finish
// before the exclusive output borrow ends or any subsequent stage reads it.
unsafe impl Sync for Packing<'_> {}

impl<'input> Packing<'input> {
    /// Fills the first `operand.columns` columns of every prime row, sharing
    /// extraction across the rows.
    ///
    /// Each coefficient is an exact integer of magnitude below its prime modulus,
    /// and missing ones are zero. The suffix beyond the columns is untouched and need
    /// not contain zero: the first pruned forward stage reads only the initialized
    /// prefix.
    ///
    /// # Safety
    /// `output` holds one row of the geometry per prime,
    /// `geometry.arena_floats(true)` floats. `input` holds coefficients of
    /// `operand.value_bits` significant bits at multiples of the chunk width, at most
    /// `operand.digits` of them, and `operand` is an operand of the geometry.
    pub unsafe fn pack<E: Executor>(
        output: &mut [f64],
        (input, operand): (&'input [Limb], Operand),
        geometry: &Geometry,
        kernels: &Kernels,
        executor: &E,
    ) {
        let width = geometry.chunk_bits();
        let stride = geometry.transform_len();
        debug_assert!(
            output.len() == geometry.arena_floats(true),
            "one row per prime"
        );
        // SAFETY: admission bounds the operand's bits, and value bits are positive
        // and fit MAX_PARTS parts because they are at most the admitted chunk width,
        // so each kernel table has an entry for the part count. The caller gives at
        // most the operand's digits, whose rounded count fits the operand's columns,
        // which fit every row.
        unsafe {
            let bits = input.len().unchecked_mul(LIMB_BITS);
            let digits = bits.div_ceil(width.get());
            debug_assert!(digits <= operand.digits.get(), "the input fits the operand");
            let value_bits = operand.value_bits.get();
            let parts = value_bits.div_ceil(PART_BITS);
            let high_part = parts.unchecked_sub(1).unchecked_mul(PART_BITS);
            let high_bits = value_bits.unchecked_sub(high_part);
            // Coefficient i needs windows up to bit i*width + high_part, each
            // ending strictly inside the input. Later coefficients use checked reads.
            let last_window = high_part.unchecked_add(64);
            let windowed = if bits > last_window {
                bits.unchecked_sub(last_window)
                    .unchecked_sub(1)
                    .div(width)
                    .unchecked_add(1)
            } else {
                0
            };
            let packing = Self {
                output: output.as_mut_ptr(),
                stride: stride.get(),
                prefix: geometry.prefix(),
                input,
                width: width.get(),
                parts,
                high_bits,
                windowed,
                kernel: if parts > 1 && high_bits <= 3 {
                    *kernels.pack_short.get_unchecked(parts.unchecked_sub(2))
                } else {
                    *kernels.pack.get_unchecked(parts.unchecked_sub(1))
                },
            };
            pack_ranges(&packing, 0, digits, executor.budget(), executor);
            // Packing wrote the rows through a raw pointer that ended above.
            let packed = digits.next_multiple_of(COLUMN_ALIGN);
            if packed < operand.columns {
                for row in output.chunks_exact_mut(stride.get()) {
                    row.get_unchecked_mut(packed..operand.columns).fill(0.0);
                }
            }
        }
    }
}

fn pack_ranges<E: Executor>(
    packing: &Packing<'_>,
    first: usize,
    end: usize,
    budget: NonZeroUsize,
    executor: &E,
) {
    // Whole tiles leave only the final partial tile of the complete operation.
    if let Some(split) = Partition::range::<PACKING_TILE, PARALLEL_GRAIN>(first, end, budget) {
        executor.join(
            || pack_ranges(packing, first, split.middle, split.left, executor),
            || pack_ranges(packing, split.middle, end, split.right, executor),
        );
        return;
    }
    // SAFETY: the driver partitions 0..digits into disjoint intervals. Ordered
    // endpoints bound all subtractions; admitted input and output spans bound
    // positions. Only the final tile of the complete operation can be partial, and
    // its rounded columns fit the admitted row.
    unsafe {
        let mut digits: DigitTile = [0.0; MAX_PARTS * PACKING_TILE];
        let mut position = first;
        while position < end {
            let len = end.unchecked_sub(position).min(PACKING_TILE);
            let columns = extract_tile(&mut digits, packing, position, len);
            (packing.kernel)(
                packing,
                Columns::new(position, position.unchecked_add(columns)),
                &digits,
            );
            position = position.unchecked_add(len);
        }
    }
}

/// Fills the digit rows of `first..first+len` and returns the padded column count.
///
/// Each row holds exact integers below `2^49`, one part per coefficient. A
/// masked part occupies the low mantissa bits of `d + 2^52`, so the biased
/// pattern minus `2^52` is `d` exactly, without a 64-bit integer conversion. A
/// chunk of one whole limb reads that limb and needs no bit window.
#[expect(
    clippy::as_conversions,
    reason = "Native limbs have at most 64 bits, so widening to u64 is exact"
)]
unsafe fn extract_tile(
    digits: &mut DigitTile,
    packing: &Packing<'_>,
    first: usize,
    len: usize,
) -> usize {
    // SAFETY: parts <= MAX_PARTS and len <= PACKING_TILE bound the stack tile,
    // and columns <= PACKING_TILE because the tile length is a multiple of
    // eight. Coefficients below packing.windowed admit their 64-bit windows;
    // later reads are checked and zero-extended. A whole-limb chunk has one
    // limb per coefficient in the input, and its part offsets lie below its
    // value bits, at most 64. The final part mask has high_bits in 1..=49, so
    // its shift amount is below 49.
    unsafe {
        let windowed = packing.windowed.saturating_sub(first).min(len);
        let columns = len.next_multiple_of(COLUMN_ALIGN);
        let high_part = packing.parts.unchecked_sub(1);
        let high_mask = PART_MASK.unchecked_shr(
            u32::try_from(PART_BITS.unchecked_sub(packing.high_bits)).unwrap_unchecked(),
        );
        let start = first.unchecked_mul(packing.width);
        for part in 0..packing.parts {
            let mask = if part == high_part {
                high_mask
            } else {
                PART_MASK
            };
            let row = digits.as_mut_ptr().add(part.unchecked_mul(PACKING_TILE));
            let offset = part.unchecked_mul(PART_BITS);
            if packing.width == LIMB_BITS {
                let shift = u32::try_from(offset).unwrap_unchecked();
                let limbs = packing.input.get_unchecked(first..first.unchecked_add(len));
                for (index, &limb) in limbs.iter().enumerate() {
                    let word = (limb as u64).unchecked_shr(shift) & mask;
                    *row.add(index) = f64::from_bits(word | BIAS.to_bits()) - BIAS;
                }
            } else {
                let mut position = start.unchecked_add(offset);
                for index in 0..windowed {
                    let word = window(packing.input, position) & mask;
                    *row.add(index) = f64::from_bits(word | BIAS.to_bits()) - BIAS;
                    position = position.unchecked_add(packing.width);
                }
                for index in windowed..len {
                    let word = checked_window(packing.input, position) & mask;
                    *row.add(index) = f64::from_bits(word | BIAS.to_bits()) - BIAS;
                    position = position.unchecked_add(packing.width);
                }
            }
            for index in len..columns {
                *row.add(index) = 0.0;
            }
        }
        columns
    }
}

/// Returns at least 49 valid low bits starting at a bit position whose 64-bit
/// window ends strictly inside the input.
///
/// # Safety
/// `position + 64 < input.len() * LIMB_BITS`.
#[cfg(target_endian = "little")]
#[inline]
unsafe fn window(input: &[Limb], position: usize) -> u64 {
    // SAFETY: the bytes of a little-endian limb array form the integer's byte
    // sequence. floor(position/8)*8 + 64 <= position + 64 bounds the load, the
    // read has no alignment requirement, and the shift amount is below eight.
    unsafe {
        let bytes = input.as_ptr().cast::<u8>().add(position >> 3);
        bytes
            .cast::<u64>()
            .read_unaligned()
            .unchecked_shr(u32::try_from(position & 7).unwrap_unchecked())
    }
}

/// Returns at least 49 valid low bits starting at a bit position whose 64-bit
/// window ends strictly inside the input.
///
/// # Safety
/// `position + 64 < input.len() * LIMB_BITS`.
#[cfg(target_endian = "big")]
#[inline]
#[expect(
    clippy::as_conversions,
    reason = "Native limbs have at most 64 bits, so widening to u64 is exact"
)]
unsafe fn window(input: &[Limb], position: usize) -> u64 {
    if LIMB_BITS != 64 {
        return checked_window(input, position);
    }
    // SAFETY: with 64-bit limbs, position + 64 < limbs*64 bounds both reads.
    // The two-step right shift gives zero for a zero offset, so both shift
    // amounts stay below 64.
    unsafe {
        let source = position >> 6;
        let offset = u32::try_from(position & 63).unwrap_unchecked();
        let low = *input.get_unchecked(source) as u64;
        let high = *input.get_unchecked(source.unchecked_add(1)) as u64;
        low.unchecked_shr(offset) | (high << 1).unchecked_shl(63_u32.unchecked_sub(offset))
    }
}

/// Returns 49 low bits at a bit position, zero-extending beyond the input.
#[expect(
    clippy::as_conversions,
    reason = "Native limbs have at most 64 bits, so widening to u64 is exact"
)]
fn checked_window(limbs: &[Limb], position: usize) -> u64 {
    let index = position.div_euclid(LIMB_BITS);
    let offset = position % LIMB_BITS;
    // SAFETY: offset < LIMB_BITS <= 64 and each later piece starts below
    // bit 49 of the result, so every shift is strictly below 64. Admitted
    // positions bound all index additions on every native pointer width.
    unsafe {
        let first = limbs.get(index).copied().unwrap_or(0);
        let mut value = (first as u64).unchecked_shr(u32::try_from(offset).unwrap_unchecked());
        let pieces = offset.unchecked_add(PART_BITS).div_ceil(LIMB_BITS);
        for piece in 1..pieces {
            let source = index.unchecked_add(piece);
            let limb = limbs.get(source).copied().unwrap_or(0);
            let shift = piece.unchecked_mul(LIMB_BITS).unchecked_sub(offset);
            value |= (limb as u64).unchecked_shl(u32::try_from(shift).unwrap_unchecked());
        }
        value
    }
}

#[expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "Catalog moduli and canonical powers are below 2^50 and convert exactly to f64"
)]
const fn packing_data() -> PackingData {
    let mut data = PackingData {
        moduli: [0.0; COLUMNS],
        reciprocals: [0.0; COLUMNS],
        powers: [[0.0; COLUMNS]; MAX_PARTS],
    };
    let mut column = 0;
    // SAFETY: the nonempty catalog admits each padded column by cycling valid
    // primes. Indices stay inside their array bounds. A canonical 50-bit
    // residue shifted by 49 uses at most 99 bits and fits u128.
    unsafe {
        while column < COLUMNS {
            let prime = &*PRIME_CATALOG.as_ptr().add(column % MAX_PRIMES);
            *data.moduli.as_mut_ptr().add(column) = prime.float_field().p();
            *data.reciprocals.as_mut_ptr().add(column) = prime.float_field().pinv();
            let modulus = prime.modulus() as u128;
            let mut power = 1_u128;
            let mut part = 0;
            while part < MAX_PARTS {
                let centered = if power > modulus >> 1 {
                    power as f64 - prime.float_field().p()
                } else {
                    power as f64
                };
                *(*data.powers.as_mut_ptr().add(part))
                    .as_mut_ptr()
                    .add(column) = centered;
                power = (power << PART_BITS).rem_euclid(modulus);
                part = part.unchecked_add(1);
            }
            column = column.unchecked_add(1);
        }
    }
    data
}
