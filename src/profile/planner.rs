//! Admitted geometry of one convolution, and its selection by estimated work.
//!
//! A [`Geometry`] is the proof later stages rely on. Its fields are private,
//! and `admit` is the only constructor, so a value always holds:
//!
//! - **Primes**: a nonempty catalog prefix whose capacity reaches every result
//!   coefficient. Every catalog prime has the roots of the table.
//! - **Operands**: each has a positive number of coefficients of `1..=chunk_bits`
//!   significant bits, and `chunk_bits <= MAX_CHUNK_BITS`.
//! - **Row**: its cyclic part `N` is a power of two of at least `MIN_TRANSFORM_LEN`,
//!   and the row has `N` or `3*N/2` points, so it holds complete vectors. Operand
//!   columns fit `N`, and output columns fit the row.
//! - **Output**: the stride is positive. A disjoint stride is a whole limb count that
//!   holds every coefficient.
//! - **Sizes**: both spectra in floats and every bit position of the operands and the
//!   output stream fit `usize`.

#![expect(
    unsafe_code,
    reason = "Admission proves the column ranges that a geometry hands out"
)]

use core::num::NonZeroUsize;

use super::{
    COLUMN_ALIGN, Cofactors, Columns, LIMB_BITS, MAX_CHUNK_BITS, MAX_ROOT_ORDER, MIN_TRANSFORM_LEN,
    PROFILES, PrimePrefix, Profile, WORDS,
};

/// Bits of a native limb, the coefficient width of a direct convolution.
const LIMB_WIDTH: NonZeroUsize = match NonZeroUsize::new(LIMB_BITS) {
    Some(bits) if matches!(bits.get(), 16 | 32 | 64) => bits,
    _ => panic!("native limbs have 16, 32, or 64 bits"),
};

/// Split of the longer operand into blocks that share one spectrum of the shorter.
///
/// The result is the sum of the products of each block with the shorter operand,
/// each added at the offset of its block. A geometry with a split describes one
/// block, whose left operand is the block and whose right operand is the shorter
/// operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Split {
    /// Limbs of the longer operand in each block, at least the shorter length.
    pub block_limbs: usize,
    /// Output limbs between the windows of consecutive blocks.
    pub step_limbs: usize,
    /// Output limbs that consecutive windows share.
    pub overlap_limbs: usize,
    /// Whether the right operand of the operation is the longer one.
    pub swapped: bool,
}

/// Coefficients of one admitted operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Operand {
    /// Coefficients of the complete operand.
    pub digits: NonZeroUsize,
    /// Columns its packing initializes: `digits` rounded up to complete vectors.
    pub columns: usize,
    /// Significant bits of each coefficient.
    pub value_bits: NonZeroUsize,
}

/// Admitted primes, operand, row, and output lengths of one convolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    prefix: PrimePrefix,
    chunk_bits: NonZeroUsize,
    left: Operand,
    right: Operand,
    conv_len: usize,
    output_columns: usize,
    transform_len: NonZeroUsize,
    cyclic_len: usize,
    table_len: NonZeroUsize,
    output_stride: NonZeroUsize,
    disjoint: bool,
    split: Option<Split>,
    spectrum_floats: usize,
}

impl Geometry {
    /// Selects the fewest primes for a direct convolution of limb coefficients.
    ///
    /// Each operand holds one coefficient per native limb, so no coefficient is
    /// split. `bound_bits` is the bit length of an upper bound of every result
    /// coefficient, and the selected capacity reaches it. `output_stride` is a
    /// whole limb count of at least `bound_bits` bits. The transform length is
    /// fixed by the convolution length, so fewer primes always cost less.
    /// Returns `None` when `admit` rejects the shape or no capacity reaches the bound.
    #[must_use]
    pub fn for_residues(
        (len_a, len_b): (NonZeroUsize, NonZeroUsize),
        (value_bits_a, value_bits_b): (NonZeroUsize, NonZeroUsize),
        (bound_bits, output_stride): (usize, usize),
    ) -> Option<Self> {
        let selected = PROFILES
            .into_iter()
            .find(|profile| Cofactors::of(profile.prefix).capacity >= bound_bits)?;
        Self::admit(
            Profile {
                prefix: selected.prefix,
                chunk_bits: LIMB_WIDTH,
            },
            ((len_a, value_bits_a), (len_b, value_bits_b)),
            (bound_bits, output_stride),
            true,
        )
    }

    /// Selects a plan by estimated transform and reconstruction work.
    ///
    /// Every nonempty catalog prefix is considered, each at the widest coefficient
    /// it holds. Ties prefer fewer primes. `crt_cost` is the cost of one CRT term in
    /// points of one transform level. Returns `None` for overflowing sizes or
    /// unavailable transform orders.
    #[must_use]
    pub fn select(len_a: NonZeroUsize, len_b: NonZeroUsize, crt_cost: u8) -> Option<Self> {
        let bits_a = len_a.checked_mul(LIMB_WIDTH)?;
        let bits_b = len_b.checked_mul(LIMB_WIDTH)?;
        let mut best: Option<(u128, Self)> = None;
        for profile in PROFILES {
            if let Some(candidate) = Self::widest_under(bits_a, bits_b, profile) {
                let (transform, crt) = candidate.work(crt_cost)?;
                let score = transform.checked_mul(3)?.checked_add(crt)?;
                if best.is_none_or(|(cost, _)| score < cost) {
                    best = Some((score, candidate));
                }
            }
        }
        best.map(|(_, candidate)| candidate)
    }

    /// Narrows the profile's coefficient until its capacity holds the products, and
    /// admits that shape.
    ///
    /// A narrower coefficient only lengthens the row, so a shape rejected for any
    /// other reason stays rejected.
    fn widest_under(
        bits_a: NonZeroUsize,
        bits_b: NonZeroUsize,
        mut profile: Profile,
    ) -> Option<Self> {
        loop {
            let width = profile.chunk_bits;
            let terms = bits_a.div_ceil(width).min(bits_b.div_ceil(width));
            let excess = product_bits(width.get(), terms)?
                .saturating_sub(Cofactors::of(profile.prefix).capacity);
            if excess == 0 {
                return Self::under(bits_a, bits_b, profile);
            }
            profile.chunk_bits = NonZeroUsize::new(width.get().checked_sub(excess.div_ceil(2))?)?;
        }
    }

    /// Admits operands of the given bit lengths under one profile as it stands.
    ///
    /// Returns `None` when the profile's capacity does not hold the coefficients or
    /// `admit` rejects the shape.
    #[must_use]
    pub fn under(bits_a: NonZeroUsize, bits_b: NonZeroUsize, profile: Profile) -> Option<Self> {
        let width = profile.chunk_bits;
        let (digits_a, digits_b) = (bits_a.div_ceil(width), bits_b.div_ceil(width));
        Self::admit(
            profile,
            ((digits_a, width), (digits_b, width)),
            (
                product_bits(width.get(), digits_a.min(digits_b))?,
                width.get(),
            ),
            false,
        )
    }

    /// Establishes every property in the module documentation, or returns `None`.
    ///
    /// Each operand is `(digits, value_bits)`, and `coefficient_bits` bounds every
    /// result coefficient.
    fn admit(
        profile: Profile,
        ((digits_a, value_bits_a), (digits_b, value_bits_b)): (
            (NonZeroUsize, NonZeroUsize),
            (NonZeroUsize, NonZeroUsize),
        ),
        (coefficient_bits, output_stride): (usize, usize),
        disjoint: bool,
    ) -> Option<Self> {
        let chunk = profile.chunk_bits;
        let holds = |stride: usize| stride.is_multiple_of(LIMB_BITS) && stride >= coefficient_bits;
        let longest = digits_a.max(digits_b);
        let conv_len = digits_a.get().checked_add(digits_b.get())?.checked_sub(1)?;
        // Later stages compute these bit positions without checks: an operand's
        // limbs and windows, and a coefficient with the words above it.
        if chunk.get() > MAX_CHUNK_BITS
            || value_bits_a > chunk
            || value_bits_b > chunk
            || coefficient_bits > Cofactors::of(profile.prefix).capacity
            || (disjoint && !holds(output_stride))
            || longest.checked_mul(profile.chunk_bits).is_none()
            || conv_len
                .checked_mul(output_stride)
                .and_then(|bits| bits.checked_add(WORDS * 64))
                .is_none()
        {
            return None;
        }
        let (transform_len, cyclic_len, table_len) = row_shape(conv_len, longest.get())?;
        // Roots of unity of order `2*table_len`, a power of two, exist in every catalog
        // prime, so the table serves every prefix.
        if table_len.checked_mul(2)?.trailing_zeros() > MAX_ROOT_ORDER {
            return None;
        }
        // Both spectra of a product fit usize.
        let spectrum_floats = profile
            .prefix
            .count()
            .checked_mul(transform_len)
            .filter(|floats| *floats <= usize::MAX >> 1)?;
        Some(Self {
            prefix: profile.prefix,
            chunk_bits: profile.chunk_bits,
            left: Operand {
                digits: digits_a,
                columns: digits_a.get().checked_next_multiple_of(COLUMN_ALIGN)?,
                value_bits: value_bits_a,
            },
            right: Operand {
                digits: digits_b,
                columns: digits_b.get().checked_next_multiple_of(COLUMN_ALIGN)?,
                value_bits: value_bits_b,
            },
            conv_len,
            output_columns: conv_len.checked_next_multiple_of(COLUMN_ALIGN)?,
            transform_len: NonZeroUsize::new(transform_len)?,
            cyclic_len,
            table_len: NonZeroUsize::new(table_len)?,
            output_stride: NonZeroUsize::new(output_stride)?,
            disjoint,
            split: None,
            spectrum_floats,
        })
    }

    /// Returns the plan of the cheapest split of the longer operand into blocks,
    /// or this whole plan when no split is estimated to cost less.
    ///
    /// `output_scale` is the output limbs per operand limb, and `crt_cost` is as in
    /// [`Self::select`]. `plan_block` plans a block of the given limbs with the
    /// shorter operand, as its left and right operand. Each candidate grows to the
    /// longest block its row holds. Blocks run one after another, so several
    /// workers keep the whole plan, whose primes and stages split across them.
    #[must_use]
    pub fn cheapest_blocks(
        self,
        (len_a, len_b): (NonZeroUsize, NonZeroUsize),
        (output_scale, workers): (usize, NonZeroUsize),
        crt_cost: u8,
        plan_block: impl Fn(NonZeroUsize) -> Option<Self>,
    ) -> Self {
        if workers.get() > 1 {
            return self;
        }
        let short = len_a.min(len_b);
        self.search_blocks(len_a.max(len_b), short, crt_cost, plan_block)
            .and_then(|(block, block_limbs)| {
                Some(Self {
                    split: Some(Split {
                        block_limbs: block_limbs.get(),
                        step_limbs: block_limbs.get().checked_mul(output_scale)?,
                        overlap_limbs: short.get().checked_mul(output_scale)?,
                        swapped: len_a < len_b,
                    }),
                    ..block
                })
            })
            .unwrap_or(self)
    }

    /// Returns the block geometry and length whose estimated work is smallest and
    /// at most three quarters of the whole plan's.
    ///
    /// A split runs `2*transform + crt` per block and one forward pass of the
    /// shorter operand. The estimate omits packing, stage setup, and adding each
    /// window, which the required margin covers. Candidates double from the
    /// shorter length up to half of the longer one.
    fn search_blocks(
        self,
        long: NonZeroUsize,
        short: NonZeroUsize,
        crt_cost: u8,
        plan_block: impl Fn(NonZeroUsize) -> Option<Self>,
    ) -> Option<(Self, NonZeroUsize)> {
        let (whole_transform, whole_crt) = self.work(crt_cost)?;
        let whole = whole_transform.checked_mul(3)?.checked_add(whole_crt)?;
        let mut cost = whole.checked_sub(whole >> 2)?;
        let mut best = None;
        let mut limbs = short;
        while limbs.get().checked_mul(2)? <= long.get() {
            let seed = plan_block(limbs)?;
            let room = seed
                .transform_len
                .get()
                .checked_add(1)?
                .checked_sub(seed.right.digits.get())?
                .min(seed.cyclic_len);
            let filled = NonZeroUsize::new(
                room.checked_mul(seed.chunk_bits.get())?
                    .div_euclid(LIMB_BITS),
            )
            .map_or(limbs, |grown| grown.max(limbs));
            let block = plan_block(filled)?;
            let (transform, crt) = block.work(crt_cost)?;
            let blocks = u128::try_from(long.div_ceil(filled).get()).ok()?;
            let split = transform
                .checked_mul(2)?
                .checked_add(crt)?
                .checked_mul(blocks)?
                .checked_add(transform)?;
            if split < cost {
                cost = split;
                best = Some((block, filled));
            }
            limbs = limbs.checked_add(limbs.get())?;
        }
        best
    }

    /// Returns the estimated work `(np*work, crt_cost*np*(np-1)*conv_len)` of one
    /// pass of every prime and of the CRT.
    ///
    /// A part of `n` points costs `n*log2(n)`. A tail of `M` points also costs
    /// `TAIL_OVERHEAD*M` for the reduction of each input and the recombination,
    /// an empirical factor of the integer-product scan.
    fn work(&self, crt_cost: u8) -> Option<(u128, u128)> {
        const TAIL_OVERHEAD: u128 = 4;
        let part = |points: usize| -> Option<u128> {
            u128::try_from(points)
                .ok()?
                .checked_mul(u128::from(points.trailing_zeros()))
        };
        let tail = self.transform_len.get().checked_sub(self.cyclic_len)?;
        let overhead = u128::try_from(tail).ok()?.checked_mul(TAIL_OVERHEAD)?;
        let work = part(self.cyclic_len)?
            .checked_add(part(tail)?)?
            .checked_add(overhead)?;
        let count = u128::try_from(self.prefix.count()).ok()?;
        let crt = u128::from(crt_cost)
            .checked_mul(count)?
            .checked_mul(count.checked_sub(1)?)?
            .checked_mul(u128::try_from(self.conv_len).ok()?)?;
        Some((count.checked_mul(work)?, crt))
    }

    /// Returns the active primes.
    #[inline]
    #[must_use]
    pub const fn prefix(&self) -> PrimePrefix {
        self.prefix
    }

    /// Returns the bits between consecutive operand coefficients.
    #[inline]
    #[must_use]
    pub const fn chunk_bits(&self) -> NonZeroUsize {
        self.chunk_bits
    }

    /// Returns the left operand.
    #[inline]
    #[must_use]
    pub const fn left(&self) -> Operand {
        self.left
    }

    /// Returns the right operand.
    #[inline]
    #[must_use]
    pub const fn right(&self) -> Operand {
        self.right
    }

    /// Returns the length of the convolution coefficient prefix.
    #[inline]
    #[must_use]
    pub const fn conv_len(&self) -> usize {
        self.conv_len
    }

    /// Returns the columns the inverse stores: `conv_len` rounded up to complete
    /// vectors.
    #[inline]
    #[must_use]
    pub const fn output_columns(&self) -> Columns {
        // SAFETY: admission rounds the positive convolution length up to a multiple
        // of COLUMN_ALIGN.
        unsafe { Columns::new(0, self.output_columns) }
    }

    /// Returns every column of one residue row.
    #[inline]
    #[must_use]
    pub const fn row_columns(&self) -> Columns {
        // SAFETY: a row holds complete vectors, so its positive length is a multiple
        // of COLUMN_ALIGN.
        unsafe { Columns::new(0, self.transform_len.get()) }
    }

    /// Returns the floats in one residue row.
    #[inline]
    #[must_use]
    pub const fn transform_len(&self) -> NonZeroUsize {
        self.transform_len
    }

    /// Returns the power-of-two length `N` of the row's cyclic part.
    #[inline]
    #[must_use]
    pub const fn cyclic_len(&self) -> usize {
        self.cyclic_len
    }

    /// Returns the root-table entries per prime, a power of two.
    #[inline]
    #[must_use]
    pub const fn table_len(&self) -> NonZeroUsize {
        self.table_len
    }

    /// Returns the bits between consecutive result coefficients.
    #[inline]
    #[must_use]
    pub const fn output_stride(&self) -> NonZeroUsize {
        self.output_stride
    }

    /// Returns whether the stride is a whole limb count holding every coefficient.
    #[inline]
    #[must_use]
    pub const fn disjoint(&self) -> bool {
        self.disjoint
    }

    /// Returns the split of the longer operand, when this geometry describes one block.
    #[inline]
    #[must_use]
    pub const fn split(&self) -> Option<Split> {
        self.split
    }

    /// Returns the output limbs that consecutive block windows share.
    #[inline]
    #[must_use]
    pub const fn overlap_limbs(&self) -> usize {
        match self.split {
            Some(split) => split.overlap_limbs,
            None => 0,
        }
    }

    /// Returns the floats of one row per prime for each spectrum: one for a square,
    /// two for a product.
    #[inline]
    #[must_use]
    pub const fn arena_floats(&self, square: bool) -> usize {
        // Admission bounds the floats of one spectrum by usize::MAX/2.
        if square {
            self.spectrum_floats
        } else {
            self.spectrum_floats << 1
        }
    }
}

/// Returns `2*width + ceil(log2(terms))`, the bits of a sum of `terms` products of
/// `width`-bit coefficients: `terms*(2^width-1)^2 < 2^bits`.
fn product_bits(width: usize, terms: NonZeroUsize) -> Option<usize> {
    let log_terms = usize::BITS.checked_sub(terms.get().saturating_sub(1).leading_zeros())?;
    width
        .checked_mul(2)?
        .checked_add(usize::try_from(log_terms).ok()?)
}

/// Returns the row length, its cyclic length, and the root-table length of the
/// shortest row holding `conv_len` coefficients and an operand of `longest`.
///
/// A row of `3*M` points evaluates modulo `(x^N-1)*(x^M-i)` with `N = 2*M`. Its
/// operands fit the cyclic part, and the tail node needs roots up to index
/// `3*N/4`, so its table has `N` entries where a row of `N` points needs `N/2`.
/// The tail is a transform of its own remainder tree, so it has at least
/// `MIN_TRANSFORM_LEN` points.
fn row_shape(conv_len: usize, longest: usize) -> Option<(usize, usize, usize)> {
    let power = conv_len.checked_next_power_of_two()?.max(MIN_TRANSFORM_LEN);
    let (cyclic, tail) = (power >> 1, power >> 2);
    let row = cyclic.checked_add(tail)?;
    Some(
        if tail >= MIN_TRANSFORM_LEN && conv_len <= row && longest <= cyclic {
            (row, cyclic, cyclic)
        } else {
            (power, power, power >> 1)
        },
    )
}
