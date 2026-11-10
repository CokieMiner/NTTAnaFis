//! The contract every residue kernel shares, stated once.
//!
//! Kernels are `unsafe` functions over raw pointers, because parallel tasks write
//! disjoint columns of one row at the same time. A kernel's `# Safety` section
//! states only what it adds to the terms below.
//!
//! # Terms
//!
//! - **Admitted**: the CPU has the kernel's instructions. A table from
//!   [`Kernels::get`] is admitted, and so are the portable kernels.
//! - **Spans**: runs of `stride` floats at `values + k*stride`. Spans are
//!   initialized unless a kernel says otherwise, and they fit the row.
//! - **Columns**: indices below `stride`, the same index in every span. A call
//!   owns its [`Columns`] exclusively, which is what makes parallel tasks disjoint.
//! - **Roots**: immutable floats centered below `p/2`, disjoint from the spans.
//!   A node `j` with `n` points is the factor `x^n - W[j]^2`, and its stage uses
//!   `W[j]`, `W[2*j]`, and `W[2*j+1]`, or their inverses going backward.
//! - **Field**: a [`FloatField`] proves `p` and `pinv`.
//!
//! # Residue bounds
//!
//! Residues are integer-valued floats, and every bound below is on the magnitude.
//!
//! | Values | Bound |
//! | :--- | :--- |
//! | Packed input | `p` |
//! | Input of a forward stage | `3*p` |
//! | Output of a forward stage | `5*p/2` |
//! | Output of the last forward stage, a spectrum | `3*p/2` |
//! | Input and output of an inverse stage | `2*p` |
//!
//! Catalog admission bounds a reduced product by `15*p/16` when `abs(a*b) < 2*p^2`
//! and by `11*p/8` when `abs(a*b) < 4*p^2`. Reducing a value below `8*p` gives
//! a value below `9*p/16`, and a value `v` below `p` its centered residue, below
//! `p/2`: the quotient estimate errs by less than `2^-52`, and an odd `p` keeps
//! `v/p` at least `1/(2*p) > 2^-51` from every half-integer.
//!
//! A forward radix-4 stage admits inputs below `3*p`. Then `abs(c*w) < 3*p^2/2`,
//! and `abs(b + d*w) < 63*p/16` gives products below `2*p^2`. It reduces only `a`,
//! the input without a product, and returns outputs below
//! `9*p/16 + 15*p/8 < 5*p/2`. The last forward stage reduces both sums instead,
//! which bounds spectra by `9*p/16 + 15*p/16 = 3*p/2` and their products by
//! `9*p^2/4`. An inverse stage admits inputs below `2*p`: `abs(u-s) < 8*p` times a
//! root is below `4*p^2`, and only `u+s` needs a reduction to stay below `2*p`.
//!
//! # Canonical bits
//!
//! A value in `(-p, p)` becomes its residue in `0..p` by adding `p` when negative.
//! Adding [`BIAS`] then places that integer in the low mantissa bits, and an
//! exclusive or with `BIAS` leaves the integer as the bits of the float. Those
//! slots are `u64` bit patterns from then on and are not residues any more.

#![expect(
    unsafe_code,
    reason = "Column ranges and rounded lengths state their alignment proofs at construction"
)]

use core::num::NonZeroUsize;

use super::{CRT_LANES, DigitTile, FloatField, MAX_PARTS, MAX_PRIMES, Packing, WORDS};

/// Every lane count divides this power of two, and vector ranges end at its
/// multiples.
pub const COLUMN_ALIGN: usize = 8;

/// Points of one leaf.
pub const LEAF_POINTS: usize = 16;

/// Shortest admitted transform. A leaf kernel may consume two leaves at once, and
/// a shorter transform has no stage of a complete vector.
pub const MIN_TRANSFORM_LEN: usize = 2 * LEAF_POINTS;

const _: () = {
    assert!(COLUMN_ALIGN.is_power_of_two(), "column alignment");
    assert!(
        MIN_TRANSFORM_LEN.is_multiple_of(COLUMN_ALIGN),
        "a transform holds complete vectors"
    );
};

/// `2^52`. Adding it places an integer below `2^52` in the low mantissa bits.
pub const BIAS: f64 = 4_503_599_627_370_496.0;

/// A nonempty range of columns that one kernel call owns.
///
/// Vector kernels run complete vectors with no scalar cleanup, so a range must
/// end at a multiple of their lane count. They run the first vector before they
/// test for the end. [`Self::complete`] checks each lane count against
/// [`COLUMN_ALIGN`] when the kernel compiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Columns {
    start: usize,
    end: usize,
}

impl Columns {
    /// Returns the columns `start..end`.
    ///
    /// # Safety
    /// `start < end`, and both are multiples of [`COLUMN_ALIGN`]. The portable
    /// kernels have one lane, so a range that reaches only them may end anywhere.
    #[inline]
    #[must_use]
    pub const unsafe fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    /// Returns the end of this range, a positive multiple of [`COLUMN_ALIGN`].
    #[inline]
    #[must_use]
    pub const fn end(self) -> usize {
        self.end
    }

    /// Returns the bounds of this range for a kernel of `LANES` lanes.
    #[inline]
    #[must_use]
    pub const fn complete<const LANES: usize>(self) -> (usize, usize) {
        const {
            assert!(
                COLUMN_ALIGN.is_multiple_of(LANES),
                "every lane count divides the column alignment"
            );
        }
        debug_assert!(
            self.start < self.end
                && self.start.is_multiple_of(LANES)
                && self.end.is_multiple_of(LANES),
            "kernels take nonempty ranges of complete vectors"
        );
        (self.start, self.end)
    }
}

/// A stage over spans: `(values, roots, stride, columns, field)`.
///
/// Spans and roots follow the kernel's arity. `roots` is unused by the unit radix-2
/// stages and by a prefix stage with one input.
pub type StageKernel = unsafe fn(*mut f64, *const f64, usize, Columns, FloatField);

/// A last inverse stage: `(values, roots, stride, columns, factor, field)`.
///
/// The kernel scales by the factor, centered below `p/2`, and stores canonical bits.
pub type FinalKernel = unsafe fn(*mut f64, *const f64, usize, Columns, f64, FloatField);

/// Leaves: `(values, right, roots, pairs, node, field)`.
///
/// `values` holds `pairs` pairs of leaves of [`LEAF_POINTS`] points each, because
/// kernels may consume two leaves at once. `node..node+2*pairs` names consecutive
/// nodes, and the roots of their descendants below `8*(node+2*pairs)` fit the table.
/// Forward output and inverse input use transposed coordinates within a leaf. A
/// product leaf reads a disjoint `right` spectrum of the same length, and other
/// leaves ignore it.
pub type LeafKernel = unsafe fn(*mut f64, *const f64, *const f64, NonZeroUsize, usize, FloatField);

/// Products of spectra: `(left, right, columns, field)`.
///
/// Both spectra are below `2*p`. Squaring ignores `right`.
pub type PointwiseKernel = unsafe fn(*mut f64, *const f64, Columns, FloatField);

/// Normalization: `(values, columns, factor, field)`.
///
/// Values are below `2*p`, and the factor lies in `0..p`. The kernel stores canonical
/// bits.
pub type ResidueKernel = unsafe fn(*mut f64, Columns, f64, FloatField);

/// Centered products with one factor: `(target, source, columns, factor, field)`.
///
/// The spans are disjoint, and sources and the factor are centered below `p/2`.
pub type ScaleKernel = unsafe fn(*mut f64, *const f64, Columns, f64, FloatField);

/// Reduces a digit tile into every selected row: `(packing, columns, digits)`.
///
/// The call owns these columns in each output row. The columns fit the tile, and
/// digits lie in `0..2^49`. A kernel serves one part count, the row count it reads.
pub type PackingKernel = unsafe fn(&Packing<'_>, Columns, &DigitTile);

/// Eight coefficients of one prefix length, shifted to their bit offsets in
/// native-word columns: `(residues, first, bit, width, output)`.
pub type CoefficientKernel =
    unsafe fn(&[&[f64]; MAX_PRIMES], usize, usize, usize, &mut [[u64; CRT_LANES]; WORDS + 1]);

/// The kernels of one vector unit, resolved once per complete operation.
#[derive(Clone, Copy, Debug)]
pub struct Kernels {
    /// Cost of one CRT term in points of one transform level, for plan selection.
    ///
    /// It is two on AVX2 and AVX-512F, whose FMA CRT trails their fused
    /// transforms, and one elsewhere. Plans are exact under any value.
    pub crt_cost: u8,
    /// Optional integer CRT packet kernels, indexed by prime count minus one.
    pub coefficients: Option<[CoefficientKernel; MAX_PRIMES]>,
    /// Reduces shared binary digits across the selected prime rows, indexed by the
    /// part count minus one.
    pub pack: [PackingKernel; MAX_PARTS],
    /// Packing with a final binary part of at most three bits, indexed by the part
    /// count minus two.
    pub pack_short: [PackingKernel; MAX_PARTS - 1],
    /// Unit radix-2 stage over two spans below `2*p`.
    pub radix2: StageKernel,
    /// Copies a packed lower span to the upper span, for a zero upper input.
    pub duplicate_half: StageKernel,
    /// Unit radix-2 with its radix-4 children, reading one to four packed spans.
    pub dif8_prefix: [StageKernel; 4],
    /// Unit radix-4 reading one, two, or three packed spans.
    pub dif4_prefix: [StageKernel; 3],
    /// Forward and inverse radix-4 stages over four spans.
    pub dif: StageKernel,
    pub dit: StageKernel,
    /// The same stages at node zero, whose first two roots are one.
    pub dif_unit: StageKernel,
    pub dit_unit: StageKernel,
    /// Last inverse stages, one per active output count.
    pub finish2: [FinalKernel; 2],
    pub finish4: [FinalKernel; 4],
    pub finish8: [FinalKernel; 8],
    /// Radix-2 stages of a node with a root, for subtrees of odd order.
    pub dif2: StageKernel,
    pub dit2: StageKernel,
    /// Writes the residue of two packed spans modulo `x^M-i` to a third span.
    pub fold: StageKernel,
    /// Writes the residue of four packed spans modulo `x^M-w` to span four or six.
    pub fold4: [StageKernel; 2],
    /// Recovers the coefficients of a row of `3*M`, `5*M`, or `7*M` points as
    /// canonical bits, with and without the top span.
    pub recombine: [FinalKernel; 2],
    pub recombine5: [FinalKernel; 2],
    pub recombine7: [FinalKernel; 2],
    /// Two radix-4 stages held in registers over sixteen spans.
    pub dif_wide: Option<StageKernel>,
    pub dit_wide: Option<StageKernel>,
    /// Forward leaves.
    pub dif16: LeafKernel,
    /// Inverse leaves over a spectrum already formed.
    pub dit16: LeafKernel,
    /// Inverse leaves that form the product or square themselves.
    pub dit16_mul: LeafKernel,
    pub dit16_sqr: LeafKernel,
    pub multiply: PointwiseKernel,
    pub square: PointwiseKernel,
    pub residues: ResidueKernel,
    pub scale: ScaleKernel,
}
