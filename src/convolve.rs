//! Exact dense convolution of nonnegative native-limb coefficients.
//!
//! With input bounds `A` and `B` and `t = min(len_a, len_b)`, every result
//! coefficient is at most `C = t*A*B`. The transform runs directly on the
//! coefficients modulo the fewest catalog primes whose capacity reaches the bit
//! length of `C`, so the cost follows the coefficient size. Each coefficient is
//! written to its own block of whole limbs, and blocks never carry into one
//! another. A short operand costs less by summing products column by column, and
//! a much longer operand by running in pieces that share one spectrum.

use core::{num::NonZeroUsize, ptr::eq};

use crate::{
    arch::Kernels,
    multiply::{Layout, Retained},
};

use super::{Budgeted, DoubleLimb, Executor, Geometry, LIMB_BITS, Limb};

/// Longest shorter operand summed directly. Both algorithms scale with the
/// longer operand, so this crossover barely depends on its length.
const SCHOOLBOOK_MAX_TERMS: usize = 24;

/// Exact convolution plan for nonnegative native-limb coefficient vectors.
///
/// The plan admits positive operand lengths and caller-supplied upper bounds on
/// every input coefficient. It chooses the smallest whole-limb block width that
/// holds the bound of every convolution coefficient. A successful convolution
/// writes its `len_a + len_b - 1` coefficients to the first
/// [`output_len`](Self::output_len) limbs of the destination: coefficient `k`
/// occupies the [`stride`](Self::stride) limbs starting at `k * stride()`,
/// least-significant limb first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConvPlan {
    max_a: Limb,
    max_b: Limb,
    stride: usize,
    schoolbook: bool,
    layout: Layout,
}

impl ConvPlan {
    /// Plans an exact convolution of nonnegative coefficient vectors.
    ///
    /// `max_a` and `max_b` must bound every coefficient in the corresponding
    /// operand. Returns `None` for an empty operand, an unaddressable shape, or
    /// a transform order outside the prime catalog.
    ///
    /// The plan suits the worker pool active when it is built: with one worker,
    /// operands of very different lengths run in blocks. It stays exact in any
    /// other pool.
    #[must_use]
    pub fn new(len_a: usize, len_b: usize, max_a: Limb, max_b: Limb) -> Option<Self> {
        let limbs = (NonZeroUsize::new(len_a)?, NonZeroUsize::new(len_b)?);
        let short = limbs.0.min(limbs.1);
        let bound_bits = bound_bits(short, max_a, max_b)?;
        let stride = bound_bits.div_ceil(LIMB_BITS).max(1);
        let bits_a = usize::try_from(Limb::BITS.saturating_sub(max_a.leading_zeros()))
            .ok()
            .and_then(NonZeroUsize::new)
            .unwrap_or(NonZeroUsize::MIN);
        let bits_b = usize::try_from(Limb::BITS.saturating_sub(max_b.leading_zeros()))
            .ok()
            .and_then(NonZeroUsize::new)
            .unwrap_or(NonZeroUsize::MIN);
        let stride_bits = stride.checked_mul(LIMB_BITS)?;
        let workers = Budgeted::resolve(Executor::budget);
        let planned = Geometry::for_residues(limbs, (bits_a, bits_b), (bound_bits, stride_bits))?
            .cheapest_blocks(limbs, (stride, workers), Kernels::get().crt_cost, |block| {
                let bits = if len_a < len_b {
                    (bits_b, bits_a)
                } else {
                    (bits_a, bits_b)
                };
                Geometry::for_residues((block, short), bits, (bound_bits, stride_bits))
            });
        // Positive lengths give at least one coefficient.
        let output_len = len_a
            .checked_add(len_b)?
            .wrapping_sub(1)
            .checked_mul(stride)?;
        Some(Self {
            max_a,
            max_b,
            stride,
            schoolbook: short.get() <= SCHOOLBOOK_MAX_TERMS,
            layout: Layout::new(planned, (len_a, len_b), output_len)?,
        })
    }

    /// Returns the limbs of one output coefficient, the distance between
    /// consecutive coefficients.
    #[must_use]
    pub const fn stride(&self) -> usize {
        self.stride
    }

    /// Returns the limbs written to the destination, `(len_a + len_b - 1) * stride()`.
    #[must_use]
    pub const fn output_len(&self) -> usize {
        self.layout.output_len()
    }

    /// Returns the caller-owned scratch length in native limbs.
    ///
    /// A shape short enough to sum directly needs none.
    #[must_use]
    pub const fn scratch_len(&self) -> usize {
        if self.schoolbook {
            0
        } else {
            self.layout.scratch_limbs(false)
        }
    }

    /// Writes the exact convolution using caller-owned scratch storage.
    ///
    /// Returns `false` when an operand length differs, an actual coefficient
    /// exceeds its admitted bound, the destination is shorter than
    /// [`Self::output_len`], or scratch is too short. Failure preserves
    /// `dst`. Success preserves any destination suffix. Passing the same slice
    /// as both operands runs one forward transform per prime.
    #[must_use]
    pub fn convolve(&self, dst: &mut [Limb], a: &[Limb], b: &[Limb], scratch: &mut [Limb]) -> bool {
        if !self.admits(a, b) {
            return false;
        }
        if self.schoolbook {
            return self.sum_columns(dst, a, b);
        }
        Budgeted::resolve(|executor| {
            if eq(a, b) {
                self.layout.run::<true, _>(dst, (a, a), scratch, executor)
            } else {
                self.layout.run::<false, _>(dst, (a, b), scratch, executor)
            }
        })
    }

    /// Tests the operand lengths and the admitted coefficient bounds.
    ///
    /// The bitwise OR of an operand bounds each of its values and needs no early
    /// exit, so the exact scan runs only for a bound that is not of the form `2^k-1`.
    fn admits(&self, a: &[Limb], b: &[Limb]) -> bool {
        self.layout.matches(a.len(), b.len())
            && [(a, self.max_a), (b, self.max_b)]
                .into_iter()
                .all(|(values, max)| {
                    values.iter().fold(0, |union, value| union | value) <= max
                        || values.iter().all(|value| *value <= max)
                })
    }

    /// Writes the convolution by direct summation, or returns `false` for a
    /// short destination.
    fn sum_columns(&self, dst: &mut [Limb], a: &[Limb], b: &[Limb]) -> bool {
        let Some(output) = dst.get_mut(..self.output_len()) else {
            return false;
        };
        // A coefficient has at most 2*LIMB_BITS + usize::BITS - 1 bits, which
        // fit three limbs on every supported pointer width.
        match self.stride {
            1 => columns::<1>(output, a, b),
            2 => columns::<2>(output, a, b),
            _ => columns::<3>(output, a, b),
        }
        true
    }
}

/// Reusable storage for repeated exact convolutions.
///
/// The workspace retains a prepared plan, the root tables, and a coefficient
/// arena. Call [`Self::prepare`] once for a fixed operand shape and coefficient
/// bound, then call [`Self::convolve`] for every matching pair. Root tables
/// serve every shorter transform and smaller prime prefix, so preparing a
/// smaller plan later regenerates nothing.
#[derive(Clone, Debug)]
pub struct ConvWorkspace {
    plan: Option<ConvPlan>,
    retained: Retained,
}

impl ConvWorkspace {
    /// Creates an empty workspace.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            plan: None,
            retained: Retained::new(),
        }
    }

    /// Retains a plan and prepares its root tables and coefficient arena.
    ///
    /// A shape short enough to sum directly prepares nothing.
    pub fn prepare(&mut self, plan: ConvPlan) {
        if !plan.schoolbook {
            Budgeted::resolve(|executor| {
                self.retained.prepare(&plan.layout, false, executor);
            });
        }
        self.plan = Some(plan);
    }

    /// Writes the exact convolution to `dst`.
    ///
    /// The destination and result layout match [`ConvPlan::convolve`]. Returns
    /// `false` when no plan has been prepared, an operand length differs, an
    /// actual coefficient exceeds its admitted bound, or the destination is
    /// shorter than the prepared plan's output. Failure preserves `dst`. Passing
    /// the same slice as both operands runs one forward transform per prime.
    #[must_use]
    pub fn convolve(&mut self, dst: &mut [Limb], a: &[Limb], b: &[Limb]) -> bool {
        let Some(plan) = self.plan.filter(|plan| plan.admits(a, b)) else {
            return false;
        };
        if plan.schoolbook {
            return plan.sum_columns(dst, a, b);
        }
        Budgeted::resolve(|executor| {
            if eq(a, b) {
                self.retained
                    .run::<true, _>(&plan.layout, dst, (a, a), executor)
            } else {
                self.retained
                    .run::<false, _>(&plan.layout, dst, (a, b), executor)
            }
        })
    }
}

impl Default for ConvWorkspace {
    fn default() -> Self {
        Self::new()
    }
}

/// Writes every block of `WORDS` limbs as the sum of its products.
///
/// Block `k` receives `a[i]*b[k-i]` for each `i` where both indices exist.
/// Partial sums of nonnegative products
/// stay below the coefficient bound, so they fit the block and wrapping
/// arithmetic never wraps.
#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Limb products widen exactly into DoubleLimb; each output limb takes its own part of the sum"
)]
fn columns<const WORDS: usize>(output: &mut [Limb], a: &[Limb], b: &[Limb]) {
    for (index, block) in output.as_chunks_mut::<WORDS>().0.iter_mut().enumerate() {
        let next = index.wrapping_add(1);
        let first = next.saturating_sub(b.len());
        let end = next.min(a.len());
        let left = a.get(first..end).unwrap_or_default();
        let right = b
            .get(next.wrapping_sub(end)..next.wrapping_sub(first))
            .unwrap_or_default();
        let terms = left.iter().zip(right.iter().rev());
        let mut low: DoubleLimb = 0;
        let mut high: Limb = 0;
        if WORDS == 1 {
            let mut sum: Limb = 0;
            for (&x, &y) in terms {
                sum = sum.wrapping_add(x.wrapping_mul(y));
            }
            low = sum as DoubleLimb;
        } else if WORDS == 2 {
            for (&x, &y) in terms {
                low = low.wrapping_add((x as DoubleLimb).wrapping_mul(y as DoubleLimb));
            }
        } else {
            for (&x, &y) in terms {
                let (sum, carry) =
                    low.overflowing_add((x as DoubleLimb).wrapping_mul(y as DoubleLimb));
                low = sum;
                high = high.wrapping_add(Limb::from(carry));
            }
        }
        for (slot, limb) in block
            .iter_mut()
            .zip([low as Limb, (low >> LIMB_BITS) as Limb, high])
        {
            *slot = limb;
        }
    }
}

/// Returns the bit length of `terms*max_a*max_b`.
///
/// Native limbs have at most 64 bits, so the product of the bounds fits `u128`.
/// A multiplication by `terms` that leaves `u128` adds the bit length of
/// `terms-1` instead, which is `ceil(log2(terms))`.
fn bound_bits(terms: NonZeroUsize, max_a: Limb, max_b: Limb) -> Option<usize> {
    let product = u128::try_from(max_a)
        .ok()?
        .checked_mul(u128::try_from(max_b).ok()?)?;
    let bits = match product.checked_mul(u128::try_from(terms.get()).ok()?) {
        Some(bound) => u128::BITS.checked_sub(bound.leading_zeros())?,
        None => u128::BITS
            .checked_sub(product.leading_zeros())?
            .checked_add(usize::BITS.checked_sub(terms.get().saturating_sub(1).leading_zeros())?)?,
    };
    usize::try_from(bits).ok()
}
