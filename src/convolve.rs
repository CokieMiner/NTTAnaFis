//! Exact dense convolution of native-limb coefficients, nonnegative or in two's
//! complement.
//!
//! With input bounds `A` and `B` and `t = min(len_a, len_b)`, every result
//! coefficient is at most `C = t*A*B` in magnitude. The transform runs directly
//! on the coefficients modulo the fewest catalog primes whose capacity reaches
//! the bit length of `C`, plus a sign bit for signed coefficients, so the cost
//! follows the coefficient size. Each coefficient is written to its own block of
//! whole limbs, and blocks never carry into one another. A short operand costs
//! less by summing products column by column, and a much longer operand by
//! running in pieces that share one spectrum.

use core::{num::NonZeroUsize, ptr::eq};

use crate::{
    arch::Kernels,
    multiply::{Layout, Retained},
};

use super::{Budgeted, DoubleLimb, Executor, Geometry, LIMB_BITS, Limb};

/// Longest shorter operand summed directly. Both algorithms scale with the
/// longer operand, so this crossover barely depends on its length.
const SCHOOLBOOK_MAX_TERMS: usize = 24;

/// Exact convolution plan for native-limb coefficient vectors, nonnegative or in
/// two's complement.
///
/// The plan admits positive operand lengths and caller-supplied bounds on the
/// magnitude of every input coefficient. It chooses the smallest whole-limb
/// block width that holds the bound of every convolution coefficient, with its
/// sign for a signed plan. A successful convolution writes its
/// `len_a + len_b - 1` coefficients to the first
/// [`output_len`](Self::output_len) limbs of the destination: coefficient `k`
/// occupies the [`stride`](Self::stride) limbs starting at `k * stride()`,
/// least-significant limb first, in two's complement for a signed plan.
///
/// # Examples
///
/// ```
/// use ntt_anafis::ConvPlan;
///
/// let (a, b) = ([1_usize, 2, 3], [4_usize, 5]);
/// let plan = ConvPlan::new(a.len(), b.len(), 3, 5, false).expect("admitted shape");
/// let mut dst = vec![0; plan.output_len()];
/// let mut scratch = vec![0; plan.scratch_len()];
///
/// assert!(plan.convolve(&mut dst, &a, &b, &mut scratch));
/// assert_eq!(dst, [4, 13, 22, 15]);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConvPlan {
    max_a: Limb,
    max_b: Limb,
    schoolbook: bool,
    layout: Layout,
}

impl ConvPlan {
    /// Plans an exact convolution of nonnegative coefficient vectors, or with
    /// `signed` of coefficient vectors in two's complement.
    ///
    /// `max_a` and `max_b` must bound the magnitude of every coefficient in the
    /// corresponding operand; signed bounds lie below `2^(LIMB_BITS-1)`, because
    /// a signed coefficient and a signed result need a sign bit beyond their
    /// magnitude. Each result coefficient is written over its stride, in two's
    /// complement when signed. Returns `None` for an empty operand, an
    /// unaddressable shape, a transform order outside the prime catalog, or a
    /// wider signed bound.
    ///
    /// The plan suits the worker pool active when it is built: an operand at least
    /// twice as long as the other may run in blocks sized for the pool. It stays
    /// exact in any other pool.
    #[must_use]
    pub fn new(len_a: usize, len_b: usize, max_a: Limb, max_b: Limb, signed: bool) -> Option<Self> {
        let limbs = (NonZeroUsize::new(len_a)?, NonZeroUsize::new(len_b)?);
        let short = limbs.0.min(limbs.1);
        let sign = usize::from(signed);
        let bound_bits = bound_bits(short, max_a, max_b)?.checked_add(sign)?;
        let stride = bound_bits.div_ceil(LIMB_BITS).max(1);
        let value_bits = |max: Limb| {
            usize::try_from(Limb::BITS.saturating_sub(max.leading_zeros()))
                .ok()?
                .checked_add(sign)
                .and_then(NonZeroUsize::new)
                .or(Some(NonZeroUsize::MIN))
                .filter(|bits| bits.get() <= LIMB_BITS)
        };
        let (bits_a, bits_b) = (value_bits(max_a)?, value_bits(max_b)?);
        let stride_bits = stride.checked_mul(LIMB_BITS)?;
        let workers = Budgeted::resolve(Executor::budget);
        let planned =
            Geometry::for_residues(limbs, (bits_a, bits_b), (bound_bits, stride_bits), signed)?
                .cheapest_blocks(limbs, (stride, workers), Kernels::get().crt_cost, |block| {
                    let bits = if len_a < len_b {
                        (bits_b, bits_a)
                    } else {
                        (bits_a, bits_b)
                    };
                    Geometry::for_residues((block, short), bits, (bound_bits, stride_bits), signed)
                });
        // Positive lengths give at least one coefficient.
        let output_len = len_a
            .checked_add(len_b)?
            .wrapping_sub(1)
            .checked_mul(stride)?;
        Some(Self {
            max_a,
            max_b,
            schoolbook: short.get() <= SCHOOLBOOK_MAX_TERMS,
            layout: Layout::new(planned, (len_a, len_b), output_len)?,
        })
    }

    /// Returns the limbs of one output coefficient, the distance between
    /// consecutive coefficients.
    #[must_use]
    pub const fn stride(&self) -> usize {
        self.layout.geometry().output_stride().get() >> LIMB_BITS.trailing_zeros()
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
    /// The bitwise OR of an unsigned operand bounds each of its values and needs no
    /// early exit, so the exact scan runs only for a bound that is not of the form
    /// `2^k-1`. A signed operand bounds each magnitude.
    fn admits(&self, a: &[Limb], b: &[Limb]) -> bool {
        let signed = self.layout.geometry().sign_bit().is_some();
        self.layout.matches(a.len(), b.len())
            && [(a, self.max_a), (b, self.max_b)]
                .into_iter()
                .all(|(values, max)| {
                    if signed {
                        values
                            .iter()
                            .all(|value| value.cast_signed().unsigned_abs() <= max)
                    } else {
                        values.iter().fold(0, |union, value| union | value) <= max
                            || values.iter().all(|value| *value <= max)
                    }
                })
    }

    /// Writes the convolution by direct summation, or returns `false` for a
    /// short destination.
    fn sum_columns(&self, dst: &mut [Limb], a: &[Limb], b: &[Limb]) -> bool {
        let Some(output) = dst.get_mut(..self.output_len()) else {
            return false;
        };
        // A coefficient has at most 2*LIMB_BITS + usize::BITS - 1 bits with its
        // sign, which fit three limbs on every supported pointer width. Single
        // limbs wrap alike for both encodings.
        match (self.stride(), self.layout.geometry().sign_bit().is_some()) {
            (1, _) => columns::<1, false>(output, a, b),
            (2, false) => columns::<2, false>(output, a, b),
            (2, true) => columns::<2, true>(output, a, b),
            (_, false) => columns::<3, false>(output, a, b),
            (_, true) => columns::<3, true>(output, a, b),
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
///
/// # Examples
///
/// ```
/// use ntt_anafis::{ConvPlan, ConvWorkspace};
///
/// let (a, b) = ([1_usize, 2, 3], [4_usize, 5]);
/// let plan = ConvPlan::new(a.len(), b.len(), 3, 5, false).expect("admitted shape");
/// let mut dst = vec![0; plan.output_len()];
///
/// let mut workspace = ConvWorkspace::new();
/// workspace.prepare(plan);
/// assert!(workspace.convolve(&mut dst, &a, &b));
/// assert_eq!(dst, [4, 13, 22, 15]);
/// ```
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
/// Unsigned partial sums stay below the coefficient bound, so they fit the
/// block and wrapping arithmetic never wraps. `SIGNED` operands are two's
/// complement: a limb `v` widens to `(v ^ h) - h` modulo `2^(2*LIMB_BITS)` with
/// `h = 2^(LIMB_BITS-1)`, a negative product adds `-1` to the third limb, and
/// the block holds the sum modulo its width, which is exact because the sum
/// fits it.
#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Limb products widen exactly into DoubleLimb; each output limb takes its own part of the sum"
)]
fn columns<const WORDS: usize, const SIGNED: bool>(output: &mut [Limb], a: &[Limb], b: &[Limb]) {
    let half: DoubleLimb = 1 << (LIMB_BITS - 1);
    let widen = |value: Limb| {
        if SIGNED {
            ((value as DoubleLimb) ^ half).wrapping_sub(half)
        } else {
            value as DoubleLimb
        }
    };
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
                low = low.wrapping_add(widen(x).wrapping_mul(widen(y)));
            }
        } else {
            for (&x, &y) in terms {
                let product = widen(x).wrapping_mul(widen(y));
                let (sum, carry) = low.overflowing_add(product);
                low = sum;
                high = high.wrapping_add(Limb::from(carry));
                if SIGNED {
                    high = high.wrapping_sub((product >> (DoubleLimb::BITS - 1)) as Limb);
                }
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
