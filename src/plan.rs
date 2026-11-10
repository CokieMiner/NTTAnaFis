//! Native-limb multiplication: plan admission, caller-owned scratch, and retained workspaces.
//!
//! Native multiplication packs binary digits below `chunk_bits`, so every
//! product limb is recovered from its residues modulo the selected primes.

use core::{num::NonZeroUsize, ptr::eq};

use crate::{
    arch::Kernels,
    multiply::{Layout, Retained},
};

use super::{Budgeted, Executor, Geometry, Limb};

/// An admitted product of two operand lengths and the scratch it needs.
///
/// A plan sizes caller-owned scratch and multiplies or squares in it through
/// [`Self::mul`] and [`Self::sqr`]. The same plan serves every call with
/// matching operand lengths.
///
/// # Examples
///
/// ```
/// use ntt_anafis::MulPlan;
///
/// let (a, b) = (vec![u32::MAX as usize; 64], vec![0x1234_5678_usize; 48]);
/// let mut dst = vec![0; a.len() + b.len()];
///
/// let plan = MulPlan::new(a.len(), b.len()).expect("admitted shape");
/// let mut scratch = vec![0; plan.scratch_len()];
/// assert!(plan.mul(&mut dst, &a, &b, &mut scratch));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MulPlan {
    layout: Layout,
}

impl MulPlan {
    /// Plans positive operand lengths, rejecting zero lengths and unavailable
    /// or unaddressable geometry.
    ///
    /// Lengths count native limbs. Leading zero limbs are permitted, but an
    /// empty operand has no plan. Returns `None` when either length is zero,
    /// the coefficient geometry cannot be admitted, or the required buffers
    /// exceed the addressable range.
    ///
    /// The plan suits this CPU and the worker pool active when it is built. One
    /// worker weighs the CRT by the kernels' cost, and several weigh it as one
    /// transform point, since more primes split across them. An operand at least
    /// twice as long as the other may run in blocks sized for the pool. The plan
    /// stays exact in any other pool.
    #[must_use]
    pub fn new(len_a: usize, len_b: usize) -> Option<Self> {
        let limbs = (NonZeroUsize::new(len_a)?, NonZeroUsize::new(len_b)?);
        let workers = Budgeted::resolve(Executor::budget);
        let crt_cost = if workers.get() > 1 {
            1
        } else {
            Kernels::get().crt_cost
        };
        let planned = Geometry::select(limbs.0, limbs.1, crt_cost)?.cheapest_blocks(
            limbs,
            (1, workers),
            crt_cost,
            |block| Geometry::select(block, limbs.0.min(limbs.1), crt_cost),
        );
        let layout = Layout::new(planned, (len_a, len_b), len_a.checked_add(len_b)?)?;
        Some(Self { layout })
    }

    /// Returns the multiplication scratch length in native limbs, including alignment padding.
    #[must_use]
    pub const fn scratch_len(&self) -> usize {
        self.layout.scratch_limbs(false)
    }

    /// Returns the square scratch length in native limbs, including alignment padding.
    ///
    /// Squaring uses a plan whose two operand lengths are equal.
    #[must_use]
    pub const fn sqr_scratch_len(&self) -> usize {
        self.layout.scratch_limbs(true)
    }

    /// Writes `a*b` to the first `a.len()+b.len()` limbs using caller-owned scratch.
    ///
    /// The operand lengths must match this plan. Plans have positive lengths, so
    /// empty operands are rejected. Returns `false` if either length differs or
    /// the destination or scratch is too short; failure preserves `dst`.
    /// Success preserves its suffix. Passing the same slice as both operands
    /// squares it, which needs only [`Self::sqr_scratch_len`] limbs. Root tables
    /// are regenerated in `scratch` on each call.
    #[must_use]
    pub fn mul(&self, dst: &mut [Limb], a: &[Limb], b: &[Limb], scratch: &mut [Limb]) -> bool {
        Budgeted::resolve(|executor| {
            if eq(a, b) {
                self.layout.run::<true, _>(dst, (a, a), scratch, executor)
            } else {
                self.layout.run::<false, _>(dst, (a, b), scratch, executor)
            }
        })
    }

    /// Writes `a*a` to the first `2*a.len()` limbs using caller-owned scratch.
    ///
    /// The plan must describe a square shape (`len_a == len_b`), and `a.len()`
    /// must match it. Plans have positive lengths, so an empty operand is
    /// rejected. Returns `false` if the shape differs or the destination or
    /// scratch is too short; failure preserves `dst`. Success preserves its
    /// suffix. Root tables are regenerated in `scratch` on each call.
    #[must_use]
    pub fn sqr(&self, dst: &mut [Limb], a: &[Limb], scratch: &mut [Limb]) -> bool {
        Budgeted::resolve(|executor| self.layout.run::<true, _>(dst, (a, a), scratch, executor))
    }
}

/// Reusable native-limb multiplication workspace.
///
/// Retains transform root tables and coefficient arenas across calls. Tables
/// serve every shorter transform and grow only when a larger operand shape arrives.
///
/// # Examples
///
/// ```
/// use ntt_anafis::{MulPlan, MulWorkspace};
///
/// let (a, b) = (vec![42_usize; 128], vec![7_usize; 128]);
/// let mut dst = vec![0; a.len() + b.len()];
///
/// let mut workspace = MulWorkspace::new();
/// workspace.prepare(MulPlan::new(a.len(), b.len()).expect("admitted shape"));
/// assert!(workspace.mul(&mut dst, &a, &b));
/// ```
#[derive(Clone, Debug)]
pub struct MulWorkspace {
    plan: Option<MulPlan>,
    retained: Retained,
}

impl MulWorkspace {
    /// Creates empty reusable storage.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            plan: None,
            retained: Retained::new(),
        }
    }

    /// Retains an admitted plan and prepares its transform tables and coefficient arena.
    pub fn prepare(&mut self, plan: MulPlan) {
        Budgeted::resolve(|executor| {
            self.retained.prepare(&plan.layout, false, executor);
        });
        self.plan = Some(plan);
    }

    /// Writes `a*b` to the first `a.len()+b.len()` limbs with default execution.
    ///
    /// Returns `false` for a short destination or unavailable plan, preserving
    /// `dst`. Success preserves the suffix. An empty operand produces zero.
    /// Passing the same slice as both operands squares it.
    #[must_use]
    pub fn mul(&mut self, dst: &mut [Limb], a: &[Limb], b: &[Limb]) -> bool {
        if eq(a, b) {
            self.run::<true>(dst, a, a)
        } else {
            self.run::<false>(dst, a, b)
        }
    }

    /// Writes `a*a` to the first `2*a.len()` limbs with default execution.
    ///
    /// Returns `false` for a short destination or unavailable plan, preserving
    /// `dst`. Success preserves the suffix. Empty input succeeds.
    #[must_use]
    pub fn sqr(&mut self, dst: &mut [Limb], a: &[Limb]) -> bool {
        self.run::<true>(dst, a, a)
    }

    /// Runs the retained matching plan, or selects and retains a new one.
    fn run<const SQUARE: bool>(&mut self, dst: &mut [Limb], a: &[Limb], b: &[Limb]) -> bool {
        if a.is_empty() || b.is_empty() {
            let output = a
                .len()
                .checked_add(b.len())
                .and_then(|output_len| dst.get_mut(..output_len));
            return output.is_some_and(|limbs| {
                limbs.fill(0);
                true
            });
        }
        let Some(plan) = self
            .plan
            .filter(|plan| plan.layout.matches(a.len(), b.len()))
            .or_else(|| MulPlan::new(a.len(), b.len()))
        else {
            return false;
        };
        self.plan = Some(plan);
        Budgeted::resolve(|executor| {
            self.retained
                .run::<SQUARE, _>(&plan.layout, dst, (a, b), executor)
        })
    }
}

impl Default for MulWorkspace {
    fn default() -> Self {
        Self::new()
    }
}
