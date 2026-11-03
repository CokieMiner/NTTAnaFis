//! Exact forced-profile fixtures, independent of automatic prime selection.

use core::num::NonZeroUsize;

use alloc::vec;

use crate::{Budgeted, Executor, Geometry, LIMB_BITS, Limb, multiply::Layout, profile::Profile};

pub fn forced(left: usize, right: usize, profile: Profile) -> Option<Geometry> {
    Geometry::under(
        NonZeroUsize::new(left.checked_mul(LIMB_BITS)?)?,
        NonZeroUsize::new(right.checked_mul(LIMB_BITS)?)?,
        profile,
    )
}

/// Writes `a*b` under a forced geometry with fresh tables.
pub fn multiply<const SQUARE: bool>(geometry: &Geometry, dst: &mut [Limb], a: &[Limb], b: &[Limb]) {
    Budgeted::resolve(|executor| {
        multiply_under::<SQUARE, _>(geometry, dst, a, b, executor);
    });
}

/// Writes `a*b` as [`multiply`] does, under the budget and joins of `executor`.
pub fn multiply_under<const SQUARE: bool, E: Executor>(
    geometry: &Geometry,
    dst: &mut [Limb],
    a: &[Limb],
    b: &[Limb],
    executor: &E,
) {
    let layout = Layout::new(*geometry, (a.len(), b.len()), dst.len()).expect("addressable shape");
    let mut scratch = vec![0; layout.scratch_limbs(SQUARE)];
    assert!(layout.run::<SQUARE, _>(dst, (a, b), &mut scratch, executor));
}
