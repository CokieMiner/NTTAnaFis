//! Portable residue kernels and exact scalar products.

#[cfg(any(
    not(any(ntt_fixed_kernels, ntt_wide_integers)),
    not(any(ntt_nearest_intrinsic, ntt_nearest_std))
))]
use super::BIAS;
#[cfg(not(ntt_fixed_kernels))]
use super::{FloatField, Kernels};

#[cfg(not(ntt_fixed_kernels))]
mod kernels;
mod product;

pub use product::Scalar;
