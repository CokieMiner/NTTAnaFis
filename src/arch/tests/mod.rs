//! Backend arithmetic, fused leaf schedules, and canonical residue conversion.

#[cfg(all(ntt_fixed_kernels, not(ntt_wide_integers)))]
use super::BIAS;
#[cfg(ntt_fixed_kernels)]
use super::{FloatField, Kernels, Scalar};

// The portable kernels are the reference for every backend, on targets where
// production compiles neither them nor the schedule they instantiate.
#[cfg(not(ntt_expression_schedule))]
#[macro_use]
#[path = "../schedule/expression/mod.rs"]
mod expression;
#[cfg(ntt_fixed_kernels)]
#[path = "../scalar/kernels.rs"]
mod portable;

mod backends;
#[cfg(any(ntt_avx2, ntt_avx512))]
mod coefficients;
mod finish;
mod leaves;
mod packing;
mod pointwise;
mod prefixes;
mod residues;
mod root;
#[cfg(ntt_avx512)]
mod stages;
mod tail;

pub use backends::admitted;
