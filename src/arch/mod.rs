//! CPU selection and exact floating-point residue kernels.
//!
//! `contract` states what every kernel assumes. `schedule` writes each kernel once
//! as a macro over register operations, or as block text for targets whose vector
//! registers cannot be `asm!` operands. [`Scalar`] is the schedule with one lane, and
//! each vector ISA in `backend` instantiates it with its own lanes; the selector
//! there resolves one kernel table per operation. `build.rs` names the backends
//! that compile.

use super::{CRT_LANES, DigitTile, FloatField, MAX_PARTS, MAX_PRIMES, Packing, WORDS};
#[cfg(any(ntt_avx2, ntt_avx512))]
use super::{Cofactors, MODULUS_BITS, PRIME_CATALOG};

mod contract;
#[macro_use]
mod schedule;
mod backend;
mod scalar;

pub use contract::{
    BIAS, COLUMN_ALIGN, CoefficientKernel, Columns, FinalKernel, Kernels, LEAF_POINTS,
    MIN_TRANSFORM_LEN, PackingKernel, StageKernel,
};
pub use scalar::Scalar;

#[cfg(test)]
mod tests;
