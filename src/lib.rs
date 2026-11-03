//! Exact multi-prime number-theoretic transforms for native-limb multiplication.
//!
//! Operands and results are slices of native unsigned words ([`usize`]) stored
//! least-significant limb first. A planned product is exact and occupies exactly
//! `a.len() + b.len()` limbs; a square occupies exactly `2 * a.len()` limbs.
//!
//! # Execution models
//!
//! * [`MulPlan`] proves the transform geometry and sizes caller-owned scratch.
//!   Its [`MulPlan::mul`] and [`MulPlan::sqr`] methods execute without retained
//!   state. Root tables are regenerated in the scratch buffer on every call.
//! * [`MulWorkspace`] retains transform tables and coefficient arenas across
//!   calls. It is the preferred model for repeated products: its tables serve
//!   every shorter transform and grow only when a longer shape arrives.
//! * [`ConvPlan`] and [`ConvWorkspace`] compute exact dense convolutions of
//!   nonnegative coefficient vectors, using the fewest primes that hold the
//!   caller's coefficient bound. Each result coefficient occupies a whole number
//!   of native limbs, so coefficient blocks never carry into one another. A short
//!   operand is summed directly and needs no scratch.
//!
//! Every model rejects unsupported shapes and short buffers with `false` and
//! leaves the destination unchanged on failure. Success writes only the product
//! prefix, preserving any additional destination limbs. Passing the same slice
//! as both operands runs the square path.
//!
//! # Modular arithmetic and Chinese remaindering
//!
//! * [`PrimeField`] exposes modular arithmetic in 50-bit prime fields
//!   (`2^49 < p < 2^50`).
//! * [`Residue`] represents canonical integers in `0..p`.
//! * [`CrtPair`] reconstructs signed 128-bit integers from two distinct prime fields
//!   using Garner's algorithm.
//!
//! # Planning
//!
//! [`MulPlan::new`] validates a pair of positive operand lengths. A plan can be
//! reused for every call whose operand lengths match. Planning returns `None`
//! when a shape exceeds the capacity of the prime catalog or the addressable
//! buffer geometry.
//!
//! # Examples
//!
//! Caller-owned scratch multiplication:
//!
//! ```
//! use ntt_anafis::MulPlan;
//!
//! let a: Vec<usize> = vec![0xFFFF_FFFF; 64];
//! let b: Vec<usize> = vec![0x1234_5678; 48];
//! let mut dst = vec![0usize; a.len() + b.len()];
//!
//! let plan = MulPlan::new(a.len(), b.len()).unwrap();
//! let mut scratch = vec![0usize; plan.scratch_len()];
//!
//! assert!(plan.mul(&mut dst, &a, &b, &mut scratch));
//! ```
//!
//! Retained workspace for repeated shapes:
//!
//! ```
//! use ntt_anafis::{MulPlan, MulWorkspace};
//!
//! let a: Vec<usize> = vec![42; 128];
//! let b: Vec<usize> = vec![7; 128];
//! let mut dst = vec![0usize; a.len() + b.len()];
//!
//! let plan = MulPlan::new(a.len(), b.len()).unwrap();
//! let mut workspace = MulWorkspace::new();
//! workspace.prepare(plan);
//!
//! // Subsequent calls with the same shape reuse the prepared tables.
//! assert!(workspace.mul(&mut dst, &a, &b));
//! ```
//!
//! Squaring:
//!
//! ```
//! use ntt_anafis::MulPlan;
//!
//! let a: Vec<usize> = vec![3; 96];
//! let mut dst = vec![0usize; 2 * a.len()];
//!
//! let plan = MulPlan::new(a.len(), a.len()).unwrap();
//! let mut scratch = vec![0usize; plan.sqr_scratch_len()];
//!
//! assert!(plan.sqr(&mut dst, &a, &mut scratch));
//! ```
//!
//! Exact dense convolution:
//!
//! ```
//! use ntt_anafis::ConvPlan;
//!
//! let a: Vec<usize> = vec![1, 2, 3];
//! let b: Vec<usize> = vec![4, 5];
//! let plan = ConvPlan::new(a.len(), b.len(), 3, 5).unwrap();
//! let mut dst = vec![0usize; plan.output_len()];
//! let mut scratch = vec![0usize; plan.scratch_len()];
//!
//! assert!(plan.convolve(&mut dst, &a, &b, &mut scratch));
//! assert_eq!(dst, [4, 13, 22, 15]);
//! ```
//!
//! Modular arithmetic and Chinese remaindering:
//!
//! ```
//! use ntt_anafis::{CrtPair, PrimeField, Residue};
//!
//! let mut catalog = PrimeField::catalog();
//! let f1 = catalog.next().unwrap();
//! let f2 = catalog.next().unwrap();
//!
//! let r1 = f1.reduce(12345);
//! let r2 = f2.reduce(12345);
//!
//! let pair = CrtPair::new(f1, f2).unwrap();
//! assert_eq!(pair.combine(r1, r2), 12345);
//! ```
//!
//! # Empty operands
//!
//! [`MulWorkspace::mul`] and [`MulWorkspace::sqr`] accept empty operands and
//! write zero. A plan requires positive operand lengths, so callers of the
//! scratch path should handle an empty operand directly.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

mod arch;
mod convolve;
mod field;
mod limb;
mod multiply;
mod partition;
mod plan;
mod profile;
mod transform;

pub(crate) use core_anafis::{Budgeted, DoubleLimb, Executor, LIMB_BITS, Limb, Sequential};
pub(crate) use field::{
    FloatField, MAX_PRIMES, MAX_ROOT_ORDER, MODULUS_BITS, PRIME_CATALOG, Prime, PrimePrefix,
};
pub(crate) use limb::{
    CRT_LANES, Cofactors, DigitTile, MAX_PARTS, PACKING_DATA, PACKING_TILE, PART_BITS, Packing,
    WORDS,
};
pub(crate) use partition::Partition;
pub(crate) use profile::{Geometry, Operand};

// Stable public surface
pub use convolve::{ConvPlan, ConvWorkspace};
pub use field::{CrtPair, PrimeField, Residue};
pub use plan::{MulPlan, MulWorkspace};

#[cfg(test)]
mod tests;
