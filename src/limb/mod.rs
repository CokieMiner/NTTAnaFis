//! Conversion between native limbs and residue vectors.
//!
//! Packing splits limbs into binary digits and reduces them into each
//! selected prime field. Reconstruction recovers every coefficient from its
//! residues by cofactor CRT and propagates carries into native limbs.

use super::{
    Executor, Geometry, LIMB_BITS, Limb, MAX_PRIMES, MODULUS_BITS, Operand, PRIME_CATALOG,
    Partition, PrimePrefix,
};

mod cofactor;
mod packing;
mod reconstruction;

pub use cofactor::{CRT_LANES, Cofactors, WORDS};
pub use packing::{DigitTile, MAX_PARTS, PACKING_DATA, PACKING_TILE, PART_BITS, Packing};
pub use reconstruction::Crt;

#[cfg(test)]
mod tests;
