//! Prime fields: the transform's prime catalog and public modular arithmetic.

mod catalog;
mod prime_field;

pub use catalog::{
    FloatField, MAX_MODULUS, MAX_PRIMES, MAX_ROOT_ORDER, MIN_MODULUS, MODULUS_BITS, PRIME_CATALOG,
    Prime, PrimePrefix, WITNESSES,
};
pub use prime_field::{CrtPair, PrimeField, Residue};

#[cfg(test)]
mod tests;
