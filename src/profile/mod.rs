//! Profile catalog and selection for multi-modulus floating-point NTT.

use crate::arch::{COLUMN_ALIGN, Columns, MIN_TRANSFORM_LEN};

use super::{
    Cofactors, LIMB_BITS, MAX_PARTS, MAX_PRIMES, MAX_ROOT_ORDER, PART_BITS, PrimePrefix, WORDS,
};

mod catalog;
mod planner;

pub use catalog::{MAX_CHUNK_BITS, PROFILES, Profile};
pub use planner::{Geometry, Operand, Split};
