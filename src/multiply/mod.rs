//! Prime convolutions, parallel native-limb reconstruction, and their storage.
//!
//! [`Layout`] and [`Retained`] are the safe entries. They check that the
//! operands and the destination match an admitted geometry and that the storage fits
//! it, so that everything below them runs without checks.

use crate::{
    Packing,
    arch::Kernels,
    limb::{Cofactors, Crt},
    transform::{Normalization, PARALLEL_MIN_LEN, Tables, Transform, Twiddles},
};

use super::{
    Executor, Geometry, LIMB_BITS, Limb, PRIME_CATALOG, Partition, PrimePrefix, Sequential, Split,
};

mod execute;
mod schedule;
mod storage;

pub use execute::{Multiplication, RunContext};
pub use schedule::PRODUCT_FUSION_MIN_LEN;
pub use storage::{Layout, Retained};
