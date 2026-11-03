//! Remainder-tree transforms of one residue row.
//!
//! Every method of [`Transform`] states only what it adds to the terms below, which
//! plan admission establishes for each row. `arch/contract.rs` states the residue
//! bounds.
//!
//! # Terms
//!
//! - **Transform**: `data` holds a power of two of at least `MIN_TRANSFORM_LEN`
//!   points, so it holds complete vectors and an even number of leaves.
//! - **Row**: the `N` points of one remainder tree, or `3*M` points with `N = 2*M`:
//!   the tree of `x^N-1` followed by the tree of `x^M-i` at `TAIL_NODE`.
//! - **Roots**: [`Roots`] names a node `j` of a table that covers the node and every
//!   descendant of a transform of this length.
//! - **Input prefix**: a positive multiple of `COLUMN_ALIGN` columns that fits the
//!   transform. A prefix shorter than the transform requires node zero, and missing
//!   coefficients represent zero.
//! - **Budget**: see `partition`.

use crate::arch::{
    COLUMN_ALIGN, Columns, FinalKernel, Kernels, LEAF_POINTS, MIN_TRANSFORM_LEN, Scalar,
    StageKernel,
};

use super::{Executor, FloatField, Partition, Prime};

mod finish;
mod forward;
mod inverse;
mod stage;
mod twiddle;

pub use forward::{CACHE_LEAF, PARALLEL_MIN_LEN, ROOT_MIN_LEN, TAIL_NODE, Transform};
pub use inverse::Normalization;
pub use stage::COLUMN_GRAIN;
pub use twiddle::{Roots, Tables, Twiddles};

#[cfg(test)]
mod tests;
