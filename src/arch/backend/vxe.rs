//! z/Architecture vector-enhancements-1 kernels (z14 and later).
//!
//! The same two-lane schedules as the vector facility; the selector detects
//! the enhancements facility at run time. Its negated multiply-subtract
//! `vfnmsdb` evaluates `RN(c - a*b)` in one instruction.

#![expect(
    unsafe_code,
    reason = "The selector detects the vector enhancements facility 1; kernel contracts bound all initialized vector spans"
)]

use super::{
    BIAS, Kernels, Vector, vfadb, vfchdb, vfidb, vfmdb, vfnmsdb, vfsdb, vl, vmrhg, vmrlg, vn,
    vrepg, vst, vx, vzero,
};

vector_backend!(
    Vxe,
    Vector,
    2,
    [target_feature(enable = "vector,vector-enhancements-1")],
    vl,
    vst,
    vrepg,
    vfadb,
    vfsdb,
    vfmdb,
    vfnmsdb,
    vfidb,
    |a, p| {
        let negative = vfchdb(vzero(), a);
        let canonical = vfadb(a, vn(negative, p));
        let bias = vrepg(BIAS);
        vx(vfadb(canonical, bias), bias)
    }
);

impl Vxe {
    /// Vector-enhancements-1 stages, leaves, and products.
    pub const KERNELS: Kernels = kernel_table!(Vxe, Vxe, None, None, None);
}

leaf16_pair_backend!(
    Vxe,
    [target_feature(enable = "vector,vector-enhancements-1")],
    Vector,
    vrepg,
    vl,
    vst,
    vmrhg,
    vmrlg
);
