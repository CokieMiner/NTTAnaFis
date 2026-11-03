//! z/Architecture vector-facility kernels (z13 and later).
//!
//! Two-lane intrinsic functions from [`zarch`](super::zarch) instantiate the
//! shared schedules; the selector detects the facility at run time. The
//! facility has no negated multiply-subtract, so `RN(c - a*b)` negates one
//! factor with `vflcdb` before `vfmadb`.

#![expect(
    unsafe_code,
    reason = "The selector detects the vector facility; kernel contracts bound all initialized vector spans"
)]

use super::{
    BIAS, Kernels, Vector, vfadb, vfchdb, vfidb, vflcdb, vfmadb, vfmdb, vfsdb, vl, vmrhg, vmrlg,
    vn, vrepg, vst, vx, vzero,
};

vector_backend!(
    Zvector,
    Vector,
    2,
    [target_feature(enable = "vector")],
    vl,
    vst,
    vrepg,
    vfadb,
    vfsdb,
    vfmdb,
    |a, b, c| vfmadb(vflcdb(a), b, c),
    vfidb,
    |a, p| {
        let negative = vfchdb(vzero(), a);
        let canonical = vfadb(a, vn(negative, p));
        let bias = vrepg(BIAS);
        vx(vfadb(canonical, bias), bias)
    }
);

impl Zvector {
    /// Vector-facility stages, leaves, and products.
    pub const KERNELS: Kernels = kernel_table!(Zvector, Zvector, None, None, None);
}

leaf16_pair_backend!(
    Zvector,
    [target_feature(enable = "vector")],
    Vector,
    vrepg,
    vl,
    vst,
    vmrhg,
    vmrlg
);
