//! Two-lane WebAssembly `simd128` kernels.
//!
//! WebAssembly has no fused multiply-add and no 64-bit integer conversion of
//! `f64x2`, so exact products split both factors with Dekker's algorithm in
//! `f64x2` arithmetic, as the portable kernels do on targets without either
//! (Dekker 1971). `f64x2.nearest` rounds ties to even.

#![expect(
    unsafe_code,
    reason = "The compile-time simd128 gate admits the intrinsics; kernel contracts bound all initialized vector spans"
)]

use core::arch::wasm32::{
    f64x2_add, f64x2_lt, f64x2_mul, f64x2_nearest, f64x2_splat, f64x2_sub, i64x2_shuffle, v128,
    v128_and, v128_load, v128_store, v128_xor,
};

use super::{BIAS, Kernels};

/// `2^27 + 1`, Veltkamp's constant for binary64: it splits a float into two
/// halves of at most 26 significant bits each.
const SPLITTER: f64 = 134_217_729.0;

/// WebAssembly `simd128` residue kernels.
#[derive(Debug)]
pub struct Simd128;

schedule_backend!(
    Simd128,
    v128,
    2,
    [target_feature(enable = "simd128")],
    |values: *const f64| v128_load(values.cast()),
    |values: *mut f64, vector: v128| v128_store(values.cast(), vector),
    f64x2_splat,
    f64x2_add,
    f64x2_sub,
    f64x2_mul,
    reduce_vector,
    multiply_vector,
    |a, p| {
        let negative = f64x2_lt(a, f64x2_splat(0.0));
        let canonical = f64x2_add(a, v128_and(negative, p));
        let bias = f64x2_splat(BIAS);
        v128_xor(f64x2_add(canonical, bias), bias)
    }
);

impl Simd128 {
    /// Two-lane `simd128` stages, paired-vector leaves, and products.
    pub const KERNELS: Kernels = kernel_table!(Simd128, Simd128, None, None, None);
}

leaf16_pair_backend!(
    Simd128,
    [target_feature(enable = "simd128")],
    v128,
    f64x2_splat,
    |values: *const f64| v128_load(values.cast()),
    |values: *mut f64, vector: v128| v128_store(values.cast(), vector),
    i64x2_shuffle::<0, 2>,
    i64x2_shuffle::<1, 3>
);

/// Reduces integer-valued lanes of magnitude below `8*p`.
///
/// A quotient of magnitude at most eight makes its product with `p < 2^50` and
/// the difference exact.
#[inline]
fn reduce_vector(a: v128, p: v128, pinv: v128) -> v128 {
    f64x2_sub(a, f64x2_mul(f64x2_nearest(f64x2_mul(a, pinv)), p))
}

/// Computes exact modular products for `abs(a*b) < 4*p^2`.
///
/// Veltkamp splitting gives `a*b = h + l` and `q*p = g + m` exactly, so this
/// returns the remainder of `Scalar::mulmod`, bound by the same proof.
#[inline]
fn multiply_vector(a: v128, b: v128, p: v128, pinv: v128) -> v128 {
    let split = |x: v128| {
        let scaled = f64x2_mul(x, f64x2_splat(SPLITTER));
        let high = f64x2_sub(scaled, f64x2_sub(scaled, x));
        (high, f64x2_sub(x, high))
    };
    let product = |x: v128, y: v128| {
        let high = f64x2_mul(x, y);
        let ((x_high, x_low), (y_high, y_low)) = (split(x), split(y));
        let low = f64x2_add(
            f64x2_add(
                f64x2_add(
                    f64x2_sub(f64x2_mul(x_high, y_high), high),
                    f64x2_mul(x_high, y_low),
                ),
                f64x2_mul(x_low, y_high),
            ),
            f64x2_mul(x_low, y_low),
        );
        (high, low)
    };
    let (high, low) = product(a, b);
    let (scaled, error) = product(f64x2_nearest(f64x2_mul(high, pinv)), p);
    f64x2_add(f64x2_sub(high, scaled), f64x2_sub(low, error))
}
