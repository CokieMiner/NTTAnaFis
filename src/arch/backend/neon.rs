//! Two-lane NEON kernels with fused product-error recovery.
//!
//! Sixteen-point leaves hold each four-lane row as a pair of registers.

#![expect(
    unsafe_code,
    reason = "The module gate establishes NEON; kernel contracts bound all initialized intrinsic spans"
)]

use core::arch::aarch64::{
    float64x2_t, vaddq_f64, vbslq_f64, vcltq_f64, vdupq_n_f64, veorq_u64, vfmsq_f64, vld1q_f64,
    vmulq_f64, vreinterpretq_f64_u64, vreinterpretq_u64_f64, vrndnq_f64, vst1q_f64, vsubq_f64,
    vzip1q_f64, vzip2q_f64,
};

use super::{BIAS, Kernels};

vector_backend!(
    Neon,
    float64x2_t,
    2,
    [target_feature(enable = "neon")],
    vld1q_f64,
    vst1q_f64,
    vdupq_n_f64,
    vaddq_f64,
    vsubq_f64,
    vmulq_f64,
    |a, b, c| vfmsq_f64(c, a, b),
    vrndnq_f64,
    |a, p| {
        let negative = vcltq_f64(a, vdupq_n_f64(0.0));
        let canonical = vbslq_f64(negative, vaddq_f64(a, p), a);
        let bias = vdupq_n_f64(BIAS);
        vreinterpretq_f64_u64(veorq_u64(
            vreinterpretq_u64_f64(vaddq_f64(canonical, bias)),
            vreinterpretq_u64_f64(bias),
        ))
    }
);

impl Neon {
    /// NEON stages, paired-vector leaves, and products.
    pub const KERNELS: Kernels = kernel_table!(Neon, Neon, None, None, None);
}

leaf16_pair_backend!(
    Neon,
    [target_feature(enable = "neon")],
    float64x2_t,
    vdupq_n_f64,
    vld1q_f64,
    vst1q_f64,
    vzip1q_f64,
    vzip2q_f64
);
