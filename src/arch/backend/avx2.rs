//! Four-lane AVX2 and FMA kernels for x86 and x86-64.

#![expect(
    unsafe_code,
    reason = "The selector validates AVX2 and FMA before calling span-checked intrinsic kernels"
)]

#[cfg(target_arch = "x86")]
use core::arch::x86::{
    __m256d, _mm256_add_pd, _mm256_and_pd, _mm256_castpd_si256, _mm256_castsi256_pd, _mm256_cmp_pd,
    _mm256_fnmadd_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_permute2f128_pd,
    _mm256_permute4x64_pd, _mm256_round_pd, _mm256_set1_pd, _mm256_setzero_pd, _mm256_storeu_pd,
    _mm256_sub_pd, _mm256_unpackhi_pd, _mm256_unpacklo_pd, _mm256_xor_si256,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m256d, _mm256_add_pd, _mm256_and_pd, _mm256_castpd_si256, _mm256_castsi256_pd, _mm256_cmp_pd,
    _mm256_fnmadd_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_permute2f128_pd,
    _mm256_permute4x64_pd, _mm256_round_pd, _mm256_set1_pd, _mm256_setzero_pd, _mm256_storeu_pd,
    _mm256_sub_pd, _mm256_unpackhi_pd, _mm256_unpacklo_pd, _mm256_xor_si256,
};

use super::{BIAS, Kernels};

vector_backend!(
    Avx2,
    __m256d,
    4,
    [target_feature(enable = "avx2,fma")],
    _mm256_loadu_pd,
    _mm256_storeu_pd,
    _mm256_set1_pd,
    _mm256_add_pd,
    _mm256_sub_pd,
    _mm256_mul_pd,
    _mm256_fnmadd_pd,
    |a| _mm256_round_pd::<8>(a),
    |a, p| {
        let negative = _mm256_cmp_pd::<1>(a, _mm256_setzero_pd());
        let canonical = _mm256_add_pd(a, _mm256_and_pd(negative, p));
        let bias = _mm256_set1_pd(BIAS);
        _mm256_castsi256_pd(_mm256_xor_si256(
            _mm256_castpd_si256(_mm256_add_pd(canonical, bias)),
            _mm256_castpd_si256(bias),
        ))
    }
);

impl Avx2 {
    /// Kernels requiring AVX2 and FMA, admitted by the selector.
    pub const KERNELS: Kernels = Kernels {
        crt_cost: 3,
        ..kernel_table!(Avx2, Avx2, None, None, None)
    };
}

leaf16_backend!(
    Avx2,
    [target_feature(enable = "avx2,fma")],
    16,
    _mm256_set1_pd,
    |values: *const f64| (
        _mm256_loadu_pd(values),
        _mm256_loadu_pd(values.add(4)),
        _mm256_loadu_pd(values.add(8)),
        _mm256_loadu_pd(values.add(12)),
    ),
    |rows| rows,
    |values: *mut f64, (a, b, c, d)| {
        _mm256_storeu_pd(values, a);
        _mm256_storeu_pd(values.add(4), b);
        _mm256_storeu_pd(values.add(8), c);
        _mm256_storeu_pd(values.add(12), d);
    },
    |roots: *const f64, node: usize| {
        let twice = node.unchecked_mul(2);
        let four = node.unchecked_mul(4);
        let eight = node.unchecked_mul(8);
        let first = _mm256_loadu_pd(roots.add(eight));
        let second = _mm256_loadu_pd(roots.add(eight.unchecked_add(4)));
        (
            (
                _mm256_set1_pd(*roots.add(node)),
                _mm256_set1_pd(*roots.add(twice)),
                _mm256_set1_pd(*roots.add(twice.unchecked_add(1))),
            ),
            (
                _mm256_loadu_pd(roots.add(four)),
                _mm256_permute4x64_pd::<0xd8>(_mm256_unpacklo_pd(first, second)),
                _mm256_permute4x64_pd::<0xd8>(_mm256_unpackhi_pd(first, second)),
            ),
        )
    },
    |(a, b, c, d)| {
        let low_ab = _mm256_unpacklo_pd(a, b);
        let high_ab = _mm256_unpackhi_pd(a, b);
        let low_cd = _mm256_unpacklo_pd(c, d);
        let high_cd = _mm256_unpackhi_pd(c, d);
        (
            _mm256_permute2f128_pd::<0x20>(low_ab, low_cd),
            _mm256_permute2f128_pd::<0x20>(high_ab, high_cd),
            _mm256_permute2f128_pd::<0x31>(low_ab, low_cd),
            _mm256_permute2f128_pd::<0x31>(high_ab, high_cd),
        )
    }
);
