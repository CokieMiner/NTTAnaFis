//! Eight-lane AVX-512F kernels for x86 and x86-64.

#![expect(
    unsafe_code,
    reason = "The selector validates AVX-512F before calling span-checked intrinsic kernels"
)]

#[cfg(target_arch = "x86")]
use core::arch::x86::{
    __m512d, _mm512_add_pd, _mm512_castpd_si512, _mm512_castsi512_pd, _mm512_cmp_pd_mask,
    _mm512_fnmadd_pd, _mm512_loadu_pd, _mm512_mask_add_pd, _mm512_maskz_loadu_pd, _mm512_mul_pd,
    _mm512_permutex2var_pd, _mm512_permutexvar_pd, _mm512_roundscale_pd, _mm512_set1_pd,
    _mm512_setr_epi64, _mm512_setzero_pd, _mm512_storeu_pd, _mm512_sub_pd, _mm512_unpackhi_pd,
    _mm512_unpacklo_pd, _mm512_xor_si512,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m512d, _mm512_add_pd, _mm512_castpd_si512, _mm512_castsi512_pd, _mm512_cmp_pd_mask,
    _mm512_fnmadd_pd, _mm512_loadu_pd, _mm512_mask_add_pd, _mm512_maskz_loadu_pd, _mm512_mul_pd,
    _mm512_permutex2var_pd, _mm512_permutexvar_pd, _mm512_roundscale_pd, _mm512_set1_pd,
    _mm512_setr_epi64, _mm512_setzero_pd, _mm512_storeu_pd, _mm512_sub_pd, _mm512_unpackhi_pd,
    _mm512_unpacklo_pd, _mm512_xor_si512,
};

#[cfg(not(all(ntt_fixed_kernels, ntt_ifma)))]
use super::Ifma;
use super::{BIAS, Columns, FloatField, Kernels};

type FourVectors = (__m512d, __m512d, __m512d, __m512d);
type RootVectors = (__m512d, __m512d, __m512d);

vector_backend!(
    Avx512,
    __m512d,
    8,
    [target_feature(enable = "avx512f")],
    _mm512_loadu_pd,
    _mm512_storeu_pd,
    _mm512_set1_pd,
    _mm512_add_pd,
    _mm512_sub_pd,
    _mm512_mul_pd,
    _mm512_fnmadd_pd,
    |a| _mm512_roundscale_pd::<8>(a),
    |a, p| {
        let negative = _mm512_cmp_pd_mask::<1>(a, _mm512_setzero_pd());
        let canonical = _mm512_mask_add_pd(a, negative, a, p);
        let bias = _mm512_set1_pd(BIAS);
        _mm512_castsi512_pd(_mm512_xor_si512(
            _mm512_castpd_si512(_mm512_add_pd(canonical, bias)),
            _mm512_castpd_si512(bias),
        ))
    }
);

leaf16_backend!(
    Avx512,
    [target_feature(enable = "avx512f")],
    32,
    _mm512_set1_pd,
    |values: *const f64| (
        _mm512_loadu_pd(values),
        _mm512_loadu_pd(values.add(8)),
        _mm512_loadu_pd(values.add(16)),
        _mm512_loadu_pd(values.add(24)),
    ),
    |(first, second, third, fourth)| {
        let low = _mm512_setr_epi64(0, 1, 2, 3, 8, 9, 10, 11);
        let high = _mm512_setr_epi64(4, 5, 6, 7, 12, 13, 14, 15);
        (
            _mm512_permutex2var_pd(first, low, third),
            _mm512_permutex2var_pd(first, high, third),
            _mm512_permutex2var_pd(second, low, fourth),
            _mm512_permutex2var_pd(second, high, fourth),
        )
    },
    |values: *mut f64, (a, b, c, d)| {
        let low = _mm512_setr_epi64(0, 1, 2, 3, 8, 9, 10, 11);
        let high = _mm512_setr_epi64(4, 5, 6, 7, 12, 13, 14, 15);
        _mm512_storeu_pd(values, _mm512_permutex2var_pd(a, low, b));
        _mm512_storeu_pd(values.add(8), _mm512_permutex2var_pd(c, low, d));
        _mm512_storeu_pd(values.add(16), _mm512_permutex2var_pd(a, high, b));
        _mm512_storeu_pd(values.add(24), _mm512_permutex2var_pd(c, high, d));
    },
    |roots: *const f64, node: usize| {
        let twice = node.unchecked_mul(2);
        let four = node.unchecked_mul(4);
        let eight = node.unchecked_mul(8);
        let parent = _mm512_maskz_loadu_pd(0x03, roots.add(node));
        let branches = _mm512_maskz_loadu_pd(0x0f, roots.add(twice));
        let first = _mm512_loadu_pd(roots.add(eight));
        let second = _mm512_loadu_pd(roots.add(eight.unchecked_add(8)));
        (
            (
                _mm512_permutexvar_pd(_mm512_setr_epi64(0, 0, 0, 0, 1, 1, 1, 1), parent),
                _mm512_permutexvar_pd(_mm512_setr_epi64(0, 0, 0, 0, 2, 2, 2, 2), branches),
                _mm512_permutexvar_pd(_mm512_setr_epi64(1, 1, 1, 1, 3, 3, 3, 3), branches),
            ),
            (
                _mm512_loadu_pd(roots.add(four)),
                _mm512_permutex2var_pd(first, _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14), second),
                _mm512_permutex2var_pd(first, _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15), second),
            ),
        )
    },
    |(a, b, c, d)| {
        let low_ab = _mm512_unpacklo_pd(a, b);
        let high_ab = _mm512_unpackhi_pd(a, b);
        let low_cd = _mm512_unpacklo_pd(c, d);
        let high_cd = _mm512_unpackhi_pd(c, d);
        let low = _mm512_setr_epi64(0, 1, 8, 9, 4, 5, 12, 13);
        let high = _mm512_setr_epi64(2, 3, 10, 11, 6, 7, 14, 15);
        (
            _mm512_permutex2var_pd(low_ab, low, low_cd),
            _mm512_permutex2var_pd(high_ab, low, high_cd),
            _mm512_permutex2var_pd(low_ab, high, low_cd),
            _mm512_permutex2var_pd(high_ab, high, high_cd),
        )
    }
);

impl Avx512 {
    /// AVX-512F kernels with FMA CRT packets; the IFMA table replaces the packets.
    #[cfg(not(all(ntt_fixed_kernels, ntt_ifma)))]
    pub const KERNELS: Kernels = Kernels {
        crt_cost: 2,
        ..kernel_table!(
            Avx512,
            Avx512,
            Some(Ifma::FMA_KERNELS),
            Some(Self::radix16::<false>),
            Some(Self::radix16::<true>)
        )
    };

    /// AVX-512F stages, which the IFMA table extends with integer CRT packets.
    #[cfg(all(ntt_fixed_kernels, ntt_ifma))]
    pub const KERNELS: Kernels = Kernels {
        crt_cost: 2,
        ..kernel_table!(
            Avx512,
            Avx512,
            None,
            Some(Self::radix16::<false>),
            Some(Self::radix16::<true>)
        )
    };

    /// Fuses adjacent radix-4 stages across sixteen spans.
    ///
    /// # Safety
    /// The kernel contract holds with residues below `3*p`, or `2*p` for the
    /// inverse. `twiddles` covers fifteen roots: three of the parent followed by
    /// four for each of the three child-root columns.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn radix16<const INVERSE: bool>(
        values: *mut f64,
        twiddles: *const f64,
        quarter: usize,
        columns: Columns,
        field: FloatField,
    ) {
        // SAFETY: the selector establishes AVX-512F. Each vector covers eight
        // admitted indices in sixteen disjoint spans. The four parent columns
        // and four child rows form the same two stages as separate radix-4
        // passes, with the same residue bounds; each child row uses its three
        // admitted constant roots. The fifteen initialized, aligned roots are
        // disjoint from every output store.
        unsafe {
            let (start, end) = columns.complete::<8>();
            let modulus = _mm512_set1_pd(field.p());
            let reciprocal = _mm512_set1_pd(field.pinv());
            let parent_roots = (
                _mm512_set1_pd(*twiddles),
                _mm512_set1_pd(*twiddles.add(1)),
                _mm512_set1_pd(*twiddles.add(2)),
            );
            let mut index = start;
            loop {
                let base = values.add(index);
                // Fixed tuple positions make every intermediate vector a
                // separate value; no addressable temporary matrix is needed.
                if INVERSE {
                    inverse_pair(base, twiddles, quarter, parent_roots, modulus, reciprocal);
                } else {
                    forward_pair(base, twiddles, quarter, parent_roots, modulus, reciprocal);
                }
                index = index.unchecked_add(8);
                if index == end {
                    break;
                }
            }
        }
    }
}

#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn forward_pair(
    base: *mut f64,
    parent: *const f64,
    quarter: usize,
    parent_roots: RootVectors,
    modulus: __m512d,
    reciprocal: __m512d,
) {
    // SAFETY: radix16 admits sixteen disjoint initialized vectors, three
    // parent roots, twelve child roots, and AVX-512F. Fixed tuple coordinates
    // preserve the parent-column then child-row sequence without aliasing.
    unsafe {
        let span = quarter.unchecked_mul(4);
        let column = |offset: usize| {
            butterfly::<false, false, false>(
                load_four(base.add(offset), span),
                parent_roots,
                modulus,
                reciprocal,
            )
        };
        let col0 = column(0);
        let col1 = column(quarter);
        let col2 = column(quarter.unchecked_mul(2));
        let col3 = column(quarter.unchecked_mul(3));
        store_four(
            base,
            quarter,
            child_row::<false>(
                (col0.0, col1.0, col2.0, col3.0),
                parent,
                0,
                modulus,
                reciprocal,
            ),
        );
        store_four(
            base.add(span),
            quarter,
            child_row::<false>(
                (col0.1, col1.1, col2.1, col3.1),
                parent,
                1,
                modulus,
                reciprocal,
            ),
        );
        store_four(
            base.add(span.unchecked_mul(2)),
            quarter,
            child_row::<false>(
                (col0.2, col1.2, col2.2, col3.2),
                parent,
                2,
                modulus,
                reciprocal,
            ),
        );
        store_four(
            base.add(span.unchecked_mul(3)),
            quarter,
            child_row::<false>(
                (col0.3, col1.3, col2.3, col3.3),
                parent,
                3,
                modulus,
                reciprocal,
            ),
        );
    }
}

#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn inverse_pair(
    base: *mut f64,
    parent: *const f64,
    quarter: usize,
    parent_roots: RootVectors,
    modulus: __m512d,
    reciprocal: __m512d,
) {
    // SAFETY: radix16 admits the ISA and all initialized value and root
    // spans. Child rows are disjoint. Parent columns combine only their
    // corresponding row entries, and each output replaces its own vector.
    unsafe {
        let span = quarter.unchecked_mul(4);
        let row = |index: usize| {
            child_row::<true>(
                load_four(base.add(index.unchecked_mul(span)), quarter),
                parent,
                index,
                modulus,
                reciprocal,
            )
        };
        let row0 = row(0);
        let row1 = row(1);
        let row2 = row(2);
        let row3 = row(3);
        store_four(
            base,
            span,
            butterfly::<true, false, false>(
                (row0.0, row1.0, row2.0, row3.0),
                parent_roots,
                modulus,
                reciprocal,
            ),
        );
        store_four(
            base.add(quarter),
            span,
            butterfly::<true, false, false>(
                (row0.1, row1.1, row2.1, row3.1),
                parent_roots,
                modulus,
                reciprocal,
            ),
        );
        store_four(
            base.add(quarter.unchecked_mul(2)),
            span,
            butterfly::<true, false, false>(
                (row0.2, row1.2, row2.2, row3.2),
                parent_roots,
                modulus,
                reciprocal,
            ),
        );
        store_four(
            base.add(quarter.unchecked_mul(3)),
            span,
            butterfly::<true, false, false>(
                (row0.3, row1.3, row2.3, row3.3),
                parent_roots,
                modulus,
                reciprocal,
            ),
        );
    }
}

#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn child_row<const INVERSE: bool>(
    values: FourVectors,
    roots: *const f64,
    row: usize,
    modulus: __m512d,
    reciprocal: __m512d,
) -> FourVectors {
    // SAFETY: paired-stage callers establish AVX-512F, row < 4, and fifteen
    // centered constants. Indices row+3, row+7, and row+11 fit the child
    // columns. Inputs are exact residues in the bounds of the parent stage.
    unsafe {
        let twiddles = (
            _mm512_set1_pd(*roots.add(row.unchecked_add(3))),
            _mm512_set1_pd(*roots.add(row.unchecked_add(7))),
            _mm512_set1_pd(*roots.add(row.unchecked_add(11))),
        );
        butterfly::<INVERSE, false, false>(values, twiddles, modulus, reciprocal)
    }
}

#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn load_four(base: *const f64, stride: usize) -> FourVectors {
    // SAFETY: radix16 admits AVX-512F and four eight-element vectors at
    // base+k*stride, k in 0..4. Every offset fits its full sixteen-span input.
    unsafe {
        (
            _mm512_loadu_pd(base),
            _mm512_loadu_pd(base.add(stride)),
            _mm512_loadu_pd(base.add(stride.unchecked_mul(2))),
            _mm512_loadu_pd(base.add(stride.unchecked_mul(3))),
        )
    }
}

#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn store_four(base: *mut f64, stride: usize, (a, b, c, d): FourVectors) {
    // SAFETY: radix16 owns four disjoint eight-element vectors at
    // base+k*stride, k in 0..4, and establishes AVX-512F before this call.
    unsafe {
        _mm512_storeu_pd(base, a);
        _mm512_storeu_pd(base.add(stride), b);
        _mm512_storeu_pd(base.add(stride.unchecked_mul(2)), c);
        _mm512_storeu_pd(base.add(stride.unchecked_mul(3)), d);
    }
}
