//! Eight independent CRT coefficients using radix-2^52 IFMA or FMA accumulators.
//!
//! Each coefficient is evaluated in radix `2^52` over the precomputed digits of
//! `P/p_i` and `q*P`. AVX-512 IFMA splits each 52x50-bit product with
//! `vpmadd52luq` and `vpmadd52huq`. AVX2 and AVX-512F split the same product in
//! binary64: `h = RN(r*F + 2^104) - 2^104` rounds `r*F` to the nearest multiple
//! of `2^52`, leaving `l = r*F - h` in `[-2^51, 2^51]`. Adding `3*2^51` places
//! `l` in the mantissa bits, and a fixed offset per word makes the low sum
//! nonnegative before its carry shift.

#![expect(
    unsafe_code,
    reason = "Admitted cofactor tables, coefficient spans, and ISA features prove SIMD products and stores"
)]

#[cfg(all(not(all(ntt_fixed_kernels, ntt_ifma)), target_arch = "x86"))]
use core::arch::x86::{
    __m256d, __m256i, _mm256_add_epi32, _mm256_add_epi64, _mm256_add_pd, _mm256_and_si256,
    _mm256_blendv_epi8, _mm256_castpd_si256, _mm256_castsi256_pd, _mm256_cmpgt_epi32,
    _mm256_fmadd_pd, _mm256_fmsub_pd, _mm256_fnmadd_pd, _mm256_loadu_si256, _mm256_mul_epu32,
    _mm256_or_pd, _mm256_or_si256, _mm256_permutevar8x32_epi32, _mm256_set1_epi32,
    _mm256_set1_epi64x, _mm256_set1_pd, _mm256_setr_epi32, _mm256_setr_epi64x,
    _mm256_setzero_si256, _mm256_shuffle_epi32, _mm256_sllv_epi64, _mm256_srli_epi64,
    _mm256_srlv_epi64, _mm256_storeu_si256, _mm256_sub_epi64, _mm256_sub_pd, _mm256_xor_si256,
};
#[cfg(all(ntt_ifma, target_arch = "x86"))]
use core::arch::x86::{
    __m512i, _mm512_add_epi64, _mm512_and_si512, _mm512_castpd_si512, _mm512_castsi512_pd,
    _mm512_cvttpd_epu64, _mm512_fnmadd_pd, _mm512_loadu_si512, _mm512_madd52hi_epu64,
    _mm512_madd52lo_epu64, _mm512_mul_epu32, _mm512_or_si512, _mm512_permutexvar_epi64,
    _mm512_set1_epi64, _mm512_set1_pd, _mm512_setr_epi64, _mm512_setzero_pd, _mm512_setzero_si512,
    _mm512_sllv_epi64, _mm512_srli_epi64, _mm512_srlv_epi64, _mm512_storeu_si512, _mm512_sub_epi64,
    _mm512_sub_pd, _mm512_xor_si512,
};
#[cfg(all(not(all(ntt_fixed_kernels, ntt_ifma)), target_arch = "x86_64"))]
use core::arch::x86_64::{
    __m256d, __m256i, _mm256_add_epi32, _mm256_add_epi64, _mm256_add_pd, _mm256_and_si256,
    _mm256_blendv_epi8, _mm256_castpd_si256, _mm256_castsi256_pd, _mm256_cmpgt_epi32,
    _mm256_fmadd_pd, _mm256_fmsub_pd, _mm256_fnmadd_pd, _mm256_loadu_si256, _mm256_mul_epu32,
    _mm256_or_pd, _mm256_or_si256, _mm256_permutevar8x32_epi32, _mm256_set1_epi32,
    _mm256_set1_epi64x, _mm256_set1_pd, _mm256_setr_epi32, _mm256_setr_epi64x,
    _mm256_setzero_si256, _mm256_shuffle_epi32, _mm256_sllv_epi64, _mm256_srli_epi64,
    _mm256_srlv_epi64, _mm256_storeu_si256, _mm256_sub_epi64, _mm256_sub_pd, _mm256_xor_si256,
};
#[cfg(all(ntt_ifma, target_arch = "x86_64"))]
use core::arch::x86_64::{
    __m512i, _mm512_add_epi64, _mm512_and_si512, _mm512_castpd_si512, _mm512_castsi512_pd,
    _mm512_cvttpd_epu64, _mm512_fnmadd_pd, _mm512_loadu_si512, _mm512_madd52hi_epu64,
    _mm512_madd52lo_epu64, _mm512_mul_epu32, _mm512_or_si512, _mm512_permutexvar_epi64,
    _mm512_set1_epi64, _mm512_set1_pd, _mm512_setr_epi64, _mm512_setzero_pd, _mm512_setzero_si512,
    _mm512_sllv_epi64, _mm512_srli_epi64, _mm512_srlv_epi64, _mm512_storeu_si512, _mm512_sub_epi64,
    _mm512_sub_pd, _mm512_xor_si512,
};
use core::array::from_fn;

use super::{
    BIAS, CRT_LANES, CoefficientKernel, Cofactors, MAX_PRIMES, MODULUS_BITS, PRIME_CATALOG, WORDS,
};

const RADIX_WORDS: usize = const {
    assert!(
        MAX_PRIMES == 8,
        "quotients select prime-product multiples from one vector"
    );
    (MAX_PRIMES * MODULUS_BITS).div_ceil(52)
};

/// `2^104`. Adding it to an integer product below `2^102` rounds the sum to
/// the nearest multiple of `2^52`, ties to even, and places that multiple's
/// 52-bit index in the low mantissa bits.
#[cfg(not(all(ntt_fixed_kernels, ntt_ifma)))]
const HIGH_BIAS: f64 = f64::from_bits(0x4670_0000_0000_0000);

/// `3 * 2^51`. Adding it to a signed integer in `[-2^51, 2^51]` places that
/// integer's two's complement representation, offset by the constant's bits,
/// in the binary64 bit pattern.
#[cfg(not(all(ntt_fixed_kernels, ntt_ifma)))]
const SIGNED_BIAS: f64 = f64::from_bits(0x4338_0000_0000_0000);

/// Transposed cofactors for a nonempty catalog prefix.
#[derive(Clone, Copy)]
struct Radix52 {
    /// Digits of `q*P` indexed by digit and then by the quotient `q`.
    multiples: [[u64; CRT_LANES]; RADIX_WORDS],
    #[cfg(ntt_ifma)]
    factors: [[u64; MAX_PRIMES]; RADIX_WORDS],
    #[cfg(not(all(ntt_fixed_kernels, ntt_ifma)))]
    float_factors: [[f64; MAX_PRIMES]; RADIX_WORDS],
    quotient_offset: f64,
    #[cfg(not(all(ntt_fixed_kernels, ntt_ifma)))]
    low_offset: i64,
    #[cfg(not(all(ntt_fixed_kernels, ntt_ifma)))]
    high_offset: i64,
    words: usize,
    digits: usize,
    factor_words: usize,
}

/// SIMD CRT packet kernels over canonical 50-bit residues.
#[derive(Debug)]
pub struct Ifma;

#[cfg(ntt_ifma)]
impl Ifma {
    /// AVX-512 IFMA packet kernels indexed by the prime count minus one.
    pub const KERNELS: [CoefficientKernel; MAX_PRIMES] = [
        Self::coefficients::<1>,
        Self::coefficients::<2>,
        Self::coefficients::<3>,
        Self::coefficients::<4>,
        Self::coefficients::<5>,
        Self::coefficients::<6>,
        Self::coefficients::<7>,
        Self::coefficients::<8>,
    ];

    /// Reconstructs eight coefficients into native-word columns, each shifted
    /// to its offset below 64 bits.
    ///
    /// `bit` is the output bit position of the first coefficient, and later
    /// coefficients follow at multiples of `width`.
    ///
    /// # Safety
    /// AVX-512F, AVX-512DQ, and AVX-512IFMA are supported. Each of the first
    /// `COUNT` rows covers `first..first+CRT_LANES` with canonical
    /// cofactor-weighted u64 residue bits. Each coefficient is below the
    /// capacity `c` of the prefix. `bit` and `width` fit 63 bits. The first
    /// `ceil(c/64)+1` output rows are written.
    #[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
    pub unsafe fn coefficients<const COUNT: usize>(
        residues: &[&[f64]; MAX_PRIMES],
        first: usize,
        bit: usize,
        width: usize,
        output: &mut [[u64; CRT_LANES]; WORDS + 1],
    ) {
        let table = const { &radix52_prefix(COUNT - 1) };
        let (words, digits, factor_words) = (table.words, table.digits, table.factor_words);
        // SAFETY: each 52x50-bit product splits into low 52 and high 50 bits.
        // Their sums and incoming carry stay below 2^64. Radix subtraction
        // lies in (-2^52-1, 2^52), so its sign bit is exactly the borrow. The
        // quotient is below COUNT and selects a table digit of q*P. Digit
        // indices fit ceil(50*MAX_PRIMES/52). The stream of ceil(c/52) digits
        // emits floor(ceil(c/52)*52/64) <= ceil(c/64) complete words and at
        // most one partial word; the shifted stream adds one word. Row loads
        // cover eight initialized slots. Bit offsets are masked below 64, and
        // the pending count stays below 64 after each emitted word.
        unsafe {
            let rows: [__m512i; COUNT] = from_fn(|row| {
                _mm512_loadu_si512(residues.get_unchecked(row).as_ptr().add(first).cast())
            });
            let quotient = estimate_quotients::<COUNT>(table, &rows);
            let (offsets, inverse) = lane_offsets(bit, width);
            let mask = _mm512_set1_epi64((1_i64 << 52) - 1);
            let mut sum_carry = _mm512_setzero_si512();
            let mut borrow = _mm512_setzero_si512();
            let mut pending = _mm512_setzero_si512();
            let mut pending_bits = 0_u32;
            let mut previous = _mm512_setzero_si512();
            let mut destination = 0_usize;
            for word in 0..digits {
                let mut low = sum_carry;
                let mut high = _mm512_setzero_si512();
                if word < factor_words {
                    let factors = table.factors.get_unchecked(word);
                    for (row, residue) in rows.iter().enumerate() {
                        let factor = _mm512_set1_epi64((*factors.get_unchecked(row)).cast_signed());
                        low = _mm512_madd52lo_epu64(low, *residue, factor);
                        high = _mm512_madd52hi_epu64(high, *residue, factor);
                    }
                }
                sum_carry = _mm512_add_epi64(_mm512_srli_epi64::<52>(low), high);
                let multiple = _mm512_permutexvar_epi64(
                    quotient,
                    _mm512_loadu_si512(table.multiples.get_unchecked(word).as_ptr().cast()),
                );
                let difference = _mm512_sub_epi64(
                    _mm512_sub_epi64(_mm512_and_si512(low, mask), multiple),
                    borrow,
                );
                borrow = _mm512_srli_epi64::<63>(difference);
                let digit = _mm512_and_si512(difference, mask);
                pending = _mm512_or_si512(
                    pending,
                    _mm512_sllv_epi64(digit, _mm512_set1_epi64(i64::from(pending_bits))),
                );
                let total = pending_bits.unchecked_add(52);
                if total >= 64 {
                    store_shifted(
                        output.get_unchecked_mut(destination),
                        pending,
                        &mut previous,
                        offsets,
                        inverse,
                    );
                    destination = destination.unchecked_add(1);
                    pending = _mm512_srlv_epi64(
                        digit,
                        _mm512_set1_epi64(i64::from(64_u32.unchecked_sub(pending_bits))),
                    );
                    pending_bits = total.unchecked_sub(64);
                } else {
                    pending_bits = total;
                }
            }
            if destination < words {
                store_shifted(
                    output.get_unchecked_mut(destination),
                    pending,
                    &mut previous,
                    offsets,
                    inverse,
                );
            }
            _mm512_storeu_si512(
                output.get_unchecked_mut(words).as_mut_ptr().cast(),
                _mm512_srlv_epi64(_mm512_srli_epi64::<1>(previous), inverse),
            );
        }
    }
}

#[cfg(not(all(ntt_fixed_kernels, ntt_ifma)))]
impl Ifma {
    /// AVX2 and FMA packet kernels indexed by the prime count minus one.
    pub const FMA_KERNELS: [CoefficientKernel; MAX_PRIMES] = [
        Self::coefficients_fma::<1>,
        Self::coefficients_fma::<2>,
        Self::coefficients_fma::<3>,
        Self::coefficients_fma::<4>,
        Self::coefficients_fma::<5>,
        Self::coefficients_fma::<6>,
        Self::coefficients_fma::<7>,
        Self::coefficients_fma::<8>,
    ];

    /// Reconstructs eight coefficients into native-word columns using AVX2 and FMA.
    ///
    /// # Safety
    /// AVX2 and FMA are supported, with the operand and output contracts of
    /// [`Self::coefficients`].
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn coefficients_fma<const COUNT: usize>(
        residues: &[&[f64]; MAX_PRIMES],
        first: usize,
        bit: usize,
        width: usize,
        output: &mut [[u64; CRT_LANES]; WORDS + 1],
    ) {
        let table = const { &radix52_prefix(COUNT - 1) };
        let (words, digits, factor_words) = (table.words, table.digits, table.factor_words);
        // SAFETY: the caller admits AVX2, FMA, and eight initialized slots in
        // each active row. For r < 2^50 and F < 2^52, RN(r*F + 2^104) equals
        // 2^104 + H*2^52 with H in 0..=2^50, and L = r*F - H*2^52 lies in
        // [-2^51, 2^51]. Adding 3*2^51 places L in the mantissa bits. Offsetting
        // low by ceil(COUNT/2)*2^52 and high by -ceil(COUNT/2) keeps low
        // nonnegative and below 2^55, so the logical right shift extracts the
        // exact carry. The two four-lane iterations write disjoint halves of
        // every eight-lane output row.
        unsafe {
            let b_hi = _mm256_set1_pd(HIGH_BIAS);
            let b_lo = _mm256_set1_pd(SIGNED_BIAS);
            let mask = _mm256_set1_epi64x((1_i64 << 52) - 1);
            let base = output.as_mut_ptr().cast::<u64>();
            for half in [0_usize, 4] {
                let target = base.add(half);
                let (rows, idx, high_mask) =
                    fma_inputs::<COUNT>(table, residues, first.unchecked_add(half), b_lo);
                let (offsets, inverse) =
                    lane_offsets_256(bit.unchecked_add(width.unchecked_mul(half)), width);
                let mut sum_carry = _mm256_setzero_si256();
                let mut borrow = _mm256_setzero_si256();
                let mut pending = _mm256_setzero_si256();
                let mut pending_bits = 0_u32;
                let mut previous = _mm256_setzero_si256();
                let mut destination = 0_usize;
                for word in 0..digits {
                    let (mut low, mut high) = if word < factor_words {
                        let mut low =
                            _mm256_add_epi64(sum_carry, _mm256_set1_epi64x(table.low_offset));
                        let mut high = _mm256_set1_epi64x(table.high_offset);
                        let factors = table.float_factors.get_unchecked(word);
                        for (row, &residue) in rows.iter().enumerate() {
                            let factor = _mm256_set1_pd(*factors.get_unchecked(row));
                            let hi_biased = _mm256_fmadd_pd(residue, factor, b_hi);
                            let hi_scaled = _mm256_sub_pd(hi_biased, b_hi);
                            let lo_biased =
                                _mm256_add_pd(_mm256_fmsub_pd(residue, factor, hi_scaled), b_lo);
                            low = _mm256_add_epi64(low, _mm256_castpd_si256(lo_biased));
                            high = _mm256_add_epi64(high, _mm256_castpd_si256(hi_biased));
                        }
                        (low, high)
                    } else {
                        (sum_carry, _mm256_setzero_si256())
                    };
                    sum_carry = _mm256_add_epi64(_mm256_srli_epi64::<52>(low), high);
                    let multiples = table.multiples.get_unchecked(word).as_ptr();
                    let low_mult =
                        _mm256_permutevar8x32_epi32(_mm256_loadu_si256(multiples.cast()), idx);
                    let multiple = if COUNT <= 4 {
                        low_mult
                    } else {
                        let high_mult = _mm256_permutevar8x32_epi32(
                            _mm256_loadu_si256(multiples.add(4).cast()),
                            idx,
                        );
                        _mm256_blendv_epi8(low_mult, high_mult, high_mask)
                    };
                    low = _mm256_and_si256(low, mask);
                    high = _mm256_sub_epi64(_mm256_sub_epi64(low, multiple), borrow);
                    borrow = _mm256_srli_epi64::<63>(high);
                    let digit = _mm256_and_si256(high, mask);
                    pending = _mm256_or_si256(
                        pending,
                        _mm256_sllv_epi64(digit, _mm256_set1_epi64x(i64::from(pending_bits))),
                    );
                    let total = pending_bits.unchecked_add(52);
                    if total >= 64 {
                        store_shifted_256(
                            target.add(destination.unchecked_mul(CRT_LANES)),
                            pending,
                            &mut previous,
                            offsets,
                            inverse,
                        );
                        destination = destination.unchecked_add(1);
                        pending = _mm256_srlv_epi64(
                            digit,
                            _mm256_set1_epi64x(i64::from(64_u32.unchecked_sub(pending_bits))),
                        );
                        pending_bits = total.unchecked_sub(64);
                    } else {
                        pending_bits = total;
                    }
                }
                if destination < words {
                    store_shifted_256(
                        target.add(destination.unchecked_mul(CRT_LANES)),
                        pending,
                        &mut previous,
                        offsets,
                        inverse,
                    );
                }
                _mm256_storeu_si256(
                    target.add(words.unchecked_mul(CRT_LANES)).cast(),
                    _mm256_srlv_epi64(_mm256_srli_epi64::<1>(previous), inverse),
                );
            }
        }
    }
}

/// Returns the bit offset of each lane below 64, and 63 minus that offset.
#[cfg(ntt_ifma)]
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
unsafe fn lane_offsets(bit: usize, width: usize) -> (__m512i, __m512i) {
    // SAFETY: the kernel admits these features, and its contract bounds both
    // values by 63 bits. Widths fit the low 32 bits read by the product.
    unsafe {
        let lanes = _mm512_mul_epu32(
            _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7),
            _mm512_set1_epi64(i64::try_from(width).unwrap_unchecked()),
        );
        let mask = _mm512_set1_epi64(63);
        let offsets = _mm512_and_si512(
            _mm512_add_epi64(
                _mm512_set1_epi64(i64::try_from(bit).unwrap_unchecked()),
                lanes,
            ),
            mask,
        );
        (offsets, _mm512_xor_si512(offsets, mask))
    }
}

/// Stores one coefficient word shifted to the offset of each lane.
#[cfg(ntt_ifma)]
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
unsafe fn store_shifted(
    target: &mut [u64; CRT_LANES],
    word: __m512i,
    previous: &mut __m512i,
    offsets: __m512i,
    inverse: __m512i,
) {
    // SAFETY: the kernel admits AVX-512F and `target` holds eight u64 words.
    // A zero offset shifts the previous word out completely through the
    // preliminary one-bit shift.
    unsafe {
        _mm512_storeu_si512(
            target.as_mut_ptr().cast(),
            _mm512_or_si512(
                _mm512_sllv_epi64(word, offsets),
                _mm512_srlv_epi64(_mm512_srli_epi64::<1>(*previous), inverse),
            ),
        );
    }
    *previous = word;
}

#[cfg(ntt_ifma)]
#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
fn estimate_quotients<const COUNT: usize>(table: &Radix52, rows: &[__m512i; COUNT]) -> __m512i {
    // Each row holds canonical integers below 2^50. Setting the exponent bits
    // of 2^52 constructs 2^52+r_i. The offset cancels the matching 2^52/p_i
    // terms. Constant sums stay below 9*n and fused partial sums below 10*n,
    // bounding absolute error by 20*2^-53*n*(n+3) < 1/16. With X/P < 2/3,
    // truncation of the biased estimate gives q in 0..n.
    let bias = _mm512_castpd_si512(_mm512_set1_pd(BIAS));
    let mut estimate = _mm512_set1_pd(table.quotient_offset);
    for (row, prime) in rows.iter().zip(PRIME_CATALOG) {
        let encoded = _mm512_or_si512(*row, bias);
        estimate = _mm512_fnmadd_pd(
            _mm512_castsi512_pd(encoded),
            _mm512_set1_pd(prime.float_field().pinv()),
            estimate,
        );
    }
    _mm512_cvttpd_epu64(_mm512_sub_pd(_mm512_setzero_pd(), estimate))
}

/// Loads four residues per prime as exact floats and returns their quotient
/// permutation indices and high-half mask.
#[cfg(not(all(ntt_fixed_kernels, ntt_ifma)))]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn fma_inputs<const COUNT: usize>(
    table: &Radix52,
    residues: &[&[f64]; MAX_PRIMES],
    first: usize,
    b_lo: __m256d,
) -> ([__m256d; COUNT], __m256i, __m256i) {
    // SAFETY: `coefficients_fma` establishes AVX2, FMA, and four readable canonical
    // residues in each active row. Adding 0.5 to `quotient_offset` centers the
    // estimate around q, so subtracting it from `3*2^51` rounds ties-to-even to
    // `q` in the low 32-bit word of each lane.
    unsafe {
        let bias = _mm256_set1_pd(BIAS);
        let mut estimate = _mm256_set1_pd(table.quotient_offset + 0.5);
        let rows: [__m256d; COUNT] = from_fn(|row| {
            let raw = _mm256_loadu_si256(residues.get_unchecked(row).as_ptr().add(first).cast());
            let encoded = _mm256_or_pd(_mm256_castsi256_pd(raw), bias);
            let prime = PRIME_CATALOG.get_unchecked(row);
            estimate = _mm256_fnmadd_pd(
                encoded,
                _mm256_set1_pd(prime.float_field().pinv()),
                estimate,
            );
            _mm256_sub_pd(encoded, bias)
        });
        let q_bits = _mm256_castpd_si256(_mm256_sub_pd(b_lo, estimate));
        let q_dup = _mm256_shuffle_epi32::<0xA0>(q_bits);
        let idx = _mm256_add_epi32(
            _mm256_add_epi32(q_dup, q_dup),
            _mm256_setr_epi32(0, 1, 0, 1, 0, 1, 0, 1),
        );
        (rows, idx, _mm256_cmpgt_epi32(q_dup, _mm256_set1_epi32(3)))
    }
}

/// Returns the bit offset of each of four lanes below 64, and 63 minus that offset.
#[cfg(not(all(ntt_fixed_kernels, ntt_ifma)))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn lane_offsets_256(bit: usize, width: usize) -> (__m256i, __m256i) {
    // SAFETY: the kernel admits AVX2, and its contract bounds both values by
    // 63 bits. Widths fit the low 32 bits read by the product.
    unsafe {
        let lanes = _mm256_mul_epu32(
            _mm256_setr_epi64x(0, 1, 2, 3),
            _mm256_set1_epi64x(i64::try_from(width).unwrap_unchecked()),
        );
        let mask = _mm256_set1_epi64x(63);
        let offsets = _mm256_and_si256(
            _mm256_add_epi64(
                _mm256_set1_epi64x(i64::try_from(bit).unwrap_unchecked()),
                lanes,
            ),
            mask,
        );
        (offsets, _mm256_xor_si256(offsets, mask))
    }
}

/// Stores four coefficient words shifted to the offset of each lane.
#[cfg(not(all(ntt_fixed_kernels, ntt_ifma)))]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn store_shifted_256(
    target: *mut u64,
    word: __m256i,
    previous: &mut __m256i,
    offsets: __m256i,
    inverse: __m256i,
) {
    // SAFETY: the kernel admits AVX2 and `target` covers four u64 words.
    unsafe {
        _mm256_storeu_si256(
            target.cast(),
            _mm256_or_si256(
                _mm256_sllv_epi64(word, offsets),
                _mm256_srlv_epi64(_mm256_srli_epi64::<1>(*previous), inverse),
            ),
        );
    }
    *previous = word;
}

/// Transposes the cofactors of the prefix of `prefix + 1` primes.
///
/// # Panics
/// Outside the catalog. Kernels evaluate this in a constant, where that fails the
/// build.
#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Each extracted radix digit is masked to 52 bits before narrowing to u64"
)]
const fn radix52_prefix(prefix: usize) -> Radix52 {
    assert!(prefix < MAX_PRIMES, "prefix exceeds the catalog");
    // SAFETY: prefix < MAX_PRIMES bounds the selected cofactor table and all
    // prime rows. Each radix digit starts below 50*MAX_PRIMES bits, so its low
    // binary word exists. The optional next word is checked. Positions and
    // products are bounded by addressable array sizes; extracted digits fit
    // 52 bits.
    unsafe {
        let count = prefix.unchecked_add(1);
        let source = Cofactors::of_count(count);
        #[cfg(not(all(ntt_fixed_kernels, ntt_ifma)))]
        let (low_offset, high_offset) = {
            let primes = count as u64;
            let shift = count.div_ceil(2) as u64;
            (
                (shift << 52)
                    .wrapping_sub(primes.wrapping_mul(SIGNED_BIAS.to_bits()))
                    .cast_signed(),
                0_u64
                    .wrapping_sub(shift)
                    .wrapping_sub(primes.wrapping_mul(HIGH_BIAS.to_bits()))
                    .cast_signed(),
            )
        };
        let mut table = Radix52 {
            multiples: [[0; CRT_LANES]; RADIX_WORDS],
            #[cfg(ntt_ifma)]
            factors: [[0; MAX_PRIMES]; RADIX_WORDS],
            #[cfg(not(all(ntt_fixed_kernels, ntt_ifma)))]
            float_factors: [[0.0; MAX_PRIMES]; RADIX_WORDS],
            quotient_offset: -0.25,
            #[cfg(not(all(ntt_fixed_kernels, ntt_ifma)))]
            low_offset,
            #[cfg(not(all(ntt_fixed_kernels, ntt_ifma)))]
            high_offset,
            words: source.capacity.div_ceil(64),
            digits: source.capacity.div_ceil(52),
            factor_words: if count == 1 {
                1
            } else {
                MODULUS_BITS.unchecked_mul(prefix).div_ceil(52)
            },
        };
        let mask = (1_u128 << 52) - 1;
        let mut digit = 0_usize;
        while digit < RADIX_WORDS {
            let position = digit.unchecked_mul(52);
            let word = position >> 6;
            let offset = position % 64;
            let next = word.unchecked_add(1);
            let mut quotient = 0_usize;
            while quotient < count {
                let multiple = &*source.multiples.as_ptr().add(quotient);
                let multiple_high = if next < WORDS {
                    *multiple.as_ptr().add(next)
                } else {
                    0
                };
                let multiple_pair =
                    (*multiple.as_ptr().add(word) as u128) | ((multiple_high as u128) << 64);
                *(*table.multiples.as_mut_ptr().add(digit))
                    .as_mut_ptr()
                    .add(quotient) = ((multiple_pair >> offset) & mask) as u64;
                quotient = quotient.unchecked_add(1);
            }
            let mut row = 0_usize;
            while row < count {
                let factor = &*source.factors.as_ptr().add(row);
                let factor_high = if next < WORDS {
                    *factor.as_ptr().add(next)
                } else {
                    0
                };
                let factor_pair =
                    (*factor.as_ptr().add(word) as u128) | ((factor_high as u128) << 64);
                let extracted = ((factor_pair >> offset) & mask) as u64;
                #[cfg(ntt_ifma)]
                {
                    *(*table.factors.as_mut_ptr().add(digit))
                        .as_mut_ptr()
                        .add(row) = extracted;
                }
                #[cfg(not(all(ntt_fixed_kernels, ntt_ifma)))]
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "52-bit radix digits convert to binary64 without rounding"
                )]
                {
                    *(*table.float_factors.as_mut_ptr().add(digit))
                        .as_mut_ptr()
                        .add(row) = extracted as f64;
                }
                row = row.unchecked_add(1);
            }
            digit = digit.unchecked_add(1);
        }
        let mut row = 0_usize;
        while row < count {
            let scaled = BIAS * (&*PRIME_CATALOG.as_ptr().add(row)).float_field().pinv();
            table.quotient_offset += scaled;
            row = row.unchecked_add(1);
        }
        table
    }
}
