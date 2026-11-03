//! Eight independent CRT coefficients using radix-2^52 IFMA accumulators.

#![expect(
    unsafe_code,
    reason = "Admitted cofactor tables, coefficient spans, and ISA features prove SIMD products and stores"
)]

#[cfg(target_arch = "x86")]
use core::arch::x86::{
    __m512i, _mm512_add_epi64, _mm512_and_si512, _mm512_castpd_si512, _mm512_castsi512_pd,
    _mm512_cvttpd_epu64, _mm512_fnmadd_pd, _mm512_loadu_si512, _mm512_madd52hi_epu64,
    _mm512_madd52lo_epu64, _mm512_mul_epu32, _mm512_or_si512, _mm512_permutexvar_epi64,
    _mm512_set1_epi64, _mm512_set1_pd, _mm512_setr_epi64, _mm512_setzero_pd, _mm512_setzero_si512,
    _mm512_sllv_epi64, _mm512_srli_epi64, _mm512_srlv_epi64, _mm512_storeu_si512, _mm512_sub_epi64,
    _mm512_sub_pd, _mm512_xor_si512,
};
#[cfg(target_arch = "x86_64")]
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

/// Transposed cofactors for a nonempty catalog prefix.
#[derive(Clone, Copy)]
struct Radix52 {
    /// Digits of `q*P` indexed by digit and then by the quotient `q`.
    multiples: [[u64; CRT_LANES]; RADIX_WORDS],
    factors: [[u64; MAX_PRIMES]; RADIX_WORDS],
    quotient_offset: f64,
}

/// SIMD integer products over canonical 50-bit residues.
#[derive(Debug)]
pub struct Ifma;

impl Ifma {
    /// Packet kernels indexed by the prime count minus one.
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
        let (words, digits, factor_words) = const {
            let capacity = Cofactors::of_count(COUNT).capacity;
            (
                capacity.div_ceil(64),
                capacity.div_ceil(52),
                if COUNT == 1 {
                    1
                } else {
                    (MODULUS_BITS * (COUNT - 1)).div_ceil(52)
                },
            )
        };
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

/// Returns the bit offset of each lane below 64, and 63 minus that offset.
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

#[inline]
#[target_feature(enable = "avx512f,avx512dq,avx512ifma")]
fn estimate_quotients<const COUNT: usize>(table: &Radix52, rows: &[__m512i; COUNT]) -> __m512i {
    // Each row holds canonical integers below 2^50. Setting the exponent bits
    // of 2^52 constructs exactly 2^52+r_i. The offset cancels the matching
    // 2^52/p_i terms. Constant sums stay below 9*n and fused partial sums
    // below 10*n, bounding absolute error by 20*2^-53*n*(n+3) < 1/16. With
    // X/P < 2/3, truncation of the biased estimate gives q in 0..n. These
    // positive finite values fit u64.
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
        let mut table = Radix52 {
            multiples: [[0; CRT_LANES]; RADIX_WORDS],
            factors: [[0; MAX_PRIMES]; RADIX_WORDS],
            quotient_offset: -0.25,
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
                *(*table.factors.as_mut_ptr().add(digit))
                    .as_mut_ptr()
                    .add(row) = ((factor_pair >> offset) & mask) as u64;
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
