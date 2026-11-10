//! Limb and polynomial products against FLINT and GMP on identical operands.
//!
//! Every library runs one thread and keeps its tables between calls. A fair
//! comparison builds all of them for one target: GMP, MPFR, and FLINT for that CPU
//! under `REFERENCE_PREFIX`, and this crate with the same `-C target-cpu`. The crate
//! README lists the commands.
//!
//! `flint_mpn_mul`, `flint_mpn_sqr`, and `fmpz_poly_mul` choose FLINT's best algorithm
//! at each length, up to the small-prime transforms of `fft_small`, whose vector code
//! is AVX2 or NEON. `ntt_poly` and `flint_poly` take coefficients of 16 and 32 bits: at
//! 16 bits every FLINT coefficient stays inline, and at 32 bits results are bignums.

use core::ptr::null_mut;

use divan::{Bencher, black_box};
use ntt_anafis::{ConvPlan, ConvWorkspace, MulPlan, MulWorkspace};
use ntt_anafis_bench::operand;

/// FLINT's `fmpz_poly_struct`.
#[repr(C)]
struct FmpzPoly {
    coeffs: *mut i64,
    alloc: isize,
    length: isize,
}

impl FmpzPoly {
    /// Returns the polynomial with these nonnegative coefficients.
    fn new(values: &[usize]) -> Self {
        let mut poly = Self {
            coeffs: null_mut(),
            alloc: 0,
            length: 0,
        };
        // SAFETY: the first call initializes `poly`, and every index is nonnegative.
        unsafe {
            fmpz_poly_init(&mut poly);
            for (index, value) in values.iter().enumerate().rev() {
                fmpz_poly_set_coeff_ui(&mut poly, index.cast_signed(), *value);
            }
        }
        poly
    }
}

impl Drop for FmpzPoly {
    fn drop(&mut self) {
        // SAFETY: `new` initialized this polynomial, and it is cleared once.
        unsafe { fmpz_poly_clear(self) };
    }
}

unsafe extern "C" {
    fn fmpz_poly_init(poly: *mut FmpzPoly);
    fn fmpz_poly_clear(poly: *mut FmpzPoly);
    fn fmpz_poly_set_coeff_ui(poly: *mut FmpzPoly, index: isize, value: usize);
    fn fmpz_poly_get_coeff_ui(poly: *const FmpzPoly, index: isize) -> usize;
    fn fmpz_poly_mul(target: *mut FmpzPoly, left: *const FmpzPoly, right: *const FmpzPoly);
    fn flint_mpn_mul(
        target: *mut usize,
        left: *const usize,
        left_len: isize,
        right: *const usize,
        right_len: isize,
    ) -> usize;
    fn flint_mpn_sqr(target: *mut usize, source: *const usize, len: isize) -> usize;
    fn __gmpn_mul(
        target: *mut usize,
        left: *const usize,
        left_len: isize,
        right: *const usize,
        right_len: isize,
    ) -> usize;
    fn __gmpn_sqr(target: *mut usize, source: *const usize, len: isize);
}

/// Operand lengths in limbs: `2^k` and `3*2^(k-1)` from 8 through `2^24`.
const LENGTHS: [usize; 43] = {
    let mut lengths = [8; 43];
    let mut index = 1;
    while index < lengths.len() {
        lengths[index] = if index % 2 == 1 {
            lengths[index - 1] / 2 * 3
        } else {
            lengths[index - 2] * 2
        };
        index += 1;
    }
    lengths
};

/// Polynomial lengths in coefficients, through `2^22`.
const POLY_LENGTHS: [usize; 39] = *LENGTHS.first_chunk().unwrap();

/// Seconds of sampling per length; the longest products take one sample.
const SECONDS: f64 = 3.0;

/// Returns `length` coefficients of at most `bits` bits and their bound.
fn coefficients(length: usize, seed: u64, bits: u32) -> (Vec<usize>, usize) {
    let bound = usize::MAX >> (usize::BITS - bits);
    let values = operand(length, seed)
        .into_iter()
        .map(|limb| limb & bound)
        .collect();
    (values, bound)
}

fn main() {
    // The timings compare only products that agree limb for limb.
    for length in LENGTHS.into_iter().filter(|length| *length <= 1 << 16) {
        let (a, b) = (operand(length, 1), operand(length, 2));
        let (mut expected, mut flint, mut gmp) = (
            vec![0; 2 * length],
            vec![0; 2 * length],
            vec![0; 2 * length],
        );
        let len = isize::try_from(length).expect("benchmark length");
        let mut workspace = MulWorkspace::new();
        assert!(workspace.mul(&mut expected, &a, &b));
        // SAFETY: both operands hold `len >= 1` limbs, and each target holds `2*len`
        // limbs and overlaps neither operand.
        unsafe {
            flint_mpn_mul(flint.as_mut_ptr(), a.as_ptr(), len, b.as_ptr(), len);
            __gmpn_mul(gmp.as_mut_ptr(), a.as_ptr(), len, b.as_ptr(), len);
        }
        assert_eq!(flint, expected, "FLINT product of {length} limbs");
        assert_eq!(gmp, expected, "GMP product of {length} limbs");
        assert!(workspace.sqr(&mut expected, &a));
        // SAFETY: the operand holds `len >= 1` limbs, and each target holds `2*len`
        // limbs and does not overlap it.
        unsafe {
            flint_mpn_sqr(flint.as_mut_ptr(), a.as_ptr(), len);
            __gmpn_sqr(gmp.as_mut_ptr(), a.as_ptr(), len);
        }
        assert_eq!(flint, expected, "FLINT square of {length} limbs");
        assert_eq!(gmp, expected, "GMP square of {length} limbs");
    }
    // These coefficients and lengths leave every result coefficient in one limb.
    for length in POLY_LENGTHS.into_iter().filter(|length| *length <= 1 << 12) {
        let (a, bound) = coefficients(length, 1, 16);
        let (b, _) = coefficients(length, 2, 16);
        let plan = ConvPlan::new(length, length, bound, bound, false).expect("benchmark shape");
        let mut expected = vec![0; plan.output_len()];
        let mut workspace = ConvWorkspace::new();
        workspace.prepare(plan);
        assert!(workspace.convolve(&mut expected, &a, &b));
        let (a, b) = (FmpzPoly::new(&a), FmpzPoly::new(&b));
        let mut product = FmpzPoly::new(&[]);
        // SAFETY: all three polynomials are initialized, and every index is nonnegative.
        let actual: Vec<usize> = unsafe {
            fmpz_poly_mul(&mut product, &a, &b);
            (0..expected.len())
                .map(|index| fmpz_poly_get_coeff_ui(&product, index.cast_signed()))
                .collect()
        };
        assert_eq!(
            actual, expected,
            "FLINT polynomial of {length} coefficients"
        );
    }
    divan::main();
}

#[divan::bench(args = POLY_LENGTHS, consts = [16, 32], max_time = SECONDS)]
fn ntt_poly<const BITS: u32>(bencher: Bencher<'_, '_>, length: usize) {
    let (a, bound) = coefficients(length, 1, BITS);
    let (b, _) = coefficients(length, 2, BITS);
    let plan = ConvPlan::new(length, length, bound, bound, false).expect("benchmark shape");
    let mut dst = vec![0; plan.output_len()];
    let mut workspace = ConvWorkspace::new();
    workspace.prepare(plan);
    bencher.bench_local(|| {
        assert!(workspace.convolve(black_box(&mut dst), black_box(&a), black_box(&b)));
    });
}

#[divan::bench(args = POLY_LENGTHS, consts = [16, 32], max_time = SECONDS)]
fn flint_poly<const BITS: u32>(bencher: Bencher<'_, '_>, length: usize) {
    let (a, _) = coefficients(length, 1, BITS);
    let (b, _) = coefficients(length, 2, BITS);
    let (a, b) = (FmpzPoly::new(&a), FmpzPoly::new(&b));
    let mut dst = FmpzPoly::new(&[]);
    bencher.bench_local(|| {
        // SAFETY: all three polynomials are initialized and pairwise distinct.
        unsafe { fmpz_poly_mul(black_box(&mut dst), black_box(&a), black_box(&b)) };
    });
}

#[divan::bench(args = LENGTHS, max_time = SECONDS)]
fn ntt_mul(bencher: Bencher<'_, '_>, length: usize) {
    let (a, b) = (operand(length, 1), operand(length, 2));
    let mut dst = vec![0; 2 * length];
    let mut workspace = MulWorkspace::new();
    workspace.prepare(MulPlan::new(length, length).expect("benchmark shape"));
    bencher.bench_local(|| {
        assert!(workspace.mul(black_box(&mut dst), black_box(&a), black_box(&b)));
    });
}

#[divan::bench(args = LENGTHS, max_time = SECONDS)]
fn ntt_sqr(bencher: Bencher<'_, '_>, length: usize) {
    let a = operand(length, 1);
    let mut dst = vec![0; 2 * length];
    let mut workspace = MulWorkspace::new();
    workspace.prepare(MulPlan::new(length, length).expect("benchmark shape"));
    bencher.bench_local(|| {
        assert!(workspace.sqr(black_box(&mut dst), black_box(&a)));
    });
}

#[divan::bench(args = LENGTHS, max_time = SECONDS)]
fn flint_mul(bencher: Bencher<'_, '_>, length: usize) {
    let (a, b) = (operand(length, 1), operand(length, 2));
    let mut dst = vec![0; 2 * length];
    let len = isize::try_from(length).expect("benchmark length");
    bencher.bench_local(|| {
        // SAFETY: both operands hold `len >= 1` limbs, the target holds `2*len`
        // limbs, and the target overlaps neither operand.
        black_box(unsafe {
            flint_mpn_mul(
                black_box(dst.as_mut_ptr()),
                black_box(a.as_ptr()),
                len,
                black_box(b.as_ptr()),
                len,
            )
        })
    });
}

#[divan::bench(args = LENGTHS, max_time = SECONDS)]
fn flint_sqr(bencher: Bencher<'_, '_>, length: usize) {
    let a = operand(length, 1);
    let mut dst = vec![0; 2 * length];
    let len = isize::try_from(length).expect("benchmark length");
    bencher.bench_local(|| {
        // SAFETY: the operand holds `len >= 1` limbs, the target holds `2*len`
        // limbs, and the two do not overlap.
        black_box(unsafe { flint_mpn_sqr(black_box(dst.as_mut_ptr()), black_box(a.as_ptr()), len) })
    });
}

#[divan::bench(args = LENGTHS, max_time = SECONDS)]
fn gmp_mul(bencher: Bencher<'_, '_>, length: usize) {
    let (a, b) = (operand(length, 1), operand(length, 2));
    let mut dst = vec![0; 2 * length];
    let len = isize::try_from(length).expect("benchmark length");
    bencher.bench_local(|| {
        // SAFETY: both operands hold `len >= 1` limbs, the target holds `2*len`
        // limbs, and the target overlaps neither operand.
        black_box(unsafe {
            __gmpn_mul(
                black_box(dst.as_mut_ptr()),
                black_box(a.as_ptr()),
                len,
                black_box(b.as_ptr()),
                len,
            )
        })
    });
}

#[divan::bench(args = LENGTHS, max_time = SECONDS)]
fn gmp_sqr(bencher: Bencher<'_, '_>, length: usize) {
    let a = operand(length, 1);
    let mut dst = vec![0; 2 * length];
    let len = isize::try_from(length).expect("benchmark length");
    bencher.bench_local(|| {
        // SAFETY: the operand holds `len >= 1` limbs, the target holds `2*len`
        // limbs, and the two do not overlap.
        unsafe { __gmpn_sqr(black_box(dst.as_mut_ptr()), black_box(a.as_ptr()), len) };
    });
}
