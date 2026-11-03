//! Exact modular products of integer-valued binary64 residues.
//!
//! For integers `a` and `b` with `abs(a*b) < 4*p^2`, `high = RN(a*b)`, and
//! the nearest integer `q` to `RN(high/p)`, catalog admission bounds the
//! remainder `a*b - q*p` by `3*p/2`. Targets with a hardware fused
//! multiply-add evaluate it as `fma(a, b, -high) + fma(-q, p, high)`; the
//! first term is the exact rounding error of the product (Ogita, Rump, and
//! Oishi 2005; van der Hoeven, Lecerf, and Quintin 2016). Other targets with
//! native 64-bit integers evaluate the same remainder in wrapping integer
//! arithmetic: its magnitude is below `2^63`, so its residue modulo `2^64`
//! determines it. The remaining targets split both products exactly with
//! Dekker's algorithm. All methods return the same integer.
//!
//! Rounding to the nearest integer uses the target's instruction where one
//! exists (`roundsd`, `frintn`, `fidbra`, `frint.d`, `f64.nearest`,
//! `xsrdpic`, `fround.d`, `vrintn.f64`), and otherwise adds and subtracts a
//! signed `2^52`. Rounding conversion to an integer uses `cvtsd2si` on
//! `x86_64` and `fcvtns` on `aarch64`.

#![cfg_attr(
    not(ntt_safe_scalar),
    expect(
        unsafe_code,
        reason = "Compiled ISA gates admit the intrinsics; integer-valued residues below 2^53 admit unchecked i64 conversion"
    )
)]

#[cfg(ntt_neon)]
use core::arch::aarch64::{vcvtnd_s64_f64, vdup_n_f64, vfma_f64, vget_lane_f64, vrndn_f64};
#[cfg(all(ntt_x86_fma, target_arch = "x86"))]
use core::arch::x86::_mm_fmadd_sd;
#[cfg(all(ntt_x86_sse41, target_arch = "x86"))]
use core::arch::x86::{_mm_cvtsd_f64, _mm_round_sd, _mm_set_sd};
#[cfg(all(ntt_x86_fma, target_arch = "x86_64"))]
use core::arch::x86_64::_mm_fmadd_sd;
#[cfg(all(ntt_x86_sse41, target_arch = "x86_64"))]
use core::arch::x86_64::{_mm_cvtsd_f64, _mm_round_sd};
#[cfg(all(ntt_nearest_integer_intrinsic, target_arch = "x86_64"))]
use core::arch::x86_64::{_mm_cvtsd_si64, _mm_set_sd};

#[cfg(not(any(ntt_nearest_intrinsic, ntt_nearest_std)))]
use super::BIAS;

/// Exact scalar products, and the portable residue kernels where they compile.
#[derive(Debug)]
pub struct Scalar;

#[cfg(ntt_nearest_integer_intrinsic)]
impl Scalar {
    /// Converts `x` with `abs(x) < 2^52` to the nearest integer, ties to
    /// even, with the target's rounding instructions. Other inputs return
    /// an unspecified integer.
    #[inline]
    pub fn nearest_integer(x: f64) -> i64 {
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: SSE2 belongs to the x86-64 baseline. `cvtsd2si`
            // rounds with the MXCSR mode, which Rust keeps at round to
            // nearest, ties to even, and defines every out-of-range result.
            unsafe { _mm_cvtsd_si64(_mm_set_sd(x)) }
        }
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: the compiled NEON feature admits `fcvtns`, which
            // saturates out-of-range inputs.
            unsafe { vcvtnd_s64_f64(x) }
        }
        #[cfg(target_arch = "x86")]
        #[expect(
            clippy::as_conversions,
            clippy::cast_possible_truncation,
            reason = "The rounded value is an integer; the saturating conversion is exact below 2^52"
        )]
        {
            nearest(x) as i64
        }
    }
}

#[cfg(not(ntt_nearest_integer_intrinsic))]
impl Scalar {
    /// Converts `x` with `abs(x) < 2^52` to the nearest integer, ties to
    /// even. Other inputs return an unspecified integer.
    #[inline]
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "The rounded value is an integer; the saturating conversion is exact below 2^52"
    )]
    pub const fn nearest_integer(x: f64) -> i64 {
        nearest(x) as i64
    }
}

#[cfg(ntt_fused_scalar)]
impl Scalar {
    /// Reduces an integer-valued residue of magnitude below `8*p`.
    #[inline]
    pub fn reduce(value: f64, modulus: f64, pinv: f64) -> f64 {
        mul_add(-nearest(value * pinv), modulus, value)
    }

    /// Computes an exact modular product for `abs(a*b) < 4*p^2`.
    #[inline]
    pub fn mulmod(a: f64, b: f64, modulus: f64, pinv: f64) -> f64 {
        let high = a * b;
        mul_add(a, b, -high) + mul_add(-nearest(high * pinv), modulus, high)
    }
}

#[cfg(not(ntt_fused_scalar))]
impl Scalar {
    /// Reduces an integer-valued residue of magnitude below `8*p`.
    #[inline]
    #[cfg_attr(
        feature = "std",
        expect(
            clippy::suboptimal_flops,
            reason = "These targets lack a fused instruction; the separately rounded form is exact here"
        )
    )]
    pub fn reduce(value: f64, modulus: f64, pinv: f64) -> f64 {
        // abs(value/p) < 8 rounds to a quotient of magnitude at most eight,
        // so its product with p < 2^50 and the difference are exact.
        value - nearest(value * pinv) * modulus
    }

    /// Computes an exact modular product for `abs(a*b) < 4*p^2`.
    #[cfg(ntt_wide_integers)]
    #[inline]
    pub fn mulmod(a: f64, b: f64, modulus: f64, pinv: f64) -> f64 {
        // abs(a*b/p) < 4*p < 2^52 bounds the rounded quotient.
        let quotient = Self::nearest_integer(a * b * pinv);
        // SAFETY: residues and p are finite integers below 2^53 in
        // magnitude, inside i64. The exact remainder lies below
        // 3*p/2 < 2^51 in magnitude, so wrapping arithmetic, which is
        // exact modulo 2^64, returns it unchanged.
        let remainder = unsafe {
            a.to_int_unchecked::<i64>()
                .wrapping_mul(b.to_int_unchecked::<i64>())
                .wrapping_sub(quotient.wrapping_mul(modulus.to_int_unchecked::<i64>()))
        };
        #[expect(
            clippy::as_conversions,
            clippy::cast_precision_loss,
            reason = "The remainder is an integer below 2^51 in magnitude, exact in binary64"
        )]
        {
            remainder as f64
        }
    }

    /// Computes an exact modular product for `abs(a*b) < 4*p^2`.
    ///
    /// Veltkamp splitting gives `a*b = h + l` and `q*p = g + m` exactly
    /// (Dekker 1971). A nonzero nearest quotient keeps `h/g` in `[1/2, 2]`,
    /// so `h - g` is exact (Sterbenz); a zero quotient gives `g = m = 0`.
    /// `l - m` and the final sum are integers below `2^52`.
    #[cfg(not(ntt_wide_integers))]
    #[inline]
    #[cfg_attr(
        feature = "std",
        expect(
            clippy::suboptimal_flops,
            reason = "These targets lack a fused instruction; Dekker's product is exact with separate roundings"
        )
    )]
    pub fn mulmod(a: f64, b: f64, modulus: f64, pinv: f64) -> f64 {
        let split = |x: f64| {
            let scaled = x * 134_217_729.0;
            let high = scaled - (scaled - x);
            (high, x - high)
        };
        let product = |x: f64, y: f64| {
            let high = x * y;
            let ((x_high, x_low), (y_high, y_low)) = (split(x), split(y));
            let low = ((x_high * y_high - high) + x_high * y_low + x_low * y_high) + x_low * y_low;
            (high, low)
        };
        let (high, low) = product(a, b);
        let (scaled, error) = product(nearest(high * pinv), modulus);
        (high - scaled) + (low - error)
    }
}

/// Rounds `abs(x) < 2^52` to the nearest integer, ties to even, with an
/// intrinsic rounding instruction.
#[cfg(ntt_nearest_intrinsic)]
#[inline]
fn nearest(x: f64) -> f64 {
    #[cfg(ntt_x86_sse41)]
    {
        // SAFETY: the compiled SSE4.1 feature establishes CPU and OS
        // support. Mode 8 rounds to nearest, ties to even, without
        // raising the inexact exception.
        unsafe { _mm_cvtsd_f64(_mm_round_sd::<8>(_mm_set_sd(x), _mm_set_sd(x))) }
    }
    #[cfg(ntt_neon)]
    {
        // SAFETY: the compiled NEON feature admits the one-lane rounding;
        // lane zero is the sole initialized result lane.
        unsafe { vget_lane_f64::<0>(vrndn_f64(vdup_n_f64(x))) }
    }
}

/// Rounds `abs(x) < 2^52` to the nearest integer, ties to even.
#[cfg(not(ntt_nearest_intrinsic))]
#[inline]
const fn nearest(x: f64) -> f64 {
    #[cfg(ntt_nearest_std)]
    {
        x.round_ties_even()
    }
    #[cfg(not(ntt_nearest_std))]
    {
        // Adding a signed 2^52 leaves no fractional bits, rounding ties
        // to even; subtraction removes the exact offset.
        let offset = BIAS.copysign(x);
        (x + offset) - offset
    }
}

/// Returns `RN(a*b+c)` with one rounding.
#[cfg(ntt_fused_scalar)]
#[allow(
    clippy::missing_const_for_fn,
    reason = "The intrinsic branches of x86 and aarch64 are not const"
)]
#[inline]
fn mul_add(a: f64, b: f64, c: f64) -> f64 {
    #[cfg(ntt_x86_fma)]
    {
        // SAFETY: the compiled fma feature establishes CPU and OS support.
        unsafe { _mm_cvtsd_f64(_mm_fmadd_sd(_mm_set_sd(a), _mm_set_sd(b), _mm_set_sd(c))) }
    }
    #[cfg(ntt_neon)]
    {
        // SAFETY: the compiled NEON feature admits a one-lane fused
        // multiply-add; lane zero is the sole initialized result lane.
        unsafe { vget_lane_f64::<0>(vfma_f64(vdup_n_f64(c), vdup_n_f64(a), vdup_n_f64(b))) }
    }
    #[cfg(not(any(ntt_x86_fma, ntt_neon)))]
    {
        a.mul_add(b, c)
    }
}
