//! Prime fields below `2^50` for modular algorithms outside the transform.
//!
//! [`PrimeField`] holds a prime `2^49 < p < 2^50` and `RN(1/p)`; a
//! [`Residue`] holds an element of `0..p`. For residues `a` and `b`, the
//! estimate `RN(RN(a*b)*RN(1/p))` lies within `3/8` of `a*b/p < 2^50`, so
//! its nearest integer `q` leaves `a*b - q*p` in `(-p, p)`. Wrapping 64-bit
//! arithmetic evaluates that remainder exactly, and one masked addition
//! makes it canonical. Products by a fixed factor `w` use Shoup's
//! precomputed quotient `w' = floor(w*2^64/p)`: for every 64-bit `a`, the
//! remainder `a*w - floor(a*w'/2^64)*p` lies in `0..2*p` (Harvey 2014).
//!
//! [`CrtPair`] combines the residues of two fields into a signed integer by
//! Garner's algorithm (Garner 1959).

use core::num::NonZeroU64;

use crate::arch::Scalar;

use super::{MAX_MODULUS, MIN_MODULUS, PRIME_CATALOG, WITNESSES};

/// A prime field `Z/pZ` with `2^49 < p < 2^50`.
///
/// # Examples
///
/// ```
/// use ntt_anafis::PrimeField;
///
/// let field = PrimeField::catalog().next().expect("catalog prime");
/// let (a, b) = (field.reduce(100), field.reduce(25));
/// assert_eq!(field.add(a, b), field.reduce(125));
/// assert_eq!(field.mul(a, b), field.reduce(2500));
/// ```
#[derive(Clone, Copy, Debug)]
pub struct PrimeField {
    modulus: u64,
    reciprocal: f64,
}

/// An element `0..p` of a [`PrimeField`].
///
/// Field operations expect residues produced by the same field; residues of
/// another field give unspecified values.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Residue(u64);

/// Chinese remaindering over two distinct prime fields.
///
/// # Examples
///
/// ```
/// use ntt_anafis::{CrtPair, PrimeField};
///
/// let mut catalog = PrimeField::catalog();
/// let f1 = catalog.next().expect("catalog prime");
/// let f2 = catalog.next().expect("catalog prime");
/// let pair = CrtPair::new(f1, f2).expect("distinct primes");
///
/// let (r1, r2) = (f1.reduce(-12345), f2.reduce(-12345));
/// assert_eq!(pair.combine(r1, r2), -12345);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct CrtPair {
    first: PrimeField,
    second: PrimeField,
    /// The inverse of the first modulus in the second field.
    inverse: Residue,
}

impl Residue {
    /// The additive identity.
    pub const ZERO: Self = Self(0);
    /// The multiplicative identity.
    pub const ONE: Self = Self(1);

    /// Returns the canonical integer in `0..p`.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl PartialEq for PrimeField {
    fn eq(&self, other: &Self) -> bool {
        self.modulus == other.modulus
    }
}

impl Eq for PrimeField {}

impl PrimeField {
    /// Returns the field of `modulus`, or `None` unless `modulus` is a
    /// prime with `2^49 < modulus < 2^50`.
    ///
    /// Primality is decided by strong-probable-prime tests to the first
    /// nine prime bases, which admit no composite below `3.8*10^18` (Jiang and
    /// Deng 2014).
    #[must_use]
    pub fn new(modulus: u64) -> Option<Self> {
        if modulus <= MIN_MODULUS || modulus >= MAX_MODULUS {
            return None;
        }
        #[expect(
            clippy::as_conversions,
            clippy::cast_precision_loss,
            reason = "Moduli below 2^50 are exact in binary64"
        )]
        let field = Self {
            modulus,
            reciprocal: 1.0 / modulus as f64,
        };
        field.is_prime().then_some(field)
    }

    /// Returns the field of the largest prime below `bound` and `2^50`, or
    /// `None` when no such prime exceeds `2^49`.
    #[must_use]
    pub fn below(bound: u64) -> Option<Self> {
        // The largest odd integer below the bound.
        let mut candidate = bound.min(MAX_MODULUS).checked_sub(2)? | 1;
        while candidate > MIN_MODULUS {
            if let Some(field) = Self::new(candidate) {
                return Some(field);
            }
            candidate = candidate.checked_sub(2)?;
        }
        None
    }

    /// Returns the transform's catalog fields in reconstruction order.
    pub fn catalog() -> impl ExactSizeIterator<Item = Self> + Clone {
        PRIME_CATALOG.iter().map(|prime| Self {
            modulus: prime.modulus(),
            reciprocal: prime.float_field().pinv(),
        })
    }

    /// Returns the prime modulus.
    #[must_use]
    pub const fn modulus(self) -> u64 {
        self.modulus
    }

    /// Returns `value` as a residue, or `None` unless `value < p`.
    #[inline]
    #[must_use]
    pub const fn residue(self, value: u64) -> Option<Residue> {
        if value < self.modulus {
            Some(Residue(value))
        } else {
            None
        }
    }

    /// Reduces a signed integer modulo `p`.
    #[inline]
    #[must_use]
    pub fn reduce(self, value: i64) -> Residue {
        // abs(value/p) < 2^14, so the estimate is within 2^-38 of value/p
        // and the rounded quotient leaves a remainder in (-p, p).
        #[expect(
            clippy::as_conversions,
            clippy::cast_precision_loss,
            reason = "The quotient estimate needs a relative accuracy of 2^-15 only"
        )]
        let quotient = Scalar::nearest_integer(value as f64 * self.reciprocal);
        self.canonical(value.wrapping_sub(quotient.wrapping_mul(self.modulus.cast_signed())))
    }

    /// Reduces an unsigned integer modulo `p`.
    #[inline]
    #[must_use]
    pub fn reduce_u64(self, value: u64) -> Residue {
        // value/p < 2^15, so the estimate is within 2^-37 of value/p and
        // the rounded quotient leaves a remainder in (-p, p), which the
        // wrapping difference represents exactly.
        #[expect(
            clippy::as_conversions,
            clippy::cast_precision_loss,
            reason = "The quotient estimate needs a relative accuracy of 2^-16 only"
        )]
        let quotient = Scalar::nearest_integer(value as f64 * self.reciprocal).cast_unsigned();
        self.canonical(
            value
                .wrapping_sub(quotient.wrapping_mul(self.modulus))
                .cast_signed(),
        )
    }

    /// Returns the representative of `residue` in `-(p-1)/2..=(p-1)/2`.
    #[inline]
    #[must_use]
    pub const fn signed(self, residue: Residue) -> i64 {
        let value = residue.0.cast_signed();
        if residue.0 > self.modulus >> 1 {
            value.wrapping_sub(self.modulus.cast_signed())
        } else {
            value
        }
    }

    /// Returns `a + b`.
    #[inline]
    #[must_use]
    pub fn add(self, a: Residue, b: Residue) -> Residue {
        // a + b < 2*p < 2^51. When the sum is below p, subtracting p wraps
        // above it, so the minimum selects the canonical value.
        let sum = a.0.wrapping_add(b.0);
        Residue(sum.min(sum.wrapping_sub(self.modulus)))
    }

    /// Returns `a - b`.
    #[inline]
    #[must_use]
    pub fn sub(self, a: Residue, b: Residue) -> Residue {
        // A borrow wraps the difference above 2^63; adding p returns it to
        // 0..p, while a nonnegative difference stays the smaller value.
        let difference = a.0.wrapping_sub(b.0);
        Residue(difference.min(difference.wrapping_add(self.modulus)))
    }

    /// Returns `-a`.
    #[inline]
    #[must_use]
    pub fn neg(self, a: Residue) -> Residue {
        self.sub(Residue::ZERO, a)
    }

    /// Returns `a * b`.
    #[inline]
    #[must_use]
    pub fn mul(self, a: Residue, b: Residue) -> Residue {
        #[expect(
            clippy::as_conversions,
            clippy::cast_precision_loss,
            reason = "Residues below 2^50 are exact in binary64"
        )]
        let estimate = a.0.cast_signed() as f64 * b.0.cast_signed() as f64 * self.reciprocal;
        let quotient = Scalar::nearest_integer(estimate).cast_unsigned();
        self.canonical(
            a.0.wrapping_mul(b.0)
                .wrapping_sub(quotient.wrapping_mul(self.modulus))
                .cast_signed(),
        )
    }

    /// Returns `base^exponent`, with `0^0 = 1`.
    #[must_use]
    pub fn pow(self, base: Residue, mut exponent: u64) -> Residue {
        let mut power = base;
        let mut result = Residue::ONE;
        while exponent != 0 {
            if exponent & 1 != 0 {
                result = self.mul(result, power);
            }
            exponent >>= 1;
            if exponent != 0 {
                power = self.mul(power, power);
            }
        }
        result
    }

    /// Returns `a^-1`, or `None` for zero.
    ///
    /// The extended Euclidean algorithm on `(p, a)` keeps each remainder
    /// `r_i` congruent to `t_i*a` with `|t_i| <= p/r_(i-1)`. For prime `p`
    /// the last nonzero remainder is `1`, and its cofactor, of magnitude at
    /// most `p/2`, is the inverse (Knuth 1997, Algorithm 4.5.2X).
    #[must_use]
    pub fn inv(self, a: Residue) -> Option<Residue> {
        let mut divisor = NonZeroU64::new(a.0)?;
        let mut dividend = self.modulus;
        let mut previous = 0_i64;
        let mut current = 1_i64;
        loop {
            // `dividend - quotient*divisor` is the remainder, so each step
            // divides once.
            let quotient = dividend.div_euclid(divisor.get());
            let Some(remainder) =
                NonZeroU64::new(dividend.wrapping_sub(quotient.wrapping_mul(divisor.get())))
            else {
                return Some(self.canonical(current));
            };
            let next = previous.wrapping_sub(quotient.cast_signed().wrapping_mul(current));
            dividend = divisor.get();
            divisor = remainder;
            previous = current;
            current = next;
        }
    }

    /// Replaces each `target[i]` by `target[i] - factor*source[i]`.
    ///
    /// Elements pair up to the shorter length.
    pub fn mul_sub(self, target: &mut [Residue], source: &[Residue], factor: Residue) {
        let shoup = self.shoup(factor);
        for (slot, &term) in target.iter_mut().zip(source) {
            *slot = self.sub(*slot, self.mul_shoup(term, factor, shoup));
        }
    }

    /// Replaces each `target[i]` by `factor*target[i]`.
    pub fn scale(self, target: &mut [Residue], factor: Residue) {
        let shoup = self.shoup(factor);
        for slot in target {
            *slot = self.mul_shoup(*slot, factor, shoup);
        }
    }

    /// Evaluates `sum(coefficients[i]*point^i)` by Horner's rule.
    #[must_use]
    pub fn evaluate(self, coefficients: &[Residue], point: Residue) -> Residue {
        let shoup = self.shoup(point);
        coefficients
            .iter()
            .rev()
            .fold(Residue::ZERO, |value, &coefficient| {
                self.add(self.mul_shoup(value, point, shoup), coefficient)
            })
    }

    /// Decides primality of an odd modulus above `2^49` by the
    /// Miller-Rabin test with the fixed bases (Rabin 1980).
    fn is_prime(self) -> bool {
        let modulus = self.modulus;
        if WITNESSES
            .iter()
            .any(|&witness| modulus.checked_rem(witness) == Some(0))
        {
            return false;
        }
        let shift = modulus.wrapping_sub(1).trailing_zeros();
        let odd = modulus.wrapping_sub(1) >> shift;
        let minus_one = Residue(modulus.wrapping_sub(1));
        WITNESSES.iter().all(|&witness| {
            let mut power = self.pow(Residue(witness), odd);
            if power == Residue::ONE || power == minus_one {
                return true;
            }
            for _ in 1..shift {
                power = self.mul(power, power);
                if power == minus_one {
                    return true;
                }
            }
            false
        })
    }

    /// Returns Shoup's quotient `floor(factor*2^64/p)`.
    ///
    /// `q1`, the nearest integer to `RN(factor*RN(1/p))*2^51`, lies within one
    /// of `factor*2^51/p`, so `r = factor*2^64 - q1*2^13*p` has magnitude below
    /// `2^13*p < 2^63`, and arithmetic modulo `2^64` evaluates it exactly. The
    /// nearest integer `q2` to `r/p` leaves `r - q2*p` within `5*p/8` of zero,
    /// and one masked correction gives the remainder in `0..p`.
    #[expect(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        reason = "Residues below 2^50 are exact in binary64; r needs a relative accuracy of 2^-16 only"
    )]
    fn shoup(self, factor: Residue) -> u64 {
        let modulus = self.modulus.cast_signed();
        let scaled = factor.0.cast_signed() as f64 * self.reciprocal * 2_251_799_813_685_248.0;
        let first = Scalar::nearest_integer(scaled);
        let remainder = first.wrapping_mul(modulus).wrapping_shl(13).wrapping_neg();
        let second = Scalar::nearest_integer(remainder as f64 * self.reciprocal);
        let centered = remainder.wrapping_sub(second.wrapping_mul(modulus));
        first
            .wrapping_shl(13)
            .wrapping_add(second)
            .wrapping_add(centered >> 63)
            .cast_unsigned()
    }

    /// Returns `value*factor` from Shoup's quotient of `factor`.
    #[inline]
    fn mul_shoup(self, value: Residue, factor: Residue, shoup: u64) -> Residue {
        #[expect(
            clippy::as_conversions,
            reason = "The high half of a 128-bit product fits u64"
        )]
        let quotient = (u128::from(value.0).wrapping_mul(u128::from(shoup)) >> 64) as u64;
        // The remainder lies in 0..2*p; one conditional subtraction ends it.
        let product = value
            .0
            .wrapping_mul(factor.0)
            .wrapping_sub(quotient.wrapping_mul(self.modulus));
        Residue(product.min(product.wrapping_sub(self.modulus)))
    }

    /// Maps a remainder in `(-p, p)` to `0..p` by a masked addition.
    #[inline]
    const fn canonical(self, remainder: i64) -> Residue {
        Residue(
            remainder
                .wrapping_add((remainder >> 63) & self.modulus.cast_signed())
                .cast_unsigned(),
        )
    }
}

impl CrtPair {
    /// Returns the pair of `first` and `second`, or `None` when they coincide.
    #[must_use]
    pub fn new(first: PrimeField, second: PrimeField) -> Option<Self> {
        let inverse = second.inv(second.reduce_u64(first.modulus))?;
        Some(Self {
            first,
            second,
            inverse,
        })
    }

    /// Returns the product `M` of both moduli, below `2^100`.
    #[must_use]
    pub fn modulus(self) -> u128 {
        u128::from(self.first.modulus).wrapping_mul(u128::from(self.second.modulus))
    }

    /// Returns the integer in `-(M-1)/2..=(M-1)/2` congruent to `first`
    /// modulo the first prime and to `second` modulo the second.
    ///
    /// Garner's form `x = a + p1*((b - a)*p1^-1 mod p2)` gives `0 <= x < M`
    /// before the symmetric shift.
    #[must_use]
    pub fn combine(self, first: Residue, second: Residue) -> i128 {
        let digit = self.second.mul(
            self.second.sub(second, self.second.reduce_u64(first.0)),
            self.inverse,
        );
        let value = u128::from(first.0)
            .wrapping_add(u128::from(self.first.modulus).wrapping_mul(u128::from(digit.0)));
        let modulus = self.modulus();
        if value > modulus >> 1 {
            value.cast_signed().wrapping_sub(modulus.cast_signed())
        } else {
            value.cast_signed()
        }
    }
}
