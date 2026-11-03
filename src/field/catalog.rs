//! The primes of the transform, each proved during constant evaluation.
//!
//! `catalog_prime` admits a prime only if it holds every property the crate
//! assumes: `2^49 < p < 2^50`, primality, a generator of the two-power roots, a
//! valuation within the stored roots, and a reciprocal within the error bound of
//! the residue bounds. The catalog also asserts that its primes are distinct. A
//! [`PrimePrefix`] names the primes of one operation.

#![expect(
    unsafe_code,
    reason = "Catalog admission and canonical residue contracts prove all unchecked integer operations"
)]

use core::num::{NonZeroU64, NonZeroUsize};

/// Bits of every catalog prime and of every [`PrimeField`](super::PrimeField) modulus.
pub const MODULUS_BITS: usize = 50;
/// Every admitted prime exceeds this power of two.
pub const MIN_MODULUS: u64 = 1 << (MODULUS_BITS - 1);
/// Every admitted prime is below this power of two.
pub const MAX_MODULUS: u64 = 1 << MODULUS_BITS;

/// Strong-probable-prime bases that decide primality below
/// `3_825_123_056_546_413_051` (Jiang and Deng 2014).
pub const WITNESSES: [u64; 9] = [2, 3, 5, 7, 11, 13, 17, 19, 23];

/// Centered level roots stored per prime; `W[2^m]` has order `2^(m+2)`.
const LEVEL_ROOTS: usize = 43;

/// A catalog prime as an exact float, with its rounded reciprocal.
///
/// A value proves that `p` is a catalog prime and that `pinv = RN(1/p)` satisfies
/// the reciprocal error bound of the residue bounds. Only [`Prime::float_field`]
/// hands one out. Every residue kernel takes this type, so none of them can receive
/// other parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FloatField {
    p: f64,
    pinv: f64,
}

impl FloatField {
    /// Returns the exact modulus.
    #[inline]
    #[must_use]
    pub const fn p(self) -> f64 {
        self.p
    }

    /// Returns the rounded reciprocal of the modulus.
    #[inline]
    #[must_use]
    pub const fn pinv(self) -> f64 {
        self.pinv
    }
}

/// Parameters of a catalog prime and its power-of-two roots.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Prime {
    modulus: NonZeroU64,
    field: FloatField,
    level_roots: [f64; LEVEL_ROOTS],
}

/// Pairwise distinct prime fields, ordered by the reconstruction schedule.
pub const PRIME_CATALOG: &[Prime] = &[
    catalog_prime(1_108_307_720_798_209, 11),
    catalog_prime(1_086_317_488_242_689, 3),
    catalog_prime(910_395_627_798_529, 7),
    catalog_prime(659_706_976_665_601, 11),
    catalog_prime(699_289_395_265_537, 5),
    catalog_prime(1_022_545_813_831_681, 7),
    catalog_prime(1_013_749_720_809_473, 3),
    catalog_prime(868_614_185_943_041, 3),
];

const _: () = {
    let mut rest = PRIME_CATALOG;
    while let [prime, others @ ..] = rest {
        let mut compared = others;
        while let [other, remaining @ ..] = compared {
            assert!(
                prime.modulus.get() != other.modulus.get(),
                "catalog primes are pairwise distinct"
            );
            compared = remaining;
        }
        rest = others;
    }
};

/// Number of available prime moduli.
pub const MAX_PRIMES: usize = PRIME_CATALOG.len();

/// The exponent `k` of the largest root order `2^k` that every catalog prime has.
///
/// Admission bounds every root table by it, so a retained table of any admitted
/// length serves every prefix.
pub const MAX_ROOT_ORDER: u32 = {
    let mut order = u32::MAX;
    let mut rest = PRIME_CATALOG;
    while let [prime, others @ ..] = rest {
        let valuation = prime.modulus.get().wrapping_sub(1).trailing_zeros();
        if valuation < order {
            order = valuation;
        }
        rest = others;
    }
    order
};

/// A nonempty prefix of [`PRIME_CATALOG`], the primes of one operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct PrimePrefix {
    count: NonZeroUsize,
}

impl PrimePrefix {
    /// Returns the prefix of `count` primes, or `None` outside `1..=MAX_PRIMES`.
    #[inline]
    #[must_use]
    pub const fn new(count: usize) -> Option<Self> {
        match NonZeroUsize::new(count) {
            Some(positive) if positive.get() <= MAX_PRIMES => Some(Self { count: positive }),
            _ => None,
        }
    }

    /// Returns the number of primes, which is positive.
    #[inline]
    #[must_use]
    pub const fn count(self) -> usize {
        self.count.get()
    }

    /// Returns the primes.
    #[inline]
    #[must_use]
    pub const fn primes(self) -> &'static [Prime] {
        // SAFETY: construction bounds the count by the catalog length.
        unsafe { PRIME_CATALOG.split_at_unchecked(self.count.get()).0 }
    }

    /// Returns the entry of this prefix in a table with one entry per prefix.
    #[inline]
    #[must_use]
    pub const fn select<T>(self, table: &[T; MAX_PRIMES]) -> &T {
        // SAFETY: construction gives 0 < count <= MAX_PRIMES, the table length.
        unsafe { &*table.as_ptr().add(self.count.get().unchecked_sub(1)) }
    }
}

impl Prime {
    /// Returns the centered root `g^((p-1)/2^(level+2))` of order `2^(level+2)`,
    /// where `g` is the catalog generator.
    ///
    /// # Safety
    /// `level + 2` is at most the 2-adic valuation of `p-1`, which is at least
    /// [`MAX_ROOT_ORDER`].
    #[must_use]
    pub const unsafe fn level_root(&self, level: usize) -> f64 {
        // SAFETY: catalog admission bounds every valuation by LEVEL_ROOTS + 1.
        unsafe { *self.level_roots.as_ptr().add(level) }
    }

    /// Returns the centered square root of minus one, the root of order four.
    ///
    /// Catalog admission gives every prime a valuation of at least two.
    #[must_use]
    pub const fn imaginary(&self) -> f64 {
        let [root, ..] = self.level_roots;
        root
    }

    /// Returns the integer modulus.
    #[must_use]
    pub const fn modulus(&self) -> u64 {
        self.modulus.get()
    }

    /// Returns the field parameters that residue kernels take.
    #[inline]
    #[must_use]
    pub const fn float_field(&self) -> FloatField {
        self.field
    }

    /// Multiplies two canonical residues using an approximate quotient and exact correction.
    ///
    /// # Safety
    /// Both arguments are in `0..p`.
    #[expect(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        reason = "50-bit factors are exact in f64; the estimated quotient is finite, nonnegative, and below 2^51"
    )]
    pub unsafe fn mul_mod(&self, left: u64, right: u64) -> u64 {
        let modulus = self.modulus.get();
        let estimated = (left as f64) * (right as f64) * self.field.pinv;
        // SAFETY: canonical factors are below 2^50, so the finite estimate
        // is below 2^51 and truncates within one of floor(left*right/p).
        // The exact difference lies in (-p, 2p), inside i64. Computing it
        // modulo 2^64 and interpreting its sign therefore recovers that
        // difference exactly. Both sign-mask corrections stay within (-p, 2p).
        unsafe {
            let quotient = estimated.to_int_unchecked::<u64>();
            let residue = left
                .wrapping_mul(right)
                .wrapping_sub(quotient.wrapping_mul(modulus))
                .cast_signed();
            let signed_modulus = modulus.cast_signed();
            let nonnegative = residue.unchecked_add((residue >> 63) & signed_modulus);
            let excess = nonnegative.unchecked_sub(signed_modulus);
            excess
                .unchecked_add((excess >> 63) & signed_modulus)
                .cast_unsigned()
        }
    }
}

#[expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "Admitted moduli lie below 2^50 and are exact in f64"
)]
const fn catalog_prime(modulus: u64, root: u64) -> Prime {
    assert!(
        modulus > MIN_MODULUS && modulus < MAX_MODULUS,
        "catalog primes must lie between 2^49 and 2^50"
    );
    assert!(
        root > 0 && root < modulus,
        "catalog generators must be canonical nonzero residues"
    );
    let two_valuation = modulus.wrapping_sub(1).trailing_zeros();
    assert!(
        two_valuation >= 2 && two_valuation as usize <= LEVEL_ROOTS + 1,
        "catalog valuations must have stored roots"
    );
    // SAFETY: the domain check proves p > 1 and makes the u64 and f64
    // conversions exact. With p < 2^50 and M < 2^53, p*M < 2^103.
    // Reciprocal rounding gives error <= p/2, so the error bound fits
    // 101 bits. Each level below the table length has a shift below 64, and
    // the valuation assertion bounds every stored level.
    unsafe {
        let reciprocal = 1.0 / modulus as f64;
        // p has 50 bits, so RN(1/p) = M/2^102 with a 53-bit mantissa M.
        // For |a*b| < 2*p^2, the quotient error is bounded by
        // (2*p*abs(p*M-2^102) + 2^47*M)/2^102 + 5/8.
        // A bound below 15/16 gives a reduced product below 15*p/16, and the
        // same terms double for the 4*p^2 case, below 11/8.
        let mantissa = ((reciprocal.to_bits() & ((1_u64 << 52) - 1)) | (1_u64 << 52)) as u128;
        let error = (modulus as u128)
            .unchecked_mul(mantissa)
            .abs_diff(1_u128 << 102);
        let bound = (modulus as u128)
            .unchecked_mul(2)
            .unchecked_mul(error)
            .unchecked_add(mantissa << 47);
        assert!(
            bound < (5_u128 << 98),
            "reciprocal error must preserve the lazy residue bounds"
        );
        let prime = Prime {
            modulus: NonZeroU64::new_unchecked(modulus),
            field: FloatField {
                p: modulus as f64,
                pinv: reciprocal,
            },
            level_roots: [0.0; LEVEL_ROOTS],
        };
        assert!(is_prime(&prime), "catalog moduli must be prime");
        // A nonresidue generates the whole two-power subgroup, so the power
        // below has order exactly 2^(level+2).
        assert!(
            prime.pow_mod(root, modulus.unchecked_sub(1) >> 1) == modulus.unchecked_sub(1),
            "catalog generators must be quadratic nonresidues"
        );
        let half = modulus >> 1;
        let mut level_roots = [0.0; LEVEL_ROOTS];
        let mut level = 0_u32;
        while level.unchecked_add(2) <= two_valuation {
            let power = prime.pow_mod(
                root,
                modulus
                    .unchecked_sub(1)
                    .unchecked_shr(level.unchecked_add(2)),
            );
            *level_roots.as_mut_ptr().add(level as usize) = if power > half {
                -((modulus.unchecked_sub(power)) as f64)
            } else {
                power as f64
            };
            level = level.unchecked_add(1);
        }
        Prime {
            level_roots,
            ..prime
        }
    }
}

/// Decides primality of an odd modulus above `2^49` by strong-probable-prime tests
/// to the bases [`WITNESSES`].
const fn is_prime(prime: &Prime) -> bool {
    let minus_one = prime.modulus.get().wrapping_sub(1);
    let shift = minus_one.trailing_zeros();
    let odd = minus_one >> shift;
    let mut bases = WITNESSES.as_slice();
    while let [base, others @ ..] = bases {
        let mut power = prime.pow_mod(*base, odd);
        let mut passed = power == 1 || power == minus_one;
        let mut round = 1;
        while !passed && round < shift {
            power = prime.pow_mod(power, 2);
            passed = power == minus_one;
            round = round.wrapping_add(1);
        }
        if !passed {
            return false;
        }
        bases = others;
    }
    true
}

impl Prime {
    /// Computes `base^exponent mod p` by binary exponentiation.
    #[must_use]
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "Const widening is lossless; each narrowed remainder lies below the 50-bit prime"
    )]
    pub const fn pow_mod(&self, base: u64, mut exponent: u64) -> u64 {
        let modulus = self.modulus.get() as u128;
        let mut value = 1_u128;
        let mut factor = (base as u128).rem_euclid(modulus);
        // SAFETY: factor and value stay below p < 2^50. Their products use
        // fewer than 100 bits and fit u128.
        unsafe {
            while exponent != 0 {
                if exponent & 1 != 0 {
                    value = value.unchecked_mul(factor).rem_euclid(modulus);
                }
                exponent >>= 1;
                if exponent != 0 {
                    factor = factor.unchecked_mul(factor).rem_euclid(modulus);
                }
            }
        }
        value as u64
    }
}
