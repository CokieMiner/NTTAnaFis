//! Weighted CRT sums with one approximate quotient.
//!
//! For `P = product(p_i)`, `M_i = P/p_i`, and `r_i = X*M_i^-1 mod p_i`,
//! `S = sum(r_i*M_i)` satisfies `S = X mod P` and `0 <= S < np*P`. Every catalog
//! prefix has a coefficient capacity `c = floor(log2(2*P/3))`. For `X < 2^c`,
//! `S/P = q + X/P` with `X/P < 2/3`, so a biased estimate of `S/P` with absolute
//! error below `1/4` determines the integer `q`. `S` fits the `words = ceil(c/64)`
//! coefficient words, its top word gives that estimate, and `S - q*P` subtracts a
//! precomputed multiple of `P`.

#![expect(
    unsafe_code,
    reason = "Catalog prefixes and admitted residues bound fixed word arrays, products, and quotient corrections"
)]

use super::{MAX_PRIMES, MODULUS_BITS, PRIME_CATALOG, PrimePrefix};

/// Words of the prime-product capacity, with room for the weighted sum.
pub const WORDS: usize = (MAX_PRIMES * MODULUS_BITS).div_ceil(64);

/// Coefficients in one integer CRT packet.
pub const CRT_LANES: usize = 8;

/// Precomputed cofactor products and reduction parameters for one prime prefix.
#[derive(Clone, Copy, Debug)]
pub struct Cofactors {
    /// Inverses of `P/p_i` in each selected field.
    pub inverses: [u64; MAX_PRIMES],
    /// Rows of `P/p_i` in radix 2^64, padded with zero high words.
    pub factors: [[u64; WORDS]; MAX_PRIMES],
    /// Low words of `q*P` for every quotient `q` below the prime count.
    pub multiples: [[u64; WORDS]; MAX_PRIMES],
    /// Coefficient capacity in bits, `floor(log2(2*P/3))`.
    pub capacity: usize,
    /// Reciprocal `R <= 2^(64*words)/P` of the prefix, with error below two.
    quotient_factor: u64,
}

impl Cofactors {
    /// Reduction data for every nonempty prefix.
    const TABLES: [Self; MAX_PRIMES] = cofactor_tables();

    /// Reconstructs one coefficient below the capacity of the prefix of
    /// `COUNT` primes.
    ///
    /// The prefix table, its factor words, and the coefficient words are
    /// constants, so rows and words unroll over constant operands.
    ///
    /// # Safety
    /// Each of the first `COUNT` residue rows covers `index` and contains a
    /// u64 bit pattern in `0..p_i`, multiplied by the corresponding inverse in
    /// the table. The unweighted coefficient is below `2^capacity`. Exactly the
    /// first `ceil(capacity/64)` entries of `value` are written.
    #[expect(
        clippy::as_conversions,
        clippy::cast_possible_truncation,
        reason = "Low-word stores compute modulo 2^(64*words)"
    )]
    pub unsafe fn bounded<const COUNT: usize>(
        residues: &[&[f64]; MAX_PRIMES],
        index: usize,
        value: &mut [u64; WORDS + 1],
    ) {
        let table = const { Self::of_count(COUNT) };
        let words = const { Self::of_count(COUNT).capacity.div_ceil(64) };
        // P/p_i is a product of COUNT-1 primes below 2^50; its higher words
        // are zero and contribute no product.
        let factor_words = const {
            if COUNT == 1 {
                1
            } else {
                (MODULUS_BITS * (COUNT - 1)).div_ceil(64)
            }
        };
        // With t = 64*(words-1), H = floor(S/2^t), and T = S/P, the value
        // H*R/2^64 lies in (T - 2^t/P - 2*H/2^64, T]. Table construction bounds
        // both error terms by 1/8. Since q <= T < q+2/3, adding 1/4 and
        // truncating gives q.
        // SAFETY: residue rows cover index, and words <= WORDS covers the
        // written entries. A factor word times a residue plus two words is
        // below 2^128. Table construction proves S < 2^(64*words), so no carry
        // leaves the top word. The quotient is below COUNT, inside the
        // multiples table, and S-q*P = X fits the same words.
        unsafe {
            let residue = |row: usize| residues.get_unchecked(row).get_unchecked(index).to_bits();
            let mut sum_carry = 0_u128;
            for word in 0..words {
                let mut sum = sum_carry;
                if word < factor_words {
                    for row in 0..COUNT {
                        let factor = *table.factors.get_unchecked(row).get_unchecked(word);
                        sum = sum.unchecked_add(
                            u128::from(factor).unchecked_mul(u128::from(residue(row))),
                        );
                    }
                }
                sum_carry = sum >> 64;
                *value.get_unchecked_mut(word) = sum as u64;
            }
            let top = *value.get_unchecked(words.unchecked_sub(1));
            let estimate = u128::from(top)
                .unchecked_mul(u128::from(table.quotient_factor))
                .unchecked_add(1 << 62);
            let multiple = table.multiples.get_unchecked((estimate >> 64) as usize);
            let mut borrow = false;
            for word in 0..words {
                let slot = value.get_unchecked_mut(word);
                (*slot, borrow) = slot.borrowing_sub(*multiple.get_unchecked(word), borrow);
            }
        }
    }

    /// Returns the reduction data of the prefix of `count` primes.
    ///
    /// # Panics
    /// When `count` lies outside `1..=MAX_PRIMES`. Kernels evaluate this in a
    /// constant, where that fails the build.
    #[must_use]
    pub const fn of_count(count: usize) -> &'static Self {
        Self::of(PrimePrefix::new(count).expect("prefix counts lie in 1..=MAX_PRIMES"))
    }

    /// Returns the reduction data of a prefix.
    #[inline]
    #[must_use]
    pub const fn of(prefix: PrimePrefix) -> &'static Self {
        prefix.select(&Self::TABLES)
    }
}

#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Const products propagate high words; canonical remainders fit u64"
)]
const fn cofactor_tables() -> [Cofactors; MAX_PRIMES] {
    let empty = Cofactors {
        inverses: [0; MAX_PRIMES],
        factors: [[0; WORDS]; MAX_PRIMES],
        multiples: [[0; WORDS]; MAX_PRIMES],
        capacity: 0,
        quotient_factor: 0,
    };
    let mut tables = [empty; MAX_PRIMES];
    let mut product = [0_u64; WORDS];
    let mut words = 1_usize;
    let mut count = 0_usize;
    // SAFETY: the catalog is nonempty and WORDS validates MAX_PRIMES*50.
    // Each loop index count < MAX_PRIMES gives a nonzero prefix count+1.
    // Product storage contains ceil(50*MAX_PRIMES/64) words. Each single-word
    // product uses at most 114 bits. Exact long division has a remainder
    // below its 50-bit divisor, so each 114-bit dividend fits u128.
    unsafe {
        *product.as_mut_ptr() = 1;
        while count < MAX_PRIMES {
            let prime = &*PRIME_CATALOG.as_ptr().add(count);
            let mut carry = 0_u128;
            let mut index = 0_usize;
            while index < words {
                let slot = product.as_mut_ptr().add(index);
                let full = (*slot as u128)
                    .unchecked_mul(prime.modulus() as u128)
                    .unchecked_add(carry);
                *slot = full as u64;
                carry = full >> 64;
                index = index.unchecked_add(1);
            }
            if carry != 0 {
                *product.as_mut_ptr().add(words) = carry as u64;
                words = words.unchecked_add(1);
            }
            let table = &mut *tables.as_mut_ptr().add(count);
            let mut row = 0_usize;
            while row <= count {
                let field = &*PRIME_CATALOG.as_ptr().add(row);
                let factor = &mut *table.factors.as_mut_ptr().add(row);
                let mut remainder = 0_u128;
                index = words;
                while index > 0 {
                    index = index.unchecked_sub(1);
                    let dividend = (remainder << 64) | (*product.as_ptr().add(index) as u128);
                    *factor.as_mut_ptr().add(index) =
                        dividend.div_euclid(field.modulus() as u128) as u64;
                    remainder = dividend.rem_euclid(field.modulus() as u128);
                }
                assert!(
                    remainder == 0,
                    "each cofactor divides the prime product exactly"
                );
                index = words;
                while index > 0 {
                    index = index.unchecked_sub(1);
                    remainder = ((remainder << 64) | (*factor.as_ptr().add(index) as u128))
                        .rem_euclid(field.modulus() as u128);
                }
                // Fermat's identity; catalog admission proves p > 2.
                *table.inverses.as_mut_ptr().add(row) =
                    field.pow_mod(remainder as u64, field.modulus().unchecked_sub(2));
                row = row.unchecked_add(1);
            }
            // S < (count+1)*P, so the quotient is below the prime count. Each
            // q*P adds P to the preceding multiple; only the low words matter.
            let mut multiple = [0_u64; WORDS];
            let mut quotient = 0_usize;
            let mut overflow = false;
            while quotient <= count {
                *table.multiples.as_mut_ptr().add(quotient) = multiple;
                let mut word = 0_usize;
                overflow = false;
                while word < WORDS {
                    let (sum, first) =
                        (*multiple.as_ptr().add(word)).overflowing_add(*product.as_ptr().add(word));
                    let (total, second) = sum.overflowing_add(overflow as u64);
                    *multiple.as_mut_ptr().add(word) = total;
                    overflow = first | second;
                    word = word.unchecked_add(1);
                }
                quotient = quotient.unchecked_add(1);
            }
            table.capacity = capacity(&product, words);
            // `multiple` now holds (count+1)*P, the bound of the weighted sum S.
            assert!(!overflow, "the weighted sum fits the coefficient words");
            table.quotient_factor = quotient_factor(&product, &multiple, table.capacity);
            count = count.unchecked_add(1);
        }
    }
    tables
}

/// Returns `floor(log2(2*P/3))` for a product `P` of `words` significant words.
///
/// With `bits` the bit length of `P`, `3*2^(bits-2) <= P` holds exactly when the
/// second highest bit of `P` is set. Then `2^(bits-1)` is the largest power of two
/// at most `2*P/3`; otherwise it is `2^(bits-2)`.
#[expect(
    clippy::as_conversions,
    reason = "A leading-zero count below 64 widens exactly"
)]
const fn capacity(product: &[u64; WORDS], words: usize) -> usize {
    // SAFETY: every catalog product has at least 50 bits in `words` positive
    // words, so `bits - 2` indexes an existing bit and each subtraction is ordered.
    unsafe {
        let top = *product.as_ptr().add(words.unchecked_sub(1));
        let bits = words
            .unchecked_sub(1)
            .unchecked_mul(64)
            .unchecked_add(64_u32.unchecked_sub(top.leading_zeros()) as usize);
        let second = bits.unchecked_sub(2);
        if (*product.as_ptr().add(second >> 6) >> (second % 64)) & 1 != 0 {
            bits.unchecked_sub(1)
        } else {
            second
        }
    }
}

/// Returns the reciprocal that estimates `S/P` from the top coefficient word.
///
/// `bound` holds an upper bound of the weighted sum `S`. The assertions prove
/// that `S` fits the coefficient words of `capacity` and that the two errors of
/// [`Cofactors::bounded`] are each below `1/8`. The value `floor((2^128-1)/(H+1))`,
/// for the top two words `H` of `P`, is at most `2^(64*words)/P` and misses it by
/// less than two.
#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "The assertion bounds the reciprocal below 2^61; product words widen exactly"
)]
const fn quotient_factor(product: &[u64; WORDS], bound: &[u64; WORDS], capacity: usize) -> u64 {
    let words = capacity.div_ceil(64);
    assert!(
        words > 0 && words <= WORDS,
        "the coefficient words fit the fixed word arrays"
    );
    // SAFETY: the assertion gives 0 < words <= WORDS, which bounds every index
    // and subtraction. P is positive, so its leading pair plus one fits u128.
    unsafe {
        let mut word = words;
        while word < WORDS {
            assert!(
                *bound.as_ptr().add(word) == 0,
                "the weighted sum fits the coefficient words"
            );
            word = word.unchecked_add(1);
        }
        let top = words.unchecked_sub(1);
        assert!(
            (*bound.as_ptr().add(top)).leading_zeros() >= 4,
            "2*H/2^64 stays below 1/8 for the top word H of the weighted sum"
        );
        let low = if top == 0 {
            0
        } else {
            *product.as_ptr().add(top.unchecked_sub(1))
        };
        let leading = ((*product.as_ptr().add(top) as u128) << 64) | (low as u128);
        let factor = u128::MAX.div_euclid(leading.unchecked_add(1));
        assert!(factor < 1 << 61, "2^(64*(words-1))/P stays below 1/8");
        factor as u64
    }
}
