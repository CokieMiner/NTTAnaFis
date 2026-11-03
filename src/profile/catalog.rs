//! Coefficient bounds for every admitted prefix of the prime catalog.

use core::num::NonZeroUsize;

use super::{Cofactors, MAX_PARTS, MAX_PRIMES, PART_BITS, PrimePrefix};

/// Widest admitted coefficient.
pub const MAX_CHUNK_BITS: usize = PART_BITS * MAX_PARTS;

/// Primes and coefficient width of a convolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Profile {
    /// Active primes.
    pub prefix: PrimePrefix,
    /// Bits between consecutive operand coefficients.
    pub chunk_bits: NonZeroUsize,
}

/// Starting profiles; the planner narrows chunk widths to the actual coefficient bound.
pub const PROFILES: [Profile; MAX_PRIMES] = standard_profiles();

const fn standard_profiles() -> [Profile; MAX_PRIMES] {
    let mut profiles = [Profile {
        prefix: PrimePrefix::new(1).expect("the catalog is nonempty"),
        chunk_bits: NonZeroUsize::MIN,
    }; MAX_PRIMES];
    let mut rest = profiles.as_mut_slice();
    let mut count = 1_usize;
    while let [slot, others @ ..] = rest {
        let prefix = PrimePrefix::new(count).expect("one prefix per table entry");
        let half = Cofactors::of(prefix).capacity >> 1;
        let bits = if half < MAX_CHUNK_BITS {
            half
        } else {
            MAX_CHUNK_BITS
        };
        *slot = Profile {
            prefix,
            chunk_bits: NonZeroUsize::new(bits).expect("every capacity is at least 49 bits"),
        };
        rest = others;
        count = count.wrapping_add(1);
    }
    profiles
}
