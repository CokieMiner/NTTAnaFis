#![expect(
    unsafe_code,
    reason = "Tests construct canonical weighted residues for coefficients below each catalog capacity"
)]

use alloc::vec;

use proptest::prelude::*;

use crate::{MAX_PRIMES, PRIME_CATALOG, WORDS};

use super::super::Cofactors;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 128 }))]

    #[test]
    fn cofactor_sums_recover_coefficients_at_capacity_boundaries(
        random in prop::array::uniform::<_, WORDS>(any::<u64>()),
    ) {
        for count in 1..=MAX_PRIMES {
            let table = Cofactors::of_count(count);
            let words = table.capacity.div_ceil(64);
            let mut bounded = random;
            let mut bounded_maximum = [u64::MAX; WORDS];
            for (index, word) in bounded.iter_mut().enumerate() {
                let keep = table.capacity.saturating_sub(index.checked_mul(64).expect("word bit position")).min(64);
                let mask = if keep == 0 { 0 } else { u64::MAX >> 64_usize.checked_sub(keep).expect("mask shift") };
                *word &= mask;
                *bounded_maximum.get_mut(index).expect("coefficient word") = mask;
            }
            let mut single_word = [0_u64; WORDS];
            *single_word.first_mut().expect("nonempty coefficient") = random.first().expect("nonempty seed") & ((1_u64 << 49).checked_sub(1).expect("49-bit mask"));
            for expected in [[0; WORDS], bounded, bounded_maximum, single_word] {
                let mut storage = vec![vec![0.0; 1]; count];
                for (index, field) in PRIME_CATALOG.iter().take(count).enumerate() {
                    let modulus = u128::from(field.modulus());
                    let mut remainder = 0_u128;
                    for word in expected.iter().rev() {
                        remainder = ((remainder << 64) | u128::from(*word)).rem_euclid(modulus);
                    }
                    let inverse = *table.inverses.get(index).expect("active inverse");
                    let weighted = remainder.checked_mul(u128::from(inverse)).expect("100-bit weighted residue").rem_euclid(modulus);
                    *storage.get_mut(index).expect("prime row").first_mut().expect("one coefficient") = f64::from_bits(u64::try_from(weighted).expect("canonical weighted residue"));
                }
                let mut residues = [&[][..]; MAX_PRIMES];
                for (row, values) in residues.iter_mut().zip(&storage) { *row = values; }
                let mut actual = [u64::MAX; WORDS + 1];
                // SAFETY: this coefficient is below the capacity of the prefix,
                // the kernel matches the prefix, and each row holds its
                // canonical cofactor-weighted residue at index zero.
                unsafe {
                    match count {
                        1 => Cofactors::bounded::<1>(&residues, 0, &mut actual),
                        2 => Cofactors::bounded::<2>(&residues, 0, &mut actual),
                        3 => Cofactors::bounded::<3>(&residues, 0, &mut actual),
                        4 => Cofactors::bounded::<4>(&residues, 0, &mut actual),
                        5 => Cofactors::bounded::<5>(&residues, 0, &mut actual),
                        6 => Cofactors::bounded::<6>(&residues, 0, &mut actual),
                        7 => Cofactors::bounded::<7>(&residues, 0, &mut actual),
                        _ => Cofactors::bounded::<8>(&residues, 0, &mut actual),
                    }
                }
                prop_assert_eq!(actual.get(..words).expect("bounded coefficient words"), expected.get(..words).expect("expected bounded words"), "prime count {}", count);
                prop_assert!(actual.get(words..).expect("preserved coefficient suffix").iter().all(|&word| word == u64::MAX));
            }
        }
    }
}

#[test]
fn capacities_are_the_largest_powers_of_two_below_two_thirds_of_each_product() {
    let mut product = [0_u64; WORDS + 1];
    *product.first_mut().expect("nonempty accumulator") = 1;
    for (count, prime) in PRIME_CATALOG.iter().enumerate() {
        let mut carry = 0_u128;
        for word in &mut product {
            let full = u128::from(*word)
                .checked_mul(u128::from(prime.modulus()))
                .and_then(|value| value.checked_add(carry))
                .expect("114-bit product");
            *word = u64::try_from(full & u128::from(u64::MAX)).expect("low word");
            carry = full >> 64;
        }
        assert_eq!(carry, 0);
        let capacity = Cofactors::of_count(count.checked_add(1).expect("small catalog")).capacity;
        // 3*2^b <= 2*P compares through 3*2^(b-1) <= P, in radix 2^64 from the top.
        let admits = |bits: usize| {
            let shift = bits.checked_sub(1).expect("positive capacity");
            let mut bound = [0_u64; WORDS + 1];
            let full = u128::from(3_u8) << (shift % 64);
            *bound.get_mut(shift >> 6).expect("low bound word") =
                u64::try_from(full & u128::from(u64::MAX)).expect("low word");
            *bound
                .get_mut((shift >> 6).checked_add(1).expect("high word index"))
                .expect("high bound word") = u64::try_from(full >> 64).expect("high word");
            bound.iter().rev().le(product.iter().rev())
        };
        assert!(admits(capacity), "3*2^c <= 2*P for prefix {count}");
        assert!(
            !admits(capacity.checked_add(1).expect("next power")),
            "2*P < 3*2^(c+1) for prefix {count}"
        );
    }
}
