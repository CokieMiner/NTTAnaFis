#![expect(
    unsafe_code,
    reason = "Tests supply admitted packet kernels, independent canonical residues, and complete guarded coefficient packets"
)]

use alloc::vec;

use proptest::prelude::*;

use crate::{CRT_LANES, Cofactors, MAX_PRIMES, PRIME_CATALOG, WORDS, arch::Kernels};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(12))]

    #[test]
    fn coefficient_packets_shift_lanes_words_and_guards_across_every_prime_prefix(
        seed in prop_oneof![
            Just([0_u64; WORDS * CRT_LANES]),
            Just([u64::MAX; WORDS * CRT_LANES]),
            prop::array::uniform::<_, { WORDS * CRT_LANES }>(any::<u64>()),
        ], sampled_bits in 1_usize..=MAX_PRIMES.checked_mul(49).expect("catalog capacity"), first in 0_usize..=3,
        sampled_bit in 0_usize..=4096, sampled_width in 1_usize..=196,
    ) {
        let Some(kernels) = Kernels::get().coefficients else { return Ok(()); };
        for count in 1..=MAX_PRIMES {
            let table = Cofactors::of_count(count);
            let kernel = *kernels.get(count.checked_sub(1).expect("prefix index")).expect("prefix kernel");
            let capacity = table.capacity;
            let words = capacity.div_ceil(64);
            for bits in [1, 49, 50, 51, 52, 53, 63, 64, 65, 103, 104, 105, 127, 128, 129, capacity, sampled_bits].into_iter().filter(|&width| width <= capacity) {
                let mut expected = [[0_u64; WORDS]; CRT_LANES];
                for (lane, coefficient) in expected.iter_mut().enumerate() {
                    for (word, value) in coefficient.iter_mut().enumerate() {
                        let keep = bits.saturating_sub(word.checked_mul(64).expect("word bit position")).min(64);
                        let mask = if keep == 0 { 0 } else { u64::MAX >> 64_usize.checked_sub(keep).expect("mask width") };
                        *value = seed.get(lane.checked_mul(WORDS).and_then(|offset| offset.checked_add(word)).expect("seed index")).expect("coefficient seed") & mask;
                    }
                }
                let span = first.checked_add(CRT_LANES).and_then(|length| length.checked_add(1)).expect("guarded residue rows");
                let mut storage = vec![vec![f64::from_bits(1); span]; count];
                for (row, prime) in PRIME_CATALOG.iter().take(count).enumerate() {
                    let modulus = u128::from(prime.modulus());
                    let inverse = u128::from(*table.inverses.get(row).expect("cofactor inverse"));
                    for (lane, coefficient) in expected.iter().enumerate() {
                        let mut remainder = 0_u128;
                        for word in coefficient.iter().rev() {
                            remainder = ((remainder << 64) | u128::from(*word)).rem_euclid(modulus);
                        }
                        let weighted = remainder.checked_mul(inverse).expect("100-bit weighted residue").rem_euclid(modulus);
                        *storage.get_mut(row).expect("prime row").get_mut(first.checked_add(lane).expect("coefficient index")).expect("residue slot") =
                            f64::from_bits(u64::try_from(weighted).expect("canonical weighted residue"));
                    }
                }
                let mut residues = [&[][..]; MAX_PRIMES];
                for (row, values) in residues.iter_mut().zip(&storage) { *row = values; }
                for (bit, width) in [(0, 64), (0, 1), (5, 49), (sampled_bit, sampled_width)] {
                    let mut output = [[u64::MAX; CRT_LANES]; WORDS + 1];
                    // SAFETY: the detected kernel admits this CPU. Every active
                    // row covers first..first+8 with canonical weighted residues.
                    // Independent coefficients are masked below the capacity.
                    unsafe { kernel(&residues, first, bit, width, &mut output); }
                    for (lane, coefficient) in expected.iter().enumerate() {
                        let position = bit.checked_add(lane.checked_mul(width).expect("lane bits")).expect("lane position");
                        let offset = position % 64;
                        let mut previous = 0_u64;
                        for word in 0..=words {
                            let current = coefficient.get(word).copied().filter(|_| word < words).unwrap_or(0);
                            let shifted = (u128::from(current) << offset) | (u128::from(previous) << offset >> 64);
                            let wanted = u64::try_from(shifted & u128::from(u64::MAX)).expect("low word");
                            let actual = *output.get(word).expect("output word").get(lane).expect("output lane");
                            prop_assert_eq!(actual, wanted, "np={}, bits={}, lane={}, word={}", count, bits, lane, word);
                            previous = current;
                        }
                    }
                    for column in output.iter().skip(words.checked_add(1).expect("written rows")) {
                        prop_assert!(column.iter().all(|&word| word == u64::MAX));
                    }
                }
            }
        }
    }
}
