#![expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "Expected canonical residues lie below 2^50 and convert exactly to f64"
)]
#![expect(
    unsafe_code,
    reason = "Packing receives rows and operands of the geometry it is given"
)]

use core::num::NonZeroUsize;

use alloc::{vec, vec::Vec};

use core_anafis::{Budgeted, Sequential};
use proptest::{collection, prelude::*};

use crate::{
    Geometry, LIMB_BITS, Limb, MAX_PRIMES, PRIME_CATALOG, Packing, PrimePrefix, profile::Profile,
};

use super::admitted;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 1 } else { 24 }))]

    #[test]
    fn shared_packing_matches_binary_reduction_for_all_prefixes_backends_and_padding(
        input in prop_oneof![
            2 => collection::vec(any::<Limb>(), if cfg!(miri) { 8 } else { 1 }..=if cfg!(miri) { 8 } else { 64 }),
            1 => collection::vec(Just(Limb::MAX), if cfg!(miri) { 8 } else { 1 }..=if cfg!(miri) { 8 } else { 64 }),
        ], sampled_width in 1_usize..=MAX_PRIMES.checked_mul(49).expect("catalog capacity").div_euclid(2),
    ) {
        let input_bits = input.len().checked_mul(LIMB_BITS).expect("test input bits");
        let log_terms = usize::try_from(usize::BITS.checked_sub(input_bits.saturating_sub(1).leading_zeros()).expect("input bit logarithm")).expect("native bit logarithm");
        for count in 1..=MAX_PRIMES {
            let capacity = count.checked_mul(49).and_then(|bits| bits.checked_sub(log_terms)).expect("positive field capacity").div_euclid(2);
            let maximum = count.checked_mul(49).expect("field capacity").div_euclid(2);
            for width in [1, 31, 32, 33, 48, 49, 50, 51, 63, 64, 65, 67, 96, 97, 98, 99, 100, 127, 128, 129, 139, 146, 147, 148, 149, 150, 151, 160, 161, capacity, maximum, sampled_width].into_iter().filter(|&bits| bits <= maximum) {
                if cfg!(miri) && !matches!(width, 1 | 49 | 50 | 65 | 67 | 98 | 99 | 147 | 148) && width != capacity && width != maximum { continue; }
                let digits = input_bits.div_ceil(width);
                for ratio in [1, 4] {
                let other_bits = input_bits.checked_mul(ratio).expect("other input bits");
                let profile = Profile { prefix: PrimePrefix::new(count).expect("catalog prefix"), chunk_bits: NonZeroUsize::new(width).expect("positive width") };
                let positive = |bits| NonZeroUsize::new(bits).expect("positive input bits");
                let Some(geometry) = Geometry::under(positive(input_bits), positive(other_bits), profile) else { continue; };
                let len = geometry.transform_len().get();
                let padded = digits.next_multiple_of(8);
                for kernels in admitted() {
                    let span = count.checked_mul(len).expect("row storage");
                    let mut output = vec![-123.0; span.checked_add(2).expect("guarded storage")];
                    // SAFETY: the output holds one row of the geometry per prime, and the
                    // input is the operand the geometry was planned for.
                    unsafe { Packing::pack(output.get_mut(1..span.checked_add(1).expect("suffix guard")).expect("output rows"), (&input, geometry.left()), &geometry, &kernels, &Sequential) };
                    prop_assert_eq!(output.first(), Some(&-123.0));
                    prop_assert_eq!(output.last(), Some(&-123.0));
                    for (prime, values) in PRIME_CATALOG.iter().take(count).zip(output.get(1..).expect("prefix guard").chunks_exact(len)) {
                        for (index, &actual) in values.iter().take(digits).enumerate() {
                            prop_assert!(actual.abs() < prime.float_field().p(), "packed residues satisfy the pruned-stage input bound");
                            let start = index.checked_mul(width).expect("coefficient bit offset");
                            let end = start.checked_add(width).expect("coefficient end").min(input_bits);
                            let mut expected = 0_u64;
                            for bit in (start..end).rev() {
                                let limb = *input.get(bit.div_euclid(LIMB_BITS)).expect("source bit limb");
                                let value = u64::try_from((limb >> (bit % LIMB_BITS)) & 1).expect("single bit");
                                expected = ((expected << 1) | value).rem_euclid(prime.modulus());
                            }
                            let canonical = if actual < 0.0 { actual + prime.float_field().p() } else { actual };
                            prop_assert_eq!((canonical + 0.0).to_bits(), (expected as f64).to_bits(), "np={}, width={}, index={}", count, width, index);
                        }
                        prop_assert!(values.get(digits..padded).expect("zero padding").iter().all(|&value| value == 0.0));
                        prop_assert!(values.get(padded..).expect("untouched suffix").iter().all(|&value| value.to_bits() == (-123.0_f64).to_bits()));
                    }
                }
                }
            }
        }
    }
}

#[test]
fn packing_range_partitions_preserve_guards_and_unused_storage_with_odd_budgets() {
    for length in [33_usize, 64, 65, 127, 128, 129] {
        if cfg!(miri) && !matches!(length, 33 | 65 | 129) {
            continue;
        }
        let input: Vec<_> = (0..length)
            .map(|index| if index & 1 == 0 { Limb::MAX } else { 0 })
            .collect();
        let digits = length.checked_mul(LIMB_BITS).expect("test digit count");
        let bits = NonZeroUsize::new(digits).expect("positive input bits");
        let geometry = Geometry::under(
            bits,
            bits,
            Profile {
                prefix: PrimePrefix::new(1).expect("catalog prefix"),
                chunk_bits: NonZeroUsize::MIN,
            },
        )
        .expect("admitted shape");
        let len = geometry.transform_len().get();
        let padded = digits.next_multiple_of(8);
        for backend in admitted() {
            let mut output = vec![-123.0; len.checked_add(2).expect("guarded allocation")];
            for workers in [1, 3, 5, 8, 1] {
                output.fill(-123.0);
                let values = output
                    .get_mut(1..len.checked_add(1).expect("suffix guard"))
                    .expect("inner allocation");
                let budget = NonZeroUsize::new(workers).expect("positive budget");
                // SAFETY: the output holds the one row of a one-prime geometry, and the
                // input is the operand the geometry was planned for.
                unsafe {
                    Packing::pack(
                        values,
                        (&input, geometry.left()),
                        &geometry,
                        &backend,
                        &Budgeted::new(&Sequential, budget),
                    );
                }
                for (index, &value) in values.iter().take(digits).enumerate() {
                    assert_eq!(
                        value,
                        f64::from(u8::from(index.div_euclid(LIMB_BITS) & 1 == 0))
                    );
                }
                assert!(
                    values
                        .get(digits..padded)
                        .expect("zero padding")
                        .iter()
                        .all(|&v| v == 0.0)
                );
                assert!(
                    values
                        .get(padded..)
                        .expect("omitted suffix")
                        .iter()
                        .all(|&v| v.to_bits() == (-123.0_f64).to_bits())
                );
                assert_eq!(output.first(), Some(&-123.0));
                assert_eq!(output.last(), Some(&-123.0));
            }
        }
    }
}
