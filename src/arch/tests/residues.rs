#![expect(
    unsafe_code,
    reason = "Tests provide guarded initialized prefixes and exact integer-domain normalization factors"
)]

use alloc::{vec, vec::Vec};

use proptest::{collection, prelude::*, test_runner::TestRunner};

use crate::{PRIME_CATALOG, arch::Columns};

use super::admitted;

#[test]
#[expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "Catalog residues and lazy signed representatives have magnitude below 2^51 and are exact in binary64"
)]
fn normalization_produces_canonical_integer_bits_at_boundaries_and_complete_vectors() {
    let backends = admitted();
    let check = |words: &[u64], sample_factor: u64| {
        for prime in PRIME_CATALOG {
            let modulus = i64::try_from(prime.modulus()).expect("50-bit modulus");
            let limit = modulus.checked_mul(2).expect("51-bit lazy bound");
            let range = u64::try_from(
                limit
                    .checked_mul(2)
                    .expect("52-bit range")
                    .checked_sub(1)
                    .expect("positive range"),
            )
            .expect("positive range");
            let edges = [
                0,
                1,
                -1,
                modulus,
                -modulus,
                modulus.checked_sub(1).expect("positive modulus"),
                limit.checked_sub(1).expect("positive limit"),
                1_i64.checked_sub(limit).expect("signed limit"),
            ];
            let source: Vec<_> = words
                .iter()
                .enumerate()
                .map(|(index, word)| {
                    edges.get(index).copied().unwrap_or_else(|| {
                        i64::try_from(word % range)
                            .expect("52-bit value")
                            .checked_sub(limit.checked_sub(1).expect("positive limit"))
                            .expect("signed residue")
                    })
                })
                .collect();
            for factor in [
                0,
                1,
                prime.modulus().checked_sub(1).expect("positive modulus"),
                sample_factor % prime.modulus(),
            ] {
                for backend in &backends {
                    let mut actual =
                        vec![f64::NAN; source.len().checked_add(2).expect("guarded prefix")];
                    for (slot, &value) in actual.iter_mut().skip(1).zip(&source) {
                        *slot = value as f64;
                    }
                    // SAFETY: the prefix is initialized and excludes both
                    // guards. abs(value) < 2*p and 0 <= factor < p; the backend
                    // has been admitted by runtime detection or is scalar.
                    unsafe {
                        (backend.residues)(
                            actual.as_mut_ptr().add(1),
                            Columns::new(0, source.len()),
                            factor as f64,
                            prime.float_field(),
                        );
                    }
                    assert!(actual.first().expect("guard").is_nan());
                    assert!(actual.last().expect("guard").is_nan());
                    for (stored, &input) in actual.iter().skip(1).zip(&source) {
                        let expected = i128::from(input)
                            .checked_mul(i128::from(factor))
                            .expect("101-bit product")
                            .rem_euclid(i128::from(modulus));
                        assert_eq!(
                            stored.to_bits(),
                            u64::try_from(expected).expect("canonical residue")
                        );
                    }
                }
            }
        }
    };
    for len in [8, 16, 24, 32] {
        check(&vec![u64::MAX; len], u64::MAX);
    }
    TestRunner::new(ProptestConfig::with_cases(if cfg!(miri) { 2 } else { 32 }))
        .run(
            &(collection::vec(any::<u64>(), 8..=88), any::<u64>()),
            |(words, factor)| {
                let complete = words.len() & !7;
                check(words.get(..complete).expect("complete vectors"), factor);
                Ok(())
            },
        )
        .expect("normalization and conversion property");
}
