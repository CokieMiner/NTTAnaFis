use alloc::vec::Vec;

use proptest::{collection::vec, prelude::*};

use crate::{CrtPair, PrimeField, Residue};

/// Fields under test: the transform catalog and the largest admitted prime.
fn fields() -> impl Iterator<Item = PrimeField> {
    PrimeField::catalog().chain(PrimeField::below(1 << 50))
}

/// Returns `value mod p` by wide integer division.
fn wide(field: PrimeField, value: u128) -> u64 {
    u64::try_from(value.rem_euclid(u128::from(field.modulus()))).expect("canonical residue")
}

/// Returns `a*b mod p` by wide integer arithmetic.
fn product(field: PrimeField, a: Residue, b: Residue) -> u64 {
    wide(
        field,
        u128::from(a.get())
            .checked_mul(u128::from(b.get()))
            .expect("100-bit product"),
    )
}

/// Returns the canonical residue of an unsigned integer.
fn residue(field: PrimeField, value: u64) -> Residue {
    field
        .residue(wide(field, u128::from(value)))
        .expect("canonical residue")
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 256 }))]

    #[test]
    fn field_arithmetic_matches_wide_integers(
        left in any::<u64>(),
        right in any::<u64>(),
        signed in any::<i64>(),
        exponent in any::<u64>(),
    ) {
        for field in fields() {
            let modulus = field.modulus();
            let maximum = modulus.checked_sub(1).expect("positive modulus");
            prop_assert_eq!(field.reduce_unsigned(left).get(), wide(field, u128::from(left)));
            let reduced = signed.rem_euclid(modulus.cast_signed()).cast_unsigned();
            prop_assert_eq!(field.reduce(signed).get(), reduced);
            for (first, second) in [
                (left, right),
                (0, right),
                (1, right),
                (maximum, right),
                (maximum, maximum),
                (modulus >> 1, modulus >> 1),
            ] {
                let a = residue(field, first);
                let b = residue(field, second);
                let sum = u128::from(a.get()).checked_add(u128::from(b.get())).expect("sum");
                let difference = u128::from(a.get())
                    .checked_add(u128::from(modulus))
                    .and_then(|value| value.checked_sub(u128::from(b.get())))
                    .expect("difference");
                prop_assert_eq!(field.add(a, b).get(), wide(field, sum));
                prop_assert_eq!(field.sub(a, b).get(), wide(field, difference));
                prop_assert_eq!(field.add(field.neg(a), a), Residue::ZERO);
                prop_assert_eq!(field.mul(a, b).get(), product(field, a, b));
                let centered = field.signed(a);
                prop_assert!(centered.unsigned_abs() <= maximum >> 1);
                prop_assert_eq!(field.reduce(centered), a);
                if let Some(inverse) = field.inv(a) {
                    prop_assert_eq!(field.mul(a, inverse), Residue::ONE);
                } else {
                    prop_assert_eq!(a, Residue::ZERO);
                }
                let mut expected = Residue::ONE;
                let mut power = a;
                let mut remaining = exponent;
                while remaining != 0 {
                    if remaining & 1 != 0 {
                        expected = residue(field, product(field, expected, power));
                    }
                    power = residue(field, product(field, power, power));
                    remaining >>= 1;
                }
                prop_assert_eq!(field.pow(a, exponent), expected);
            }
        }
    }

    #[test]
    fn slice_kernels_match_scalar_products(
        targets in vec(any::<u64>(), 0..48),
        sources in vec(any::<u64>(), 0..48),
        multiplier in any::<u64>(),
    ) {
        for field in fields() {
            let factor = residue(field, multiplier);
            let target: Vec<Residue> = targets.iter().map(|&value| residue(field, value)).collect();
            let source: Vec<Residue> = sources.iter().map(|&value| residue(field, value)).collect();
            let mut subtracted = target.clone();
            field.mul_sub(&mut subtracted, &source, factor);
            for (index, &actual) in subtracted.iter().enumerate() {
                let original = *target.get(index).expect("target element");
                let expected = source.get(index).map_or(original, |&term| {
                    field.sub(original, field.mul(term, factor))
                });
                prop_assert_eq!(actual, expected);
            }
            let mut scaled = target.clone();
            field.scale(&mut scaled, factor);
            for (&actual, &original) in scaled.iter().zip(&target) {
                prop_assert_eq!(actual, field.mul(original, factor));
            }
            let mut expected = Residue::ZERO;
            let mut power = Residue::ONE;
            for &coefficient in &target {
                expected = field.add(expected, field.mul(coefficient, power));
                power = field.mul(power, factor);
            }
            prop_assert_eq!(field.evaluate(&target, factor), expected);
        }
    }

    #[test]
    fn fixed_factor_products_stay_canonical_at_residue_boundaries(multiplier in any::<u64>()) {
        for field in fields() {
            let modulus = field.modulus();
            let maximum = modulus.checked_sub(1).expect("positive modulus");
            let half = modulus >> 1;
            let edges = [0, 1, 2, half, half.checked_add(1).expect("midpoint"), maximum];
            let values = edges.map(|value| residue(field, value));
            for factor in edges.into_iter().chain([multiplier]).map(|value| residue(field, value)) {
                let mut scaled = values;
                field.scale(&mut scaled, factor);
                for (&actual, &original) in scaled.iter().zip(&values) {
                    prop_assert_eq!(actual.get(), product(field, original, factor));
                }
            }
        }
    }

    #[test]
    fn crt_pairs_recover_signed_integers(value in any::<i64>()) {
        let catalog: Vec<PrimeField> = fields().collect();
        for (index, &first) in catalog.iter().enumerate() {
            for &second in catalog.iter().skip(index.checked_add(1).expect("next index")) {
                let pair = CrtPair::new(first, second).expect("distinct fields");
                let combined = pair.combine(first.reduce(value), second.reduce(value));
                prop_assert_eq!(combined, i128::from(value));
            }
        }
    }
}

#[test]
fn primality_matches_trial_division_near_both_range_ends() {
    // Odd divisors up to 2^25 decide every candidate below 2^50.
    let mut composite = alloc::vec![false; 1 << 25];
    let mut divisors = Vec::new();
    for candidate in 3..composite.len() {
        if *composite.get(candidate).expect("sieve slot") || candidate % 2 == 0 {
            continue;
        }
        divisors.push(u64::try_from(candidate).expect("sieve index"));
        // A square beyond the native width also lies beyond the sieve.
        let Some(mut multiple) = candidate.checked_mul(candidate) else {
            continue;
        };
        while let Some(slot) = composite.get_mut(multiple) {
            *slot = true;
            multiple = multiple.checked_add(candidate).expect("multiple");
        }
    }
    let windows = [(1_u64 << 49).checked_add(1).expect("start"), (1 << 50) - 61];
    for start in windows {
        for offset in (0..60).step_by(2) {
            let candidate = start.checked_add(offset).expect("candidate");
            let prime = divisors
                .iter()
                .take_while(|&&divisor| divisor.checked_mul(divisor).expect("square") <= candidate)
                .all(|&divisor| candidate % divisor != 0);
            assert_eq!(PrimeField::new(candidate).is_some(), prime, "{candidate}");
        }
    }
}

#[test]
fn admission_and_search_respect_the_prime_range() {
    for field in PrimeField::catalog() {
        assert_eq!(PrimeField::new(field.modulus()), Some(field));
    }
    for rejected in [0, 1, 2, 1 << 49, (1 << 50) + 1, 28_000_001 * 28_000_003] {
        assert_eq!(PrimeField::new(rejected), None, "{rejected}");
    }
    let largest = PrimeField::below(1 << 50).expect("prime below 2^50");
    assert!(((largest.modulus() + 1)..(1 << 50)).all(|odd| PrimeField::new(odd).is_none()));
    assert_eq!(
        PrimeField::below(largest.modulus()).map(|field| field.modulus() < largest.modulus()),
        Some(true)
    );
    assert_eq!(PrimeField::below((1 << 49) + 2), None);
    assert_eq!(PrimeField::below(0), None);
    assert!(CrtPair::new(largest, largest).is_none());
}
