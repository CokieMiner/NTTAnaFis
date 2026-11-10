#![expect(
    unsafe_code,
    reason = "Tests reduce arbitrary inputs into each catalog field before invoking canonical-product kernels"
)]

use proptest::prelude::*;

use crate::PRIME_CATALOG;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 128 }))]

    #[test]
    fn reciprocal_products_match_wide_integer_arithmetic(left in any::<u64>(), right in any::<u64>()) {
        for prime in PRIME_CATALOG {
            let modulus = prime.modulus();
            let maximum = modulus.checked_sub(1).expect("positive catalog modulus");
            let a = left.rem_euclid(modulus);
            let b = right.rem_euclid(modulus);
            for (first, second) in [(a, b), (0, b), (1, b), (maximum, b), (maximum, maximum), (modulus >> 1, modulus >> 1)] {
                let expected = u128::from(first).checked_mul(u128::from(second)).expect("100-bit product").rem_euclid(u128::from(modulus));
                // SAFETY: each operand lies in 0..modulus, including the
                // explicit zero, unit, midpoint, and maximum representatives.
                let actual = unsafe { prime.mul_mod(first, second) };
                prop_assert_eq!(actual, u64::try_from(expected).expect("canonical residue"));
            }
        }
    }
}
