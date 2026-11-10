#![expect(
    unsafe_code,
    reason = "Tests allocate complete disjoint buffers and invoke only detected ISA backends"
)]

use alloc::vec::Vec;

use crate::{
    PRIME_CATALOG,
    arch::{Columns, Scalar},
};

use super::admitted;

#[test]
#[expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "Catalog boundary residues are exact integers below 2^50 in f64 and u64"
)]
fn admitted_backends_match_exact_integer_products_and_scalar_spans() {
    let backends = admitted();
    for prime in PRIME_CATALOG {
        let modulus = prime.modulus();
        let inputs = [
            0,
            1,
            2,
            modulus.div_euclid(2),
            modulus.checked_sub(1).expect("positive prime"),
        ];
        // Kernels consume complete vectors, so the pairs repeat to a multiple of eight.
        let left: Vec<f64> = inputs
            .iter()
            .flat_map(|&a| inputs.iter().map(move |_| a as f64))
            .cycle()
            .take(32)
            .collect();
        let right: Vec<f64> = inputs
            .iter()
            .flat_map(|_| inputs.iter().map(|&b| b as f64))
            .cycle()
            .take(32)
            .collect();
        for backend in &backends {
            let mut values = left.clone();
            // SAFETY: arrays have equal initialized lengths and are disjoint.
            // Each backend was admitted by runtime detection or is scalar.
            unsafe {
                (backend.multiply)(
                    values.as_mut_ptr(),
                    right.as_ptr(),
                    Columns::new(0, values.len()),
                    prime.float_field(),
                );
            }
            for ((&actual, &a), &b) in values.iter().zip(&left).zip(&right) {
                let expected = u128::from(a as u64)
                    .checked_mul(u128::from(b as u64))
                    .expect("100-bit product")
                    % u128::from(modulus);
                let centered =
                    Scalar::reduce(actual, prime.float_field().p(), prime.float_field().pinv());
                let canonical = if centered < 0.0 {
                    centered + prime.float_field().p()
                } else {
                    centered
                };
                assert_eq!(
                    canonical,
                    u64::try_from(expected).expect("canonical residue") as f64
                );
            }
        }
    }
}
