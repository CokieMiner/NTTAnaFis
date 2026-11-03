//! Prefix stages preserve modular values without reading omitted input spans.

#![expect(
    unsafe_code,
    reason = "Tests admit detected backends over disjoint, complete stage spans and valid root tables"
)]

use core::{num::NonZeroUsize, ops::Range, slice::from_ref};

use alloc::{vec, vec::Vec};

use core_anafis::Sequential;
use proptest::{collection, prelude::*, test_runner::TestRunner};

use crate::{
    PRIME_CATALOG, Prime,
    arch::{Columns, Kernels, Scalar, StageKernel},
    transform::Twiddles,
};

use super::admitted;

#[test]
fn pruned_radix4_matches_zero_padded_stages_across_fields_ranges_and_lane_boundaries() {
    let backends = admitted();
    let check = |words: &[u64]| {
        for prime in PRIME_CATALOG
            .iter()
            .take(if cfg!(miri) { 1 } else { PRIME_CATALOG.len() })
        {
            for quarter in if cfg!(miri) {
                &[8_usize, 16][..]
            } else {
                &[8_usize, 16, 64][..]
            } {
                let len = quarter.checked_mul(4).expect("four small spans");
                let mut roots = vec![0.0; len >> 1];
                let mut inverse = vec![0.0; len >> 1];
                // SAFETY: one power-of-two stride per prime, within the catalog roots.
                unsafe {
                    Twiddles::prepare(
                        &mut roots,
                        &mut inverse,
                        NonZeroUsize::new(len >> 1).expect("positive stride"),
                        from_ref(prime),
                        &Kernels::SCALAR,
                        &Sequential,
                    );
                }
                let constants = [1.0, 1.0, *roots.get(1).expect("fourth root")];
                for inputs in 1..=3 {
                    let radius = prime.modulus().checked_sub(1).expect("positive radius");
                    let mut source = lazy_inputs(words, len, radius);
                    let initialized = quarter.checked_mul(inputs).expect("active spans");
                    source
                        .get_mut(initialized..)
                        .expect("omitted input spans")
                        .fill(f64::NAN);
                    let tail = quarter.checked_sub(8).expect("positive span");
                    for requested in [0..*quarter, 0..8, tail..*quarter] {
                        let mut expected = source.clone();
                        expected
                            .get_mut(initialized..)
                            .expect("omitted spans")
                            .fill(0.0);
                        // SAFETY: four initialized spans and three complete
                        // root spans fit the requested quarter range. Inputs
                        // satisfy the packed bound.
                        unsafe {
                            Scalar::radix4::<false, true>(
                                expected.as_mut_ptr(),
                                constants.as_ptr(),
                                *quarter,
                                Columns::new(requested.start, requested.end),
                                prime.float_field(),
                            );
                        }
                        for backend in &backends {
                            let kernel = *backend
                                .dif4_prefix
                                .get(inputs.checked_sub(1).expect("positive prefix arity"))
                                .expect("one to three input spans");
                            compare_stage(
                                kernel,
                                &source,
                                &constants,
                                *quarter,
                                requested.clone(),
                                prime,
                                &expected,
                            );
                        }
                    }
                }
            }
        }
    };
    check(&[0, 1, u64::MAX]);
    TestRunner::new(ProptestConfig::with_cases(if cfg!(miri) { 1 } else { 16 }))
        .run(&collection::vec(any::<u64>(), 1..=32), |words| {
            check(&words);
            Ok(())
        })
        .expect("pruned stage property");
}

#[test]
fn radix2_matches_modular_recurrences_with_full_and_omitted_upper_halves() {
    let backends = admitted();
    let check = |words: &[u64]| {
        for prime in PRIME_CATALOG
            .iter()
            .take(if cfg!(miri) { 1 } else { PRIME_CATALOG.len() })
        {
            let radius = prime
                .modulus()
                .checked_mul(2)
                .and_then(|v| v.checked_sub(1))
                .expect("51-bit radius");
            for half in &[8_usize, 16, 64] {
                let len = half.checked_mul(2).expect("two small spans");
                let values = lazy_inputs(words, len, radius);
                for zero in [false, true] {
                    let roots = &[];
                    let mut source = values.clone();
                    if zero {
                        source
                            .get_mut(*half..)
                            .expect("omitted upper half")
                            .fill(f64::NAN);
                    }
                    let tail = half.checked_sub(8).expect("positive span");
                    for requested in [0..*half, 0..8, tail..*half] {
                        let mut expected = source.clone();
                        for index in requested.clone() {
                            let first = *source.get(index).expect("lower input");
                            let upper = half.checked_add(index).expect("upper index");
                            let second = if zero {
                                0.0
                            } else {
                                *source.get(upper).expect("upper input")
                            };
                            let (a, b) = (first + second, first - second);
                            *expected.get_mut(index).expect("lower output") = a;
                            *expected.get_mut(upper).expect("upper output") = b;
                        }
                        for backend in &backends {
                            let kernel = if zero {
                                backend.duplicate_half
                            } else {
                                backend.radix2
                            };
                            compare_stage(
                                kernel,
                                &source,
                                roots,
                                *half,
                                requested.clone(),
                                prime,
                                &expected,
                            );
                        }
                    }
                }
            }
        }
    };
    check(&[0, 1, u64::MAX]);
    TestRunner::new(ProptestConfig::with_cases(if cfg!(miri) { 1 } else { 16 }))
        .run(&collection::vec(any::<u64>(), 1..=32), |words| {
            check(&words);
            Ok(())
        })
        .expect("radix-2 stage property");
}

#[expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "Lazy signed test residues have magnitude at most 2*p-1 < 2^51 and are exact binary64 integers"
)]
fn lazy_inputs(words: &[u64], len: usize, radius: u64) -> Vec<f64> {
    let range = radius
        .checked_mul(2)
        .and_then(|v| v.checked_add(1))
        .expect("52-bit range");
    words
        .iter()
        .cycle()
        .take(len)
        .map(|word| {
            let position = if *word == u64::MAX {
                range.checked_sub(1).expect("last residue")
            } else {
                word.rem_euclid(range)
            };
            i64::try_from(position)
                .expect("52-bit position")
                .checked_sub(i64::try_from(radius).expect("51-bit radius"))
                .expect("signed residue") as f64
        })
        .collect()
}

fn compare_stage(
    kernel: StageKernel,
    source: &[f64],
    roots: &[f64],
    stride: usize,
    requested: Range<usize>,
    prime: &Prime,
    expected: &[f64],
) {
    let mut actual = vec![f64::NAN; source.len().checked_add(2).expect("guarded span")];
    actual
        .get_mut(1..=source.len())
        .expect("output span")
        .copy_from_slice(source);
    // SAFETY: the caller selects an admitted backend and matching spans.
    // Read inputs satisfy its residue bounds; omitted spans are only written.
    // Complete root spans cover the ordered subrange, and guards are disjoint.
    unsafe {
        kernel(
            actual.as_mut_ptr().add(1),
            roots.as_ptr(),
            stride,
            Columns::new(requested.start, requested.end),
            prime.float_field(),
        );
    }
    assert!(actual.first().expect("prefix guard").is_nan());
    assert!(actual.last().expect("suffix guard").is_nan());
    for (position, ((&value, &original), &reference)) in actual
        .iter()
        .skip(1)
        .take(source.len())
        .zip(source)
        .zip(expected)
        .enumerate()
    {
        if !requested.contains(&position.rem_euclid(stride)) {
            assert_eq!(value.to_bits(), original.to_bits());
            continue;
        }
        assert!(value.abs() < 3.0 * prime.float_field().p());
        assert_eq!(
            Scalar::reduce(
                value - reference,
                prime.float_field().p(),
                prime.float_field().pinv()
            ),
            0.0,
            "stride={stride}, position={position}, prime={}",
            prime.modulus()
        );
    }
}
