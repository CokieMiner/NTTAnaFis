#![expect(
    unsafe_code,
    reason = "Tests admit detected kernels over guarded three-span allocations with bounded residues"
)]

use core::ops::Range;

use alloc::{vec, vec::Vec};

use proptest::{collection, prelude::*, test_runner::TestRunner};

use crate::{PRIME_CATALOG, Prime, arch::Columns};

use super::admitted;

/// Returns signed integers of magnitude below `limit`, holding both extremes
/// for the words zero and `u64::MAX`.
#[expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "Test residues are below 2^52 in magnitude and convert exactly to binary64"
)]
fn residues(words: &[u64], len: usize, limit: u64) -> Vec<f64> {
    let radius = limit.checked_sub(1).expect("positive limit");
    let range = radius
        .checked_mul(2)
        .and_then(|width| width.checked_add(1))
        .expect("53-bit range");
    words
        .iter()
        .cycle()
        .take(len)
        .map(|&word| {
            let position = match word {
                0 => 0,
                u64::MAX => range.checked_sub(1).expect("last position"),
                _ => word.checked_rem(range).expect("positive range"),
            };
            i64::try_from(position)
                .expect("53-bit position")
                .checked_sub(i64::try_from(radius).expect("52-bit radius"))
                .expect("signed residue") as f64
        })
        .collect()
}

/// Reduces an integer-valued float modulo the prime, independently of the kernels.
#[expect(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    reason = "Test values are exact integers below 2^53 in magnitude"
)]
fn canonical(value: f64, prime: &Prime) -> i128 {
    (value as i128).rem_euclid(i128::from(prime.modulus()))
}

/// Returns the product modulo the prime of two canonical residues.
fn times(left: i128, right: i128, prime: &Prime) -> i128 {
    left.checked_mul(right)
        .expect("100-bit product")
        .rem_euclid(i128::from(prime.modulus()))
}

/// Returns the sum modulo the prime of canonical residues.
fn plus(left: i128, right: i128, prime: &Prime) -> i128 {
    left.checked_add(right)
        .expect("small sum")
        .rem_euclid(i128::from(prime.modulus()))
}

/// Returns the difference modulo the prime of canonical residues.
fn minus(left: i128, right: i128, prime: &Prime) -> i128 {
    left.checked_sub(right)
        .expect("small difference")
        .rem_euclid(i128::from(prime.modulus()))
}

/// Returns the column ranges of a kernel over spans of `span` residues.
fn ranges(span: usize) -> Vec<Range<usize>> {
    let last = span.checked_sub(8).expect("positive span");
    let mut ranges = vec![0..span, 0..8, last..span];
    ranges.sort_unstable_by_key(|range| (range.start, range.end));
    ranges.dedup();
    ranges
}

/// Returns a copy of `source` between two NaN guards.
fn guarded(source: &[f64]) -> Vec<f64> {
    let mut values = vec![f64::NAN];
    values.extend_from_slice(source);
    values.push(f64::NAN);
    values
}

/// Returns the entries of a guarded allocation and checks both guards.
fn inner(values: &[f64]) -> &[f64] {
    assert!(values.first().expect("prefix guard").is_nan());
    assert!(values.last().expect("suffix guard").is_nan());
    let end = values.len().checked_sub(1).expect("suffix guard");
    values.get(1..end).expect("inner entries")
}

/// Fixed and random words run for every prime and span.
fn for_all_cases(check: impl Fn(&Prime, &[u64], usize)) {
    let run = |words: &[u64]| {
        for prime in PRIME_CATALOG
            .iter()
            .take(if cfg!(miri) { 1 } else { PRIME_CATALOG.len() })
        {
            for span in [8_usize, 16, 64] {
                check(prime, words, span);
            }
        }
    };
    run(&[0, 1, u64::MAX]);
    TestRunner::new(ProptestConfig::with_cases(if cfg!(miri) { 1 } else { 16 }))
        .run(&collection::vec(any::<u64>(), 1..=32), |words| {
            run(&words);
            Ok(())
        })
        .expect("tail kernel property");
}

#[test]
fn folds_write_the_residue_to_the_third_span_and_preserve_the_inputs() {
    for_all_cases(|prime, words, span| {
        let double = span.checked_mul(2).expect("small span");
        let triple = span.checked_mul(3).expect("small span");
        let mut source = residues(words, double, prime.modulus());
        source.resize(triple, f64::NAN);
        let root = prime.imaginary();
        let canonical_root = canonical(root, prime);
        for backend in admitted() {
            for range in ranges(span) {
                let mut actual = guarded(&source);
                // SAFETY: three spans fit the guarded allocation, inputs are below
                // p in magnitude, the root is centered, and endpoints are
                // multiples of eight. The table admits the current CPU.
                unsafe {
                    (backend.fold)(
                        actual.as_mut_ptr().add(1),
                        &raw const root,
                        span,
                        Columns::new(range.start, range.end),
                        prime.float_field(),
                    );
                }
                let values = inner(&actual);
                for column in 0..span {
                    let upper = span.checked_add(column).expect("second span");
                    let target = double.checked_add(column).expect("third span");
                    let value = *values.get(target).expect("output");
                    if !range.contains(&column) {
                        assert!(value.is_nan(), "columns outside the range are untouched");
                        continue;
                    }
                    let a = canonical(*source.get(column).expect("first input"), prime);
                    let b = canonical(*source.get(upper).expect("second input"), prime);
                    assert!(value.abs() < 13.0 * prime.float_field().p() / 8.0);
                    assert_eq!(
                        canonical(value, prime),
                        plus(a, times(b, canonical_root, prime), prime)
                    );
                }
                assert_eq!(
                    values
                        .get(..double)
                        .expect("inputs")
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    source
                        .get(..double)
                        .expect("inputs")
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                );
            }
        }
    });
}

#[test]
fn node_radix2_stages_match_modular_arithmetic_at_their_residue_bounds() {
    for_all_cases(|prime, words, span| {
        let double = span.checked_mul(2).expect("small span");
        let root = prime.imaginary();
        let canonical_root = canonical(root, prime);
        for inverse in [false, true] {
            let factor: u64 = if inverse { 2 } else { 3 };
            let limit = prime.modulus().checked_mul(factor).expect("52-bit limit");
            let source = residues(words, double, limit);
            for backend in admitted() {
                let kernel = if inverse { backend.dit2 } else { backend.dif2 };
                for range in ranges(span) {
                    let mut actual = guarded(&source);
                    // SAFETY: two halves fit the guarded allocation, residues
                    // satisfy the stage bound, the root is centered, and endpoints
                    // are multiples of eight. The table admits the current CPU.
                    unsafe {
                        kernel(
                            actual.as_mut_ptr().add(1),
                            &raw const root,
                            span,
                            Columns::new(range.start, range.end),
                            prime.float_field(),
                        );
                    }
                    let values = inner(&actual);
                    for column in 0..span {
                        let upper = span.checked_add(column).expect("second half");
                        let first = *values.get(column).expect("first output");
                        let second = *values.get(upper).expect("second output");
                        let lower_input = *source.get(column).expect("first input");
                        let upper_input = *source.get(upper).expect("second input");
                        if !range.contains(&column) {
                            assert_eq!(first.to_bits(), lower_input.to_bits());
                            assert_eq!(second.to_bits(), upper_input.to_bits());
                            continue;
                        }
                        let a = canonical(lower_input, prime);
                        let b = canonical(upper_input, prime);
                        let (low, high) = if inverse {
                            (
                                plus(a, b, prime),
                                times(minus(a, b, prime), canonical_root, prime),
                            )
                        } else {
                            let rotated = times(b, canonical_root, prime);
                            (plus(a, rotated, prime), minus(a, rotated, prime))
                        };
                        assert_eq!(canonical(first, prime), low);
                        assert_eq!(canonical(second, prime), high);
                        let bound = if inverse { 2.0 } else { 1.5 } * prime.float_field().p();
                        assert!(first.abs() < bound && second.abs() < bound);
                    }
                }
            }
        }
    });
}

#[test]
fn recombination_recovers_canonical_coefficients_from_both_parts() {
    for_all_cases(|prime, words, span| {
        let double = span.checked_mul(2).expect("small span");
        let triple = span.checked_mul(3).expect("small span");
        let limit = prime.modulus().checked_mul(2).expect("51-bit limit");
        let source = residues(words, triple, limit);
        let half = (1.0 - prime.float_field().p()) * 0.5;
        let constants = [prime.imaginary(), half];
        for factor in [0.0, 1.0, -1.0, half, -half] {
            let scale = canonical(factor, prime);
            for backend in admitted() {
                for (active, kernel) in
                    [(true, backend.recombine), (false, backend.recombine_suffix)]
                {
                    for range in ranges(span) {
                        let mut actual = guarded(&source);
                        // SAFETY: three spans fit the guarded allocation, residues
                        // are below 2*p, the constants and factor are centered,
                        // and endpoints are multiples of eight. The table admits
                        // the current CPU.
                        unsafe {
                            kernel(
                                actual.as_mut_ptr().add(1),
                                constants.as_ptr(),
                                span,
                                Columns::new(range.start, range.end),
                                factor,
                                prime.float_field(),
                            );
                        }
                        let values = inner(&actual);
                        for column in 0..span {
                            let middle = span.checked_add(column).expect("second span");
                            let top = double.checked_add(column).expect("third span");
                            let bits = |slot: usize| values.get(slot).expect("output").to_bits();
                            let original = |slot: usize| *source.get(slot).expect("input");
                            if !range.contains(&column) {
                                for slot in [column, middle, top] {
                                    assert_eq!(bits(slot), original(slot).to_bits());
                                }
                                continue;
                            }
                            let u = times(canonical(original(column), prime), scale, prime);
                            let v = times(canonical(original(middle), prime), scale, prime);
                            let t = times(canonical(original(top), prime), scale, prime);
                            let high = if active {
                                let rotated = times(v, canonical(constants[0], prime), prime);
                                minus(
                                    times(plus(u, rotated, prime), canonical(half, prime), prime),
                                    t,
                                    prime,
                                )
                            } else {
                                assert_eq!(bits(top), original(top).to_bits());
                                0
                            };
                            let expect = |value: i128| u64::try_from(value).expect("canonical");
                            assert_eq!(bits(column), expect(minus(u, high, prime)));
                            assert_eq!(bits(middle), expect(v));
                            if active {
                                assert_eq!(bits(top), expect(high));
                            }
                        }
                    }
                }
            }
        }
    });
}
