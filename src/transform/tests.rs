#![expect(
    unsafe_code,
    reason = "Tests use canonical nonzero transform lengths admitted by each catalog prime"
)]

use core::{num::NonZeroUsize, slice::from_ref};

use alloc::{vec, vec::Vec};

use core_anafis::Sequential;

use crate::{
    PRIME_CATALOG,
    arch::{Columns, Kernels, Scalar},
};

use super::{COLUMN_ALIGN, COLUMN_GRAIN, Normalization, Partition, Roots, Transform, Twiddles};

const FIELD_CASES: usize = if cfg!(miri) { 1 } else { PRIME_CATALOG.len() };

#[test]
fn column_partitions_preserve_coverage_budgets_alignment_and_balanced_leaf_work() {
    for length in [8191_usize, 8192, 12_288, 16_384, 24_576, 65_536, 65_543] {
        for budget in 1_usize..=9 {
            assert!(
                Partition::range::<COLUMN_ALIGN, COLUMN_GRAIN>(
                    length,
                    0,
                    NonZeroUsize::new(budget).expect("positive budget")
                )
                .is_none(),
                "a reversed range has no split"
            );
            let mut pending = vec![(
                0,
                length,
                NonZeroUsize::new(budget).expect("positive budget"),
            )];
            let mut leaves = Vec::new();
            while let Some((first, end, workers)) = pending.pop() {
                if let Some(split) =
                    Partition::range::<COLUMN_ALIGN, COLUMN_GRAIN>(first, end, workers)
                {
                    assert!(first < split.middle && split.middle < end);
                    assert_eq!(split.middle & 7, 0);
                    assert_eq!(
                        split.left.get().checked_add(split.right.get()),
                        Some(workers.get())
                    );
                    let left_columns = split.middle.checked_sub(first).expect("left interval");
                    let right_columns = end.checked_sub(split.middle).expect("right interval");
                    assert!(left_columns >= COLUMN_GRAIN);
                    assert!(right_columns >= COLUMN_GRAIN);
                    pending.extend([
                        (first, split.middle, split.left),
                        (split.middle, end, split.right),
                    ]);
                } else {
                    leaves.push((first, end, workers));
                }
            }
            leaves.sort_unstable();
            assert_eq!(leaves.first().map(|leaf| leaf.0), Some(0));
            assert_eq!(leaves.last().map(|leaf| leaf.1), Some(length));
            assert!(leaves.windows(2).all(
                |pair| pair.first().expect("left leaf").1 == pair.last().expect("right leaf").0
            ));
            assert_eq!(
                leaves.iter().map(|leaf| leaf.2.get()).sum::<usize>(),
                budget
            );
            if length >= budget.checked_mul(COLUMN_GRAIN).expect("test work budget") {
                assert_eq!(leaves.len(), budget, "all admitted workers receive a leaf");
                let widths: Vec<_> = leaves
                    .iter()
                    .map(|leaf| leaf.1.checked_sub(leaf.0).expect("ordered leaf"))
                    .collect();
                let spread = widths
                    .iter()
                    .max()
                    .expect("largest leaf")
                    .checked_sub(*widths.iter().min().expect("smallest leaf"))
                    .expect("ordered widths");
                assert!(
                    spread < 16,
                    "worker leaves differ by at most one aligned unit and the final tail: {widths:?}"
                );
            }
        }
    }
}

#[test]
fn child_partitions_minimize_the_largest_work_per_worker_and_accept_extreme_budgets() {
    for arity in [2_usize, 4, 8, 16] {
        assert!(
            Partition::children(arity, NonZeroUsize::MIN).is_none(),
            "one worker runs the children in place"
        );
        for budget in
            (2_usize..=17).chain([usize::MAX.checked_sub(1).expect("large budget"), usize::MAX])
        {
            let split =
                Partition::children(arity, NonZeroUsize::new(budget).expect("positive budget"))
                    .expect("two workers split the children");
            assert!(split.middle > 0 && split.middle < arity);
            assert_eq!(
                split.left.get().checked_add(split.right.get()),
                Some(budget)
            );
            if budget > 17 {
                continue;
            }
            let cost = |cut: usize| {
                let right = arity.checked_sub(cut).expect("interior cut");
                if cut.checked_mul(split.right.get()).expect("test work")
                    >= right.checked_mul(split.left.get()).expect("test work")
                {
                    (cut, split.left.get())
                } else {
                    (right, split.right.get())
                }
            };
            let selected = cost(split.middle);
            for other in (1..arity).map(cost) {
                assert!(
                    selected.0.checked_mul(other.1).expect("selected ratio")
                        <= other.0.checked_mul(selected.1).expect("candidate ratio"),
                    "arity={arity}, budget={budget}, cut={}",
                    split.middle
                );
            }
        }
    }
}

#[test]
#[expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "Canonical test residues and inverse scales are below 2^50 and exactly representable as f64"
)]
fn forward_and_inverse_preserve_prime_fields_across_cache_boundaries_and_zero_prefixes() {
    let kernels = Kernels::get();
    for prime in PRIME_CATALOG.iter().take(FIELD_CASES) {
        for len in [32_usize, 64, 128, 256, 512, 1024, 2048, 4096] {
            if cfg!(miri) && !matches!(len, 32 | 512) {
                continue;
            }
            let mut forward = vec![0.0; len >> 1];
            let mut inverse = vec![0.0; len >> 1];
            // SAFETY: one power-of-two stride per prime, within the catalog roots.
            unsafe {
                Twiddles::prepare(
                    &mut forward,
                    &mut inverse,
                    NonZeroUsize::new(len >> 1).expect("positive stride"),
                    from_ref(prime),
                    kernels,
                    &Sequential,
                );
            }
            let mut prefixes = vec![
                8,
                len.div_euclid(4),
                len.div_euclid(4).checked_add(8).expect("quarter boundary"),
                len.div_euclid(2),
                len.div_euclid(2).checked_add(8).expect("half boundary"),
                len.checked_sub(len >> 2).expect("three quarters"),
                len.checked_sub(8).expect("positive length"),
                len,
            ];
            prefixes.sort_unstable();
            prefixes.dedup();
            for active in prefixes {
                let mut values = vec![0.0; len];
                for (index, value) in values.iter_mut().take(active).enumerate() {
                    *value = (prime
                        .modulus()
                        .checked_sub(1)
                        .expect("positive prime")
                        .checked_sub(u64::try_from(index).expect("small index"))
                        .expect("test index is below prime")) as f64;
                }
                let original = values.clone();
                values
                    .get_mut(active..)
                    .expect("inactive prefix suffix")
                    .fill(f64::NAN);
                let transform = Transform::new(prime, kernels, &Sequential, false);
                transform.forward(
                    &mut values,
                    Roots {
                        values: &forward,
                        node: 0,
                    },
                    active,
                    NonZeroUsize::MIN,
                );
                if len <= 64 {
                    check_forward_values(&values, &original, prime);
                }
                let factor = prime.pow_mod(
                    u64::try_from(len).expect("small transform length"),
                    prime.modulus().checked_sub(2).expect("odd prime"),
                );
                transform.finish::<false, false>(
                    (&mut values, &[]),
                    &inverse,
                    // SAFETY: a transform holds complete vectors.
                    Normalization::new(unsafe { Columns::new(0, len) }, factor as f64, prime),
                    NonZeroUsize::MIN,
                );
                for (actual, expected) in values.iter().zip(&original) {
                    assert_eq!(
                        actual.to_bits(),
                        // SAFETY: test coefficients are integers below 2^50.
                        unsafe { expected.to_int_unchecked::<u64>() },
                        "len={len}, active={active}, prime={}",
                        prime.modulus()
                    );
                }
            }
        }
    }
}

#[expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "Canonical reference coefficients are below 2^50 and convert exactly to binary64"
)]
fn check_forward_values(values: &[f64], coefficients: &[f64], prime: &crate::Prime) {
    let bits = values.len().trailing_zeros();
    let shift = usize::BITS
        .checked_sub(bits)
        .expect("small transform order");
    // SAFETY: test lengths are admitted powers of two between 32 and 64.
    // Every original coefficient is an exact canonical integer below p.
    // Horner steps retain canonical residues, so each modular product has
    // valid canonical inputs and each sum is below 2*p < 2^51.
    unsafe {
        let level_root = prime.level_root(
            usize::try_from(bits.checked_sub(2).expect("large transform")).expect("small level"),
        );
        let root = (if level_root < 0.0 {
            level_root + prime.float_field().p()
        } else {
            level_root
        })
        .to_int_unchecked::<u64>();
        for (index, &value) in values.iter().enumerate() {
            let position = (index & !15) | ((index & 3) << 2) | ((index >> 2) & 3);
            let exponent = position.reverse_bits() >> shift;
            let point = prime.pow_mod(root, u64::try_from(exponent).expect("small exponent"));
            let mut expected = 0_u64;
            for &coefficient in coefficients.iter().rev() {
                let product = prime.mul_mod(expected, point);
                expected = product
                    .checked_add(coefficient.to_int_unchecked::<u64>())
                    .expect("51-bit sum")
                    .rem_euclid(prime.modulus());
            }
            let centered =
                Scalar::reduce(value, prime.float_field().p(), prime.float_field().pinv());
            let canonical = if centered < 0.0 {
                centered + prime.float_field().p()
            } else {
                centered
            };
            assert_eq!(
                canonical,
                expected as f64,
                "len={}, position={index}",
                values.len()
            );
        }
    }
}

#[test]
#[expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "Lazy signed test residues have magnitude below 2^51 and are exact binary64 integers"
)]
fn transforms_match_separate_stages_and_products_across_boundaries() {
    let kernels = Kernels::get();
    let mut separate = *kernels;
    separate.dif_wide = None;
    separate.dit_wide = None;
    for prime in PRIME_CATALOG.iter().take(FIELD_CASES) {
        let limit = i64::try_from(prime.modulus())
            .expect("50-bit prime")
            .checked_mul(2)
            .expect("51-bit lazy bound");
        let range = u64::try_from(
            limit
                .checked_mul(2)
                .and_then(|v| v.checked_sub(1))
                .expect("52-bit range"),
        )
        .expect("positive range");
        for len in [
            32_usize, 64, 128, 256, 512, 1024, 2048, 8192, 16_384, 32_768, 65_536, 131_072,
            262_144, 2_097_152,
        ] {
            if (cfg!(miri) && len != 512) || (len > 16_384 && kernels.dit_wide.is_none()) {
                continue;
            }
            let mut forward = vec![0.0; len >> 1];
            let mut inverse = vec![0.0; len >> 1];
            // SAFETY: one power-of-two stride per prime, within the catalog roots.
            unsafe {
                Twiddles::prepare(
                    &mut forward,
                    &mut inverse,
                    NonZeroUsize::new(len >> 1).expect("positive stride"),
                    from_ref(prime),
                    kernels,
                    &Sequential,
                );
            }
            let source: Vec<_> = (0..len)
                .map(|index| {
                    let word = u64::try_from(index)
                        .expect("small index")
                        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                        .rotate_left(17);
                    i64::try_from(word % range)
                        .expect("52-bit value")
                        .checked_sub(limit.checked_sub(1).expect("positive limit"))
                        .expect("lazy signed residue") as f64
                })
                .collect();
            let factors: Vec<_> = source.iter().rev().map(|value| -value).collect();
            if len > 16_384 {
                check_forward_fusion(&source, &forward, prime, kernels, &separate);
            }
            let roots = inverse.as_slice();
            // SAFETY: a transform holds complete vectors.
            let unit = Normalization::new(unsafe { Columns::new(0, len) }, 1.0, prime);
            let early = Transform::new(prime, kernels, &Sequential, true);
            for squared in [false, true] {
                let mut expected = source.clone();
                for (value, &factor) in expected.iter_mut().zip(&factors) {
                    let multiplier = if squared { *value } else { factor };
                    *value = Scalar::mulmod(
                        *value,
                        multiplier,
                        prime.float_field().p(),
                        prime.float_field().pinv(),
                    );
                }
                Transform::new(prime, &separate, &Sequential, false).finish::<false, false>(
                    (&mut expected, &[]),
                    roots,
                    unit,
                    NonZeroUsize::new(2).expect("positive budget"),
                );
                let mut actual = source.clone();
                let budget = NonZeroUsize::new(2).expect("positive budget");
                if squared {
                    early.finish::<true, true>((&mut actual, &[]), roots, unit, budget);
                } else {
                    early.finish::<true, false>((&mut actual, &factors), roots, unit, budget);
                }
                for (value, reference) in actual.into_iter().zip(expected) {
                    assert_eq!(
                        value.to_bits(),
                        reference.to_bits(),
                        "len={len}, square={squared}, prime={}",
                        prime.modulus(),
                    );
                }
            }
        }
    }
}

/// Forms one product in a row of `3*M` points and in the doubled power-of-two
/// transform, which holds the same coefficients.
#[expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "Canonical inverse scales and test residues are below 2^50 and exact in binary64"
)]
fn check_row(
    (prime, kernels, early_pairs): (&crate::Prime, &Kernels, bool),
    (forward, inverse): (&[f64], &[f64]),
    span: usize,
    (len_a, len_b): (usize, usize),
    (square, fused, budget): (bool, bool, NonZeroUsize),
) {
    let cyclic = span.checked_mul(2).expect("small row");
    let row_len = span.checked_mul(3).expect("small row");
    let order = span.checked_mul(4).expect("small transform");
    let conv_len = len_a
        .checked_add(if square { len_a } else { len_b })
        .and_then(|sum| sum.checked_sub(1))
        .expect("positive shape");
    let outputs = conv_len.next_multiple_of(8);
    let packed = |len: usize, seed: u64, total: usize| {
        let mut values = vec![f64::NAN; total];
        for (index, value) in values.iter_mut().take(len).enumerate() {
            let word = u64::try_from(index)
                .expect("small index")
                .wrapping_add(seed)
                .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                .rotate_left(29);
            let residue = word.checked_rem(prime.modulus()).expect("positive modulus") as f64;
            *value = if word & 1 != 0 {
                residue - prime.float_field().p()
            } else {
                residue
            };
        }
        values
    };
    let scale = |length: usize| {
        let inverse_length = prime.pow_mod(
            u64::try_from(length).expect("small transform length"),
            prime.modulus().checked_sub(2).expect("odd prime"),
        );
        // SAFETY: the outputs round a positive convolution length up to complete vectors.
        Normalization::new(
            unsafe { Columns::new(0, outputs) },
            inverse_length as f64,
            prime,
        )
    };
    let plain = Transform::new(prime, kernels, &Sequential, false);
    let rows = Transform::new(prime, kernels, &Sequential, early_pairs);
    let product = |left: &mut [f64], right: &[f64]| {
        for (value, &factor) in left.iter_mut().zip(right) {
            let multiplier = if square { *value } else { factor };
            *value = Scalar::mulmod(
                *value,
                multiplier,
                prime.float_field().p(),
                prime.float_field().pinv(),
            );
        }
    };
    let mut expected = packed(len_a, 1, order);
    let mut right = packed(len_b, 2, order);
    let roots = |values| Roots { values, node: 0 };
    plain.forward(&mut expected, roots(forward), len_a, NonZeroUsize::MIN);
    plain.forward(&mut right, roots(forward), len_b, NonZeroUsize::MIN);
    product(&mut expected, &right);
    plain.finish::<false, false>(
        (&mut expected, &[]),
        inverse,
        scale(order),
        NonZeroUsize::MIN,
    );
    let mut actual = packed(len_a, 1, row_len);
    let mut other = packed(len_b, 2, row_len);
    // SAFETY: both rows have the shape of the cyclic length, the tables cover it, and
    // the packed prefixes fit the cyclic part.
    unsafe {
        rows.forward_row(&mut actual, forward, (cyclic, len_a), budget);
        if !square {
            rows.forward_row(&mut other, forward, (cyclic, len_b), budget);
        }
    }
    let row_roots = (inverse, cyclic);
    // SAFETY: the rows hold spectra of that shape below 3*p/2, and the outputs fit them.
    unsafe {
        if fused && square {
            rows.finish_row::<true, true>((&mut actual, &[]), row_roots, scale(cyclic), budget);
        } else if fused {
            rows.finish_row::<true, false>((&mut actual, &other), row_roots, scale(cyclic), budget);
        } else {
            product(&mut actual, &other);
            rows.finish_row::<false, false>((&mut actual, &[]), row_roots, scale(cyclic), budget);
        }
    }
    for (index, (value, reference)) in actual.iter().zip(&expected).take(conv_len).enumerate() {
        assert_eq!(
            value.to_bits(),
            reference.to_bits(),
            "span={span}, shape={len_a}x{len_b}, index={index}, square={square}, fused={fused}, prime={}",
            prime.modulus()
        );
    }
}

#[test]
fn rows_of_three_spans_match_the_doubled_transform_across_boundaries() {
    let kernels = Kernels::get();
    let mut separate = *kernels;
    separate.dif_wide = None;
    separate.dit_wide = None;
    for prime in PRIME_CATALOG.iter().take(FIELD_CASES) {
        for span in [
            32_usize, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 65_536, 131_072,
        ] {
            if (cfg!(miri) && span != 64) || (span > 8192 && kernels.dit_wide.is_none()) {
                continue;
            }
            let cyclic = span.checked_mul(2).expect("small row");
            let mut forward = vec![0.0; cyclic];
            let mut inverse = vec![0.0; cyclic];
            // SAFETY: one power-of-two stride per prime, within the catalog roots.
            unsafe {
                Twiddles::prepare(
                    &mut forward,
                    &mut inverse,
                    NonZeroUsize::new(cyclic).expect("positive stride"),
                    from_ref(prime),
                    kernels,
                    &Sequential,
                );
            }
            let tables = (forward.as_slice(), inverse.as_slice());
            let half = span >> 1;
            let above = span.checked_add(8).expect("small span");
            let three_halves = span.checked_add(half).expect("small span");
            for shape in [
                (above, above),
                (above, span),
                (span, above),
                (three_halves, three_halves),
                (cyclic, span),
                (span, cyclic),
                (cyclic, 8),
                (8, cyclic),
            ] {
                for (square, fused) in [(false, false), (false, true), (true, false), (true, true)]
                {
                    let flags = (
                        square,
                        fused,
                        NonZeroUsize::new(3).expect("positive budget"),
                    );
                    if square && shape.0 > three_halves {
                        continue;
                    }
                    check_row((prime, kernels, false), tables, span, shape, flags);
                    if span > 8192 {
                        check_row((prime, kernels, true), tables, span, shape, flags);
                        check_row((prime, &separate, false), tables, span, shape, flags);
                    }
                }
            }
        }
    }
}

fn check_forward_fusion(
    source: &[f64],
    roots: &[f64],
    prime: &crate::Prime,
    kernels: &Kernels,
    separate: &Kernels,
) {
    let mut expected = source.to_vec();
    let mut actual = source.to_vec();
    let unit_root = Roots {
        values: roots,
        node: 0,
    };
    Transform::new(prime, separate, &Sequential, false).forward(
        &mut expected,
        unit_root,
        source.len(),
        NonZeroUsize::new(2).expect("positive budget"),
    );
    Transform::new(prime, kernels, &Sequential, true).forward(
        &mut actual,
        unit_root,
        source.len(),
        NonZeroUsize::new(2).expect("positive budget"),
    );
    for (value, reference) in actual.into_iter().zip(expected) {
        assert_eq!(
            Scalar::reduce(value, prime.float_field().p(), prime.float_field().pinv()),
            Scalar::reduce(
                reference,
                prime.float_field().p(),
                prime.float_field().pinv()
            ),
            "len={}, prime={}",
            source.len(),
            prime.modulus(),
        );
    }
}
