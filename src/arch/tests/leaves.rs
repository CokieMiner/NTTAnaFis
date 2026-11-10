#![expect(
    unsafe_code,
    reason = "Tests admit complete sixteen-point leaves, twiddle spans, and detected backends"
)]

use core::{num::NonZeroUsize, ptr::null, slice::from_ref};

use alloc::{vec, vec::Vec};

use core_anafis::Sequential;
use proptest::{collection, prelude::*, test_runner::TestRunner};

use crate::{
    PRIME_CATALOG, Prime,
    arch::{Columns, Kernels, Scalar},
    transform::Twiddles,
};

use super::admitted;

// (inverse, product, square) selects the four admitted leaf variants.
const LEAF_MODES: [(bool, bool, bool); 4] = [
    (false, false, false),
    (true, false, false),
    (true, true, false),
    (true, true, true),
];

#[test]
#[expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "Signed test residues have magnitude below 2*p < 2^51 and are exact binary64 integers"
)]
#[expect(
    clippy::too_many_lines,
    reason = "One property covers every leaf mode, field, and backend"
)]
fn fused_leaves_match_separate_stages_and_preserve_guards_for_every_backend() {
    let backends = admitted();
    let check = |words: &[u64]| {
        // Vector kernels may consume leaves in pairs.
        let blocks = words
            .len()
            .div_euclid(32)
            .checked_mul(2)
            .expect("small leaf count");
        let len = blocks.checked_mul(16).expect("small leaf span");
        if blocks == 0 {
            return;
        }
        for prime in PRIME_CATALOG {
            let modulus = i64::try_from(prime.modulus()).expect("50-bit modulus");
            let limit = modulus.checked_mul(2).expect("51-bit bound");
            let range = u64::try_from(
                limit
                    .checked_mul(2)
                    .expect("52-bit range")
                    .checked_sub(1)
                    .expect("positive range"),
            )
            .expect("positive range");
            let table_len = len.next_power_of_two();
            let mut forward = vec![0.0; table_len >> 1];
            let mut inverse = vec![0.0; table_len >> 1];
            // SAFETY: one power-of-two stride per prime, within the catalog roots.
            unsafe {
                Twiddles::prepare(
                    &mut forward,
                    &mut inverse,
                    NonZeroUsize::new(table_len >> 1).expect("positive stride"),
                    from_ref(prime),
                    &Kernels::SCALAR,
                    &Sequential,
                );
            }
            let source: Vec<_> = words
                .iter()
                .take(len)
                .map(|word| {
                    i64::try_from(word % range)
                        .expect("52-bit integer")
                        .checked_sub(limit.checked_sub(1).expect("positive limit"))
                        .expect("signed residue") as f64
                })
                .collect();
            let factors: Vec<_> = source.iter().rev().map(|value| -value).collect();
            for (inverted, product, squared) in LEAF_MODES {
                let roots = if inverted { &inverse } else { &forward };
                let mut expected = source.clone();
                if inverted && product {
                    for (value, &factor) in expected.iter_mut().zip(&factors) {
                        let multiplier = if squared { *value } else { factor };
                        *value = Scalar::mulmod(
                            *value,
                            multiplier,
                            prime.float_field().p(),
                            prime.float_field().pinv(),
                        );
                    }
                }
                staged(&mut expected, roots, prime, inverted);
                for backend in &backends {
                    let mut actual = vec![f64::NAN; len.checked_add(2).expect("guarded span")];
                    actual
                        .get_mut(1..=len)
                        .expect("data between guards")
                        .copy_from_slice(&source);
                    let (kernel, factor_ptr) = match (inverted, product, squared) {
                        (false, _, _) => (backend.dif16, null()),
                        (true, false, _) => (backend.dit16, null()),
                        (true, true, false) => (backend.dit16_mul, factors.as_ptr()),
                        (true, true, true) => (backend.dit16_sqr, null()),
                    };
                    // SAFETY: the data span contains complete initialized leaves;
                    // the two guards are outside it. The right spectrum is
                    // initialized and disjoint, or unused. Both input bounds
                    // are below 2*p; ISA support is detected.
                    unsafe {
                        kernel(
                            actual.as_mut_ptr().add(1),
                            factor_ptr,
                            roots.as_ptr(),
                            NonZeroUsize::new(blocks >> 1).expect("positive leaf pairs"),
                            0,
                            prime.float_field(),
                        );
                    }
                    assert!(actual.first().expect("guard").is_nan());
                    assert!(actual.last().expect("guard").is_nan());
                    for (&value, &reference) in actual.iter().skip(1).take(len).zip(&expected) {
                        assert!(value.abs() < 2.0 * prime.float_field().p());
                        assert_eq!(
                            Scalar::reduce(
                                value,
                                prime.float_field().p(),
                                prime.float_field().pinv()
                            ),
                            Scalar::reduce(
                                reference,
                                prime.float_field().p(),
                                prime.float_field().pinv()
                            )
                        );
                    }
                }
            }
        }
    };
    for len in [32, 64, 96, 128, 160] {
        let words: Vec<_> = [0, 1, u64::MAX, u64::MAX.checked_sub(1).expect("nonzero")]
            .into_iter()
            .cycle()
            .take(len)
            .collect();
        check(&words);
    }
    TestRunner::new(ProptestConfig::with_cases(if cfg!(miri) { 2 } else { 32 }))
        .run(&collection::vec(any::<u64>(), 32..=160), |words| {
            check(&words);
            Ok(())
        })
        .expect("fused leaf property");
}

fn staged(values: &mut [f64], roots: &[f64], prime: &Prime, inverse: bool) {
    let weights = |node: usize| {
        let twice = node.checked_mul(2).expect("small node");
        [
            *roots.get(node).expect("node root"),
            *roots.get(twice).expect("left root"),
            *roots
                .get(twice.checked_add(1).expect("right node"))
                .expect("right root"),
        ]
    };
    for (node, block) in values.as_chunks_mut::<16>().0.iter_mut().enumerate() {
        let parent = weights(node);
        if inverse {
            transpose_leaf(block);
        }
        // SAFETY: the block has sixteen initialized residues and both
        // complete root tables. Four-point child spans are disjoint. These short
        // ranges reach only the portable kernel, which admits any range.
        unsafe {
            let field = prime.float_field();
            if inverse {
                for (index, child) in block.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                    let child_roots = weights(
                        node.checked_mul(4)
                            .and_then(|v| v.checked_add(index))
                            .expect("child node"),
                    );
                    Scalar::radix4::<true, false>(
                        child.as_mut_ptr(),
                        child_roots.as_ptr(),
                        1,
                        Columns::new(0, 1),
                        field,
                    );
                }
                Scalar::radix4::<true, false>(
                    block.as_mut_ptr(),
                    parent.as_ptr(),
                    4,
                    Columns::new(0, 4),
                    field,
                );
            } else {
                Scalar::radix4::<false, false>(
                    block.as_mut_ptr(),
                    parent.as_ptr(),
                    4,
                    Columns::new(0, 4),
                    field,
                );
                for (index, child) in block.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                    let child_roots = weights(
                        node.checked_mul(4)
                            .and_then(|v| v.checked_add(index))
                            .expect("child node"),
                    );
                    Scalar::radix4::<false, false>(
                        child.as_mut_ptr(),
                        child_roots.as_ptr(),
                        1,
                        Columns::new(0, 1),
                        field,
                    );
                }
            }
        }
        if !inverse {
            transpose_leaf(block);
        }
    }
}

fn transpose_leaf(block: &mut [f64; 16]) {
    for row in 0_usize..4 {
        for column in 0_usize..row {
            let left = row
                .checked_mul(4)
                .and_then(|offset| offset.checked_add(column))
                .expect("leaf position");
            let right = column
                .checked_mul(4)
                .and_then(|offset| offset.checked_add(row))
                .expect("transposed position");
            block.swap(left, right);
        }
    }
}
