#![expect(
    unsafe_code,
    reason = "Tests admit detected SIMD kernels over complete, disjoint stage spans and aligned ranges"
)]

use core::{num::NonZeroUsize, ops::Range, slice::from_ref};

use alloc::{vec, vec::Vec};

use core_anafis::Sequential;
use proptest::{collection, prelude::*, test_runner::TestRunner};

use crate::{
    PRIME_CATALOG, Prime,
    arch::{Columns, Kernels, Scalar},
    transform::Twiddles,
};

use super::admitted;

#[test]
#[cfg_attr(miri, ignore = "Paired SIMD kernels require native execution")]
#[expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "Signed test residues are exact binary64 integers of magnitude below 2*p < 2^51"
)]
fn paired_stages_match_separate_stages_at_residue_and_range_boundaries() {
    let mut backends = admitted();
    backends.retain(|backend| backend.dif_wide.is_some() || backend.dit_wide.is_some());
    if backends.is_empty() {
        return;
    }
    let check = |words: &[u64]| {
        for prime in PRIME_CATALOG {
            let (p, pinv) = (prime.float_field().p(), prime.float_field().pinv());
            let limit = prime.modulus().checked_mul(2).expect("51-bit magnitude");
            let radius = limit.checked_sub(1).expect("positive radius");
            let range = radius
                .checked_mul(2)
                .and_then(|n| n.checked_add(1))
                .expect("52-bit range");
            for len in [256, 1024, 4096] {
                let quarter = len >> 4;
                let (mut forward, mut inverse) = (vec![0.0; len >> 1], vec![0.0; len >> 1]);
                // SAFETY: one power-of-two stride per prime, within the catalog roots.
                unsafe {
                    Twiddles::prepare(
                        &mut forward,
                        &mut inverse,
                        NonZeroUsize::new(len >> 1).expect("positive stride"),
                        from_ref(prime),
                        &Kernels::SCALAR,
                        &Sequential,
                    );
                }
                let source: Vec<_> = words
                    .iter()
                    .cycle()
                    .take(len)
                    .map(|word| {
                        let position = if *word == u64::MAX {
                            range.checked_sub(1).expect("last residue")
                        } else {
                            word % range
                        };
                        i64::try_from(position)
                            .expect("52-bit position")
                            .checked_sub(i64::try_from(radius).expect("51-bit radius"))
                            .expect("signed residue") as f64
                    })
                    .collect();
                for inverted in [false, true] {
                    let table = if inverted { &inverse } else { &forward };
                    let root = |index| *table.get(index).expect("admitted node root");
                    let packet = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 9, 11, 13, 15].map(root);
                    for span in [0..quarter, 0..8, 8..quarter] {
                        let mut expected = source.clone();
                        staged(&mut expected, &packet, quarter, &span, prime, inverted);
                        for backend in &backends {
                            let Some(kernel) = (if inverted {
                                backend.dit_wide
                            } else {
                                backend.dif_wide
                            }) else {
                                continue;
                            };
                            let mut actual =
                                vec![f64::NAN; len.checked_add(2).expect("guarded span")];
                            actual
                                .get_mut(1..=len)
                                .expect("output span")
                                .copy_from_slice(&source);
                            // SAFETY: the sixteen spans and fifteen root spans
                            // are complete, residues satisfy the stage bound,
                            // endpoints are eight-aligned, and ISA is detected.
                            unsafe {
                                kernel(
                                    actual.as_mut_ptr().add(1),
                                    packet.as_ptr(),
                                    quarter,
                                    Columns::new(span.start, span.end),
                                    prime.float_field(),
                                );
                            }
                            assert!(actual.first().expect("prefix guard").is_nan());
                            assert!(actual.last().expect("suffix guard").is_nan());
                            for (position, (&value, &reference)) in
                                actual.iter().skip(1).take(len).zip(&expected).enumerate()
                            {
                                if !span.contains(&(position % quarter)) {
                                    assert_eq!(value.to_bits(), reference.to_bits());
                                }
                                assert!(value.abs() < 2.5 * p);
                                assert_eq!(
                                    Scalar::reduce(value, p, pinv),
                                    Scalar::reduce(reference, p, pinv)
                                );
                            }
                        }
                    }
                }
            }
        }
    };
    check(&[0, 1, u64::MAX]);
    TestRunner::new(ProptestConfig::with_cases(16))
        .run(&collection::vec(any::<u64>(), 1..=64), |words| {
            check(&words);
            Ok(())
        })
        .expect("paired stage property");
}

fn staged(
    values: &mut [f64],
    roots: &[f64],
    quarter: usize,
    span: &Range<usize>,
    prime: &Prime,
    inverse: bool,
) {
    let parent_quarter = quarter.checked_mul(4).expect("parent span");
    let child_roots = |row: usize| {
        [
            *roots
                .get(row.checked_add(3).expect("child root"))
                .expect("first root"),
            *roots
                .get(row.checked_add(7).expect("child root"))
                .expect("left root"),
            *roots
                .get(row.checked_add(11).expect("child root"))
                .expect("right root"),
        ]
    };
    let field = prime.float_field();
    // SAFETY: values contains sixteen initialized quarter spans; roots has
    // the complete parent and child stages. Each requested range lies inside
    // its stage. Child slices are disjoint and stage roots are immutable.
    unsafe {
        let columns = Columns::new(span.start, span.end);
        if inverse {
            for (row, block) in values.chunks_exact_mut(parent_quarter).enumerate() {
                let child = child_roots(row);
                Scalar::radix4::<true, false>(
                    block.as_mut_ptr(),
                    child.as_ptr(),
                    quarter,
                    columns,
                    field,
                );
            }
        }
        for column in 0_usize..4 {
            let first = column
                .checked_mul(quarter)
                .and_then(|offset| offset.checked_add(span.start))
                .expect("parent start");
            let last = first
                .checked_add(span.end.checked_sub(span.start).expect("ordered range"))
                .expect("parent end");
            if inverse {
                Scalar::radix4::<true, false>(
                    values.as_mut_ptr(),
                    roots.as_ptr(),
                    parent_quarter,
                    Columns::new(first, last),
                    field,
                );
            } else {
                Scalar::radix4::<false, false>(
                    values.as_mut_ptr(),
                    roots.as_ptr(),
                    parent_quarter,
                    Columns::new(first, last),
                    field,
                );
            }
        }
        if !inverse {
            for (row, block) in values.chunks_exact_mut(parent_quarter).enumerate() {
                let child = child_roots(row);
                Scalar::radix4::<false, false>(
                    block.as_mut_ptr(),
                    child.as_ptr(),
                    quarter,
                    columns,
                    field,
                );
            }
        }
    }
}
