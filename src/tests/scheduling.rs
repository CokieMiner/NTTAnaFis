//! Dense products across prime-task thresholds and changing worker budgets.

use core::num::NonZeroUsize;

use alloc::{vec, vec::Vec};

use core_anafis::{Budgeted, Sequential};
use rayon::ThreadPoolBuilder;

use crate::{
    ConvPlan, ConvWorkspace, Geometry, LIMB_BITS, Limb, MAX_PRIMES, MulPlan, MulWorkspace,
    PrimePrefix, profile::Profile,
};

use super::{
    convolution::signed_accumulate,
    plans::{forced, multiply, multiply_under},
    schoolbook::product,
};

/// Returns a profile of `count` primes with the widest tested coefficient.
fn wide_profile(count: usize) -> Profile {
    let bits = count
        .checked_mul(49)
        .and_then(|width| width.checked_sub(16))
        .expect("catalog coefficient capacity")
        .div_euclid(2);
    Profile {
        prefix: PrimePrefix::new(count).expect("catalog prefix"),
        chunk_bits: NonZeroUsize::new(bits).expect("positive coefficient width"),
    }
}

/// Returns `(B^left-1)*(B^right-1)`, where `B` is the native radix and `right <= left`.
fn all_ones_product(left: usize, right: usize) -> Vec<Limb> {
    let mut expected = vec![0; left.checked_add(right).expect("product span")];
    *expected.first_mut().expect("positive product span") = 1;
    expected
        .get_mut(right..)
        .expect("high product band")
        .fill(Limb::MAX);
    *expected.get_mut(left).expect("subtracted high power") =
        Limb::MAX.checked_sub(1).expect("native radix exceeds two");
    expected
}

#[test]
#[cfg_attr(
    miri,
    ignore = "The dense parallel boundary matrix uses large transform buffers"
)]
fn prime_tasks_form_products_and_squares_across_worker_budgets() {
    let pools = [1, 2, 3, 4, 5, 6, 7, 8, 2, 1].map(|workers| {
        let pool = ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .expect("test worker pool");
        (workers, pool)
    });
    for count in 1..=MAX_PRIMES {
        let profile = wide_profile(count);
        for length in [1023_usize, 1024, 1025, 4095, 4096, 4097] {
            let shorter = length.checked_sub(7).expect("positive shorter operand");
            for (right_len, square) in [(shorter, false), (length, true)] {
                let left = vec![Limb::MAX; length];
                let right = vec![Limb::MAX; right_len];
                let expected = all_ones_product(length, right_len);
                let geometry = forced(length, right_len, profile).expect("admitted dense shape");
                for (workers, pool) in &pools {
                    pool.install(|| {
                        let mut actual = vec![0; expected.len()];
                        if square {
                            multiply::<true>(&geometry, &mut actual, &left, &left);
                        } else {
                            multiply::<false>(&geometry, &mut actual, &left, &right);
                        }
                        assert_eq!(
                            actual, expected,
                            "np={count}, len={length}, workers={workers}, square={square}"
                        );
                    });
                }
            }
        }
    }
}

/// Budgets above the pool width only change how far work splits. A sequential
/// executor runs each split in order, so this covers the budget arithmetic of
/// every partition up to the largest budget an executor can report.
#[test]
#[cfg_attr(miri, ignore = "The budget matrix runs many complete dense products")]
fn products_accept_budgets_of_every_divisibility_up_to_the_largest() {
    let largest = usize::MAX.checked_sub(1).expect("large budget");
    let length = 4096_usize;
    let right_len = length.checked_sub(7).expect("positive shorter operand");
    let left = vec![Limb::MAX; length];
    let right = vec![Limb::MAX; right_len];
    let product = all_ones_product(length, right_len);
    let square = all_ones_product(length, length);
    for count in 1..=MAX_PRIMES {
        let profile = wide_profile(count);
        let mul_geometry = forced(length, right_len, profile).expect("admitted dense shape");
        let sqr_geometry = forced(length, length, profile).expect("admitted dense shape");
        for budget in (1_usize..=17).chain([largest, usize::MAX]) {
            let workers = NonZeroUsize::new(budget).expect("positive budget");
            let executor = Budgeted::new(&Sequential, workers);
            let mut mul_output = vec![0; product.len()];
            multiply_under::<false, _>(&mul_geometry, &mut mul_output, &left, &right, &executor);
            assert_eq!(mul_output, product, "np={count}, budget={budget}");
            let mut sqr_output = vec![0; square.len()];
            multiply_under::<true, _>(&sqr_geometry, &mut sqr_output, &left, &left, &executor);
            assert_eq!(sqr_output, square, "np={count}, budget={budget}, square");
        }
    }
}

/// A plan built for one worker splits into blocks, and stays exact when it runs
/// in a wider pool, whose workers split the primes of each block.
#[test]
#[cfg_attr(miri, ignore = "Worker pools run complete dense products")]
fn block_plans_form_products_and_convolutions_in_any_pool() {
    let single = ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("test worker pool");
    let (long_len, short_len) = (6000_usize, 300_usize);
    let long = vec![Limb::MAX; long_len];
    let short = vec![Limb::MAX; short_len];
    let expected = all_ones_product(long_len, short_len);
    let plan = single
        .install(|| MulPlan::new(long_len, short_len))
        .expect("small native shape");
    let bound = Limb::MAX >> 1;
    let (a, b) = (vec![bound; long_len], vec![bound; short_len]);
    let convolution = single
        .install(|| ConvPlan::new(long_len, short_len, bound, bound, false))
        .expect("small shape");
    let blocks = saturated_blocks(long_len, short_len, bound, convolution.stride());
    for workers in [1, 2, 3, 4, 8] {
        let pool = ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .expect("test worker pool");
        pool.install(|| {
            let mut mul_scratch = vec![0; plan.scratch_len()];
            let mut actual = vec![0; expected.len()];
            assert!(plan.mul(&mut actual, &long, &short, &mut mul_scratch));
            assert_eq!(actual, expected, "product, workers={workers}");
            let mut conv_scratch = vec![0; convolution.scratch_len()];
            let mut output = vec![Limb::MAX; blocks.len()];
            assert!(convolution.convolve(&mut output, &a, &b, &mut conv_scratch));
            assert_eq!(output, blocks, "convolution, workers={workers}");
            let mut retained = ConvWorkspace::new();
            retained.prepare(convolution);
            output.fill(Limb::MAX);
            assert!(retained.convolve(&mut output, &a, &b));
            assert_eq!(output, blocks, "retained convolution, workers={workers}");
        });
    }
}

/// A geometry planned for eight workers runs the blocks after the first in one
/// slot per worker, and stays exact under any budget: all-one limbs carry across
/// every window and every boundary between runs.
#[test]
#[cfg_attr(miri, ignore = "Dense products of every budget use large buffers")]
fn block_runs_stay_exact_under_every_budget() {
    let (long_len, short_len) = (16_384_usize, 64_usize);
    let lengths = (
        NonZeroUsize::new(long_len).expect("positive length"),
        NonZeroUsize::new(short_len).expect("positive length"),
    );
    let eight = NonZeroUsize::new(8).expect("positive budget");
    let geometry = Geometry::select(lengths.0, lengths.1, 1)
        .expect("small native shape")
        .cheapest_blocks(lengths, (1, eight), 1, |limbs| {
            Geometry::select(limbs, lengths.1, 1)
        });
    assert!(geometry.split().is_some_and(|split| split.slots == eight));
    let long = vec![Limb::MAX; long_len];
    let short = vec![Limb::MAX; short_len];
    let expected = all_ones_product(long_len, short_len);
    let largest = usize::MAX.checked_sub(1).expect("large budget");
    for budget in (1_usize..=17).chain([largest, usize::MAX]) {
        let workers = NonZeroUsize::new(budget).expect("positive budget");
        let mut actual = vec![0; expected.len()];
        let executor = Budgeted::new(&Sequential, workers);
        multiply_under::<false, _>(&geometry, &mut actual, &long, &short, &executor);
        assert_eq!(actual, expected, "budget={budget}");
    }
}

/// Plans built in a pool of eight workers stay exact in any pool, in either
/// operand order, with saturated convolutions of both signs that fill each
/// coefficient block.
#[test]
#[cfg_attr(miri, ignore = "Worker pools run complete dense products")]
fn block_runs_form_products_and_convolutions_in_any_pool() {
    let wide = ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .expect("test worker pool");
    let (long_len, short_len) = (16_384_usize, 64_usize);
    let long = vec![Limb::MAX; long_len];
    let short = vec![Limb::MAX; short_len];
    let expected = all_ones_product(long_len, short_len);
    let mixed: Vec<Limb> = (0..long_len)
        .map(|index| {
            Limb::MAX.rotate_left(u32::try_from(index % 61).expect("small shift"))
                ^ Limb::try_from(index).expect("small index")
        })
        .collect();
    let mixed_expected = product(&mixed, &short);
    let plan = wide
        .install(|| MulPlan::new(long_len, short_len))
        .expect("small native shape");
    let swapped = wide
        .install(|| MulPlan::new(short_len, long_len))
        .expect("small native shape");
    let bound = Limb::MAX >> 2;
    let (conv_long, conv_short) = (6000_usize, 300_usize);
    let unsigned = wide
        .install(|| ConvPlan::new(conv_long, conv_short, bound, bound, false))
        .expect("small shape");
    let (a, b) = (vec![bound; conv_long], vec![bound; conv_short]);
    let saturated = saturated_blocks(conv_long, conv_short, bound, unsigned.stride());
    let signed_long: Vec<Limb> = (0..conv_long)
        .map(|index| {
            if index % 3 == 0 {
                bound.wrapping_neg()
            } else {
                bound >> (index % 7)
            }
        })
        .collect();
    let signed_short: Vec<Limb> = (0..conv_short)
        .map(|index| {
            if index % 2 == 0 {
                bound
            } else {
                bound.wrapping_neg()
            }
        })
        .collect();
    let signed = wide
        .install(|| ConvPlan::new(conv_long, conv_short, bound, bound, true))
        .expect("small shape");
    let signed_expected = signed_accumulate(&signed_long, &signed_short, signed.stride());
    for workers in [1, 2, 3, 8, 16] {
        let pool = ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .expect("test worker pool");
        pool.install(|| {
            let mut scratch = vec![0; plan.scratch_len()];
            let mut actual = vec![0; expected.len()];
            assert!(plan.mul(&mut actual, &long, &short, &mut scratch));
            assert_eq!(actual, expected, "product, workers={workers}");
            let mut swapped_scratch = vec![0; swapped.scratch_len()];
            assert!(swapped.mul(&mut actual, &short, &mixed, &mut swapped_scratch));
            assert_eq!(actual, mixed_expected, "swapped product, workers={workers}");
            let mut workspace = MulWorkspace::new();
            workspace.prepare(plan);
            assert!(workspace.mul(&mut actual, &mixed, &short));
            assert_eq!(actual, mixed_expected, "mixed product, workers={workers}");
            let mut output = vec![Limb::MAX; saturated.len()];
            let mut conv_scratch = vec![0; unsigned.scratch_len()];
            assert!(unsigned.convolve(&mut output, &a, &b, &mut conv_scratch));
            assert_eq!(output, saturated, "convolution, workers={workers}");
            let mut retained = ConvWorkspace::new();
            retained.prepare(signed);
            let mut signed_output = vec![Limb::MAX; signed_expected.len()];
            assert!(retained.convolve(&mut signed_output, &signed_long, &signed_short));
            assert_eq!(
                signed_output, signed_expected,
                "signed convolution, workers={workers}"
            );
        });
    }
}

/// A signed plan built for one worker splits into blocks whose windows add
/// two's complement partial sums of mixed signs, which must not carry from one
/// coefficient block into the next, in any pool.
#[test]
#[cfg_attr(miri, ignore = "Worker pools run complete dense products")]
fn signed_block_plans_keep_each_coefficient_block_in_any_pool() {
    let single = ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("test worker pool");
    let (long_len, short_len) = (6000_usize, 300_usize);
    let bound = Limb::MAX >> 2;
    let long: Vec<Limb> = (0..long_len)
        .map(|index| {
            if index % 3 == 0 {
                bound.wrapping_neg()
            } else {
                bound >> (index % 7)
            }
        })
        .collect();
    let short: Vec<Limb> = (0..short_len)
        .map(|index| {
            if index % 2 == 0 {
                bound
            } else {
                bound.wrapping_neg()
            }
        })
        .collect();
    let plan = single
        .install(|| ConvPlan::new(long_len, short_len, bound, bound, true))
        .expect("small shape");
    let expected = signed_accumulate(&long, &short, plan.stride());
    for workers in [1, 3, 8] {
        let pool = ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .expect("test worker pool");
        pool.install(|| {
            let mut scratch = vec![0; plan.scratch_len()];
            let mut output = vec![Limb::MAX; expected.len()];
            assert!(plan.convolve(&mut output, &long, &short, &mut scratch));
            assert_eq!(output, expected, "signed convolution, workers={workers}");
            let mut retained = ConvWorkspace::new();
            retained.prepare(plan);
            output.fill(Limb::MAX);
            assert!(retained.convolve(&mut output, &long, &short));
            assert_eq!(
                output, expected,
                "retained signed convolution, workers={workers}"
            );
        });
    }
}

/// Returns the blocks of a convolution of `left >= right` copies of `max`.
///
/// Coefficient `k` has `min(k+1, right, left+right-1-k)` equal products, which
/// gives an oracle independent of the transform at lengths where parallel
/// packing and reconstruction split.
fn saturated_blocks(left: usize, right: usize, max: Limb, words: usize) -> Vec<Limb> {
    let term = product(&[max], &[max]);
    let last = left
        .checked_add(right)
        .and_then(|blocks| blocks.checked_sub(1))
        .expect("positive shape");
    let mut output = vec![0; last.checked_mul(words).expect("small shape")];
    for (index, block) in output.chunks_exact_mut(words).enumerate() {
        let count = index
            .checked_add(1)
            .expect("small index")
            .min(right)
            .min(last.checked_sub(index).expect("ordered index"));
        for (slot, limb) in block.iter_mut().zip(product(&term, &[count])) {
            *slot = limb;
        }
    }
    output
}

#[test]
#[cfg_attr(miri, ignore = "The parallel shapes use large transform buffers")]
fn parallel_convolutions_match_saturated_operands_across_worker_budgets() {
    let pools = [1, 2, 3, 4, 7, 8].map(|workers| {
        let pool = ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .expect("test worker pool");
        (workers, pool)
    });
    let (left, right) = (3000_usize, 2500_usize);
    let half = LIMB_BITS.div_euclid(2);
    for bits in [half.div_euclid(2), half, LIMB_BITS] {
        let unused = u32::try_from(LIMB_BITS.checked_sub(bits).expect("native bits"));
        let max = Limb::MAX >> unused.expect("small shift");
        let (a, b) = (vec![max; left], vec![max; right]);
        let plan = ConvPlan::new(left, right, max, max, false).expect("small shape");
        let words = plan.stride();
        let expected = saturated_blocks(left, right, max, words);
        for (workers, pool) in &pools {
            pool.install(|| {
                let mut scratch = vec![0; plan.scratch_len()];
                let mut planned = vec![Limb::MAX; expected.len()];
                assert!(plan.convolve(&mut planned, &a, &b, &mut scratch));
                assert_eq!(planned, expected, "bits={bits}, workers={workers}");
                let mut workspace = ConvWorkspace::new();
                workspace.prepare(plan);
                let mut retained = vec![Limb::MAX; expected.len()];
                assert!(workspace.convolve(&mut retained, &a, &b));
                assert_eq!(retained, expected, "bits={bits}, workers={workers}");
            });
        }
        let stride = words.checked_mul(LIMB_BITS).expect("small stride");
        let value_bits = NonZeroUsize::new(bits).expect("positive bits");
        let lengths = (
            NonZeroUsize::new(left).expect("positive length"),
            NonZeroUsize::new(right).expect("positive length"),
        );
        let geometry =
            Geometry::for_residues(lengths, (value_bits, value_bits), (stride, stride), false)
                .expect("small shape");
        let largest = usize::MAX.checked_sub(1).expect("large budget");
        for budget in (1_usize..=17).chain([largest, usize::MAX]) {
            let workers = NonZeroUsize::new(budget).expect("positive budget");
            let executor = Budgeted::new(&Sequential, workers);
            let mut actual = vec![Limb::MAX; expected.len()];
            multiply_under::<false, _>(&geometry, &mut actual, &a, &b, &executor);
            assert_eq!(actual, expected, "bits={bits}, budget={budget}");
        }
    }
}
