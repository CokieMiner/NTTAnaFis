//! Dense products across prime-task thresholds and changing worker budgets.

use core::num::NonZeroUsize;

use alloc::{vec, vec::Vec};

use core_anafis::{Budgeted, Sequential};
use rayon::ThreadPoolBuilder;

use crate::{
    ConvPlan, ConvWorkspace, Geometry, LIMB_BITS, Limb, MAX_PRIMES, MulPlan, PrimePrefix,
    profile::Profile,
};

use super::{
    planning::{forced, multiply, multiply_under},
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
        .install(|| ConvPlan::new(long_len, short_len, bound, bound))
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
        let plan = ConvPlan::new(left, right, max, max).expect("small shape");
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
        let geometry = Geometry::for_residues(lengths, (value_bits, value_bits), (stride, stride))
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
