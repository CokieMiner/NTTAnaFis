//! Differential multiplication across every catalog prefix and coefficient width.

use core::num::NonZeroUsize;

use alloc::{vec, vec::Vec};

use proptest::{collection, prelude::*};

use crate::{Geometry, Limb, MAX_PRIMES, MulPlan, MulWorkspace, PrimePrefix, profile::Profile};

use super::{
    plans::{forced, multiply},
    schoolbook::product,
};

/// All-one limbs give the longest carries, including across block windows.
#[test]
fn saturated_block_windows_carry_into_the_next_window() {
    for (long_len, short_len) in [(4096, 64), (4097, 64), (5000, 300), (8192, 25), (3000, 1)] {
        let long = vec![Limb::MAX; long_len];
        let short = vec![Limb::MAX; short_len];
        let expected = product(&long, &short);
        let plan = MulPlan::new(long_len, short_len).expect("small native shape");
        let mut scratch = vec![0; plan.scratch_len()];
        let mut actual = vec![0; expected.len()];
        assert!(plan.mul(&mut actual, &long, &short, &mut scratch));
        assert_eq!(actual, expected, "{long_len}x{short_len}");
    }
}

/// A long operand and a short one split into blocks, and similar lengths keep the
/// whole plan. Several workers either share the rows of blocks that hold 12288
/// spectrum floats per task that shares a prime's row, or run the blocks after
/// the first in one slot per worker, when they fill more than three quarters of
/// their waves of one block per worker.
#[test]
fn only_unbalanced_shapes_split_and_several_workers_share_rows_or_run_blocks_at_once() {
    let plan_of = |left: usize, right: usize, workers: NonZeroUsize| {
        let lengths = (
            NonZeroUsize::new(left).expect("positive length"),
            NonZeroUsize::new(right).expect("positive length"),
        );
        Geometry::select(lengths.0, lengths.1, 1)
            .expect("small native shape")
            .cheapest_blocks(lengths, (1, workers), 1, |limbs| {
                Geometry::select(limbs, lengths.0.min(lengths.1), 1)
            })
    };
    let split = plan_of(65_536, 64, NonZeroUsize::MIN)
        .split()
        .expect("unbalanced shape");
    assert!(!split.swapped);
    assert!(split.block_limbs >= 64 && split.block_limbs <= 32_768);
    assert_eq!(split.step_limbs, split.block_limbs);
    assert_eq!(split.overlap_limbs, 64);
    assert_eq!(split.slots, NonZeroUsize::MIN);
    let swapped = plan_of(64, 65_536, NonZeroUsize::MIN).split();
    assert_eq!(swapped.map(|plan| plan.swapped), Some(true));
    for (long, short) in [(65_536, 64), (16_384, 64), (6000, 300), (4096, 64)] {
        for workers in [2, 3, 8, 16] {
            let budget = NonZeroUsize::new(workers).expect("positive budget");
            let parallel = plan_of(long, short, budget);
            let Some(blocks) = parallel.split() else {
                continue;
            };
            let count = parallel.prefix().count();
            let shares = parallel.prefix().tasks(budget).get().div_euclid(count);
            let floats = parallel
                .transform_len()
                .get()
                .checked_mul(count)
                .expect("admitted spectra");
            let least = shares.checked_mul(12_288).expect("few tasks per prime");
            if blocks.slots == NonZeroUsize::MIN {
                assert!(floats >= least, "{long}x{short}, workers={workers}");
            } else {
                assert_eq!(blocks.slots, budget, "{long}x{short}, workers={workers}");
                let later = long
                    .div_ceil(blocks.block_limbs)
                    .checked_sub(1)
                    .expect("at least two blocks");
                let capacity = later
                    .div_ceil(workers)
                    .checked_mul(workers)
                    .expect("small counts");
                assert!(
                    later.checked_mul(4) > capacity.checked_mul(3),
                    "{long}x{short}, workers={workers}"
                );
            }
        }
    }
    assert!(
        plan_of(65_536, 64, NonZeroUsize::new(8).expect("positive budget"))
            .split()
            .is_some_and(|blocks| blocks.slots.get() == 8)
    );
    assert_eq!(plan_of(65_536, 32_768, NonZeroUsize::MIN).split(), None);
    assert_eq!(plan_of(1000, 1000, NonZeroUsize::MIN).split(), None);
}

/// A costlier CRT term favors fewer primes, and every plan stays exact.
#[test]
fn crt_cost_selects_fewer_primes_and_stays_exact() {
    let count_of = |len: usize, crt_cost: u8| {
        let limbs = NonZeroUsize::new(len).expect("positive length");
        Geometry::select(limbs, limbs, crt_cost)
            .expect("small native shape")
            .prefix()
            .count()
    };
    assert!(count_of(1024, 3) < count_of(1024, 1));
    for crt_cost in [1, 2, 3, 4, 8] {
        for (left_len, right_len) in [(64, 64), (65, 64), (300, 257)] {
            let left = vec![Limb::MAX; left_len];
            let right = vec![Limb::MAX; right_len];
            let geometry = Geometry::select(
                NonZeroUsize::new(left_len).expect("positive length"),
                NonZeroUsize::new(right_len).expect("positive length"),
                crt_cost,
            )
            .expect("small native shape");
            let expected = product(&left, &right);
            let mut actual = vec![0; expected.len()];
            multiply::<false>(&geometry, &mut actual, &left, &right);
            assert_eq!(actual, expected, "{crt_cost}: {left_len}x{right_len}");
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 48 }))]
    #[test]
    fn multiplication_and_squaring_match_schoolbook_for_every_prime_count(
        left in collection::vec(any::<Limb>(), 1..=40), right in collection::vec(any::<Limb>(), 1..=40),
    ) {
        let expected = product(&left, &right);
        let expected_square = product(&left, &left);
        for count in 1..=MAX_PRIMES {
            let bits = count.checked_mul(49).expect("catalog capacity").checked_sub(16).expect("positive width").div_euclid(2);
            let profile = Profile { prefix: PrimePrefix::new(count).expect("catalog prefix"), chunk_bits: NonZeroUsize::new(bits).expect("positive width") };
            let geometry = forced(left.len(), right.len(), profile).expect("conservative profile");
            let mut actual = vec![Limb::MAX; expected.len()];
            multiply::<false>(&geometry, &mut actual, &left, &right);
            prop_assert_eq!(&actual, &expected, "prime count {}", count);
            let square = forced(left.len(), left.len(), profile).expect("conservative square profile");
            actual.resize(expected_square.len(), Limb::MAX);
            multiply::<true>(&square, &mut actual, &left, &left);
            prop_assert_eq!(&actual, &expected_square, "square prime count {}", count);
        }
    }

    #[test]
    fn planned_and_retained_execution_match_schoolbook(
        left in collection::vec(any::<Limb>(), 1..=40), right in collection::vec(any::<Limb>(), 1..=40),
    ) {
        let expected = product(&left, &right);
        let expected_square = product(&left, &left);
        let mut workspace = MulWorkspace::new();
        let mut actual = vec![Limb::MAX; expected.len()];
        prop_assert!(workspace.mul(&mut actual, &left, &right));
        prop_assert_eq!(&actual, &expected);
        let mut square = vec![Limb::MAX; expected_square.len()];
        prop_assert!(workspace.sqr(&mut square, &left));
        prop_assert_eq!(&square, &expected_square);
        let mul_plan = MulPlan::new(left.len(), right.len()).expect("small native shape");
        let mut mul_scratch = vec![0; mul_plan.scratch_len()];
        actual.fill(Limb::MAX);
        prop_assert!(mul_plan.mul(&mut actual, &left, &right, &mut mul_scratch));
        prop_assert_eq!(&actual, &expected);
        let sqr_plan = MulPlan::new(left.len(), left.len()).expect("small native shape");
        let mut sqr_scratch = vec![0; sqr_plan.sqr_scratch_len()];
        square.fill(Limb::MAX);
        prop_assert!(sqr_plan.sqr(&mut square, &left, &mut sqr_scratch));
        prop_assert_eq!(&square, &expected_square);
    }

    /// An operand much longer than the other runs in blocks, in either order, through
    /// both execution models.
    #[test]
    fn unbalanced_products_match_schoolbook_in_either_operand_order(
        short in collection::vec(any::<Limb>(), 1..=40),
        long_len in 200_usize..=1500,
        seed in any::<Limb>(),
        swapped in any::<bool>(),
    ) {
        let long: Vec<_> = (0..long_len)
            .map(|index| seed.rotate_left(u32::try_from(index % 61).expect("small")) ^ index)
            .collect();
        let (left, right) = if swapped { (&short, &long) } else { (&long, &short) };
        let expected = product(left, right);
        let plan = MulPlan::new(left.len(), right.len()).expect("small native shape");
        let mut scratch = vec![0; plan.scratch_len()];
        let mut actual = vec![Limb::MAX; expected.len()];
        prop_assert!(plan.mul(&mut actual, left, right, &mut scratch));
        prop_assert_eq!(&actual, &expected);
        let mut workspace = MulWorkspace::new();
        actual.fill(Limb::MAX);
        prop_assert!(workspace.mul(&mut actual, left, right));
        prop_assert_eq!(&actual, &expected);
    }

    #[test]
    fn retained_tables_serve_shorter_and_longer_shapes_in_any_order(
        shapes in collection::vec((1_usize..=300, 1_usize..=300), 1..=6),
        seed in any::<Limb>(),
    ) {
        let mut workspace = MulWorkspace::new();
        for (left_len, right_len) in shapes {
            let left: Vec<_> = (0..left_len).map(|index| seed.wrapping_mul(index | 1)).collect();
            let right: Vec<_> = (0..right_len).map(|index| seed.rotate_left(13) ^ index).collect();
            let expected = product(&left, &right);
            let mut actual = vec![Limb::MAX; expected.len()];
            prop_assert!(workspace.mul(&mut actual, &left, &right));
            prop_assert_eq!(&actual, &expected);
        }
    }
}
