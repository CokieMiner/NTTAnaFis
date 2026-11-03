//! Direct residue convolutions checked against independent block accumulation.

use core::num::NonZeroUsize;

use alloc::{vec, vec::Vec};

use proptest::{collection, prelude::*};

use crate::{
    Budgeted, ConvPlan, ConvWorkspace, Geometry, LIMB_BITS, Limb, MAX_PRIMES, limb::Cofactors,
};

use super::{planning::multiply_under, schoolbook::product};

/// Returns the largest coefficient of `bits` bits.
fn bound_of(bits: usize) -> Limb {
    let unused = LIMB_BITS.checked_sub(bits).expect("native coefficient");
    Limb::MAX
        .checked_shr(u32::try_from(unused).expect("small shift"))
        .unwrap_or(0)
}

/// Returns `len_a+len_b-1` blocks of `words` limbs by schoolbook accumulation.
fn accumulate(a: &[Limb], b: &[Limb], words: usize) -> Vec<Limb> {
    let blocks = a
        .len()
        .checked_add(b.len())
        .and_then(|blocks| blocks.checked_sub(1))
        .expect("small oracle span");
    let mut output = vec![0; blocks.checked_mul(words).expect("small oracle span")];
    let mut chunks: Vec<&mut [Limb]> = output.chunks_exact_mut(words).collect();
    for (i, &x) in a.iter().enumerate() {
        for (j, &y) in b.iter().enumerate() {
            let term = product(&[x], &[y]);
            let block = chunks
                .get_mut(i.checked_add(j).expect("oracle index"))
                .expect("oracle block");
            let mut carry = false;
            for (position, slot) in block.iter_mut().enumerate() {
                let limb = term.get(position).copied().unwrap_or(0);
                (*slot, carry) = slot.carrying_add(limb, carry);
            }
            assert!(!carry, "every coefficient fits its block");
            assert!(
                term.iter().skip(words).all(|&limb| limb == 0),
                "every term fits its block"
            );
        }
    }
    output
}

/// Runs every execution path of one plan and compares it with the oracle.
fn assert_all_paths(a: &[Limb], b: &[Limb], max_a: Limb, max_b: Limb) {
    let plan = ConvPlan::new(a.len(), b.len(), max_a, max_b).expect("small shape");
    let expected = accumulate(a, b, plan.stride());
    assert_eq!(expected.len(), plan.output_len());
    let mut scratch = vec![Limb::MAX; plan.scratch_len()];
    let mut planned = vec![Limb::MAX; plan.output_len()];
    assert!(plan.convolve(&mut planned, a, b, &mut scratch));
    assert_eq!(planned, expected);
    let mut workspace = ConvWorkspace::new();
    workspace.prepare(plan);
    let mut retained = vec![Limb::MAX; plan.output_len()];
    assert!(workspace.convolve(&mut retained, a, b));
    assert_eq!(retained, expected);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(if cfg!(miri) { 4 } else { 192 }))]
    #[test]
    fn convolutions_match_schoolbook_for_all_bounds_and_shapes(
        left in collection::vec(any::<Limb>(), 1..=70),
        right in collection::vec(any::<Limb>(), 1..=70),
        bits_a in 0..=LIMB_BITS,
        bits_b in 0..=LIMB_BITS,
    ) {
        let (max_a, max_b) = (bound_of(bits_a), bound_of(bits_b));
        let a: Vec<Limb> = left.iter().map(|value| value & max_a).collect();
        let b: Vec<Limb> = right.iter().map(|value| value & max_b).collect();
        assert_all_paths(&a, &b, max_a, max_b);
    }

    #[test]
    fn one_operand_slice_used_twice_matches_two_distinct_copies(
        values in collection::vec(any::<Limb>(), 1..=70),
        bits in 0..=LIMB_BITS,
    ) {
        let max = bound_of(bits);
        let a: Vec<Limb> = values.iter().map(|value| value & max).collect();
        let copy = a.clone();
        let plan = ConvPlan::new(a.len(), a.len(), max, max).expect("small shape");
        let mut scratch = vec![0; plan.scratch_len()];
        let mut distinct = vec![0; plan.output_len()];
        let mut aliased = vec![Limb::MAX; plan.output_len()];
        prop_assert!(plan.convolve(&mut distinct, &a, &copy, &mut scratch));
        prop_assert!(plan.convolve(&mut aliased, &a, &a, &mut scratch));
        prop_assert_eq!(&aliased, &distinct);
        let mut workspace = ConvWorkspace::new();
        workspace.prepare(plan);
        aliased.fill(Limb::MAX);
        prop_assert!(workspace.convolve(&mut aliased, &a, &a));
        prop_assert_eq!(&aliased, &distinct);
    }
}

/// Saturated operands reach `t*A*B`, the largest coefficient the bound admits.
#[test]
fn saturated_operands_reach_the_bound_at_every_block_edge() {
    let half = LIMB_BITS.div_euclid(2);
    for (left, right) in [
        (1, 1),
        (1, 40),
        (8, 8),
        (9, 9),
        (24, 24),
        (24, 100),
        (25, 25),
        (25, 100),
        (31, 2),
        (32, 1),
        (33, 33),
    ] {
        for (bits_a, bits_b) in [
            (0, 0),
            (1, 1),
            (half, half),
            (half, LIMB_BITS),
            (LIMB_BITS, 1),
            (LIMB_BITS, LIMB_BITS),
        ] {
            let (max_a, max_b) = (bound_of(bits_a), bound_of(bits_b));
            assert_all_paths(&vec![max_a; left], &vec![max_b; right], max_a, max_b);
        }
    }
}

/// Operands of very different lengths run in blocks, in either order, and block
/// windows accumulate the columns they share at every coefficient width.
#[test]
fn unbalanced_convolutions_match_the_oracle_across_block_windows() {
    for (long, short) in [(1000, 25), (1001, 30), (2048, 64), (2500, 100), (3000, 200)] {
        for (bits_long, bits_short) in [
            (1, 1),
            (8, LIMB_BITS),
            (LIMB_BITS, 8),
            (LIMB_BITS, LIMB_BITS),
        ] {
            let (max_long, max_short) = (bound_of(bits_long), bound_of(bits_short));
            let saturated = (vec![max_long; long], vec![max_short; short]);
            assert_all_paths(&saturated.0, &saturated.1, max_long, max_short);
            assert_all_paths(&saturated.1, &saturated.0, max_short, max_long);
            let mixed: Vec<Limb> = (0..long)
                .map(|index| index.wrapping_mul(0x9E37) & max_long)
                .collect();
            assert_all_paths(&mixed, &saturated.1, max_long, max_short);
        }
    }
}

/// A prefix serves a bound equal to its capacity and rejects nothing below it.
/// A capacity of `c` admits coefficients below `2^c`, so the next bit needs one
/// more prime. Larger prefixes run with blocks wider than any real coefficient.
#[test]
fn every_prime_prefix_serves_a_bound_at_its_capacity() {
    let a: Vec<Limb> = (1..=40).map(|value| Limb::MAX - value).collect();
    let b: Vec<Limb> = (1..=33).map(|value| Limb::MAX - 3 * value).collect();
    for count in 1..=MAX_PRIMES {
        let table = Cofactors::of_count(count);
        let width = NonZeroUsize::new(LIMB_BITS).expect("positive limb width");
        let select = |bound: usize| {
            let words = bound.div_ceil(LIMB_BITS);
            Geometry::for_residues(
                (
                    NonZeroUsize::new(a.len()).expect("positive length"),
                    NonZeroUsize::new(b.len()).expect("positive length"),
                ),
                (width, width),
                (bound, words.checked_mul(LIMB_BITS).expect("small stride")),
            )
        };
        let geometry = select(table.capacity).expect("bound at capacity");
        assert_eq!(geometry.prefix().count(), count);
        let next = table.capacity.checked_add(1).expect("small capacity");
        if count < MAX_PRIMES {
            assert_eq!(
                select(next).expect("next prefix").prefix().count(),
                count.checked_add(1).expect("small catalog")
            );
        } else {
            assert!(select(next).is_none());
        }
        // Sums of near-maximal two-limb products need three limbs per block.
        if geometry.output_stride().get() >= 3 * LIMB_BITS {
            let words = geometry.output_stride().get().div_euclid(LIMB_BITS);
            let expected = accumulate(&a, &b, words);
            Budgeted::resolve(|executor| {
                let mut actual = vec![Limb::MAX; expected.len()];
                multiply_under::<false, _>(&geometry, &mut actual, &a, &b, executor);
                assert_eq!(actual, expected, "np={count}");
            });
        }
    }
}

#[test]
fn block_width_is_the_smallest_that_holds_the_bound() {
    let half = bound_of(LIMB_BITS.div_euclid(2));
    let words = |terms, max_a, max_b| {
        ConvPlan::new(terms, terms, max_a, max_b)
            .expect("small shape")
            .stride()
    };
    assert_eq!(words(1, 0, 0), 1);
    assert_eq!(words(1, 0, Limb::MAX), 1);
    assert_eq!(words(1, Limb::MAX, 1), 1);
    assert_eq!(words(1, half, half), 1);
    assert_eq!(words(2, half, half), 2);
    assert_eq!(words(1, Limb::MAX, Limb::MAX), 2);
    assert_eq!(words(2, Limb::MAX, Limb::MAX), 3);
}

#[test]
fn success_preserves_the_destination_suffix() {
    let plan = ConvPlan::new(3, 2, 7, 7).expect("small shape");
    let long = plan.output_len().checked_add(2).expect("small output");
    let mut scratch = vec![0; plan.scratch_len()];
    let mut dst = vec![Limb::MAX; long];
    assert!(plan.convolve(&mut dst, &[1, 2, 3], &[4, 5], &mut scratch));
    assert_eq!(dst, [4, 13, 22, 15, Limb::MAX, Limb::MAX]);
}

#[test]
fn rejected_calls_preserve_the_destination() {
    // The first shape sums columns without scratch; the second runs the transform.
    for (left, right, needs_scratch) in [(3, 2, false), (30, 30, true)] {
        let plan = ConvPlan::new(left, right, 7, 7).expect("small shape");
        let (a, b) = (vec![7; left], vec![7; right]);
        let mut above = a.clone();
        *above.last_mut().expect("positive shape") = 8;
        let short_a = a.get(1..).expect("positive shape");
        let short_b = b.get(1..).expect("positive shape");
        let mut scratch = vec![0; plan.scratch_len()];
        assert_eq!(!scratch.is_empty(), needs_scratch);
        let long = plan.output_len().checked_add(1).expect("small output");
        let short_output = plan.output_len().checked_sub(1).expect("positive output");
        let mut workspace = ConvWorkspace::new();
        let mut untouched = vec![Limb::MAX; long];
        assert!(!workspace.convolve(&mut untouched, &a, &b));
        workspace.prepare(plan);
        for (first, second) in [(short_a, &b[..]), (&a[..], short_b), (&above[..], &b[..])] {
            assert!(!plan.convolve(&mut untouched, first, second, &mut scratch));
            assert!(!workspace.convolve(&mut untouched, first, second));
        }
        if let Some(last) = scratch.len().checked_sub(1) {
            let tight = scratch.get_mut(..last).expect("short scratch");
            assert!(!plan.convolve(&mut untouched, &a, &b, tight));
        }
        let short = untouched.get_mut(..short_output).expect("short output");
        assert!(!plan.convolve(short, &a, &b, &mut scratch));
        assert!(!workspace.convolve(short, &a, &b));
        assert!(untouched.iter().all(|&limb| limb == Limb::MAX));
    }
}

#[test]
fn unavailable_shapes_have_no_plan() {
    assert!(ConvPlan::new(0, 1, 1, 1).is_none());
    assert!(ConvPlan::new(1, 0, 1, 1).is_none());
    assert!(ConvPlan::new(usize::MAX, 1, 1, 1).is_none());
    assert!(ConvPlan::new(usize::MAX, usize::MAX, 1, 1).is_none());
    assert!(ConvPlan::new(usize::MAX.div_euclid(LIMB_BITS), 1, 1, 1).is_none());
}

/// Tables grow with the longest transform and the largest prime prefix, and a
/// smaller later plan runs in the storage of a larger earlier one.
#[test]
fn one_workspace_serves_plans_of_any_order() {
    let mut workspace = ConvWorkspace::new();
    for (left, right, bits) in [
        (40, 40, 8),
        (300, 200, LIMB_BITS),
        (3, 2, 1),
        (100, 100, 16),
        (1, 1, LIMB_BITS),
        (500, 3, LIMB_BITS.div_euclid(2)),
    ] {
        let max = bound_of(bits);
        let a = vec![max; left];
        let b = vec![max; right];
        let plan = ConvPlan::new(left, right, max, max).expect("small shape");
        workspace.prepare(plan);
        let mut actual = vec![Limb::MAX; plan.output_len()];
        assert!(workspace.convolve(&mut actual, &a, &b));
        assert_eq!(actual, accumulate(&a, &b, plan.stride()));
    }
}
