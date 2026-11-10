//! Direct residue convolutions checked against independent block accumulation.

use core::num::NonZeroUsize;

use alloc::{vec, vec::Vec};

use proptest::{collection, prelude::*};

use crate::{
    Budgeted, ConvPlan, ConvWorkspace, Geometry, LIMB_BITS, Limb, MAX_PRIMES, limb::Cofactors,
};

use super::{plans::multiply_under, schoolbook::product};

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
    let plan = ConvPlan::new(a.len(), b.len(), max_a, max_b, false).expect("small shape");
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

/// Returns `len_a+len_b-1` blocks of `words` limbs holding the two's complement
/// sums of signed products, each formed in `i128` and sign-extended.
pub fn signed_accumulate(a: &[Limb], b: &[Limb], words: usize) -> Vec<Limb> {
    let blocks = a
        .len()
        .checked_add(b.len())
        .and_then(|blocks| blocks.checked_sub(1))
        .expect("small oracle span");
    let mask = u128::try_from(Limb::MAX).expect("a native limb fits u128");
    let mut output = vec![0; blocks.checked_mul(words).expect("small oracle span")];
    let mut chunks: Vec<&mut [Limb]> = output.chunks_exact_mut(words).collect();
    for (i, &x) in a.iter().enumerate() {
        for (j, &y) in b.iter().enumerate() {
            let left = i128::try_from(x.cast_signed()).expect("native limb");
            let right = i128::try_from(y.cast_signed()).expect("native limb");
            let term = left.checked_mul(right).expect("magnitudes below 2^63");
            let block = chunks
                .get_mut(i.checked_add(j).expect("oracle index"))
                .expect("oracle block");
            let mut carry = false;
            for (position, slot) in block.iter_mut().enumerate() {
                let shift = u32::try_from(position.checked_mul(LIMB_BITS).expect("small shift"))
                    .expect("small shift");
                let part = term.checked_shr(shift).unwrap_or(term >> 127);
                let limb = Limb::try_from(part.cast_unsigned() & mask).expect("one limb");
                (*slot, carry) = slot.carrying_add(limb, carry);
            }
        }
    }
    output
}

/// Signed values of magnitude at most `max`, a mask below the sign bit, with the
/// sign taken from each generated limb's top bit.
fn signed_values(values: &[Limb], max: Limb) -> Vec<Limb> {
    values
        .iter()
        .map(|value| {
            let magnitude = value & max;
            if value.cast_signed() < 0 {
                magnitude.wrapping_neg()
            } else {
                magnitude
            }
        })
        .collect()
}

/// Runs both execution paths of one signed plan and compares them with the oracle.
fn assert_signed_paths(a: &[Limb], b: &[Limb], max_a: Limb, max_b: Limb) {
    let plan = ConvPlan::new(a.len(), b.len(), max_a, max_b, true).expect("small shape");
    let expected = signed_accumulate(a, b, plan.stride());
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
    fn signed_convolutions_match_an_independent_oracle(
        left in collection::vec(any::<Limb>(), 1..=70),
        right in collection::vec(any::<Limb>(), 1..=70),
        bits_a in 0..LIMB_BITS,
        bits_b in 0..LIMB_BITS,
    ) {
        let (max_a, max_b) = (bound_of(bits_a), bound_of(bits_b));
        assert_signed_paths(&signed_values(&left, max_a), &signed_values(&right, max_b), max_a, max_b);
    }

    #[test]
    fn one_operand_slice_used_twice_matches_two_distinct_copies(
        values in collection::vec(any::<Limb>(), 1..=70),
        bits in 0..=LIMB_BITS,
    ) {
        let max = bound_of(bits);
        let a: Vec<Limb> = values.iter().map(|value| value & max).collect();
        let copy = a.clone();
        let plan = ConvPlan::new(a.len(), a.len(), max, max, false).expect("small shape");
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

/// Operands of unequal lengths match the oracle in either order at every
/// coefficient width. Plans that split into blocks are checked in the
/// scheduling tests, which plan them for one worker.
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
                false,
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
        ConvPlan::new(terms, terms, max_a, max_b, false)
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
    let plan = ConvPlan::new(3, 2, 7, 7, false).expect("small shape");
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
        let plan = ConvPlan::new(left, right, 7, 7, false).expect("small shape");
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
    assert!(ConvPlan::new(0, 1, 1, 1, false).is_none());
    assert!(ConvPlan::new(1, 0, 1, 1, false).is_none());
    assert!(ConvPlan::new(usize::MAX, 1, 1, 1, false).is_none());
    assert!(ConvPlan::new(usize::MAX, usize::MAX, 1, 1, false).is_none());
    assert!(ConvPlan::new(usize::MAX.div_euclid(LIMB_BITS), 1, 1, 1, false).is_none());
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
        let plan = ConvPlan::new(left, right, max, max, false).expect("small shape");
        workspace.prepare(plan);
        let mut actual = vec![Limb::MAX; plan.output_len()];
        assert!(workspace.convolve(&mut actual, &a, &b));
        assert_eq!(actual, accumulate(&a, &b, plan.stride()));
    }
}

/// Extreme signed operands, all negative or alternating, reach the bound in
/// summed columns, whole transforms, and block windows alike.
#[test]
fn signed_extremes_reach_the_bound_in_columns_transforms_and_blocks() {
    let top = LIMB_BITS.checked_sub(1).expect("positive limb width");
    for (left, right) in [
        (1, 1),
        (8, 8),
        (24, 100),
        (25, 25),
        (33, 33),
        (1000, 25),
        (2048, 64),
    ] {
        for (bits_a, bits_b) in [(0, 0), (1, 1), (top, 1), (top, top)] {
            let (max_a, max_b) = (bound_of(bits_a), bound_of(bits_b));
            let negative = (
                vec![max_a.wrapping_neg(); left],
                vec![max_b.wrapping_neg(); right],
            );
            assert_signed_paths(&negative.0, &negative.1, max_a, max_b);
            let alternating: Vec<Limb> = (0..left)
                .map(|index| {
                    if index % 2 == 0 {
                        max_a
                    } else {
                        max_a.wrapping_neg()
                    }
                })
                .collect();
            assert_signed_paths(&alternating, &negative.1, max_a, max_b);
            assert_signed_paths(&negative.1, &alternating, max_b, max_a);
        }
    }
}

/// A signed bound needs a sign bit in one limb, and a value beyond the bound is
/// rejected without writing.
#[test]
fn signed_plans_reject_wide_bounds_and_values_beyond_them() {
    let half = bound_of(LIMB_BITS.checked_sub(1).expect("positive limb width"));
    let past = half.checked_add(1).expect("half of the range");
    assert!(ConvPlan::new(4, 4, past, 1, true).is_none());
    let plan = ConvPlan::new(4, 4, 3, 3, true).expect("small shape");
    let mut scratch = vec![0; plan.scratch_len()];
    let mut dst = vec![7; plan.output_len()];
    let beyond = [1, Limb::MAX.wrapping_sub(3), 0, 2];
    assert!(!plan.convolve(&mut dst, &beyond, &[1, 2, 3, 0], &mut scratch));
    assert!(dst.iter().all(|&limb| limb == 7));
}
