#![expect(
    unsafe_code,
    reason = "Root-order checks use canonical bases and exponents within the catalog valuation"
)]

use core::num::NonZeroUsize;

use crate::{
    Geometry, LIMB_BITS, MAX_PRIMES, MAX_ROOT_ORDER, MulPlan, PRIME_CATALOG, PrimePrefix,
    profile::Profile,
};

use super::planning::forced;

/// A row of `3*2^k` points holds the convolution and both operands, and needs
/// the table of the doubled transform.
#[test]
fn rows_take_the_shortest_length_that_holds_the_convolution_and_its_operands() {
    let byte = NonZeroUsize::new(8).expect("positive bits");
    for ((left, right), expected) in [
        ((1, 1), (32, 32, 16)),
        ((30, 30), (64, 64, 32)),
        ((40, 40), (96, 64, 64)),
        ((49, 48), (96, 64, 64)),
        ((49, 49), (128, 128, 64)),
        ((64, 33), (96, 64, 64)),
        ((65, 32), (128, 128, 64)),
        ((70, 20), (128, 128, 64)),
        ((3000, 2500), (6144, 4096, 4096)),
        ((4097, 1000), (8192, 8192, 4096)),
    ] {
        let lengths = (
            NonZeroUsize::new(left).expect("positive length"),
            NonZeroUsize::new(right).expect("positive length"),
        );
        let geometry =
            Geometry::for_residues(lengths, (byte, byte), (16, LIMB_BITS)).expect("small shape");
        assert_eq!(
            (
                geometry.transform_len().get(),
                geometry.cyclic_len(),
                geometry.table_len().get()
            ),
            expected,
            "{left}x{right}"
        );
    }
}

#[test]
fn native_plans_cover_profiles_shapes_and_reject_unaddressable_geometry() {
    for count in 1..=MAX_PRIMES {
        let chunk_bits = NonZeroUsize::new(
            count
                .checked_mul(49)
                .expect("small catalog")
                .checked_sub(16)
                .expect("positive capacity")
                .div_euclid(2),
        )
        .expect("positive coefficient width");
        let profile = Profile {
            prefix: PrimePrefix::new(count).expect("catalog prefix"),
            chunk_bits,
        };
        for (left, right) in [(1, 1), (3, 7), (4, 4), (16, 65), (257, 256)] {
            assert!(forced(left, right, profile).is_some());
        }
        assert!(forced(0, 1, profile).is_none());
        assert!(forced(1, 0, profile).is_none());
    }
    for (left, right) in [(1, 1), (3, 7), (4, 4), (16, 65), (257, 256)] {
        let plan = MulPlan::new(left, right).expect("small native shape");
        assert!(plan.scratch_len() >= plan.sqr_scratch_len());
    }
    for (left, right) in [(0, 1), (1, 0), (usize::MAX, 1)] {
        assert!(MulPlan::new(left, right).is_none());
    }
    assert!(Geometry::select(NonZeroUsize::MAX, NonZeroUsize::MIN, 1).is_none());
    let oversized = usize::MAX.div_euclid(LIMB_BITS);
    assert!(MulPlan::new(oversized, oversized).is_none());
    assert!(PrimePrefix::new(0).is_none());
    assert!(PrimePrefix::new(MAX_PRIMES.checked_add(1).expect("catalog count")).is_none());
    let too_wide = Profile {
        prefix: PrimePrefix::new(1).expect("catalog prefix"),
        chunk_bits: NonZeroUsize::new(50).expect("positive width"),
    };
    assert!(forced(2, 2, too_wide).is_none());
    for prime in PRIME_CATALOG {
        let valuation = prime
            .modulus()
            .checked_sub(1)
            .expect("positive prime")
            .trailing_zeros();
        assert!(
            valuation >= MAX_ROOT_ORDER,
            "every prime has the catalog roots"
        );
        for level in 0..valuation.checked_sub(1).expect("large valuation") {
            let log = level.checked_add(2).expect("level order");
            // SAFETY: the level order log is within the catalog valuation.
            let centered =
                unsafe { prime.level_root(usize::try_from(level).expect("small level")) };
            assert!(centered.abs() < prime.float_field().p() / 2.0);
            let canonical = if centered < 0.0 {
                centered + prime.float_field().p()
            } else {
                centered
            };
            // SAFETY: the integer-valued root is below p, inside u64.
            let root = unsafe { canonical.to_int_unchecked::<u64>() };
            assert_eq!(
                prime.pow_mod(root, 1_u64.checked_shl(log).expect("catalog order")),
                1
            );
            assert_eq!(
                prime.pow_mod(
                    root,
                    1_u64
                        .checked_shl(log.checked_sub(1).expect("positive log"))
                        .expect("catalog half-order"),
                ),
                prime.modulus().checked_sub(1).expect("positive prime")
            );
        }
    }
}
