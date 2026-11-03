#![expect(
    unsafe_code,
    reason = "Tests provide exact weighted convolutions and native product spans admitted by their geometry"
)]

use core::{
    num::NonZeroUsize,
    sync::atomic::{AtomicUsize, Ordering},
};

use alloc::{vec, vec::Vec};

use crate::{
    Executor, Geometry, LIMB_BITS, Limb, MAX_PRIMES, PRIME_CATALOG, PrimePrefix, arch::Kernels,
    profile::Profile,
};

use super::super::{Cofactors, Crt};

struct Budget {
    workers: NonZeroUsize,
    joins: AtomicUsize,
}

impl Executor for Budget {
    fn budget(&self) -> NonZeroUsize {
        self.workers
    }

    fn join<A, B, RA, RB>(&self, left: A, right: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send,
    {
        let _previous = self.joins.fetch_add(1, Ordering::Relaxed);
        (left(), right())
    }
}

#[test]
fn parallel_blocks_preserve_carries_guards_and_complete_native_output() {
    for count in 1..=MAX_PRIMES {
        if cfg!(miri) && count != MAX_PRIMES {
            continue;
        }
        let widest = count
            .checked_mul(49)
            .and_then(|bits| bits.checked_sub(8))
            .expect("field capacity")
            .div_euclid(2);
        for (width, digits) in [
            (
                1,
                1024_usize
                    .checked_sub(LIMB_BITS)
                    .expect("positive digit count"),
            ),
            (1, 1024),
            (
                1,
                1024_usize
                    .checked_add(LIMB_BITS)
                    .expect("boundary digit count"),
            ),
            (
                1,
                2048_usize
                    .checked_add(LIMB_BITS)
                    .expect("nested digit count"),
            ),
            (widest, 256),
        ]
        .into_iter()
        .chain(
            [63, 64, 65, 127, 128, 129, 191]
                .into_iter()
                .filter(|&width| width < widest && (!cfg!(miri) || width == 63))
                .map(|width| (width, 256)),
        )
        .chain(
            [64, 128, 192]
                .into_iter()
                .filter(|&width| width <= widest)
                .map(|width| (width, 1)),
        ) {
            if cfg!(miri)
                && width == 1
                && digits
                    != 1024_usize
                        .checked_add(LIMB_BITS)
                        .expect("boundary digit count")
            {
                continue;
            }
            let limbs = digits
                .checked_mul(width)
                .expect("native input bits")
                .div_euclid(LIMB_BITS);
            let bits = NonZeroUsize::new(digits.checked_mul(width).expect("native input bits"))
                .expect("positive input bits");
            let geometry = Geometry::under(
                bits,
                bits,
                Profile {
                    prefix: PrimePrefix::new(count).expect("catalog prefix"),
                    chunk_bits: NonZeroUsize::new(width).expect("positive width"),
                },
            )
            .expect("admitted shape");
            let storage = weighted_triangle(digits, count, width);
            let mut residues = [&[][..]; MAX_PRIMES];
            for (row, values) in residues.iter_mut().zip(&storage) {
                *row = values;
            }
            let output_len = limbs.checked_mul(2).expect("native product span");
            let mut expected = vec![0; output_len];
            *expected.first_mut().expect("positive output") = 1;
            expected
                .get_mut(limbs..)
                .expect("upper product half")
                .fill(Limb::MAX);
            *expected.get_mut(limbs).expect("middle limb") =
                Limb::MAX.checked_sub(1).expect("native radix");
            check_budgets(&residues, &geometry, &expected);
        }
    }
}

fn weighted_triangle(digits: usize, count: usize, width: usize) -> Vec<Vec<f64>> {
    let conv_len = digits
        .checked_mul(2)
        .and_then(|length| length.checked_sub(1))
        .expect("convolution span");
    let mut storage = vec![vec![0.0; conv_len]; count];
    let table = Cofactors::of_count(count);
    for (row, prime) in PRIME_CATALOG.iter().take(count).enumerate() {
        let inverse = u128::from(*table.inverses.get(row).expect("active inverse"));
        let modulus = u128::from(prime.modulus());
        let mut digit = 0_u64;
        for _ in 0..width {
            digit = ((digit << 1) | 1).rem_euclid(prime.modulus());
        }
        let square = u128::from(digit)
            .checked_mul(u128::from(digit))
            .expect("100-bit digit square")
            .rem_euclid(modulus);
        let factor = square
            .checked_mul(inverse)
            .expect("100-bit weighted square")
            .rem_euclid(modulus);
        for (index, value) in storage
            .get_mut(row)
            .expect("prime row")
            .iter_mut()
            .enumerate()
        {
            let coefficient = if index < digits {
                index.checked_add(1).expect("ascending coefficient")
            } else {
                conv_len.checked_sub(index).expect("descending coefficient")
            };
            let weighted = u128::try_from(coefficient)
                .expect("native coefficient")
                .checked_mul(factor)
                .expect("bounded coefficient product")
                .rem_euclid(modulus);
            *value = f64::from_bits(u64::try_from(weighted).expect("canonical weighted residue"));
        }
    }
    storage
}

fn check_budgets(residues: &[&[f64]; MAX_PRIMES], geometry: &Geometry, expected: &[Limb]) {
    let output_len = expected.len();
    for workers in [1, 3, 5, 7, 8] {
        if cfg!(miri) && workers != 3 {
            continue;
        }
        let executor = Budget {
            workers: NonZeroUsize::new(workers).expect("positive budget"),
            joins: AtomicUsize::new(0),
        };
        let mut guarded = vec![123; output_len.checked_add(2).expect("guarded output")];
        // SAFETY: the driver supplies the exact convolution of radix-2^width
        // digits, each equal to 2^width-1. Their canonical residues include
        // the table's CRT weights. The admitted geometry proves the field
        // bound and the exact native product span used for expected.
        unsafe {
            Crt::reconstruct(
                guarded
                    .get_mut(1..output_len.checked_add(1).expect("suffix guard"))
                    .expect("output span"),
                residues,
                geometry,
                Kernels::get(),
                &executor,
            );
        }
        assert_eq!(guarded.first(), Some(&123));
        assert_eq!(guarded.last(), Some(&123));
        assert_eq!(
            guarded
                .get(1..output_len.checked_add(1).expect("suffix guard"))
                .expect("output span"),
            expected
        );
        assert_eq!(
            executor.joins.load(Ordering::Relaxed) > 0,
            workers > 1 && geometry.conv_len() >= 512
        );
    }
}
