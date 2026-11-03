//! Task partitions under an explicit worker budget.
//!
//! A budget is a positive task count, a [`NonZeroUsize`]. It only limits how far
//! work splits and reserves no thread, so every value is valid, up to the largest an
//! executor can report.

#![expect(
    unsafe_code,
    reason = "Ordered ranges and bounded quotient-remainder partitions prove exact arithmetic"
)]

use core::{
    num::NonZeroUsize,
    ops::{Div, Rem},
};

/// One interior cut and two worker budgets.
#[derive(Debug)]
pub struct Partition {
    /// First element of the right partition.
    pub middle: usize,
    /// Worker budget assigned to the left partition.
    pub left: NonZeroUsize,
    /// Worker budget assigned to the right partition.
    pub right: NonZeroUsize,
}

impl Partition {
    /// Partitions two, four, eight, or sixteen children when two workers exist, or
    /// returns `None`.
    ///
    /// The cut minimizes the larger child-count/worker ratio. For odd B and
    /// k=arity/2, the two adjacent candidates are k-ceil(k/B) and its successor.
    /// When B exceeds the arity, equal child counts minimize that ratio. The cut is
    /// interior for every arity of two to sixteen.
    #[must_use]
    pub fn children(arity: usize, budget: NonZeroUsize) -> Option<Self> {
        if budget.get() < 2 {
            return None;
        }
        // SAFETY: the caller's arity is even in 2..=16, and B>=2 was tested above.
        // For odd 3<=B<arity and k>=2, 1<=q=ceil(k/B)<k. Both adjacent cuts are
        // interior. The floor cut is optimal exactly when (2*q-1)*B <= arity+1;
        // these products are below 2*arity <= 32. Other modes divide children in
        // half. B>=2 gives both halves a worker.
        unsafe {
            let half = arity >> 1;
            let middle = if budget.get() & 1 != 0 && budget.get() < arity {
                let quotient = half
                    .div(budget)
                    .unchecked_add(usize::from(half.rem(budget) != 0));
                let floor = half.unchecked_sub(quotient);
                let left_cost = quotient
                    .unchecked_mul(2)
                    .unchecked_sub(1)
                    .unchecked_mul(budget.get());
                floor.unchecked_add(usize::from(left_cost > arity.unchecked_add(1)))
            } else {
                half
            };
            let left = budget.get() >> 1;
            Some(Self {
                middle,
                left: NonZeroUsize::new_unchecked(left),
                right: NonZeroUsize::new_unchecked(budget.get().unchecked_sub(left)),
            })
        }
    }

    /// Splits `first..end` when two tasks of `GRAIN` elements fit and two workers
    /// exist, or returns `None`.
    ///
    /// A range with `first` a multiple of `UNIT` cuts at a multiple of `UNIT`. Each
    /// child keeps at least `GRAIN` elements, and a reversed range has none. Odd
    /// budgets divide the whole units in proportion to the tasks that fit each worker
    /// group.
    #[must_use]
    pub fn range<const UNIT: usize, const GRAIN: usize>(
        first: usize,
        end: usize,
        budget: NonZeroUsize,
    ) -> Option<Self> {
        const {
            assert!(
                UNIT.is_power_of_two() && GRAIN >= UNIT && GRAIN.is_multiple_of(UNIT),
                "a task holds whole units"
            );
        };
        // SAFETY: a split has length >= 2*GRAIN, so first < end, and budget >= 2,
        // so at least two tasks fit. Quotient and remainder assign a
        // strict, positive prefix of both work and workers, whose sums stay within
        // length and budget. A cut at a whole unit leaves fewer than UNIT residual
        // elements in the right partition.
        unsafe {
            let length = end.saturating_sub(first);
            if budget.get() < 2 || length < GRAIN << 1 {
                return None;
            }
            let unit_bits = UNIT.trailing_zeros();
            let (prefix, workers) = if budget.get() & 1 == 0 {
                ((length >> 1) & !UNIT.unchecked_sub(1), budget.get() >> 1)
            } else {
                let grain = NonZeroUsize::new_unchecked(GRAIN);
                let tasks = NonZeroUsize::new_unchecked(budget.get().min(length.div(grain)));
                let first_tasks = tasks.get() >> 1;
                let units = length >> unit_bits;
                let prefix = units
                    .div(tasks)
                    .unchecked_mul(first_tasks)
                    .unchecked_add(units.rem(tasks).min(first_tasks));
                let workers = budget
                    .get()
                    .div(tasks)
                    .unchecked_mul(first_tasks)
                    .unchecked_add(budget.get().rem(tasks).min(first_tasks));
                (prefix << unit_bits, workers)
            };
            Some(Self {
                middle: first.unchecked_add(prefix),
                left: NonZeroUsize::new_unchecked(workers),
                right: NonZeroUsize::new_unchecked(budget.get().unchecked_sub(workers)),
            })
        }
    }
}
