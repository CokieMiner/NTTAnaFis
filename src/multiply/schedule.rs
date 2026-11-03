//! Synchronous prime partitions under one worker budget.

#![expect(
    unsafe_code,
    reason = "Admitted nonempty prime partitions bound complete spectra, root tables, and positive worker budgets"
)]

use core::{
    num::NonZeroUsize,
    ops::{Div, Range},
};

use super::{Executor, RunContext, Tables};

impl<E: Executor> RunContext<'_, E> {
    /// Evaluates a nonempty partition of the admitted prefix under its worker budget.
    ///
    /// # Safety
    /// `primes` is a nonempty range of catalog indices in the geometry's prefix.
    /// `left`, a multiplication's `right`, and both tables hold one row per index, in
    /// order, as [`Self::process_prime`] requires of each.
    pub unsafe fn run_primes<const SQUARE: bool>(
        &self,
        left: &mut [f64],
        right: &mut [f64],
        tables: Tables<'_>,
        primes: Range<usize>,
        budget: NonZeroUsize,
    ) {
        if budget.get() > 1 && primes.len() > 1 {
            // SAFETY: the caller's rows cover the range, which has two primes, and the
            // budget has two tasks.
            unsafe { self.split_primes::<SQUARE>(left, right, tables, primes, budget) };
            return;
        }
        let length = self.geometry.transform_len();
        for (index, (((prime, lbuf), forward), inverse)) in primes
            .zip(left.chunks_exact_mut(length.get()))
            .zip(tables.forward.chunks_exact(tables.stride.get()))
            .zip(tables.inverse.chunks_exact(tables.stride.get()))
            .enumerate()
        {
            // SAFETY: the right rows correspond to this prime partition, and each index
            // of the range lies in the prefix with its rows.
            unsafe {
                let rbuf = if SQUARE {
                    &mut [][..]
                } else {
                    let offset = index.unchecked_mul(length.get());
                    right.get_unchecked_mut(offset..offset.unchecked_add(length.get()))
                };
                self.process_prime::<SQUARE>(
                    lbuf,
                    rbuf,
                    Tables {
                        forward,
                        inverse,
                        stride: tables.stride,
                    },
                    (prime, budget),
                );
            }
        }
    }

    /// Splits a prime partition into two parts with proportional task budgets.
    ///
    /// `Multiplication` starts with at least two tasks per prime, and this split
    /// preserves that ratio. Every prime therefore starts at once and work stealing
    /// balances workers of different speeds. Budgets only limit splitting and
    /// reserve no thread.
    ///
    /// # Safety
    /// As for [`Self::run_primes`], with at least two primes and a budget of two.
    unsafe fn split_primes<const SQUARE: bool>(
        &self,
        left: &mut [f64],
        right: &mut [f64],
        tables: Tables<'_>,
        primes: Range<usize>,
        budget: NonZeroUsize,
    ) {
        let prefix = primes.len() >> 1;
        // SAFETY: the caller gives at least two primes and two tasks, so
        // 0 < prefix < count and 0 < first_tasks < budget for any budget: a budget
        // below the count gives each part one task and the rest. Right rows split with
        // their primes; squaring needs no right row. Root rows cover the corresponding
        // spectra.
        let ((lp, ls), (rp, rs), (fp, fs), (ip, is), middle, workers, remaining) = unsafe {
            let span = prefix.unchecked_mul(self.geometry.transform_len().get());
            let rows = prefix.unchecked_mul(tables.stride.get());
            let count = NonZeroUsize::new_unchecked(primes.len());
            let first_tasks = budget
                .get()
                .div(count)
                .unchecked_mul(prefix)
                .clamp(1, budget.get().unchecked_sub(1));
            (
                left.split_at_mut_unchecked(span),
                if SQUARE {
                    (&mut [][..], &mut [][..])
                } else {
                    right.split_at_mut_unchecked(span)
                },
                tables.forward.split_at_unchecked(rows),
                tables.inverse.split_at_unchecked(rows),
                primes.start.unchecked_add(prefix),
                NonZeroUsize::new_unchecked(first_tasks),
                NonZeroUsize::new_unchecked(budget.get().unchecked_sub(first_tasks)),
            )
        };
        // SAFETY: each nonempty half of the range keeps the rows of its primes.
        self.executor.join(
            || unsafe {
                self.run_primes::<SQUARE>(
                    lp,
                    rp,
                    Tables {
                        forward: fp,
                        inverse: ip,
                        stride: tables.stride,
                    },
                    primes.start..middle,
                    workers,
                );
            },
            || unsafe {
                self.run_primes::<SQUARE>(
                    ls,
                    rs,
                    Tables {
                        forward: fs,
                        inverse: is,
                        stride: tables.stride,
                    },
                    middle..primes.end,
                    remaining,
                );
            },
        );
    }
}
