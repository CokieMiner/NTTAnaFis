//! Synchronous prime partitions under one worker budget.

#![expect(
    unsafe_code,
    reason = "Admitted nonempty prime partitions bound complete spectra, root tables, and positive worker budgets"
)]

use core::{
    num::NonZeroUsize,
    ops::{Div, Range},
};

use super::{
    Cofactors, Executor, Normalization, PARALLEL_MIN_LEN, PRIME_CATALOG, RunContext, Tables,
    Transform,
};

/// Minimum transform length for inverse-leaf product fusion.
pub const PRODUCT_FUSION_MIN_LEN: usize = 2_usize.saturating_pow(20);

impl<E: Executor> RunContext<'_, E> {
    /// Evaluates a nonempty partition of the admitted prefix under its worker budget.
    ///
    /// # Safety
    /// `primes` is a nonempty range of catalog indices in the geometry's prefix.
    /// `left`, both tables, and the `right` of a multiplication without shared right
    /// spectra hold one row per index, in order, as [`Self::process_prime`] requires
    /// of each.
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
        let length = self.geometry.transform_len().get();
        let stride = tables.stride.get();
        for (index, prime) in primes.enumerate() {
            // SAFETY: the caller gives one row of each slice per prime of the range,
            // in order, so row `index` of each lies inside it, and each index of the
            // range lies in the prefix.
            unsafe {
                let row = index.unchecked_mul(length);
                let roots = index.unchecked_mul(stride);
                let rbuf = if SQUARE || self.right.is_some() {
                    &mut [][..]
                } else {
                    right.get_unchecked_mut(row..row.unchecked_add(length))
                };
                self.process_prime::<SQUARE>(
                    left.get_unchecked_mut(row..row.unchecked_add(length)),
                    rbuf,
                    Tables {
                        forward: tables
                            .forward
                            .get_unchecked(roots..roots.unchecked_add(stride)),
                        inverse: tables
                            .inverse
                            .get_unchecked(roots..roots.unchecked_add(stride)),
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
        // their primes; squaring and shared right spectra need none. Root rows cover
        // the corresponding spectra.
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
                if SQUARE || self.right.is_some() {
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

    /// Evaluates and normalizes one prime with its packed spectra.
    ///
    /// # Safety
    /// `prime_index` is a catalog index in the geometry's prefix. `left` is a row of
    /// the geometry packed for this prime. A multiplication reads `right`, a
    /// disjoint row packed the same way, unless the context holds the right
    /// spectra, whose row of this prime it reads instead; squaring ignores both.
    /// `tables` holds the prime's roots for the geometry's table.
    unsafe fn process_prime<const SQUARE: bool>(
        &self,
        left: &mut [f64],
        right: &mut [f64],
        tables: Tables<'_>,
        (prime_index, budget): (usize, NonZeroUsize),
    ) {
        let geometry = self.geometry;
        let cyclic_len = geometry.cyclic_len();
        // SAFETY: the index lies in the admitted prefix of the catalog.
        let prime = unsafe { PRIME_CATALOG.get_unchecked(prime_index) };
        let transform = Transform::new(prime, self.kernels, self.executor, self.early_pairs);
        let forward = |row: &mut [f64], input_len, workers| {
            // SAFETY: the caller gives admitted rows and tables, and an operand's
            // columns fit the cyclic part.
            unsafe {
                transform.forward_row(row, tables.forward, (cyclic_len, input_len), workers);
            }
        };
        let (left_inputs, right_inputs) = (geometry.left().columns, geometry.right().columns);
        // Shared right spectra are only read, by the products.
        let forward_right = !SQUARE && self.right.is_none();
        if forward_right && budget.get() > 1 && geometry.transform_len().get() >= PARALLEL_MIN_LEN {
            // SAFETY: budget > 1 proves 0 < workers < budget.
            let (workers, remaining) = unsafe {
                let workers = budget.get() >> 1;
                (
                    NonZeroUsize::new_unchecked(workers),
                    NonZeroUsize::new_unchecked(budget.get().unchecked_sub(workers)),
                )
            };
            self.executor.join(
                || forward(left, left_inputs, workers),
                || forward(right, right_inputs, remaining),
            );
        } else {
            forward(left, left_inputs, budget);
            if forward_right {
                forward(right, right_inputs, NonZeroUsize::MIN);
            }
        }
        // SAFETY: the forward stages left spectra below 3*p/2 in the caller's rows,
        // and shared spectra hold one row per prime of the prefix, in order.
        unsafe {
            let spectrum = match self.right {
                Some(spectra) => {
                    let length = geometry.transform_len().get();
                    let offset = prime_index.unchecked_mul(length);
                    spectra.get_unchecked(offset..offset.unchecked_add(length))
                }
                None => right,
            };
            self.inverse_prime::<SQUARE>(left, spectrum, tables, &transform, (prime_index, budget));
        }
    }

    /// Multiplies the spectra of one prime, inverts them, and normalizes the result.
    ///
    /// # Safety
    /// As for [`Self::process_prime`], with both spectra formed.
    #[expect(
        clippy::as_conversions,
        clippy::cast_precision_loss,
        reason = "The inverse-normalization factor is canonical below a 50-bit prime and converts exactly to f64"
    )]
    unsafe fn inverse_prime<const SQUARE: bool>(
        &self,
        left: &mut [f64],
        right: &[f64],
        tables: Tables<'_>,
        transform: &Transform<'_, E>,
        (prime_index, budget): (usize, NonZeroUsize),
    ) {
        let geometry = self.geometry;
        let prime = transform.prime;
        let cyclic_len = geometry.cyclic_len();
        // SAFETY: the admitted index selects a prime of this prefix. N divides p-1, so
        // p-((p-1)/N) is N^-1 mod p. Multiplying by the cofactor inverse combines both
        // normalizations into one exact factor, canonical below the 50-bit prime.
        let factor = unsafe {
            let inverse_scale = prime.modulus().unchecked_sub(
                prime
                    .modulus()
                    .unchecked_sub(1)
                    .unchecked_shr(cyclic_len.trailing_zeros()),
            );
            prime.mul_mod(
                inverse_scale,
                *Cofactors::of(geometry.prefix())
                    .inverses
                    .get_unchecked(prime_index),
            ) as f64
        };
        let normalization = Normalization::new(geometry.output_columns(), factor, prime);
        let row_roots = (tables.inverse, cyclic_len);
        // SAFETY: the spectra are complete rows below 3*p/2, and the tables cover
        // them. Multiplication reads a disjoint right row; squaring ignores it. The
        // output columns fit the row.
        unsafe {
            if geometry.transform_len().get() >= PRODUCT_FUSION_MIN_LEN {
                transform.finish_row::<true, SQUARE>(
                    (left, right),
                    row_roots,
                    normalization,
                    budget,
                );
            } else {
                let product = if SQUARE {
                    self.kernels.square
                } else {
                    self.kernels.multiply
                };
                product(
                    left.as_mut_ptr(),
                    right.as_ptr(),
                    geometry.row_columns(),
                    prime.float_field(),
                );
                transform.finish_row::<false, false>((left, &[]), row_roots, normalization, budget);
            }
        }
    }
}
