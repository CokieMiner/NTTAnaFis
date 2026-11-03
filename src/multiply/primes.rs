//! Forward spectra, inverse products, and cofactor-weighted residue normalization.

#![expect(
    unsafe_code,
    reason = "Admitted geometry and exclusive prime rows prove transform spans and canonical normalization"
)]

use core::num::NonZeroUsize;

use super::{
    Cofactors, Executor, Normalization, PARALLEL_MIN_LEN, PRIME_CATALOG, RunContext, Tables,
    Transform,
};

/// Minimum transform length for inverse-leaf product fusion.
pub const PRODUCT_FUSION_MIN_LEN: usize = 2_usize.saturating_pow(20);

impl<E: Executor> RunContext<'_, E> {
    /// Evaluates and normalizes one prime with its packed spectra.
    ///
    /// # Safety
    /// `prime_index` is a catalog index in the geometry's prefix. `left` is a row of
    /// the geometry packed for this prime. A multiplication reads `right`, a
    /// disjoint row packed the same way, or its spectrum once `right_ready`; squaring
    /// ignores it. `tables` holds the prime's roots for the geometry's table.
    pub unsafe fn process_prime<const SQUARE: bool>(
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
        // A retained right spectrum is only read from here on, by the products.
        let forward_right = !SQUARE && !self.right_ready;
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
        // SAFETY: the forward stages left spectra below 3*p/2 in the caller's rows.
        unsafe {
            self.inverse_prime::<SQUARE>(left, right, tables, &transform, (prime_index, budget));
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
