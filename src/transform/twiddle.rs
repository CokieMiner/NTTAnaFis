//! Bit-reversed roots shared by every level of the polynomial remainder tree.

#![expect(
    unsafe_code,
    reason = "Admitted power-of-two geometry proves disjoint table spans and centered field products"
)]

use core::num::NonZeroUsize;

use super::{COLUMN_ALIGN, Columns, Executor, Kernels, Prime, Scalar};

/// Shortest table per prime whose primes split across workers.
const PARALLEL_MIN_ENTRIES: usize = 2048;

/// Immutable tables for all admitted prime spectra.
///
/// Each prime occupies `stride` entries of both tables. Every root depends
/// only on its index, so a table serves every transform of at most
/// `2*stride` points.
#[derive(Clone, Copy, Debug)]
pub struct Tables<'tables> {
    /// Forward roots in bit-reversed order.
    pub forward: &'tables [f64],
    /// Inverse roots in bit-reversed order.
    pub inverse: &'tables [f64],
    /// Entries per prime, a power of two.
    pub stride: NonZeroUsize,
}

/// Shared roots and the index of one contiguous remainder-tree node.
#[derive(Clone, Copy, Debug)]
pub struct Roots<'tables> {
    /// The table of one prime.
    pub values: &'tables [f64],
    /// The node, whose factor is `x^n - W[node]^2`.
    pub node: usize,
}

/// Prepared power-of-two root tables.
#[derive(Debug)]
pub struct Twiddles;

impl Twiddles {
    /// Initializes independent prime tables across the worker pool.
    ///
    /// # Safety
    /// Both tables hold exactly `stride` entries per prime, and `stride` is a power
    /// of two with `log2(2*stride) <= MAX_ROOT_ORDER`, so every prime has roots of
    /// order `2*stride`.
    pub unsafe fn prepare<E: Executor>(
        forward: &mut [f64],
        inverse: &mut [f64],
        stride: NonZeroUsize,
        primes: &[Prime],
        kernels: &Kernels,
        executor: &E,
    ) {
        prepare_primes(
            (forward, inverse),
            stride,
            primes,
            kernels,
            executor.budget(),
            executor,
        );
    }
}

/// Initializes the tables of `primes` under a partitioned worker budget.
fn prepare_primes<E: Executor>(
    (forward, inverse): (&mut [f64], &mut [f64]),
    stride: NonZeroUsize,
    primes: &[Prime],
    kernels: &Kernels,
    budget: NonZeroUsize,
    executor: &E,
) {
    if budget.get() > 1 && primes.len() > 1 && stride.get() >= PARALLEL_MIN_ENTRIES {
        let count = primes.len() >> 1;
        // SAFETY: prime_count*stride is the admitted table span. count is a
        // positive proper prefix, and both table borrows are exclusively split.
        // budget > 1 proves 0 < left < budget.
        let ((fp, fs), (ip, is), (pp, ps), left, right) = unsafe {
            let left = budget.get() >> 1;
            (
                forward.split_at_mut_unchecked(count.unchecked_mul(stride.get())),
                inverse.split_at_mut_unchecked(count.unchecked_mul(stride.get())),
                primes.split_at_unchecked(count),
                NonZeroUsize::new_unchecked(left),
                NonZeroUsize::new_unchecked(budget.get().unchecked_sub(left)),
            )
        };
        executor.join(
            || prepare_primes((fp, ip), stride, pp, kernels, left, executor),
            || prepare_primes((fs, is), stride, ps, kernels, right, executor),
        );
    } else {
        for ((fwd, inv), prime) in forward
            .chunks_exact_mut(stride.get())
            .zip(inverse.chunks_exact_mut(stride.get()))
            .zip(primes)
        {
            generate_tables(fwd, inv, prime, kernels);
        }
    }
}

/// Stores `W[j] = omega^rev(j)` and its inverse.
///
/// Reversal uses `bits = log2(len)` bits, where `len` is the table length and
/// `omega` has order `2^(bits+1)`. For `j = 2^m + k` with `k < 2^m`,
/// `rev(j) = rev(2^m) + rev(k)`, so `W[j] = W[2^m]*W[k]`, and `W[2^m]` has
/// order `2^(m+2)` for every table length. Each dyadic level multiplies the
/// completed prefix by its level root. Levels shorter than a vector use the
/// portable kernel.
fn generate_tables(forward: &mut [f64], inverse: &mut [f64], prime: &Prime, kernels: &Kernels) {
    let len = forward.len();
    debug_assert!(
        len.is_power_of_two() && len >= 2 && inverse.len() == len,
        "table geometry is admitted"
    );
    // SAFETY: plan admission bounds the transform order by the prime's
    // valuation, so each level below log2(len) has a stored root. Stored
    // entries are centered below p/2. Level m reads indices below 2^m and
    // writes 2^m..2^(m+1) in both tables, which are disjoint spans. Levels of
    // at least COLUMN_ALIGN entries are complete vectors, and shorter levels hold
    // too few entries for one, so they multiply each entry alone. Reversal gives
    // rev(end-1-index) = len-rev(width+index), and omega^len = -1, so the inverse
    // root at end-1-index is -value.
    unsafe {
        *forward.get_unchecked_mut(0) = 1.0;
        *inverse.get_unchecked_mut(0) = 1.0;
        let field = prime.float_field();
        let mut width = 1_usize;
        while width < len {
            let root = prime.level_root(usize::try_from(width.trailing_zeros()).unwrap_unchecked());
            // Both pointers derive from one exclusive borrow of the table.
            let source = forward.as_mut_ptr();
            let target = source.add(width);
            if width < COLUMN_ALIGN {
                for index in 0..width {
                    let product = Scalar::mulmod(*source.add(index), root, field.p(), field.pinv());
                    *target.add(index) = Scalar::reduce(product, field.p(), field.pinv());
                }
            } else {
                (kernels.scale)(target, source, Columns::new(0, width), root, field);
            }
            let end = width.unchecked_mul(2);
            for index in 0..width {
                *inverse.get_unchecked_mut(end.unchecked_sub(1).unchecked_sub(index)) =
                    -*forward.get_unchecked(width.unchecked_add(index));
            }
            width = end;
        }
    }
}
