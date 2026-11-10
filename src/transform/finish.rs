//! Final inverse roots with scaling and canonical stores in the same kernel.

#![expect(
    unsafe_code,
    reason = "Admitted inverse geometry proves initialized spans and disjoint canonical residue stores"
)]

use core::num::NonZeroUsize;

use super::{
    COLUMN_ALIGN, COLUMN_GRAIN, Columns, Executor, FinalKernel, FloatField, Normalization,
    Partition, Scalar, Transform,
};

struct FinalJob {
    values: *mut f64,
    twiddles: *const f64,
    stride: usize,
    kernel: FinalKernel,
    suffix_kernel: FinalKernel,
    transition: usize,
    factor: f64,
    field: FloatField,
}

// SAFETY: recursive ranges are disjoint in every strided output span.
// Kernels read all required floating inputs before replacing their columns
// with integer bits. Roots remain live through every synchronous join.
unsafe impl Sync for FinalJob {}

impl<E: Executor> Transform<'_, E> {
    /// Completes the unit inverse root of `roots`, of two, four, or eight spans, and
    /// normalizes its active output prefix.
    pub fn finish_stage<const SPANS: usize>(
        &self,
        values: &mut [f64],
        roots: &[f64],
        normalization: Normalization,
        budget: NonZeroUsize,
    ) {
        const { assert!(matches!(SPANS, 2 | 4 | 8), "unit inverse root arity") };
        let kernels = self.kernels;
        // SAFETY: this inverse has more than a cache block of points, a power of two,
        // so its stride is a positive power of two, and its table covers W1..W3. The
        // output prefix has complete <= SPANS full spans, with a partial span only
        // when complete < SPANS, so kernel selection indexes positive output counts
        // within its table.
        unsafe {
            let stride = NonZeroUsize::new_unchecked(values.len() >> SPANS.trailing_zeros());
            let complete = normalization.columns.end() >> stride.trailing_zeros();
            let transition = normalization.columns.end() & stride.get().unchecked_sub(1);
            let table: &[FinalKernel] = if SPANS == 2 {
                &kernels.finish2
            } else if SPANS == 4 {
                &kernels.finish4
            } else {
                &kernels.finish8
            };
            let kernel = *table.get_unchecked(if transition != 0 {
                complete
            } else {
                complete.unchecked_sub(1)
            });
            let suffix_kernel = if complete == 0 {
                kernel
            } else {
                *table.get_unchecked(complete.unchecked_sub(1))
            };
            let constants = if SPANS == 8 {
                [
                    *roots.get_unchecked(1),
                    *roots.get_unchecked(2),
                    *roots.get_unchecked(3),
                ]
            } else {
                [1.0, 1.0, *roots.get_unchecked(1)]
            };
            let job = FinalJob {
                values: values.as_mut_ptr(),
                twiddles: constants.as_ptr(),
                stride: stride.get(),
                kernel,
                suffix_kernel,
                transition,
                factor: normalization.centered,
                field: self.prime.float_field(),
            };
            let end = if complete == 0 {
                transition
            } else {
                stride.get()
            };
            finish_ranges(&job, 0, end, budget, self.executor);
        }
    }

    /// Recovers the coefficients of a row of `3*span`, `5*span`, or `7*span`
    /// points and normalizes them.
    ///
    /// With `N` the cyclic length (`2*span` or `4*span`), `x^N-1` equals `-2`
    /// modulo every tail factor, so subtracting a scaled tail from the cyclic
    /// residue modulo that factor recovers the high coefficients above `x^N`.
    /// The active columns bound the degree of the result; columns of the last
    /// span at or above `columns - (SPANS-1)*span` are zero and neither read nor
    /// write that span.
    pub fn recombine<const SPANS: usize>(
        &self,
        values: &mut [f64],
        span: usize,
        normalization: Normalization,
        budget: NonZeroUsize,
    ) {
        const { assert!(matches!(SPANS, 3 | 5 | 7), "row span count") };
        let field = self.prime.float_field();
        let (p, pinv) = (field.p(), field.pinv());
        let w1 = self.prime.imaginary();
        let w2 = self.prime.eighth_root();
        // -(p-1)/2 is the centered residue of 1/2.
        let half = (1.0 - p) * 0.5;
        // Each reduced value lies below p, so its reduction is centered below p/2.
        let twice = Scalar::reduce(normalization.centered * 2.0, p, pinv);
        let w3 = Scalar::reduce(Scalar::mulmod(w1, w2, p, pinv), p, pinv);
        let neg_half_i = Scalar::reduce(Scalar::mulmod(-w1, half, p, pinv), p, pinv);
        let constants = if SPANS == 3 {
            [w1, half, 0.0, 0.0, 0.0]
        } else if SPANS == 5 {
            [w1, w2, half, twice, 0.0]
        } else {
            [w1, w3, half, neg_half_i, twice]
        };
        let [kernel, suffix_kernel] = if SPANS == 3 {
            self.kernels.recombine
        } else if SPANS == 5 {
            self.kernels.recombine5
        } else {
            self.kernels.recombine7
        };
        // SAFETY: the row holds SPANS spans of span floats, and active columns
        // end in the last span. The constants outlive the synchronous tasks.
        let transition = unsafe {
            normalization
                .columns
                .end()
                .saturating_sub(span.unchecked_mul(SPANS.unchecked_sub(1)))
        };
        let job = FinalJob {
            values: values.as_mut_ptr(),
            twiddles: constants.as_ptr(),
            stride: span,
            kernel,
            suffix_kernel,
            transition,
            factor: normalization.centered,
            field,
        };
        finish_ranges(&job, 0, span, budget, self.executor);
    }
}

fn finish_ranges<E: Executor>(
    job: &FinalJob,
    first: usize,
    end: usize,
    budget: NonZeroUsize,
    executor: &E,
) {
    if let Some(split) = Partition::range::<COLUMN_ALIGN, COLUMN_GRAIN>(first, end, budget) {
        executor.join(
            || finish_ranges(job, first, split.middle, split.left, executor),
            || finish_ranges(job, split.middle, end, split.right, executor),
        );
        return;
    }
    // SAFETY: first <= end <= stride bounds all kernel columns, and kernel selection
    // fixes each range's positive output count. Every input span is initialized, and
    // selected outputs fit the active prefix. Strides and transitions are multiples of
    // COLUMN_ALIGN, so the guards leave nonempty ranges of complete vectors, and
    // synchronous joins own disjoint ranges.
    unsafe {
        let middle = job.transition.max(first).min(end);
        if first < middle {
            (job.kernel)(
                job.values,
                job.twiddles,
                job.stride,
                Columns::new(first, middle),
                job.factor,
                job.field,
            );
        }
        if middle < end {
            (job.suffix_kernel)(
                job.values,
                job.twiddles,
                job.stride,
                Columns::new(middle, end),
                job.factor,
                job.field,
            );
        }
    }
}
