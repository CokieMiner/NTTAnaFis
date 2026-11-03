//! Final inverse roots with scaling and canonical stores in the same kernel.

#![expect(
    unsafe_code,
    reason = "Admitted inverse geometry proves initialized spans and disjoint canonical residue stores"
)]

use core::{
    num::NonZeroUsize,
    ops::{Div, Rem},
};

use super::{
    COLUMN_ALIGN, COLUMN_GRAIN, Columns, Executor, FinalKernel, FloatField, Normalization,
    Partition, Transform,
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
        // SAFETY: this inverse has more than a cache block of points, and its table
        // covers W1..W3. The output prefix has complete <= SPANS full spans, with a
        // partial span only when complete < SPANS, so kernel selection indexes
        // positive output counts within its table.
        unsafe {
            let stride = NonZeroUsize::new_unchecked(values.len() >> SPANS.trailing_zeros());
            let complete = normalization.columns.end().div(stride);
            let transition = normalization.columns.end().rem(stride);
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

    /// Recovers the coefficients of a row of `3*span` points and normalizes them.
    ///
    /// The spans hold `N*(a0, a1)` for `a0 + x^M*a1 = C mod (x^N-1)` and
    /// `M*(C mod (x^M-i))`, below `2*p` in magnitude, with `N = 2*M`. Since
    /// `x^N-1` equals `-2` modulo `x^M-i`, the polynomial
    /// `D = ((a0 + i*a1) - C mod (x^M-i))/2` gives
    /// `C = (a0 - D) + x^M*a1 + x^N*D`. The factor is `c/N` for the CRT inverse `c`.
    /// The active columns bound the degree of `C`, so columns of `D` at or above
    /// `columns - N` are zero. They read no third span and write no output above the
    /// active columns.
    pub fn recombine(
        &self,
        values: &mut [f64],
        span: usize,
        normalization: Normalization,
        budget: NonZeroUsize,
    ) {
        // -(p-1)/2 is the centered residue of 1/2.
        let constants = [
            self.prime.imaginary(),
            (1.0 - self.prime.float_field().p()) * 0.5,
        ];
        // SAFETY: the row holds three spans of span floats, and columns stay below
        // span. The constants outlive the synchronous range tasks.
        let transition = unsafe {
            normalization
                .columns
                .end()
                .saturating_sub(span.unchecked_mul(2))
        };
        let job = FinalJob {
            values: values.as_mut_ptr(),
            twiddles: constants.as_ptr(),
            stride: span,
            kernel: self.kernels.recombine,
            suffix_kernel: self.kernels.recombine_suffix,
            transition,
            factor: normalization.centered,
            field: self.prime.float_field(),
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
