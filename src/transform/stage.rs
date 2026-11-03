//! Disjoint butterfly ranges scheduled under one explicit worker budget.

#![expect(
    unsafe_code,
    reason = "Validated transform geometry proves pointer spans and disjoint synchronous task ranges"
)]

use core::{
    num::NonZeroUsize,
    ops::{Div, Rem},
};

use super::{
    COLUMN_ALIGN, Columns, Executor, FloatField, Partition, Roots, StageKernel, Transform,
};

/// Columns of the smallest parallel stage task.
pub const COLUMN_GRAIN: usize = 1024;

struct RangeJob {
    values: *mut f64,
    twiddles: *const f64,
    stride: usize,
    kernel: StageKernel,
    suffix_kernel: StageKernel,
    transition: usize,
    field: FloatField,
}

// SAFETY: RangeJob is private and borrowed only by run_ranges. Its recursive
// splits give each task disjoint indices in every output span. Twiddles are
// immutable, and synchronous joins finish before the stage releases either slice.
unsafe impl Sync for RangeJob {}

impl<E: Executor> Transform<'_, E> {
    /// Runs a complete stage of `SPANS` spans with the roots of its node.
    pub fn stage<const SPANS: usize>(
        &self,
        values: &mut [f64],
        roots: Roots<'_>,
        kernel: StageKernel,
        budget: NonZeroUsize,
    ) {
        const {
            assert!(
                matches!(SPANS, 2 | 4 | 16),
                "stage spans must be 2, 4, or 16"
            );
        }
        let stride = values.len() >> SPANS.trailing_zeros();
        let output = values.as_mut_ptr();
        let run = |twiddles| {
            let job = RangeJob {
                values: output,
                twiddles,
                stride,
                kernel,
                suffix_kernel: kernel,
                transition: stride,
                field: self.prime.float_field(),
            };
            run_ranges::<false, E>(&job, 0, stride, budget, self.executor);
        };
        // SAFETY: the table covers this node and its descendants, and the local
        // arrays remain live through every synchronous range task.
        unsafe {
            let j = roots.node;
            let root = |index| *roots.values.get_unchecked(index);
            if SPANS == 2 {
                let constants = [root(j)];
                run(constants.as_ptr());
            } else {
                let twice = j.unchecked_mul(2);
                if SPANS == 4 {
                    let constants = [root(j), root(twice), root(twice.unchecked_add(1))];
                    run(constants.as_ptr());
                } else {
                    let four = j.unchecked_mul(4);
                    let eight = j.unchecked_mul(8);
                    let constants = [
                        root(j),
                        root(twice),
                        root(twice.unchecked_add(1)),
                        root(four),
                        root(four.unchecked_add(1)),
                        root(four.unchecked_add(2)),
                        root(four.unchecked_add(3)),
                        root(eight),
                        root(eight.unchecked_add(2)),
                        root(eight.unchecked_add(4)),
                        root(eight.unchecked_add(6)),
                        root(eight.unchecked_add(1)),
                        root(eight.unchecked_add(3)),
                        root(eight.unchecked_add(5)),
                        root(eight.unchecked_add(7)),
                    ];
                    run(constants.as_ptr());
                }
            }
        }
    }

    /// Evaluates the unit root of `roots` without reading outside its input prefix.
    ///
    /// Two-span and four-span roots admit `input_len < values.len()`; eight-span roots
    /// admit at most four input spans.
    pub fn prefix<const SPANS: usize>(
        &self,
        values: &mut [f64],
        roots: &[f64],
        input_len: usize,
        budget: NonZeroUsize,
    ) {
        const { assert!(matches!(SPANS, 2 | 4 | 8), "prefix root arity") };
        let kernels = self.kernels;
        // SAFETY: the root has at least SPANS points. Its prefix contains at most
        // SPANS-1 complete spans, or four for SPANS=8, and a nonempty partial span
        // adds exactly one active input. All root constants lie in the table.
        unsafe {
            let stride = NonZeroUsize::new_unchecked(values.len() >> SPANS.trailing_zeros());
            let complete = input_len.div(stride);
            let transition = input_len.rem(stride);
            let select = |inputs: usize| -> StageKernel {
                if inputs == 0 {
                    zero_spans::<SPANS>
                } else if SPANS == 2 {
                    if inputs == 1 {
                        kernels.duplicate_half
                    } else {
                        kernels.radix2
                    }
                } else if SPANS == 4 {
                    if inputs == 4 {
                        kernels.dif_unit
                    } else {
                        *kernels.dif4_prefix.get_unchecked(inputs.unchecked_sub(1))
                    }
                } else {
                    *kernels.dif8_prefix.get_unchecked(inputs.unchecked_sub(1))
                }
            };
            let suffix_kernel = select(complete);
            let kernel = if transition == 0 {
                suffix_kernel
            } else {
                select(complete.unchecked_add(1))
            };
            let constants = if SPANS == 8 {
                [
                    *roots.get_unchecked(1),
                    *roots.get_unchecked(2),
                    *roots.get_unchecked(3),
                ]
            } else if SPANS == 4 {
                [1.0, 1.0, *roots.get_unchecked(1)]
            } else {
                [0.0; 3]
            };
            let job = RangeJob {
                values: values.as_mut_ptr(),
                twiddles: constants.as_ptr(),
                stride: stride.get(),
                kernel,
                suffix_kernel,
                transition,
                field: self.prime.float_field(),
            };
            run_ranges::<true, E>(&job, 0, stride.get(), budget, self.executor);
        }
    }

    /// Writes the residue of a packed row modulo `x^M-i` to its third span.
    ///
    /// The row holds three spans of `span` floats, and the input prefix fits two of
    /// them. With the input `a0 + x^M*a1`, the residue is `a0 + i*a1`. The input stays
    /// in place for the transform modulo `x^(2*M)-1`.
    pub fn fold(&self, values: &mut [f64], span: usize, input_len: usize, budget: NonZeroUsize) {
        let overlap = input_len.saturating_sub(span);
        let active = input_len.min(span);
        // SAFETY: the row holds 3*span floats and overlap <= active <= span, so
        // the two spans split at 2*span and every copied range fits both sides. The
        // local root outlives the synchronous range tasks, and the kernel reads the
        // initialized input columns below overlap and writes their third span.
        // Endpoints are multiples of COLUMN_ALIGN.
        unsafe {
            let (inputs, target) = values.split_at_mut_unchecked(span.unchecked_mul(2));
            target
                .get_unchecked_mut(overlap..active)
                .copy_from_slice(inputs.get_unchecked(overlap..active));
            target.get_unchecked_mut(active..).fill(0.0);
            if overlap > 0 {
                let root = [self.prime.imaginary()];
                let job = RangeJob {
                    values: values.as_mut_ptr(),
                    twiddles: root.as_ptr(),
                    stride: span,
                    kernel: self.kernels.fold,
                    suffix_kernel: self.kernels.fold,
                    transition: span,
                    field: self.prime.float_field(),
                };
                run_ranges::<false, E>(&job, 0, overlap, budget, self.executor);
            }
        }
    }
}

fn run_ranges<const PREFIX: bool, E: Executor>(
    job: &RangeJob,
    start: usize,
    end: usize,
    budget: NonZeroUsize,
    executor: &E,
) {
    if let Some(split) = Partition::range::<COLUMN_ALIGN, COLUMN_GRAIN>(start, end, budget) {
        executor.join(
            || run_ranges::<PREFIX, E>(job, start, split.middle, split.left, executor),
            || run_ranges::<PREFIX, E>(job, split.middle, end, split.right, executor),
        );
    } else {
        // SAFETY: the stage admits its spans with the corresponding roots, and the
        // selected kernel reads only its initialized input spans. Strides and prefix
        // transitions are multiples of COLUMN_ALIGN, and partitions preserve that, so
        // the guards leave nonempty ranges of complete vectors. Tasks access disjoint
        // ranges.
        unsafe {
            if PREFIX {
                let middle = job.transition.max(start).min(end);
                if start < middle {
                    (job.kernel)(
                        job.values,
                        job.twiddles,
                        job.stride,
                        Columns::new(start, middle),
                        job.field,
                    );
                }
                if middle < end {
                    (job.suffix_kernel)(
                        job.values,
                        job.twiddles,
                        job.stride,
                        Columns::new(middle, end),
                        job.field,
                    );
                }
            } else {
                (job.kernel)(
                    job.values,
                    job.twiddles,
                    job.stride,
                    Columns::new(start, end),
                    job.field,
                );
            }
        }
    }
}

unsafe fn zero_spans<const SPANS: usize>(
    values: *mut f64,
    _twiddles: *const f64,
    stride: usize,
    columns: Columns,
    _field: FloatField,
) {
    // SAFETY: the columns fit each disjoint span. This mode has no active
    // inputs, so all butterfly outputs are exactly zero. All-zero bytes
    // represent 0.0; every byte count and offset fits the admitted slice.
    unsafe {
        let (start, end) = columns.complete::<1>();
        for span in 0..SPANS {
            values
                .add(span.unchecked_mul(stride).unchecked_add(start))
                .write_bytes(0, end.unchecked_sub(start));
        }
    }
}
