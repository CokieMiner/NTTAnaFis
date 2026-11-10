//! Forward polynomial remainder stages and contiguous cache-resident descendants.

#![expect(
    unsafe_code,
    reason = "Admitted power-of-two geometry proves stage spans and worker partitions"
)]

use core::{num::NonZeroUsize, ptr::null};

use super::{Executor, Kernels, LEAF_POINTS, MIN_TRANSFORM_LEN, Partition, Prime, Roots};

/// Table node of the factor `x^M-i`. Node `j` of a transform of `M` points is the
/// factor `x^M - W[j]^2`, and `W[2]^2 = W[1] = i`.
pub const TAIL_NODE: usize = 2;

/// Table node of the factor `x^M-w`, where `w = W[2]` and `W[4]^2 = W[2]`.
pub const FIFTH_NODE: usize = 4;

/// Table node of the factor `x^M-i*w`, where `W[6]^2 = W[3] = i*w`.
pub const SEVENTH_NODE: usize = 6;

/// Maximum length evaluated as one contiguous cache block.
pub const CACHE_LEAF: usize = 256;

/// Minimum transform length for splitting work across a worker budget.
pub const PARALLEL_MIN_LEN: usize = 2048;

/// Minimum global radix-8 root size retaining a full cache block per child.
pub const ROOT_MIN_LEN: usize = CACHE_LEAF << 3;

/// Minimum transform length for amortizing paired-stage address setup.
const FUSED_MIN_LEN: usize = 2_usize.saturating_pow(24);

/// The same with early pairs, when workers share a large product.
const EARLY_FUSED_MIN_LEN: usize = 2_usize.saturating_pow(16);

const _: () = {
    assert!(
        CACHE_LEAF.is_power_of_two()
            && CACHE_LEAF >= MIN_TRANSFORM_LEN
            && (CACHE_LEAF.trailing_zeros() - LEAF_POINTS.trailing_zeros()).is_multiple_of(2),
        "radix-4 stages of a cache block end at its leaves"
    );
    assert!(
        EARLY_FUSED_MIN_LEN >= CACHE_LEAF << 4 && FUSED_MIN_LEN >= EARLY_FUSED_MIN_LEN,
        "paired stages leave children of at least a cache block"
    );
    assert!(
        PARALLEL_MIN_LEN > CACHE_LEAF,
        "a split transform runs its children recursively"
    );
};

/// Field, kernels, and worker pool of the transforms of one prime.
#[derive(Debug)]
pub struct Transform<'run, E> {
    /// Field of every residue.
    pub prime: &'run Prime,
    /// Kernels of the resolved vector unit.
    pub kernels: &'run Kernels,
    /// Worker pool of the complete operation.
    pub executor: &'run E,
    /// Shortest non-root transform that runs paired stages.
    pub fused_min_len: usize,
}

impl<'run, E: Executor> Transform<'run, E> {
    /// Returns the transforms of one prime; `early_pairs` lowers the paired-stage
    /// crossover for workers that share a large product.
    #[must_use]
    pub const fn new(
        prime: &'run Prime,
        kernels: &'run Kernels,
        executor: &'run E,
        early_pairs: bool,
    ) -> Self {
        Self {
            prime,
            kernels,
            executor,
            fused_min_len: if early_pairs {
                EARLY_FUSED_MIN_LEN
            } else {
                FUSED_MIN_LEN
            },
        }
    }

    /// Evaluates one row with a known input prefix.
    ///
    /// # Safety
    /// `row` is a row of the module terms whose cyclic part has `cyclic_len` points,
    /// packed below `p`, and `roots` covers its nodes. `input_len` is an input prefix
    /// that fits the cyclic part.
    pub unsafe fn forward_row(
        &self,
        row: &mut [f64],
        roots: &[f64],
        (cyclic_len, input_len): (usize, usize),
        budget: NonZeroUsize,
    ) {
        let cyclic_roots = Roots {
            values: roots,
            node: 0,
        };
        let row_len = row.len();
        let tails = row_len.saturating_sub(cyclic_len);
        if tails == 0 {
            self.forward(row, cyclic_roots, input_len, budget);
            return;
        }
        let half = cyclic_len >> 1;
        let quarter = cyclic_len >> 2;
        if tails == half {
            self.fold(row, half, input_len, budget);
        } else if tails == quarter {
            self.fold4::<false>(row, quarter, input_len, budget);
        } else {
            self.fold(row, half, input_len, budget);
            self.fold4::<true>(row, quarter, input_len, budget);
        }
        let parallel = budget.get() > 1 && row.len() >= PARALLEL_MIN_LEN;
        // SAFETY: the row holds cyclic_len + tails floats, where tails is M,
        // 2*M, or 3*M with N = 4*M = cyclic_len. When parallel, budget >= 2
        // gives both partitions a positive worker count.
        unsafe {
            let (cyclic, tail) = row.split_at_mut_unchecked(cyclic_len);
            let run_tails = |span: &mut [f64], workers: NonZeroUsize| {
                if tails == half {
                    self.forward(
                        span,
                        Roots {
                            values: roots,
                            node: TAIL_NODE,
                        },
                        half,
                        workers,
                    );
                } else if tails == quarter {
                    self.forward(
                        span,
                        Roots {
                            values: roots,
                            node: FIFTH_NODE,
                        },
                        quarter,
                        workers,
                    );
                } else {
                    let (first, second) = span.split_at_mut_unchecked(half);
                    let first_roots = Roots {
                        values: roots,
                        node: TAIL_NODE,
                    };
                    let second_roots = Roots {
                        values: roots,
                        node: SEVENTH_NODE,
                    };
                    if workers.get() > 1 && tails >= PARALLEL_MIN_LEN {
                        let tail_workers =
                            NonZeroUsize::new_unchecked(workers.get().div_euclid(3).max(1));
                        let first_workers = NonZeroUsize::new_unchecked(
                            workers.get().unchecked_sub(tail_workers.get()),
                        );
                        self.executor.join(
                            || self.forward(first, first_roots, half, first_workers),
                            || self.forward(second, second_roots, quarter, tail_workers),
                        );
                    } else {
                        self.forward(first, first_roots, half, NonZeroUsize::MIN);
                        self.forward(second, second_roots, quarter, NonZeroUsize::MIN);
                    }
                }
            };
            if parallel {
                // The tails hold a third, a fifth, or three sevenths of the row.
                let share = if tails == half {
                    budget.get().div_euclid(3)
                } else if tails == quarter {
                    budget.get().div_euclid(5)
                } else {
                    budget.get().saturating_mul(3).div_euclid(7)
                };
                let workers =
                    NonZeroUsize::new_unchecked(share.clamp(1, budget.get().unchecked_sub(1)));
                let remaining =
                    NonZeroUsize::new_unchecked(budget.get().unchecked_sub(workers.get()));
                self.executor.join(
                    || self.forward(cyclic, cyclic_roots, input_len, remaining),
                    || run_tails(tail, workers),
                );
            } else {
                self.forward(cyclic, cyclic_roots, input_len, NonZeroUsize::MIN);
                run_tails(tail, NonZeroUsize::MIN);
            }
        }
    }

    /// Evaluates a transform with a known input prefix.
    pub fn forward(
        &self,
        data: &mut [f64],
        mut roots: Roots<'_>,
        input_len: usize,
        budget: NonZeroUsize,
    ) {
        let kernels = self.kernels;
        let len = data.len();
        debug_assert!(
            len.is_power_of_two() && len >= MIN_TRANSFORM_LEN,
            "a transform holds complete vectors"
        );
        debug_assert!(
            input_len == len || roots.node == 0,
            "a shorter input prefix requires the unit root"
        );
        let odd = len.trailing_zeros() & 1 != 0;
        let paired_root = odd && roots.node == 0 && len >= ROOT_MIN_LEN && input_len <= len >> 1;
        let wide = kernels.dif_wide.filter(|_| {
            !odd && roots.node != 0 && len >= self.fused_min_len && input_len > len >> 1
        });
        let arity = if wide.is_some() {
            16_usize
        } else if paired_root {
            8
        } else if odd {
            2
        } else {
            4
        };
        // SAFETY: a transform has at least 32 points. Radix-2, radix-4, and root
        // radix-8 leave nonempty children; paired stages leave at least a cache
        // block.
        let child_len = unsafe { NonZeroUsize::new_unchecked(len >> arity.trailing_zeros()) };
        if let Some(kernel) = wide {
            self.stage::<16>(data, roots, kernel, budget);
        } else if paired_root {
            self.prefix::<8>(data, roots.values, input_len, budget);
        } else if odd {
            if input_len < len {
                self.prefix::<2>(data, roots.values, input_len, budget);
            } else {
                let kernel = if roots.node == 0 {
                    kernels.radix2
                } else {
                    kernels.dif2
                };
                self.stage::<2>(data, roots, kernel, budget);
            }
        } else if input_len < len {
            self.prefix::<4>(data, roots.values, input_len, budget);
        } else {
            let kernel = if roots.node == 0 {
                kernels.dif_unit
            } else {
                kernels.dif
            };
            self.stage::<4>(data, roots, kernel, budget);
        }
        // SAFETY: arity*j names the first child at the next depth, inside the table.
        roots.node = unsafe { roots.node.unchecked_mul(arity) };
        if len <= CACHE_LEAF {
            self.forward_leaf(data, roots, child_len.get());
            return;
        }
        let run_children = |span: &mut [f64], threads, start: usize| {
            let mut child_roots = roots;
            // SAFETY: start is zero or half the parent's arity.
            child_roots.node = unsafe { child_roots.node.unchecked_add(start) };
            for child in span.chunks_exact_mut(child_len.get()) {
                self.forward(child, child_roots, child_len.get(), threads);
                // SAFETY: this advances at most one past the complete child
                // interval, which the table bounds.
                child_roots.node = unsafe { child_roots.node.unchecked_add(1) };
            }
        };
        if len >= PARALLEL_MIN_LEN
            && let Some(split) = Partition::children(arity, budget)
        {
            // SAFETY: the arity partitions len into child_len slots, so an interior
            // child cut lies strictly inside data.
            let (left, right) =
                unsafe { data.split_at_mut_unchecked(split.middle.unchecked_mul(child_len.get())) };
            self.executor.join(
                || run_children(left, split.left, 0),
                || run_children(right, split.right, split.middle),
            );
        } else {
            run_children(data, NonZeroUsize::MIN, 0);
        }
    }

    /// Runs the contiguous descendants of a cache block, whose consecutive nodes
    /// follow one another in the shared table.
    ///
    /// The cache block's constant assertion makes the steps end at its leaves, and a
    /// transform holds an even number of them.
    fn forward_leaf(&self, data: &mut [f64], mut roots: Roots<'_>, mut step: usize) {
        let kernels = self.kernels;
        while step > LEAF_POINTS {
            for (index, block) in data.chunks_exact_mut(step).enumerate() {
                // SAFETY: consecutive blocks name the nodes at this depth.
                let node = unsafe { roots.node.unchecked_add(index) };
                let kernel = if node == 0 {
                    kernels.dif_unit
                } else {
                    kernels.dif
                };
                self.stage::<4>(
                    block,
                    Roots {
                        values: roots.values,
                        node,
                    },
                    kernel,
                    NonZeroUsize::MIN,
                );
            }
            // SAFETY: every node has four children at the next depth.
            roots.node = unsafe { roots.node.unchecked_mul(4) };
            step >>= 2;
        }
        // SAFETY: parent stages initialize every leaf, and the table covers each node
        // and its descendant roots. A transform holds at least MIN_TRANSFORM_LEN, two
        // leaves, so it holds a positive number of leaf pairs. Other leaves ignore the
        // right spectrum.
        unsafe {
            (kernels.dif16)(
                data.as_mut_ptr(),
                null(),
                roots.values.as_ptr(),
                NonZeroUsize::new_unchecked(data.len() >> MIN_TRANSFORM_LEN.trailing_zeros()),
                roots.node,
                self.prime.float_field(),
            );
        }
    }
}
