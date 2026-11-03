//! Inverse remainder-tree execution consumes spectral products at its leaves.

#![expect(
    unsafe_code,
    reason = "Admitted power-of-two geometry proves ascending stages and disjoint worker spans"
)]

use core::num::NonZeroUsize;

use super::{
    CACHE_LEAF, Columns, Executor, LEAF_POINTS, MIN_TRANSFORM_LEN, PARALLEL_MIN_LEN, Partition,
    Prime, ROOT_MIN_LEN, Roots, TAIL_NODE, Transform,
};

/// The final scaling of an inverse, fused with its canonical store.
#[derive(Clone, Copy, Debug)]
pub struct Normalization {
    /// Active output columns from zero. They fit the row and bound the degree of its
    /// polynomial.
    pub columns: Columns,
    /// The scale of the cyclic part in `0..p`, combined with the CRT inverse.
    pub factor: f64,
    /// The same scale centered below `p/2`.
    pub centered: f64,
}

impl Normalization {
    /// Returns the normalization of `columns` output columns by `factor` in `0..p`.
    #[must_use]
    pub fn new(columns: Columns, factor: f64, prime: &Prime) -> Self {
        let modulus = prime.float_field().p();
        Self {
            columns,
            factor,
            centered: if factor > modulus * 0.5 {
                factor - modulus
            } else {
                factor
            },
        }
    }
}

impl<E: Executor> Transform<'_, E> {
    /// Completes a row inverse and converts its active prefix to CRT bits.
    ///
    /// # Safety
    /// `row` is a row of the module terms whose cyclic part has `cyclic_len` points,
    /// below `2*p`, and `roots` covers its nodes. A multiplication reads `source`,
    /// a disjoint row of the same length and bound; other modes ignore it. The
    /// normalization's columns fit the row.
    pub unsafe fn finish_row<const PRODUCT: bool, const SQUARE: bool>(
        &self,
        (row, source): (&mut [f64], &[f64]),
        (roots, cyclic_len): (&[f64], usize),
        normalization: Normalization,
        budget: NonZeroUsize,
    ) {
        if row.len() == cyclic_len {
            self.finish::<PRODUCT, SQUARE>((row, source), roots, normalization, budget);
            return;
        }
        let cyclic_roots = Roots {
            values: roots,
            node: 0,
        };
        let tail_roots = Roots {
            values: roots,
            node: TAIL_NODE,
        };
        let parallel = budget.get() > 1 && row.len() >= PARALLEL_MIN_LEN;
        // SAFETY: the row and a read multiplication source hold N+M floats, and
        // budget > 1 proves 1 <= max(budget/3, 1) < budget.
        unsafe {
            let (cyclic, tail) = row.split_at_mut_unchecked(cyclic_len);
            let (cyclic_source, tail_source) = if !PRODUCT || SQUARE {
                (&[][..], &[][..])
            } else {
                source.split_at_unchecked(cyclic_len)
            };
            if parallel {
                let workers = NonZeroUsize::new_unchecked(budget.get().div_euclid(3).max(1));
                let remaining =
                    NonZeroUsize::new_unchecked(budget.get().unchecked_sub(workers.get()));
                self.executor.join(
                    || {
                        self.inverse::<PRODUCT, SQUARE>(
                            (cyclic, cyclic_source),
                            cyclic_roots,
                            remaining,
                        );
                    },
                    || self.inverse::<PRODUCT, SQUARE>((tail, tail_source), tail_roots, workers),
                );
            } else {
                self.inverse::<PRODUCT, SQUARE>(
                    (cyclic, cyclic_source),
                    cyclic_roots,
                    NonZeroUsize::MIN,
                );
                self.inverse::<PRODUCT, SQUARE>((tail, tail_source), tail_roots, NonZeroUsize::MIN);
            }
        }
        self.recombine(row, cyclic_len >> 1, normalization, budget);
    }

    /// Completes a product inverse at the unit root of `roots` and converts its
    /// active prefix to CRT bits.
    pub fn finish<const PRODUCT: bool, const SQUARE: bool>(
        &self,
        spectra: (&mut [f64], &[f64]),
        roots: &[f64],
        normalization: Normalization,
        budget: NonZeroUsize,
    ) {
        if spectra.0.len() > CACHE_LEAF {
            self.finish_recursive::<PRODUCT, SQUARE>(spectra, roots, normalization, budget);
        } else {
            let (data, source) = spectra;
            self.inverse::<PRODUCT, SQUARE>(
                (&mut *data, source),
                Roots {
                    values: roots,
                    node: 0,
                },
                budget,
            );
            // SAFETY: the complete inverse initializes the active prefix below 2*p.
            unsafe {
                (self.kernels.residues)(
                    data.as_mut_ptr(),
                    normalization.columns,
                    normalization.factor,
                    self.prime.float_field(),
                );
            }
        }
    }

    /// Inverts the children of the global root, then its fused scaling stage.
    fn finish_recursive<const PRODUCT: bool, const SQUARE: bool>(
        &self,
        spectra: (&mut [f64], &[f64]),
        roots: &[f64],
        normalization: Normalization,
        budget: NonZeroUsize,
    ) {
        let len = spectra.0.len();
        let odd = len.trailing_zeros() & 1 != 0;
        let paired_root = odd && len >= ROOT_MIN_LEN;
        let arity = if paired_root {
            8
        } else if odd {
            2
        } else {
            4
        };
        let data = self.inverse_children::<PRODUCT, SQUARE>(
            spectra,
            Roots {
                values: roots,
                node: 0,
            },
            arity,
            budget,
        );
        if paired_root {
            self.finish_stage::<8>(data, roots, normalization, budget);
        } else if odd {
            self.finish_stage::<2>(data, roots, normalization, budget);
        } else {
            self.finish_stage::<4>(data, roots, normalization, budget);
        }
    }

    /// Inverts a spectrum, optionally forming products at its leaves.
    ///
    /// Residues are below `2*p`. With `PRODUCT`, leaves multiply an equal-length
    /// second spectrum with the same bound, or square the first. Other modes ignore
    /// the second spectrum.
    pub fn inverse<const PRODUCT: bool, const SQUARE: bool>(
        &self,
        (data, source): (&mut [f64], &[f64]),
        roots: Roots<'_>,
        budget: NonZeroUsize,
    ) {
        let kernels = self.kernels;
        let len = data.len();
        if len > CACHE_LEAF {
            self.inverse_recursive::<PRODUCT, SQUARE>((data, source), roots, budget);
            return;
        }
        let odd = len.trailing_zeros() & 1 != 0;
        let limit = if odd { len >> 1 } else { len };
        let blocks = len >> LEAF_POINTS.trailing_zeros();
        // SAFETY: the data holds complete initialized leaves and the table covers
        // their nodes. A transform holds at least MIN_TRANSFORM_LEN, two leaves, so
        // it holds a positive number of leaf pairs. Product multiplication reads an
        // equal-sized, disjoint source; other modes ignore it.
        unsafe {
            let leaf = if !PRODUCT {
                kernels.dit16
            } else if SQUARE {
                kernels.dit16_sqr
            } else {
                kernels.dit16_mul
            };
            leaf(
                data.as_mut_ptr(),
                source.as_ptr(),
                roots.values.as_ptr(),
                NonZeroUsize::new_unchecked(len >> MIN_TRANSFORM_LEN.trailing_zeros()),
                roots.node.unchecked_mul(blocks),
                self.prime.float_field(),
            );
        }
        let mut step = LEAF_POINTS << 2;
        while step <= limit {
            // SAFETY: step is a power of four no greater than len, and len/step nodes
            // at this depth name the contiguous blocks below this parent.
            let first = unsafe { roots.node.unchecked_mul(len >> step.trailing_zeros()) };
            for (index, block) in data.chunks_exact_mut(step).enumerate() {
                // SAFETY: the block index remains in the node interval.
                let node = unsafe { first.unchecked_add(index) };
                let kernel = if node == 0 {
                    kernels.dit_unit
                } else {
                    kernels.dit
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
            step <<= 2;
        }
        if odd {
            let kernel = if roots.node == 0 {
                kernels.radix2
            } else {
                kernels.dit2
            };
            self.stage::<2>(data, roots, kernel, NonZeroUsize::MIN);
        }
    }

    /// Inverts every child of a node above the cache leaf, then its stage.
    fn inverse_recursive<const PRODUCT: bool, const SQUARE: bool>(
        &self,
        spectra: (&mut [f64], &[f64]),
        roots: Roots<'_>,
        budget: NonZeroUsize,
    ) {
        let kernels = self.kernels;
        let len = spectra.0.len();
        let odd = len.trailing_zeros() & 1 != 0;
        let wide = kernels
            .dit_wide
            .filter(|_| !odd && roots.node != 0 && len >= self.fused_min_len);
        let arity = if wide.is_some() {
            16_usize
        } else if odd {
            2
        } else {
            4
        };
        let data = self.inverse_children::<PRODUCT, SQUARE>(spectra, roots, arity, budget);
        if let Some(kernel) = wide {
            self.stage::<16>(data, roots, kernel, budget);
        } else if odd {
            let kernel = if roots.node == 0 {
                kernels.radix2
            } else {
                kernels.dit2
            };
            self.stage::<2>(data, roots, kernel, budget);
        } else {
            let kernel = if roots.node == 0 {
                kernels.dit_unit
            } else {
                kernels.dit
            };
            self.stage::<4>(data, roots, kernel, budget);
        }
    }

    fn inverse_children<'data, const PRODUCT: bool, const SQUARE: bool>(
        &self,
        (data, source): (&'data mut [f64], &[f64]),
        roots: Roots<'_>,
        arity: usize,
        budget: NonZeroUsize,
    ) -> &'data mut [f64] {
        let len = data.len();
        // SAFETY: len > CACHE_LEAF is a power of two, so dividing by two, four, eight,
        // or sixteen leaves positive complete children.
        let child_len = unsafe { NonZeroUsize::new_unchecked(len >> arity.trailing_zeros()) };
        // SAFETY: the next depth has arity times as many nodes; this parent and its
        // complete child interval lie in the table.
        let first = unsafe { roots.node.unchecked_mul(arity) };
        let run_children = |(values, factors): (&mut [f64], &[f64]), threads, start: usize| {
            for (index, child) in values.chunks_exact_mut(child_len.get()).enumerate() {
                let source_child = if !PRODUCT || SQUARE {
                    &[][..]
                } else {
                    // SAFETY: both spectra have equal lengths, and
                    // index*child_len+child_len <= factors.len() bounds the complete
                    // source interval.
                    unsafe {
                        let offset = index.unchecked_mul(child_len.get());
                        factors.get_unchecked(offset..offset.unchecked_add(child_len.get()))
                    }
                };
                self.inverse::<PRODUCT, SQUARE>(
                    (child, source_child),
                    Roots {
                        values: roots.values,
                        // SAFETY: start is zero or arity/2, and values covers the
                        // corresponding complete children, so start+index < arity
                        // inside the node interval.
                        node: unsafe { first.unchecked_add(start).unchecked_add(index) },
                    },
                    threads,
                );
            }
        };
        if len >= PARALLEL_MIN_LEN
            && let Some(split) = Partition::children(arity, budget)
        {
            // SAFETY: the arity partitions len into child_len slots, so an interior
            // child cut lies inside both equally sized spectra. Non-product and square
            // modes do not read the second spectrum.
            let ((left, right), (left_source, right_source)) = unsafe {
                let cut = split.middle.unchecked_mul(child_len.get());
                (
                    data.split_at_mut_unchecked(cut),
                    if !PRODUCT || SQUARE {
                        (&[][..], &[][..])
                    } else {
                        source.split_at_unchecked(cut)
                    },
                )
            };
            self.executor.join(
                || run_children((left, left_source), split.left, 0),
                || run_children((right, right_source), split.right, split.middle),
            );
        } else {
            run_children((&mut *data, source), NonZeroUsize::MIN, 0);
        }
        data
    }
}
