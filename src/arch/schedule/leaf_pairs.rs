//! Sixteen-point leaves for two-lane vectors.
//!
//! Each four-lane row is a pair of two-lane vectors holding lanes 0..2 and
//! 2..4. Row butterflies act on the low and the high halves separately. The
//! four-by-four transpose takes column `2*h + k` from the `k`-th lanes of the
//! `h` halves: `low(a, b) = [a0, b0]` and `high(a, b) = [a1, b1]`.

/// Instantiates leaves for two-lane registers.
///
/// `$name`, `[$attribute]`, `$vector`, `$splat`, `$load`, and `$store` are as in
/// `schedule_backend!`. `$low` and `$high` interleave the first and the second
/// lanes of two registers.
macro_rules! leaf16_pair_backend {
    ($name:ident, [$($attribute:meta),*], $vector:ty,
     $splat:path, $load:expr, $store:expr, $low:expr, $high:expr) => {
        impl $name {
            /// Executes forward leaves or inverse leaves with optional products.
            ///
            /// # Safety
            /// The contract of `LeafKernel` holds with residues below `3*p`, or
            /// `2*p` for the inverse. Forward outputs are below `3*p/2`.
            $(#[$attribute])*
            pub unsafe fn leaf16<const INVERSE: bool, const PRODUCT: bool, const SQUARE: bool>(
                values: *mut f64,
                right: *const f64,
                twiddles: *const f64,
                pairs: ::core::num::NonZeroUsize,
                node: usize,
                field: $crate::FloatField,
            ) {
                // SAFETY: each iteration owns one complete leaf of sixteen
                // residues, read and written as eight two-lane vectors. Its
                // largest root index is 8*(node+block)+7, inside the admitted
                // table. Transposition stays within the leaf and commutes with
                // componentwise products, whose bound abs(a*b) < 9*p^2/4 gives
                // residues below 3*p/2.
                unsafe {
                    let modulus = $splat(field.p());
                    let reciprocal = $splat(field.pinv());
                    let blocks = pairs.get().unchecked_mul(2);
                    let mut block = 0_usize;
                    loop {
                        let offset = block.unchecked_mul(16);
                        let base = values.add(offset);
                        let current = node.unchecked_add(block);
                        let row = |source: *const f64, index: usize| {
                            let start = source.add(index.unchecked_mul(4));
                            (($load)(start), ($load)(start.add(2)))
                        };
                        let mut rows = [row(base, 0), row(base, 1), row(base, 2), row(base, 3)];
                        if INVERSE && PRODUCT {
                            let factors = if SQUARE {
                                rows
                            } else {
                                let other = right.add(offset);
                                [row(other, 0), row(other, 1), row(other, 2), row(other, 3)]
                            };
                            for ((low, high), (left, upper)) in rows.iter_mut().zip(factors) {
                                *low = multiply_vector(*low, left, modulus, reciprocal);
                                *high = multiply_vector(*high, upper, modulus, reciprocal);
                            }
                        }
                        let twice = current.unchecked_mul(2);
                        let top = (
                            $splat(*twiddles.add(current)),
                            $splat(*twiddles.add(twice)),
                            $splat(*twiddles.add(twice.unchecked_add(1))),
                        );
                        // Child j of the leaf uses lane j: W = roots[4*n+j],
                        // left = roots[8*n+2*j], right = roots[8*n+2*j+1].
                        let quad = twiddles.add(current.unchecked_mul(4));
                        let octet = twiddles.add(current.unchecked_mul(8));
                        let (r01, r23) = (($load)(octet), ($load)(octet.add(2)));
                        let (r45, r67) = (($load)(octet.add(4)), ($load)(octet.add(6)));
                        let children = [
                            (($load)(quad), ($low)(r01, r23), ($high)(r01, r23)),
                            (($load)(quad.add(2)), ($low)(r45, r67), ($high)(r45, r67)),
                        ];
                        let transpose = |[(a0, a1), (b0, b1), (c0, c1), (d0, d1)]: [($vector, $vector); 4]| {
                            [
                                (($low)(a0, b0), ($low)(c0, d0)),
                                (($high)(a0, b0), ($high)(c0, d0)),
                                (($low)(a1, b1), ($low)(c1, d1)),
                                (($high)(a1, b1), ($high)(c1, d1)),
                            ]
                        };
                        if INVERSE {
                            rows = leaf_halves::<true, false>(rows, children, modulus, reciprocal);
                            rows = transpose(rows);
                            rows = leaf_halves::<true, false>(rows, [top, top], modulus, reciprocal);
                        } else {
                            rows = leaf_halves::<false, false>(rows, [top, top], modulus, reciprocal);
                            rows = transpose(rows);
                            rows = leaf_halves::<false, true>(rows, children, modulus, reciprocal);
                        }
                        for (index, (low, high)) in rows.into_iter().enumerate() {
                            let start = base.add(index.unchecked_mul(4));
                            ($store)(start, low);
                            ($store)(start.add(2), high);
                        }
                        block = block.unchecked_add(1);
                        if block == blocks {
                            break;
                        }
                    }
                }
            }
        }

        // Always inlined into the leaf kernel, whose target features the body
        // then compiles with; see the butterfly in `stages`.
        #[inline(always)]
        unsafe fn leaf_halves<const INVERSE: bool, const LAST: bool>(
            [(a0, a1), (b0, b1), (c0, c1), (d0, d1)]: [($vector, $vector); 4],
            [(w0, l0, r0), (w1, l1, r1)]: [($vector, $vector, $vector); 2],
            modulus: $vector,
            reciprocal: $vector,
        ) -> [($vector, $vector); 4] {
            // SAFETY: the caller is a leaf kernel of this backend, whose target
            // features admit the butterflies, with residues in their bounds.
            unsafe {
                let lower = butterfly::<INVERSE, false, LAST>(
                    (a0, b0, c0, d0),
                    (w0, l0, r0),
                    modulus,
                    reciprocal,
                );
                let upper = butterfly::<INVERSE, false, LAST>(
                    (a1, b1, c1, d1),
                    (w1, l1, r1),
                    modulus,
                    reciprocal,
                );
                [
                    (lower.0, upper.0),
                    (lower.1, upper.1),
                    (lower.2, upper.2),
                    (lower.3, upper.3),
                ]
            }
        }
    };
}
