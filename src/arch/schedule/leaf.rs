//! Two radix-4 stages and inverse spectral products retained in registers.
//!
//! DIF output and DIT input share transposed four-by-four leaf coordinates.
//! A common permutation commutes with pointwise products, so the final DIF
//! transpose and initial DIT transpose are unnecessary for convolution.

/// Instantiates leaves that hold each four-lane row in one register.
///
/// | Argument | Meaning |
/// | :--- | :--- |
/// | `$name`, `[$attribute]`, `$splat` | as in `schedule_backend!` |
/// | `$points` | points of one iteration, one or two leaves |
/// | `$load_vectors` | `(pointer) -> vectors` of one iteration |
/// | `$unpack_rows` | `(vectors) -> rows`, the butterfly order of those registers |
/// | `$store_rows` | `(pointer, rows)`, the inverse of the two above |
/// | `$roots` | `(table, node) -> (top, children)`, the roots of one iteration |
/// | `$transpose` | `(rows) -> rows`, the four-by-four transpose within each leaf |
macro_rules! leaf16_backend {
    ($name:ident, [$($attribute:meta),*], $points:literal,
     $splat:path, $load_vectors:expr, $unpack_rows:expr, $store_rows:expr, $roots:expr, $transpose:expr) => {
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
                // SAFETY: each iteration owns one or two complete leaves.
                // Parent and child root loads fit below 8*(node+2*pairs).
                // Two-leaf vectors load sixteen consecutive child roots;
                // one-leaf vectors load eight. Transposition stays within each leaf
                // and commutes with componentwise products. The product
                // bound abs(a*b) < 9*p^2/4 gives residues below 3*p/2.
                unsafe {
                    let modulus = $splat(field.p());
                    let reciprocal = $splat(field.pinv());
                    let mut offset = 0;
                    let len = pairs.get().unchecked_mul(32);
                    loop {
                        let current = node.unchecked_add(offset >> 4);
                        let (top, children) = ($roots)(twiddles, current);
                        let mut vectors = ($load_vectors)(values.add(offset));
                        if INVERSE && PRODUCT {
                            let factors = if SQUARE {
                                vectors
                            } else {
                                ($load_vectors)(right.add(offset))
                            };
                            vectors = (
                                multiply_vector(vectors.0, factors.0, modulus, reciprocal),
                                multiply_vector(vectors.1, factors.1, modulus, reciprocal),
                                multiply_vector(vectors.2, factors.2, modulus, reciprocal),
                                multiply_vector(vectors.3, factors.3, modulus, reciprocal),
                            );
                        }
                        // Permute the product once, using P(a*b)=P(a)*P(b).
                        let mut rows = ($unpack_rows)(vectors);
                        if INVERSE {
                            rows = butterfly::<true, false, false>(rows, children, modulus, reciprocal);
                            rows = ($transpose)(rows);
                            rows = butterfly::<true, false, false>(rows, top, modulus, reciprocal);
                        } else {
                            rows = butterfly::<false, false, false>(rows, top, modulus, reciprocal);
                            rows = ($transpose)(rows);
                            rows = butterfly::<false, false, true>(rows, children, modulus, reciprocal);
                        }
                        ($store_rows)(values.add(offset), rows);
                        offset = offset.unchecked_add($points);
                        if offset == len {
                            break;
                        }
                    }
                }
            }
        }
    };
}
