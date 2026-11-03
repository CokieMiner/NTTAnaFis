//! Stages that only a row of `3*M` points needs.
//!
//! Such a row holds residues modulo `x^(2*M)-1` and modulo `x^M-i`. The second
//! factor enters the remainder tree below the node `x^(2*M)+1`, so it needs the
//! reduction of a packed input and, when `M` has odd order, a radix-2 stage with
//! a root.

macro_rules! tail_backend {
    ($name:ident, $lanes:literal, [$($attribute:meta),*],
     $load:expr, $store:expr, $splat:expr, $add:expr, $sub:expr,
     $reduce:expr, $multiply:expr) => {
        impl $name {
            /// Writes `a + w*b` for two packed input spans to a third span.
            ///
            /// # Safety
            /// The kernel contract holds for two packed spans and a third writable
            /// span that need not be initialized, with one root.
            $(#[$attribute])*
            pub unsafe fn fold(
                values: *mut f64,
                twiddles: *const f64,
                span: usize,
                columns: $crate::arch::Columns,
                field: $crate::FloatField,
            ) {
                // SAFETY: complete vectors stay inside the three disjoint
                // spans. The root product has a residue below 5*p/8, so the
                // output stays below 13*p/8, inside the forward bound of 3*p.
                unsafe {
                    let (start, end) = columns.complete::<$lanes>();
                    let modulus = ($splat)(field.p());
                    let reciprocal = ($splat)(field.pinv());
                    let root = ($splat)(*twiddles);
                    let second = values.add(span);
                    let target = second.add(span);
                    let mut index = start;
                    loop {
                        let a = ($load)(values.add(index));
                        let b = ($load)(second.add(index));
                        let rotated = ($multiply)(b, root, modulus, reciprocal);
                        ($store)(target.add(index), ($add)(a, rotated));
                        index = index.unchecked_add($lanes);
                        if index == end {
                            break;
                        }
                    }
                }
            }

            /// Executes a radix-2 stage with one root.
            ///
            /// # Safety
            /// The kernel contract holds for two halves below `3*p`, or `2*p` for
            /// the inverse, with the node root or its inverse.
            $(#[$attribute])*
            pub unsafe fn radix2_node<const INVERSE: bool>(
                values: *mut f64,
                twiddles: *const f64,
                half: usize,
                columns: $crate::arch::Columns,
                field: $crate::FloatField,
            ) {
                // SAFETY: complete vectors stay inside both disjoint halves.
                // Forward: abs(b*w) < 3*p^2/2 bounds the product by 15*p/16 and
                // the reduced input by 9*p/16, so outputs stay below 3*p/2.
                // Inverse: the sum reduces below 9*p/16, and abs(a-b) < 4*p times
                // a centered root is below 2*p^2, which bounds the product by
                // 15*p/16. Both outputs stay below 2*p.
                unsafe {
                    let (start, end) = columns.complete::<$lanes>();
                    let modulus = ($splat)(field.p());
                    let reciprocal = ($splat)(field.pinv());
                    let root = ($splat)(*twiddles);
                    let upper = values.add(half);
                    let mut index = start;
                    loop {
                        let a = ($load)(values.add(index));
                        let b = ($load)(upper.add(index));
                        if INVERSE {
                            let sum = ($add)(a, b);
                            let difference = ($sub)(a, b);
                            ($store)(values.add(index), ($reduce)(sum, modulus, reciprocal));
                            ($store)(
                                upper.add(index),
                                ($multiply)(difference, root, modulus, reciprocal),
                            );
                        } else {
                            let rotated = ($multiply)(b, root, modulus, reciprocal);
                            let centered = ($reduce)(a, modulus, reciprocal);
                            ($store)(values.add(index), ($add)(centered, rotated));
                            ($store)(upper.add(index), ($sub)(centered, rotated));
                        }
                        index = index.unchecked_add($lanes);
                        if index == end {
                            break;
                        }
                    }
                }
            }
        }
    };
}
