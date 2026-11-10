//! Stages that only rows with tails need.
//!
//! A row of `3*M`, `5*M`, or `7*M` points holds residues modulo `x^N-1` and
//! modulo one or two tail factors: `x^M-i` with `N = 2*M`, or `x^M-w`, or
//! `x^(2*M)-i` and `x^M-i*w` with `N = 4*M`. Every tail factor divides `x^N+1`,
//! so a tail needs the reduction of a packed input and, when its length has odd
//! order, a radix-2 stage with a root.

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
                // spans. The root product has a residue below 15*p/16, so the
                // output stays below 31*p/16, inside the forward bound of 3*p.
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

            /// Writes `(a0 + r1*a2) + r2*(a1 + r1*a3)` for four packed input
            /// spans to span `TARGET`.
            ///
            /// # Safety
            /// The kernel contract holds for four packed spans and a writable
            /// span at `TARGET * span` that need not be initialized, with two
            /// roots.
            $(#[$attribute])*
            pub unsafe fn fold4<const TARGET: usize>(
                values: *mut f64,
                twiddles: *const f64,
                span: usize,
                columns: $crate::arch::Columns,
                field: $crate::FloatField,
            ) {
                const { assert!(matches!(TARGET, 4 | 6), "four-span fold target") };
                // SAFETY: complete vectors stay inside the five disjoint spans.
                // Each centered root product has a residue below 15*p/16, so
                // `a0 + r1*a2` and `a1 + r1*a3` stay below 31*p/16, the outer
                // product stays below 15*p/16, and the output stays below
                // 23*p/8 < 3*p.
                unsafe {
                    let (start, end) = columns.complete::<$lanes>();
                    let modulus = ($splat)(field.p());
                    let reciprocal = ($splat)(field.pinv());
                    let r1 = ($splat)(*twiddles);
                    let r2 = ($splat)(*twiddles.add(1));
                    let s1 = values.add(span);
                    let s2 = s1.add(span);
                    let s3 = s2.add(span);
                    let target = values.add(span.unchecked_mul(TARGET));
                    let mut index = start;
                    loop {
                        let a0 = ($load)(values.add(index));
                        let a1 = ($load)(s1.add(index));
                        let a2 = ($load)(s2.add(index));
                        let a3 = ($load)(s3.add(index));
                        let even = ($add)(a0, ($multiply)(a2, r1, modulus, reciprocal));
                        let odd = ($add)(a1, ($multiply)(a3, r1, modulus, reciprocal));
                        let rotated = ($multiply)(odd, r2, modulus, reciprocal);
                        ($store)(target.add(index), ($add)(even, rotated));
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
