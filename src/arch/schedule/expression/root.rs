//! Unit radix-2 and its radix-4 children evaluated in one coefficient pass.

macro_rules! root8_backend {
    ($name:ident, $lanes:literal, [$($attribute:meta),*],
     $load:expr, $store:expr, $splat:expr, $add:expr, $sub:expr,
     $reduce:expr, $multiply:expr) => {
        impl $name {
            /// Fuses a forward unit radix-2 stage with its two radix-4 children.
            ///
            /// # Safety
            /// The kernel contract holds for eight spans, whose first INPUTS spans
            /// are packed; omitted spans represent zero and need not be
            /// initialized. Roots are W1, W2, W3.
            $(#[$attribute])*
            pub unsafe fn root8<const INPUTS: usize>(
                values: *mut f64,
                twiddles: *const f64,
                stride: usize,
                columns: $crate::arch::Columns,
                field: $crate::FloatField,
            ) {
                const { assert!(matches!(INPUTS, 1..=4), "root stage requires one to four inputs") };
                // SAFETY: complete vectors fit the columns in every active span.
                // Packed values are below p, so their sums are below 2*p.
                // Centered root products have residues below 3*p/4, and a
                // reduced sum is below 5*p/8. Every output is below 3*p, the
                // input bound of the next stage. All eight output spans are
                // disjoint. Three constants are readable; four-entry
                // temporaries are indexed only in 0..4. Omitted inputs are
                // never read.
                unsafe {
                    let (start, end) = columns.complete::<$lanes>();
                    let modulus = ($splat)(field.p());
                    let reciprocal = ($splat)(field.pinv());
                    let roots = [
                        ($splat)(*twiddles),
                        ($splat)(*twiddles.add(1)),
                        ($splat)(*twiddles.add(2)),
                    ];
                    let mut index = start;
                    loop {
                        let base = values.add(index);
                        let read = |row: usize| ($load)(base.add(row.unchecked_mul(stride)));
                        let reduce = |value| ($reduce)(value, modulus, reciprocal);
                        let multiply = |value, root| ($multiply)(value, root, modulus, reciprocal);
                        let a = read(0);
                        let b = if INPUTS == 1 { a } else { read(1) };
                        let (first, last) = if INPUTS == 1 {
                            ([a; 4], [a; 4])
                        } else if INPUTS == 2 {
                            let t = multiply(b, roots[0]);
                            let v = multiply(b, roots[1]);
                            let w = multiply(b, roots[2]);
                            (
                                [($add)(a, b), ($sub)(a, b), ($add)(a, t), ($sub)(a, t)],
                                [($add)(a, v), ($sub)(a, v), ($add)(a, w), ($sub)(a, w)],
                            )
                        } else {
                            let c = read(2);
                            let u = ($add)(a, c);
                            let s = ($sub)(a, c);
                            let (v, t, d1) = if INPUTS == 3 {
                                (b, multiply(b, roots[0]), ($splat)(0.0))
                            } else {
                                let d = read(3);
                                (
                                    reduce(($add)(b, d)),
                                    multiply(($sub)(b, d), roots[0]),
                                    multiply(d, roots[0]),
                                )
                            };
                            let left = [($add)(u, v), ($sub)(u, v), ($add)(s, t), ($sub)(s, t)];
                            let c1 = multiply(c, roots[0]);
                            let r0 = ($add)(a, c1);
                            let r1 = ($sub)(a, c1);
                            let t0 = multiply(($add)(b, d1), roots[1]);
                            let t1 = multiply(($sub)(b, d1), roots[2]);
                            (
                                left,
                                [($add)(r0, t0), ($sub)(r0, t0), ($add)(r1, t1), ($sub)(r1, t1)],
                            )
                        };
                        for row in 0_usize..4 {
                            ($store)(base.add(row.unchecked_mul(stride)), *first.get_unchecked(row));
                            ($store)(
                                base.add(row.unchecked_add(4).unchecked_mul(stride)),
                                *last.get_unchecked(row),
                            );
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
