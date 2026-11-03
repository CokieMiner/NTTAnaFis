//! Last inverse stages with register-resident scaling and canonical CRT stores.

macro_rules! finish_backend {
    ($name:ident, $lanes:literal, [$($attribute:meta),*],
     $load:expr, $store:expr, $splat:expr, $add:expr, $sub:expr,
     $reduce:expr, $multiply:expr, $canonical:expr) => {
        impl $name {
            /// Completes a unit inverse root and stores only its active spans.
            ///
            /// # Safety
            /// The kernel contract holds for SPANS spans below `2*p`. The three
            /// roots use the unit radix-4 layout, or W1..W3 for SPANS=8. `factor`
            /// is centered below `p/2`. The first OUTPUTS spans receive canonical
            /// bits.
            $(#[$attribute])*
            pub unsafe fn finish<const SPANS: usize, const OUTPUTS: usize>(
                values: *mut f64,
                twiddles: *const f64,
                stride: usize,
                columns: $crate::arch::Columns,
                factor: f64,
                field: $crate::FloatField,
            ) {
                const {
                    assert!(matches!(SPANS, 2 | 4 | 8), "unit inverse root arity");
                    assert!(OUTPUTS > 0 && OUTPUTS <= SPANS, "active inverse outputs");
                }
                // SAFETY: each complete vector lies inside the selected columns
                // of all disjoint spans. Four-span outputs are below 2*p;
                // two/eight-span outputs are below 4*p. Multiplication by a
                // centered factor therefore has magnitude below 2*p^2, whose
                // exact remainder lies in (-p,p), the input of canonical
                // conversion. Every source vector is loaded before its columns
                // receive integer bit patterns.
                unsafe {
                    let (start, end) = columns.complete::<$lanes>();
                    let modulus = ($splat)(field.p());
                    let reciprocal = ($splat)(field.pinv());
                    let multiplier = ($splat)(factor);
                    let roots = if SPANS == 2 {
                        [($splat)(0.0); 3]
                    } else {
                        [($splat)(*twiddles), ($splat)(*twiddles.add(1)), ($splat)(*twiddles.add(2))]
                    };
                    let mut index = start;
                    loop {
                        let base = values.add(index);
                        let read = |row: usize| ($load)(base.add(row.unchecked_mul(stride)));
                        let reduce = |value| ($reduce)(value, modulus, reciprocal);
                        let multiply = |value, root| ($multiply)(value, root, modulus, reciprocal);
                        let write = |row: usize, value| {
                            if row < OUTPUTS {
                                let scaled = multiply(value, multiplier);
                                ($store)(base.add(row.unchecked_mul(stride)), ($canonical)(scaled, modulus));
                            }
                        };
                        let a = read(0);
                        let b = read(1);
                        if SPANS == 2 {
                            write(0, ($add)(a, b));
                            write(1, ($sub)(a, b));
                        } else {
                            let c = read(2);
                            let d = read(3);
                            let u = reduce(($add)(a, b));
                            let s = reduce(($add)(c, d));
                            let v = reduce(($sub)(a, b));
                            let t = multiply(($sub)(c, d), if SPANS == 4 { roots[2] } else { roots[0] });
                            let left = [($add)(u, s), ($add)(v, t), ($sub)(u, s), ($sub)(v, t)];
                            // Rows are constants, so each inactive output folds away.
                            if SPANS == 4 {
                                write(0, left[0]);
                                write(1, left[1]);
                                write(2, left[2]);
                                write(3, left[3]);
                            } else {
                                let e = read(4);
                                let f = read(5);
                                let g = read(6);
                                let h = read(7);
                                let r0 = ($add)(e, f);
                                let r1 = ($add)(g, h);
                                let t0 = multiply(($sub)(e, f), roots[1]);
                                let t1 = multiply(($sub)(g, h), roots[2]);
                                let right = [
                                    reduce(($add)(r0, r1)),
                                    ($add)(t0, t1),
                                    multiply(($sub)(r0, r1), roots[0]),
                                    multiply(($sub)(t0, t1), roots[0]),
                                ];
                                // Each pair is consumed before the next pair's
                                // final sum and difference become live.
                                write(0, ($add)(left[0], right[0]));
                                write(4, ($sub)(left[0], right[0]));
                                write(1, ($add)(left[1], right[1]));
                                write(5, ($sub)(left[1], right[1]));
                                write(2, ($add)(left[2], right[2]));
                                write(6, ($sub)(left[2], right[2]));
                                write(3, ($add)(left[3], right[3]));
                                write(7, ($sub)(left[3], right[3]));
                            }
                        }
                        index = index.unchecked_add($lanes);
                        if index == end {
                            break;
                        }
                    }
                }
            }

            /// Recovers coefficients from residues modulo `x^(2*M)-1` and
            /// `x^M-i`, and stores them as canonical bits.
            ///
            /// Spans hold `u, v` modulo the first factor and `t` modulo the
            /// second; `D = (f*u + i*f*v)/2 - f*t` is the high part. The outputs
            /// are `f*u - D`, `f*v`, and `D`. `ACTIVE` false admits `D = 0`,
            /// which reads and writes only the first two spans.
            ///
            /// # Safety
            /// The kernel contract holds for three spans below `2*p`. `twiddles`
            /// covers `i` and `1/2`, and `factor` is centered below `p/2`.
            $(#[$attribute])*
            pub unsafe fn recombine<const ACTIVE: bool>(
                values: *mut f64,
                twiddles: *const f64,
                span: usize,
                columns: $crate::arch::Columns,
                factor: f64,
                field: $crate::FloatField,
            ) {
                // SAFETY: complete vectors stay inside the three disjoint spans.
                // Each input times the centered factor is below p^2, which bounds
                // its product by 15*p/16. So is f*v times the centered root, so
                // the sum stays below 15*p/8 and its centered half has a product
                // below 15*p/16. D before reduction stays below 15*p/8, and both
                // reductions return magnitudes below 9*p/16. Every stored value
                // therefore lies in (-p, p), the range of canonical conversion.
                unsafe {
                    let (start, end) = columns.complete::<$lanes>();
                    let modulus = ($splat)(field.p());
                    let reciprocal = ($splat)(field.pinv());
                    let multiplier = ($splat)(factor);
                    let root = ($splat)(*twiddles);
                    let half = ($splat)(*twiddles.add(1));
                    let middle = values.add(span);
                    let high = middle.add(span);
                    let mut index = start;
                    loop {
                        let multiply = |value, other| ($multiply)(value, other, modulus, reciprocal);
                        let reduce = |value| ($reduce)(value, modulus, reciprocal);
                        let u = multiply(($load)(values.add(index)), multiplier);
                        let v = multiply(($load)(middle.add(index)), multiplier);
                        let low = if ACTIVE {
                            let t = multiply(($load)(high.add(index)), multiplier);
                            let sum = ($add)(u, multiply(v, root));
                            let d = reduce(($sub)(multiply(sum, half), t));
                            ($store)(high.add(index), ($canonical)(d, modulus));
                            reduce(($sub)(u, d))
                        } else {
                            u
                        };
                        ($store)(values.add(index), ($canonical)(low, modulus));
                        ($store)(middle.add(index), ($canonical)(v, modulus));
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
