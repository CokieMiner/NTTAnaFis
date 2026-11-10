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

            /// Recovers coefficients from residues modulo `x^(4*M)-1` and
            /// `x^M-w`, and stores them as canonical bits.
            ///
            /// # Safety
            /// The kernel contract holds for five spans below `2*p`. `twiddles`
            /// covers `i`, `w`, `1/2`, and `2*factor mod p`, all centered below
            /// `p/2`.
            $(#[$attribute])*
            pub unsafe fn recombine5<const ACTIVE: bool>(
                values: *mut f64,
                twiddles: *const f64,
                span: usize,
                columns: $crate::arch::Columns,
                factor: f64,
                field: $crate::FloatField,
            ) {
                // SAFETY: complete vectors stay inside the five disjoint spans.
                // Scaled inputs and root products stay below 15*p/16, `s02` and
                // `s13` below 15*p/8, and `sum` below 45*p/16, so its product
                // with `half` is below 45*p^2/32 < 2*p^2. Both reductions return
                // magnitudes below 9*p/16, placing every stored value in (-p, p).
                unsafe {
                    let (start, end) = columns.complete::<$lanes>();
                    let modulus = ($splat)(field.p());
                    let reciprocal = ($splat)(field.pinv());
                    let multiplier = ($splat)(factor);
                    let w1 = ($splat)(*twiddles);
                    let w2 = ($splat)(*twiddles.add(1));
                    let half = ($splat)(*twiddles.add(2));
                    let twice = ($splat)(*twiddles.add(3));
                    let (s1, s2) = (values.add(span), values.add(span.unchecked_mul(2)));
                    let (s3, s4) = (s2.add(span), s2.add(span.unchecked_mul(2)));
                    let mut index = start;
                    loop {
                        let multiply = |value, other| ($multiply)(value, other, modulus, reciprocal);
                        let reduce = |value| ($reduce)(value, modulus, reciprocal);
                        let u0 = multiply(($load)(values.add(index)), multiplier);
                        let u1 = multiply(($load)(s1.add(index)), multiplier);
                        let u2 = multiply(($load)(s2.add(index)), multiplier);
                        let u3 = multiply(($load)(s3.add(index)), multiplier);
                        let low = if ACTIVE {
                            let t = multiply(($load)(s4.add(index)), twice);
                            let s02 = ($add)(u0, multiply(u2, w1));
                            let s13 = ($add)(u1, multiply(u3, w1));
                            let sum = ($add)(s02, multiply(s13, w2));
                            let d = reduce(($sub)(multiply(sum, half), t));
                            ($store)(s4.add(index), ($canonical)(d, modulus));
                            reduce(($sub)(u0, d))
                        } else {
                            u0
                        };
                        ($store)(values.add(index), ($canonical)(low, modulus));
                        ($store)(s1.add(index), ($canonical)(u1, modulus));
                        ($store)(s2.add(index), ($canonical)(u2, modulus));
                        ($store)(s3.add(index), ($canonical)(u3, modulus));
                        index = index.unchecked_add($lanes);
                        if index == end {
                            break;
                        }
                    }
                }
            }

            /// Recovers coefficients from residues modulo `x^(4*M)-1`,
            /// `x^(2*M)-i`, and `x^M-w3`, and stores them as canonical bits.
            ///
            /// # Safety
            /// The kernel contract holds for seven spans below `2*p`. `twiddles`
            /// covers `i`, `w3`, `1/2`, `-i/2`, and `2*factor mod p`, all
            /// centered below `p/2`.
            $(#[$attribute])*
            pub unsafe fn recombine7<const ACTIVE: bool>(
                values: *mut f64,
                twiddles: *const f64,
                span: usize,
                columns: $crate::arch::Columns,
                factor: f64,
                field: $crate::FloatField,
            ) {
                // SAFETY: complete vectors stay inside the seven disjoint spans.
                // Scaled inputs and root products stay below 15*p/16, `d4` below
                // 15*p/8, and `conj` below 9*p/16, so `d4 ± conj` stays
                // below 39*p/16 and its products with `half` and `neg_half_i`
                // lie below 39*p^2/32 < 2*p^2, returning magnitudes below
                // 15*p/16 < p. Every stored value lies in (-p, p).
                unsafe {
                    let (start, end) = columns.complete::<$lanes>();
                    let modulus = ($splat)(field.p());
                    let reciprocal = ($splat)(field.pinv());
                    let multiplier = ($splat)(factor);
                    let w1 = ($splat)(*twiddles);
                    let w3 = ($splat)(*twiddles.add(1));
                    let half = ($splat)(*twiddles.add(2));
                    let neg_half_i = ($splat)(*twiddles.add(3));
                    let twice = ($splat)(*twiddles.add(4));
                    let s1 = values.add(span);
                    let s2 = s1.add(span);
                    let s3 = s2.add(span);
                    let s4 = s3.add(span);
                    let s5 = s4.add(span);
                    let s6 = s5.add(span);
                    let mut index = start;
                    loop {
                        let multiply = |value, other| ($multiply)(value, other, modulus, reciprocal);
                        let reduce = |value| ($reduce)(value, modulus, reciprocal);
                        let u0 = multiply(($load)(values.add(index)), multiplier);
                        let u1 = multiply(($load)(s1.add(index)), multiplier);
                        let u2 = multiply(($load)(s2.add(index)), multiplier);
                        let u3 = multiply(($load)(s3.add(index)), multiplier);
                        let t0 = multiply(($load)(s4.add(index)), multiplier);
                        let t1 = multiply(($load)(s5.add(index)), multiplier);
                        let u2_i = multiply(u2, w1);
                        let u3_i = multiply(u3, w1);
                        let c5 = reduce(($sub)(multiply(($add)(u1, u3_i), half), t1));
                        let c1 = reduce(($sub)(u1, c5));
                        let d4 = ($sub)(multiply(($add)(u0, u2_i), half), t0);
                        let (c4, c2) = if ACTIVE {
                            let t2 = multiply(($load)(s6.add(index)), twice);
                            let sum6 = ($add)(($sub)(u0, u2_i), multiply(($sub)(u1, u3_i), w3));
                            let q6 = ($sub)(multiply(sum6, half), t2);
                            let conj = reduce(($sub)(q6, multiply(c5, w3)));
                            let low4 = multiply(($add)(d4, conj), half);
                            let c6 = multiply(($sub)(d4, conj), neg_half_i);
                            ($store)(s6.add(index), ($canonical)(c6, modulus));
                            (low4, reduce(($sub)(u2, c6)))
                        } else {
                            (reduce(d4), u2)
                        };
                        let c0 = reduce(($sub)(u0, c4));
                        ($store)(values.add(index), ($canonical)(c0, modulus));
                        ($store)(s1.add(index), ($canonical)(c1, modulus));
                        ($store)(s2.add(index), ($canonical)(c2, modulus));
                        ($store)(s3.add(index), ($canonical)(u3, modulus));
                        ($store)(s4.add(index), ($canonical)(c4, modulus));
                        ($store)(s5.add(index), ($canonical)(c5, modulus));
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
