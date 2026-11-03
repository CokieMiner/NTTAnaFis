//! The complete schedule of one backend, and its radix-2 and radix-4 stages.

/// Instantiates every kernel of the schedule for a backend.
///
/// Leaves are separate, because their register shapes differ between backends.
///
/// | Argument | Meaning |
/// | :--- | :--- |
/// | `$name` | type that owns the kernels |
/// | `$vector` | register type, `f64` for the portable kernels |
/// | `$lanes` | floats per register, which must divide `COLUMN_ALIGN` |
/// | `[$attribute]` | attributes of every kernel, such as its target features |
/// | `$load`, `$store` | `(pointer) -> register`, `(pointer, register)`, unaligned |
/// | `$splat` | `(float) -> register`, the same value in every lane |
/// | `$add`, `$sub`, `$mul` | lane operations, each rounded to nearest |
/// | `$reduce` | `(a, p, pinv)`, the exact residue of `a` below `8*p` |
/// | `$multiply` | `(a, b, p, pinv)`, the exact residue of `a*b` below `4*p^2` |
/// | `$canonical` | `(a, p)`, canonical bits of an integer in `(-p, p)` |
macro_rules! schedule_backend {
    ($name:ident, $vector:ty, $lanes:literal, [$($attribute:meta),*],
     $load:expr, $store:expr, $splat:expr, $add:expr, $sub:expr, $mul:expr,
     $reduce:expr, $multiply:expr, $canonical:expr) => {
        stage_backend!(
            $name, $vector, $lanes, [$($attribute),*],
            $load, $store, $splat, $add, $sub,
            $reduce, $multiply, $canonical
        );
        finish_backend!(
            $name, $lanes, [$($attribute),*],
            $load, $store, $splat, $add, $sub,
            $reduce, $multiply, $canonical
        );
        root8_backend!(
            $name, $lanes, [$($attribute),*],
            $load, $store, $splat, $add, $sub,
            $reduce, $multiply
        );
        tail_backend!(
            $name, $lanes, [$($attribute),*],
            $load, $store, $splat, $add, $sub,
            $reduce, $multiply
        );
        packing_backend!(
            $name, $vector, $lanes, [$($attribute),*],
            $load, $store, $splat, $add, $mul,
            $reduce, $multiply
        );
    };
}

macro_rules! stage_backend {
    ($name:ident, $vector:ty, $lanes:literal, [$($attribute:meta),*],
     $load:expr, $store:expr, $splat:expr, $add:expr, $sub:expr,
     $reduce:expr, $multiply:expr, $canonical:expr) => {
        impl $name {
            /// Executes a unit radix-2 stage over two halves.
            ///
            /// # Safety
            /// The kernel contract holds with residues below `2*p`. `ZERO` omits
            /// reads of the upper half, which need not be initialized, and
            /// preserves the lower half. The root pointer is unused.
            $(#[$attribute])*
            pub unsafe fn radix2<const ZERO: bool>(
                values: *mut f64,
                _twiddles: *const f64,
                half: usize,
                columns: $crate::arch::Columns,
                field: $crate::FloatField,
            ) {
                // SAFETY: complete vectors stay in the columns of both disjoint
                // halves. Sums remain exact below 4*p < 2^52, and reductions give
                // magnitudes below p. The omitted upper half is written without
                // being read.
                unsafe {
                    let (start, end) = columns.complete::<$lanes>();
                    let modulus = ($splat)(field.p());
                    let reciprocal = ($splat)(field.pinv());
                    let mut index = start;
                    loop {
                        let first = ($load)(values.add(index));
                        let upper = values.add(half.unchecked_add(index));
                        let last = if ZERO {
                            first
                        } else {
                            let second = ($load)(upper);
                            ($store)(
                                values.add(index),
                                ($reduce)(($add)(first, second), modulus, reciprocal),
                            );
                            ($reduce)(($sub)(first, second), modulus, reciprocal)
                        };
                        ($store)(upper, last);
                        index = index.unchecked_add($lanes);
                        if index == end {
                            break;
                        }
                    }
                }
            }

            /// Evaluates a unit radix-4 stage from one, two, or three input quarters.
            ///
            /// # Safety
            /// The kernel contract holds with the first `INPUTS` quarters packed;
            /// omitted quarters represent zero and need not be initialized. The
            /// first two roots are one, and the third is read with two or more
            /// inputs.
            $(#[$attribute])*
            pub unsafe fn radix4_prefix<const INPUTS: usize>(
                values: *mut f64,
                twiddles: *const f64,
                quarter: usize,
                columns: $crate::arch::Columns,
                field: $crate::FloatField,
            ) {
                const { assert!(matches!(INPUTS, 1..=3), "radix-4 prefix arity") };
                // SAFETY: complete vectors fit all four output quarters. The
                // first INPUTS quarters are initialized, below p in magnitude.
                // The centered root product is below p^2/2, so its residue is
                // below 5*p/8. Outputs remain below 3*p, the input bound of the
                // next forward stage. Omitted inputs are never read.
                unsafe {
                    let (start, end) = columns.complete::<$lanes>();
                    let modulus = ($splat)(field.p());
                    let reciprocal = ($splat)(field.pinv());
                    let root = if INPUTS == 1 {
                        ($splat)(0.0)
                    } else {
                        ($splat)(*twiddles.add(2))
                    };
                    let q2 = quarter.unchecked_mul(2);
                    let q3 = quarter.unchecked_mul(3);
                    let mut index = start;
                    loop {
                        let a = ($load)(values.add(index));
                        let (second, third, fourth) = if INPUTS == 1 {
                            (a, a, a)
                        } else {
                            let b = ($load)(values.add(quarter.unchecked_add(index)));
                            let (sum, difference) = if INPUTS == 3 {
                                let c = ($load)(values.add(q2.unchecked_add(index)));
                                (($add)(a, c), ($sub)(a, c))
                            } else {
                                (a, a)
                            };
                            let rotated = ($multiply)(b, root, modulus, reciprocal);
                            ($store)(values.add(index), ($add)(sum, b));
                            (
                                ($sub)(sum, b),
                                ($add)(difference, rotated),
                                ($sub)(difference, rotated),
                            )
                        };
                        ($store)(values.add(quarter.unchecked_add(index)), second);
                        ($store)(values.add(q2.unchecked_add(index)), third);
                        ($store)(values.add(q3.unchecked_add(index)), fourth);
                        index = index.unchecked_add($lanes);
                        if index == end {
                            break;
                        }
                    }
                }
            }

            /// Executes radix-4 butterflies over four quarter spans.
            ///
            /// # Safety
            /// The kernel contract holds with residues below `3*p`, or `2*p` for
            /// the inverse. Three roots are readable, and `UNIT` requires the
            /// first two to be one.
            $(#[$attribute])*
            pub unsafe fn radix4<const INVERSE: bool, const UNIT: bool>(
                values: *mut f64,
                twiddles: *const f64,
                quarter: usize,
                columns: $crate::arch::Columns,
                field: $crate::FloatField,
            ) {
                // SAFETY: each vector iteration stays inside the four disjoint
                // quarters. Three constant roots are broadcast once.
                unsafe {
                    let (start, end) = columns.complete::<$lanes>();
                    let modulus = ($splat)(field.p());
                    let reciprocal = ($splat)(field.pinv());
                    let roots = if UNIT {
                        (($splat)(1.0), ($splat)(1.0), ($splat)(*twiddles.add(2)))
                    } else {
                        (
                            ($splat)(*twiddles),
                            ($splat)(*twiddles.add(1)),
                            ($splat)(*twiddles.add(2)),
                        )
                    };
                    let q2 = quarter.unchecked_mul(2);
                    let q3 = quarter.unchecked_mul(3);
                    let mut index = start;
                    loop {
                        let a = ($load)(values.add(index));
                        let b = ($load)(values.add(quarter.unchecked_add(index)));
                        let c = ($load)(values.add(q2.unchecked_add(index)));
                        let d = ($load)(values.add(q3.unchecked_add(index)));
                        let (y0, y1, y2, y3) = butterfly::<INVERSE, UNIT, false>(
                            (a, b, c, d),
                            roots,
                            modulus,
                            reciprocal,
                        );
                        ($store)(values.add(index), y0);
                        ($store)(values.add(quarter.unchecked_add(index)), y1);
                        ($store)(values.add(q2.unchecked_add(index)), y2);
                        ($store)(values.add(q3.unchecked_add(index)), y3);
                        index = index.unchecked_add($lanes);
                        if index == end {
                            break;
                        }
                    }
                }
            }

            /// Multiplies or squares spectra in place.
            ///
            /// # Safety
            /// The kernel contract holds for `left`, and for a disjoint `right`
            /// unless `SQUARE`, which ignores it. Magnitudes are below `2*p`.
            $(#[$attribute])*
            pub unsafe fn pointwise<const SQUARE: bool>(
                left: *mut f64,
                right: *const f64,
                columns: $crate::arch::Columns,
                field: $crate::FloatField,
            ) {
                // SAFETY: complete vectors stay inside the columns, and the
                // product is below 4*p^2.
                unsafe {
                    let (start, end) = columns.complete::<$lanes>();
                    let modulus = ($splat)(field.p());
                    let reciprocal = ($splat)(field.pinv());
                    let mut index = start;
                    loop {
                        let a = ($load)(left.add(index));
                        let b = if SQUARE { a } else { ($load)(right.add(index)) };
                        ($store)(left.add(index), ($multiply)(a, b, modulus, reciprocal));
                        index = index.unchecked_add($lanes);
                        if index == end {
                            break;
                        }
                    }
                }
            }

            /// Normalizes residues and stores canonical bits.
            ///
            /// # Safety
            /// The kernel contract holds with magnitudes below `2*p`, and `factor`
            /// lies in `0..p`.
            $(#[$attribute])*
            pub unsafe fn residues(
                values: *mut f64,
                columns: $crate::arch::Columns,
                factor: f64,
                field: $crate::FloatField,
            ) {
                // SAFETY: vector accesses remain in the columns. abs(a*factor) <
                // 2*p^2 gives an exact product residue in (-p, p), the input of
                // canonical conversion.
                unsafe {
                    let (start, end) = columns.complete::<$lanes>();
                    let modulus = ($splat)(field.p());
                    let reciprocal = ($splat)(field.pinv());
                    let multiplier = ($splat)(factor);
                    let mut index = start;
                    loop {
                        let a = ($load)(values.add(index));
                        let scaled = ($multiply)(a, multiplier, modulus, reciprocal);
                        ($store)(values.add(index), ($canonical)(scaled, modulus));
                        index = index.unchecked_add($lanes);
                        if index == end {
                            break;
                        }
                    }
                }
            }

            /// Stores the centered product of a source span and one factor.
            ///
            /// # Safety
            /// The kernel contract holds for the disjoint `source` and `target`,
            /// and sources and `factor` are centered below `p/2`.
            $(#[$attribute])*
            pub unsafe fn scale(
                target: *mut f64,
                source: *const f64,
                columns: $crate::arch::Columns,
                factor: f64,
                field: $crate::FloatField,
            ) {
                // SAFETY: vector accesses remain in both disjoint spans. The
                // product residue is below 5*p/8, whose reduction is the unique
                // centered residue of magnitude below p/2.
                unsafe {
                    let (start, end) = columns.complete::<$lanes>();
                    let modulus = ($splat)(field.p());
                    let reciprocal = ($splat)(field.pinv());
                    let multiplier = ($splat)(factor);
                    let mut index = start;
                    loop {
                        let product =
                            ($multiply)(($load)(source.add(index)), multiplier, modulus, reciprocal);
                        ($store)(target.add(index), ($reduce)(product, modulus, reciprocal));
                        index = index.unchecked_add($lanes);
                        if index == end {
                            break;
                        }
                    }
                }
            }
        }

        /// The radix-4 butterfly of a node.
        ///
        /// Reduction modulo `x^(4q)-w^2` splits into the factors `x^(2q)-w` and
        /// `x^(2q)+w`, then into `x^q-left` and `x^q-right`. `UNIT` identifies the
        /// node `w = left = 1`. `LAST` selects the two forward reductions that
        /// bound spectra for pointwise products. The contract derives the bounds.
        // Always inlined into the kernels. Without a target-feature attribute of
        // its own, the inlined body compiles with the features of the kernel.
        #[inline(always)]
        #[allow(
            unused_unsafe,
            reason = "Register operations are unsafe on vector units and safe on the portable kernels"
        )]
        unsafe fn butterfly<const INVERSE: bool, const UNIT: bool, const LAST: bool>(
            (a, b, c, d): ($vector, $vector, $vector, $vector),
            (w, left, right): ($vector, $vector, $vector),
            p: $vector,
            pinv: $vector,
        ) -> ($vector, $vector, $vector, $vector) {
            // SAFETY: callers are kernels of this backend, whose target features
            // admit every operation of the inlined body.
            unsafe {
                if INVERSE {
                    if UNIT {
                        let u = ($reduce)(($add)(a, b), p, pinv);
                        let s = ($reduce)(($add)(c, d), p, pinv);
                        let v = ($reduce)(($sub)(a, b), p, pinv);
                        let t = ($multiply)(($sub)(c, d), right, p, pinv);
                        (($add)(u, s), ($add)(v, t), ($sub)(u, s), ($sub)(v, t))
                    } else {
                        let u = ($add)(a, b);
                        let s = ($add)(c, d);
                        let v = ($multiply)(($sub)(a, b), left, p, pinv);
                        let t = ($multiply)(($sub)(c, d), right, p, pinv);
                        (
                            ($reduce)(($add)(u, s), p, pinv),
                            ($add)(v, t),
                            ($multiply)(($sub)(u, s), w, p, pinv),
                            ($multiply)(($sub)(v, t), w, p, pinv),
                        )
                    }
                } else if UNIT {
                    let u = ($reduce)(($add)(a, c), p, pinv);
                    let s = ($reduce)(($sub)(a, c), p, pinv);
                    let v = ($reduce)(($add)(b, d), p, pinv);
                    let t = ($multiply)(($sub)(b, d), right, p, pinv);
                    (($add)(u, v), ($sub)(u, v), ($add)(s, t), ($sub)(s, t))
                } else {
                    let c1 = ($multiply)(c, w, p, pinv);
                    let d1 = ($multiply)(d, w, p, pinv);
                    let (u, s) = if LAST {
                        (
                            ($reduce)(($add)(a, c1), p, pinv),
                            ($reduce)(($sub)(a, c1), p, pinv),
                        )
                    } else {
                        let centered = ($reduce)(a, p, pinv);
                        (($add)(centered, c1), ($sub)(centered, c1))
                    };
                    let v = ($multiply)(($add)(b, d1), left, p, pinv);
                    let t = ($multiply)(($sub)(b, d1), right, p, pinv);
                    (($add)(u, v), ($sub)(u, v), ($add)(s, t), ($sub)(s, t))
                }
            }
        }
    };
}
