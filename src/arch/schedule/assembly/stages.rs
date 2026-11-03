//! Radix-2 and radix-4 stages, pointwise products, and residue normalization.

/// Instantiates a backend from an instruction table.
///
/// | Argument | Meaning |
/// | :--- | :--- |
/// | `$name` | type that owns the kernels |
/// | `$isa` | instruction table, described in the `assembly` module |
/// | `$run` | macro that runs a block with this unit's clobbers |
/// | `$lanes` | floats per register, which must divide `COLUMN_ALIGN` |
/// | `$pointer` | register class of an address operand |
/// | `$leaf` | type that owns the sixteen-point leaves |
macro_rules! assembly_backend {
    ($name:ident, $isa:ident, $run:ident, $lanes:literal, $pointer:ident, $leaf:ident) => {
        /// Inline-assembly residue kernels.
        #[derive(Debug)]
        pub struct $name;

        impl $name {
            /// Stages, leaves, and products of this vector unit.
            pub const KERNELS: $crate::arch::Kernels =
                kernel_table!($name, $leaf, None, None, None);

            /// Executes a unit radix-2 stage over two halves.
            ///
            /// # Safety
            /// The kernel contract holds with residues below `2*p`. `ZERO` omits
            /// reads of the upper half, which need not be initialized, and
            /// preserves the lower half. The root pointer is unused.
            pub unsafe fn radix2<const ZERO: bool>(
                values: *mut f64,
                _twiddles: *const f64,
                half: usize,
                columns: $crate::arch::Columns,
                field: $crate::FloatField,
            ) {
                // SAFETY: the block runs a positive multiple of the lane count
                // of columns inside both disjoint halves. Sums remain exact
                // below 4*p < 2^52. The omitted upper half is written without
                // being read. The clobbers cover every register the text writes.
                unsafe {
                    let (start, end) = columns.complete::<$lanes>();
                    let constants = [field.p(), field.pinv()];
                    $run!(
                        $isa!(enter),
                        $isa!(splat 22, "{c}", 0),
                        $isa!(splat 23, "{c}", 8),
                        "2:",
                        $isa!(load 0, "{r0}", 0),
                        ".if {zero}",
                        $isa!(store 0, "{r1}", 0),
                        ".else",
                        $isa!(load 1, "{r1}", 0),
                        $isa!(add 2, 0, 1),
                        $isa!(sub 0, 0, 1),
                        asm_reduce!($isa, 2),
                        asm_reduce!($isa, 0),
                        $isa!(store 2, "{r0}", 0),
                        $isa!(store 0, "{r1}", 0),
                        ".endif",
                        $isa!(step "{r0}"),
                        $isa!(step "{r1}"),
                        $isa!(next "{n}"),
                        $isa!(leave),
                        zero = const if ZERO { 1 } else { 0 },
                        c = in($pointer) constants.as_ptr(),
                        r0 = inout($pointer) values.add(start) => _,
                        r1 = inout($pointer) values.add(half.unchecked_add(start)) => _,
                        n = inout($pointer) end.unchecked_sub(start) => _
                    );
                }
            }

            /// Evaluates a unit radix-4 stage from one, two, or three input quarters.
            ///
            /// # Safety
            /// The kernel contract holds with the first `INPUTS` quarters packed;
            /// omitted quarters represent zero and need not be initialized. The
            /// first two roots are one, and the third is read with two or more
            /// inputs.
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
                // The centered root product has a residue below 5*p/8. Outputs
                // remain below 3*p. Omitted inputs are never read, and the root
                // is read only with two or more inputs.
                unsafe {
                    let (start, end) = columns.complete::<$lanes>();
                    let root = if INPUTS == 1 { 0.0 } else { *twiddles.add(2) };
                    let constants = [field.p(), field.pinv(), root];
                    let q2 = quarter.unchecked_mul(2);
                    let q3 = quarter.unchecked_mul(3);
                    $run!(
                        $isa!(enter),
                        $isa!(splat 22, "{c}", 0),
                        $isa!(splat 23, "{c}", 8),
                        $isa!(splat 14, "{c}", 16),
                        "2:",
                        $isa!(load 0, "{r0}", 0),
                        ".if {inputs} == 1",
                        $isa!(store 0, "{r1}", 0),
                        $isa!(store 0, "{r2}", 0),
                        $isa!(store 0, "{r3}", 0),
                        ".else",
                        $isa!(load 1, "{r1}", 0),
                        ".if {inputs} == 3",
                        $isa!(load 2, "{r2}", 0),
                        $isa!(add 3, 0, 2),
                        $isa!(sub 0, 0, 2),
                        $isa!(add 4, 3, 1),
                        $isa!(sub 5, 3, 1),
                        ".else",
                        $isa!(add 4, 0, 1),
                        $isa!(sub 5, 0, 1),
                        ".endif",
                        asm_multiply!($isa, 6, 1, 14),
                        $isa!(add 7, 0, 6),
                        $isa!(sub 0, 0, 6),
                        $isa!(store 4, "{r0}", 0),
                        $isa!(store 5, "{r1}", 0),
                        $isa!(store 7, "{r2}", 0),
                        $isa!(store 0, "{r3}", 0),
                        ".endif",
                        $isa!(step "{r0}"),
                        $isa!(step "{r1}"),
                        $isa!(step "{r2}"),
                        $isa!(step "{r3}"),
                        $isa!(next "{n}"),
                        $isa!(leave),
                        inputs = const INPUTS,
                        c = in($pointer) constants.as_ptr(),
                        r0 = inout($pointer) values.add(start) => _,
                        r1 = inout($pointer) values.add(quarter.unchecked_add(start)) => _,
                        r2 = inout($pointer) values.add(q2.unchecked_add(start)) => _,
                        r3 = inout($pointer) values.add(q3.unchecked_add(start)) => _,
                        n = inout($pointer) end.unchecked_sub(start) => _
                    );
                }
            }

            /// Executes radix-4 butterflies over four quarter spans.
            ///
            /// # Safety
            /// The kernel contract holds with residues below `3*p`, or `2*p` for
            /// the inverse. Three roots are readable, and `UNIT` requires the
            /// first two to be one.
            pub unsafe fn radix4<const INVERSE: bool, const UNIT: bool>(
                values: *mut f64,
                twiddles: *const f64,
                quarter: usize,
                columns: $crate::arch::Columns,
                field: $crate::FloatField,
            ) {
                // SAFETY: the block runs complete vectors inside the four
                // disjoint quarters. Three constant roots are broadcast once;
                // UNIT requires the first two to equal one, and they are not
                // read. Outputs satisfy the butterfly bounds of the contract.
                unsafe {
                    let (start, end) = columns.complete::<$lanes>();
                    let (first, second) = if UNIT {
                        (1.0, 1.0)
                    } else {
                        (*twiddles, *twiddles.add(1))
                    };
                    let constants = [field.p(), field.pinv(), first, second, *twiddles.add(2)];
                    let q2 = quarter.unchecked_mul(2);
                    let q3 = quarter.unchecked_mul(3);
                    $run!(
                        $isa!(enter),
                        $isa!(splat 22, "{c}", 0),
                        $isa!(splat 23, "{c}", 8),
                        $isa!(splat 12, "{c}", 16),
                        $isa!(splat 13, "{c}", 24),
                        $isa!(splat 14, "{c}", 32),
                        "2:",
                        $isa!(load 0, "{r0}", 0),
                        $isa!(load 1, "{r1}", 0),
                        $isa!(load 2, "{r2}", 0),
                        $isa!(load 3, "{r3}", 0),
                        ".if {inverse}",
                        asm_dit!($isa, "{unit}", 0, 1, 2, 3, 4, 5, 12, 13, 14),
                        $isa!(store 1, "{r0}", 0),
                        $isa!(store 3, "{r1}", 0),
                        $isa!(store 4, "{r2}", 0),
                        $isa!(store 0, "{r3}", 0),
                        ".else",
                        asm_dif!($isa, "{unit}", "0", 0, 1, 2, 3, 4, 5, 12, 13, 14),
                        $isa!(store 2, "{r0}", 0),
                        $isa!(store 4, "{r1}", 0),
                        $isa!(store 3, "{r2}", 0),
                        $isa!(store 0, "{r3}", 0),
                        ".endif",
                        $isa!(step "{r0}"),
                        $isa!(step "{r1}"),
                        $isa!(step "{r2}"),
                        $isa!(step "{r3}"),
                        $isa!(next "{n}"),
                        $isa!(leave),
                        inverse = const if INVERSE { 1 } else { 0 },
                        unit = const if UNIT { 1 } else { 0 },
                        c = in($pointer) constants.as_ptr(),
                        r0 = inout($pointer) values.add(start) => _,
                        r1 = inout($pointer) values.add(quarter.unchecked_add(start)) => _,
                        r2 = inout($pointer) values.add(q2.unchecked_add(start)) => _,
                        r3 = inout($pointer) values.add(q3.unchecked_add(start)) => _,
                        n = inout($pointer) end.unchecked_sub(start) => _
                    );
                }
            }

            /// Writes `a + w*b` for two packed input spans to a third span.
            ///
            /// # Safety
            /// The kernel contract holds for two packed spans and a third writable
            /// span that need not be initialized, with one root.
            pub unsafe fn fold(
                values: *mut f64,
                twiddles: *const f64,
                span: usize,
                columns: $crate::arch::Columns,
                field: $crate::FloatField,
            ) {
                // SAFETY: the block runs complete vectors inside the three
                // disjoint spans. The root product has a residue below 5*p/8, so
                // the output stays below 13*p/8, inside the forward bound of 3*p.
                unsafe {
                    let (start, end) = columns.complete::<$lanes>();
                    let constants = [field.p(), field.pinv(), *twiddles];
                    $run!(
                        $isa!(enter),
                        $isa!(splat 22, "{c}", 0),
                        $isa!(splat 23, "{c}", 8),
                        $isa!(splat 12, "{c}", 16),
                        "2:",
                        $isa!(load 0, "{r0}", 0),
                        $isa!(load 1, "{r1}", 0),
                        asm_multiply!($isa, 1, 1, 12),
                        $isa!(add 0, 0, 1),
                        $isa!(store 0, "{r2}", 0),
                        $isa!(step "{r0}"),
                        $isa!(step "{r1}"),
                        $isa!(step "{r2}"),
                        $isa!(next "{n}"),
                        $isa!(leave),
                        c = in($pointer) constants.as_ptr(),
                        r0 = inout($pointer) values.add(start) => _,
                        r1 = inout($pointer) values.add(span.unchecked_add(start)) => _,
                        r2 = inout($pointer) values.add(span.unchecked_mul(2).unchecked_add(start)) => _,
                        n = inout($pointer) end.unchecked_sub(start) => _
                    );
                }
            }

            /// Executes a radix-2 stage with one root.
            ///
            /// # Safety
            /// The kernel contract holds for two halves below `3*p`, or `2*p` for
            /// the inverse, with the node root or its inverse.
            pub unsafe fn radix2_node<const INVERSE: bool>(
                values: *mut f64,
                twiddles: *const f64,
                half: usize,
                columns: $crate::arch::Columns,
                field: $crate::FloatField,
            ) {
                // SAFETY: the block runs complete vectors inside both disjoint
                // halves and performs the arithmetic of the shared schedule,
                // whose proof gives the same output bounds.
                unsafe {
                    let (start, end) = columns.complete::<$lanes>();
                    let constants = [field.p(), field.pinv(), *twiddles];
                    $run!(
                        $isa!(enter),
                        $isa!(splat 22, "{c}", 0),
                        $isa!(splat 23, "{c}", 8),
                        $isa!(splat 12, "{c}", 16),
                        "2:",
                        $isa!(load 0, "{r0}", 0),
                        $isa!(load 1, "{r1}", 0),
                        ".if {inverse}",
                        $isa!(add 2, 0, 1),
                        $isa!(sub 1, 0, 1),
                        asm_reduce!($isa, 2),
                        asm_multiply!($isa, 1, 1, 12),
                        $isa!(store 2, "{r0}", 0),
                        $isa!(store 1, "{r1}", 0),
                        ".else",
                        asm_multiply!($isa, 1, 1, 12),
                        asm_reduce!($isa, 0),
                        $isa!(add 2, 0, 1),
                        $isa!(sub 0, 0, 1),
                        $isa!(store 2, "{r0}", 0),
                        $isa!(store 0, "{r1}", 0),
                        ".endif",
                        $isa!(step "{r0}"),
                        $isa!(step "{r1}"),
                        $isa!(next "{n}"),
                        $isa!(leave),
                        inverse = const if INVERSE { 1 } else { 0 },
                        c = in($pointer) constants.as_ptr(),
                        r0 = inout($pointer) values.add(start) => _,
                        r1 = inout($pointer) values.add(half.unchecked_add(start)) => _,
                        n = inout($pointer) end.unchecked_sub(start) => _
                    );
                }
            }

            /// Multiplies or squares spectra in place.
            ///
            /// # Safety
            /// The kernel contract holds for `left`, and for a disjoint `right`
            /// unless `SQUARE`, which ignores it. Magnitudes are below `2*p`.
            pub unsafe fn pointwise<const SQUARE: bool>(
                left: *mut f64,
                right: *const f64,
                columns: $crate::arch::Columns,
                field: $crate::FloatField,
            ) {
                // SAFETY: complete vectors stay inside the columns of both spans.
                // `right` is read only when SQUARE is false.
                unsafe {
                    let (start, end) = columns.complete::<$lanes>();
                    let constants = [field.p(), field.pinv()];
                    $run!(
                        $isa!(enter),
                        $isa!(splat 22, "{c}", 0),
                        $isa!(splat 23, "{c}", 8),
                        "2:",
                        $isa!(load 0, "{a}", 0),
                        ".if {square}",
                        asm_multiply!($isa, 0, 0, 0),
                        ".else",
                        $isa!(load 1, "{b}", 0),
                        asm_multiply!($isa, 0, 0, 1),
                        $isa!(step "{b}"),
                        ".endif",
                        $isa!(store 0, "{a}", 0),
                        $isa!(step "{a}"),
                        $isa!(next "{n}"),
                        $isa!(leave),
                        square = const if SQUARE { 1 } else { 0 },
                        c = in($pointer) constants.as_ptr(),
                        a = inout($pointer) left.add(start) => _,
                        b = inout($pointer) right.wrapping_add(start) => _,
                        n = inout($pointer) end.unchecked_sub(start) => _
                    );
                }
            }

            /// Normalizes residues and stores canonical bits.
            ///
            /// # Safety
            /// The kernel contract holds with magnitudes below `2*p`, and `factor`
            /// lies in `0..p`.
            pub unsafe fn residues(
                values: *mut f64,
                columns: $crate::arch::Columns,
                factor: f64,
                field: $crate::FloatField,
            ) {
                // SAFETY: vector accesses remain in the columns. abs(a*factor) <
                // 2*p^2 gives an exact product residue in (-p, p) by catalog
                // admission; canonical conversion maps it to [0, p) and exposes
                // its bits.
                unsafe {
                    let (start, end) = columns.complete::<$lanes>();
                    let constants = [field.p(), field.pinv(), factor, $crate::arch::BIAS];
                    $run!(
                        $isa!(enter),
                        $isa!(splat 22, "{c}", 0),
                        $isa!(splat 23, "{c}", 8),
                        $isa!(splat 15, "{c}", 16),
                        $isa!(splat 16, "{c}", 24),
                        $isa!(zero 17),
                        "2:",
                        $isa!(load 0, "{a}", 0),
                        asm_multiply!($isa, 0, 0, 15),
                        $isa!(canonical 0),
                        $isa!(store 0, "{a}", 0),
                        $isa!(step "{a}"),
                        $isa!(next "{n}"),
                        $isa!(leave),
                        c = in($pointer) constants.as_ptr(),
                        a = inout($pointer) values.add(start) => _,
                        n = inout($pointer) end.unchecked_sub(start) => _
                    );
                }
            }

            /// Stores the centered product of a source span and one factor.
            ///
            /// # Safety
            /// The kernel contract holds for the disjoint `source` and `target`,
            /// and sources and `factor` are centered below `p/2`.
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
                    let constants = [field.p(), field.pinv(), factor];
                    $run!(
                        $isa!(enter),
                        $isa!(splat 22, "{c}", 0),
                        $isa!(splat 23, "{c}", 8),
                        $isa!(splat 15, "{c}", 16),
                        "2:",
                        $isa!(load 0, "{s}", 0),
                        asm_multiply!($isa, 0, 0, 15),
                        asm_reduce!($isa, 0),
                        $isa!(store 0, "{t}", 0),
                        $isa!(step "{s}"),
                        $isa!(step "{t}"),
                        $isa!(next "{n}"),
                        $isa!(leave),
                        c = in($pointer) constants.as_ptr(),
                        t = inout($pointer) target.add(start) => _,
                        s = inout($pointer) source.add(start) => _,
                        n = inout($pointer) end.unchecked_sub(start) => _
                    );
                }
            }
        }

        assembly_finish!($name, $isa, $run, $lanes, $pointer);
        assembly_root!($name, $isa, $run, $lanes, $pointer);
        assembly_packing!($name, $isa, $run, $lanes, $pointer);
    };
}
