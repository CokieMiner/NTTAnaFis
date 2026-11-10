//! Unit inverse roots with register-resident scaling and canonical CRT stores.

/// Scales a residue, converts it to canonical bits, and stores it.
macro_rules! asm_write {
    ($isa:ident, $v:tt, $row:tt) => {
        asm_lines!(
            asm_multiply!($isa, $v, $v, 15),
            $isa!(canonical $v),
            $isa!(store $v, $row, 0),
        )
    };
}

macro_rules! assembly_finish {
    ($name:ident, $isa:ident, $run:ident, $lanes:literal, $pointer:ident) => {
        impl $name {
            /// Completes a unit inverse root and stores only its active spans.
            ///
            /// # Safety
            /// The kernel contract holds for SPANS spans below `2*p`. The three
            /// roots use the unit radix-4 layout, or W1..W3 for SPANS=8. `factor`
            /// is centered below `p/2`. The first OUTPUTS spans receive canonical
            /// bits.
            #[expect(
                clippy::too_many_lines,
                reason = "One assembly block holds the complete kernel"
            )]
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
                // SAFETY: each complete vector lies inside the selected
                // columns of all disjoint spans. Roots and source residues
                // have the admitted bounds. Four-span outputs are below 2*p;
                // two/eight-span outputs are below 4*p. Multiplication by a
                // centered factor therefore has magnitude below 2*p^2, whose
                // exact remainder lies in (-p,p). Canonical conversion is valid
                // below p < 2^50. Every source vector is loaded before its
                // columns receive integer bit patterns. Rows at or beyond SPANS
                // are null and are neither dereferenced nor advanced.
                unsafe {
                    let (start, end) = columns.complete::<$lanes>();
                    let (w0, w1, w2) = if SPANS == 2 {
                        (0.0, 0.0, 0.0)
                    } else {
                        (*twiddles, *twiddles.add(1), *twiddles.add(2))
                    };
                    let constants = [
                        field.p(),
                        field.pinv(),
                        factor,
                        $crate::arch::BIAS,
                        w0,
                        w1,
                        w2,
                    ];
                    let row = |index: usize| {
                        if index < SPANS {
                            values.add(index.unchecked_mul(stride).unchecked_add(start))
                        } else {
                            core::ptr::null_mut()
                        }
                    };
                    $run!(
                        $isa!(enter),
                        $isa!(splat 22, "{c}", 0),
                        $isa!(splat 23, "{c}", 8),
                        $isa!(splat 15, "{c}", 16),
                        $isa!(splat 16, "{c}", 24),
                        $isa!(splat 12, "{c}", 32),
                        $isa!(splat 13, "{c}", 40),
                        $isa!(splat 14, "{c}", 48),
                        $isa!(zero 17),
                        "2:",
                        $isa!(load 0, "{r0}", 0),
                        $isa!(load 1, "{r1}", 0),
                        ".if {spans} == 2",
                        ".if {outputs} > 1",
                        $isa!(sub 2, 0, 1),
                        asm_write!($isa, 2, "{r1}"),
                        ".endif",
                        $isa!(add 0, 0, 1),
                        asm_write!($isa, 0, "{r0}"),
                        ".elseif {spans} == 4",
                        $isa!(load 2, "{r2}", 0),
                        $isa!(load 3, "{r3}", 0),
                        asm_dit!($isa, "1", 0, 1, 2, 3, 8, 9, 14, 14, 14),
                        asm_write!($isa, 1, "{r0}"),
                        ".if {outputs} > 1",
                        asm_write!($isa, 3, "{r1}"),
                        ".endif",
                        ".if {outputs} > 2",
                        asm_write!($isa, 8, "{r2}"),
                        ".endif",
                        ".if {outputs} > 3",
                        asm_write!($isa, 0, "{r3}"),
                        ".endif",
                        ".else",
                        // Left children keep y in 1, 3, 8, 0 and right
                        // children in 5, 7, 10, 4.
                        $isa!(load 2, "{r2}", 0),
                        $isa!(load 3, "{r3}", 0),
                        $isa!(load 4, "{r4}", 0),
                        $isa!(load 5, "{r5}", 0),
                        $isa!(load 6, "{r6}", 0),
                        $isa!(load 7, "{r7}", 0),
                        asm_dit!($isa, "1", 0, 1, 2, 3, 8, 9, 12, 12, 12),
                        asm_dit!($isa, "0", 4, 5, 6, 7, 10, 11, 12, 13, 14),
                        ".if {outputs} > 4",
                        $isa!(sub 2, 1, 5),
                        asm_write!($isa, 2, "{r4}"),
                        ".endif",
                        $isa!(add 1, 1, 5),
                        asm_write!($isa, 1, "{r0}"),
                        ".if {outputs} > 5",
                        $isa!(sub 2, 3, 7),
                        asm_write!($isa, 2, "{r5}"),
                        ".endif",
                        ".if {outputs} > 1",
                        $isa!(add 3, 3, 7),
                        asm_write!($isa, 3, "{r1}"),
                        ".endif",
                        ".if {outputs} > 6",
                        $isa!(sub 2, 8, 10),
                        asm_write!($isa, 2, "{r6}"),
                        ".endif",
                        ".if {outputs} > 2",
                        $isa!(add 8, 8, 10),
                        asm_write!($isa, 8, "{r2}"),
                        ".endif",
                        ".if {outputs} > 7",
                        $isa!(sub 2, 0, 4),
                        asm_write!($isa, 2, "{r7}"),
                        ".endif",
                        ".if {outputs} > 3",
                        $isa!(add 0, 0, 4),
                        asm_write!($isa, 0, "{r3}"),
                        ".endif",
                        ".endif",
                        $isa!(step "{r0}"),
                        $isa!(step "{r1}"),
                        ".if {spans} > 2",
                        $isa!(step "{r2}"),
                        $isa!(step "{r3}"),
                        ".endif",
                        ".if {spans} > 4",
                        $isa!(step "{r4}"),
                        $isa!(step "{r5}"),
                        $isa!(step "{r6}"),
                        $isa!(step "{r7}"),
                        ".endif",
                        $isa!(next "{n}"),
                        $isa!(leave),
                        spans = const SPANS,
                        outputs = const OUTPUTS,
                        c = in($pointer) constants.as_ptr(),
                        r0 = inout($pointer) row(0) => _,
                        r1 = inout($pointer) row(1) => _,
                        r2 = inout($pointer) row(2) => _,
                        r3 = inout($pointer) row(3) => _,
                        r4 = inout($pointer) row(4) => _,
                        r5 = inout($pointer) row(5) => _,
                        r6 = inout($pointer) row(6) => _,
                        r7 = inout($pointer) row(7) => _,
                        n = inout($pointer) end.unchecked_sub(start) => _
                    );
                }
            }

            /// Recovers coefficients from residues modulo `x^(2*M)-1` and
            /// `x^M-i`, and stores them as canonical bits.
            ///
            /// # Safety
            /// The kernel contract holds for three spans below `2*p`. `twiddles`
            /// covers `i` and `1/2`, and `factor` is centered below `p/2`.
            pub unsafe fn recombine<const ACTIVE: bool>(
                values: *mut f64,
                twiddles: *const f64,
                span: usize,
                columns: $crate::arch::Columns,
                factor: f64,
                field: $crate::FloatField,
            ) {
                // SAFETY: the block runs complete vectors inside the three
                // disjoint spans and performs the arithmetic of the shared
                // schedule, whose proof places every stored value in (-p, p).
                // Without ACTIVE the third span is neither read nor advanced.
                unsafe {
                    let (start, end) = columns.complete::<$lanes>();
                    let constants = [
                        field.p(),
                        field.pinv(),
                        factor,
                        $crate::arch::BIAS,
                        *twiddles,
                        *twiddles.add(1),
                    ];
                    $run!(
                        $isa!(enter),
                        $isa!(splat 22, "{c}", 0),
                        $isa!(splat 23, "{c}", 8),
                        $isa!(splat 15, "{c}", 16),
                        $isa!(splat 16, "{c}", 24),
                        $isa!(splat 12, "{c}", 32),
                        $isa!(splat 13, "{c}", 40),
                        $isa!(zero 17),
                        "2:",
                        $isa!(load 0, "{r0}", 0),
                        $isa!(load 1, "{r1}", 0),
                        asm_multiply!($isa, 0, 0, 15),
                        asm_multiply!($isa, 1, 1, 15),
                        ".if {active}",
                        $isa!(load 2, "{r2}", 0),
                        asm_multiply!($isa, 2, 2, 15),
                        asm_multiply!($isa, 3, 1, 12),
                        $isa!(add 3, 0, 3),
                        asm_multiply!($isa, 3, 3, 13),
                        $isa!(sub 3, 3, 2),
                        asm_reduce!($isa, 3),
                        $isa!(sub 0, 0, 3),
                        asm_reduce!($isa, 0),
                        $isa!(canonical 3),
                        $isa!(store 3, "{r2}", 0),
                        $isa!(step "{r2}"),
                        ".endif",
                        $isa!(canonical 0),
                        $isa!(store 0, "{r0}", 0),
                        $isa!(canonical 1),
                        $isa!(store 1, "{r1}", 0),
                        $isa!(step "{r0}"),
                        $isa!(step "{r1}"),
                        $isa!(next "{n}"),
                        $isa!(leave),
                        active = const if ACTIVE { 1 } else { 0 },
                        c = in($pointer) constants.as_ptr(),
                        r0 = inout($pointer) values.add(start) => _,
                        r1 = inout($pointer) values.add(span.unchecked_add(start)) => _,
                        r2 = inout($pointer) values.add(span.unchecked_mul(2).unchecked_add(start)) => _,
                        n = inout($pointer) end.unchecked_sub(start) => _
                    );
                }
            }

            /// Recovers coefficients from residues modulo `x^(4*M)-1` and
            /// `x^M-w`, and stores them as canonical bits.
            ///
            /// # Safety
            /// The kernel contract holds for five spans below `2*p`. `twiddles`
            /// covers `i`, `w`, `1/2`, and `2*factor mod p`, all centered below
            /// `p/2`.
            pub unsafe fn recombine5<const ACTIVE: bool>(
                values: *mut f64,
                twiddles: *const f64,
                span: usize,
                columns: $crate::arch::Columns,
                factor: f64,
                field: $crate::FloatField,
            ) {
                // SAFETY: the block runs complete vectors inside the five
                // disjoint spans and performs the arithmetic of the shared
                // schedule, whose proof places every stored value in (-p, p).
                // Without ACTIVE the fifth span is neither read nor advanced.
                unsafe {
                    let (start, end) = columns.complete::<$lanes>();
                    let constants = [
                        field.p(),
                        field.pinv(),
                        factor,
                        $crate::arch::BIAS,
                        *twiddles,
                        *twiddles.add(1),
                        *twiddles.add(2),
                        *twiddles.add(3),
                    ];
                    let row = |index: usize| {
                        values.add(index.unchecked_mul(span).unchecked_add(start))
                    };
                    $run!(
                        $isa!(enter),
                        $isa!(splat 22, "{c}", 0),
                        $isa!(splat 23, "{c}", 8),
                        $isa!(splat 15, "{c}", 16),
                        $isa!(splat 16, "{c}", 24),
                        $isa!(splat 12, "{c}", 32),
                        $isa!(splat 13, "{c}", 40),
                        $isa!(splat 14, "{c}", 48),
                        $isa!(splat 11, "{c}", 56),
                        $isa!(zero 17),
                        "2:",
                        $isa!(load 0, "{r0}", 0),
                        $isa!(load 1, "{r1}", 0),
                        $isa!(load 2, "{r2}", 0),
                        $isa!(load 3, "{r3}", 0),
                        asm_multiply!($isa, 0, 0, 15),
                        asm_multiply!($isa, 1, 1, 15),
                        asm_multiply!($isa, 2, 2, 15),
                        asm_multiply!($isa, 3, 3, 15),
                        ".if {active}",
                        $isa!(load 4, "{r4}", 0),
                        asm_multiply!($isa, 4, 4, 11),
                        asm_multiply!($isa, 5, 2, 12),
                        $isa!(add 5, 0, 5),
                        asm_multiply!($isa, 6, 3, 12),
                        $isa!(add 6, 1, 6),
                        asm_multiply!($isa, 6, 6, 13),
                        $isa!(add 5, 5, 6),
                        asm_multiply!($isa, 5, 5, 14),
                        $isa!(sub 4, 5, 4),
                        asm_reduce!($isa, 4),
                        $isa!(sub 0, 0, 4),
                        asm_reduce!($isa, 0),
                        $isa!(canonical 4),
                        $isa!(store 4, "{r4}", 0),
                        $isa!(step "{r4}"),
                        ".endif",
                        $isa!(canonical 0),
                        $isa!(store 0, "{r0}", 0),
                        $isa!(canonical 1),
                        $isa!(store 1, "{r1}", 0),
                        $isa!(canonical 2),
                        $isa!(store 2, "{r2}", 0),
                        $isa!(canonical 3),
                        $isa!(store 3, "{r3}", 0),
                        $isa!(step "{r0}"),
                        $isa!(step "{r1}"),
                        $isa!(step "{r2}"),
                        $isa!(step "{r3}"),
                        $isa!(next "{n}"),
                        $isa!(leave),
                        active = const if ACTIVE { 1 } else { 0 },
                        c = in($pointer) constants.as_ptr(),
                        r0 = inout($pointer) row(0) => _,
                        r1 = inout($pointer) row(1) => _,
                        r2 = inout($pointer) row(2) => _,
                        r3 = inout($pointer) row(3) => _,
                        r4 = inout($pointer) row(4) => _,
                        n = inout($pointer) end.unchecked_sub(start) => _
                    );
                }
            }

            /// Recovers coefficients from residues modulo `x^(4*M)-1`,
            /// `x^(2*M)-i`, and `x^M-w3`, and stores them as canonical bits.
            ///
            /// # Safety
            /// The kernel contract holds for seven spans below `2*p`. `twiddles`
            /// covers `i`, `w3`, `1/2`, `-i/2`, and `2*factor mod p`, all
            /// centered below `p/2`.
            #[expect(
                clippy::too_many_lines,
                reason = "One assembly block holds the complete kernel"
            )]
            pub unsafe fn recombine7<const ACTIVE: bool>(
                values: *mut f64,
                twiddles: *const f64,
                span: usize,
                columns: $crate::arch::Columns,
                factor: f64,
                field: $crate::FloatField,
            ) {
                // SAFETY: the block runs complete vectors inside the seven
                // disjoint spans and performs the arithmetic of the shared
                // schedule, whose proof places every stored value in (-p, p).
                // Without ACTIVE the seventh span is neither read nor advanced.
                unsafe {
                    let (start, end) = columns.complete::<$lanes>();
                    let constants = [
                        field.p(),
                        field.pinv(),
                        factor,
                        $crate::arch::BIAS,
                        *twiddles,
                        *twiddles.add(1),
                        *twiddles.add(2),
                        *twiddles.add(3),
                        *twiddles.add(4),
                    ];
                    let row = |index: usize| {
                        values.add(index.unchecked_mul(span).unchecked_add(start))
                    };
                    $run!(
                        $isa!(enter),
                        $isa!(splat 22, "{c}", 0),
                        $isa!(splat 23, "{c}", 8),
                        $isa!(splat 15, "{c}", 16),
                        $isa!(splat 16, "{c}", 24),
                        $isa!(splat 12, "{c}", 32),
                        $isa!(splat 13, "{c}", 40),
                        $isa!(splat 14, "{c}", 48),
                        $isa!(splat 10, "{c}", 56),
                        $isa!(splat 11, "{c}", 64),
                        $isa!(zero 17),
                        "2:",
                        $isa!(load 0, "{r0}", 0),
                        $isa!(load 1, "{r1}", 0),
                        $isa!(load 2, "{r2}", 0),
                        $isa!(load 3, "{r3}", 0),
                        $isa!(load 4, "{r4}", 0),
                        $isa!(load 5, "{r5}", 0),
                        asm_multiply!($isa, 0, 0, 15),
                        asm_multiply!($isa, 1, 1, 15),
                        asm_multiply!($isa, 2, 2, 15),
                        asm_multiply!($isa, 3, 3, 15),
                        asm_multiply!($isa, 4, 4, 15),
                        asm_multiply!($isa, 5, 5, 15),
                        asm_multiply!($isa, 6, 2, 12),
                        asm_multiply!($isa, 7, 3, 12),
                        $isa!(canonical 3),
                        $isa!(store 3, "{r3}", 0),
                        $isa!(step "{r3}"),
                        $isa!(add 8, 1, 7),
                        asm_multiply!($isa, 8, 8, 14),
                        $isa!(sub 5, 8, 5),
                        asm_reduce!($isa, 5),
                        $isa!(sub 3, 1, 5),
                        asm_reduce!($isa, 3),
                        $isa!(canonical 3),
                        $isa!(store 3, "{r1}", 0),
                        $isa!(step "{r1}"),
                        $isa!(add 8, 0, 6),
                        asm_multiply!($isa, 8, 8, 14),
                        $isa!(sub 4, 8, 4),
                        ".if {active}",
                        $isa!(load 8, "{r6}", 0),
                        asm_multiply!($isa, 8, 8, 11),
                        $isa!(sub 6, 0, 6),
                        $isa!(sub 7, 1, 7),
                        asm_multiply!($isa, 7, 7, 13),
                        $isa!(add 6, 6, 7),
                        asm_multiply!($isa, 6, 6, 14),
                        $isa!(sub 6, 6, 8),
                        asm_multiply!($isa, 7, 5, 13),
                        $isa!(sub 6, 6, 7),
                        asm_reduce!($isa, 6),
                        $isa!(add 7, 4, 6),
                        $isa!(sub 6, 4, 6),
                        asm_multiply!($isa, 4, 7, 14),
                        asm_multiply!($isa, 6, 6, 10),
                        $isa!(sub 2, 2, 6),
                        asm_reduce!($isa, 2),
                        $isa!(canonical 6),
                        $isa!(store 6, "{r6}", 0),
                        $isa!(step "{r6}"),
                        ".else",
                        asm_reduce!($isa, 4),
                        ".endif",
                        $isa!(sub 0, 0, 4),
                        asm_reduce!($isa, 0),
                        $isa!(canonical 0),
                        $isa!(store 0, "{r0}", 0),
                        $isa!(canonical 2),
                        $isa!(store 2, "{r2}", 0),
                        $isa!(canonical 4),
                        $isa!(store 4, "{r4}", 0),
                        $isa!(canonical 5),
                        $isa!(store 5, "{r5}", 0),
                        $isa!(step "{r0}"),
                        $isa!(step "{r2}"),
                        $isa!(step "{r4}"),
                        $isa!(step "{r5}"),
                        $isa!(next "{n}"),
                        $isa!(leave),
                        active = const if ACTIVE { 1 } else { 0 },
                        c = in($pointer) constants.as_ptr(),
                        r0 = inout($pointer) row(0) => _,
                        r1 = inout($pointer) row(1) => _,
                        r2 = inout($pointer) row(2) => _,
                        r3 = inout($pointer) row(3) => _,
                        r4 = inout($pointer) row(4) => _,
                        r5 = inout($pointer) row(5) => _,
                        r6 = inout($pointer) row(6) => _,
                        n = inout($pointer) end.unchecked_sub(start) => _
                    );
                }
            }
        }
    };
}
