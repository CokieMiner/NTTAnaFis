//! Unit radix-2 and its radix-4 children evaluated in one coefficient pass.

macro_rules! assembly_root {
    ($name:ident, $isa:ident, $run:ident, $lanes:literal, $pointer:ident) => {
        impl $name {
            /// Fuses a forward unit radix-2 stage with its two radix-4 children.
            ///
            /// # Safety
            /// The kernel contract holds for eight spans, whose first INPUTS spans
            /// are packed; omitted spans represent zero and need not be
            /// initialized. Roots are W1, W2, W3.
            #[expect(
                clippy::too_many_lines,
                reason = "One assembly block holds the complete kernel"
            )]
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
                // Root products have residues below 15*p/16, and a reduced sum
                // is below 9*p/16, so every output is below 3*p, the input
                // bound of the next stage. All eight output spans are
                // disjoint. Omitted inputs are never read.
                unsafe {
                    let (start, end) = columns.complete::<$lanes>();
                    let constants = [
                        field.p(),
                        field.pinv(),
                        *twiddles,
                        *twiddles.add(1),
                        *twiddles.add(2),
                    ];
                    let row = |index: usize| {
                        values.add(index.unchecked_mul(stride).unchecked_add(start))
                    };
                    $run!(
                        $isa!(enter),
                        $isa!(splat 22, "{c}", 0),
                        $isa!(splat 23, "{c}", 8),
                        $isa!(splat 12, "{c}", 16),
                        $isa!(splat 13, "{c}", 24),
                        $isa!(splat 14, "{c}", 32),
                        "2:",
                        $isa!(load 0, "{r0}", 0),
                        ".if {inputs} == 1",
                        $isa!(store 0, "{r1}", 0),
                        $isa!(store 0, "{r2}", 0),
                        $isa!(store 0, "{r3}", 0),
                        $isa!(store 0, "{r4}", 0),
                        $isa!(store 0, "{r5}", 0),
                        $isa!(store 0, "{r6}", 0),
                        $isa!(store 0, "{r7}", 0),
                        ".elseif {inputs} == 2",
                        $isa!(load 1, "{r1}", 0),
                        asm_multiply!($isa, 2, 1, 12),
                        asm_multiply!($isa, 3, 1, 13),
                        asm_multiply!($isa, 4, 1, 14),
                        $isa!(add 5, 0, 1),
                        $isa!(store 5, "{r0}", 0),
                        $isa!(sub 5, 0, 1),
                        $isa!(store 5, "{r1}", 0),
                        $isa!(add 5, 0, 2),
                        $isa!(store 5, "{r2}", 0),
                        $isa!(sub 5, 0, 2),
                        $isa!(store 5, "{r3}", 0),
                        $isa!(add 5, 0, 3),
                        $isa!(store 5, "{r4}", 0),
                        $isa!(sub 5, 0, 3),
                        $isa!(store 5, "{r5}", 0),
                        $isa!(add 5, 0, 4),
                        $isa!(store 5, "{r6}", 0),
                        $isa!(sub 5, 0, 4),
                        $isa!(store 5, "{r7}", 0),
                        ".else",
                        $isa!(load 1, "{r1}", 0),
                        $isa!(load 2, "{r2}", 0),
                        $isa!(add 4, 0, 2),
                        $isa!(sub 5, 0, 2),
                        ".if {inputs} == 3",
                        asm_multiply!($isa, 7, 1, 12),
                        $isa!(add 8, 4, 1),
                        $isa!(store 8, "{r0}", 0),
                        $isa!(sub 8, 4, 1),
                        $isa!(store 8, "{r1}", 0),
                        ".else",
                        $isa!(load 3, "{r3}", 0),
                        $isa!(add 6, 1, 3),
                        asm_reduce!($isa, 6),
                        $isa!(sub 7, 1, 3),
                        asm_multiply!($isa, 7, 7, 12),
                        asm_multiply!($isa, 3, 3, 12),
                        $isa!(add 8, 4, 6),
                        $isa!(store 8, "{r0}", 0),
                        $isa!(sub 8, 4, 6),
                        $isa!(store 8, "{r1}", 0),
                        ".endif",
                        $isa!(add 8, 5, 7),
                        $isa!(store 8, "{r2}", 0),
                        $isa!(sub 8, 5, 7),
                        $isa!(store 8, "{r3}", 0),
                        asm_multiply!($isa, 2, 2, 12),
                        $isa!(add 4, 0, 2),
                        $isa!(sub 5, 0, 2),
                        ".if {inputs} == 3",
                        asm_multiply!($isa, 6, 1, 13),
                        asm_multiply!($isa, 7, 1, 14),
                        ".else",
                        $isa!(add 6, 1, 3),
                        $isa!(sub 7, 1, 3),
                        asm_multiply!($isa, 6, 6, 13),
                        asm_multiply!($isa, 7, 7, 14),
                        ".endif",
                        $isa!(add 8, 4, 6),
                        $isa!(store 8, "{r4}", 0),
                        $isa!(sub 8, 4, 6),
                        $isa!(store 8, "{r5}", 0),
                        $isa!(add 8, 5, 7),
                        $isa!(store 8, "{r6}", 0),
                        $isa!(sub 8, 5, 7),
                        $isa!(store 8, "{r7}", 0),
                        ".endif",
                        $isa!(step "{r0}"),
                        $isa!(step "{r1}"),
                        $isa!(step "{r2}"),
                        $isa!(step "{r3}"),
                        $isa!(step "{r4}"),
                        $isa!(step "{r5}"),
                        $isa!(step "{r6}"),
                        $isa!(step "{r7}"),
                        $isa!(next "{n}"),
                        $isa!(leave),
                        inputs = const INPUTS,
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
        }
    };
}
