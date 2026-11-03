//! Sixteen-point leaves holding each four-lane row in one register.
//!
//! DIF output and DIT input share transposed leaf coordinates, so a common
//! permutation commutes with the inverse spectral product. The table supplies
//! the transpose: `unpack_low d, a, b` gives `[a0, b0, a2, b2]`, `unpack_high`
//! gives `[a1, b1, a3, b3]`, and `cross x, y` replaces `x` and `y` by
//! `[x.low, y.low]` and `[x.high, y.high]`. `root_pairs l, r, ptr` loads eight
//! roots into `l = [r0, r2, r4, r6]` and `r = [r1, r3, r5, r7]`; `r` is the
//! register after `l`.

macro_rules! assembly_leaf_rows {
    ($name:ident, $isa:ident, $run:ident, $pointer:ident) => {
        impl $name {
            /// Executes forward leaves or inverse leaves with optional products.
            ///
            /// # Safety
            /// The contract of `LeafKernel` holds with residues below `3*p`, or
            /// `2*p` for the inverse. Forward outputs are below `3*p/2`.
            pub unsafe fn leaf16<const INVERSE: bool, const PRODUCT: bool, const SQUARE: bool>(
                values: *mut f64,
                right: *const f64,
                twiddles: *const f64,
                pairs: ::core::num::NonZeroUsize,
                node: usize,
                field: $crate::FloatField,
            ) {
                // SAFETY: each iteration owns one complete leaf of four 32-byte
                // rows. Its largest root index is 8*(node+block)+7, inside the
                // admitted table. Transposition stays within the leaf and
                // commutes with componentwise products, whose bound
                // abs(a*b) < 9*p^2/4 gives residues below 3*p/2.
                unsafe {
                    let constants = [field.p(), field.pinv()];
                    $run!(
                        $isa!(enter),
                        $isa!(mask),
                        $isa!(splat 22, "{c}", 0),
                        $isa!(splat 23, "{c}", 8),
                        "2:",
                        $isa!(load 0, "{base}", 0),
                        $isa!(load 1, "{base}", 32),
                        $isa!(load 2, "{base}", 64),
                        $isa!(load 3, "{base}", 96),
                        ".if {inverse}",
                        ".if {product}",
                        ".if {square}",
                        asm_multiply!($isa, 0, 0, 0),
                        asm_multiply!($isa, 1, 1, 1),
                        asm_multiply!($isa, 2, 2, 2),
                        asm_multiply!($isa, 3, 3, 3),
                        ".else",
                        $isa!(load 8, "{other}", 0),
                        asm_multiply!($isa, 0, 0, 8),
                        $isa!(load 8, "{other}", 32),
                        asm_multiply!($isa, 1, 1, 8),
                        $isa!(load 8, "{other}", 64),
                        asm_multiply!($isa, 2, 2, 8),
                        $isa!(load 8, "{other}", 96),
                        asm_multiply!($isa, 3, 3, 8),
                        $isa!(advance "{other}", 128),
                        ".endif",
                        ".endif",
                        // Child roots: lane j uses W = roots[4*n+j],
                        // l = roots[8*n+2*j], and r = roots[8*n+2*j+1].
                        $isa!(load 12, "{quad}", 0),
                        $isa!(root_pairs 13, 14, "{octet}"),
                        // Children: y0..y3 in 1, 3, 4, 0; 2 and 5 are free.
                        asm_dit!($isa, "0", 0, 1, 2, 3, 4, 5, 12, 13, 14),
                        $isa!(unpack_low 2, 1, 3),
                        $isa!(unpack_high 5, 1, 3),
                        $isa!(unpack_low 6, 4, 0),
                        $isa!(unpack_high 7, 4, 0),
                        $isa!(cross 2, 6),
                        $isa!(cross 5, 7),
                        // Outer: y0..y3 in 5, 7, 0, 2.
                        $isa!(splat 12, "{top}", 0),
                        $isa!(splat 13, "{twice}", 0),
                        $isa!(splat 14, "{twice}", 8),
                        asm_dit!($isa, "0", 2, 5, 6, 7, 0, 1, 12, 13, 14),
                        $isa!(store 5, "{base}", 0),
                        $isa!(store 7, "{base}", 32),
                        $isa!(store 0, "{base}", 64),
                        $isa!(store 2, "{base}", 96),
                        ".else",
                        // Child roots wait in 8, 9, 10 during the outer butterfly.
                        $isa!(load 8, "{quad}", 0),
                        $isa!(root_pairs 9, 10, "{octet}"),
                        $isa!(splat 12, "{top}", 0),
                        $isa!(splat 13, "{twice}", 0),
                        $isa!(splat 14, "{twice}", 8),
                        // Outer: y0..y3 in 2, 4, 3, 0; 1 and 5 are free.
                        asm_dif!($isa, "0", "0", 0, 1, 2, 3, 4, 5, 12, 13, 14),
                        $isa!(unpack_low 1, 2, 4),
                        $isa!(unpack_high 5, 2, 4),
                        $isa!(unpack_low 6, 3, 0),
                        $isa!(unpack_high 7, 3, 0),
                        $isa!(cross 1, 6),
                        $isa!(cross 5, 7),
                        // Children: y0..y3 in 6, 0, 7, 1.
                        asm_dif!($isa, "0", "1", 1, 5, 6, 7, 0, 2, 8, 9, 10),
                        $isa!(store 6, "{base}", 0),
                        $isa!(store 0, "{base}", 32),
                        $isa!(store 7, "{base}", 64),
                        $isa!(store 1, "{base}", 96),
                        ".endif",
                        $isa!(advance "{base}", 128),
                        $isa!(advance "{top}", 8),
                        $isa!(advance "{twice}", 16),
                        $isa!(advance "{quad}", 32),
                        $isa!(advance "{octet}", 64),
                        $isa!(next_block "{n}"),
                        $isa!(leave),
                        inverse = const if INVERSE { 1 } else { 0 },
                        product = const if PRODUCT { 1 } else { 0 },
                        square = const if SQUARE { 1 } else { 0 },
                        c = in($pointer) constants.as_ptr(),
                        base = inout($pointer) values => _,
                        other = inout($pointer) right => _,
                        top = inout($pointer) twiddles.add(node) => _,
                        twice = inout($pointer) twiddles.add(node.unchecked_mul(2)) => _,
                        quad = inout($pointer) twiddles.add(node.unchecked_mul(4)) => _,
                        octet = inout($pointer) twiddles.add(node.unchecked_mul(8)) => _,
                        n = inout($pointer) pairs.get().unchecked_mul(2) => _
                    );
                }
            }
        }
    };
}
