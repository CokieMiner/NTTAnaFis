//! Sixteen-point leaves for two-lane vectors.
//!
//! Each four-lane row is a pair of registers holding lanes 0..2 and 2..4. Row
//! butterflies act on the low and the high halves separately. The four-by-four
//! transpose takes column `2*h + k` from the `k`-th lanes of the `h` halves.
//! DIF output and DIT input share transposed leaf coordinates, so a common
//! permutation commutes with the inverse spectral product.

/// Forms the roots of child butterfly `half`: lane `j` of the first half uses
/// `W = roots[4*n+j]`, `l = roots[8*n+2*j]`, and `r = roots[8*n+2*j+1]`.
macro_rules! asm_child_roots {
    ($isa:ident, $quad:tt, $first:tt, $second:tt) => {
        asm_lines!(
            $isa!(load 12, "{quad}", $quad),
            $isa!(load 15, "{octet}", $first),
            $isa!(load 16, "{octet}", $second),
            $isa!(low 13, 15, 16),
            $isa!(high 14, 15, 16),
        )
    };
}

/// Broadcasts the roots of the leaf's outer butterflies.
macro_rules! asm_top_roots {
    ($isa:ident) => {
        asm_lines!(
            $isa!(splat 12, "{top}", 0),
            $isa!(splat 13, "{twice}", 0),
            $isa!(splat 14, "{twice}", 8),
        )
    };
}

macro_rules! assembly_leaf_pairs {
    ($name:ident, $isa:ident, $run:ident, $pointer:ident) => {
        impl $name {
            /// Executes forward leaves or inverse leaves with optional products.
            ///
            /// # Safety
            /// The contract of `LeafKernel` holds with residues below `3*p`, or
            /// `2*p` for the inverse. Forward outputs are below `3*p/2`.
            #[expect(
                clippy::too_many_lines,
                reason = "One assembly block holds the complete kernel"
            )]
            pub unsafe fn leaf16<const INVERSE: bool, const PRODUCT: bool, const SQUARE: bool>(
                values: *mut f64,
                right: *const f64,
                twiddles: *const f64,
                pairs: ::core::num::NonZeroUsize,
                node: usize,
                field: $crate::FloatField,
            ) {
                // SAFETY: each iteration owns one complete leaf of sixteen
                // residues, read and written as eight two-lane vectors. Its
                // largest root index is 8*(node+block)+7, inside the admitted
                // table. Transposition stays within the leaf and commutes with
                // componentwise products, whose bound abs(a*b) < 9*p^2/4 gives
                // residues below 3*p/2.
                unsafe {
                    let constants = [field.p(), field.pinv()];
                    $run!(
                        $isa!(enter),
                        $isa!(mask),
                        $isa!(splat 22, "{c}", 0),
                        $isa!(splat 23, "{c}", 8),
                        "2:",
                        $isa!(load 0, "{base}", 0),
                        $isa!(load 4, "{base}", 16),
                        $isa!(load 1, "{base}", 32),
                        $isa!(load 5, "{base}", 48),
                        $isa!(load 2, "{base}", 64),
                        $isa!(load 6, "{base}", 80),
                        $isa!(load 3, "{base}", 96),
                        $isa!(load 7, "{base}", 112),
                        ".if {inverse}",
                        ".if {product}",
                        ".if {square}",
                        asm_multiply!($isa, 0, 0, 0),
                        asm_multiply!($isa, 4, 4, 4),
                        asm_multiply!($isa, 1, 1, 1),
                        asm_multiply!($isa, 5, 5, 5),
                        asm_multiply!($isa, 2, 2, 2),
                        asm_multiply!($isa, 6, 6, 6),
                        asm_multiply!($isa, 3, 3, 3),
                        asm_multiply!($isa, 7, 7, 7),
                        ".else",
                        $isa!(load 8, "{other}", 0),
                        asm_multiply!($isa, 0, 0, 8),
                        $isa!(load 8, "{other}", 16),
                        asm_multiply!($isa, 4, 4, 8),
                        $isa!(load 8, "{other}", 32),
                        asm_multiply!($isa, 1, 1, 8),
                        $isa!(load 8, "{other}", 48),
                        asm_multiply!($isa, 5, 5, 8),
                        $isa!(load 8, "{other}", 64),
                        asm_multiply!($isa, 2, 2, 8),
                        $isa!(load 8, "{other}", 80),
                        asm_multiply!($isa, 6, 6, 8),
                        $isa!(load 8, "{other}", 96),
                        asm_multiply!($isa, 3, 3, 8),
                        $isa!(load 8, "{other}", 112),
                        asm_multiply!($isa, 7, 7, 8),
                        $isa!(advance "{other}", 128),
                        ".endif",
                        ".endif",
                        // Children: y0..y3 in 1, 3, 8, 0 and in 5, 7, 10, 4.
                        asm_child_roots!($isa, 0, 0, 16),
                        asm_dit!($isa, "0", 0, 1, 2, 3, 8, 9, 12, 13, 14),
                        asm_child_roots!($isa, 16, 32, 48),
                        asm_dit!($isa, "0", 4, 5, 6, 7, 10, 11, 12, 13, 14),
                        // Half rows: a in 2, 1; b in 9, 3; c in 8, 5; d in 0, 7.
                        $isa!(low 2, 1, 3),
                        $isa!(high 9, 1, 3),
                        $isa!(low 1, 8, 0),
                        $isa!(high 3, 8, 0),
                        $isa!(low 8, 5, 7),
                        $isa!(high 0, 5, 7),
                        $isa!(low 5, 10, 4),
                        $isa!(high 7, 10, 4),
                        // Outer: y0..y3 in 9, 0, 4, 2 and in 3, 7, 10, 1.
                        asm_top_roots!($isa),
                        asm_dit!($isa, "0", 2, 9, 8, 0, 4, 6, 12, 13, 14),
                        asm_dit!($isa, "0", 1, 3, 5, 7, 10, 11, 12, 13, 14),
                        $isa!(store 9, "{base}", 0),
                        $isa!(store 3, "{base}", 16),
                        $isa!(store 0, "{base}", 32),
                        $isa!(store 7, "{base}", 48),
                        $isa!(store 4, "{base}", 64),
                        $isa!(store 10, "{base}", 80),
                        $isa!(store 2, "{base}", 96),
                        $isa!(store 1, "{base}", 112),
                        ".else",
                        // Outer: y0..y3 in 2, 8, 3, 0 and in 6, 10, 7, 4.
                        asm_top_roots!($isa),
                        asm_dif!($isa, "0", "0", 0, 1, 2, 3, 8, 9, 12, 13, 14),
                        asm_dif!($isa, "0", "0", 4, 5, 6, 7, 10, 11, 12, 13, 14),
                        // Half rows: a in 1, 2; b in 9, 8; c in 3, 6; d in 0, 10.
                        $isa!(low 1, 2, 8),
                        $isa!(high 9, 2, 8),
                        $isa!(low 2, 3, 0),
                        $isa!(high 8, 3, 0),
                        $isa!(low 3, 6, 10),
                        $isa!(high 0, 6, 10),
                        $isa!(low 6, 7, 4),
                        $isa!(high 10, 7, 4),
                        // Children: y0..y3 in 3, 4, 0, 1 and in 6, 7, 10, 2.
                        asm_child_roots!($isa, 0, 0, 16),
                        asm_dif!($isa, "0", "1", 1, 9, 3, 0, 4, 5, 12, 13, 14),
                        asm_child_roots!($isa, 16, 32, 48),
                        asm_dif!($isa, "0", "1", 2, 8, 6, 10, 7, 11, 12, 13, 14),
                        $isa!(store 3, "{base}", 0),
                        $isa!(store 6, "{base}", 16),
                        $isa!(store 4, "{base}", 32),
                        $isa!(store 7, "{base}", 48),
                        $isa!(store 0, "{base}", 64),
                        $isa!(store 10, "{base}", 80),
                        $isa!(store 1, "{base}", 96),
                        $isa!(store 2, "{base}", 112),
                        ".endif",
                        $isa!(advance "{base}", 128),
                        $isa!(advance "{top}", 8),
                        $isa!(advance "{twice}", 16),
                        $isa!(advance "{quad}", 32),
                        $isa!(advance "{octet}", 64),
                        $isa!(next "{n}"),
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
                        n = inout($pointer) pairs.get().unchecked_mul(4) => _
                    );
                }
            }
        }
    };
}
