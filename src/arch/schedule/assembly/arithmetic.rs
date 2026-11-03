//! Text of the exact modular products and the radix-4 butterflies.
//!
//! The two fused forms of the instruction tables overwrite either the addend
//! or the first factor, so the text needs no register copies on the targets
//! whose fused instructions take three operands.

/// Joins template lines.
macro_rules! asm_lines {
    ($($line:expr),* $(,)?) => {
        concat!($($line, "\n"),*)
    };
}

/// `a = RN(a - RI(RN(a/p))*p)`; temporary 21.
macro_rules! asm_reduce {
    ($isa:ident, $a:tt) => {
        asm_lines!(
            $isa!(mul 21, $a, 23),
            $isa!(round 21),
            $isa!(nmsub_acc $a, 21, 22),
        )
    };
}

/// `d = a*b - q*p`, an exact integer; temporaries 20 and 21.
///
/// With `h = RN(a*b)` and `q = RI(RN(h/p))`, the fused `h - q*p` and
/// `h - a*b` are exact, so their difference is the exact remainder. `d` may
/// equal `a` or `b`, which stay live until the last instruction.
macro_rules! asm_multiply {
    ($isa:ident, $d:tt, $a:tt, $b:tt) => {
        asm_lines!(
            $isa!(mul 20, $a, $b),
            $isa!(mul 21, 20, 23),
            $isa!(round 21),
            $isa!(nmsub_mul 21, 22, 20),
            $isa!(nmsub_acc 20, $a, $b),
            $isa!(sub $d, 21, 20),
        )
    };
}

/// Forward radix-4 butterfly over `a, b, c, d`; temporaries `e`, `f`.
///
/// Outputs `y0..y3` are in `c, e, d, a`; `b` and `f` become free. `unit` is
/// the text of an assembly-time flag for the four-point stage, where the
/// roots `w` and `l` equal one. `last` is the flag of the final forward stage,
/// whose two reductions bound spectra for pointwise products; other stages
/// reduce only `a`, the input without a product.
macro_rules! asm_dif {
    ($isa:ident, $unit:tt, $last:tt, $a:tt, $b:tt, $c:tt, $d:tt, $e:tt, $f:tt, $w:tt, $l:tt, $r:tt) => {
        asm_lines!(
            concat!(".if ", $unit, " == 0"),
            asm_multiply!($isa, $c, $c, $w),
            asm_multiply!($isa, $d, $d, $w),
            ".endif",
            concat!(".if ", $unit, " == 0 && ", $last, " == 0"),
            asm_reduce!($isa, $a),
            ".endif",
            $isa!(add $e, $a, $c),
            $isa!(sub $a, $a, $c),
            $isa!(add $f, $b, $d),
            $isa!(sub $b, $b, $d),
            concat!(".if ", $unit, " || ", $last),
            asm_reduce!($isa, $e),
            asm_reduce!($isa, $a),
            ".endif",
            concat!(".if ", $unit),
            asm_reduce!($isa, $f),
            ".else",
            asm_multiply!($isa, $f, $f, $l),
            ".endif",
            asm_multiply!($isa, $b, $b, $r),
            $isa!(add $c, $e, $f),
            $isa!(sub $e, $e, $f),
            $isa!(add $d, $a, $b),
            $isa!(sub $a, $a, $b),
        )
    };
}

/// Inverse radix-4 butterfly over `a, b, c, d`; temporaries `e`, `f`.
///
/// Outputs `y0..y3` are in `b, d, e, a`; `c` and `f` become free. `unit` is
/// the text of an assembly-time flag for the four-point stage, where the
/// roots `w` and `l` equal one. Other stages reduce only the sum `y0`.
macro_rules! asm_dit {
    ($isa:ident, $unit:tt, $a:tt, $b:tt, $c:tt, $d:tt, $e:tt, $f:tt, $w:tt, $l:tt, $r:tt) => {
        asm_lines!(
            $isa!(add $e, $a, $b),
            $isa!(sub $a, $a, $b),
            $isa!(add $f, $c, $d),
            $isa!(sub $c, $c, $d),
            concat!(".if ", $unit),
            asm_reduce!($isa, $e),
            asm_reduce!($isa, $f),
            asm_reduce!($isa, $a),
            ".else",
            asm_multiply!($isa, $a, $a, $l),
            ".endif",
            asm_multiply!($isa, $c, $c, $r),
            $isa!(add $b, $e, $f),
            $isa!(sub $e, $e, $f),
            $isa!(add $d, $a, $c),
            $isa!(sub $a, $a, $c),
            concat!(".if ", $unit, " == 0"),
            asm_reduce!($isa, $b),
            asm_multiply!($isa, $e, $e, $w),
            asm_multiply!($isa, $a, $a, $w),
            ".endif",
        )
    };
}
