//! POWER VSX kernels on 128-bit registers (POWER7 and later).
//!
//! The module gate requires VSX at compile time; little-endian `powerpc64`
//! targets assume POWER8. `lxvd2x` and `stxvd2x` place the float at the lower
//! address in doubleword 0 on either byte order, and `xxpermdi` numbers
//! doublewords the same way, so the text does not depend on the byte order.
//! Logical registers are `vs32..=vs51` and `vs0..=vs3`, all caller-saved.

#![expect(
    unsafe_code,
    reason = "The compile-time VSX gate admits the instructions; kernel contracts bound all initialized vector spans"
)]

/// Maps logical registers to VSX register numbers.
#[rustfmt::skip]
macro_rules! vs {
    (0) => { "32" };
    (1) => { "33" };
    (2) => { "34" };
    (3) => { "35" };
    (4) => { "36" };
    (5) => { "37" };
    (6) => { "38" };
    (7) => { "39" };
    (8) => { "40" };
    (9) => { "41" };
    (10) => { "42" };
    (11) => { "43" };
    (12) => { "44" };
    (13) => { "45" };
    (14) => { "46" };
    (15) => { "47" };
    (16) => { "48" };
    (17) => { "49" };
    (18) => { "50" };
    (19) => { "51" };
    (20) => { "0" };
    (21) => { "1" };
    (22) => { "2" };
    (23) => { "3" };
}

/// VSX instructions over logical registers; `r12` holds displacements.
#[rustfmt::skip]
macro_rules! vsx {
    (enter) => { "" };
    (leave) => { "" };
    (mask) => { "" };
    (load $v:tt, $p:tt, 0) => { concat!("lxvd2x ", vs!($v), ", 0, ", $p) };
    (load $v:tt, $p:tt, $o:tt) => { concat!("li 12, ", $o, "\nlxvd2x ", vs!($v), ", ", $p, ", 12") };
    (store $v:tt, $p:tt, 0) => { concat!("stxvd2x ", vs!($v), ", 0, ", $p) };
    (store $v:tt, $p:tt, $o:tt) => { concat!("li 12, ", $o, "\nstxvd2x ", vs!($v), ", ", $p, ", 12") };
    (splat $v:tt, $p:tt, 0) => { concat!("lxvdsx ", vs!($v), ", 0, ", $p) };
    (splat $v:tt, $p:tt, $o:tt) => { concat!("li 12, ", $o, "\nlxvdsx ", vs!($v), ", ", $p, ", 12") };
    (advance $p:tt, $b:tt) => { concat!("addi ", $p, ", ", $p, ", ", $b) };
    (step $p:tt) => { concat!("addi ", $p, ", ", $p, ", 16") };
    (next $n:tt) => { concat!("addic. ", $n, ", ", $n, ", -2\nbne 0, 2b") };
    (add $d:tt, $a:tt, $b:tt) => { concat!("xvadddp ", vs!($d), ", ", vs!($a), ", ", vs!($b)) };
    (sub $d:tt, $a:tt, $b:tt) => { concat!("xvsubdp ", vs!($d), ", ", vs!($a), ", ", vs!($b)) };
    (mul $d:tt, $a:tt, $b:tt) => { concat!("xvmuldp ", vs!($d), ", ", vs!($a), ", ", vs!($b)) };
    (round $v:tt) => { concat!("xvrdpic ", vs!($v), ", ", vs!($v)) };
    (nmsub_acc $c:tt, $a:tt, $b:tt) => { concat!("xvnmsubadp ", vs!($c), ", ", vs!($a), ", ", vs!($b)) };
    (nmsub_mul $a:tt, $b:tt, $c:tt) => { concat!("xvnmsubmdp ", vs!($a), ", ", vs!($b), ", ", vs!($c)) };
    (zero $v:tt) => { concat!("xxlxor ", vs!($v), ", ", vs!($v), ", ", vs!($v)) };
    (canonical $v:tt) => {
        concat!(
            "xvcmpgtdp ", vs!(18), ", ", vs!(17), ", ", vs!($v), "\n",
            "xxland ", vs!(18), ", ", vs!(18), ", ", vs!(22), "\n",
            "xvadddp ", vs!($v), ", ", vs!($v), ", ", vs!(18), "\n",
            "xvadddp ", vs!($v), ", ", vs!($v), ", ", vs!(16), "\n",
            "xxlxor ", vs!($v), ", ", vs!($v), ", ", vs!(16)
        )
    };
    (low $d:tt, $a:tt, $b:tt) => { concat!("xxpermdi ", vs!($d), ", ", vs!($a), ", ", vs!($b), ", 0") };
    (high $d:tt, $a:tt, $b:tt) => { concat!("xxpermdi ", vs!($d), ", ", vs!($a), ", ", vs!($b), ", 3") };
}

/// Executes one kernel. `addic.` writes the carry and condition field 0, which
/// the compiler does not otherwise treat as clobbered on this target.
macro_rules! vsx_run {
    ($($body:tt)*) => {
        core::arch::asm!(
            $($body)*,
            out("vs0") _, out("vs1") _, out("vs2") _, out("vs3") _,
            out("vs32") _, out("vs33") _, out("vs34") _, out("vs35") _,
            out("vs36") _, out("vs37") _, out("vs38") _, out("vs39") _,
            out("vs40") _, out("vs41") _, out("vs42") _, out("vs43") _,
            out("vs44") _, out("vs45") _, out("vs46") _, out("vs47") _,
            out("vs48") _, out("vs49") _, out("vs50") _, out("vs51") _,
            out("r12") _, out("cr0") _, out("xer") _,
            options(nostack)
        )
    };
}

assembly_backend!(Vsx, vsx, vsx_run, 2, reg_nonzero, Vsx);
assembly_leaf_pairs!(Vsx, vsx, vsx_run, reg_nonzero);
