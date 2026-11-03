//! `loongarch64` LSX kernels on 128-bit registers.
//!
//! The selector detects the extension. LSX has fused and rounding
//! instructions in every lane form (`vfnmsub.d`, `vfrintrne.d`).

#![expect(
    unsafe_code,
    reason = "The selector detects LSX; kernel contracts bound all initialized vector spans"
)]

/// LSX instructions over logical registers.
#[rustfmt::skip]
macro_rules! lsx {
    (enter) => { "" };
    (leave) => { "" };
    (mask) => { "" };
    (load $v:tt, $p:tt, $o:tt) => { concat!("vld $vr", $v, ", ", $p, ", ", $o) };
    (store $v:tt, $p:tt, $o:tt) => { concat!("vst $vr", $v, ", ", $p, ", ", $o) };
    (splat $v:tt, $p:tt, $o:tt) => { concat!("vldrepl.d $vr", $v, ", ", $p, ", ", $o) };
    (advance $p:tt, $b:tt) => { concat!("addi.d ", $p, ", ", $p, ", ", $b) };
    (step $p:tt) => { concat!("addi.d ", $p, ", ", $p, ", 16") };
    (next $n:tt) => { concat!("addi.d ", $n, ", ", $n, ", -2\nbnez ", $n, ", 2b") };
    (add $d:tt, $a:tt, $b:tt) => { concat!("vfadd.d $vr", $d, ", $vr", $a, ", $vr", $b) };
    (sub $d:tt, $a:tt, $b:tt) => { concat!("vfsub.d $vr", $d, ", $vr", $a, ", $vr", $b) };
    (mul $d:tt, $a:tt, $b:tt) => { concat!("vfmul.d $vr", $d, ", $vr", $a, ", $vr", $b) };
    (round $v:tt) => { concat!("vfrintrne.d $vr", $v, ", $vr", $v) };
    (nmsub_acc $c:tt, $a:tt, $b:tt) => {
        concat!("vfnmsub.d $vr", $c, ", $vr", $a, ", $vr", $b, ", $vr", $c)
    };
    (nmsub_mul $a:tt, $b:tt, $c:tt) => {
        concat!("vfnmsub.d $vr", $a, ", $vr", $a, ", $vr", $b, ", $vr", $c)
    };
    (zero $v:tt) => { concat!("vxor.v $vr", $v, ", $vr", $v, ", $vr", $v) };
    (canonical $v:tt) => {
        concat!(
            "vfcmp.clt.d $vr18, $vr", $v, ", $vr17\n",
            "vand.v $vr18, $vr18, $vr22\n",
            "vfadd.d $vr", $v, ", $vr", $v, ", $vr18\n",
            "vfadd.d $vr", $v, ", $vr", $v, ", $vr16\n",
            "vxor.v $vr", $v, ", $vr", $v, ", $vr16"
        )
    };
    (low $d:tt, $a:tt, $b:tt) => { concat!("vilvl.d $vr", $d, ", $vr", $b, ", $vr", $a) };
    (high $d:tt, $a:tt, $b:tt) => { concat!("vilvh.d $vr", $d, ", $vr", $b, ", $vr", $a) };
}

/// Executes one kernel; registers `$f0..=$f23` alias the vector registers and
/// are caller-saved.
macro_rules! lsx_run {
    ($($body:tt)*) => {
        core::arch::asm!(
            $($body)*,
            out("$f0") _, out("$f1") _, out("$f2") _, out("$f3") _,
            out("$f4") _, out("$f5") _, out("$f6") _, out("$f7") _,
            out("$f8") _, out("$f9") _, out("$f10") _, out("$f11") _,
            out("$f12") _, out("$f13") _, out("$f14") _, out("$f15") _,
            out("$f16") _, out("$f17") _, out("$f18") _, out("$f19") _,
            out("$f20") _, out("$f21") _, out("$f22") _, out("$f23") _,
            options(nostack)
        )
    };
}

assembly_backend!(Lsx, lsx, lsx_run, 2, reg, Lsx);
assembly_leaf_pairs!(Lsx, lsx, lsx_run, reg);
