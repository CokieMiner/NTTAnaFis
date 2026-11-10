//! `loongarch64` LASX kernels on 256-bit registers.
//!
//! The selector detects the extension. Sixteen-point leaves hold each row in
//! one register, as AVX2 does.

#![expect(
    unsafe_code,
    reason = "The selector detects LASX; kernel contracts bound all initialized vector spans"
)]

/// LASX instructions over logical registers.
#[rustfmt::skip]
macro_rules! lasx {
    (enter) => { "" };
    (leave) => { "" };
    (mask) => { "" };
    (load $v:tt, $p:tt, $o:tt) => { concat!("xvld $xr", $v, ", ", $p, ", ", $o) };
    (store $v:tt, $p:tt, $o:tt) => { concat!("xvst $xr", $v, ", ", $p, ", ", $o) };
    (splat $v:tt, $p:tt, $o:tt) => { concat!("xvldrepl.d $xr", $v, ", ", $p, ", ", $o) };
    (advance $p:tt, $b:tt) => { concat!("addi.d ", $p, ", ", $p, ", ", $b) };
    (step $p:tt) => { concat!("addi.d ", $p, ", ", $p, ", 32") };
    (next $n:tt) => { concat!("addi.d ", $n, ", ", $n, ", -4\nbnez ", $n, ", 2b") };
    (next_block $n:tt) => { concat!("addi.d ", $n, ", ", $n, ", -1\nbnez ", $n, ", 2b") };
    (add $d:tt, $a:tt, $b:tt) => { concat!("xvfadd.d $xr", $d, ", $xr", $a, ", $xr", $b) };
    (sub $d:tt, $a:tt, $b:tt) => { concat!("xvfsub.d $xr", $d, ", $xr", $a, ", $xr", $b) };
    (mul $d:tt, $a:tt, $b:tt) => { concat!("xvfmul.d $xr", $d, ", $xr", $a, ", $xr", $b) };
    (round $v:tt) => { concat!("xvfrintrne.d $xr", $v, ", $xr", $v) };
    (nmsub_acc $c:tt, $a:tt, $b:tt) => {
        concat!("xvfnmsub.d $xr", $c, ", $xr", $a, ", $xr", $b, ", $xr", $c)
    };
    (nmsub_mul $a:tt, $b:tt, $c:tt) => {
        concat!("xvfnmsub.d $xr", $a, ", $xr", $a, ", $xr", $b, ", $xr", $c)
    };
    (zero $v:tt) => { concat!("xvxor.v $xr", $v, ", $xr", $v, ", $xr", $v) };
    (canonical $v:tt) => {
        concat!(
            "xvfcmp.clt.d $xr18, $xr", $v, ", $xr17\n",
            "xvand.v $xr18, $xr18, $xr22\n",
            "xvfadd.d $xr", $v, ", $xr", $v, ", $xr18\n",
            "xvfadd.d $xr", $v, ", $xr", $v, ", $xr16\n",
            "xvxor.v $xr", $v, ", $xr", $v, ", $xr16"
        )
    };
    (unpack_low $d:tt, $a:tt, $b:tt) => { concat!("xvilvl.d $xr", $d, ", $xr", $b, ", $xr", $a) };
    (unpack_high $d:tt, $a:tt, $b:tt) => { concat!("xvilvh.d $xr", $d, ", $xr", $b, ", $xr", $a) };
    // Temporary 19.
    (cross $x:tt, $y:tt) => {
        concat!(
            "xvor.v $xr19, $xr", $y, ", $xr", $y, "\n",
            "xvpermi.q $xr", $y, ", $xr", $x, ", 0x31\n",
            "xvpermi.q $xr", $x, ", $xr19, 0x02"
        )
    };
    // Temporaries 15 and 16.
    (root_pairs $l:tt, $r:tt, $p:tt) => {
        concat!(
            "xvld $xr15, ", $p, ", 0\n",
            "xvld $xr16, ", $p, ", 32\n",
            "xvilvl.d $xr", $l, ", $xr16, $xr15\n",
            "xvilvh.d $xr", $r, ", $xr16, $xr15\n",
            "xvpermi.d $xr", $l, ", $xr", $l, ", 0xd8\n",
            "xvpermi.d $xr", $r, ", $xr", $r, ", 0xd8"
        )
    };
}

/// Executes one kernel; registers `$f0..=$f23` alias the vector registers and
/// are caller-saved.
macro_rules! lasx_run {
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

assembly_backend!(Lasx, lasx, lasx_run, 4, reg, Lasx);
assembly_leaf_rows!(Lasx, lasx, lasx_run, reg);
