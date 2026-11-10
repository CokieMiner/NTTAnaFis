//! RISC-V V-extension kernels at `LMUL = 1`.
//!
//! The selector detects the extension and the register width at run time.
//! [`Rvv128`] runs two lanes for `VLEN >= 128`, [`Rvv256`] runs four lanes for
//! `VLEN >= 256`, and [`Rvv512`] runs eight lanes for `VLEN >= 512`. Leaves hold
//! two lanes per register on [`Rvv128`] and four lanes otherwise.
//! Each kernel enables the instructions locally and sets its own vector
//! length, so the crate does not require the `v` target feature. Logical
//! registers are `v1..=v24`; `v0` is the mask register.
//!
//! Rounding converts through a signed integer with the dynamic rounding mode
//! (`vfcvt.x.f.v`, `vfcvt.f.x.v`), which holds ties to even. Quotients stay
//! below `2^51`, inside the exact integer range of binary64.

#![expect(
    unsafe_code,
    reason = "The selector detects V and the register width; kernel contracts bound all initialized vector spans"
)]

use core::arch::asm;
#[cfg(all(target_os = "linux", ntt_rvv_wide))]
use core::ffi::c_ulong;

/// Maps logical registers to vector registers.
#[rustfmt::skip]
macro_rules! vr {
    (0) => { "v1" };
    (1) => { "v2" };
    (2) => { "v3" };
    (3) => { "v4" };
    (4) => { "v5" };
    (5) => { "v6" };
    (6) => { "v7" };
    (7) => { "v8" };
    (8) => { "v9" };
    (9) => { "v10" };
    (10) => { "v11" };
    (11) => { "v12" };
    (12) => { "v13" };
    (13) => { "v14" };
    (14) => { "v15" };
    (15) => { "v16" };
    (16) => { "v17" };
    (17) => { "v18" };
    (18) => { "v19" };
    (19) => { "v20" };
    (20) => { "v21" };
    (21) => { "v22" };
    (22) => { "v23" };
    (23) => { "v24" };
}

/// V-extension instructions shared by both widths; `t0` holds addresses and
/// raw constants.
#[rustfmt::skip]
macro_rules! rvv {
    (leave) => { ".option pop" };
    (load $v:tt, $p:tt, 0) => { concat!("vle64.v ", vr!($v), ", (", $p, ")") };
    (load $v:tt, $p:tt, $o:tt) => { concat!("addi t0, ", $p, ", ", $o, "\nvle64.v ", vr!($v), ", (t0)") };
    (store $v:tt, $p:tt, 0) => { concat!("vse64.v ", vr!($v), ", (", $p, ")") };
    (store $v:tt, $p:tt, $o:tt) => { concat!("addi t0, ", $p, ", ", $o, "\nvse64.v ", vr!($v), ", (t0)") };
    (splat $v:tt, $p:tt, $o:tt) => { concat!("ld t0, ", $o, "(", $p, ")\nvmv.v.x ", vr!($v), ", t0") };
    (advance $p:tt, $b:tt) => { concat!("addi ", $p, ", ", $p, ", ", $b) };
    (add $d:tt, $a:tt, $b:tt) => { concat!("vfadd.vv ", vr!($d), ", ", vr!($a), ", ", vr!($b)) };
    (sub $d:tt, $a:tt, $b:tt) => { concat!("vfsub.vv ", vr!($d), ", ", vr!($a), ", ", vr!($b)) };
    (mul $d:tt, $a:tt, $b:tt) => { concat!("vfmul.vv ", vr!($d), ", ", vr!($a), ", ", vr!($b)) };
    (round $v:tt) => { concat!("vfcvt.x.f.v ", vr!($v), ", ", vr!($v), "\nvfcvt.f.x.v ", vr!($v), ", ", vr!($v)) };
    (nmsub_acc $c:tt, $a:tt, $b:tt) => { concat!("vfnmsac.vv ", vr!($c), ", ", vr!($a), ", ", vr!($b)) };
    (nmsub_mul $a:tt, $b:tt, $c:tt) => { concat!("vfnmsub.vv ", vr!($a), ", ", vr!($b), ", ", vr!($c)) };
    (zero $v:tt) => { concat!("vmv.v.i ", vr!($v), ", 0") };
    (canonical $v:tt) => {
        concat!(
            "vmflt.vv v0, ", vr!($v), ", ", vr!(17), "\n",
            "vfadd.vv ", vr!($v), ", ", vr!($v), ", ", vr!(22), ", v0.t\n",
            "vfadd.vv ", vr!($v), ", ", vr!($v), ", ", vr!(16), "\n",
            "vxor.vv ", vr!($v), ", ", vr!($v), ", ", vr!(16)
        )
    };
    // Lane pairs of two registers under a mask of the odd lanes.
    (low $d:tt, $a:tt, $b:tt) => {
        concat!(
            "vslideup.vi ", vr!(19), ", ", vr!($b), ", 1\n",
            "vmerge.vvm ", vr!($d), ", ", vr!($a), ", ", vr!(19), ", v0"
        )
    };
    (high $d:tt, $a:tt, $b:tt) => {
        concat!(
            "vslidedown.vi ", vr!(19), ", ", vr!($a), ", 1\n",
            "vmerge.vvm ", vr!($d), ", ", vr!(19), ", ", vr!($b), ", v0"
        )
    };
}

/// Two lanes at `VLEN >= 128`.
#[rustfmt::skip]
macro_rules! rvv2 {
    (enter) => { ".option push\n.option arch, +v\nvsetivli zero, 2, e64, m1, ta, mu" };
    // Mask bit 1 selects the second lane of `low` and `high`.
    (mask) => { "vmv.v.i v0, 2" };
    (step $p:tt) => { concat!("addi ", $p, ", ", $p, ", 16") };
    (next $n:tt) => { concat!("addi ", $n, ", ", $n, ", -2\nbnez ", $n, ", 2b") };
    ($($operation:tt)*) => { rvv!($($operation)*) };
}

/// Four lanes at `VLEN >= 256`.
#[cfg(ntt_rvv_wide)]
#[rustfmt::skip]
macro_rules! rvv4 {
    (enter) => { ".option push\n.option arch, +v\nvsetivli zero, 4, e64, m1, ta, mu" };
    // Mask bits 1 and 3 select the odd lanes of `unpack_low` and `unpack_high`.
    (mask) => { "vmv.v.i v0, 10" };
    (step $p:tt) => { concat!("addi ", $p, ", ", $p, ", 32") };
    (next $n:tt) => { concat!("addi ", $n, ", ", $n, ", -4\nbnez ", $n, ", 2b") };
    (next_block $n:tt) => { concat!("addi ", $n, ", ", $n, ", -1\nbnez ", $n, ", 2b") };
    (unpack_low $d:tt, $a:tt, $b:tt) => { rvv!(low $d, $a, $b) };
    (unpack_high $d:tt, $a:tt, $b:tt) => { rvv!(high $d, $a, $b) };
    // Temporaries 18 and 19.
    (cross $x:tt, $y:tt) => {
        concat!(
            "vslidedown.vi ", vr!(18), ", ", vr!($x), ", 2\n",
            "vslidedown.vi ", vr!(19), ", ", vr!($y), ", 2\n",
            "vslideup.vi ", vr!($x), ", ", vr!($y), ", 2\n",
            "vslideup.vi ", vr!(18), ", ", vr!(19), ", 2\n",
            "vmv.v.v ", vr!($y), ", ", vr!(18)
        )
    };
    // A segmented load separates the even and the odd roots.
    (root_pairs $l:tt, $r:tt, $p:tt) => { concat!("vlseg2e64.v ", vr!($l), ", (", $p, ")") };
    ($($operation:tt)*) => { rvv!($($operation)*) };
}

/// Eight lanes at `VLEN >= 512`.
#[cfg(ntt_rvv_wide)]
#[rustfmt::skip]
macro_rules! rvv8 {
    (enter) => { ".option push\n.option arch, +v\nvsetivli zero, 8, e64, m1, ta, mu" };
    (step $p:tt) => { concat!("addi ", $p, ", ", $p, ", 64") };
    (next $n:tt) => { concat!("addi ", $n, ", ", $n, ", -8\nbnez ", $n, ", 2b") };
    ($($operation:tt)*) => { rvv!($($operation)*) };
}

/// Executes one kernel.
macro_rules! rvv_run {
    ($($body:tt)*) => {
        asm!(
            $($body)*,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _,
            out("v5") _, out("v6") _, out("v7") _, out("v8") _, out("v9") _,
            out("v10") _, out("v11") _, out("v12") _, out("v13") _, out("v14") _,
            out("v15") _, out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _, out("v24") _,
            out("t0") _,
            options(nostack)
        )
    };
}

assembly_backend!(Rvv128, rvv2, rvv_run, 2, reg, Rvv128);
assembly_leaf_pairs!(Rvv128, rvv2, rvv_run, reg);

#[cfg(ntt_rvv_wide)]
assembly_backend!(Rvv256, rvv4, rvv_run, 4, reg, Rvv256);
#[cfg(ntt_rvv_wide)]
assembly_leaf_rows!(Rvv256, rvv4, rvv_run, reg);
#[cfg(ntt_rvv_wide)]
assembly_backend!(Rvv512, rvv8, rvv_run, 8, reg, Rvv256);

impl Rvv128 {
    /// Returns the vector register width in bytes, or `None` without V.
    ///
    /// Bit 21 of the Linux hardware capabilities is the letter V. Other
    /// operating systems report V only through the compile-time feature.
    #[cfg(ntt_rvv_wide)]
    #[must_use]
    pub fn register_bytes() -> Option<usize> {
        #[cfg(target_os = "linux")]
        let detected = {
            const AT_HWCAP: c_ulong = 16;
            unsafe extern "C" {
                safe fn getauxval(kind: c_ulong) -> c_ulong;
            }
            cfg!(target_feature = "v") || getauxval(AT_HWCAP) & (1 << 21) != 0
        };
        #[cfg(not(target_os = "linux"))]
        let detected = cfg!(target_feature = "v");
        if !detected {
            return None;
        }
        let bytes;
        // SAFETY: V is present, which makes `vlenb` readable. The directive
        // only enables the mnemonic for this statement.
        unsafe {
            asm!(
                ".option push",
                ".option arch, +v",
                "csrr {0}, vlenb",
                ".option pop",
                out(reg) bytes,
                options(nomem, nostack, preserves_flags)
            );
        }
        Some(bytes)
    }
}
