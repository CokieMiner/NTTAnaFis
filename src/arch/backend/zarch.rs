//! z/Architecture vector instructions as inline functions.
//!
//! Each function emits one instruction of the vector facility (z13) or of its
//! enhancements facility 1 (z14). A [`Vector`] holds two binary64 lanes in one
//! 128-bit register; element 0 is the lane at the lower address.

#![expect(
    unsafe_code,
    reason = "Each assembly statement is one instruction whose facility the target feature attribute establishes"
)]

use core::arch::asm;

/// Two binary64 lanes in one vector register.
pub type Vector = u128;

/// Loads two consecutive floats (`vl`).
///
/// # Safety
/// `source` covers two readable initialized floats.
#[inline]
#[target_feature(enable = "vector")]
pub unsafe fn vl(source: *const f64) -> Vector {
    let result;
    // SAFETY: the caller admits sixteen readable bytes; `vl` has no alignment
    // requirement.
    unsafe {
        asm!(
            "vl {0}, 0({1})",
            lateout(vreg) result,
            in(reg_addr) source,
            options(pure, readonly, nostack, preserves_flags)
        );
    }
    result
}

/// Stores two consecutive floats (`vst`).
///
/// # Safety
/// `destination` covers two exclusively writable floats.
#[inline]
#[target_feature(enable = "vector")]
pub unsafe fn vst(destination: *mut f64, value: Vector) {
    // SAFETY: the caller admits sixteen writable bytes; `vst` has no alignment
    // requirement.
    unsafe {
        asm!(
            "vst {0}, 0({1})",
            in(vreg) value,
            in(reg_addr) destination,
            options(nostack, preserves_flags)
        );
    }
}

/// Broadcasts one float to both lanes (`vrepg`).
#[inline]
#[target_feature(enable = "vector")]
pub fn vrepg(value: f64) -> Vector {
    let result;
    // SAFETY: the instruction reads element 0, which holds `value`, and only
    // writes its result register.
    unsafe {
        asm!(
            "vrepg {0}, {1}, 0",
            lateout(vreg) result,
            in(vreg) value,
            options(pure, nomem, nostack, preserves_flags)
        );
    }
    result
}

/// Returns the all-zero vector (`vzero`).
#[inline]
#[target_feature(enable = "vector")]
pub fn vzero() -> Vector {
    let result;
    // SAFETY: the instruction only writes its result register.
    unsafe {
        asm!("vzero {0}", lateout(vreg) result, options(pure, nomem, nostack, preserves_flags));
    }
    result
}

/// Rounds each lane to the nearest integer, ties to even, without raising
/// the inexact exception (`vfidb` with rounding method 4).
#[inline]
#[target_feature(enable = "vector")]
pub fn vfidb(a: Vector) -> Vector {
    let result;
    // SAFETY: the instruction only reads and writes its register operands.
    unsafe {
        asm!(
            "vfidb {0}, {1}, 4, 4",
            lateout(vreg) result,
            in(vreg) a,
            options(pure, nomem, nostack, preserves_flags)
        );
    }
    result
}

/// Negates each lane (`vflcdb`).
#[cfg(ntt_zvector)]
#[inline]
#[target_feature(enable = "vector")]
pub fn vflcdb(a: Vector) -> Vector {
    let result;
    // SAFETY: the instruction only reads and writes its register operands.
    unsafe {
        asm!(
            "vflcdb {0}, {1}",
            lateout(vreg) result,
            in(vreg) a,
            options(pure, nomem, nostack, preserves_flags)
        );
    }
    result
}

/// Defines two-operand vector instructions.
macro_rules! binary_instructions {
    ($($(#[$documentation:meta])* $name:ident),* $(,)?) => {
        $(
            $(#[$documentation])*
            #[inline]
            #[target_feature(enable = "vector")]
            pub fn $name(a: Vector, b: Vector) -> Vector {
                let result;
                // SAFETY: the instruction only reads and writes its register
                // operands and leaves the condition code unchanged.
                unsafe {
                    asm!(
                        concat!(stringify!($name), " {0}, {1}, {2}"),
                        lateout(vreg) result,
                        in(vreg) a,
                        in(vreg) b,
                        options(pure, nomem, nostack, preserves_flags)
                    );
                }
                result
            }
        )*
    };
}

binary_instructions!(
    /// Adds lanes (`vfadb`).
    vfadb,
    /// Subtracts lanes, `a - b` (`vfsdb`).
    vfsdb,
    /// Multiplies lanes (`vfmdb`).
    vfmdb,
    /// Compares lanes, all ones where `a > b` (`vfchdb`).
    vfchdb,
    /// Bitwise conjunction (`vn`).
    vn,
    /// Bitwise exclusive or (`vx`).
    vx,
    /// Interleaves the first lanes, `[a0, b0]` (`vmrhg`).
    vmrhg,
    /// Interleaves the second lanes, `[a1, b1]` (`vmrlg`).
    vmrlg,
);

/// Returns `RN(a*b + c)` in each lane (`vfmadb`).
#[cfg(ntt_zvector)]
#[inline]
#[target_feature(enable = "vector")]
pub fn vfmadb(a: Vector, b: Vector, c: Vector) -> Vector {
    let result;
    // SAFETY: the instruction only reads and writes its register operands.
    unsafe {
        asm!(
            "vfmadb {0}, {1}, {2}, {3}",
            lateout(vreg) result,
            in(vreg) a,
            in(vreg) b,
            in(vreg) c,
            options(pure, nomem, nostack, preserves_flags)
        );
    }
    result
}

/// Returns `RN(c - a*b)` in each lane (`vfnmsdb`, enhancements facility 1).
#[cfg(ntt_vxe)]
#[inline]
#[target_feature(enable = "vector,vector-enhancements-1")]
pub fn vfnmsdb(a: Vector, b: Vector, c: Vector) -> Vector {
    let result;
    // SAFETY: the instruction only reads and writes its register operands.
    unsafe {
        asm!(
            "vfnmsdb {0}, {1}, {2}, {3}",
            lateout(vreg) result,
            in(vreg) a,
            in(vreg) b,
            in(vreg) c,
            options(pure, nomem, nostack, preserves_flags)
        );
    }
    result
}
