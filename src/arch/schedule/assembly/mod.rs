//! Complete kernels written as single inline-assembly blocks.
//!
//! Stable Rust admits vector registers as `asm!` operands only on `s390x`, so
//! a vector value cannot cross a statement on `loongarch64`, `powerpc64`, or
//! `riscv64`. Each kernel on these targets is therefore one block with an
//! explicit register assignment, and its code is independent of the
//! optimizer's vectorizers.
//!
//! The text is shared. A backend supplies an instruction table `isa!(op ...)`
//! that expands to the target's mnemonics and a runner that appends its
//! clobbers. Const generics select sections at assembly time through `.if`.
//!
//! Tables address 24 logical vector registers, all caller-saved:
//!
//! | Registers | Use |
//! | :--- | :--- |
//! | `0..=11` | residues and butterfly temporaries |
//! | `12..=14` | roots `w`, `l`, `r` |
//! | `15` | outer scale factor |
//! | `16`, `17` | `2^52` and zero for canonical conversion |
//! | `18`, `19` | temporaries of the table's `canonical`, `low`, and `high` |
//! | `20`, `21` | product temporaries |
//! | `22`, `23` | `p` and `RN(1/p)` |
//!
//! A table provides:
//!
//! | Operation | Effect |
//! | :--- | :--- |
//! | `enter`, `leave` | begin and end a kernel |
//! | `mask` | prepare `low` and `high` |
//! | `load v, ptr, off` | `v = [ptr + off]` |
//! | `store v, ptr, off` | `[ptr + off] = v` |
//! | `splat v, ptr, off` | every lane of `v` holds the float at `ptr + off` |
//! | `advance ptr, bytes` | `ptr += bytes` |
//! | `step ptr` | `ptr += ` one vector |
//! | `next n` | `n -= ` lanes; branch to label `2` while nonzero |
//! | `add d, a, b` | `d = RN(a + b)` |
//! | `sub d, a, b` | `d = RN(a - b)` |
//! | `mul d, a, b` | `d = RN(a * b)` |
//! | `round v` | `v` rounded to the nearest integer, ties to even |
//! | `nmsub_acc c, a, b` | `c = RN(c - a*b)` |
//! | `nmsub_mul a, b, c` | `a = RN(c - a*b)` |
//! | `zero v` | `v = 0` |
//! | `canonical v` | integer-valued `v` in `(-p, p)` to canonical u64 bits |
//! | `low d, a, b` | `d = [a0, b0]`, two-lane leaves only |
//! | `high d, a, b` | `d = [a1, b1]`, two-lane leaves only |
//! | `next_block n` | `n -= 1`; branch to label `2` while nonzero, four-lane leaves only |

#[macro_use]
mod arithmetic;
#[macro_use]
mod finish;
#[cfg(any(ntt_lsx, ntt_vsx, ntt_rvv))]
#[macro_use]
mod leaf_pairs;
#[cfg(any(ntt_lasx, ntt_rvv_wide))]
#[macro_use]
mod leaf_rows;
#[macro_use]
mod packing;
#[macro_use]
mod root;
#[macro_use]
mod stages;
