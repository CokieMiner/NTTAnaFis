//! Kernel tables of every vector ISA.
//!
//! x86 and x86-64 use AVX2 with FMA, AVX-512F, and AVX-512 IFMA intrinsics;
//! `aarch64` uses NEON intrinsics. `s390x` uses the instruction functions of
//! `zarch` for the vector facility and its enhancements facility 1. The
//! `loongarch64` LSX and LASX extensions, `powerpc64` VSX, and the `riscv64` V
//! extension use instruction tables over complete assembly blocks.
//! The selector resolves one table per complete operation.

#[cfg(any(ntt_avx2, ntt_avx512, ntt_ifma, ntt_neon, ntt_vxe, ntt_zvector))]
use super::BIAS;
use super::Kernels;
#[cfg(ntt_ifma)]
use super::{
    CRT_LANES, CoefficientKernel, Cofactors, MAX_PRIMES, MODULUS_BITS, PRIME_CATALOG, WORDS,
};
#[cfg(ntt_avx512)]
use super::{Columns, FloatField};

#[cfg(ntt_avx2)]
mod avx2;
#[cfg(ntt_avx512)]
mod avx512;
#[cfg(ntt_ifma)]
mod ifma;
#[cfg(ntt_lasx)]
mod lasx;
#[cfg(ntt_lsx)]
mod lsx;
#[cfg(ntt_neon)]
mod neon;
#[cfg(ntt_rvv)]
mod rvv;
mod selector;
#[cfg(ntt_vsx)]
mod vsx;
#[cfg(ntt_vxe)]
mod vxe;
#[cfg(ntt_zarch)]
mod zarch;
#[cfg(ntt_zvector)]
mod zvector;

#[cfg(ntt_avx2)]
pub use avx2::Avx2;
#[cfg(ntt_avx512)]
pub use avx512::Avx512;
#[cfg(ntt_ifma)]
pub use ifma::Ifma;
#[cfg(ntt_lasx)]
pub use lasx::Lasx;
#[cfg(ntt_lsx)]
pub use lsx::Lsx;
#[cfg(ntt_neon)]
pub use neon::Neon;
#[cfg(ntt_rvv)]
pub use rvv::Rvv128;
#[cfg(ntt_rvv_wide)]
pub use rvv::{Rvv256, Rvv512};
#[cfg(ntt_vsx)]
pub use vsx::Vsx;
#[cfg(ntt_vxe)]
pub use vxe::Vxe;
#[cfg(ntt_vxe)]
pub use zarch::vfnmsdb;
#[cfg(ntt_zarch)]
pub use zarch::{
    Vector, vfadb, vfchdb, vfidb, vfmdb, vfsdb, vl, vmrhg, vmrlg, vn, vrepg, vst, vx, vzero,
};
#[cfg(ntt_zvector)]
pub use zarch::{vflcdb, vfmadb};
#[cfg(ntt_zvector)]
pub use zvector::Zvector;
