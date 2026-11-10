//! Kernel schedules, each written once.
//!
//! The schedules in `expression` are macros over a backend's load, store,
//! arithmetic, and exact-product operations. `schedule_backend!` instantiates all
//! of them. [`Scalar`](super::Scalar) uses it with one lane; `vector_backend!` adds
//! the fused register arithmetic of the targets with intrinsic functions. Targets
//! whose vector registers cannot be `asm!` operands use the block text in `assembly`.
//!
//! Admitted ranges end at multiples of every lane count, so vector loops cover
//! complete ranges with no scalar cleanup. The kernel contract is in `contract`.

#[macro_use]
mod table;
#[cfg(ntt_expression_schedule)]
#[macro_use]
mod expression;
#[cfg(any(ntt_lsx, ntt_lasx, ntt_vsx, ntt_rvv))]
#[macro_use]
mod assembly;
#[cfg(any(ntt_avx2, ntt_avx512, ntt_neon, ntt_zvector, ntt_vxe))]
#[macro_use]
mod vector;
#[cfg(any(ntt_avx2, ntt_avx512))]
#[macro_use]
mod leaf;
#[cfg(any(ntt_neon, ntt_zvector, ntt_vxe, ntt_simd128))]
#[macro_use]
mod leaf_pairs;
