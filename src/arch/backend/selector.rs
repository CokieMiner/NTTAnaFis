//! One-time selection of an admissible residue kernel table.

#[cfg(all(ntt_runtime_selection, target_arch = "loongarch64"))]
use std::arch::is_loongarch_feature_detected;
#[cfg(all(ntt_runtime_selection, target_arch = "s390x"))]
use std::arch::is_s390x_feature_detected;
#[cfg(all(
    ntt_runtime_selection,
    any(target_arch = "x86", target_arch = "x86_64")
))]
use std::is_x86_feature_detected;

#[cfg(ntt_avx2)]
use super::Avx2;
#[cfg(ntt_avx512)]
use super::Avx512;
#[cfg(ntt_ifma)]
use super::Ifma;
use super::Kernels;
#[cfg(ntt_lasx)]
use super::Lasx;
#[cfg(ntt_lsx)]
use super::Lsx;
#[cfg(ntt_neon)]
use super::Neon;
#[cfg(ntt_rvv)]
use super::Rvv128;
#[cfg(ntt_simd128)]
use super::Simd128;
#[cfg(ntt_vsx)]
use super::Vsx;
#[cfg(ntt_vxe)]
use super::Vxe;
#[cfg(ntt_zvector)]
use super::Zvector;
#[cfg(ntt_rvv_wide)]
use super::{Rvv256, Rvv512};

impl Kernels {
    /// Resolves runtime CPU features at the complete-operation boundary.
    #[must_use]
    #[cfg(ntt_runtime_selection)]
    pub fn get() -> &'static Self {
        runtime_kernels().unwrap_or(&Self::SCALAR)
    }

    /// Selects the one table the target features guarantee.
    #[must_use]
    #[cfg(not(ntt_runtime_selection))]
    pub const fn get() -> &'static Self {
        #[cfg(ntt_ifma)]
        {
            &AVX512_IFMA
        }
        #[cfg(all(ntt_avx512, not(ntt_ifma)))]
        {
            &Avx512::KERNELS
        }
        #[cfg(ntt_avx2)]
        {
            &Avx2::KERNELS
        }
        #[cfg(ntt_neon)]
        {
            &Neon::KERNELS
        }
        #[cfg(ntt_vsx)]
        {
            &Vsx::KERNELS
        }
        #[cfg(ntt_rvv)]
        {
            &Rvv128::KERNELS
        }
        #[cfg(ntt_lasx)]
        {
            &Lasx::KERNELS
        }
        #[cfg(ntt_lsx)]
        {
            &Lsx::KERNELS
        }
        #[cfg(ntt_vxe)]
        {
            &Vxe::KERNELS
        }
        #[cfg(ntt_zvector)]
        {
            &Zvector::KERNELS
        }
        #[cfg(ntt_simd128)]
        {
            &Simd128::KERNELS
        }
        #[cfg(not(ntt_fixed_kernels))]
        {
            &Self::SCALAR
        }
    }
}

/// AVX-512F stages with integer CRT packets.
#[cfg(ntt_ifma)]
const AVX512_IFMA: Kernels = Kernels {
    crt_cost: 1,
    coefficients: Some(Ifma::KERNELS),
    ..Avx512::KERNELS
};

#[cfg(all(
    ntt_runtime_selection,
    any(target_arch = "x86", target_arch = "x86_64")
))]
fn runtime_kernels() -> Option<&'static Kernels> {
    if is_x86_feature_detected!("avx512f") {
        return Some(
            if is_x86_feature_detected!("avx512ifma") && is_x86_feature_detected!("avx512dq") {
                &AVX512_IFMA
            } else {
                &Avx512::KERNELS
            },
        );
    }
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return Some(&Avx2::KERNELS);
    }
    None
}

#[cfg(all(ntt_runtime_selection, target_arch = "s390x"))]
fn runtime_kernels() -> Option<&'static Kernels> {
    if is_s390x_feature_detected!("vector-enhancements-1") {
        return Some(&Vxe::KERNELS);
    }
    is_s390x_feature_detected!("vector").then_some(&Zvector::KERNELS)
}

#[cfg(all(ntt_runtime_selection, target_arch = "loongarch64"))]
fn runtime_kernels() -> Option<&'static Kernels> {
    if is_loongarch_feature_detected!("lasx") {
        return Some(&Lasx::KERNELS);
    }
    is_loongarch_feature_detected!("lsx").then_some(&Lsx::KERNELS)
}

#[cfg(all(ntt_runtime_selection, target_arch = "riscv64"))]
fn runtime_kernels() -> Option<&'static Kernels> {
    let bytes = Rvv128::register_bytes()?;
    Some(if bytes >= 64 {
        &Rvv512::KERNELS
    } else if bytes >= 32 {
        &Rvv256::KERNELS
    } else {
        &Rvv128::KERNELS
    })
}
