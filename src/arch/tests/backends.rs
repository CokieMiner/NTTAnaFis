extern crate std;

#[cfg(all(ntt_lsx, feature = "std"))]
use std::arch::is_loongarch_feature_detected;
#[cfg(all(ntt_zvector, feature = "std"))]
use std::arch::is_s390x_feature_detected;
#[cfg(ntt_avx2)]
use std::is_x86_feature_detected;

use alloc::{vec, vec::Vec};

use crate::arch::Kernels;

pub fn admitted() -> Vec<Kernels> {
    let mut backends = vec![Kernels::SCALAR];
    if !cfg!(miri) {
        backends.push(*Kernels::get());
    }
    #[cfg(ntt_avx2)]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        use crate::arch::backend::Avx2;

        backends.push(Avx2::KERNELS);
    }
    #[cfg(all(ntt_lsx, feature = "std"))]
    if is_loongarch_feature_detected!("lsx") {
        use crate::arch::backend::Lsx;

        backends.push(Lsx::KERNELS);
    }
    #[cfg(all(ntt_zvector, feature = "std"))]
    if is_s390x_feature_detected!("vector") {
        use crate::arch::backend::Zvector;

        backends.push(Zvector::KERNELS);
    }
    #[cfg(ntt_rvv_wide)]
    {
        use crate::arch::backend::Rvv128;

        if Rvv128::register_bytes().is_some() {
            backends.push(Rvv128::KERNELS);
        }
    }
    backends
}
