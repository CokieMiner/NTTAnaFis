//! Names what the target compiles as `cfg` flags.
//!
//! This file is the one definition of which vector backends and scalar
//! instructions exist on a target. `src/arch` gates on the names. Run-time
//! selection then chooses among the backends that compiled: with `std` a backend
//! compiles when the CPU may have it, and without `std` only when the target
//! features guarantee it, so at most one backend of each family compiles.

use std::env::{var, var_os};

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    let arch = var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let features = var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    let little_endian = var("CARGO_CFG_TARGET_ENDIAN").is_ok_and(|endian| endian == "little");
    let has = |name: &str| features.split(',').any(|feature| feature == name);
    let std = var_os("CARGO_FEATURE_STD").is_some();
    // Miri has no vector instructions, so it runs the portable kernels.
    let native = var_os("CARGO_CFG_MIRI").is_none();
    let on = |name: &str| native && arch == name;
    let x86 = on("x86") || on("x86_64");

    // Vector backends.
    let backends = [
        (
            "ntt_avx2",
            x86 && (std || (!has("avx512f") && has("avx2") && has("fma"))),
        ),
        ("ntt_avx512", x86 && (std || has("avx512f"))),
        (
            "ntt_ifma",
            x86 && (std || (has("avx512f") && has("avx512ifma") && has("avx512dq"))),
        ),
        ("ntt_neon", on("aarch64") && has("neon")),
        ("ntt_zarch", on("s390x") && (std || has("vector"))),
        (
            "ntt_zvector",
            on("s390x") && (std || (has("vector") && !has("vector-enhancements-1"))),
        ),
        (
            "ntt_vxe",
            on("s390x") && (std || has("vector-enhancements-1")),
        ),
        (
            "ntt_lsx",
            on("loongarch64") && (std || (has("lsx") && !has("lasx"))),
        ),
        ("ntt_lasx", on("loongarch64") && (std || has("lasx"))),
        ("ntt_vsx", on("powerpc64") && (has("vsx") || little_endian)),
        ("ntt_rvv", on("riscv64") && (std || has("v"))),
        // Wider registers are known only at run time.
        ("ntt_rvv_wide", on("riscv64") && std),
    ];
    for (name, enabled) in backends {
        flag(name, enabled);
    }
    let runtime = std && (x86 || on("s390x") || on("loongarch64") || on("riscv64"));
    flag("ntt_runtime_selection", runtime);
    // The target guarantees a backend, so production does not compile the portable
    // kernels.
    let fixed = !runtime && backends.iter().any(|&(_, enabled)| enabled);
    flag("ntt_fixed_kernels", fixed);
    // The portable kernels and intrinsic backends instantiate the expression
    // schedule; a guaranteed backend on the other targets is block text.
    flag(
        "ntt_expression_schedule",
        !fixed || x86 || on("aarch64") || on("s390x"),
    );

    // Scalar instructions of the exact modular product.
    let std_on = |name: &str| std && arch == name;
    let neon = on("aarch64") && has("neon");
    flag("ntt_x86_fma", x86 && has("fma"));
    flag("ntt_x86_sse41", x86 && has("sse4.1"));
    // A hardware fused multiply-add, through intrinsics or `f64::mul_add`.
    let std_fused = std_on("powerpc")
        || std_on("powerpc64")
        || std_on("riscv64")
        || std_on("s390x")
        || (std_on("loongarch64") && has("d"))
        || (std_on("arm") && has("vfp4"));
    flag("ntt_fused_scalar", (x86 && has("fma")) || neon || std_fused);
    // Rounding to the nearest float integer, through intrinsics or `std`.
    flag("ntt_nearest_intrinsic", (x86 && has("sse4.1")) || neon);
    flag(
        "ntt_nearest_std",
        std_on("s390x")
            || std_on("wasm32")
            || (std_on("loongarch64") && has("d"))
            || (std_on("powerpc64") && (has("vsx") || little_endian))
            || (std_on("riscv64") && has("zfa"))
            || (std_on("arm") && has("fp-armv8")),
    );
    // Rounding conversion to an integer through a dedicated instruction.
    flag(
        "ntt_nearest_integer_intrinsic",
        on("x86_64") || neon || (on("x86") && has("sse4.1")),
    );
    // Native 64-bit integers: exact remainders and canonical residues use wrapping
    // integer arithmetic instead of float splits and selects.
    let width = var("CARGO_CFG_TARGET_POINTER_WIDTH").unwrap_or_default();
    let family = var("CARGO_CFG_TARGET_FAMILY").unwrap_or_default();
    let wide_integers = width == "64" || family.split(',').any(|name| name == "wasm");
    flag("ntt_wide_integers", wide_integers);
    // The product needs no unsafe operation: `std` has its fused instruction, or
    // a 32-bit target splits it with Dekker's product.
    let x86_intrinsics = on("x86") && (has("fma") || has("sse4.1"));
    flag(
        "ntt_safe_scalar",
        std_fused || (!wide_integers && !x86_intrinsics),
    );
}

/// Declares a flag and sets it when its condition holds.
fn flag(name: &str, enabled: bool) {
    println!("cargo::rustc-check-cfg=cfg({name})");
    if enabled {
        println!("cargo::rustc-cfg={name}");
    }
}
