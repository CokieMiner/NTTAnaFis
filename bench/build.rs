//! Links the `reference` benchmark to the libraries built for this CPU.

use std::env::{var, var_os};

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-env-changed=REFERENCE_PREFIX");
    if var_os("CARGO_FEATURE_REFERENCE").is_some() {
        let prefix = var("REFERENCE_PREFIX").expect("the reference feature needs REFERENCE_PREFIX");
        // Full paths, because the search path of a dependency holds another GMP.
        for library in ["flint", "mpfr", "gmp"] {
            println!("cargo::rustc-link-arg-benches={prefix}/lib/lib{library}.a");
        }
        println!("cargo::rustc-link-arg-benches=-lm");
    }
}
