//! Kernels as macros over a backend's load, store, arithmetic, and exact-product
//! operations; `schedule_backend!` instantiates all of them.
//!
//! The portable kernels and the backends of intrinsic functions use it. A target
//! whose guaranteed backend is block text does not compile it in production, and
//! `arch/tests` includes it to build the portable reference.

#[macro_use]
mod finish;
#[macro_use]
mod packing;
#[macro_use]
mod root;
#[macro_use]
mod tail;
#[macro_use]
mod stages;
