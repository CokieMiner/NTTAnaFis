//! Planning, execution policies, and products checked against schoolbook arithmetic.

mod convolution;
mod plans;
mod products;
#[cfg(feature = "rayon")]
mod scheduling;
mod schoolbook;
