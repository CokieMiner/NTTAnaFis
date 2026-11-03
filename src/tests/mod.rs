//! Planning, execution policies, and products checked against schoolbook arithmetic.

mod convolution;
mod planning;
mod plans;
mod products;
#[cfg(feature = "rayon")]
mod scheduling;
mod schoolbook;
