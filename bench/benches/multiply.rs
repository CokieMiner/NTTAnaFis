//! Native-limb products on identical operands.
//!
//! `workspace_*` measures execution with retained tables. `scratch_*`
//! regenerates the root tables in caller-owned scratch on every call.
//! `gmp_*` runs the same products in GMP on 64-bit x86. `convolve_*` runs dense
//! convolutions of coefficients bounded by `2^BITS-1`, whose cost follows the
//! coefficient size rather than one limb per coefficient.

use divan::{Bencher, black_box};
use ntt_anafis::{ConvPlan, ConvWorkspace, MulPlan, MulWorkspace};
use ntt_anafis_bench::operand;

/// Operand lengths in native limbs.
const LENGTHS: [usize; 8] = [
    64,
    256,
    1 << 10,
    1 << 12,
    1 << 14,
    1 << 16,
    1 << 18,
    1 << 20,
];

fn main() {
    divan::main();
}

#[divan::bench(args = LENGTHS)]
fn workspace_mul(bencher: Bencher<'_, '_>, length: usize) {
    let (a, b) = (operand(length, 1), operand(length, 2));
    let mut dst = vec![0; 2 * length];
    let mut workspace = MulWorkspace::new();
    workspace.prepare(MulPlan::new(length, length).expect("benchmark shape"));
    bencher.bench_local(|| {
        assert!(workspace.mul(black_box(&mut dst), black_box(&a), black_box(&b)));
    });
}

#[divan::bench(args = LENGTHS)]
fn workspace_sqr(bencher: Bencher<'_, '_>, length: usize) {
    let a = operand(length, 1);
    let mut dst = vec![0; 2 * length];
    let mut workspace = MulWorkspace::new();
    workspace.prepare(MulPlan::new(length, length).expect("benchmark shape"));
    bencher.bench_local(|| {
        assert!(workspace.sqr(black_box(&mut dst), black_box(&a)));
    });
}

#[divan::bench(args = LENGTHS)]
fn scratch_mul(bencher: Bencher<'_, '_>, length: usize) {
    let (a, b) = (operand(length, 1), operand(length, 2));
    let mut dst = vec![0; 2 * length];
    let plan = MulPlan::new(length, length).expect("benchmark shape");
    let mut scratch = vec![0; plan.scratch_len()];
    bencher.bench_local(|| {
        assert!(plan.mul(
            black_box(&mut dst),
            black_box(&a),
            black_box(&b),
            &mut scratch
        ));
    });
}

/// Long and short operand lengths of products that may run in blocks.
const UNBALANCED: [(usize, usize); 4] = [
    (1 << 12, 1 << 6),
    (1 << 16, 1 << 6),
    (1 << 16, 1 << 10),
    (1 << 18, 1 << 10),
];

#[divan::bench(args = UNBALANCED)]
fn workspace_unbalanced(bencher: Bencher<'_, '_>, (long, short): (usize, usize)) {
    let (a, b) = (operand(long, 1), operand(short, 2));
    let mut dst = vec![0; long + short];
    let mut workspace = MulWorkspace::new();
    workspace.prepare(MulPlan::new(long, short).expect("benchmark shape"));
    bencher.bench_local(|| {
        assert!(workspace.mul(black_box(&mut dst), black_box(&a), black_box(&b)));
    });
}

/// Coefficient lengths, including shapes summed directly and shapes transformed,
/// with rows of `2^k` and of `3*2^k` points.
const CONVOLUTION_LENGTHS: [usize; 10] = [
    8,
    24,
    32,
    256,
    1 << 10,
    1 << 12,
    5000,
    1 << 16,
    80_000,
    1 << 18,
];

/// Returns `length` coefficients of at most `bits` bits and their bound.
fn coefficients(length: usize, seed: u64, bits: u32) -> (Vec<usize>, usize) {
    let bound = usize::MAX >> (usize::BITS - bits);
    let values = operand(length, seed)
        .into_iter()
        .map(|limb| limb & bound)
        .collect();
    (values, bound)
}

#[divan::bench(args = CONVOLUTION_LENGTHS, consts = [16, 32, 64])]
fn convolve_workspace<const BITS: u32>(bencher: Bencher<'_, '_>, length: usize) {
    let (a, bound) = coefficients(length, 1, BITS);
    let (b, _) = coefficients(length, 2, BITS);
    let plan = ConvPlan::new(length, length, bound, bound).expect("benchmark shape");
    let mut dst = vec![0; plan.output_len()];
    let mut workspace = ConvWorkspace::new();
    workspace.prepare(plan);
    bencher.bench_local(|| {
        assert!(workspace.convolve(black_box(&mut dst), black_box(&a), black_box(&b)));
    });
}

#[divan::bench(args = CONVOLUTION_LENGTHS, consts = [16, 32, 64])]
fn convolve_square<const BITS: u32>(bencher: Bencher<'_, '_>, length: usize) {
    let (a, bound) = coefficients(length, 1, BITS);
    let plan = ConvPlan::new(length, length, bound, bound).expect("benchmark shape");
    let mut dst = vec![0; plan.output_len()];
    let mut workspace = ConvWorkspace::new();
    workspace.prepare(plan);
    bencher.bench_local(|| {
        let values = black_box(&a);
        assert!(workspace.convolve(black_box(&mut dst), values, values));
    });
}

#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
mod gmp {
    use divan::{Bencher, black_box};
    use ntt_anafis_bench::operand;
    use rug::{Assign, Integer, integer::Order};

    use super::LENGTHS;

    #[divan::bench(args = LENGTHS)]
    fn gmp_mul(bencher: Bencher<'_, '_>, length: usize) {
        let a = Integer::from_digits(&operand(length, 1), Order::Lsf);
        let b = Integer::from_digits(&operand(length, 2), Order::Lsf);
        let mut dst = Integer::with_capacity(2 * length * 64);
        bencher.bench_local(|| {
            dst.assign(black_box(&a) * black_box(&b));
        });
    }

    #[divan::bench(args = LENGTHS)]
    fn gmp_sqr(bencher: Bencher<'_, '_>, length: usize) {
        let a = Integer::from_digits(&operand(length, 1), Order::Lsf);
        let mut dst = Integer::with_capacity(2 * length * 64);
        bencher.bench_local(|| {
            dst.assign(black_box(&a).square_ref());
        });
    }
}
