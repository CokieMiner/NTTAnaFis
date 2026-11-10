#![expect(
    unsafe_code,
    reason = "Tests admit detected kernels over guarded eight-span allocations and verify initialization of omitted inputs"
)]

use core::{
    mem::MaybeUninit,
    num::NonZeroUsize,
    ops::Range,
    ptr::null,
    slice::{from_raw_parts, from_ref},
};

use alloc::{vec, vec::Vec};

use core_anafis::Sequential;
use proptest::{collection, prelude::*, test_runner::TestRunner};

use crate::{
    PRIME_CATALOG, Prime,
    arch::{Columns, Kernels, Scalar},
    transform::Twiddles,
};

use super::admitted;

#[test]
fn unit_root_stages_match_separate_stages_and_initialize_omitted_inputs() {
    let check = |words: &[u64]| {
        for prime in PRIME_CATALOG
            .iter()
            .take(if cfg!(miri) { 1 } else { PRIME_CATALOG.len() })
        {
            for stride in [8_usize, 16, 64] {
                if cfg!(miri) && stride != 16 {
                    continue;
                }
                let mut spans = vec![
                    0..stride,
                    0..8,
                    stride.checked_sub(8).expect("span")..stride,
                ];
                spans.sort_unstable_by_key(|span| (span.start, span.end));
                spans.dedup();
                for inputs in [1, 2, 3, 4] {
                    for span in &spans {
                        check_root(prime, words, stride, inputs, span);
                    }
                }
            }
        }
    };
    check(&[0, 1, u64::MAX]);
    TestRunner::new(ProptestConfig::with_cases(if cfg!(miri) { 1 } else { 8 }))
        .run(&collection::vec(any::<u64>(), 1..=32), |words| {
            check(&words);
            Ok(())
        })
        .expect("unit root stage property");
}

#[expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "Signed test residues have magnitude below p < 2^50 and are exact binary64 integers"
)]
fn check_root(prime: &Prime, words: &[u64], stride: usize, inputs: usize, span: &Range<usize>) {
    let mut forward = [0.0; 4];
    let mut backward = [0.0; 4];
    // SAFETY: one power-of-two stride per prime, within the catalog roots.
    unsafe {
        Twiddles::prepare(
            &mut forward,
            &mut backward,
            NonZeroUsize::new(4).expect("positive stride"),
            from_ref(prime),
            &Kernels::SCALAR,
            &Sequential,
        );
    }
    let packet = [forward[1], forward[2], forward[3]];
    let radius = prime.modulus().checked_sub(1).expect("positive radius");
    let range = radius
        .checked_mul(2)
        .and_then(|v| v.checked_add(1))
        .expect("51-bit range");
    let len = stride.checked_mul(8).expect("test allocation width");
    let active = stride.checked_mul(inputs).expect("active input width");
    let source: Vec<_> = words
        .iter()
        .cycle()
        .take(len)
        .enumerate()
        .map(|(position, &word)| {
            if position < active {
                let residue = match word {
                    1 => radius,
                    u64::MAX => range.checked_sub(1).expect("last residue"),
                    _ => word.checked_rem(range).expect("positive residue range"),
                };
                i64::try_from(residue)
                    .expect("51-bit position")
                    .checked_sub(i64::try_from(radius).expect("50-bit radius"))
                    .expect("signed residue") as f64
            } else {
                0.0
            }
        })
        .collect();
    let mut expected = source.clone();
    separate_stages(&mut expected, &packet, stride, span, prime);
    for backend in admitted() {
        let kernel = *backend
            .dif8_prefix
            .get(inputs.checked_sub(1).expect("positive prefix arity"))
            .expect("one to four input spans");
        let mut actual = guarded_input(&source, active, stride, span);
        // SAFETY: eight disjoint spans fit the allocation; active inputs
        // satisfy the packed bound. Omitted selected slots remain
        // uninitialized and represent zero. Roots are complete and shared,
        // the selected kernel admits this CPU, and the requested range fits
        // with endpoints at multiples of eight.
        unsafe {
            kernel(
                actual.as_mut_ptr().add(1).cast(),
                packet.as_ptr(),
                stride,
                Columns::new(span.start, span.end),
                prime.float_field(),
            );
        }
        // SAFETY: guards and untouched slots were initialized above. The
        // kernel initializes every selected output, including omitted input
        // slots. MaybeUninit<f64> has the layout and alignment of f64.
        let values = unsafe { from_raw_parts(actual.as_ptr().cast::<f64>(), actual.len()) };
        assert!(values.first().expect("prefix guard").is_nan());
        assert!(values.last().expect("suffix guard").is_nan());
        for (position, (&value, &reference)) in
            values.iter().skip(1).take(len).zip(&expected).enumerate()
        {
            if span.contains(&position.checked_rem(stride).expect("positive stride")) {
                assert!(value.abs() < 3.0 * prime.float_field().p());
                assert_eq!(
                    Scalar::reduce(value, prime.float_field().p(), prime.float_field().pinv()),
                    Scalar::reduce(
                        reference,
                        prime.float_field().p(),
                        prime.float_field().pinv()
                    ),
                    "stride={stride}, inputs={inputs}, span={span:?}, position={position}"
                );
            } else if position < active {
                assert_eq!(value.to_bits(), reference.to_bits());
            } else {
                assert!(value.is_nan());
            }
        }
    }
}

fn separate_stages(
    values: &mut [f64],
    roots: &[f64; 3],
    stride: usize,
    span: &Range<usize>,
    prime: &Prime,
) {
    let half = stride.checked_mul(4).expect("parent half");
    for column in 0_usize..4 {
        let offset = column.checked_mul(stride).expect("parent column");
        let start = offset.checked_add(span.start).expect("parent start");
        let end = offset.checked_add(span.end).expect("parent end");
        // SAFETY: both parent halves are initialized. Each column range
        // stays inside the half, and the unit radix-2 ignores its root pointer.
        unsafe {
            Scalar::radix2::<false>(
                values.as_mut_ptr(),
                null(),
                half,
                Columns::new(start, end),
                prime.float_field(),
            );
        }
    }
    let stages = [
        Scalar::radix4::<false, true>,
        Scalar::radix4::<false, false>,
    ];
    for ((row, child), kernel) in values.chunks_exact_mut(half).enumerate().zip(stages) {
        let packet = if row == 0 {
            [1.0, 1.0, roots[0]]
        } else {
            *roots
        };
        // SAFETY: this child has four initialized spans, the admitted
        // unit/generic root packet, and a range inside every span.
        unsafe {
            kernel(
                child.as_mut_ptr(),
                packet.as_ptr(),
                stride,
                Columns::new(span.start, span.end),
                prime.float_field(),
            );
        }
    }
}

fn guarded_input(
    source: &[f64],
    active: usize,
    stride: usize,
    span: &Range<usize>,
) -> Vec<MaybeUninit<f64>> {
    let len = source.len();
    let mut guarded = vec![MaybeUninit::uninit(); len.checked_add(2).expect("guarded allocation")];
    *guarded.first_mut().expect("prefix guard") = MaybeUninit::new(f64::NAN);
    *guarded.last_mut().expect("suffix guard") = MaybeUninit::new(f64::NAN);
    for (position, slot) in guarded.iter_mut().skip(1).take(len).enumerate() {
        if position < active {
            *slot = MaybeUninit::new(*source.get(position).expect("input slot"));
        } else if !span.contains(&position.checked_rem(stride).expect("positive stride")) {
            *slot = MaybeUninit::new(f64::NAN);
        }
    }
    guarded
}
