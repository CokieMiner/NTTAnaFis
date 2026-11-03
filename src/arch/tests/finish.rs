#![expect(
    unsafe_code,
    reason = "Guarded allocations and admitted backend tables establish final-stage span and residue contracts"
)]

use core::{num::NonZeroUsize, ptr::null, slice::from_ref};

use alloc::{vec, vec::Vec};

use core_anafis::Sequential;
use proptest::{collection, prelude::*, test_runner::TestRunner};

use crate::{
    PRIME_CATALOG, Prime,
    arch::{Columns, Kernels, Scalar},
    transform::{Normalization, Transform, Twiddles},
};

use super::admitted;

#[test]
fn fused_final_inverse_matches_separate_normalization_across_spans_prefixes_and_backends() {
    let check = |words: &[u64]| {
        for prime in PRIME_CATALOG
            .iter()
            .take(if cfg!(miri) { 1 } else { PRIME_CATALOG.len() })
        {
            for stride in [8_usize, 64, 512, 1024, 8192, 16_384] {
                if cfg!(miri) && stride != 8 {
                    continue;
                }
                check_tiles(prime, words, stride);
            }
        }
    };
    check(&[0, 1, u64::MAX]);
    TestRunner::new(ProptestConfig::with_cases(if cfg!(miri) { 1 } else { 2 }))
        .run(&collection::vec(any::<u64>(), 1..=8), |words| {
            check(&words);
            Ok(())
        })
        .expect("final inverse tile property");
}

#[expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "Test inputs are signed integers below 2*p in magnitude; canonical factors are below 2^50"
)]
fn check_tiles(prime: &Prime, words: &[u64], stride: usize) {
    for spans in [2_usize, 4, 8] {
        let len = stride.checked_mul(spans).expect("test span");
        let (source, inverse, expected) = reference_inverse(prime, words, stride, spans);
        for factor in [
            0.0,
            1.0,
            prime.modulus().div_euclid(2) as f64,
            prime
                .modulus()
                .div_euclid(2)
                .checked_add(1)
                .expect("centered boundary") as f64,
            prime.modulus().checked_sub(1).expect("positive factor") as f64,
        ] {
            let mut normalized = expected.clone();
            // SAFETY: the reference stages initialize every coefficient below
            // 4*p; factor is canonical and the exclusive output has len slots.
            // Each residue is reduced first, inside the kernel's input bound.
            unsafe {
                for value in &mut normalized {
                    *value =
                        Scalar::reduce(*value, prime.float_field().p(), prime.float_field().pinv());
                }
                Scalar::residues(
                    normalized.as_mut_ptr(),
                    Columns::new(0, len),
                    factor,
                    prime.float_field(),
                );
            }
            for backend in admitted() {
                let mut prefixes = vec![8, stride, len];
                if len > 8 {
                    prefixes.push(len.checked_sub(8).expect("positive span"));
                }
                prefixes.push(stride.checked_add(8).expect("prefix fits"));
                for boundary in
                    (1..spans).map(|span| span.checked_mul(stride).expect("span boundary"))
                {
                    prefixes.push(boundary);
                    if boundary > 8 {
                        prefixes.push(boundary.checked_sub(8).expect("last column"));
                    }
                    prefixes.push(boundary.checked_add(8).expect("first column"));
                }
                prefixes.retain(|&prefix| prefix <= len);
                prefixes.sort_unstable();
                prefixes.dedup();
                for prefix in prefixes {
                    let mut actual = vec![f64::NAN; len.checked_add(2).expect("guards fit")];
                    let values = actual
                        .get_mut(1..len.checked_add(1).expect("suffix fits"))
                        .expect("inner allocation");
                    values.copy_from_slice(&source);
                    let transform = Transform::new(prime, &backend, &Sequential, false);
                    // SAFETY: every prefix is a positive multiple of COLUMN_ALIGN.
                    let columns = unsafe { Columns::new(0, prefix) };
                    let normalization = Normalization::new(columns, factor, prime);
                    let budget = NonZeroUsize::new(3).expect("positive budget");
                    match spans {
                        2 => transform.finish_stage::<2>(values, &inverse, normalization, budget),
                        4 => transform.finish_stage::<4>(values, &inverse, normalization, budget),
                        _ => transform.finish_stage::<8>(values, &inverse, normalization, budget),
                    }
                    for (position, &value) in values.iter().take(prefix).enumerate() {
                        assert_eq!(
                            value.to_bits(),
                            normalized
                                .get(position)
                                .expect("reference prefix")
                                .to_bits(),
                            "spans={spans}, stride={stride}, prefix={prefix}"
                        );
                    }
                    for (&value, &original) in values.iter().zip(&source).skip(prefix) {
                        assert_eq!(
                            value.to_bits(),
                            original.to_bits(),
                            "unused inverse suffix is untouched"
                        );
                    }
                    assert!(actual.first().expect("prefix guard").is_nan());
                    assert!(actual.last().expect("suffix guard").is_nan());
                }
            }
        }
    }
}

#[expect(
    clippy::as_conversions,
    clippy::cast_precision_loss,
    reason = "Signed reference inputs have magnitude below 2*p < 2^51 and convert exactly to binary64"
)]
fn reference_inverse(
    prime: &Prime,
    words: &[u64],
    stride: usize,
    spans: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let len = stride.checked_mul(spans).expect("test span");
    let mut forward = vec![0.0; len >> 1];
    let mut inverse = forward.clone();
    // SAFETY: one power-of-two stride per prime, within the catalog roots.
    unsafe {
        Twiddles::prepare(
            &mut forward,
            &mut inverse,
            NonZeroUsize::new(len >> 1).expect("positive stride"),
            from_ref(prime),
            &Kernels::SCALAR,
            &Sequential,
        );
    }
    let root = |index| *inverse.get(index).expect("admitted root");
    let radius = prime
        .modulus()
        .checked_mul(2)
        .and_then(|v| v.checked_sub(1))
        .expect("lazy radius");
    let range = radius
        .checked_mul(2)
        .and_then(|v| v.checked_add(1))
        .expect("signed residue range");
    let source: Vec<_> = words
        .iter()
        .cycle()
        .take(len)
        .map(|word| {
            i64::try_from(word.checked_rem(range).expect("positive range"))
                .expect("52-bit residue")
                .checked_sub(i64::try_from(radius).expect("51-bit radius"))
                .expect("signed residue") as f64
        })
        .collect();
    let mut expected = source.clone();
    let field = prime.float_field();
    // SAFETY: reference spans cover len initialized lazy residues, each packet
    // has its unit or generic root constants, and the complete strides are
    // admitted. The radix-2 root reads children below 2*p.
    unsafe {
        let columns = Columns::new(0, stride);
        if spans == 2 {
            Scalar::radix2::<false>(expected.as_mut_ptr(), null(), stride, columns, field);
        } else if spans == 4 {
            let packet = [1.0, 1.0, root(1)];
            Scalar::radix4::<true, true>(
                expected.as_mut_ptr(),
                packet.as_ptr(),
                stride,
                columns,
                field,
            );
        } else {
            let half = stride.checked_mul(4).expect("parent half");
            let packet = [1.0, 1.0, root(1)];
            let upper = [root(1), root(2), root(3)];
            Scalar::radix4::<true, true>(
                expected.as_mut_ptr(),
                packet.as_ptr(),
                stride,
                columns,
                field,
            );
            Scalar::radix4::<true, false>(
                expected.as_mut_ptr().add(half),
                upper.as_ptr(),
                stride,
                columns,
                field,
            );
            Scalar::radix2::<false>(
                expected.as_mut_ptr(),
                null(),
                half,
                Columns::new(0, half),
                field,
            );
        }
    }
    (source, inverse, expected)
}
