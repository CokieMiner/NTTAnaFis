//! Portable residue kernels: the schedule with one lane, and sixteen-point leaves.
//!
//! A target that guarantees a vector backend does not compile this file in
//! production, and `arch/tests` includes it as the reference for every backend.

#![expect(
    unsafe_code,
    reason = "Kernel callers prove initialized spans, exclusive ownership, and residue bounds"
)]

use core::{
    num::NonZeroUsize,
    ptr::{copy_nonoverlapping, read, write},
};

#[cfg(not(ntt_wide_integers))]
use super::BIAS;
use super::{FloatField, Kernels, Scalar};

// Canonical inputs are products or reductions, integers in (-p, p). With native
// 64-bit integers they convert exactly, and a sign mask adds p without a select,
// which targets without a float select would compile to a branch on the sign.
schedule_backend!(
    Scalar,
    f64,
    1,
    [],
    read::<f64>,
    write::<f64>,
    |value| value,
    |a, b| a + b,
    |a, b| a - b,
    |a, b| a * b,
    Scalar::reduce,
    Scalar::mulmod,
    |value: f64, p: f64| {
        #[cfg(ntt_wide_integers)]
        {
            let integer = value.to_int_unchecked::<i64>();
            let modulus = p.to_int_unchecked::<i64>();
            f64::from_bits(
                integer
                    .wrapping_add((integer >> 63) & modulus)
                    .cast_unsigned(),
            )
        }
        #[cfg(not(ntt_wide_integers))]
        {
            let canonical = if value < 0.0 { value + p } else { value };
            f64::from_bits((canonical + BIAS).to_bits() ^ BIAS.to_bits())
        }
    }
);

impl Kernels {
    /// Portable kernels with the same geometry and residue contracts as SIMD backends.
    pub const SCALAR: Self = kernel_table!(Scalar, Scalar, None, None, None);
}

impl Scalar {
    /// Executes forward leaves or inverse leaves with optional products.
    ///
    /// # Safety
    /// The contract of `LeafKernel` holds with residues below `3*p`, or `2*p` for
    /// the inverse. Forward outputs are below `3*p/2`.
    pub unsafe fn leaf16<const INVERSE: bool, const PRODUCT: bool, const SQUARE: bool>(
        values: *mut f64,
        right: *const f64,
        twiddles: *const f64,
        pairs: NonZeroUsize,
        node: usize,
        field: FloatField,
    ) {
        let (p, pinv) = (field.p(), field.pinv());
        // SAFETY: each block has sixteen coefficients; its largest root
        // index is 8*(node+block)+7, inside the admitted table.
        // The fixed row and column loops stay below four, and every row slot
        // is initialized before either butterfly sequence reads it. Inverse
        // products are below 9*p^2/4 and reduce below 3*p/2. Multiplication
        // commutes with the common transpose of both spectra. The portable
        // butterfly has no target features to admit.
        unsafe {
            let blocks = pairs.get().unchecked_mul(2);
            let mut block = 0_usize;
            loop {
                let base = values.add(block.unchecked_mul(16));
                let current = node.unchecked_add(block);
                let twice = current.unchecked_mul(2);
                let top = [
                    *twiddles.add(current),
                    *twiddles.add(twice),
                    *twiddles.add(twice.unchecked_add(1)),
                ];
                let child = current.unchecked_mul(4);
                let mut rows = [[0.0; 4]; 4];
                for (row, slots) in rows.iter_mut().enumerate() {
                    if INVERSE {
                        for (column, slot) in slots.iter_mut().enumerate() {
                            let position = column.unchecked_mul(4).unchecked_add(row);
                            let value = *base.add(position);
                            *slot = if PRODUCT {
                                let factor = if SQUARE {
                                    value
                                } else {
                                    *right.add(block.unchecked_mul(16).unchecked_add(position))
                                };
                                Self::mulmod(value, factor, p, pinv)
                            } else {
                                value
                            };
                        }
                    } else {
                        copy_nonoverlapping(base.add(row.unchecked_mul(4)), slots.as_mut_ptr(), 4);
                    }
                }
                let child_roots = |index: usize| {
                    let child_node = child.unchecked_add(index);
                    let twice_child = child_node.unchecked_mul(2);
                    [
                        *twiddles.add(child_node),
                        *twiddles.add(twice_child),
                        *twiddles.add(twice_child.unchecked_add(1)),
                    ]
                };
                if INVERSE {
                    for (index, row) in rows.iter_mut().enumerate() {
                        *row = butterfly::<true, false, false>(
                            (*row).into(),
                            child_roots(index).into(),
                            p,
                            pinv,
                        )
                        .into();
                    }
                }
                for column in 0_usize..4 {
                    let inputs = rows.map(|row| *row.get_unchecked(column));
                    let outputs: [f64; 4] =
                        butterfly::<INVERSE, false, false>(inputs.into(), top.into(), p, pinv)
                            .into();
                    for (row, output) in rows.iter_mut().zip(outputs) {
                        *row.get_unchecked_mut(column) = output;
                    }
                }
                if !INVERSE {
                    for (index, row) in rows.iter_mut().enumerate() {
                        *row = butterfly::<false, false, true>(
                            (*row).into(),
                            child_roots(index).into(),
                            p,
                            pinv,
                        )
                        .into();
                    }
                }
                for (row, slots) in rows.iter().enumerate() {
                    if INVERSE {
                        copy_nonoverlapping(slots.as_ptr(), base.add(row.unchecked_mul(4)), 4);
                    } else {
                        for (column, &slot) in slots.iter().enumerate() {
                            *base.add(column.unchecked_mul(4).unchecked_add(row)) = slot;
                        }
                    }
                }
                block = block.unchecked_add(1);
                if block == blocks {
                    break;
                }
            }
        }
    }
}
