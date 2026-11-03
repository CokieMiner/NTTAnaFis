//! The kernel table of one backend.

/// Builds a kernel table from the kernels of `$owner`.
///
/// `$leaf` supplies the sixteen-point leaves, `$coefficients` the integer CRT
/// packets, and `$dif_wide` and `$dit_wide` the paired stages, where they exist.
/// A CRT term costs one transform point unless the backend overrides `crt_cost`.
macro_rules! kernel_table {
    ($owner:ident, $leaf:ident, $coefficients:expr, $dif_wide:expr, $dit_wide:expr) => {
        $crate::arch::Kernels {
            crt_cost: 1,
            coefficients: $coefficients,
            pack: [
                $owner::pack::<false, 1>,
                $owner::pack::<false, 2>,
                $owner::pack::<false, 3>,
                $owner::pack::<false, 4>,
            ],
            pack_short: [
                $owner::pack::<true, 2>,
                $owner::pack::<true, 3>,
                $owner::pack::<true, 4>,
            ],
            radix2: $owner::radix2::<false>,
            duplicate_half: $owner::radix2::<true>,
            dif8_prefix: [
                $owner::root8::<1>,
                $owner::root8::<2>,
                $owner::root8::<3>,
                $owner::root8::<4>,
            ],
            dif4_prefix: [
                $owner::radix4_prefix::<1>,
                $owner::radix4_prefix::<2>,
                $owner::radix4_prefix::<3>,
            ],
            dif: $owner::radix4::<false, false>,
            dit: $owner::radix4::<true, false>,
            dif_unit: $owner::radix4::<false, true>,
            dit_unit: $owner::radix4::<true, true>,
            finish2: [$owner::finish::<2, 1>, $owner::finish::<2, 2>],
            finish4: [
                $owner::finish::<4, 1>,
                $owner::finish::<4, 2>,
                $owner::finish::<4, 3>,
                $owner::finish::<4, 4>,
            ],
            finish8: [
                $owner::finish::<8, 1>,
                $owner::finish::<8, 2>,
                $owner::finish::<8, 3>,
                $owner::finish::<8, 4>,
                $owner::finish::<8, 5>,
                $owner::finish::<8, 6>,
                $owner::finish::<8, 7>,
                $owner::finish::<8, 8>,
            ],
            dif2: $owner::radix2_node::<false>,
            dit2: $owner::radix2_node::<true>,
            fold: $owner::fold,
            recombine: $owner::recombine::<true>,
            recombine_suffix: $owner::recombine::<false>,
            dif_wide: $dif_wide,
            dit_wide: $dit_wide,
            dif16: $leaf::leaf16::<false, false, false>,
            dit16: $leaf::leaf16::<true, false, false>,
            dit16_mul: $leaf::leaf16::<true, true, false>,
            dit16_sqr: $leaf::leaf16::<true, true, true>,
            multiply: $owner::pointwise::<false>,
            square: $owner::pointwise::<true>,
            residues: $owner::residues,
            scale: $owner::scale,
        }
    };
}
