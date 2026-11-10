//! Reduction of shared binary digits into one or several prime fields.

macro_rules! packing_backend {
    ($name:ident, $vector:ty, $lanes:literal, [$($attribute:meta),*],
     $load:expr, $store:expr, $splat:expr, $add:expr, $mul:expr,
     $reduce:expr, $multiply:expr) => {
        impl $name {
            /// Reduces consecutive coefficients of each selected prime in lanes.
            ///
            /// # Safety
            /// The kernel contract of `PackingKernel` holds with `PARTS` parts.
            /// `SHORT_HIGH` requires at least two parts and a final digit below eight.
            $(#[$attribute])*
            pub unsafe fn pack<const SHORT_HIGH: bool, const PARTS: usize>(
                packing: &$crate::Packing<'_>,
                columns: $crate::arch::Columns,
                digits: &$crate::DigitTile,
            ) {
                const {
                    assert!(
                        PARTS > 0 && PARTS <= $crate::MAX_PARTS && (PARTS > 1 || !SHORT_HIGH),
                        "packing part count"
                    );
                };
                // SAFETY: the caller admits this ISA. Each selected prime lies in
                // the catalog. Complete vectors fit the digit tile and the owned
                // nonempty columns. Row offsets stay within count*stride output slots.
                unsafe {
                    let (first, end) = columns.complete::<$lanes>();
                    let len = end.unchecked_sub(first);
                    let data = &$crate::PACKING_DATA;
                    for prime in 0..packing.prefix.count() {
                        let modulus = ($splat)(*data.moduli.get_unchecked(prime));
                        let reciprocal = ($splat)(*data.reciprocals.get_unchecked(prime));
                        let powers: [$vector; PARTS] = core::array::from_fn(|part| {
                            ($splat)(*data.powers.get_unchecked(part).get_unchecked(prime))
                        });
                        let output = packing
                            .output
                            .add(prime.unchecked_mul(packing.stride).unchecked_add(first));
                        let mut index = 0_usize;
                        loop {
                            let digit_at = |part: usize| {
                                ($load)(
                                    digits
                                        .as_ptr()
                                        .add(part.unchecked_mul($crate::PACKING_TILE).unchecked_add(index)),
                                )
                            };
                            let sum = packing_sum::<SHORT_HIGH, PARTS, _>(
                                digit_at,
                                &powers,
                                modulus,
                                reciprocal,
                            );
                            ($store)(output.add(index), sum);
                            index = index.unchecked_add($lanes);
                            if index == len {
                                break;
                            }
                        }
                    }
                }
            }
        }

        // Always inlined into the packing kernel, whose target features the
        // body then compiles with; see the butterfly in `stages`.
        #[inline(always)]
        unsafe fn packing_sum<const SHORT_HIGH: bool, const PARTS: usize, D>(
            digit_at: D,
            powers: &[$vector; PARTS],
            modulus: $vector,
            reciprocal: $vector,
        ) -> $vector
        where
            D: Fn(usize) -> $vector,
        {
            // SAFETY: callers are packing kernels of this backend, whose
            // target features admit every ISA operation of the inlined body.
            // Their loaders admit every part in 0..PARTS. The first digit
            // lies below 2^49 < p. A 49-bit digit times a centered power has
            // a product residue below p, so a digit and three products stay
            // below 4*p. A three-bit final digit times a centered power is
            // exact below 7*p/2; the short sum stays below 13*p/2 < 2^53, so
            // all additions remain exact integers. One reduction centers it.
            unsafe {
                let mut sum = digit_at(0);
                if PARTS > 1 {
                    let last = PARTS.unchecked_sub(1);
                    for part in 1..last {
                        sum = ($add)(
                            sum,
                            ($multiply)(digit_at(part), *powers.get_unchecked(part), modulus, reciprocal),
                        );
                    }
                    let digit = digit_at(last);
                    let power = *powers.get_unchecked(last);
                    let product = if SHORT_HIGH {
                        ($mul)(digit, power)
                    } else {
                        ($multiply)(digit, power, modulus, reciprocal)
                    };
                    sum = ($reduce)(($add)(sum, product), modulus, reciprocal);
                }
                sum
            }
        }
    };
}
