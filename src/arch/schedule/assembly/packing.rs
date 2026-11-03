//! Reduction of shared binary digits into one or several prime fields.

macro_rules! assembly_packing {
    ($name:ident, $isa:ident, $run:ident, $lanes:literal, $pointer:ident) => {
        impl $name {
            /// Reduces consecutive coefficients of each selected prime in vector lanes.
            ///
            /// # Safety
            /// The contract of `PackingKernel` holds with `PARTS` parts. `SHORT_HIGH`
            /// requires at least two parts and a final digit below eight.
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
                // SAFETY: each selected prime lies in the catalog; part indices
                // are clipped below PARTS. Complete vectors fit the digit tile
                // and the owned nonempty coefficient span. Row offsets stay within
                // count*stride output slots. The first digit lies below
                // 2^49 < p. A 49-bit digit times a centered power has a product
                // residue below p, so a digit and three products stay below
                // 4*p. A three-bit final digit times a centered power is exact
                // below 7*p/2; the short sum stays below 13*p/2 < 2^53, so all
                // additions remain exact integers. One reduction centers it.
                unsafe {
                    let (first, end) = columns.complete::<$lanes>();
                    let len = end.unchecked_sub(first);
                    let last = PARTS.unchecked_sub(1);
                    let data = &$crate::PACKING_DATA;
                    let digit = |part: usize| {
                        digits.as_ptr().add(part.min(last).unchecked_mul($crate::PACKING_TILE))
                    };
                    for prime in 0..packing.prefix.count() {
                        let power = |part: usize| {
                            *data.powers.get_unchecked(part.min(last)).get_unchecked(prime)
                        };
                        let constants = [
                            *data.moduli.get_unchecked(prime),
                            *data.reciprocals.get_unchecked(prime),
                            power(1),
                            power(2),
                            power(last),
                        ];
                        let output = packing
                            .output
                            .add(prime.unchecked_mul(packing.stride).unchecked_add(first));
                        $run!(
                            $isa!(enter),
                            $isa!(splat 22, "{c}", 0),
                            $isa!(splat 23, "{c}", 8),
                            $isa!(splat 12, "{c}", 16),
                            $isa!(splat 13, "{c}", 24),
                            $isa!(splat 14, "{c}", 32),
                            "2:",
                            $isa!(load 0, "{d0}", 0),
                            ".if {parts} > 1",
                            ".if {parts} > 2",
                            $isa!(load 1, "{d1}", 0),
                            asm_multiply!($isa, 1, 1, 12),
                            $isa!(add 0, 0, 1),
                            $isa!(step "{d1}"),
                            ".endif",
                            ".if {parts} > 3",
                            $isa!(load 1, "{d2}", 0),
                            asm_multiply!($isa, 1, 1, 13),
                            $isa!(add 0, 0, 1),
                            $isa!(step "{d2}"),
                            ".endif",
                            $isa!(load 1, "{dl}", 0),
                            ".if {short}",
                            $isa!(mul 1, 1, 14),
                            ".else",
                            asm_multiply!($isa, 1, 1, 14),
                            ".endif",
                            $isa!(add 0, 0, 1),
                            asm_reduce!($isa, 0),
                            $isa!(step "{dl}"),
                            ".endif",
                            $isa!(store 0, "{out}", 0),
                            $isa!(step "{d0}"),
                            $isa!(step "{out}"),
                            $isa!(next "{n}"),
                            $isa!(leave),
                            parts = const PARTS,
                            short = const if SHORT_HIGH { 1 } else { 0 },
                            c = in($pointer) constants.as_ptr(),
                            d0 = inout($pointer) digit(0) => _,
                            d1 = inout($pointer) digit(1) => _,
                            d2 = inout($pointer) digit(2) => _,
                            dl = inout($pointer) digit(last) => _,
                            out = inout($pointer) output => _,
                            n = inout($pointer) len => _
                        );
                    }
                }
            }
        }
    };
}
