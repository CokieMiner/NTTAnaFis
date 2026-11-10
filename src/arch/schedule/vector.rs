//! The fused register arithmetic of backends built from intrinsic functions.

/// Instantiates a backend from intrinsic functions.
///
/// Adds the exact products to `schedule_backend!`, whose table describes the other
/// arguments. `$fnmadd` is `(a, b, c) -> RN(c - a*b)` with one rounding, and `$round`
/// rounds each lane to the nearest integer, ties to even.
macro_rules! vector_backend {
    ($name:ident, $vector:ty, $lanes:literal, [$($attribute:meta),*],
     $load:path, $store:path, $splat:path, $add:path, $sub:path, $mul:path,
     $fnmadd:expr, $round:expr, $canonical:expr) => {
        /// Architecture-specific compensated residue kernels.
        #[derive(Debug)]
        pub struct $name;

        schedule_backend!(
            $name, $vector, $lanes, [$($attribute),*],
            $load, $store, $splat, $add, $sub, $mul,
            reduce_vector, multiply_vector, $canonical
        );

        // The modular products are always inlined into the kernels. Without a
        // target-feature attribute of their own, the inlined bodies compile with
        // the features of the calling kernel.
        #[inline(always)]
        unsafe fn reduce_vector(a: $vector, p: $vector, pinv: $vector) -> $vector {
            // SAFETY: callers are kernels of this backend, whose target
            // features admit these ISA operations.
            unsafe { ($fnmadd)(($round)($mul(a, pinv)), p, a) }
        }

        #[inline(always)]
        unsafe fn multiply_vector(a: $vector, b: $vector, p: $vector, pinv: $vector) -> $vector {
            // SAFETY: callers are kernels of this backend, whose target
            // features admit these ISA operations. The fused `high - a*b`
            // is the exact negated rounding error of the product, so
            // `(high - q*p) - (high - a*b) = a*b - q*p` is an exact integer
            // of magnitude below 2*p.
            unsafe {
                let high = $mul(a, b);
                let quotient = ($round)($mul(high, pinv));
                let error = ($fnmadd)(a, b, high);
                $sub(($fnmadd)(quotient, p, high), error)
            }
        }
    };
}
