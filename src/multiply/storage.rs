//! Caller-owned scratch and retained tables of one admitted transform geometry.

#![expect(
    unsafe_code,
    reason = "Admitted geometry proves exclusive aligned arenas and complete transform table spans"
)]

use core::{
    mem::{align_of, size_of},
    num::NonZeroUsize,
};

use alloc::{vec, vec::Vec};

use super::{
    Executor, Geometry, Kernels, LIMB_BITS, Limb, Multiplication, PrimePrefix, Tables, Twiddles,
};

/// Operand lengths, output span, and scratch partition of one admitted geometry.
///
/// Fields are private, and [`Self::new`] is the only constructor, so a value always
/// holds: both arenas with both root tables fit `isize::MAX` bytes, the larger
/// being the multiplication scratch, and the output bit count fits `usize`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    geometry: Geometry,
    mul_limbs: usize,
    sqr_limbs: usize,
    len_a: usize,
    len_b: usize,
    table_floats: usize,
    output_limbs: usize,
    mul_floats: usize,
    sqr_floats: usize,
}

impl Layout {
    /// Sizes the scratch of `geometry` for operands of the given lengths and a result
    /// of `output_limbs` native limbs.
    ///
    /// Each arena contains one transform per prime. Multiplication stores two
    /// spectra and both root tables; squaring stores one spectrum and both tables.
    /// A block split stores one left spectrum per slot of its plan's workers, and
    /// keeps the limbs that windows share. Returns `None` when a buffer or a bit
    /// count exceeds the addressable range.
    #[must_use]
    pub fn new(
        geometry: Geometry,
        (len_a, len_b): (usize, usize),
        output_limbs: usize,
    ) -> Option<Self> {
        let table_floats = geometry
            .prefix()
            .count()
            .checked_mul(geometry.table_len().get())?
            .checked_mul(2)?;
        let sqr_floats = geometry.arena_floats(true).checked_add(table_floats)?;
        let mul_floats = geometry.arena_floats(false).checked_add(table_floats)?;
        let padding = align_of::<f64>()
            .saturating_sub(align_of::<Limb>())
            .div_ceil(size_of::<Limb>())
            .checked_add(geometry.kept_limbs())?;
        let mul_limbs = mul_floats
            .checked_mul(size_of::<f64>())?
            .div_ceil(size_of::<Limb>())
            .checked_add(padding)?;
        let sqr_limbs = sqr_floats
            .checked_mul(size_of::<f64>())?
            .div_ceil(size_of::<Limb>())
            .checked_add(padding)?;
        // The multiplication scratch holds the larger arena in whole limbs. The
        // output bits bound every bit position of reconstruction.
        if mul_limbs.checked_mul(size_of::<Limb>())? > isize::MAX.cast_unsigned()
            || output_limbs.checked_mul(LIMB_BITS).is_none()
        {
            return None;
        }
        Some(Self {
            geometry,
            mul_limbs,
            sqr_limbs,
            len_a,
            len_b,
            table_floats,
            output_limbs,
            mul_floats,
            sqr_floats,
        })
    }

    /// Writes the result to the first limbs of `dst` using caller-owned scratch.
    ///
    /// Returns `false`, preserving `dst`, when an operand length differs or `dst` or
    /// `scratch` is too short. Root tables are regenerated in `scratch` on each call.
    pub fn run<const SQUARE: bool, E: Executor>(
        &self,
        dst: &mut [Limb],
        operands: (&[Limb], &[Limb]),
        scratch: &mut [Limb],
        executor: &E,
    ) -> bool {
        let (limbs, floats) = if SQUARE {
            (self.sqr_limbs, self.sqr_floats)
        } else {
            (self.mul_limbs, self.mul_floats)
        };
        let Some(output) = self.output(dst, operands) else {
            return false;
        };
        if scratch.len() < limbs {
            return false;
        }
        let geometry = &self.geometry;
        // SAFETY: all native integer and float bit patterns are valid. The
        // admitted scratch size includes the kept limbs and worst-case
        // alignment padding, so align_to_mut retains the spectra and both tables.
        // The float count is the spectra plus the tables.
        let (kept, arena, forward, inverse) = unsafe {
            let (kept, rest) = scratch.split_at_mut_unchecked(geometry.kept_limbs());
            let (_, aligned, _) = rest.align_to_mut::<f64>();
            let values = aligned.get_unchecked_mut(..floats);
            let (spectra, roots) =
                values.split_at_mut_unchecked(floats.unchecked_sub(self.table_floats));
            let (fwd, inv) = roots.split_at_mut_unchecked(self.table_floats >> 1);
            (kept, spectra, fwd, inv)
        };
        let stride = geometry.table_len();
        let kernels = Kernels::get();
        // SAFETY: `output` checked the operands and the output span, and the split
        // above sized the arena, the kept limbs, and both tables of this geometry,
        // whose table length its primes admit.
        unsafe {
            Twiddles::prepare(
                forward,
                inverse,
                stride,
                geometry.prefix().primes(),
                kernels,
                executor,
            );
            Multiplication::execute::<SQUARE, E>(
                output,
                operands,
                (arena, kept),
                Tables {
                    forward,
                    inverse,
                    stride,
                },
                geometry,
                kernels,
                executor,
            );
        }
        true
    }
}

/// Root tables and a coefficient arena retained across operations.
#[derive(Clone, Debug)]
pub struct Retained {
    /// Entries per prime of both tables.
    stride: NonZeroUsize,
    /// Primes of both tables, once any exist.
    primes: Option<PrimePrefix>,
    forward: Vec<f64>,
    inverse: Vec<f64>,
    arena: Vec<f64>,
    kept: Vec<Limb>,
}

impl Retained {
    /// Creates empty storage.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            stride: NonZeroUsize::MIN,
            primes: None,
            forward: Vec::new(),
            inverse: Vec::new(),
            arena: Vec::new(),
            kept: Vec::new(),
        }
    }

    /// Writes the result to the first limbs of `dst`, growing the storage to fit.
    ///
    /// Returns `false`, preserving `dst`, when an operand length differs or `dst` is
    /// too short.
    pub fn run<const SQUARE: bool, E: Executor>(
        &mut self,
        layout: &Layout,
        dst: &mut [Limb],
        operands: (&[Limb], &[Limb]),
        executor: &E,
    ) -> bool {
        let Some(output) = layout.output(dst, operands) else {
            return false;
        };
        self.prepare(layout, SQUARE, executor);
        let geometry = &layout.geometry;
        // SAFETY: `output` checked the operands and the output span, and `prepare` grew
        // the tables, the arena, and the kept limbs to this geometry. The kept limbs
        // are exactly its length here.
        unsafe {
            Multiplication::execute::<SQUARE, E>(
                output,
                operands,
                (
                    &mut self.arena,
                    self.kept.get_unchecked_mut(..geometry.kept_limbs()),
                ),
                Tables {
                    forward: &self.forward,
                    inverse: &self.inverse,
                    stride: self.stride,
                },
                geometry,
                Kernels::get(),
                executor,
            );
        }
        true
    }

    /// Grows the tables, the arena, and the kept limbs to fit a layout.
    ///
    /// `square` selects the one-spectrum arena used by squaring. Every root depends
    /// only on its index and prime, so existing tables serve every shorter transform
    /// and smaller prime prefix unchanged.
    pub fn prepare<E: Executor>(&mut self, layout: &Layout, square: bool, executor: &E) {
        let geometry = &layout.geometry;
        let floats = geometry.arena_floats(square);
        // The arena and the kept limbs hold no value between operations, and grown
        // tables are regenerated whole. Freeing a buffer before allocating its
        // replacement keeps peak memory at the new size, and a zeroed allocation
        // maps fresh pages without writing them.
        if self.arena.len() < floats {
            self.arena = Vec::new();
            self.arena = vec![0.0; floats];
        }
        if self.kept.len() < geometry.kept_limbs() {
            self.kept = Vec::new();
            self.kept = vec![0; geometry.kept_limbs()];
        }
        let stride = self.stride.max(geometry.table_len());
        let prefix = geometry.prefix();
        let primes = self.primes.map_or(prefix, |primes| primes.max(prefix));
        if self.stride != stride || self.primes != Some(primes) {
            // SAFETY: the stride is the table length of a layout, whose two tables of
            // 8-byte floats fit isize::MAX bytes, so stride < 2^(usize::BITS-5). A
            // prefix has at most MAX_PRIMES = 8 primes, so the product fits usize, and
            // both tables hold exactly that many entries. Admission bounds every table
            // length by MAX_ROOT_ORDER, which every catalog prime admits, so the
            // largest stride serves the largest prefix.
            unsafe {
                let entries = primes.count().unchecked_mul(stride.get());
                self.forward = Vec::new();
                self.inverse = Vec::new();
                self.forward = vec![0.0; entries];
                self.inverse = vec![0.0; entries];
                Twiddles::prepare(
                    &mut self.forward,
                    &mut self.inverse,
                    stride,
                    primes.primes(),
                    Kernels::get(),
                    executor,
                );
            }
            self.stride = stride;
            self.primes = Some(primes);
        }
    }
}

impl Layout {
    /// Returns the output span of `dst`, when the operands match and it is long enough.
    fn output<'dst>(
        &self,
        dst: &'dst mut [Limb],
        (a, b): (&[Limb], &[Limb]),
    ) -> Option<&'dst mut [Limb]> {
        if self.matches(a.len(), b.len()) {
            dst.get_mut(..self.output_limbs)
        } else {
            None
        }
    }

    /// Tests whether operands of these lengths match this layout.
    #[inline]
    #[must_use]
    pub const fn matches(&self, len_a: usize, len_b: usize) -> bool {
        self.len_a == len_a && self.len_b == len_b
    }

    /// Returns the scratch in native limbs, including alignment padding, of a square
    /// or of a product.
    #[inline]
    #[must_use]
    pub const fn scratch_limbs(&self, square: bool) -> usize {
        if square {
            self.sqr_limbs
        } else {
            self.mul_limbs
        }
    }

    /// Returns the limbs of the output span.
    #[inline]
    #[must_use]
    pub const fn output_len(&self) -> usize {
        self.output_limbs
    }

    /// Returns the admitted geometry.
    #[inline]
    #[must_use]
    pub const fn geometry(&self) -> &Geometry {
        &self.geometry
    }
}
