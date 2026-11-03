//! Entry point of an admitted native-limb multiplication or squaring.

#![expect(
    unsafe_code,
    reason = "Admitted geometry and exclusive aligned arenas prove kernel spans and residue bounds"
)]

use core::num::NonZeroUsize;

use super::{
    Crt, Executor, Geometry, Kernels, Limb, MAX_PRIMES, PRODUCT_FUSION_MIN_LEN, Packing,
    Sequential, Tables,
};

/// Minimum transform length for early pairs in parallel products.
const PARALLEL_PAIR_MIN_LEN: usize = 2_usize.saturating_pow(21);
const PARALLEL_PAIR_MIN_WORKERS: usize = 8;

const _: () = assert!(
    PARALLEL_PAIR_MIN_LEN >= PRODUCT_FUSION_MIN_LEN,
    "early pairs imply products at the inverse leaves"
);

/// Smallest `np*conv_len` of an operation that enters the worker pool, about a
/// thousand limbs of an integer product. Below this, waking a worker costs more
/// than the split saves.
const PARALLEL_MIN_WORK: usize = 5000;

/// Internal execution of an admitted prime convolution.
#[derive(Debug)]
pub struct Multiplication;

/// Immutable geometry, kernels, and worker pool of one operation.
#[derive(Debug)]
pub struct RunContext<'operation, E> {
    /// Admitted transform geometry.
    pub geometry: &'operation Geometry,
    /// Kernels of the resolved vector unit.
    pub kernels: &'operation Kernels,
    /// Worker pool of the complete operation.
    pub executor: &'operation E,
    /// Whether workers run paired stages from the lower crossover.
    pub early_pairs: bool,
    /// Whether the right rows already hold their forward spectra.
    pub right_ready: bool,
}

impl Multiplication {
    /// Runs admitted operands in exclusive arenas and immutable transform tables.
    ///
    /// `storage` holds the spectra and, for a block split, the overlap limbs. An
    /// operation with enough work enters the worker pool once, so its phases need
    /// no further transfer between threads. A smaller one stays on the calling
    /// thread, where a worker hand-off would cost more than it saves.
    ///
    /// # Safety
    /// The operands match the geometry, of which `dst` is exactly the output span.
    /// `storage` holds at least `geometry.arena_floats(SQUARE)` floats and exactly
    /// `geometry.overlap_limbs()` limbs, and `tables` covers the geometry's table and
    /// primes.
    pub unsafe fn execute<const SQUARE: bool, E: Executor>(
        dst: &mut [Limb],
        operands: (&[Limb], &[Limb]),
        storage: (&mut [f64], &mut [Limb]),
        tables: Tables<'_>,
        geometry: &Geometry,
        kernels: &Kernels,
        executor: &E,
    ) {
        if executor.budget().get() > 1
            && geometry
                .prefix()
                .count()
                .saturating_mul(geometry.conv_len())
                >= PARALLEL_MIN_WORK
        {
            executor.on_workers(|| {
                let context = RunContext {
                    geometry,
                    kernels,
                    executor,
                    early_pairs: !SQUARE
                        && executor.budget().get() >= PARALLEL_PAIR_MIN_WORKERS
                        && geometry.transform_len().get() >= PARALLEL_PAIR_MIN_LEN,
                    right_ready: false,
                };
                // SAFETY: the caller's contract is this function's.
                unsafe { run_blocks::<SQUARE, E>(&context, dst, operands, storage, tables) };
            });
        } else {
            let context = RunContext {
                geometry,
                kernels,
                executor: &Sequential,
                early_pairs: false,
                right_ready: false,
            };
            // SAFETY: the caller's contract is this function's.
            unsafe {
                run_blocks::<SQUARE, Sequential>(&context, dst, operands, storage, tables);
            }
        }
    }
}

/// Multiplies every block of the longer operand with the shorter one.
///
/// A plan without a split has one block. Each later window starts inside the
/// previous one, whose high limbs are kept and added back, because reconstruction
/// replaces its whole window. Partial sums of nonnegative products stay below the
/// result, so no carry leaves a window.
///
/// # Safety
/// As for [`Multiplication::execute`].
unsafe fn run_blocks<const SQUARE: bool, E: Executor>(
    context: &RunContext<'_, E>,
    dst: &mut [Limb],
    operands: (&[Limb], &[Limb]),
    (arena, overlap): (&mut [f64], &mut [Limb]),
    tables: Tables<'_>,
) {
    let geometry = context.geometry;
    let spectrum_floats = geometry.arena_floats(true);
    // SAFETY: the arena holds one spectrum per prime for a square and two otherwise.
    let (left, right) = unsafe {
        if SQUARE {
            (arena.get_unchecked_mut(..spectrum_floats), &mut [][..])
        } else {
            let (prefix, suffix) = arena.split_at_mut_unchecked(spectrum_floats);
            (prefix, suffix.get_unchecked_mut(..spectrum_floats))
        }
    };
    let Some(split) = geometry.split() else {
        // SAFETY: the caller's contract is this function's, for one block.
        unsafe { run_block::<SQUARE, E>(context, dst, operands, (left, right), tables) };
        return;
    };
    let (long, short) = if split.swapped {
        (operands.1, operands.0)
    } else {
        operands
    };
    let mut rest = long;
    let mut start = 0_usize;
    let mut right_ready = false;
    // SAFETY: admission gives positive block lengths and an output of
    // (long+short)*scale limbs. The window of the last block ends at the end of
    // the output, because that block takes what remains of the longer operand.
    // Blocks are at least as long as the shorter operand, so every window holds
    // the overlap, whose retained length is exactly overlap_limbs. Each block is
    // an admitted pair of operands with its window as the output span.
    unsafe {
        while !rest.is_empty() {
            let (block, tail) = rest.split_at_unchecked(rest.len().min(split.block_limbs));
            let end = start
                .unchecked_add(split.step_limbs)
                .unchecked_add(split.overlap_limbs)
                .min(dst.len());
            let window = dst.get_unchecked_mut(start..end);
            if right_ready {
                overlap.copy_from_slice(window.get_unchecked(..overlap.len()));
            }
            run_block::<SQUARE, E>(
                &RunContext {
                    right_ready,
                    ..*context
                },
                window,
                (block, short),
                (&mut *left, &mut *right),
                tables,
            );
            if right_ready {
                // Adds the kept limbs back and propagates the carry.
                let (low, high) = window.split_at_mut_unchecked(overlap.len());
                let mut carry = false;
                for (slot, &limb) in low.iter_mut().zip(&*overlap) {
                    (*slot, carry) = slot.carrying_add(limb, carry);
                }
                for slot in high {
                    if !carry {
                        break;
                    }
                    (*slot, carry) = slot.overflowing_add(1);
                }
                debug_assert!(!carry, "the product fits its native span");
            }
            start = start.unchecked_add(split.step_limbs);
            rest = tail;
            right_ready = true;
        }
    }
}

/// Packs one pair of operands, runs every prime, and reconstructs the limbs.
///
/// # Safety
/// As for [`Multiplication::execute`], with the geometry of one block: `dst` is
/// the output span of these operands, and `left` and `right` hold one spectrum
/// per prime.
unsafe fn run_block<const SQUARE: bool, E: Executor>(
    context: &RunContext<'_, E>,
    dst: &mut [Limb],
    operands: (&[Limb], &[Limb]),
    (left, right): (&mut [f64], &mut [f64]),
    tables: Tables<'_>,
) {
    let geometry = context.geometry;
    let kernels = context.kernels;
    let executor = context.executor;
    let budget = executor.budget();
    let count = geometry.prefix().count();
    // Two or more tasks per prime start every prime at once, and stealing
    // balances workers of different speeds. A budget too large to round to a
    // multiple of the prime count stays as it is.
    let tasks = if budget.get() > 1 {
        // SAFETY: rounding up and the maximum never fall below the positive budget.
        unsafe {
            NonZeroUsize::new_unchecked(
                budget
                    .get()
                    .checked_next_multiple_of(count)
                    .unwrap_or_else(|| budget.get())
                    .max(count.saturating_mul(2)),
            )
        }
    } else {
        NonZeroUsize::MIN
    };
    // SAFETY: the caller gives each operand of this block with its geometry, and
    // rows of one spectrum per prime for both. The tables cover the prefix, whose
    // indices are 0..count.
    unsafe {
        Packing::pack(
            left,
            (operands.0, geometry.left()),
            geometry,
            kernels,
            executor,
        );
        if !SQUARE && !context.right_ready {
            Packing::pack(
                right,
                (operands.1, geometry.right()),
                geometry,
                kernels,
                executor,
            );
        }
        context.run_primes::<SQUARE>(left, right, tables, 0..count, tasks);
    }
    let mut residues: [&[f64]; MAX_PRIMES] = [&[]; MAX_PRIMES];
    for (slot, spectrum) in residues
        .iter_mut()
        .zip(left.chunks_exact(geometry.transform_len().get()))
    {
        *slot = spectrum;
    }
    // SAFETY: geometry admits coefficient uniqueness, and the caller gives `dst` as
    // the output span. Each active slot contains a canonical u64 residue bit
    // pattern produced by kernels.residues, not a floating-point value.
    unsafe {
        Crt::reconstruct(dst, &residues, geometry, kernels, executor);
    }
}
