//! Entry point of an admitted native-limb multiplication or squaring.

#![expect(
    unsafe_code,
    reason = "Admitted geometry and exclusive aligned arenas prove kernel spans and residue bounds"
)]

use core::num::NonZeroUsize;

use super::{
    Crt, Executor, Geometry, Kernels, LIMB_BITS, Limb, PRODUCT_FUSION_MIN_LEN, Packing, Partition,
    Sequential, Split, Tables,
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
    /// Tasks of the prefix under the pool's budget.
    pub tasks: NonZeroUsize,
    /// Whether workers run paired stages from the lower crossover.
    pub early_pairs: bool,
    /// Right spectra that the first block formed, one row per prime, which the
    /// later blocks of a split share; `None` while each block packs its own.
    pub right: Option<&'operation [f64]>,
}

impl Multiplication {
    /// Runs admitted operands in exclusive arenas and immutable transform tables.
    ///
    /// `storage` holds the spectra and, for a block split, the limbs its windows
    /// keep. An operation with enough work, counting every block, enters the worker
    /// pool once, so its phases need no further transfer between threads. A smaller
    /// one stays on the calling thread, where a worker hand-off would cost more than
    /// it saves.
    ///
    /// # Safety
    /// The operands match the geometry, of which `dst` is exactly the output span.
    /// `storage` holds at least `geometry.arena_floats(SQUARE)` floats and exactly
    /// `geometry.kept_limbs()` limbs, and `tables` covers the geometry's table and
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
        let blocks = geometry.split().map_or(1, |split| split.blocks.get());
        if executor.budget().get() > 1
            && geometry
                .prefix()
                .count()
                .saturating_mul(geometry.conv_len())
                .saturating_mul(blocks)
                >= PARALLEL_MIN_WORK
        {
            executor.on_workers(|| {
                let context = RunContext {
                    geometry,
                    kernels,
                    executor,
                    tasks: geometry.prefix().tasks(executor.budget()),
                    early_pairs: !SQUARE
                        && executor.budget().get() >= PARALLEL_PAIR_MIN_WORKERS
                        && geometry.transform_len().get() >= PARALLEL_PAIR_MIN_LEN,
                    right: None,
                };
                // SAFETY: the caller's contract is this function's.
                unsafe { run_blocks::<SQUARE, E>(&context, dst, operands, storage, tables) };
            });
        } else {
            let context = RunContext {
                geometry,
                kernels,
                executor: &Sequential,
                tasks: NonZeroUsize::MIN,
                early_pairs: false,
                right: None,
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
/// A plan without a split has one block. Otherwise window `k` of the output
/// starts `k*step` limbs in and holds the product of block `k`, and the result is
/// the sum of all windows at their offsets. The first block forms the spectra of
/// the shorter operand, which the other blocks share in runs of consecutive
/// blocks, one per slot. Each window replaces its limbs, so the limbs it shares
/// with the window before it are kept and added back.
///
/// Every addition is exact. Partial sums of nonnegative products stay below the
/// result, so no carry leaves the windows added so far. A convolution adds each
/// disjoint coefficient block modulo its width: its partial sums, unsigned or in
/// two's complement, are bounded as the result is, so the wrapped sum is exact
/// and no carry crosses a block.
///
/// # Safety
/// As for [`Multiplication::execute`].
unsafe fn run_blocks<const SQUARE: bool, E: Executor>(
    context: &RunContext<'_, E>,
    dst: &mut [Limb],
    operands: (&[Limb], &[Limb]),
    (arena, kept): (&mut [f64], &mut [Limb]),
    tables: Tables<'_>,
) {
    let geometry = context.geometry;
    let spectrum_floats = geometry.arena_floats(true);
    let Some(split) = geometry.split() else {
        // SAFETY: the arena holds one spectrum per prime for a square and two
        // otherwise, and the caller's contract is this function's for one block.
        unsafe {
            let (left, right) = if SQUARE {
                (arena.get_unchecked_mut(..spectrum_floats), &mut [][..])
            } else {
                let (prefix, suffix) = arena.split_at_mut_unchecked(spectrum_floats);
                (prefix, suffix.get_unchecked_mut(..spectrum_floats))
            };
            run_block::<SQUARE, E>(context, dst, operands, (left, right), tables);
        }
        return;
    };
    let (long, short) = if split.swapped {
        (operands.1, operands.0)
    } else {
        operands
    };
    // SAFETY: a split has at least two blocks, each at least as long as the shorter
    // operand, so the first block is whole and its window of step+overlap limbs
    // lies inside the output, and the remaining blocks are nonempty. The arena
    // holds the right spectra and a left spectrum per slot, and the kept limbs hold
    // the first overlap, then an overlap per slot, then a window per slot but the
    // last. The runs number at most the slots and the remaining blocks.
    unsafe {
        let (right, lefts) = arena.split_at_mut_unchecked(spectrum_floats);
        let (head, rest) = kept.split_at_mut_unchecked(split.overlap_limbs);
        let (keeps, windows) =
            rest.split_at_mut_unchecked(split.overlap_limbs.unchecked_mul(split.slots.get()));
        let (first, blocks) = long.split_at_unchecked(split.block_limbs);
        run_block::<SQUARE, E>(
            context,
            dst.get_unchecked_mut(..split.step_limbs.unchecked_add(split.overlap_limbs)),
            (first, short),
            (lefts.get_unchecked_mut(..spectrum_floats), right),
            tables,
        );
        let region = dst.get_unchecked_mut(split.step_limbs..);
        head.copy_from_slice(region.get_unchecked(..split.overlap_limbs));
        let later = NonZeroUsize::new_unchecked(split.blocks.get().unchecked_sub(1));
        let runs = split.slots.min(context.executor.budget()).min(later);
        let shared = RunContext {
            right: Some(&*right),
            ..*context
        };
        let buffers = (lefts, keeps, windows);
        let beyond = if runs.get() == 1 {
            run_slot::<SQUARE, E>(
                &shared,
                region,
                (blocks, short),
                buffers,
                (split, true),
                tables,
            )
        } else {
            divide::<SQUARE, E>(
                &shared,
                region,
                (blocks, short),
                buffers,
                (split, true),
                (later.get(), runs),
                tables,
            )
        };
        add_window(geometry, region, head, beyond);
    }
}

/// Runs the blocks of `long` in `runs` concurrent runs of consecutive blocks, one
/// per slot, and returns the high limbs that the last window keeps beyond
/// `region`, empty when the run is final.
///
/// Each cut divides the blocks and the slots in proportion, and once both halves
/// are done, adds the high limbs the left half kept into the right half's region,
/// carrying into the limbs the right half keeps. A single slot runs its blocks in
/// turn on its worker.
///
/// # Safety
/// As for [`run_slot`], with `long` of `blocks` blocks, `runs` at most `blocks`,
/// and buffers for that many slots.
unsafe fn divide<'kept, const SQUARE: bool, E: Executor>(
    context: &RunContext<'_, E>,
    region: &mut [Limb],
    (long, short): (&[Limb], &[Limb]),
    (lefts, keeps, windows): (&mut [f64], &mut [Limb], &'kept mut [Limb]),
    (split, last): (Split, bool),
    (blocks, runs): (usize, NonZeroUsize),
    tables: Tables<'_>,
) -> &'kept mut [Limb] {
    let Some(cut) = Partition::range::<1, 1>(0, blocks, runs) else {
        // One slot runs its blocks in turn on this worker.
        let sequential = RunContext {
            geometry: context.geometry,
            kernels: context.kernels,
            executor: &Sequential,
            tasks: NonZeroUsize::MIN,
            early_pairs: false,
            right: context.right,
        };
        // SAFETY: the caller's contract is this function's for one slot.
        return unsafe {
            run_slot::<SQUARE, Sequential>(
                &sequential,
                region,
                (long, short),
                (lefts, keeps, windows),
                (split, last),
                tables,
            )
        };
    };
    let geometry = context.geometry;
    // SAFETY: the cut gives each half at least as many blocks as slots, so every
    // left block is whole and the left region holds one step per block. Each half
    // of the buffers holds its slots; the left half is never final, so each of its
    // slots has a window.
    unsafe {
        let window = split.step_limbs.unchecked_add(split.overlap_limbs);
        let (left_long, right_long) =
            long.split_at_unchecked(cut.middle.unchecked_mul(split.block_limbs));
        let (left_region, right_region) =
            region.split_at_mut_unchecked(cut.middle.unchecked_mul(split.step_limbs));
        let (left_lefts, right_lefts) =
            lefts.split_at_mut_unchecked(cut.left.get().unchecked_mul(geometry.arena_floats(true)));
        let (left_keeps, right_keeps) =
            keeps.split_at_mut_unchecked(cut.left.get().unchecked_mul(split.overlap_limbs));
        let (left_windows, right_windows) =
            windows.split_at_mut_unchecked(cut.left.get().unchecked_mul(window));
        let (boundary, beyond) = context.executor.join(
            || {
                divide::<SQUARE, E>(
                    context,
                    left_region,
                    (left_long, short),
                    (left_lefts, left_keeps, left_windows),
                    (split, false),
                    (cut.middle, cut.left),
                    tables,
                )
            },
            || {
                divide::<SQUARE, E>(
                    context,
                    &mut *right_region,
                    (right_long, short),
                    (right_lefts, right_keeps, right_windows),
                    (split, last),
                    (blocks.unchecked_sub(cut.middle), cut.right),
                    tables,
                )
            },
        );
        add_window(geometry, right_region, boundary, beyond);
        beyond
    }
}

/// Runs consecutive blocks in one slot, in turn, and returns the high limbs that
/// its last window keeps beyond `region`, empty when the run is final.
///
/// Each window but the last is written in place. The last window of a run that
/// is not final would reach into the next run's region, so it is written to the
/// slot's window, whose low limbs are copied into place and whose high limbs are
/// returned.
///
/// # Safety
/// `long` holds the run's blocks of the context's split, and `short` the shorter
/// operand, whose spectra the context holds. `region` holds the run's output
/// from its first window: one step per block, or for a final run, every output
/// limb from there on. The first slot's buffers hold a left spectrum, an overlap,
/// and for a run that is not final, a window.
unsafe fn run_slot<'kept, const SQUARE: bool, E: Executor>(
    context: &RunContext<'_, E>,
    region: &mut [Limb],
    (long, short): (&[Limb], &[Limb]),
    (lefts, keeps, windows): (&mut [f64], &mut [Limb], &'kept mut [Limb]),
    (split, last): (Split, bool),
    tables: Tables<'_>,
) -> &'kept mut [Limb] {
    let geometry = context.geometry;
    // SAFETY: blocks before the last are whole, so each window but the last of a
    // final run, and every window of a run that is not final, holds step+overlap
    // limbs inside the region, and the last window of a final run ends with the
    // output. Each window starts with the overlap that the window before it in
    // this run keeps there.
    unsafe {
        let left = lefts.get_unchecked_mut(..geometry.arena_floats(true));
        let kept = keeps.get_unchecked_mut(..split.overlap_limbs);
        let window = split.step_limbs.unchecked_add(split.overlap_limbs);
        let mut rest = long;
        let mut start = 0_usize;
        loop {
            let (block, tail) = rest.split_at_unchecked(rest.len().min(split.block_limbs));
            if start != 0 {
                kept.copy_from_slice(region.get_unchecked(start..start.unchecked_add(kept.len())));
            }
            if tail.is_empty() && !last {
                let target = windows.get_unchecked_mut(..window);
                run_block::<SQUARE, E>(context, target, (block, short), (left, &mut []), tables);
                if start != 0 {
                    add_window(geometry, target, kept, &mut []);
                }
                let (low, high) = target.split_at_mut_unchecked(split.step_limbs);
                region
                    .get_unchecked_mut(start..start.unchecked_add(split.step_limbs))
                    .copy_from_slice(low);
                return high;
            }
            let end = start.unchecked_add(window).min(region.len());
            let target = region.get_unchecked_mut(start..end);
            run_block::<SQUARE, E>(
                context,
                target,
                (block, short),
                (&mut *left, &mut []),
                tables,
            );
            if start != 0 {
                add_window(geometry, target, kept, &mut []);
            }
            if tail.is_empty() {
                return &mut [];
            }
            start = start.unchecked_add(split.step_limbs);
            rest = tail;
        }
    }
}

/// Adds the high limbs that an earlier window keeps to the first limbs of
/// `target`.
///
/// A product carries through `target` and then `beyond`, the limbs kept after it;
/// the windows added so far sum below the result, so no carry leaves `beyond`. A
/// convolution adds each kept coefficient block modulo its width.
///
/// # Safety
/// `kept` is no longer than `target`, and both belong to a split of `geometry`.
unsafe fn add_window(geometry: &Geometry, target: &mut [Limb], kept: &[Limb], beyond: &mut [Limb]) {
    // SAFETY: the caller bounds `kept` by `target`.
    let (low, high) = unsafe { target.split_at_mut_unchecked(kept.len()) };
    if geometry.disjoint() {
        // A disjoint stride is a positive whole limb count, and the carry clears
        // at the end of each block.
        let width = geometry.output_stride().get() >> LIMB_BITS.trailing_zeros();
        let mut remaining = width;
        let mut carry = false;
        for (slot, &limb) in low.iter_mut().zip(kept) {
            (*slot, carry) = slot.carrying_add(limb, carry);
            remaining = remaining.wrapping_sub(1);
            if remaining == 0 {
                carry = false;
                remaining = width;
            }
        }
    } else {
        let mut carry = false;
        for (slot, &limb) in low.iter_mut().zip(kept) {
            (*slot, carry) = slot.carrying_add(limb, carry);
        }
        for slot in high.iter_mut().chain(beyond) {
            if !carry {
                break;
            }
            (*slot, carry) = slot.overflowing_add(1);
        }
        debug_assert!(!carry, "the windows sum below the result");
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
    let count = geometry.prefix().count();
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
        if !SQUARE && context.right.is_none() {
            Packing::pack(
                right,
                (operands.1, geometry.right()),
                geometry,
                kernels,
                executor,
            );
        }
        context.run_primes::<SQUARE>(left, right, tables, 0..count, context.tasks);
    }
    // SAFETY: geometry admits coefficient uniqueness, and the caller gives `dst` as
    // the output span. `left` holds one row of transform_len slots per prime, whose
    // active slots contain canonical u64 residue bit patterns produced by
    // kernels.residues, not floating-point values.
    unsafe {
        Crt::reconstruct(dst, left, geometry, kernels, executor);
    }
}
