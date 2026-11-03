# NTTAnaFis: Contributor Guide

`ntt_anafis` computes exact multi-prime number-theoretic transforms and native-limb
integer products, and exposes its prime fields for modular algorithms. Its public
surface is declared in [src/lib.rs](src/lib.rs). Limb types and the parallel executor
come from `core_anafis` (`../CoreAnaFis`).

**Priorities, in order:** exactness, memory safety, portability, speed, ergonomics.
Speed never excuses an unproved invariant, and a proved invariant must not add
avoidable allocation or branching.

## Layout

| Path | Holds |
| :--- | :--- |
| `src/plan.rs` | `MulPlan`, `MulWorkspace` |
| `src/convolve.rs` | Dense convolution by residue transforms or direct summation |
| `src/profile/` | Catalog prefixes, and `Geometry`, the admitted geometry |
| `src/multiply/` | `Layout` and `Retained`, the safe entries, and prime scheduling |
| `src/limb/` | Packing limbs into residues, and cofactor CRT reconstruction |
| `src/transform/` | `Transform`, the transforms of one prime, and root tables |
| `src/field/` | The prime catalog, `PrimeField`, `Residue`, `CrtPair` |
| `src/partition.rs` | Task partitions under a worker budget |
| `src/arch/contract.rs` | What every kernel assumes |
| `src/arch/schedule/` | Each kernel once: macros in `expression/`, block text in `assembly/` |
| `src/arch/scalar/` | Portable kernels and exact scalar products |
| `src/arch/backend/` | One file per vector ISA, and the kernel selector |
| `build.rs` | The `ntt_*` flags, the one definition of target conditions |
| `bench/` | Separate crate holding the benchmarks and their dependencies |
| `docs/` | The README chart, drawn from the `reference` benchmark |
| `tools/` | Source audits, see [tools/audit/README.md](tools/audit/README.md) |
| `.github/` | CI over every target and vector unit, see [its README](.github/workflows/README.md) |

## Invariants

Each rule has one definition. Change it there, and cite it elsewhere.

| Rule | Defined in |
| :--- | :--- |
| Catalog primes: range, primality, generator, valuation; `MAX_ROOT_ORDER` | `catalog_prime`, `field/catalog.rs` |
| Stage residue bounds, `FloatField`, `Columns`, complete vectors | `arch/contract.rs` |
| Exact modular products and their citations | `arch/scalar/product.rs` |
| Admitted geometry; `admit` is the only constructor | `profile/planner.rs` |
| Addressable scratch and output; `Layout::new` is the only constructor | `multiply/storage.rs` |
| Row shapes `2^k` and `3*2^k`, and what `Transform` methods assume | `transform/mod.rs` |
| Coefficient capacity `floor(log2(2*P/3))` | `limb/cofactor.rs` |
| Worker budgets | `partition.rs` |

These rules cross modules:

- **Exactness:** a planned result is exact when the coefficient bound is below the
  product of the selected primes. Do not weaken that bound without a proof.
- **Failure leaves no trace:** a rejected plan or a short buffer returns `false` or
  `None` and leaves `dst` unmodified. A product writes exactly `a.len() + b.len()`
  limbs, and a square exactly `2 * a.len()`.
- **One check:** `Layout::run` and `Retained::run` check operands and storage.
  `Multiplication::execute` and everything below it assume those checks, so it is
  `unsafe`, and so is each entry into another subsystem: `Packing::pack`,
  `RunContext::run_primes`, `Transform::forward_row` and `finish_row`,
  `Twiddles::prepare`, `Crt::reconstruct`. Later stages cite admission and do not
  repeat it. Functions inside one subsystem rely on its module terms.
- **Complete vectors:** transforms have at least `MIN_TRANSFORM_LEN` points, and active
  prefixes end at multiples of `COLUMN_ALIGN`, so kernels have no scalar cleanup.
  Column ranges are nonempty, so a kernel tests for the end only after a vector.
- **One schedule:** a kernel's arithmetic is written once. Assembly targets repeat it
  as block text, which `admitted()` in `src/arch/tests/backends.rs` compares with the
  portable kernels. Where a target guarantees a backend, production does not compile
  the portable kernels (`ntt_fixed_kernels`), or the expression schedule for block-text
  backends (`ntt_expression_schedule`), and `arch/tests` includes those files as the
  reference.
- **Retained tables:** a root depends only on its index and prime, so a table serves
  every shorter transform and smaller prefix. Tables only grow. Admission bounds every
  table by `MAX_ROOT_ORDER`, which every catalog prime has, so the largest stride serves
  the largest prefix.
- **Block splits:** only planned for one worker, and a plan stays exact in any pool.
  The right spectrum is transformed once, and read only after the first block.
- **Pool entry:** `Multiplication::execute` enters the pool once per operation; phases
  must not enter it separately. Below `PARALLEL_MIN_WORK` it stays sequential.
- **Convolution capacity:** check it against the bit length of
  `min(len_a, len_b)*max_a*max_b`, not the integer-product bound.
- **Arithmetic:** sizes and indices use checked arithmetic, and casts need a range
  proof. Kernels assume IEEE binary64 in the default rounding mode.

## Code Rules

- **Unsafe:** one file-level `#![expect(unsafe_code, reason = "...")]`, and a
  `// SAFETY:` proof before each block. State the proof, not a checklist.
- **Lint attributes:** every `allow` or `expect` has a `reason`. Never suppress
  `dead_code`; gate the item for the configurations that use it.
- **No small helpers:** a function under about five lines of logic needs a reason: an
  API boundary, private-field access, or a single proof site. Repeat a line rather
  than wrap it.
- **Functions belong to types:** a function another module calls is an associated
  function, such as `Crt::reconstruct`, never a re-exported free function.
- **Paths:** import with `use`; write inline paths (`::core::…`, `$crate::…`) only
  inside macros.
- **Modules:** private `mod` with `pub use`, never `pub mod`. A `mod.rs` only registers
  children, and a module is a file or a folder, never both.
- **Visibility:** `pub(crate)` only for `pub(crate) use` at the top of `lib.rs`. Shared
  items are plain `pub` inside sealed modules.
- **Imports:** `super::Thing` for parent and sibling items, `crate::<subsystem>::Thing`
  across subsystems. Group `core`, `alloc`, external, `crate::`, `super::`.
- **Target conditions:** define them once in `build.rs` as `ntt_*` flags.
- **Tests:** in `tests.rs` or `tests/` beside the module. Production names nothing only
  tests use, and the only test item it contains is the final `#[cfg(test)] mod tests;`.
- **Source order:** declarations, driver, scratch partition, algorithm stages, helpers,
  leaves. Callers come before callees.
- **Size:** files stay near 500 lines (`src/arch/` backends are exempt). Do not split a
  file only to meet that.
- **Docs:** direct, descriptive, and scientific. Cite the source of a named algorithm.
  No history or commented-out code.
- **Editing:** by hand, not with bulk rewrite scripts. The library depends only on
  `core_anafis`, and stable public APIs keep their names.

## Performance Work

Work in this order: reduce the mathematical work, then the algorithmic cost and memory
traffic, then review MIR, LLVM IR, and assembly, and only then benchmark the complete
change. Unchecked indexing, forced inlining, and branchless rewrites need an identified
reduction in executed work.

Measure in alternating fresh processes (A/B/B/A) on identical operands with a fixed
worker count, and below, at, and above any crossover. Check the power profile first.
Keep raw runs under the ignored `target/`.

## Verification

Run from this directory, with zero findings:

```sh
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
python3 tools/structure_audit.py
python3 tools/import_audit.py
python3 tools/check_allows.py --check --all-lints
python3 tools/audit.py --tests
```

A change to a backend or to `build.rs` also needs clippy on each affected target, with
and without `--no-default-features` and the backend's target features:

```sh
cargo clippy --lib --no-default-features --target s390x-unknown-linux-gnu -- -D warnings
```

Assembly text fails only when assembled, so build it with `cargo build --lib --target X`,
and run the tests under QEMU for every vector unit of the target. CI runs all of these
checks, the x86 units under Intel SDE, and the minimum Rust version.
