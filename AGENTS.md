# NTTAnaFis

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
| `src/plan.rs` | Client-facing `MulPlan`, `MulWorkspace` |
| `src/convolve.rs` | Client-facing `ConvPlan`, `ConvWorkspace` for exact convolutions |
| `src/profile/` | Transform geometry planning (`Geometry`), split policies, and prime prefixes |
| `src/multiply/` | Execution coordinators (`Multiplication`), internal storage (`Layout`, `Retained`), and prime scheduling |
| `src/limb/` | Packing limbs into residues, and cofactor CRT reconstruction |
| `src/transform/` | Core NTT butterflies (`Transform`), root/twiddle tables, forward and inverse stages |
| `src/field/` | Prime catalog (`PRIME_CATALOG`), `PrimeField`, `Residue`, `CrtPair` |
| `src/partition.rs` | Task partitions under a worker budget |
| `src/arch/contract.rs` | Architectural contracts, lane widths, and stage residue bounds |
| `src/arch/schedule/` | Kernel execution schedules: expression macros in `expression/`, vector block text in `assembly/` |
| `src/arch/scalar/` | Portable scalar fallback kernels and exact products |
| `src/arch/backend/` | Hardware vector ISA backends (AVX2, AVX-512, IFMA, NEON, s390x, etc.) and runtime selector |
| `src/arch/tests/` | Hardware backend validation, residue contracts, and packing tests |
| `src/tests/` | End-to-end integration tests (plans, products, scheduling, convolution, schoolbook) |
| `build.rs` | Target feature detection and `ntt_*` configuration flags |
| `bench/` | Isolated benchmark crate (`ntt_anafis_bench`) comparing against FLINT and GMP |
| `docs/` | Benchmark plots (`multiplication.svg`) |
| `tools/` | Automated source audits (see [tools/audit/README.md](tools/audit/README.md)) |
| `.github/` | Multi-architecture CI matrix (see [its README](.github/workflows/README.md)) |

## Invariants

Each rule has one definition. Change it there, and cite it elsewhere.

| Rule | Defined in |
| :--- | :--- |
| Catalog primes: range, primality, generator, valuation, `FloatField`; `MAX_ROOT_ORDER` | `PRIME_CATALOG`, `Prime`, `field/catalog.rs` |
| Stage residue bounds, kernel contracts, `Columns`, complete vectors | `arch/contract.rs` |
| Exact modular products and their citations | `arch/scalar/product.rs` |
| Admitted geometry; `admit` is the only constructor | `profile/planner.rs` |
| Addressable scratch and output; `Layout::new` is the only constructor | `multiply/storage.rs` |
| Row shapes `2^k`, `3*2^k`, `5*2^k`, `7*2^k`, and what `Transform` methods assume | `transform/mod.rs` |
| Coefficient capacity `floor(log2(2*P/3))` | `limb/cofactor.rs` |
| Worker budgets | `partition.rs` |
| Tasks of a prime prefix under a budget | `PrimePrefix::tasks`, `field/catalog.rs` |

These rules cross modules:

- **Exactness:** a planned result is exact when the coefficient bound is below the
  product of the selected primes. Do not weaken that bound without a proof.
- **Failure leaves no trace:** a rejected plan or a short buffer returns `false` or
  `None` and leaves `dst` unmodified. A product writes exactly `a.len() + b.len()`
  limbs, and a square exactly `2 * a.len()`.
- **One check:** Public entry points (`MulPlan`, `MulWorkspace`, `ConvPlan`, `ConvWorkspace`)
  validate operand shapes and buffer capacities once. Execution coordinators (`Layout::run`, `Retained::run`)
  assume these checks and dispatch to `unsafe` subsystem entries (`Multiplication::execute`, `Packing::pack`,
  `RunContext::run_primes`, `Transform::forward_row` and `Transform::finish_row`, `Crt::reconstruct`).
  Later stages cite admission without re-validating. Root table preparation (`Twiddles::prepare`) is safe.
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
- **Block splits:** a split has at least two blocks. The first block forms the right
  spectra, which later blocks only read. Under several workers a block either holds
  `BLOCK_SHARE_FLOATS` floats per task that shares a prime's row, counted by
  `PrimePrefix::tasks`, or the blocks after the first fill more than three quarters of
  their waves of one block per worker and run in one slot per worker, each with its
  own left spectra (`Split::slots`, sized by `Layout`). A run keeps the high limbs of its last
  window until the next run is done. A plan stays exact in any pool.
- **Pool entry:** `Multiplication::execute` enters the pool once per operation; phases
  must not enter it separately. Below `PARALLEL_MIN_WORK`, counted over every block, it
  stays sequential.
- **Convolution capacity:** check it against the bit length of
  `min(len_a, len_b)*max_a*max_b`, plus one sign bit for a signed convolution, not the
  integer-product bound. Signed coefficients reconstruct offset by `2^(b-1)` inside
  `Crt::reconstruct`, and block windows of every convolution add per coefficient block.
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
python3 -B tools/audit.py --tests
python3 -B tools/check_allows.py --check --all-lints
```

`tools/audit.py` checks structural constraints, import hierarchies, tool regressions,
and function graphs. `check_allows.py` validates `#[allow]` and `#[expect]` reasons.
Individual runners (`structure_audit.py`, `import_audit.py`) remain available for isolated troubleshooting.

A change to a backend or to `build.rs` also needs clippy on each affected target, with
and without `--no-default-features` and the backend's target features:

```sh
cargo clippy --lib --no-default-features --target s390x-unknown-linux-gnu -- -D warnings
```

Assembly text fails only when assembled, so build it with `cargo build --lib --target X`,
and run the tests under QEMU for every vector unit of the target. CI runs all of these
checks, the x86 units under Intel SDE, and the minimum Rust version.
