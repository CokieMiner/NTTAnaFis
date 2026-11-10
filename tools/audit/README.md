# Source audits

Run the verification checks from the repository root:

```sh
python3 -B tools/audit.py --tests
python3 -B tools/check_allows.py --check --all-lints
```

`audit.py` verifies Rust structure, import hierarchies, Python tools, and runs the
Python regression suite (`--tests`). `check_allows.py` inventories and validates all
`#[allow]` and `#[expect]` attributes.

Additional flags on `audit.py`:
- `--json`: outputs structured JSON results on stdout.
- `--deny-oversized`: promotes the 500-line cohesion target to a failure.
- `--function-report DIR`: writes `functions.json` and `functions.dot`.
- `--deny-function-reviews`: promotes advisory function reviews to failures.

## Checks and source coverage

| Check | Tool | Scope |
| --- | --- | --- |
| Rust structure | `tools/structure_audit.py` | Structural registries, declaration order, visibility gates, architecture isolation in `src/arch/`, and advisory function relationships. |
| Rust imports | `tools/import_audit.py` | Facade boundaries, explicit names, grouping order, duplicate bindings, inline paths, and unused aliases. |
| Rust lint attributes | `tools/check_allows.py` | Validates that every `allow` and `expect` has a nonempty literal reason and never suppresses `dead_code`. |
| Python tools | `tools/audit.py` | Package facades, syntax, explicit imports, and tests in `tools/audit/tests/`. |

Rust discovery scans `src/` (including `src/tests/`); the separate `bench/` crate is
outside the audit. The library crate name is read from `Cargo.toml` (`ntt_anafis`).
Production sources must not import `mp_anafis`.

`pub(crate)` is accepted only on explicit reexports at the top level of `src/lib.rs`.
Shared implementation items use plain `pub` inside sealed modules; external exposure
is governed strictly by the `src/lib.rs` facade.

Direct sibling dependencies pass through their parent facade. Cross-subsystem imports
use `crate::<subsystem>::Item`. Production files over 500 lines are reported for cohesion
review (architecture backends under `src/arch/` are exempt).

## Function relationships

`audit.py` and `structure_audit.py` construct a static source function call graph.
The review produces advisory observations:

- `pub` functions whose resolved callers are all within their own module or descendants.
- Helpers declared above a resolved caller in the same source file.
- Functions with a single resolved production caller in another file of the same subsystem.
- Associated functions without a receiver or explicit dependency on their type.
- Functions with no observed call or value reference.

These checks are advisory; candidate findings do not fail CI by default. Use
`--deny-function-reviews` to enforce them during deliberate refactoring reviews.

Individual tool entry points remain available for targeted diagnosis:

```sh
python3 -B tools/structure_audit.py --json
python3 -B tools/import_audit.py --json
python3 -B tools/check_allows.py --check --all-lints
```

## Implementation and limits

The shared Rust scanner masks comments and literals while preserving positions,
then reads balanced attributes, items, and use trees. Source branches are
inspected regardless of the host target. Source discovery, lint parsing,
visibility resolution, import rules, and Python checks have separate modules;
tests are grouped by those responsibilities.

These checks do not expand macros or replace Rust name resolution, type checking,
cfg compilation, Clippy, or Miri. Visibility resolution follows ordinary module
files and explicit export paths; custom `#[path]` modules and types exposed
indirectly through signatures require compiler-assisted review. A literal reason
can be checked structurally; its adequacy and scope require review.

Mathematical correctness, unsafe proofs, prime-catalog bounds, coefficient
ranges, scratch geometry, benchmark evidence, and performance claims still
require the corresponding manual review and execution checks in
[AGENTS.md](../../AGENTS.md).
