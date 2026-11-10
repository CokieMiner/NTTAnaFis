# NTTAnaFis

Exact multi-prime number-theoretic transforms for large integer and polynomial
products, in stable Rust.

`ntt_anafis` is a multiplication engine designed for operands above the basecase /
Karatsuba crossover. Short operands belong in caller-managed small-integer routines.
It also exposes its underlying 50-bit prime fields for modular algorithms.

- **Exact:** every admitted result is the exact integer, with no rounding error.
- **Portable:** vector kernels for x86, AArch64, s390x, LoongArch, POWER, RISC-V, and
  WebAssembly on stable Rust, with `no_std` + `alloc` support.
- **Allocation-free path:** `MulPlan` runs entirely in caller-owned scratch.
- **Minimal:** one dependency, `core_anafis` (and optional `rayon`).

## Performance

![Line charts of time per limb and per coefficient on the AVX-512 IFMA and AVX2 targets. With AVX-512 IFMA and with AVX2, ntt_anafis is below FLINT from about 192 limbs. GMP is far above both at large sizes.](docs/multiplication.svg)

Each library is built for its target and timed at its best, single-threaded on identical operands:

| Limbs | `ntt_anafis` (IFMA) | `ntt_anafis` (AVX2) | FLINT (AVX2) | GMP |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 5.5 µs | 6.3 µs | 7.6 µs | 7.7 µs |
| 1,024 | 23.1 µs | 27.4 µs | 37.0 µs | 53.9 µs |
| 16,384 | 468 µs | 546 µs | 681 µs | 1.6 ms |
| 262,144 | 10.1 ms | 11.2 ms | 13.7 ms | 46.9 ms |
| 1,048,576 | 43.4 ms | 50.1 ms | 61.3 ms | 216 ms |
| 16,777,216 | 849 ms | 1.02 s | 1.18 s | 4.28 s |

| Coefficients (16-bit) | `ntt_anafis` (IFMA) | `ntt_anafis` (AVX2) | FLINT (AVX2) |
| ---: | ---: | ---: | ---: |
| 1,024 | 6.6 µs | 8.1 µs | 9.8 µs |
| 65,536 | 594 µs | 723 µs | 824 µs |
| 1,048,576 | 29.7 ms | 33.4 ms | 38.5 ms |
| 4,194,304 | 134 ms | 146 ms | 172 ms |

Key takeaways against FLINT and GMP:
- **Integer products:** `ntt_anafis` outperforms FLINT starting at ~192 limbs (taking 70% of FLINT's time with IFMA, 83% with AVX2). At 1M limbs, products take a fifth of GMP's time.
- **Squares:** takes 67% (IFMA) and 82% (AVX2) of FLINT's time from 256 limbs.
- **Polynomials:** 12–27% faster than FLINT from 1,024 coefficients across 16-bit and 32-bit workloads.

*Benchmark setup: AMD Ryzen AI 7 350 (single Zen 5 core at 5.09 GHz), FLINT 3.6.0 and GMP 6.3.0 built by GCC 16. FLINT's `fft_small` transforms use AVX2 on all x86 targets, so AVX-512 IFMA is unique to `ntt_anafis`.*

## Usage

```toml
[dependencies]
ntt_anafis = "0.1"
```

Limbs are native words (`usize`), least significant first. A product of `a` and `b`
occupies exactly `a.len() + b.len()` limbs; a square occupies `2 * a.len()` limbs.
Operations return `bool`: `true` on success, `false` on unsupported shapes or short
buffers. Failure leaves `dst` unmodified; success writes only the product prefix and
preserves any additional destination limbs.

| Type | Role | Execution model |
| :--- | :--- | :--- |
| `MulWorkspace` | Integer products & squares | Retains transform tables and arenas across calls |
| `MulPlan` | Integer products & squares | Zero heap allocation in caller-owned scratch |
| `ConvWorkspace` | Polynomial convolutions | Retains tables and scratch across calls |
| `ConvPlan` | Polynomial convolutions | Computes convolutions in caller-owned scratch |
| `PrimeField`, `Residue` | 50-bit modular arithmetic | Operations modulo primes `2^49 < p < 2^50` |
| `CrtPair` | Two-prime CRT | Garner reconstruction to signed 128-bit integers |

### Reusable workspace (`MulWorkspace`)

Retains root tables and coefficient arenas across calls. Tables serve every shorter
transform and grow only when a larger shape arrives. Also accepts empty operands (writing zero):

```rust
use ntt_anafis::{MulPlan, MulWorkspace};

let (a, b) = (vec![42_usize; 1024], vec![7_usize; 1024]);
let mut dst = vec![0; a.len() + b.len()];

let mut workspace = MulWorkspace::new();
workspace.prepare(MulPlan::new(a.len(), b.len()).expect("admitted shape"));
assert!(workspace.mul(&mut dst, &a, &b));
```

### Allocation-free scratch path (`MulPlan`)

Validates operand lengths, determines geometry, and executes in caller-owned scratch without heap allocations:

```rust
use ntt_anafis::MulPlan;

let (a, b) = (vec![u32::MAX as usize; 64], vec![0x1234_5678_usize; 48]);
let mut dst = vec![0; a.len() + b.len()];

let plan = MulPlan::new(a.len(), b.len()).expect("admitted shape");
let mut scratch = vec![0; plan.scratch_len()];
assert!(plan.mul(&mut dst, &a, &b, &mut scratch));
```

### Squaring

Passing the same slice as both operands runs the cheaper square:

```rust
use ntt_anafis::MulPlan;

let a = vec![3_usize; 96];
let mut dst = vec![0; 2 * a.len()];

let plan = MulPlan::new(a.len(), a.len()).expect("admitted shape");
let mut scratch = vec![0; plan.sqr_scratch_len()];
assert!(plan.sqr(&mut dst, &a, &mut scratch));
```

### Exact dense convolutions (`ConvPlan`, `ConvWorkspace`)

Computes exact convolutions of nonnegative or two's complement coefficients, using the fewest
primes that hold the caller's coefficient bounds (`max_a`, `max_b`). Each result coefficient occupies `stride()` limbs:

```rust
use ntt_anafis::ConvPlan;

let (a, b) = ([1_usize, 2, 3], [4_usize, 5]);
let plan = ConvPlan::new(a.len(), b.len(), 3, 5, false).expect("admitted shape");
let mut dst = vec![0; plan.output_len()];
let mut scratch = vec![0; plan.scratch_len()];

assert!(plan.convolve(&mut dst, &a, &b, &mut scratch));
assert_eq!(dst, [4, 13, 22, 15]);
```

### Modular arithmetic & CRT (`PrimeField`, `CrtPair`)

Prime fields and two-prime Garner reconstruction can also be used directly:

```rust
use ntt_anafis::{CrtPair, PrimeField};

let mut catalog = PrimeField::catalog();
let (first, second) = (catalog.next().unwrap(), catalog.next().unwrap());

let (a, b) = (first.reduce(12345), second.reduce(12345));
assert_eq!(CrtPair::new(first, second).unwrap().combine(a, b), 12345);
```

## How It Works

- **Exact products in binary64:** residues are integer-valued `f64`. A fused
  multiply-add recovers the exact remainder of each product (Ogita, Rump, and Oishi 2005;
  van der Hoeven, Lecerf, and Quintin 2016). Fallback targets use wrapping integers
  or Dekker's product (Dekker 1971).
- **Eight primes:** each between `2^49` and `2^50` with roots of order at least `2^41`.
  Integer products select the prefix minimizing estimated work; convolutions select the
  fewest primes holding the declared bound (one holds coefficients below `2^49`, two below `2^99`).
- **Composite lengths (`2^k`, `3*2^k`, `5*2^k`, `7*2^k`):** a remainder tree
  splits `x^N-1` and factors of `x^N+1` (such as `(x^N-1)(x^M-i)` with `N = 2M`), avoiding
  length doubling past powers of two without requiring factors of 3, 5, or 7 in `p-1`.
- **Cache-blocked stages:** radix-4 stages run down to fused 16-point register leaves.
- **Unbalanced operands:** long operands split into blocks sharing the shorter operand's
  spectrum, processed concurrently across threads.
- **Radix-`2^52` reconstruction:** cofactor Chinese remaindering evaluates coefficients
  at their bit offset. x86 evaluates eight coefficients in parallel via IFMA or FMA; other
  targets evaluate one in radix `2^64`.

Mathematical proofs sit beside code: residue bounds in `src/arch/contract.rs`, exact products
in `src/arch/scalar/product.rs`, and geometry in `src/profile/planner.rs`. See [AGENTS.md](AGENTS.md)
for architecture details.

## Targets

| Target | Vector units | Lanes | Selection with `std` |
| :--- | :--- | :--- | :--- |
| `x86`, `x86_64` | AVX2 + FMA, AVX-512F, AVX-512 IFMA | 4, 8 | Run time |
| `aarch64` | NEON | 2 | Compile time |
| `s390x` | Vector facility, enhancements 1 | 2 | Run time |
| `loongarch64` | LSX, LASX | 2, 4 | Run time |
| `powerpc64le`, `powerpc64` | VSX | 2 | Compile time |
| `riscv64` | V (`VLEN` 128, 256, 512) | 2, 4, 8 | Run time |
| `wasm32` | `simd128` | 2 | Compile time |

Other targets use portable scalar kernels. Without `std`, target features select kernels at compile time. Rust 1.96 or later is required.

## Features

- **`std`** (default): run-time CPU detection.
- **`rayon`**: parallel transforms, packing, and reconstruction for operations above ~1,000 limbs.

## Benchmarks

The `bench/` crate isolates benchmark dependencies (`divan`, `gmp-mpfr-sys`, `rug`):

```sh
cargo bench -p ntt_anafis_bench
```

To compare against local FLINT and GMP installations (under `$PREFIX`):

```sh
RUSTFLAGS="-C target-cpu=native" REFERENCE_PREFIX=$PREFIX \
    cargo bench -p ntt_anafis_bench --features reference --bench reference
```

## Citation

Academic citation:

```bibtex
@software{ntt_anafis,
  author       = {Martins, Pedro},
  orcid        = {0009-0001-8170-2930},
  title        = {NTTAnaFis: Exact multi-prime number-theoretic transforms and native-limb multiplication in pure Rust},
  year         = {2026},
  url          = {https://github.com/CokieMiner/NTTAnaFis},
  version      = {0.1.0}
}
```

Contributors who make substantial contributions to the project and would like
academic credit may request inclusion in future citation metadata. Attribution
is limited to the project's principal contributors.

## License

`ntt_anafis` is licensed under the Apache License, Version 2.0.
See [LICENSE](LICENSE) for the full text.
