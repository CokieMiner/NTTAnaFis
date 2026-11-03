# Continuous integration

[ci.yml](ci.yml) runs on pushes and pull requests to `main`, on version tags, and on
manual dispatch.

| Job | Scope |
| --- | --- |
| `quality` | Formatting, source audits, nightly Clippy, and rustdoc |
| `tools` | Workflow validation and Python tool regressions |
| `msrv` | Every target with Rust 1.96.0, and CoreAnaFis with 1.85.0 |
| `check_archs` | Clippy and assembly of every target and backend configuration |
| `test` | Native tests, and benchmark compilation |
| `test_x86` | Kernel, CRT, and product tests of AVX-512 IFMA, AVX-512F, AVX2, and portable kernels under Intel SDE |
| `qemu` | QEMU 11.1.3 user emulators, cached |
| `test_cross` | NEON, s390x, VSX, LSX and LASX, and RVV under QEMU |
| `test_container` | 32-bit x86 and ARMv7 through `cross` |
| `publish` | crates.io release of a `v*` tag after every check passes |

Each job checks this repository out as `NTTAnaFis` beside `CoreAnaFis`, and
[actions/workspace](../actions/workspace/action.yml) writes the workspace manifest the
two share locally. Without a committed lockfile, dependencies resolve on each run.

`test_cross` builds one static test binary per target with `cross` and runs it once per
CPU configuration:

| Target | QEMU `-cpu` |
| :--- | :--- |
| `aarch64` | `max` |
| `s390x` | `max`, then without the enhancements facilities, then without the vector facility |
| `powerpc64le` | `power8`, `power9` |
| `loongarch64` | `la464`, `la464,lasx=off` |
| `riscv64` | `v=on` with `vlen` of 128, 256, and 512, and `v=off` |
| `powerpc64` | `power8`, experimental: it has never run, and stable Rust compiles its scalar kernels only |

The QEMU source archive and Actionlint have fixed versions and SHA-256 checks.

`publish` uses crates.io trusted publishing. The first release of a crate is manual;
afterwards, register this repository, `ci.yml`, and the `crates-io` environment as its
trusted publisher. `core_anafis` must be on crates.io first.
