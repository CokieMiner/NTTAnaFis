//! Operands shared by the `ntt_anafis` benchmarks.
//!
//! Run the benchmarks from the workspace with
//! `cargo bench -p ntt_anafis_bench`; add `--features rayon` for parallel
//! products. This crate holds the benchmark dependencies, so `ntt_anafis`
//! itself stays free of them.

/// Returns `len` deterministic limbs with uniformly distributed bits.
///
/// The generator is SplitMix64 (Steele, Lea, and Flood 2014), truncated to the
/// native limb width.
#[must_use]
pub fn operand(len: usize, seed: u64) -> Vec<usize> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut mixed = state;
            mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            #[expect(
                clippy::cast_possible_truncation,
                reason = "Truncation keeps uniform low bits on narrower limbs"
            )]
            let limb = (mixed ^ (mixed >> 31)) as usize;
            limb
        })
        .collect()
}
