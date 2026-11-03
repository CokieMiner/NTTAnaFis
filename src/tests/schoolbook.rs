//! Independent native-radix product oracle for unit and public contract tests.

use alloc::{vec, vec::Vec};

#[cfg(target_pointer_width = "64")]
type Accumulator = u128;
#[cfg(target_pointer_width = "32")]
type Accumulator = u64;
#[cfg(target_pointer_width = "16")]
type Accumulator = u32;

/// Returns the complete native-limb product by schoolbook accumulation.
pub fn product(left: &[usize], right: &[usize]) -> Vec<usize> {
    let mut output = vec![
        0;
        left.len()
            .checked_add(right.len())
            .expect("small oracle output")
    ];
    let mask = Accumulator::try_from(usize::MAX).expect("native mask fits double limb");
    for (i, &a) in left.iter().enumerate() {
        let mut carry = 0;
        for (j, &b) in right.iter().enumerate() {
            let slot = output
                .get_mut(i.checked_add(j).expect("oracle index"))
                .expect("oracle product span");
            let value = Accumulator::try_from(a)
                .expect("native input")
                .checked_mul(Accumulator::try_from(b).expect("native input"))
                .expect("double-width product")
                .checked_add(Accumulator::try_from(*slot).expect("native output"))
                .expect("product plus output")
                .checked_add(carry)
                .expect("sum is below the double-width radix");
            *slot = usize::try_from(value & mask).expect("low native limb");
            carry = value >> usize::BITS;
        }
        if !right.is_empty() {
            *output
                .get_mut(i.checked_add(right.len()).expect("oracle carry index"))
                .expect("oracle carry span") = usize::try_from(carry).expect("native carry");
        }
    }
    output
}
