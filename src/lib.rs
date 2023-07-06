use std::hash::{BuildHasher, Hash, Hasher};

use bitvec::{field::BitField, prelude::BitArray, vec::BitVec, view::BitViewSized};

struct PackedVec {
    bits: BitVec,
    elem_size: usize,
}

impl PackedVec {
    pub fn new(len: usize, elem_size: usize) -> Self {
        assert!(1 <= elem_size && elem_size <= 32);

        Self {
            bits: BitVec::repeat(false, len * elem_size),
            elem_size,
        }
    }
    pub fn get(&self, idx: usize) -> u32 {
        self.bits[idx * self.elem_size..][..self.elem_size].load_le()
    }
    pub fn set(&mut self, idx: usize, val: u32) {
        let val = val.min((1 << self.elem_size) - 1);
        self.bits[idx * self.elem_size..][..self.elem_size].store_le(val)
    }
}

pub struct HyperLogLog<H: BuildHasher> {
    registers: PackedVec,
    b: usize,
    l: usize,
    hasher: H,
}

impl<H> HyperLogLog<H>
where
    H: BuildHasher,
{
    /// parameters: hash function, log_2{number of bins}, length of hash
    pub fn new(hasher: H, b: usize, l: usize) -> Self {
        assert!(4 <= b && b <= 16);
        assert!(l == 32 || l == 64);

        let m = 1 << b;
        let registers = PackedVec::new(
            m,
            match l {
                32 => 5,
                64 => 6,
                _ => unreachable!(),
            },
        );

        Self {
            hasher,
            registers,
            b,
            l,
        }
    }
    pub fn add<T: Hash>(&mut self, val: T) {
        fn inner(registers: &mut PackedVec, b: usize, x: u64) {
            let arr: BitArray<_> = x.into_bitarray();
            let (j, w) = arr.split_at(b);
            let j: usize = j.load_le();
            let p = 1 + w.leading_zeros() as u32;

            let prev = registers.get(j);
            if p > prev {
                registers.set(j, p);
            }
        }
        let mut hasher = self.hasher.build_hasher();
        val.hash(&mut hasher);
        let hash = hasher.finish();
        inner(&mut self.registers, self.b, hash);
    }
    pub fn cardinality(&self) -> f64 {
        fn inner(registers: &PackedVec, b: usize, l: usize) -> f64 {
            let m = 1 << b;
            let m_f64 = m as f64;

            let z_recip = harmonic_sum_of_powers((0..m).map(|j| registers.get(j) as u8));

            let a = match m {
                16 => 0.673,
                32 => 0.697,
                64 => 0.709,
                _ => 0.7213 / (1f64 + 1.079 / m_f64),
            };

            let max = 2f64.powi(l as i32);

            let e_unscaled = a / z_recip;
            let e = e_unscaled * m_f64.powi(2); // the “raw” HyperLogLog estimate

            if e_unscaled * m_f64 <= 2.5f64 {
                // small range correction
                let v = (0..m).filter(|j| registers.get(*j) == 0).count();
                if v == 0 {
                    return 0f64;
                } else if v != 0 {
                    let u: f64 = (b as f64) - (v as f64).log2(); // u = log(m / V)
                    return m_f64 * u;
                }
            } else if e / max > 30.0 {
                // large range correction
                return -max * (1f64 - (e / max)).log2();
            }

            e
        }

        inner(&self.registers, self.b, self.l)
    }
}

/// calculate the harmonic sum of powers of two.
///
/// implemented using 128-bit fixed-point arithmetic to prevent floating point rounding errors
fn harmonic_sum_of_powers(powers: impl Iterator<Item = u8>) -> f64 {
    // fixed point, with 64 decimal places
    let mut fixed_sum: u128 = 0;
    const FIXED_POINT: i32 = 64;
    const MANTISSA_BITS: i32 = f64::MANTISSA_DIGITS as i32 - 1;

    for p in powers {
        let p = p as i32;
        assert!(p <= FIXED_POINT);
        let recip = 1 << (FIXED_POINT - p);
        fixed_sum += recip;
    }

    // now convert to IEEE floating point:
    if fixed_sum == 0 {
        return 0.0;
    }

    // determine the mantissa... we only get to keep the 53 most significant digits.
    // we can assume that the floating point number is not zero
    // align the first 1 bit to be the hidden bit
    let shift = (u128::BITS - f64::MANTISSA_DIGITS) as i32 - fixed_sum.leading_zeros() as i32;
    let mantissa = if shift > 0 {
        fixed_sum >> shift
    } else {
        fixed_sum << -shift
    } as u64
        & 0x000f_ffff_ffff_ffff;

    // reconstruct the floating point
    let exp = MANTISSA_BITS - FIXED_POINT + shift as i32;
    let e_biased = (exp + 1023) as u64;
    f64::from_bits(e_biased << MANTISSA_BITS | mantissa)
}

#[cfg(test)]
mod tests {
    use seahash::SeaHasher;

    use super::*;

    #[test]
    fn harmonic_mean() {
        assert_eq!(harmonic_sum_of_powers([1u8, 1].iter().copied()), 1.0);
        assert_eq!(
            harmonic_sum_of_powers([0u8, 2, 2].iter().copied()),
            3.0 / 2.0
        );

        let pows = [1u8, 2, 3, 4, 5, 6, 7, 8, 9];
        // note that this only works when we're within the range of the floating point exponent [2^-10, 2^10]
        let expected = pows
            .iter()
            .map(|pow| (2f64).powi(-(*pow as i32)))
            .sum::<f64>();
        assert_eq!(harmonic_sum_of_powers(pows.iter().copied()), expected);

        let pows = [8u8, 8, 8, 1];
        let expected = pows
            .iter()
            .map(|pow| (2f64).powi(-(*pow as i32)))
            .sum::<f64>();
        assert_eq!(harmonic_sum_of_powers(pows.iter().copied()), expected);

        let pows = [58u8, 40];
        let expected = pows
            .iter()
            .map(|pow| (2f64).powi(-(*pow as i32)))
            .sum::<f64>();
        assert_eq!(harmonic_sum_of_powers(pows.iter().copied()), expected);
        let pows = [53u8, 40];
        let expected = pows
            .iter()
            .map(|pow| (2f64).powi(-(*pow as i32)))
            .sum::<f64>();
        assert_eq!(harmonic_sum_of_powers(pows.iter().copied()), expected);

        for i in 0..=64 {
            let pows = [i];
            let expected = pows
                .iter()
                .map(|pow| (2f64).powi(-(*pow as i32)))
                .sum::<f64>();
            assert_eq!(harmonic_sum_of_powers(pows.iter().copied()), expected);
        }
    }

    struct BuildHasherClone<H: Hasher + Clone>(H);
    impl<H: Hasher + Clone> BuildHasher for BuildHasherClone<H> {
        type Hasher = H;

        fn build_hasher(&self) -> Self::Hasher {
            self.0.clone()
        }
    }

    #[test]
    fn ten_thousand() {
        let b = 4;
        let m = 1 << b;
        let sterr = 1.04 / (m as f64).sqrt();
        let mut hll = HyperLogLog::new(BuildHasherClone(SeaHasher::new()), b, 32);
        assert_eq!(hll.cardinality(), 0f64);

        for n in 1..=10000 {
            hll.add(n);

            if n % 10 == 1 {
                let c = hll.cardinality();
                let rel_error = (c / n as f64) - 1.;
                let z = rel_error / sterr;

                assert!(
                    z.abs() <= 3.0,
                    "z was {z}, c: {c}, n: {n}, rel_er: {rel_error}"
                );
            }
        }
    }

    #[test]
    fn million() {
        let b = 4;
        let m = 1 << b;
        let sterr = 1.04 / (m as f64).sqrt();
        let mut hll = HyperLogLog::new(BuildHasherClone(SeaHasher::new()), b, 32);
        assert_eq!(hll.cardinality(), 0f64);

        for n in 1..=1_000_000 {
            hll.add(n);

            if n % 100_000 == 0 {
                let c = hll.cardinality();
                let rel_error = (c / n as f64) - 1.;
                let z = rel_error / sterr;

                assert!(
                    z.abs() <= 3.0,
                    "z was {z}, c: {c}, n: {n}, rel_er: {rel_error}"
                );
            }
        }
    }

    #[test]
    fn million_b8() {
        let b = 8;
        let m = 1 << b;
        let sterr = 1.04 / (m as f64).sqrt();
        let mut hll = HyperLogLog::new(BuildHasherClone(SeaHasher::new()), b, 32);
        assert_eq!(hll.cardinality(), 0f64);

        for n in 1..=1_000_000 {
            hll.add(n);

            if n % 100_000 == 0 {
                let c = hll.cardinality();
                let rel_error = (c / n as f64) - 1.;
                let z = rel_error / sterr;

                assert!(
                    z.abs() <= 3.0,
                    "z was {z}, c: {c}, n: {n}, rel_er: {rel_error}"
                );
            }
        }
    }

    #[test]
    fn million_b16() {
        let b = 16;
        let m = 1 << b;
        let sterr = 1.04 / (m as f64).sqrt();
        let mut hll = HyperLogLog::new(BuildHasherClone(SeaHasher::new()), b, 32);
        assert_eq!(hll.cardinality(), 0f64);

        for n in 1..=1_000_000 {
            hll.add(n);

            if n % 250_000 == 0 {
                let c = hll.cardinality();
                let rel_error = (c / n as f64) - 1.;
                let z = rel_error / sterr;

                assert!(
                    z.abs() <= 4.0,
                    "z was {z}, c: {c}, n: {n}, rel_er: {rel_error}"
                );
            }
        }
    }
}
