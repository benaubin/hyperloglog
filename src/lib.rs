use std::{hash::{BuildHasher, Hash, Hasher}};


struct Registers {
    words: Box<[u32]>,
    int_size: u32,
}

impl Registers {
    pub fn new(len: usize, int_size: u32) -> Self {
        assert!(int_size == 5 || int_size == 6);
        let ints_per_word = u32::BITS / int_size;
        let words = (len + ints_per_word as usize - 1) / ints_per_word as usize;
        Self {
            words: vec![0; words].into_boxed_slice(),
            int_size,
        }
    }
    
    pub fn incr(&mut self, j: u64, p: u32) -> Option<(u32, u32)> {
        let ints_per_word = (u32::BITS / self.int_size) as u64;
        let word = (j / ints_per_word) as usize;
        let offset = (j % ints_per_word) as u32 * self.int_size;

        let mask = (1 << self.int_size) - 1;
        let val = p & mask;

        let old_word = self.words[word];
        let old_val = (old_word >> offset) & mask;
        if old_val >= val { return None }

        let new_word = (old_word & !(mask << offset)) | (val << offset);

        self.words[word] = new_word;

        Some((old_val, val))
    }
}


const RECIP_PRECISION: u32 = 60;

pub struct HyperLogLog<H: BuildHasher> {
    registers: Registers,
    counters: Counters,
    hasher: H,
}

struct Counters {
    reciprical_sum: u128,
    zero_count: u64,
    b: u8,
    l: u8
}

impl<H> HyperLogLog<H>
where
    H: BuildHasher,
{
    /// parameters: hash function, log_2{number of bins}, length of hash
    pub fn new(hasher: H, b: u8, l: u8) -> Self {
        assert!(4 <= b && b <= 16);
        assert!(l == 32 || l == 64);

        let m = 1 << b;
        let registers = Registers::new(
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
            counters: Counters { reciprical_sum: (1u128 << RECIP_PRECISION) * m as u128, zero_count: m as u64, b, l }
        }
    }
    pub fn add<T: Hash>(&mut self, val: T) {
        let mut hasher = self.hasher.build_hasher();
        val.hash(&mut hasher);
        let x = hasher.finish();

        let j = x & ((1 << self.counters.b) - 1);
        let p = 1 + x.leading_zeros();
        
        if let Some((old, new)) = self.registers.incr(j, p) {
            let old_recip = 1u64 << RECIP_PRECISION - old;
            let new_recip = 1u64 << RECIP_PRECISION - new;

            self.counters.reciprical_sum -= (old_recip - new_recip) as u128;
            if old == 0 {
                self.counters.zero_count -= 1;
            }
        }
    }
    pub fn cardinality(&self) -> f64 {
        fn inner(c: &Counters) -> f64 {
            let max = 2f64.powi(c.l as i32);
            let m = 1 << c.b;
            let m_f64 = m as f64;

            let z_recip = fixed_point_to_floating_point(c.reciprical_sum, RECIP_PRECISION as i32);
            let a = match m {
                16 => 0.673,
                32 => 0.697,
                64 => 0.709,
                _ => 0.7213 / (1f64 + 1.079 / m_f64),
            };
            let e_unscaled = a / z_recip;
            let e = e_unscaled * m_f64.powi(2); // the “raw” HyperLogLog estimate

            if e_unscaled * m_f64 <= 2.5f64 {
                // small range correction
                if c.zero_count != 0 {
                    let u: f64 = (c.b as f64) - (c.zero_count as f64).log2(); // u = log(m / V)
                    return m_f64 * u;
                }
            } else if e / max > 30.0 {
                // large range correction
                return -max * (1f64 - (e / max)).log2();
            }

            e
        }

        inner(&self.counters)
    }
}

fn fixed_point_to_floating_point(fixed: u128, ones_place: i32) -> f64 {
    const MANTISSA_BITS: i32 = f64::MANTISSA_DIGITS as i32 - 1;
    const MANTISSA_MASK: u64 = 0x000f_ffff_ffff_ffff;

    // now convert to IEEE floating point:
    if fixed == 0 {
        return 0.0;
    }

    // determine the mantissa... we only get to keep the 53 most significant digits.
    // we can assume that the floating point number is not zero
    // align the first 1 bit to be the hidden bit
    let shift = (u128::BITS - f64::MANTISSA_DIGITS) as i32 - fixed.leading_zeros() as i32;
    let mantissa = if shift > 0 {
        fixed >> shift
    } else {
        fixed << -shift
    } as u64
        & MANTISSA_MASK;

    // reconstruct the floating point
    let exp = MANTISSA_BITS - ones_place + shift as i32;
    let e_biased = (exp + 1023) as u64;
    f64::from_bits(e_biased << MANTISSA_BITS | mantissa)
}

#[cfg(test)]
mod tests {
    use seahash::SeaHasher;

    use super::*;

    #[test]
    fn fixed_to_float() {
        for n in [
            0u128,
            1,
            0x000f_ffff_ffff_ffff,
            0xffff_ffff_ffff_ffff,
            0x1000_0000_0000_0000_0000_0000_0000,
            0x1000_0000_0000_0000_0000_0000_0001,
            0x1000_0000_0000_1000_0000_0000_0001,
            0xffff_ffff_ffff_ffff_ffff_ffff,
            0xabcd_ef12_abcd_ef45_aacc,
        ] {
            let actual = fixed_point_to_floating_point(n, 64);
            let expected = n as f64 / (2.0f64).powi(64);
            assert!(actual - expected < 0.001, "{actual} ≠ {expected}")
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
