//! # Atomic HyperLogLog
//! 
//! a concurrent, super fast, pretty-well-tested and fully safe hyperloglog for rust with no dependencies.
//! 
//! ```
//!# use std::hash::BuildHasherDefault;
//!# use seahash::SeaHasher;
//! use atomic_hyperloglog::HyperLogLog;
//! let h = HyperLogLog::new(BuildHasherDefault::<SeaHasher>::default(), 12);
//! for n in 0..10_000 {
//!     h.add(n);
//! }
//! let est = h.cardinality();
//! assert!(10_000.0 * 0.95 <= est && est <= 10_000.0 * 1.05, "{est}");
//! ```

use std::{
    hash::{BuildHasher, Hash, Hasher},
    sync::atomic::{AtomicU32, AtomicU64, Ordering},
};

struct Registers {
    words: Box<[AtomicU32]>,
    int_size: u32,
}

/// the bit length of registers
const REGISTER_SIZE: u8 = 6;

impl Registers {
    pub fn new(len: usize, int_size: u32) -> Self {
        let ints_per_word = u32::BITS / int_size;
        let words = (len + ints_per_word as usize - 1) / ints_per_word as usize;
        Self {
            words: Vec::from_iter(std::iter::repeat_with(|| AtomicU32::new(0)).take(words))
                .into_boxed_slice(),
            int_size,
        }
    }

    /// Increment the relevant register, given j and p (terms used in HLL paper).
    /// 
    /// Params:
    /// - j, the index of the register
    /// - p, `1 + leading zeros`
    pub fn incr(&self, j: u64, p: u32) -> Option<(u32, u32)> {
        let ints_per_word = (u32::BITS / self.int_size) as u64;
        let word = (j / ints_per_word) as usize;
        let offset = (j % ints_per_word) as u32 * self.int_size;

        let mask = (1 << self.int_size) - 1;
        let val = p & mask;

        let mut old_word = self.words[word].load(Ordering::Relaxed);

        loop {
            let old_val = (old_word >> offset) & mask;
            if old_val >= val {
                return None;
            }

            let new_word = (old_word & !(mask << offset)) | (val << offset);

            match self.words[word].compare_exchange(
                old_word,
                new_word,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Some((old_val, val)),
                Err(val) => old_word = val,
            };
        }
    }

    /// merge two register sets, modifying self, updating counters as our data changes
    pub fn merge(&self, other: &Self, counters: &Counters) {
        assert_eq!(self.int_size, other.int_size);
        assert_eq!(self.words.len(), other.words.len());

        let ints_per_word = (u32::BITS / self.int_size) as u64;
        let mask = (1 << self.int_size) - 1;


        for w_idx in 0..self.words.len() {
            let mut old_word = self.words[w_idx].load(Ordering::Relaxed);
            let (mut reciprocal_adj, mut zero_count_adj);
            loop {
                reciprocal_adj = 0;
                zero_count_adj = 0;
                let mut their_word = other.words[w_idx].load(Ordering::Relaxed);
                let mut our_word = old_word;
                let mut new_word = 0;
                for i in 0..ints_per_word {
                    let their_val = their_word & mask;
                    let our_val = our_word & mask;

                    let new_val = if their_val > our_val {
                        let old_recip = 1u64 << RECIP_PRECISION.saturating_sub(our_val);
                        let new_recip = 1u64 << RECIP_PRECISION.saturating_sub(their_val);
                        reciprocal_adj += old_recip - new_recip;
                        zero_count_adj += (our_val == 0) as u64;
                        their_val
                    } else {
                        our_val
                    };

                    new_word |= new_val << i * self.int_size as u64;
                    their_word = their_word >> self.int_size;
                    our_word = our_word >> self.int_size;
                }
                match self.words[w_idx].compare_exchange(old_word, new_word, Ordering::Relaxed, Ordering::Relaxed) {
                    Ok(_) => break,
                    Err(word) => {old_word = word}
                }
            }
            counters.reciprical_sum.fetch_sub(reciprocal_adj, Ordering::Relaxed);
            counters.zero_count.fetch_sub(zero_count_adj, Ordering::Relaxed);
        }
    }
}

/// fixed-point precision bits used for the reciprocal sum
const RECIP_PRECISION: u32 = 47;

/// A hyperloglog data structure, allowing count-distinct with limited memory overhead.
/// Fully concurrent with relaxed-only ordering and zero-unsafe code.
pub struct HyperLogLog<H: BuildHasher> {
    registers: Registers,
    counters: Counters,
    b: u8,
    hasher: H,
}

struct Counters {
    reciprical_sum: AtomicU64,
    zero_count: AtomicU64,
}

impl<H> HyperLogLog<H>
where
    H: BuildHasher,
{
    /// Create a new hyperloglog data structure
    /// 
    /// parameters: hasher: hash function, b = log_2{number of bins}
    pub fn new(hasher: H, b: u8) -> Self {
        assert!(4 <= b && b <= 16);

        let m = 1 << b;
        let registers = Registers::new(
            m,
            REGISTER_SIZE as u32
        );

        Self {
            hasher,
            registers,
            counters: Counters {
                reciprical_sum: AtomicU64::new((1u64 << RECIP_PRECISION) * m as u64),
                zero_count: AtomicU64::new(m as u64),
            },
            b,
        }
    }

    /// calculates the standard relative error for the given `b` parameter
    pub fn stderr(&self) -> f64 {
        let m = 1 << self.b;
        1.04 / (m as f64).sqrt()
    }

    /// Add a value to the count
    pub fn add<T: Hash>(&self, val: T) {
        let mut hasher = self.hasher.build_hasher();
        val.hash(&mut hasher);
        let x = hasher.finish();

        let j = x & ((1 << self.b) - 1);
        let p = 1 + x.leading_zeros();

        if let Some((old, new)) = self.registers.incr(j, p) {
            let old_recip = 1u64 << RECIP_PRECISION.saturating_sub(old);
            let new_recip = 1u64 << RECIP_PRECISION.saturating_sub(new);

            self.counters.reciprical_sum.fetch_sub(old_recip - new_recip, Ordering::Relaxed);
            if old == 0 {
                self.counters.zero_count.fetch_sub(1, Ordering::Relaxed);
            }
        }
    }

    /// merge other's count into self
    pub fn merge(&self, other: &Self) {
        assert_eq!(self.b, other.b);
        self.registers.merge(&other.registers, &self.counters);
    }

    /// Get the cardinality estimate
    pub fn cardinality(&self) -> f64 {
        fn inner(reciprical_sum: u64, zero_count: u64, b: u8) -> f64 {
            let max = 2f64.powi(RECIP_PRECISION as i32 + b as i32);
            let m = 1 << b;
            let m_f64 = m as f64;

            let z_recip = fixed_point_to_floating_point(reciprical_sum, RECIP_PRECISION as i32);
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
                if zero_count != 0 {
                    let u: f64 = (b as f64) - (zero_count as f64).log2(); // u = log(m / V)
                    return m_f64 * u;
                }
            } else if e / max > 30.0 {
                // large range correction
                return -max * (1f64 - (e / max)).log2();
            }

            e
        }

        inner(
            self.counters.reciprical_sum.load(Ordering::Relaxed),
            self.counters.zero_count.load(Ordering::Relaxed),
            self.b,
        )
    }
}

/// Convert a fixed point number to a floating point number
fn fixed_point_to_floating_point(fixed: u64, ones_place: i32) -> f64 {
    const MANTISSA_BITS: i32 = f64::MANTISSA_DIGITS as i32 - 1;
    const MANTISSA_MASK: u64 = 0x000f_ffff_ffff_ffff;

    // now convert to IEEE floating point:
    if fixed == 0 {
        return 0.0;
    }

    // determine the mantissa... we only get to keep the 53 most significant digits.
    // we can assume that the floating point number is not zero
    // align the first 1 bit to be the hidden bit
    let shift = (u64::BITS - f64::MANTISSA_DIGITS) as i32 - fixed.leading_zeros() as i32;
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
            0u64,
            1,
            0x000f_ffff_ffff_ffff,
            0xffff_ffff_ffff_ffff,
            0x1000_0000_0000,
            0x1000_0000_0001,
            0x1000_1000_0001,
            0xffff_ffff_ffff,
            0xabcd_ef12_abcd_ef45,
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
        let hll = HyperLogLog::new(BuildHasherClone(SeaHasher::new()), b);
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
        let hll = HyperLogLog::new(BuildHasherClone(SeaHasher::new()), b);
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
    fn merging_small() {
        let b = 8;
        let m = 1 << b;
        let sterr = 1.04 / (m as f64).sqrt();

        let hll1 = HyperLogLog::new(BuildHasherClone(SeaHasher::new()), b);
        let hll2 = HyperLogLog::new(BuildHasherClone(SeaHasher::new()), b);
        let hll3 = HyperLogLog::new(BuildHasherClone(SeaHasher::new()), b);

        hll1.add(1);
        hll2.add(2);
        hll3.add(3);

        hll1.merge(&hll2);
        hll2.merge(&hll3);

        assert_eq!(hll1.cardinality(), hll2.cardinality());
        assert_ne!(hll2.cardinality(), hll3.cardinality());
    }

    #[test]
    fn merging() {
        let b = 8;
        let m = 1 << b;
        let sterr = 1.04 / (m as f64).sqrt();

        let hll1 = HyperLogLog::new(BuildHasherClone(SeaHasher::new()), b);
        let hll2 = HyperLogLog::new(BuildHasherClone(SeaHasher::new()), b);
        let hll3 = HyperLogLog::new(BuildHasherClone(SeaHasher::new()), b);

        assert_eq!(hll1.cardinality(), 0f64);

        for n in 1..=1_000_000 {
            hll1.add(n);
            hll2.add(n);
            hll3.add(!n);
        }

        assert_eq!(hll1.cardinality(), hll2.cardinality());
        assert_ne!(hll1.cardinality(), hll3.cardinality());

        for n in 1_000_000..=2_000_000 {
            hll2.add(n);
        }

        assert_ne!(hll1.cardinality(), hll2.cardinality());

        hll1.merge(&hll2);

        assert_eq!(hll1.cardinality(), hll2.cardinality());

        let expected = hll2.cardinality() + hll3.cardinality();
        hll2.merge(&hll3);
        let error = (hll2.cardinality() - expected) / expected;
        let z = error / (sterr.powi(2) * 2.0).sqrt();
        assert!(z <= 1.0, "should be within 1 margin of error of difference after merging");
    }

    #[test]
    fn million_b8() {
        let b = 8;
        let m = 1 << b;
        let sterr = 1.04 / (m as f64).sqrt();
        let hll = HyperLogLog::new(BuildHasherClone(SeaHasher::new()), b);
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
        let hll = HyperLogLog::new(BuildHasherClone(SeaHasher::new()), b);
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
