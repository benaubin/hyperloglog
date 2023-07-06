use std::{hash::BuildHasherDefault, time::{Instant, Duration}};

use seahash::SeaHasher;

#[derive(Clone, Copy, tabled::Tabled)]
struct Case {
    b: u8,
    num_threads: usize,
    elements_per_thread: usize,
}

#[derive(tabled::Tabled)]
struct Results {
    duration: f64,
    estimate: f64,
    error: f64,
    z: f64
}

fn main() {

    let cases = [
        Case {
            b: 12,
            num_threads: 1,
            elements_per_thread: 8 * 1_000_0000
        },
        Case {
            b: 12,
            num_threads: 2,
            elements_per_thread: 4 * 1_000_0000
        },
        Case {
            b: 12,
            num_threads: 8,
            elements_per_thread: 1_000_0000
        },
        Case {
            b: 12,
            num_threads: 8,
            elements_per_thread: 20_000_0000
        },
        Case {
            b: 12,
            num_threads: 16,
            elements_per_thread: 20_000_0000
        },
        Case {
            b: 16,
            num_threads: 16,
            elements_per_thread: 20_000_0000
        },
        Case {
            b: 8,
            num_threads: 16,
            elements_per_thread: 20_000_0000
        },
        Case {
            b: 4,
            num_threads: 16,
            elements_per_thread: 20_000_0000
        },
        Case {
            b: 4,
            num_threads: 8,
            elements_per_thread: 1_000_0000
        },
        Case {
            b: 4,
            num_threads: 2,
            elements_per_thread: 4_000_0000
        },
        Case {
            b: 4,
            num_threads: 1,
            elements_per_thread: 8_000_0000
        },
        Case {
            b: 4,
            num_threads: 2,
            elements_per_thread: 8_000_0000
        },
    ];

    let iter = cases.into_iter().map(|case| {
        let Case { num_threads, elements_per_thread, b } = case;
        
        let start = Instant::now();
        let hll = hyperloglog::HyperLogLog::new(BuildHasherDefault::<SeaHasher>::default(), b);
        std::thread::scope(|s| {
            for i in 0..num_threads {
                let hll = &hll;
                s.spawn(move || {
                    let start = elements_per_thread * i;
                    for n in 0..elements_per_thread {
                        hll.add(n + start)
                    }
                });
            }
        });
        let duration = start.elapsed().as_secs_f64();
        let actual_count = num_threads * elements_per_thread;
        let estimate = hll.cardinality();
        let error = (estimate - actual_count as f64) / actual_count as f64;
        let z = error / hll.stderr();

        let results = Results {
            duration,
            estimate,
            error,
            z
        };

        (case, results)
    });

    println!("{}", tabled::Table::new(iter));
}
