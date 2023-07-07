
use std::hash::BuildHasherDefault;

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkGroup, measurement::WallTime};
use hyperloglog::HyperLogLog;
use seahash::SeaHasher;

pub fn benches() {
    let mut criterion: Criterion<_> = Criterion::default().configure_from_args();

    let mut g = criterion.benchmark_group("merge");

    g.throughput(criterion::Throughput::Elements(1_000_000));

    g.bench_with_input("8", &8, |ben, b| {
        ben.iter_batched(|| {
            let a = HyperLogLog::new(BuildHasherDefault::<SeaHasher>::default(), *b);
            let b = HyperLogLog::new(BuildHasherDefault::<SeaHasher>::default(), *b);
            for n in 0..1_000_000 { a.add(n); }
            for n in 500_000..1_500_000 { b.add(n); }
            (a, b)
        }, |(a,b)| a.merge(&b), criterion::BatchSize::LargeInput)
    });

    g.bench_with_input("16", &16, |ben, b| {
        ben.iter_batched(|| {
            let a = HyperLogLog::new(BuildHasherDefault::<SeaHasher>::default(), *b);
            let b = HyperLogLog::new(BuildHasherDefault::<SeaHasher>::default(), *b);
            for n in 0..1_000_000 { a.add(n); }
            for n in 500_000..1_500_000 { b.add(n); }
            (a, b)
        }, |(a,b)| a.merge(&b), criterion::BatchSize::LargeInput)
    });

    g.finish();

}

criterion_main!(benches);
