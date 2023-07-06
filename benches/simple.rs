
use std::hash::BuildHasherDefault;

use criterion::{criterion_group, criterion_main, Criterion};
use hyperloglog::HyperLogLog;
use seahash::SeaHasher;

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut i = 0;

    let mut g = c.benchmark_group("add");

    g.throughput(criterion::Throughput::Elements(1));
    
    let h = HyperLogLog::new(BuildHasherDefault::<SeaHasher>::default(), 4);
    g.bench_function("b: 4", |b| b.iter(|| { h.add(i); i+=1 } ));

    let h = HyperLogLog::new(BuildHasherDefault::<SeaHasher>::default(), 8);
    g.bench_function("b: 8", |b| b.iter(|| { h.add(i); i+=1 } ));

    let h = HyperLogLog::new(BuildHasherDefault::<SeaHasher>::default(), 16);
    g.bench_function("b: 16", |b| b.iter(|| { h.add(i); i+=1 } ));

    g.finish();

}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
