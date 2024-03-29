# hyperloglog

a concurrent, super fast, pretty-well-tested and fully safe hyperloglog for rust with no dependencies.

counts about ~1 billion elements per second (including time to compute SeaHash) with 16 threads. that's ~8 GiB of hashes 
**computed and counted** per second, with constant memory overhead (around 3KiB for an error rate of ≤ 1%).

the concurrency support is surprisingly really really good. it's effectively zero-cost,
there's no speed up from removing atomic support, and the speedup is nearly linear in number of cores (on my M1 MacBook Pro).

want to read more about the implementation process and how the hyperloglog structure works? i wrote a [Thread](https://www.threads.net/t/CuYj0OzsbZi) on it!

## performance:

```
+----+-------------+---------------------+--------------------+--------------------+-------------------------+----------------------+
| b  | num_threads | elements_per_thread | duration           | est. cardinality   | error                   | z score              |
+----+-------------+---------------------+--------------------+--------------------+-------------------------+----------------------+
| 12 | 1           | 80000000            | 0.783505583        | 81407162.3289801   | 0.017589529112251288    | 1.0824325607539254   |
+----+-------------+---------------------+--------------------+--------------------+-------------------------+----------------------+
| 12 | 2           | 40000000            | 0.392236           | 81407162.3289801   | 0.017589529112251288    | 1.0824325607539254   |
+----+-------------+---------------------+--------------------+--------------------+-------------------------+----------------------+
| 12 | 8           | 10000000            | 0.115582916        | 81407162.3289801   | 0.017589529112251288    | 1.0824325607539254   |
+----+-------------+---------------------+--------------------+--------------------+-------------------------+----------------------+
| 12 | 8           | 200000000           | 2.054057208        | 1585805566.3940902 | -0.00887152100369364    | -0.5459397540734547  |
+----+-------------+---------------------+--------------------+--------------------+-------------------------+----------------------+
| 12 | 16          | 200000000           | 3.7321913330000003 | 3227717320.617457  | 0.008661662692955286    | 0.533025396489556    |
+----+-------------+---------------------+--------------------+--------------------+-------------------------+----------------------+
| 16 | 16          | 200000000           | 3.728134916        | 3198889302.695715  | -0.00034709290758907796 | -0.08543825417577304 |
+----+-------------+---------------------+--------------------+--------------------+-------------------------+----------------------+
| 8  | 16          | 200000000           | 3.6748197080000002 | 2962602353.834109  | -0.074186764426841      | -1.1413348373360153  |
+----+-------------+---------------------+--------------------+--------------------+-------------------------+----------------------+
| 4  | 16          | 200000000           | 3.756755708        | 4388265829.463295  | 0.3713330717072797      | 1.4282041219510757   |
+----+-------------+---------------------+--------------------+--------------------+-------------------------+----------------------+
| 4  | 8           | 10000000            | 0.100113625        | 85015087.94729412  | 0.06268859934117645     | 0.24110999746606324  |
+----+-------------+---------------------+--------------------+--------------------+-------------------------+----------------------+
| 4  | 2           | 40000000            | 0.390384833        | 85015087.94729412  | 0.06268859934117645     | 0.24110999746606324  |
+----+-------------+---------------------+--------------------+--------------------+-------------------------+----------------------+
| 4  | 1           | 80000000            | 0.779172958        | 85015087.94729412  | 0.06268859934117645     | 0.24110999746606324  |
+----+-------------+---------------------+--------------------+--------------------+-------------------------+----------------------+
| 4  | 2           | 80000000            | 0.785492541        | 208090923.9294848  | 0.3005682745592801      | 1.1560318252280004   |
+----+-------------+---------------------+--------------------+--------------------+-------------------------+----------------------+
```

## memory usage: 

the number of registers is configurable by the parameter b, set so that the number of registers `m = 2^b`

the memory layout of this crate stores 5 6-bit registers in every 32 bit word (wasting 2 bits per word to allow for
easier atomics and better performance). `ceil(m / 5) * 4` gives the bytes of memory used for `m` registers



