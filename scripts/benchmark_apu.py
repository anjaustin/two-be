#!/usr/bin/env python3
"""
Benchmark Neural APU Performance

Measures:
- Cache hit/miss rates
- Opcode execution latency
- Throughput (ops/sec)
- Memory footprint
"""

import time
import numpy as np
from bbdos.kernel import NeuralAPU, NeuralCache, CacheLine, CacheStatus


def benchmark_cache(cache_size: int = 16, n_iterations: int = 50000):
    """Benchmark L-Cache performance."""
    print(f"\n{'=' * 60}")
    print(f"L-CACHE BENCHMARK (size={cache_size}, iterations={n_iterations})")
    print(f"{'=' * 60}")

    cache = NeuralCache(cache_size=cache_size)

    opcodes = [f"TMUL_{i}" for i in range(cache_size)]
    for i, op in enumerate(opcodes):
        line = CacheLine(opcode_id=op, config=0)
        cache.store(op, line)

    start = time.perf_counter()
    for i in range(n_iterations):
        op = opcodes[i % len(opcodes)]
        cache.lookup(op)
    elapsed = time.perf_counter() - start

    ops_per_sec = n_iterations / elapsed
    print(f"  Total ops:      {n_iterations:,}")
    print(f"  Time:           {elapsed * 1000:.2f} ms")
    print(f"  Ops/sec:        {ops_per_sec:,.0f}")
    print(f"  Hit rate:       {cache.stats.hit_rate:.2%}")
    print(f"  Evictions:      {cache.stats.evictions}")
    print(f"  Rejected:       {cache.stats.rejected}")

    return ops_per_sec


def benchmark_tadd(n_iterations: int = 1000):
    """Benchmark TADD operation."""
    print(f"\n{'=' * 60}")
    print(f"TADD BENCHMARK (iterations={n_iterations})")
    print(f"{'=' * 60}")

    apu = NeuralAPU(cache_size=16)

    for b in [1, 4, 16, 64, 256]:
        for c in [64, 256, 1024]:
            a = np.random.randn(b, c).astype(np.float32)
            b_arr = np.random.randn(b, c).astype(np.float32)

            for _ in range(10):
                apu.exec("TADD", a, b_arr)

            start = time.perf_counter()
            for _ in range(n_iterations):
                apu.exec("TADD", a, b_arr)
            elapsed = time.perf_counter() - start

            ops_per_sec = n_iterations / elapsed
            print(
                f"  TADD {b}x{c}: {ops_per_sec:,.0f} ops/sec ({elapsed / n_iterations * 1000:.4f} ms/op)"
            )


def benchmark_tgate(n_iterations: int = 1000):
    """Benchmark TGATE operation."""
    print(f"\n{'=' * 60}")
    print(f"TGATE BENCHMARK (iterations={n_iterations})")
    print(f"{'=' * 60}")

    apu = NeuralAPU(cache_size=16)

    for b in [1, 4, 16, 64, 256]:
        for c in [64, 256, 1024]:
            for tiles in [4, 8]:
                x = np.random.randn(b, c).astype(np.float32)
                logits = np.random.randn(b, tiles).astype(np.float32)

                for _ in range(10):
                    apu.exec("TGATE", x, logits)

                start = time.perf_counter()
                for _ in range(n_iterations):
                    apu.exec("TGATE", x, logits)
                elapsed = time.perf_counter() - start

                ops_per_sec = n_iterations / elapsed
                print(
                    f"  TGATE {b}x{c} tiles={tiles}: {ops_per_sec:,.0f} ops/sec ({elapsed / n_iterations * 1000:.4f} ms/op)"
                )


def benchmark_tmul_simple(n_iterations: int = 500):
    """Benchmark simple TMUL operation."""
    print(f"\n{'=' * 60}")
    print(f"TMUL BENCHMARK (iterations={n_iterations})")
    print(f"{'=' * 60}")

    apu = NeuralAPU(cache_size=16)

    for m in [1, 4, 16, 64]:
        k, n = 128, 256
        a = np.random.randn(m, k).astype(np.float32)
        w = np.random.randn(k, n).astype(np.float32)
        s = np.ones(n, dtype=np.float32)

        for _ in range(10):
            apu.exec("TMUL", a, weights=w, scales=s)

        start = time.perf_counter()
        for _ in range(n_iterations):
            apu.exec("TMUL", a, weights=w, scales=s)
        elapsed = time.perf_counter() - start

        ops_per_sec = n_iterations / elapsed
        flops = 2 * m * k * n * n_iterations
        gflops = flops / elapsed / 1e9
        print(
            f"  TMUL {m}x{k}x{n}: {ops_per_sec:,.0f} ops/sec ({elapsed / n_iterations * 1000:.2f} ms/op) {gflops:.2f} GFLOPS"
        )


def benchmark_cache_warmup(
    cache_size: int = 16, n_opcodes: int = 16, n_iterations: int = 1000
):
    """Benchmark cache warmup behavior."""
    print(f"\n{'=' * 60}")
    print(f"CACHE WARMUP BENCHMARK")
    print(f"{'=' * 60}")

    apu = NeuralAPU(cache_size=cache_size)

    a = np.random.randn(16, 64).astype(np.float32)
    b_arr = np.random.randn(16, 64).astype(np.float32)

    print(f"  First run (cold cache)...")
    start = time.perf_counter()
    for _ in range(n_iterations):
        apu.exec("TADD", a, b_arr)
    cold_time = time.perf_counter() - start
    print(
        f"  Cold: {cold_time * 1000:.2f} ms ({cold_time / n_iterations * 1000:.4f} ms/op)"
    )

    print(f"  Second run (warm cache)...")
    start = time.perf_counter()
    for _ in range(n_iterations):
        apu.exec("TADD", a, b_arr)
    warm_time = time.perf_counter() - start
    print(
        f"  Warm: {warm_time * 1000:.2f} ms ({warm_time / n_iterations * 1000:.4f} ms/op)"
    )

    speedup = cold_time / warm_time
    print(f"  Speedup: {speedup:.2f}x")

    stats = apu.cache_stats()
    print(f"  Hit rate: {stats['hit_rate']:.2%}")


def benchmark_concurrent_access(n_threads: int = 8, n_ops_per_thread: int = 1000):
    """Benchmark concurrent cache access."""
    print(f"\n{'=' * 60}")
    print(f"CONCURRENT ACCESS ({n_threads} threads, {n_ops_per_thread} ops each)")
    print(f"{'=' * 60}")

    import threading

    cache = NeuralCache(cache_size=16)

    opcodes = [f"TMUL_{i}" for i in range(16)]
    for op in opcodes:
        cache.store(op, CacheLine(opcode_id=op, config=0))

    results = []

    def worker(thread_id):
        start = time.perf_counter()
        for i in range(n_ops_per_thread):
            op = opcodes[i % len(opcodes)]
            cache.lookup(op)
        elapsed = time.perf_counter() - start
        results.append((thread_id, elapsed))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]

    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total_time = time.perf_counter() - start

    total_ops = n_threads * n_ops_per_thread
    ops_per_sec = total_ops / total_time

    print(f"  Total ops:     {total_ops:,}")
    print(f"  Time:          {total_time * 1000:.2f} ms")
    print(f"  Ops/sec:       {ops_per_sec:,.0f}")
    print(f"  Hit rate:      {cache.stats.hit_rate:.2%}")


def benchmark_memory():
    """Estimate memory footprint."""
    print(f"\n{'=' * 60}")
    print(f"MEMORY FOOTPRINT ESTIMATE")
    print(f"{'=' * 60}")

    apu = NeuralAPU(cache_size=16)

    a = np.random.randn(64, 128).astype(np.float32)
    b_arr = np.random.randn(64, 128).astype(np.float32)

    a_bytes = a.nbytes

    print(f"  Input (64x128 fp32):     {a_bytes / 1024:.1f} KB")
    print(f"  Cache entries:            16")
    print(f"  Per-entry metadata:       ~200 bytes")
    print(f"  Estimated overhead:      ~3 KB")


def main():
    print("\n" + "=" * 60)
    print("NEURAL APU BENCHMARK SUITE")
    print("=" * 60)

    benchmark_cache(cache_size=8, n_iterations=50000)
    benchmark_cache(cache_size=16, n_iterations=50000)
    benchmark_cache(cache_size=32, n_iterations=50000)

    benchmark_tadd(n_iterations=2000)
    benchmark_tgate(n_iterations=2000)
    benchmark_tmul_simple(n_iterations=500)

    benchmark_cache_warmup(cache_size=16, n_opcodes=16, n_iterations=5000)

    benchmark_concurrent_access(n_threads=8, n_ops_per_thread=5000)

    benchmark_memory()

    print(f"\n{'=' * 60}")
    print("BENCHMARK COMPLETE")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
