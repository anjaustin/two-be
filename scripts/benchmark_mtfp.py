#!/usr/bin/env python3
"""
Benchmark MTFP vs float32
"""

import time
import numpy as np
from bbdos.kernel import MTFP_8, MTFP_16, mtfp_add, mtfp_mul, NeuralAPU


def benchmark_pack(mtfp, n_ops=10000):
    """Benchmark pack operation."""
    values = np.random.randn(n_ops).astype(np.float32)

    start = time.perf_counter()
    for v in values:
        mtfp.pack(v)
    elapsed = time.perf_counter() - start

    return n_ops / elapsed


def benchmark_unpack(mtfp, n_ops=10000):
    """Benchmark unpack operation."""
    values = np.random.randn(n_ops).astype(np.float32)
    trits = mtfp.pack_array(values)

    start = time.perf_counter()
    for t in trits:
        mtfp.unpack(t)
    elapsed = time.perf_counter() - start

    return n_ops / elapsed


def benchmark_roundtrip(mtfp, n_ops=10000):
    """Benchmark pack + unpack."""
    values = np.random.randn(n_ops).astype(np.float32)

    start = time.perf_counter()
    for v in values:
        t = mtfp.pack(v)
        r = mtfp.unpack(t)
    elapsed = time.perf_counter() - start

    return n_ops / elapsed


def benchmark_mtfp_add(mtfp, n_ops=5000):
    """Benchmark MTFP add."""
    a = mtfp.pack_array(np.random.randn(n_ops).astype(np.float32))
    b = mtfp.pack_array(np.random.randn(n_ops).astype(np.float32))

    start = time.perf_counter()
    c = mtfp_add(a, b, mtfp)
    elapsed = time.perf_counter() - start

    return n_ops / elapsed


def benchmark_mtfp_mul(mtfp, n_ops=5000):
    """Benchmark MTFP multiply."""
    a = mtfp.pack_array(np.random.randn(n_ops).astype(np.float32))
    b = mtfp.pack_array(np.random.randn(n_ops).astype(np.float32))

    start = time.perf_counter()
    c = mtfp_mul(a, b, mtfp)
    elapsed = time.perf_counter() - start

    return n_ops / elapsed


def benchmark_apu_mtfp(n_ops=1000):
    """Benchmark MTFP via APU."""
    apu = NeuralAPU(cache_size=16)
    mtfp = MTFP_16

    a = mtfp.pack_array(np.random.randn(100).astype(np.float32))
    b = mtfp.pack_array(np.random.randn(100).astype(np.float32))

    start = time.perf_counter()
    for _ in range(n_ops):
        apu.exec("MTFP_ADD", a, b)
    elapsed = time.perf_counter() - start

    return n_ops / elapsed


def benchmark_float32_baseline(n_ops=10000):
    """Baseline float32 performance."""
    a = np.random.randn(n_ops).astype(np.float32)
    b = np.random.randn(n_ops).astype(np.float32)

    start = time.perf_counter()
    c = a + b
    elapsed = time.perf_counter() - start

    return n_ops / elapsed


def benchmark_memory():
    """Compare memory usage."""
    n = 10000

    fp32_values = np.random.randn(n).astype(np.float32)
    fp32_bytes = fp32_values.nbytes

    mtfp_8 = MTFP_8
    mtfp_16 = MTFP_16

    m8_trits = mtfp_8.pack_array(fp32_values)
    m16_trits = mtfp_16.pack_array(fp32_values)

    print(f"\n{'=' * 60}")
    print(f"MEMORY COMPARISON ({n} values)")
    print(f"{'=' * 60}")
    print(f"  float32:        {fp32_bytes / 1024:.1f} KB")
    print(
        f"  MTFP-8:         {m8_trits.nbytes / 1024:.1f} KB ({fp32_bytes / m8_trits.nbytes:.1f}x smaller)"
    )
    print(
        f"  MTFP-16:        {m16_trits.nbytes / 1024:.1f} KB ({fp32_bytes / m16_trits.nbytes:.1f}x smaller)"
    )


def main():
    print("\n" + "=" * 60)
    print("MTFP BENCHMARK SUITE")
    print("=" * 60)

    print(f"\n{'=' * 60}")
    print("PACK BENCHMARK (ops/sec)")
    print(f"{'=' * 60}")
    for mtfp, name in [(MTFP_8, "MTFP-8"), (MTFP_16, "MTFP-16")]:
        ops = benchmark_pack(mtfp)
        print(f"  {name}: {ops:,.0f}")

    print(f"\n{'=' * 60}")
    print("UNPACK BENCHMARK (ops/sec)")
    print(f"{'=' * 60}")
    for mtfp, name in [(MTFP_8, "MTFP-8"), (MTFP_16, "MTFP-16")]:
        ops = benchmark_unpack(mtfp)
        print(f"  {name}: {ops:,.0f}")

    print(f"\n{'=' * 60}")
    print("ROUNDTRIP BENCHMARK (ops/sec)")
    print(f"{'=' * 60}")
    for mtfp, name in [(MTFP_8, "MTFP-8"), (MTFP_16, "MTFP-16")]:
        ops = benchmark_roundtrip(mtfp)
        print(f"  {name}: {ops:,.0f}")

    print(f"\n{'=' * 60}")
    print("MTFP_ADD BENCHMARK (ops/sec)")
    print(f"{'=' * 60}")
    ops = benchmark_mtfp_add(MTFP_16)
    print(f"  MTFP-16: {ops:,.0f}")

    print(f"\n{'=' * 60}")
    print("MTFP_MUL BENCHMARK (ops/sec)")
    print(f"{'=' * 60}")
    ops = benchmark_mtfp_mul(MTFP_16)
    print(f"  MTFP-16: {ops:,.0f}")

    print(f"\n{'=' * 60}")
    print("APU MTFP BENCHMARK (ops/sec)")
    print(f"{'=' * 60}")
    ops = benchmark_apu_mtfp()
    print(f"  MTFP_ADD via APU: {ops:,.0f}")

    print(f"\n{'=' * 60}")
    print("FLOAT32 BASELINE (ops/sec)")
    print(f"{'=' * 60}")
    ops = benchmark_float32_baseline()
    print(f"  float32 add: {ops:,.0f}")

    benchmark_memory()

    print(f"\n{'=' * 60}")
    print("BENCHMARK COMPLETE")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
