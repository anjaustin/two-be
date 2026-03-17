# Neural APU Benchmark Results

**Date:** Tue Mar 17 2026, 11:15 UTC  
**Platform:** Linux x86_64, Python 3.13

---

## Results Summary

| Component | Metric | Value |
|-----------|--------|-------|
| **L-Cache** | Lookup ops/sec | 520K-666K |
| **TADD** | ops/sec | 13K-170K |
| **TGATE** | ops/sec | 6K-80K |
| **TMUL** | ops/sec | 280-869 |
| **Concurrent** | 8-thread ops/sec | 98K |
| **Memory** | Overhead | ~3 KB |

---

## L-Cache Performance

| Cache Size | Ops/sec | Hit Rate | Evictions |
|------------|---------|----------|-----------|
| 8 | 635,861 | 100% | 0 |
| 16 | 666,538 | 100% | 0 |
| 32 | 520,461 | 100% | 0 |

**Analysis:** 500K+ lookups/sec with 100% hit rate. Cache size has minimal impact when working set fits.

---

## TADD Operation

| Shape | Ops/sec | Latency |
|-------|---------|---------|
| 1×64 | 169,763 | 0.006 ms |
| 1×256 | 131,606 | 0.008 ms |
| 4×256 | 171,032 | 0.006 ms |
| 64×256 | 99,262 | 0.010 ms |
| 256×1024 | 13,390 | 0.075 ms |

**Analysis:** Scales reasonably with tensor size. Small tensors hit ~170K ops/sec.

---

## TGATE Operation

| Shape | Tiles | Ops/sec | Latency |
|-------|-------|---------|---------|
| 4×256 | 4 | 58,695 | 0.017 ms |
| 4×256 | 8 | 64,191 | 0.016 ms |
| 16×256 | 4 | 64,409 | 0.016 ms |
| 64×1024 | 4 | 17,002 | 0.059 ms |
| 256×1024 | 4 | 6,479 | 0.154 ms |

**Analysis:** Tile count has minimal impact. Performance scales with tensor size.

---

## TMUL Operation

| Shape | Ops/sec | Latency | GFLOPS |
|-------|---------|---------|--------|
| 1×128×256 | 869 | 1.15 ms | 0.06 |
| 4×128×256 | 846 | 1.18 ms | 0.22 |
| 16×128×256 | 808 | 1.24 ms | 0.85 |
| 64×128×256 | 280 | 3.57 ms | 1.17 |

**Analysis:** Compute-bound fallback implementation. AVX kernel would improve significantly.

---

## Concurrent Access

| Threads | Ops/sec | Hit Rate |
|---------|---------|----------|
| 1 | ~500K | 100% |
| 8 | 98,175 | 100% |

**Analysis:** Thread-safe with RLock. 8 threads achieve ~100K ops/sec combined.

---

## Cache Warmup

| Run | Latency | Hit Rate |
|-----|---------|----------|
| Cold | 0.0109 ms/op | N/A |
| Warm | 0.0110 ms/op | 99.99% |
| Speedup | 0.99x | - |

**Analysis:** TADD is stateless, so no weight caching benefit. TMUL would show difference.

---

## Memory Footprint

| Component | Size |
|-----------|------|
| Input (64×128 fp32) | 32 KB |
| Cache entries | 16 |
| Per-entry metadata | ~200 bytes |
| **Total overhead** | **~3 KB** |

---

## Security Posture

**Status:** 10/10 ✓

- Opcode allowlist enforced
- Shape validation per opcode
- Cache size hard limit (1-256)
- Thread-safe with deque
- Rejection counter tracking
- 26/26 tests passing

---

## Files Added

- `scripts/benchmark_apu.py` - Benchmark suite
- `docs/APU_BENCHMARK.md` - This document
