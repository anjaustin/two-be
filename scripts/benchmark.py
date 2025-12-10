#!/usr/bin/env python3
"""
TriX Kernel Benchmark

Measures speedup at various sparsity levels and outputs CSV data.
"""

import argparse
import sys
import os
import time
import csv

import numpy as np

# Add paths for imports
sys.path.insert(0, "/workspace/BBDOS")
from trix import pack_weights_np, trix_forward_np, _lib


def benchmark_sparsity(
    batch: int = 32,
    in_features: int = 512,
    out_features: int = 2048,
    num_tiles: int = 4,
    warmup_iters: int = 5,
    bench_iters: int = 50
):
    """
    Benchmark kernel at different sparsity levels.
    
    Returns:
        List of dicts with benchmark results
    """
    np.random.seed(42)
    
    # Create test data
    input_data = np.random.randn(batch, in_features).astype(np.float32)
    weights = np.random.choice([-1.0, 0.0, 1.0], size=(out_features, in_features)).astype(np.float32)
    scales = np.ones(out_features, dtype=np.float32)
    packed = pack_weights_np(weights)
    
    results = []
    
    for num_active in range(num_tiles, 0, -1):
        sparsity = 1.0 - (num_active / num_tiles)
        
        # Create gate mask with specified number of active tiles
        gate_mask = np.zeros((batch, num_tiles), dtype=np.int8)
        gate_mask[:, :num_active] = 1
        
        # Warmup
        for _ in range(warmup_iters):
            trix_forward_np(input_data, packed, scales, gate_mask, in_features, out_features, num_tiles)
        
        # Benchmark
        times = []
        for _ in range(bench_iters):
            start = time.perf_counter()
            trix_forward_np(input_data, packed, scales, gate_mask, in_features, out_features, num_tiles)
            times.append(time.perf_counter() - start)
        
        avg_time = np.mean(times) * 1000  # Convert to ms
        std_time = np.std(times) * 1000
        
        results.append({
            'sparsity': sparsity,
            'active_tiles': num_active,
            'time_ms': avg_time,
            'std_ms': std_time
        })
    
    return results


def compute_speedups(results):
    """Add speedup relative to baseline (0% sparsity)."""
    baseline = results[0]['time_ms']  # 0% sparsity is first
    
    for r in results:
        r['speedup'] = baseline / r['time_ms']
    
    return results


def print_results(results):
    """Print formatted benchmark results."""
    print("\n" + "=" * 60)
    print("TriX Kernel Benchmark Results")
    print("=" * 60)
    print(f"{'Sparsity':<12} {'Tiles':<8} {'Time (ms)':<12} {'Speedup':<10}")
    print("-" * 60)
    
    for r in results:
        print(f"{r['sparsity']*100:>6.0f}%      {r['active_tiles']:<8} {r['time_ms']:>8.2f}     {r['speedup']:>6.2f}x")
    
    print("=" * 60)


def save_csv(results, filename):
    """Save results to CSV file."""
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['sparsity', 'active_tiles', 'time_ms', 'std_ms', 'speedup'])
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description="TriX Kernel Benchmark")
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--in-features', type=int, default=512, help='Input features')
    parser.add_argument('--out-features', type=int, default=2048, help='Output features')
    parser.add_argument('--num-tiles', type=int, default=4, help='Number of tiles')
    parser.add_argument('--warmup', type=int, default=5, help='Warmup iterations')
    parser.add_argument('--iters', type=int, default=50, help='Benchmark iterations')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file')
    args = parser.parse_args()
    
    if not _lib.available:
        print("ERROR: C++ library not available. Build with:")
        print("  cd bbdos/kernel && mkdir build && cd build && cmake .. && make")
        sys.exit(1)
    
    print(f"Configuration:")
    print(f"  Batch: {args.batch}")
    print(f"  In features: {args.in_features}")
    print(f"  Out features: {args.out_features}")
    print(f"  Tiles: {args.num_tiles}")
    print(f"  Iterations: {args.iters}")
    
    results = benchmark_sparsity(
        batch=args.batch,
        in_features=args.in_features,
        out_features=args.out_features,
        num_tiles=args.num_tiles,
        warmup_iters=args.warmup,
        bench_iters=args.iters
    )
    
    results = compute_speedups(results)
    print_results(results)
    
    if args.output:
        save_csv(results, args.output)
    
    # Verify key claim: 4x speedup at 75% sparsity
    for r in results:
        if r['sparsity'] == 0.75:
            if r['speedup'] >= 3.5:
                print(f"\n✓ KEY CLAIM VERIFIED: {r['speedup']:.2f}x speedup at 75% sparsity")
            else:
                print(f"\n✗ KEY CLAIM FAILED: Only {r['speedup']:.2f}x speedup at 75% sparsity")
            break


if __name__ == "__main__":
    main()
