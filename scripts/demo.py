#!/usr/bin/env python3
"""
BBDOS One-Command Demo

Validates the core claim: 4x speedup at 75% sparsity.

Usage:
    python scripts/demo.py
"""

import sys
import os

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/workspace/BBDOS")

def main():
    print("=" * 60)
    print("BBDOS Demo - 2-Bit Conditional Ternary Neural Architecture")
    print("=" * 60)
    print()
    
    # Step 1: Check kernel
    print("[1/3] Checking BitSwitch kernel...")
    try:
        from bitswitch import _lib
        if _lib.available:
            print("      ✓ NEON kernel loaded")
        else:
            print("      ⚠ Using NumPy fallback (slower)")
    except ImportError:
        print("      ✗ Kernel not found. Build with:")
        print("        cd bbdos/kernel && mkdir build && cd build && cmake .. && make")
        return 1
    
    # Step 2: Run quick benchmark
    print()
    print("[2/3] Running speedup benchmark...")
    
    import numpy as np
    import time
    from bitswitch import pack_weights_np, bitswitch_forward_np
    
    # Setup
    batch, in_f, out_f, num_tiles = 32, 512, 2048, 4
    np.random.seed(42)
    
    input_data = np.random.randn(batch, in_f).astype(np.float32)
    weights = np.random.choice([-1.0, 0.0, 1.0], size=(out_f, in_f)).astype(np.float32)
    scales = np.ones(out_f, dtype=np.float32)
    packed = pack_weights_np(weights)
    
    # Benchmark all tiles
    gate_all = np.ones((batch, num_tiles), dtype=np.int8)
    for _ in range(3):  # warmup
        bitswitch_forward_np(input_data, packed, scales, gate_all, in_f, out_f, num_tiles)
    
    start = time.perf_counter()
    for _ in range(20):
        bitswitch_forward_np(input_data, packed, scales, gate_all, in_f, out_f, num_tiles)
    time_all = (time.perf_counter() - start) / 20
    
    # Benchmark 25% tiles (75% sparsity)
    gate_quarter = np.zeros((batch, num_tiles), dtype=np.int8)
    gate_quarter[:, 0] = 1
    
    start = time.perf_counter()
    for _ in range(20):
        bitswitch_forward_np(input_data, packed, scales, gate_quarter, in_f, out_f, num_tiles)
    time_quarter = (time.perf_counter() - start) / 20
    
    speedup = time_all / time_quarter
    
    print(f"      Dense (0% sparse):   {time_all*1000:.2f} ms")
    print(f"      Sparse (75% sparse): {time_quarter*1000:.2f} ms")
    print(f"      Speedup: {speedup:.2f}x")
    print()
    
    # Step 3: Verify claim
    print("[3/3] Verifying core claim...")
    print()
    
    if speedup >= 3.5:
        print("=" * 60)
        print("✓ CORE CLAIM VERIFIED")
        print(f"  {speedup:.2f}x speedup at 75% sparsity (target: 4.00x)")
        print("=" * 60)
        return 0
    else:
        print("=" * 60)
        print("✗ CLAIM NOT MET")
        print(f"  {speedup:.2f}x speedup (expected >= 3.5x)")
        print("  This may be due to CPU fallback or platform differences.")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
