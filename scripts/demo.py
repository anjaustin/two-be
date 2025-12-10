#!/usr/bin/env python3
"""
BBDOS One-Command Demo

Validates the core claim: 4x speedup at 75% sparsity.

Usage:
    python scripts/demo.py
"""

import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("=" * 60)
    print("BBDOS Demo - 2-Bit Conditional Ternary Neural Architecture")
    print("=" * 60)
    print()
    
    # Step 1: Check kernel
    print("[1/3] Checking TriX kernel...")
    try:
        from bbdos.kernel.bindings import TriXLinear, is_neon_available
        if is_neon_available():
            print("      ✓ NEON kernel loaded")
        else:
            print("      ⚠ Using PyTorch fallback (no NEON)")
    except Exception as e:
        print(f"      ✗ Kernel not found: {e}")
        print("        Build with: cd bbdos/kernel && mkdir build && cd build && cmake .. && make")
        return 1
    
    # Step 2: Run quick benchmark
    print()
    print("[2/3] Running speedup benchmark...")
    
    import torch
    import numpy as np
    
    # Setup
    batch, in_f, out_f, num_tiles = 32, 512, 2048, 4
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create a TriX layer
    layer = TriXLinear(in_f, out_f, num_tiles=num_tiles)
    layer.pack()  # Pack weights for NEON kernel
    layer.eval()  # Inference mode
    input_data = torch.randn(batch, in_f)
    
    # Benchmark all tiles active (dense)
    gate_all = torch.ones(batch, num_tiles)
    for _ in range(3):  # warmup
        _ = layer(input_data, gate_all)
    
    start = time.perf_counter()
    for _ in range(20):
        _ = layer(input_data, gate_all)
    time_all = (time.perf_counter() - start) / 20
    
    # Benchmark 25% tiles (75% sparsity)
    gate_quarter = torch.zeros(batch, num_tiles)
    gate_quarter[:, 0] = 1.0
    
    for _ in range(3):  # warmup
        _ = layer(input_data, gate_quarter)
    
    start = time.perf_counter()
    for _ in range(20):
        _ = layer(input_data, gate_quarter)
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
        print("⚠ PARTIAL VERIFICATION")
        print(f"  {speedup:.2f}x speedup (expected >= 3.5x)")
        print("  This may be due to CPU fallback or platform differences.")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
