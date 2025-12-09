#!/usr/bin/env python3
"""
CPU Trace Generator

Generates 6502 CPU execution traces for training the Neural CPU.
Uses multiprocessing for fast generation on multi-core systems.

Usage:
    python generate_traces.py --output data/traces.pt --cycles 50000000
"""

import argparse
import os
import sys
import time
import random
import multiprocessing
from pathlib import Path

import torch

try:
    from py65.devices.mpu6502 import MPU
except ImportError:
    print("ERROR: py65 not installed. Run: pip install py65")
    sys.exit(1)


def set_seed(seed: int, worker_id: int):
    """Set random seed for reproducibility."""
    random.seed(seed + worker_id)


def generate_shard(args):
    """Generate CPU traces for one worker."""
    worker_id, cycles_per_worker, seed, output_dir = args
    
    set_seed(seed, worker_id)
    
    # Initialize 6502 CPU
    mpu = MPU()
    memory = bytearray(65536)
    mpu.memory = memory
    
    inputs = []
    targets = []
    
    def randomize():
        """Randomize CPU state and inject random code."""
        mpu.pc = random.randint(0x0200, 0xFFFF)
        mpu.a = random.randint(0, 255)
        mpu.x = random.randint(0, 255)
        mpu.y = random.randint(0, 255)
        mpu.sp = random.randint(0, 255)
        mpu.p = random.randint(0, 255)
        # Inject random bytes as "code"
        for i in range(32):
            addr = (mpu.pc + i) & 0xFFFF
            mpu.memory[addr] = random.randint(0, 255)
    
    randomize()
    
    for i in range(cycles_per_worker):
        # Randomize every 100 cycles for variety
        if i % 100 == 0:
            randomize()
        
        # Progress reporting
        if i % 500000 == 0 and i > 0:
            print(f"  Worker {worker_id}: {i:,}/{cycles_per_worker:,} cycles")
        
        # Capture input state (State T)
        opcode = mpu.memory[mpu.pc]
        operand = mpu.memory[(mpu.pc + 1) & 0xFFFF]
        
        state_t = [
            mpu.a, mpu.x, mpu.y, mpu.sp, mpu.p,
            (mpu.pc >> 8) & 0xFF, mpu.pc & 0xFF,
            opcode, operand
        ]
        
        # Execute one instruction
        try:
            mpu.step()
        except:
            randomize()
            continue
        
        # Capture target state (State T+1)
        state_t1 = [
            mpu.a, mpu.x, mpu.y, mpu.sp, mpu.p,
            (mpu.pc >> 8) & 0xFF, mpu.pc & 0xFF
        ]
        
        inputs.append(state_t)
        targets.append(state_t1)
    
    # Save shard
    in_tensor = torch.tensor(inputs, dtype=torch.uint8)
    tgt_tensor = torch.tensor(targets, dtype=torch.uint8)
    
    shard_file = output_dir / f"shard_{worker_id:03d}.pt"
    torch.save({"input": in_tensor, "target": tgt_tensor}, shard_file)
    
    print(f"  Worker {worker_id}: Done - {len(inputs):,} cycles -> {shard_file.name}")
    return len(inputs)


def stitch_shards(output_dir: Path, output_file: Path):
    """Combine all shards into final dataset."""
    print("\nStitching shards...")
    
    shard_files = sorted(output_dir.glob("shard_*.pt"))
    
    all_inputs = []
    all_targets = []
    
    for f in shard_files:
        data = torch.load(f)
        all_inputs.append(data["input"])
        all_targets.append(data["target"])
    
    final_input = torch.cat(all_inputs)
    final_target = torch.cat(all_targets)
    
    torch.save({"input": final_input, "target": final_target}, output_file)
    
    return final_input.shape[0]


def main():
    parser = argparse.ArgumentParser(description="Generate 6502 CPU traces")
    parser.add_argument('--output', type=str, default='data/cpu_traces.pt', help='Output file')
    parser.add_argument('--cycles', type=int, default=50_000_000, help='Total cycles to generate')
    parser.add_argument('--workers', type=int, default=None, help='Number of workers (default: CPU count)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--keep-shards', action='store_true', help='Keep intermediate shard files')
    args = parser.parse_args()
    
    # Setup
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    shard_dir = output_file.parent / "shards"
    shard_dir.mkdir(exist_ok=True)
    
    num_workers = args.workers or os.cpu_count() or 8
    cycles_per_worker = args.cycles // num_workers
    
    print("=" * 60)
    print("CPU Trace Generator")
    print("=" * 60)
    print(f"  Target cycles: {args.cycles:,}")
    print(f"  Workers: {num_workers}")
    print(f"  Cycles per worker: {cycles_per_worker:,}")
    print(f"  Random seed: {args.seed}")
    print(f"  Output: {output_file}")
    print("=" * 60)
    
    # Generate in parallel
    t0 = time.time()
    
    worker_args = [
        (i, cycles_per_worker, args.seed, shard_dir)
        for i in range(num_workers)
    ]
    
    with multiprocessing.Pool(num_workers) as pool:
        results = pool.map(generate_shard, worker_args)
    
    total_cycles = sum(results)
    gen_time = time.time() - t0
    
    # Stitch shards
    final_count = stitch_shards(shard_dir, output_file)
    
    # Cleanup shards
    if not args.keep_shards:
        for f in shard_dir.glob("shard_*.pt"):
            f.unlink()
        shard_dir.rmdir()
    
    # Summary
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    
    print("\n" + "=" * 60)
    print("Generation Complete")
    print("=" * 60)
    print(f"  Total cycles: {final_count:,}")
    print(f"  Time: {gen_time:.1f}s ({final_count / gen_time / 1e6:.2f}M cycles/sec)")
    print(f"  File: {output_file}")
    print(f"  Size: {file_size_mb:.1f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
