# Reproducing BBDOS Results

This guide walks through reproducing all results from the paper.

## Prerequisites

### Hardware
- **Tested:** NVIDIA Jetson AGX Thor (ARM64, 64GB unified memory)
- **Minimum:** ARM64 with NEON support, 16GB RAM, CUDA GPU

### Software
- Ubuntu 22.04+
- Python 3.10+
- CUDA 11.8+ (for GPU training)
- CMake 3.20+

## Quick Start (Using Pre-trained Weights)

If you just want to verify claims using our pre-trained models:

```bash
# 1. Clone and setup
cd bbdos_v2
pip install -r requirements.txt

# 2. Build kernel
cd bbdos/kernel && mkdir build && cd build && cmake .. && make && cd ../../..

# 3. Run benchmark (verifies 4x speedup claim)
python scripts/benchmark.py

# 4. Run kernel tests
python -m pytest tests/test_kernel.py -v
```

## Full Reproduction (From Scratch)

### Step 1: Build the TriX Kernel

```bash
cd bbdos/kernel
mkdir build && cd build
cmake ..
make -j$(nproc)
```

Expected output:
```
-- Configuring ARM64 with NEON...
-- Build type: Release
-- Configuring done
-- Generating done
-- Build files have been written to: .../build
[100%] Built target trix
```

### Step 2: Verify Kernel Correctness

```bash
cd ../..
python -m pytest tests/test_kernel.py -v
```

Expected: All tests pass.

### Step 3: Run Speed Benchmark

```bash
python scripts/benchmark.py --output results/speedup.csv
```

Expected output:
```
============================================================
TriX Kernel Benchmark Results
============================================================
Sparsity      Tiles    Time (ms)     Speedup   
------------------------------------------------------------
     0%       4           140.70        1.00x
    25%       3           105.41        1.33x
    50%       2            70.28        2.00x
    75%       1            35.15        4.00x
============================================================

✓ KEY CLAIM VERIFIED: 4.00x speedup at 75% sparsity
```

### Step 4: Generate CPU Traces (Optional)

If you want to regenerate training data:

```bash
python scripts/generate_traces.py --output data/traces.pt --cycles 50000000
```

This takes ~20 minutes and produces ~750MB of data.

### Step 5: Train Neural 6502

```bash
python scripts/train_cpu.py --config configs/neural_cpu.yaml
```

Expected training curve:
| Epoch | Val Accuracy |
|-------|--------------|
| 1     | ~59%         |
| 5     | ~65%         |
| 10    | ~66%         |

Final model saved to `checkpoints/cpu/best.pt`.

### Step 6: Evaluate Neural 6502

```bash
python scripts/evaluate_cpu.py --checkpoint checkpoints/cpu/best.pt
```

Expected output:
```
Opcode Accuracy: 84.4%
Per-register: SP=99.9%, X=98.4%, Y=98.4%, PCH=97.3%, PCL=96.1%, A=83.5%, P=81.5%
```

## Using Docker

For guaranteed reproducibility:

```bash
# Build image
docker build -t bbdos:latest .

# Run tests
docker run --gpus all bbdos:latest

# Run benchmark
docker run --gpus all bbdos:latest python scripts/benchmark.py

# Train model
docker run --gpus all -v $(pwd)/data:/app/data bbdos:latest \
    python scripts/train_cpu.py --config configs/neural_cpu.yaml
```

## Expected Results Summary

| Claim | Expected | How to Verify |
|-------|----------|---------------|
| 4x speedup @ 75% sparsity | ≥3.5x | `python scripts/benchmark.py` |
| Kernel numerical accuracy | <1e-4 | `pytest tests/test_kernel.py` |
| Neural 6502 opcode accuracy | ~84% | `python scripts/evaluate_cpu.py` |
| Neural 6502 SP accuracy | ~99% | `python scripts/evaluate_cpu.py` |

## Troubleshooting

### Kernel build fails
- Ensure CMake 3.20+ is installed
- On ARM64, verify NEON support: `cat /proc/cpuinfo | grep neon`

### Speedup lower than expected
- Disable CPU frequency scaling: `sudo cpufreq-set -g performance`
- Close other applications
- Run multiple times and average

### Training accuracy lower than expected
- Verify random seed is 42 (in config)
- Check CUDA version compatibility
- Ensure data wasn't corrupted during generation

## Citation

If you reproduce these results, please cite:

```bibtex
@software{bbdos2024,
  author = {Tripp and Double-D and Team},
  title = {BBDOS: TriX-Based Distributed Operating System},
  year = {2024},
  url = {https://github.com/[repo]}
}
```
