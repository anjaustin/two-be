# BBDOS

**2-Bit Conditional Ternary Neural Architecture with Learned Computational Sparsity**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform: ARM64](https://img.shields.io/badge/Platform-ARM64-blue.svg)]()
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)]()

> *Paper: [PDF](docs/paper/bbdos_paper_WIP.pdf) | [LaTeX source](docs/paper/bbdos_paper.tex) (compile with `pdflatex`)*

---

## Abstract

BBDOS (BitSwitch) is a 2-bit conditional ternary neural architecture that learns to allocate computation proportionally to semantic entropy. Unlike masked sparsity approaches that execute full dense operations and zero unwanted activations, BBDOS **physically skips inactive computation** through dynamic tile-based gating, achieving linear speedup that scales directly with sparsity level.

The architecture combines:
- **Ternary weights** {-1, 0, +1} encoded in 2 bits per weight
- **Learned tile activation** patterns for conditional computation
- **16x memory compression** versus FP32
- **4.00x inference speedup** at 75% sparsity

The 2-bit encoding reserves 25% of the bit space (the **"Possibility State"**) for future extensions, enabling backward-compatible architectural evolution.

## Key Results

| Experiment | Metric | Value |
|------------|--------|-------|
| BitSwitch Kernel | Memory compression vs FP32 | **16x** |
| BitSwitch Kernel | Speedup @ 75% sparsity | **4.00x** |
| BitSwitch Kernel | Numerical accuracy vs PyTorch | **0.000069** max error |
| Neural 6502 | Full-state accuracy | **66.4%** |
| Neural 6502 | Per-register average | **~91%** |
| Neural 6502 | Stack pointer accuracy | **99.9%** |
| Neural 6502 | Shift operations (ASL, LSR) | **96-97%** |
| Neural 6502 | Carry arithmetic (ADC) | **3.1%** |
| BBDOS LM (38.2M params) | Final loss on TinyStories | **0.43** |

### Neural 6502 Accuracy Metrics

We report multiple accuracy metrics for transparency (see [docs/METRICS.md](docs/METRICS.md) for details):

| Metric | Value | Description |
|--------|-------|-------------|
| Full-state accuracy | **66.4%** | All 7 registers predicted correctly simultaneously |
| Per-register average | **~91%** | Mean accuracy across individual registers |
| Opcode-weighted | **84.4%** | Weighted average across 3,136 opcode tests |

**Per-Register Breakdown:**
| Register | Accuracy | Notes |
|----------|----------|-------|
| SP (Stack Pointer) | 99.9% | Near-perfect stack operations |
| X (Index) | 98.4% | Excellent |
| Y (Index) | 98.4% | Excellent |
| PCH (PC High) | 97.3% | Control flow mastered |
| PCL (PC Low) | 96.1% | Control flow mastered |
| A (Accumulator) | 83.5% | Arithmetic challenges |
| P (Flags) | 81.5% | Flag prediction harder |

### The "Savant CPU" Phenomenon

The Neural 6502 reveals a striking pattern: neural networks can master control flow and bitwise logic (96-99% accuracy) but fail catastrophically on multi-register arithmetic coordination (3.1% on ADC). This sharp boundary between learnable and unlearnable deterministic patterns provides new insights into neural program synthesis capabilities.

## Architecture

```
Input → [Tile Gating Network] → Active Tile Selection
                                      ↓
        ┌─────────────────────────────┴─────────────────────────────┐
        │                                                           │
    [Tile 0]    [Tile 1]    [Tile 2]    [Tile 3]                   │
    (active)    (SKIPPED)   (active)    (SKIPPED)    ← Physical skip
        │                       │                                   │
        └───────────────────────┴───────────────────────────────────┘
                                      ↓
                               Output (sparse)
```

### Memory Compression: Reality Check

The **16x compression** applies to weight storage specifically:

| Component | FP32 Baseline | BitSwitch | Compression |
|-----------|---------------|-----------|-------------|
| Weights | 32 bits/weight | 2 bits/weight | **16x** |
| Scales | - | 32 bits/output | overhead |
| Activations | 32 bits | 32 bits | 1x |
| Gate masks | - | 8 bits/tile | overhead |

**When 16x matters:** Large models, batch size 1, weight-dominated memory (e.g., LLM inference on edge devices).

**When it doesn't:** Small models, large batches, compute-bound workloads. Effective compression drops to ~7-8x.

**Research opportunities not explored here:**
- FP16/INT8 activations (potential additional 2-4x)
- Scale quantization (FP16 or per-tile sharing)
- Sparse activation storage

## Quick Start

### Docker (Recommended)
```bash
# Build and verify core claim in one command
docker build -t bbdos .
docker run bbdos

# Expected output:
# ✓ CORE CLAIM VERIFIED
#   4.xx speedup at 75% sparsity (target: 4.00x)

# Run full test suite
docker run bbdos python -m pytest tests/ -v

# Interactive shell
docker run -it bbdos bash
```

### From Source
```bash
# Install dependencies
pip install -r requirements.txt

# For development (editable install)
pip install -e .

# Build the kernel
cd bbdos/kernel && mkdir build && cd build && cmake .. && make && cd ../../..

# Run demo (verifies core claim)
python scripts/demo.py

# Run tests (22 tests)
python -m pytest tests/ -v

# Evaluate Neural 6502 with pretrained weights
python scripts/evaluate_cpu.py --checkpoint weights/neural_cpu_best.pt
```

## Pretrained Weights

| Model | File | Size | Metric | Status |
|-------|------|------|--------|--------|
| Neural 6502 | `weights/neural_cpu_best.pt` | 9.3 MB | 84.4% opcode accuracy | Included |
| BBDOS LM | `bbdos_research_final.pt` | 146 MB | 0.43 loss on TinyStories | Contact author |

*Note: LM weights exceed GitHub's 100MB limit. See [docs/WEIGHTS.md](docs/WEIGHTS.md) for options including Git LFS.*

## Repository Structure

```
two-be/
├── bbdos/                  # Main package
│   ├── kernel/             # BitSwitch NEON/CUDA kernel
│   ├── cpu/                # Neural 6502 model
│   └── lm/                 # NanoLPU language model
├── configs/                # YAML configs (seeded for reproducibility)
├── scripts/                # Training, evaluation, benchmarking
├── tests/                  # 22 pytest tests
├── docs/
│   ├── paper/              # Paper draft (WIP)
│   ├── ARCHITECTURE.md     # Technical deep-dive
│   └── REPRODUCING.md      # Step-by-step reproduction guide
└── Dockerfile              # Reproducible environment
```

## Hardware Requirements

**Tested on:**
- NVIDIA Jetson AGX Thor (ARM64)
- 64GB unified memory
- Ubuntu 22.04

**Minimum:**
- ARM64 with NEON support (or x86_64 with scalar fallback)
- 16GB RAM
- CUDA 11.8+ (for GPU training)

## Reproducing Results

See [docs/REPRODUCING.md](docs/REPRODUCING.md) for complete instructions.

```bash
# Generate 50M CPU traces
python scripts/generate_traces.py --cycles 50000000 --seed 42

# Train Neural 6502
python scripts/train_cpu.py --config configs/neural_cpu.yaml

# Verify speedup claim
python scripts/benchmark.py
# Expected: ✓ KEY CLAIM VERIFIED: 4.00x speedup at 75% sparsity
```

## Prior Work & What's Novel

**This builds on established research:**

| Technique | Prior Work | What We Use |
|-----------|------------|-------------|
| Ternary weights | TWN (Li et al., 2016), TTQ (Zhu et al., 2017) | 2-bit encoding with Possibility State |
| Conditional computation | Mixture of Experts (Shazeer et al., 2017) | Tile-based hard gating |
| Sparse inference | Structured pruning literature | Physical skip via gating |

**Novel contributions:**
- Combination of ternary weights + learned tile gating + physical skip in one architecture
- ARM NEON kernel implementation achieving linear speedup
- "Savant CPU" phenomenon documenting neural network limitations on deterministic arithmetic

**What is NOT claimed as novel:**
- Ternary quantization itself
- The concept of conditional computation
- Sparse neural network architectures

Please cite the foundational work above when building on these techniques.

## Citation

```bibtex
@article{josserandaustin2025bbdos,
  title={BBDOS: 2-Bit Conditional Ternary Neural Architecture with Learned Computational Sparsity},
  author={Josserand-Austin, Aaron (Tripp)},
  year={2025},
  month={December},
  note={Independent Research}
}
```

**Author:** Aaron (Tripp) Josserand-Austin — iam@anjaustin.com  
**Date:** Monday, 08 DEC 2025, 00:21 Hrs

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- NVIDIA Jetson AGX Thor platform
- TinyStories dataset (Microsoft Research)
- py65 6502 emulator
- Double-D (AI collaborator)

---

*"Legit 2-Bit"* — BBDOS v1.0.0
