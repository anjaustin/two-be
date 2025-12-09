# BBDOS: BitSwitch-Based Distributed Operating System

**Sparse 2-bit neural computation for efficient inference**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform: ARM64](https://img.shields.io/badge/Platform-ARM64-blue.svg)]()
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)]()

---

## What is BBDOS?

BBDOS demonstrates that **sparse 2-bit neural networks can learn meaningful computation**. We implement:

1. **BitSwitch Kernel**: A sparse matrix multiplication kernel using 2-bit (ternary) weights with tile-based routing. Achieves **4x speedup** at 75% sparsity on ARM NEON.

2. **Neural 6502**: A neural network that learns to emulate the MOS 6502 CPU by predicting register state transitions. Achieves **84.4% opcode accuracy** across 3,136 tests.

3. **BBDOS Language Model**: A 38M parameter transformer with BitSwitch layers, trained on TinyStories to **0.43 cross-entropy loss**.

## Key Results

| Component | Metric | Value |
|-----------|--------|-------|
| BitSwitch Kernel | Speedup @ 75% sparsity | **4.00x** |
| BitSwitch Kernel | Numerical accuracy | **0.000069** max diff |
| Neural 6502 | Opcode accuracy | **84.4%** |
| Neural 6502 | SP register accuracy | **99.9%** |
| BBDOS LM | Final loss | **0.4347** |

## Quick Start

```bash
# Clone and enter directory
cd bbdos_v2

# Install dependencies
pip install -r requirements.txt

# Build the kernel
cd bbdos/kernel && mkdir build && cd build && cmake .. && make && cd ../../..

# Run kernel tests
python -m pytest tests/test_kernel.py -v

# Run a quick CPU inference
python scripts/evaluate_cpu.py --checkpoint ../neural_cpu_best.pt

# Generate text with BBDOS LM
python scripts/generate.py --checkpoint ../bbdos_research_final.pt --prompt "Once upon a time"
```

## Hardware Requirements

**Tested on:**
- NVIDIA Jetson AGX Thor (ARM64, Blackwell GPU)
- 64GB unified memory
- Ubuntu 22.04

**Minimum requirements:**
- ARM64 processor with NEON support (for kernel acceleration)
- 16GB RAM
- CUDA 11.8+ (for GPU training)

## Repository Structure

```
bbdos_v2/
├── bbdos/                  # Main package
│   ├── kernel/             # BitSwitch ARM NEON kernel
│   ├── cpu/                # Neural 6502 model
│   └── lm/                 # Language model
├── configs/                # Hyperparameter configs
├── scripts/                # Training and evaluation
├── tests/                  # Test suite
└── docs/                   # Documentation
```

## Reproducing Results

See [docs/REPRODUCING.md](docs/REPRODUCING.md) for complete reproduction instructions.

**Quick reproduction:**
```bash
# 1. Generate CPU traces (requires py65)
python scripts/generate_traces.py --output data/traces.pt --cycles 50000000

# 2. Train Neural 6502 (10 epochs, ~2 hours on Thor)
python scripts/train_cpu.py --config configs/neural_cpu.yaml

# 3. Evaluate
python scripts/evaluate_cpu.py --checkpoint checkpoints/best.pt
```

## Citation

```bibtex
@software{bbdos2024,
  author = {Tripp and Double-D and Team},
  title = {BBDOS: BitSwitch-Based Distributed Operating System},
  year = {2024},
  url = {https://github.com/[repo]}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Built on Jetson AGX Thor (NVIDIA)
- TinyStories dataset (Microsoft Research)
- py65 6502 emulator

---

*"Legit 2-Bit"* - BBDOS v1.0.0
