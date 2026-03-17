# BBDOS Two-Be Repository Audit

**Date:** Tue Mar 17 2026, 09:55 UTC  
**Auditor:** opencode  
**Scope:** Full repository analysis

---

## Executive Summary

BBDOS is a 2-bit conditional ternary neural architecture with learned computational sparsity. The core insight is physically skipping computation through dynamic tile-based gating rather than dense-then-mask approaches. The repository contains working implementations of the kernel, two model architectures (Neural 6502 and NanoLPU), and training/evaluation scripts.

**Verdict:** Solid research implementation with working code. Novelty is real, useful parts are actually useful, fluff is mostly marketing.

---

## What's Novel

### 1. Tile-Based Learned Sparsity

The architecture learns which output tiles to activate per input through a gating network, then physically skips inactive tiles. This is distinct from:

- **Masked sparsity**: Dense compute then zero activations (wastes FLOPs)
- **Static pruning**: Fixed sparsity pattern (no input-dependent routing)

The gating mechanism enables computation proportional to semantic entropy.

### 2. 2-Bit Ternary with "Dark State"

| Code | Value | Meaning |
|------|-------|---------|
| 0b00 | 0 | No connection (sparse) |
| 0b01 | +1 | Positive connection |
| 0b10 | -1 | Negative connection |
| 0b11 | - | "Dark State" (reserved) |

The reservation of `0b11` for future extensions is a deliberate backward-compatibility play. If future work needs to add scaling, bias, or new weight states, the encoding space supports it without breaking existing weights.

### 3. Neural 6502 Emulator

Training a neural network to predict 6502 CPU state transitions from execution traces is genuinely novel. The model:

- Takes current state (A, X, Y, SP, P, PC, opcode, operand)
- Predicts next state for each register
- Uses 50M generated traces from py65 emulator

The "Savant CPU" finding - good at control flow/bitwise ops, terrible at multi-register arithmetic (ADC at 3.1%) - provides insight into neural program synthesis boundaries.

### 4. Super-Linear Speedup at High Sparsity

Measured 4.31x speedup at 75% sparsity exceeds theoretical 4.0x. This emergent property results from improved cache locality when fewer tiles are computed.

---

## What's Useful

### 1. NEON Kernel (`bbdos/kernel/bitswitch.cpp`)

Clean implementation that delivers measurable speedup:

- Packed weight loading (4 weights per byte)
- NEON 128-bit vector operations for ARM64
- Tile-skip early exit for inactive tiles
- Cache-friendly memory layout
- Scalar fallback for non-ARM platforms

### 2. Python Bindings (`bbdos/kernel/bindings.py`)

Well-designed wrapper bridging PyTorch with C++ kernel via ctypes. Allows seamless integration into training pipeline.

### 3. BitSwitchLinear Module

The PyTorch module that wraps the kernel handles:
- Weight packing (float32 → 2-bit)
- Gate mask application
- Scale factor management

### 4. Training Scripts

| Script | Purpose |
|--------|---------|
| `scripts/train_cpu.py` | Train Neural 6502 |
| `scripts/train_lm.py` | Train NanoLPU |
| `scripts/evaluate_cpu.py` | Evaluate with pretrained weights |
| `scripts/benchmark.py` | Verify speedup claims |
| `scripts/generate_traces.py` | Generate CPU execution traces |

### 5. Reproducible Configs (`configs/`)

YAML configs with seeded random states for reproducibility.

### 6. Docker Environment

One-command reproducibility: `docker build -t bbdos . && docker run bbdos`

---

## What's Understated

### 1. Load Balancing Mechanism

**Critical for training success.** Without it, all inputs route to one tile (mode collapse).

Two mechanisms work together:

**Noise injection** (during training):
```python
gate_logits += torch.randn_like(gate_logits) * noise_scale
```

**Load balancing loss**:
```python
usage = gates.mean(dim=0)      # Per-tile average usage
target = 1/num_tiles           # Uniform target
balance_loss = MSE(usage, target)
```

This is mentioned in ARCHITECTURE.md but underemphasized.

### 2. Straight-Through Estimator (STE)

The `Top1Gate` function in both models uses hard gating with straight-through gradient:

```python
class Top1Gate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits):
        idx = torch.argmax(logits, dim=-1)
        return torch.zeros_like(logits).scatter_(-1, idx, 1.0)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # Pass-through gradient
```

This is what makes end-to-end training work - the gating network learns via gradient signal that passes through the hard decision.

### 3. Scale Factors

The kernel computes `dot * scale` per output channel (`bitswitch.cpp:124`):

```cpp
out_row[out_start + o] = dot * tile_scales[o];
```

These scale factors compensate for quantization loss. Without them, ternary weights would have no amplitude control, destroying accuracy.

### 4. Input Embedding Strategy (Neural 6502)

The model treats each register/operand as a learned embedding and combines via position embeddings:

```python
# 9 inputs: A, X, Y, SP, P, PCH, PCL, Op, Val
embeddings = [self.register_emb(state[key]) for key in registers]
embeddings.append(self.opcode_emb(state['Op']))
embeddings.append(self.register_emb(state['Val']))
x = torch.stack(embeddings, dim=1) + position_emb
```

Clean design that avoids manual feature engineering.

### 5. Multi-Head Loss

Each register has its own output head with independent cross-entropy loss. This allows the model to specialize and fail independently - revealing which operations are learnable vs not.

---

## What's Fluff

### 1. "Savant CPU" Phenomenon

Overhyped in the README. The language implies a surprising discovery, but:

- Neural networks struggle with precise arithmetic universally
- ADC (add with carry) requires coordination across A, P registers and memory
- This isn't unique to this architecture - it's a general limitation

Interesting finding, but presented as more profound than it is.

### 2. WIP Paper PDF

Included at `docs/paper/bbdos_paper_WIP.pdf` but it's work in progress. The code is the actual contribution.

### 3. "Dark State" Reserve

A clever idea but purely speculative - there's no implementation of what `0b11` does. Good for future-proofing narrative, not current functionality.

### 4. README Badges

```
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]
[![Platform: ARM64](https://img.shields.io/badge/Platform-ARM64-blue.svg)]
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)]
```

Trivial. Focus on the metrics instead.

### 5. bbdos_topo File

A 512-byte file in root directory that appears completely unused. No references in code or docs.

### 6. Acknowledgments Section

The Double-D "AI collaborator" credit is unusual and unexplained.

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                         BBDOS Stack                             │
├─────────────────────────────────────────────────────────────────┤
│  Applications                                                   │
│  ├── Neural 6502 CPU Emulator                                   │
│  └── NanoLPU Language Model                                     │
├─────────────────────────────────────────────────────────────────┤
│  Core Library                                                   │
│  └── BitSwitch Sparse Layers (PyTorch)                          │
├─────────────────────────────────────────────────────────────────┤
│  Kernel                                                         │
│  └── 2-bit NEON Matrix Multiplication (C++)                     │
└─────────────────────────────────────────────────────────────────┘
```

### BitSwitch Kernel Flow

```
Input → [Tile Gating Network] → Active Tile Selection
                              ↓
              ┌───────────────────────────────┐
              │ Tile 0 (active)   Tile 1 (skip)│
              │ Tile 2 (active)  Tile 3 (skip)│
              └───────────────────────────────┘
                              ↓
                         Output (sparse)
```

### Performance Metrics

| Component | Metric | Value |
|-----------|--------|-------|
| Memory compression | vs FP32 | 16x |
| Speedup | @ 75% sparsity | 4.00x (measured: 4.31x) |
| Numerical accuracy | max error vs PyTorch | 0.000069 |
| Neural 6502 | Opcode accuracy | 84.4% |
| BBDOS LM | Final loss (TinyStories) | 0.43 |

---

## File Structure

```
two-be/
├── bbdos/
│   ├── kernel/
│   │   ├── bitswitch.h      # C API declarations
│   │   ├── bitswitch.cpp    # NEON implementation
│   │   ├── bindings.py      # Python ctypes wrapper
│   │   ├── CMakeLists.txt   # Build configuration
│   │   └── __init__.py
│   ├── cpu/
│   │   ├── model.py         # NeuralCPU architecture
│   │   └── __init__.py
│   └── lm/
│       ├── model.py         # NanoLPU architecture  
│       └── __init__.py
├── configs/
│   ├── neural_cpu.yaml      # Neural 6502 config
│   └── bbdos_lm.yaml        # Language model config
├── scripts/
│   ├── train_cpu.py
│   ├── train_lm.py
│   ├── evaluate_cpu.py
│   ├── benchmark.py
│   ├── generate_traces.py
│   └── demo.py
├── tests/                   # 22 pytest tests
├── docs/
│   ├── ARCHITECTURE.md
│   ├── REPRODUCING.md
│   └── paper/
│       └── bbdos_paper_WIP.pdf
├── weights/
│   └── neural_cpu_best.pt   # Pretrained (9.3 MB)
├── Dockerfile
├── README.md
└── LICENSE
```

---

## Key Implementation Details

### Weight Packing (`bitswitch.cpp:130-157`)

```cpp
// 4 ternary weights pack into 1 byte
// w3 w2 w1 w0 → [2bits][2bits][2bits][2bits]
packed |= (code << (i * 2));
```

### Tile Routing (`bitswitch.cpp:112-127`)

```cpp
// Skip inactive tiles entirely
if (gate_mask[b * num_tiles + t] == 0) {
    continue;  // Physical skip - no computation
}
```

### Hard Gating (Python, both models)

```python
gate = Top1Gate.apply(gate_logits)  # One-hot output
h = self.up_proj(x, gate)            # Only compute active tile
```

---

## Recommendations

1. **Emphasize load balancing** - Document prominently; it's critical to training
2. **Remove bbdos_topo** - Or document its purpose
3. **Calibrate "Savant CPU" language** - It's an interesting finding, not a breakthrough
4. **Add Dark State implementation** - If reserving space, implement at least a no-op behavior
5. **Clean up acknowledgments** - Remove or explain "AI collaborator"

---

## Conclusion

BBDOS is a solid research implementation with working code that delivers on its core claims. The novelty (tile gating + 2-bit ternary) is real, the useful parts (kernel, bindings, training scripts) are actually useful, and the understated parts (load balancing, STE, scale factors) are what make it work. The fluff is mostly marketing language that could be toned down.

**Rating:** Useful research code with clear novel contribution. Recommended for anyone interested in efficient inference, learned sparsity, or neural program synthesis.
