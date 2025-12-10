# BBDOS Architecture

This document explains the technical architecture of BBDOS.

## Overview

BBDOS consists of three main components:

```
┌─────────────────────────────────────────────────────────────────┐
│                         BBDOS Stack                             │
├─────────────────────────────────────────────────────────────────┤
│  Applications                                                   │
│  ├── Neural 6502 CPU Emulator                                   │
│  └── NanoLPU Language Model                                     │
├─────────────────────────────────────────────────────────────────┤
│  Core Library                                                   │
│  └── TriX Sparse Layers (PyTorch)                          │
├─────────────────────────────────────────────────────────────────┤
│  Kernel                                                         │
│  └── 2-bit NEON Matrix Multiplication (C++)                     │
└─────────────────────────────────────────────────────────────────┘
```

## 1. TriX Kernel

### Weight Representation

Standard neural networks use 32-bit floating point weights. TriX uses **2-bit ternary weights**:

| Value | 2-bit Code | Meaning |
|-------|------------|---------|
| +1    | 0b01       | Positive connection |
| -1    | 0b10       | Negative connection |
| 0     | 0b00       | No connection (sparse) |

Four weights pack into a single byte:
```
Byte: [w3:2bits][w2:2bits][w1:2bits][w0:2bits]
```

This gives **4x memory compression** compared to int8 and **16x** compared to float32.

### Tile-Based Routing

Instead of computing the full matrix multiplication, TriX divides outputs into **tiles**:

```
Output Vector (2048 dims)
├── Tile 0 [0:512]      ← Active (compute)
├── Tile 1 [512:1024]   ← Inactive (skip)
├── Tile 2 [1024:1536]  ← Active (compute)
└── Tile 3 [1536:2048]  ← Inactive (skip)
```

A learned **gating network** decides which tiles to activate per input. Inactive tiles produce zero output without computation.

**Speedup:** If 75% of tiles are inactive, we achieve ~4x speedup.

### ARM NEON Implementation

The kernel uses ARM NEON SIMD intrinsics for the inner loop:

```cpp
// Pseudocode for NEON accumulation
for each 16-element chunk:
    weights = vld1q_s8(packed_ptr)      // Load 16 packed weights
    input = vld1q_f32(input_ptr)        // Load 4 floats
    unpack weights to ternary (-1, 0, +1)
    accumulate: sum += input * weight
```

Key optimizations:
- Packed weight loads (4 weights per byte)
- NEON 128-bit vector operations
- Tile-skip early exit for inactive tiles
- Cache-friendly memory layout

## 2. Neural 6502 CPU

### Architecture

The Neural CPU predicts state transitions:

```
Input State (T)              Output State (T+1)
┌──────────────┐             ┌──────────────┐
│ A  (8-bit)   │             │ A' (8-bit)   │
│ X  (8-bit)   │             │ X' (8-bit)   │
│ Y  (8-bit)   │   ──────►   │ Y' (8-bit)   │
│ SP (8-bit)   │   Neural    │ SP'(8-bit)   │
│ P  (8-bit)   │   Network   │ P' (8-bit)   │
│ PC (16-bit)  │             │ PC'(16-bit)  │
│ Opcode       │             │              │
│ Operand      │             │              │
└──────────────┘             └──────────────┘
```

### Model Structure

```
Input Embeddings (9 × 32-dim each)
        │
        ▼
   Bus Projection (352 → 256)
        │
        ▼
┌───────────────────────────────┐
│  TriX Functional Unit    │ × 6 layers
│  ├── Gate Network (256 → 4)   │
│  ├── Up Project (256 → 512)   │
│  └── Down Project (512 → 256) │
└───────────────────────────────┘
        │
        ▼
   Output Heads (256 → 256 each)
   ├── head_A, head_X, head_Y
   ├── head_SP, head_P
   └── head_PCH, head_PCL
```

### Emergent Tile Specialization

During training, tiles learn to specialize:

| Tile | Learned Specialization |
|------|------------------------|
| 0    | ALU operations (shifts, logic) |
| 1    | Memory operations (load/store) |
| 2    | Branch operations (jumps, conditionals) |
| 3    | System operations (stack, flags) |

This emerges naturally from the gating mechanism without explicit supervision.

### Training Data

Training data consists of CPU execution traces:

```python
trace = {
    'input': [A, X, Y, SP, P, PCH, PCL, Op, Val],  # State at T
    'target': [A', X', Y', SP', P', PCH', PCL']    # State at T+1
}
```

We generate 50M traces by running random code on a reference 6502 emulator (py65).

### Loss Function

Multi-head cross-entropy loss:

```python
loss = Σ CrossEntropy(pred[reg], target[reg])
       for reg in [A, X, Y, SP, P, PCH, PCL]
```

Each register head predicts a distribution over 256 possible values.

## 3. NanoLPU Language Model

### Architecture

```
Token Embedding + Position Embedding
        │
        ▼
┌───────────────────────────────────┐
│  Transformer Block                │ × 12 layers
│  ├── Multi-Head Attention (8 heads)
│  └── TriX FFN                │
│      ├── Gate (512 → 4)           │
│      ├── Up (512 → 2048)          │
│      └── Down (2048 → 512)        │
└───────────────────────────────────┘
        │
        ▼
   Output Projection (512 → vocab)
```

### Load Balancing

To prevent **mode collapse** (all inputs routing to one tile), we add:

1. **Noise injection** during training:
   ```python
   gate_logits += torch.randn_like(gate_logits) * noise_scale
   ```

2. **Load balancing loss**:
   ```python
   usage = gates.mean(dim=0)  # Average per-tile usage
   target = 1/num_tiles       # Uniform target
   balance_loss = MSE(usage, target)
   ```

### Tokenization

Character-level tokenization with 73 characters:
```
a-z, A-Z, 0-9, space, punctuation, newline
```

## Data Flow

### Training Pipeline

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Raw Data    │────►│ Tokenize/   │────►│ DataLoader  │
│ (text/CPU)  │     │ Encode      │     │ (batched)   │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Checkpoint  │◄────│ Optimizer   │◄────│ Model       │
│ (.pt file)  │     │ (AdamW)     │     │ (forward)   │
└─────────────┘     └─────────────┘     └─────────────┘
```

### Inference Pipeline

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Input       │────►│ Pack Weights│────►│ NEON Kernel │
│ (PyTorch)   │     │ (2-bit)     │     │ (C++)       │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                                               ▼
                                        ┌─────────────┐
                                        │ Output      │
                                        │ (PyTorch)   │
                                        └─────────────┘
```

## Performance Analysis

### Memory Savings

| Component | Float32 | TriX | Savings |
|-----------|---------|-----------|---------|
| Weight storage | 4 bytes | 0.25 bytes | 16x |
| FFN layer (512→2048) | 4 MB | 256 KB | 16x |
| Full model (38M params) | 152 MB | ~10 MB | 15x |

### Compute Savings

| Sparsity | Active Tiles | Theoretical Speedup | Measured |
|----------|--------------|---------------------|----------|
| 0% | 4/4 | 1.0x | 1.0x |
| 25% | 3/4 | 1.33x | 1.34x |
| 50% | 2/4 | 2.0x | 2.04x |
| 75% | 1/4 | 4.0x | 4.31x |

The slight super-linear speedup at high sparsity is due to improved cache locality.

## File Structure

```
bbdos/
├── kernel/
│   ├── trix.h      # C API declarations
│   ├── trix.cpp    # NEON implementation
│   ├── bindings.py      # Python ctypes wrapper
│   └── CMakeLists.txt   # Build configuration
├── cpu/
│   ├── model.py         # NeuralCPU architecture
│   └── __init__.py
└── lm/
    ├── model.py         # NanoLPU architecture  
    └── __init__.py
```

## References

- [BitNet: Scaling 1-bit Transformers](https://arxiv.org/abs/2310.11453)
- [Mixture of Experts](https://arxiv.org/abs/1701.06538)
- [MOS 6502 Reference](http://www.6502.org/tutorials/6502opcodes.html)
