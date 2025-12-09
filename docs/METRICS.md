# Neural 6502 Accuracy Metrics

This document explains the different accuracy metrics reported for the Neural 6502 model and why they differ.

## Overview

The Neural 6502 is a transformer-based model that predicts CPU register state transitions. Given an initial state and an opcode, it predicts the values of 7 registers after execution: A, X, Y, SP, P (flags), PCH, and PCL.

We report three different accuracy metrics:

| Metric | Value | Use Case |
|--------|-------|----------|
| Full-state accuracy | 66.4% | Strictest measure - practical correctness |
| Per-register average | ~91% | Understanding model capabilities |
| Opcode-weighted | 84.4% | Research comparison across opcodes |

## Metric Definitions

### 1. Full-State Accuracy (66.4%)

**Definition:** The percentage of test cases where ALL 7 registers are predicted correctly simultaneously.

```
correct = (pred_A == true_A) AND (pred_X == true_X) AND ... AND (pred_PCL == true_PCL)
accuracy = sum(correct) / num_tests
```

**Why this matters:** This is the only metric that reflects whether the model could actually replace a real CPU. If any register is wrong, the emulation fails.

**Evaluation command:**
```bash
python scripts/evaluate_cpu.py --checkpoint weights/neural_cpu_best.pt
```

### 2. Per-Register Average (~91%)

**Definition:** The accuracy of each register predicted independently, then averaged.

| Register | Accuracy |
|----------|----------|
| SP (Stack Pointer) | 99.9% |
| X (Index Register) | 98.4% |
| Y (Index Register) | 98.4% |
| PCH (Program Counter High) | 97.3% |
| PCL (Program Counter Low) | 96.1% |
| A (Accumulator) | 83.5% |
| P (Status Flags) | 81.5% |
| **Average** | **~91%** |

**Why this matters:** Shows that the model has learned most register behaviors well. The lower A and P accuracy reflects the difficulty of arithmetic operations and flag computation.

### 3. Opcode-Weighted Accuracy (84.4%)

**Definition:** Accuracy computed per opcode, then averaged across all tested opcodes. This was measured on 3,136 test cases (56 opcodes × 56 test states).

**Why this differs:** Some opcodes are easier than others. This metric weights each opcode equally regardless of how many registers it affects.

**Opcode categories:**
- Perfect (≥95%): NOP, CLC, SEC, PHP, PHA, LSR, ROL, branch instructions
- Good (70-95%): Transfer ops (TAX, TXA), increment/decrement, loads
- Broken (<10%): ADC, SBC (carry-based arithmetic)

## Why Full-State Accuracy is Lower

Consider predicting a single instruction. With 7 registers at ~91% individual accuracy:

```
P(all correct) ≈ 0.91^7 ≈ 0.52
```

This rough calculation shows why 66.4% full-state accuracy is consistent with ~91% per-register accuracy. Errors compound when requiring all predictions to be correct.

## The "Savant CPU" Phenomenon

The most striking finding is the sharp accuracy boundary:

| Operation Type | Accuracy | Notes |
|----------------|----------|-------|
| Stack operations (SP) | 99.9% | Near-perfect |
| Control flow (branches) | 96-99% | Excellent |
| Bitwise shifts (ASL, LSR) | 96-97% | Excellent |
| Register transfers | 75-87% | Good |
| Carry arithmetic (ADC, SBC) | 3-6% | Catastrophic failure |

The model masters operations that can be learned from input-output patterns but fails at multi-step arithmetic requiring carry propagation. This suggests a fundamental limitation in how neural networks learn deterministic algorithms.

## Reproducing These Metrics

```bash
# Full evaluation (outputs all metrics)
python scripts/evaluate_cpu.py --checkpoint weights/neural_cpu_best.pt

# Quick verification
python scripts/evaluate_cpu.py --checkpoint weights/neural_cpu_best.pt --quick
```

## Recommendations for Reporting

When citing Neural 6502 accuracy:

1. **For practical applications:** Use full-state accuracy (66.4%)
2. **For research comparison:** Use opcode-weighted (84.4%) with methodology note
3. **For capability analysis:** Use per-register breakdown

Always specify which metric is being used to avoid confusion.

---

*See also: [PAPER_DATA.md](../PAPER_DATA.md) for complete experimental results.*
