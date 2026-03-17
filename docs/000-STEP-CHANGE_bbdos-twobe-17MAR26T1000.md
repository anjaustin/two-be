# BBDOS Step-Change: From Tile Gating to Operation Routing

**Date:** Tue Mar 17 2026, 10:00 UTC  
**Subject:** Throughline and Next Step-Change

---

## The Throughline

**"Learn what to compute, then compute it cheaply."**

BBDOS makes two hard choices at inference time:

### 1. Routing (Gating Network)

Which output tiles activate per input. The network learns to route computation proportional to semantic entropy. Inactive tiles are physically skipped - no FLOPs wasted.

### 2. Weight Encoding (2-Bit Ternary)

Only three values: `-1`, `0`, `+1`. Packed 4-to-1 in memory. No floating point math needed.

---

## What Makes It Work

Three understated components enable the above:

| Component | Role |
|-----------|------|
| **Straight-Through Estimator** | Hard gating (`argmax`) with pass-through gradient makes routing trainable end-to-end |
| **Load Balancing Loss** | Noise injection + usage regularization prevents all inputs from collapsing to one tile |
| **Scale Factors** | Per-channel scaling recovers amplitude lost to ternary quantization |

---

## Current Architecture

```
Input
   │
   ▼
[Gating Network] → Binary tile mask (which tiles to compute)
   │
   ▼
┌─────────────────────────────┐
│ Tile 0  │ Tile 1 │ Tile 2  │  Each tile: same operation (matmul)
│ active  │ skip    │ active  │             different weights
└─────────────────────────────┘
   │
   ▼
Output
```

**Limit:** Each tile does the same operation (matrix multiplication with ternary weights). The only variation is *which* tiles run.

---

## The Ceiling

The Neural 6502 reveals the boundary:

| Operation Type | Accuracy |
|----------------|----------|
| Control flow (branches, jumps) | 96-99% |
| Bitwise logic (shifts, AND/OR) | 96-97% |
| Stack/pointer operations | 99.9% |
| Multi-register arithmetic (ADC) | **3.1%** |

Neural networks route and compute well. They fail on **multi-register coordination** - the arithmetic that requires precise carry propagation across registers.

This isn't a flaw in BBDOS. It's a flaw in pure neural computation. Some operations are symbolic by nature.

---

## The Step-Change

### From Tile Gating → Operation Routing

**Current encoding:**

| Code | Value |
|------|-------|
| 0b00 | 0 (skip) |
| 0b01 | +1 |
| 0b10 | -1 |
| 0b11 | **reserved** ("Dark State") |

**Proposed encoding:**

| Code | Operation | Description |
|------|-----------|-------------|
| 0b00 | skip | No computation (current) |
| 0b01 | +1 | Positive weight (current) |
| 0b10 | -1 | Negative weight (current) |
| 0b11 | **op_select** | Dynamic operation per-tile |

The Dark State becomes an **operation selector**. Each tile can now learn which operation to apply:

- `pass_through` - Identity, no modification
- `negate` - Multiply by -1
- `abs` - Absolute value
- `threshold` - Clip to range
- `lookup` - Index into small table

### Hybrid Execution Model

```
Input
   │
   ▼
[Gating Network] → Tile mask + Operation mask
   │
   ▼
┌─────────────────────────────────────────────┐
│ Tile 0: matmul (+1/-1 weights)               │
│ Tile 1: SKIP                                 │
│ Tile 2: matmul + negate + pass_through       │  ← Multiple ops per tile
│ Tile 3: SKIP                                 │
└─────────────────────────────────────────────┘
   │
   ▼
Output
```

### The Bridge

The Dark State is the bridge:

```
Current: Tile Gating (which tiles)
         │
         ▼
Next:    Operation Routing (which operations per tile)
         │
         ▼
Future:  Hybrid Neural-Symbolic (neural handles what it's good at,
         symbolic handles what it isn't)
```

---

## Why This Matters

1. **ADC is the test case**: The Neural 6502 fails at ADC. The next architecture should route ADC to a symbolic carry lookahead instead of trying to neural it.

2. **Scaling**: If tile gating gives 4x speedup at 75% sparsity, operation routing gives more - some operations are cheaper than matmul.

3. **The pattern is established**: BBDOS already proved "learn what to compute, then compute it cheaply." Operation routing extends the same principle from *which tiles* to *which operations*.

---

## Implementation Sketch

```python
# Extended operation encoding
OPS = {
    0b00: lambda x: 0,                      # skip
    0b01: lambda x: x * weight,             # multiply
    0b10: lambda x: -x * weight,            # negate + multiply
    0b11: lambda x: op_table[opcode](x),    # lookup operation
}

# In kernel: dynamic dispatch per code
for each weight_code:
    if code == 0b11:
        result = op_dispatch(opcode, input)
    else:
        result = multiply(input, code)
```

---

## Summary

**Throughline:** Learn what to compute, then compute it cheaply.

**Current:** Tile gating + ternary weights = 4x speedup + 16x memory compression.

**Next step:** Dark State as operation selector → hybrid neural-symbolic execution.

**Why:** Neural networks are bad at precise multi-register arithmetic (ADC at 3.1%). The solution isn't a bigger network - it's routing to the right compute primitive.

**The step-change is the transition from "which tiles" to "which operations", and the Dark State is the seed of that transition.**
