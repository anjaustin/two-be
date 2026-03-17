# Neural APU Design

> **Implementation Status:** Phase 1 Complete ✓

## Ternary Compute as First-Class Opcode

**Concept:** Transform BBDOS from a model into a **compute primitive** — callable like a CPU opcode, cached like a cache line, with ternary math as the instruction set.

---

## 1. The Core Idea

### Traditional Compute Stack

```
Application
     │
     ▼
Python/TensorFlow/PyTorch
     │
     ▼
CPU (int ops) + GPU (tensor ops)
     │
     ▼
Hardware (ALU, FPU, SIMD)
```

### BBDOS as Compute Primitive

```
Application
     │
     ▼
Python/TensorFlow/PyTorch
     │
     ▼
CPU (int ops) + GPU (tensor ops) + **Neural APU (ternary)**
     │
     ▼
Hardware (ALU, FPU, SIMD, Neural Tile Array)
```

The Neural APU is a new execution unit — not a replacement for GPU, but an **accelerator for specific patterns** that neural networks handle well.

---

## 2. APU Opcode Table

### Base Opcodes

| Opcode | Description | Input | Output |
|--------|-------------|-------|--------|
| `TMUL` | Ternary matrix multiply | [M×K], [K×N] | [M×N] |
| `TADD` | Ternary elementwise add | [M×N], [M×N] | [M×N] |
| `TGATE` | Ternary gating/routing | [B×C], [num_tiles] | [B×C] |
| `TATTN` | Ternary attention | [B×T×D], [B×T×D] | [B×T×D] |
| `TNORM` | Ternary layer norm | [B×T×D] | [B×T×D] |
| `TLOOKUP` | Ternary embedding lookup | [B×T], [vocab×D] | [B×T×D] |

### Opcode Modifiers

```assembly
TMUL.4      ; 4-bit output (vs default 32-bit)
TMUL.TILE=2 ; Use tile 2 only (sparse mode)
TMUL.SPARSE ; Auto-skip zero tiles
TGATE.NOISE ; Enable noise injection (training)
```

---

## 3. L-Cache Design

### Cache Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                    Neural APU L-Cache                            │
│  ┌─────────┬─────────┬─────────┬─────────┬─────────┐             │
│  │ Slot 0  │ Slot 1  │ Slot 2  │ Slot 3  │  ...    │  16 slots   │
│  └─────────┴─────────┴─────────┴─────────┴─────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

### Cache Line Structure (per slot)

```
┌─────────────────────────────────────────────────────────────────┐
│ Cache Line (256 bytes)                                          │
├─────────────────────────────────────────────────────────────────┤
│ Header (16 bytes)                                               │
│   - opcode_id: 8 bits                                            │
│   - config: 8 bits (num_tiles, sparsity_mode, etc.)            │
│   - status: 8 bits (HOT/COLD/EVICT/PREFETCH)                    │
│   - age: 32 bits                                                 │
├─────────────────────────────────────────────────────────────────┤
│ Weights (packed, variable)                                      │
│   - packed_ternary: 2 bits per weight                          │
│   - scales: float32 per output channel                         │
├─────────────────────────────────────────────────────────────────┤
│ Metadata (variable)                                             │
│   - tile_routing: learned gating pattern                       │
│   - hit_count: uint32                                           │
│   - avg_latency: float32                                        │
└─────────────────────────────────────────────────────────────────┘
```

### Eviction Policy

**LRU with Neural Hint:**

```python
# Standard LRU + neural-specific hints
def should_evict(slot):
    base_score = slot.age * slot.hit_count
    neural_bonus = slot.is_inference_mode * 0.5  # Prefer evicting training ops
    latency_penalty = slot.avg_latency * 0.1   # Keep low-latency ops
    
    return base_score + neural_bonus - latency_penalty
```

---

## 4. Integration with Neural 6502

### Current Architecture (Monolithic)

```python
# All compute in one neural network
state = {A: 10, X: 20, ...}
next_state = neural_cpu(state)
```

### Proposed Architecture (APU-Based)

```python
# CPU handles control flow, APU handles learned patterns

def execute_opcode(op, state):
    if op.is_arithmetic:
        # Traditional CPU
        return cpu_alu(op, state)
    elif op.is_learned:
        # Neural APU
        return apu.call(op.apu_opcode, state, cache=state.l1_cache)
    else:
        # Hybrid
        cpu_result = cpu_alu(op, state)
        return apu.refine(cpu_result, state, cache=state.l1_cache)
```

### Example: ADC with APU Refinement

```assembly
; Current: Pure neural (84.4% accuracy, ADC at 3.1%)
LDA #$10
ADC #$20
STA result

; Proposed: CPU + APU hybrid
LDA #$10        ; CPU: Load to accumulator
ADC #$20        ; CPU: Basic addition
STA temp        ; Store temp

; APU: Refine carry using learned pattern
TATTN temp, carry_table, tile=0  ; APU refines carry
STA result
```

---

## 5. Ternary Instruction Set

### Weight Encoding

```
┌──────────────────────────────────────────────────────────────┐
│ 2-bit Ternary Instruction                                    │
├──────────────────────────────────────────────────────────────┤
│ 00 = skip (0)                                                │
│ 01 = +1 (identity)                                           │
│ 10 = -1 (negate)                                             │
│ 11 = op_select (Dark State - select operation)              │
└──────────────────────────────────────────────────────────────┘
```

### Operation Selection (Dark State as Opcode)

```python
OPS = {
    0b00: lambda x, w: 0,              # Zero
    0b01: lambda x, w: x * w,          # Multiply
    0b10: lambda x, w: -x * w,         # Negated multiply
    0b11: lambda x, w: custom_op(x, w) # Custom (Dark State)
}

# Custom operations could be:
# - abs(), clamp(), lookup(), sigmoid(), etc.
```

---

## 6. Implementation Phases

### Phase 1: Opcode Interface ✓ (COMPLETE)

**Implemented:**
- `bbdos/kernel/bitswitch_avx.cpp` - AVX2 matrix multiplication
- `bbdos/kernel/apu_cache.py` - L-Cache with LRU eviction
- `bbdos/kernel/bindings.py` - Python bindings with AVX detection
- `tests/test_apu.py` - 19 passing tests

**Usage:**
```python
from bbdos.kernel import NeuralAPU

apu = NeuralAPU(cache_size=16)
apu.register("TMUL_256", weights=weights, scales=scales)
result = apu.exec("TMUL_256", input_a, input_b)
print(apu.cache_stats())
# {'hits': 128, 'misses': 16, 'hit_rate': 0.888}
```

### Phase 2: L-Cache Integration ✓ (COMPLETE)

**Implemented:**
- LRU eviction with neural hints
- Hit/miss tracking
- Latency profiling
- Thread-safe access

### Phase 3: Hybrid Execution (TODO)

- Identify ops where APU outperforms CPU
- Add routing between CPU and APU
- Implement refine() for corrections

---

## 7. File Structure (Implementation)

```
bbdos/kernel/
├── bitswitch.h          # C API (original)
├── bitswitch.cpp        # NEON implementation
├── bitswitch_avx.cpp   # AVX2 implementation (NEW)
├── bindings.py         # Python bindings (updated)
├── apu_cache.py        # L-Cache (NEW)
└── __init__.py        # Exports (updated)
```

### Phase 3: Hybrid Execution

- Identify ops where APU outperforms CPU
- Add routing between CPU and APU
- Implement refine() for corrections

---

## 7. Comparison: Current vs APU

| Aspect | Current BBDOS | Neural APU |
|--------|---------------|------------|
| Compute model | Single model | Multiple opcodes |
| Caching | None | L-cache with eviction |
| Integration | Standalone | Host CPU addon |
| Flexibility | Fixed architecture | Opcode-extensible |
| Ternary ops | FFN only | Full instruction set |

---

## 8. Example Usage

```python
# Initialize APU
apu = NeuralAPU(cache_size=16)

# Register operations
apu.register('TMUL_256', weights=packed_weights, scales=scales)
apu.register('TGATE_4', gating_network=gating_net)

# Execute like CPU opcodes
result = apu.exec('TMUL_256', input_a, input_b)
gated, gates = apu.exec('TGATE_4', input, mode='inference')

# Cache statistics
print(apu.cache.stats())
# {'hits': 128, 'misses': 16, 'hit_rate': 0.888, 'avg_latency': '0.4ms'}
```

---

## Summary

**Neural APU = Ternary Compute as Opcode**

- **APU** — New execution unit, callable like CPU opcode
- **Ternary** — -1/0/+1 weights, 2-bit encoding
- **L-Cache** — Cache recent operations with LRU eviction
- **Hybrid** — CPU handles deterministic, APU handles learned patterns

**Next Step:** Implement Phase 1 — the opcode interface layer.
