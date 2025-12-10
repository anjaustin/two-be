# Wired Voltron - Connectors and Adaptors Architecture

## The Shift

**Before (Orchestrated):**
```
Python fetches → Python decodes → Python routes → Neural computes → Python commits
```

**After (Wired):**
```
Input pins → Wiring → Output pins
```

One `forward()` call. No Python in the loop.

---

## The Topology

```
                            ┌─────────────────────────────────────────┐
                            │            WIRED VOLTRON                │
                            │                                         │
  ┌──────────┐              │  ┌─────────┐    ┌──────────────────┐   │
  │ Opcode   │──────────────┼─>│ Router  │───>│ Specialist Bank  │   │
  └──────────┘              │  │ (LUT)   │    │                  │   │
                            │  └─────────┘    │  ┌────────────┐  │   │
  ┌──────────┐              │       │         │  │ ALU        │  │   │
  │ Operand  │──────────────┼───────┼────────>│  │ (Soroban)  │  │   │
  └──────────┘              │       │         │  └────────────┘  │   │
                            │       │         │  ┌────────────┐  │   │
  ┌──────────┐              │       │         │  │ Logic      │  │   │
  │ A_in     │──────────────┼───────┼────────>│  │ (Binary)   │  │   │
  └──────────┘              │       │         │  └────────────┘  │   │
                            │       │         │  ┌────────────┐  │   │
  ┌──────────┐              │       │         │  │ Transfer   │  │   │
  │ X_in     │──────────────┼───────┼────────>│  │            │  │   │
  └──────────┘              │       │         │  └────────────┘  │   │
                            │       │         │  ┌────────────┐  │   │
  ┌──────────┐              │       │         │  │ Stack      │  │   │
  │ Y_in     │──────────────┼───────┼────────>│  │            │  │   │
  └──────────┘              │       │         │  └────────────┘  │   │
                            │       │         │  ┌────────────┐  │   │
  ┌──────────┐              │       │         │  │ Flags      │  │   │
  │ SP_in    │──────────────┼───────┼────────>│  │            │  │   │
  └──────────┘              │       │         │  └────────────┘  │   │
                            │       │         │  ┌────────────┐  │   │
  ┌──────────┐              │       │         │  │ Branch     │  │   │
  │ P_in     │──────────────┼───────┼────────>│  │            │  │   │
  └──────────┘              │       │         │  └────────────┘  │   │
                            │       │         └──────────────────┘   │
                            │       │                   │            │
                            │       v                   v            │
                            │  ┌─────────┐    ┌──────────────────┐   │
                            │  │ Select  │<───│ Adaptor Bank     │   │
                            │  │ (mask)  │    │ (format convert) │   │
                            │  └─────────┘    └──────────────────┘   │
                            │       │                                │
                            │       v                                │
                            │  ┌─────────────────────────────────┐   │
                            │  │         Output Combiner         │   │
                            │  └─────────────────────────────────┘   │
                            │       │                                │
                            └───────┼────────────────────────────────┘
                                    v
                    ┌───────────────────────────────────┐
                    │  A_out, X_out, Y_out, SP_out, P_out │
                    └───────────────────────────────────┘
```

---

## The Components

### 1. Input Adaptors
Convert raw bytes to specialist-native formats:

```python
class InputAdaptor(nn.Module):
    def __init__(self):
        self.to_soroban = SorobanEncoder()  # For ALU
        self.to_binary = lambda x: x / 255.0  # For Logic/Transfer
    
    def forward(self, A, X, Y, SP, P, opcode, operand):
        return {
            'soroban': {
                'A': self.to_soroban(A),
                'operand': self.to_soroban(operand),
            },
            'binary': {
                'A': self.to_binary(A),
                'X': self.to_binary(X),
                'Y': self.to_binary(Y),
                'operand': self.to_binary(operand),
            },
            'flags': P,
            'opcode': opcode,
        }
```

### 2. Router (Learned or LUT)
Maps opcode to specialist activation mask:

```python
class Router(nn.Module):
    def __init__(self):
        # 256 opcodes → 6 specialists (one-hot or soft)
        # Could be learned or hardcoded LUT
        self.routing_table = self._build_lut()
    
    def forward(self, opcode):
        # Returns: [alu_weight, logic_weight, transfer_weight, 
        #           stack_weight, flags_weight, branch_weight]
        return self.routing_table[opcode]
```

### 3. Specialist Bank
All specialists run in parallel, outputs masked by router:

```python
class SpecialistBank(nn.Module):
    def __init__(self):
        self.alu = NeuralALU()       # Soroban-powered
        self.logic = LogicUnit()      # Binary
        self.transfer = TransferUnit()
        self.stack = StackUnit()
        self.flags = FlagsUnit()
        self.branch = BranchUnit()
    
    def forward(self, inputs, router_mask):
        # All specialists compute (or we can gate for efficiency)
        outputs = {
            'alu': self.alu(inputs['soroban']),
            'logic': self.logic(inputs['binary']),
            'transfer': self.transfer(inputs['binary']),
            'stack': self.stack(inputs['binary']),
            'flags': self.flags(inputs['flags']),
            'branch': self.branch(inputs['binary']),
        }
        return outputs, router_mask
```

### 4. Output Adaptors
Convert specialist outputs back to common format:

```python
class OutputAdaptor(nn.Module):
    def __init__(self):
        self.from_soroban = SorobanDecoder()
    
    def forward(self, specialist_outputs):
        # Normalize all outputs to [A, X, Y, SP, P] format
        return {
            'alu': self._adapt_alu(specialist_outputs['alu']),
            'logic': self._adapt_logic(specialist_outputs['logic']),
            # ...
        }
```

### 5. Output Combiner
Weighted sum based on router mask:

```python
class OutputCombiner(nn.Module):
    def forward(self, adapted_outputs, router_mask):
        # router_mask selects which specialist's output to use
        # For hard routing: just index
        # For soft routing: weighted sum
        
        A_out = sum(mask * out['A'] for mask, out in zip(router_mask, adapted_outputs))
        # ... same for X, Y, SP, P
        
        return A_out, X_out, Y_out, SP_out, P_out
```

---

## The Wired Module

```python
class WiredVoltron(nn.Module):
    def __init__(self):
        self.input_adaptor = InputAdaptor()
        self.router = Router()
        self.specialists = SpecialistBank()
        self.output_adaptor = OutputAdaptor()
        self.combiner = OutputCombiner()
    
    def forward(self, A, X, Y, SP, P, opcode, operand):
        # 1. Adapt inputs to specialist formats
        inputs = self.input_adaptor(A, X, Y, SP, P, opcode, operand)
        
        # 2. Route: which specialist handles this opcode?
        mask = self.router(opcode)
        
        # 3. Run specialists (parallel)
        outputs, mask = self.specialists(inputs, mask)
        
        # 4. Adapt outputs to common format
        adapted = self.output_adaptor(outputs)
        
        # 5. Combine based on routing
        A_out, X_out, Y_out, SP_out, P_out = self.combiner(adapted, mask)
        
        return A_out, X_out, Y_out, SP_out, P_out
```

---

## Key Decisions

### Hard vs Soft Routing
- **Hard:** LUT maps opcode to exactly one specialist. Efficient. Deterministic.
- **Soft:** Learned router outputs weights. Could blend specialists. More flexible.

**Recommendation:** Start with hard routing (we know which specialist handles what). Soft routing is for when you don't know the decomposition.

### Parallel vs Gated Execution
- **Parallel:** All specialists run, output masked. Simple but wasteful.
- **Gated:** Only activated specialist runs. Efficient but needs conditional logic.

**Recommendation:** For GPU, parallel is fine (SIMD loves it). For CPU/edge, gating matters.

### Adaptor Complexity
- **Simple:** Just reshaping and scaling. No learned params.
- **Learned:** Small networks that learn format conversion.

**Recommendation:** Start simple. If accuracy drops at boundaries, add learned adaptors.

---

## What This Buys Us

1. **One forward() = one CPU cycle.** No Python in the loop.
2. **Batched execution.** Process 10K instructions in parallel.
3. **Differentiable.** Could fine-tune end-to-end if needed.
4. **Portable.** Export to ONNX, run anywhere.
5. **Hardware-like.** Topology is explicit, not hidden in code.

---

## Migration Path

1. **Phase 1:** Wire existing specialists with hard routing, simple adaptors.
2. **Phase 2:** Test on Fibonacci (must still work).
3. **Phase 3:** Benchmark batched throughput.
4. **Phase 4:** Optimize (gating, fusion, quantization).
5. **Phase 5:** Export to standalone runtime.

---

## The Goal

```python
# Before
for cycle in range(1000):
    opcode = memory[PC]
    operand = fetch_operand(...)  # Python
    new_state = route_to_specialist(...)  # Python
    commit(new_state)  # Python

# After
states = wired_voltron(
    A_batch, X_batch, Y_batch, SP_batch, P_batch,
    opcode_batch, operand_batch
)  # One call, 1000 cycles
```

The wiring IS the computer. The forward pass IS the execution.
