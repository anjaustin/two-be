# Abby Normal: Experimental Plan

*Testing the Abacus Layer hypothesis for neural arithmetic*

---

## The Hypothesis

**Neural networks fail at carry arithmetic because they lack positional awareness, not because they lack computational capability.**

Numbers are positions. Carry is movement between positions. Without knowing the geometry, the network can't learn the operation.

**Prediction:** Adding explicit positional encoding for bit positions will improve ADC accuracy from ~3% to >80%.

---

## Theoretical Foundation

```
Observation:   Neural 6502 achieves 99.9% on stack ops, 3.1% on ADC
Diagnosis:     Stack ops are position-independent; ADC requires positional structure
Hypothesis:    Provide positional structure → ADC becomes learnable
Precedent:     Positional encoding enabled transformers to understand sequences
```

The sharp cliff (99.9% → 3.1%) is the fingerprint of a missing representation, not a missing capability.

---

## The Abacus Layer

### Core Design

```python
class AbacusLayer(nn.Module):
    """Provides positional awareness for numerical bit representations."""
    
    def __init__(self, num_bits: int = 8, embed_dim: int = 16):
        super().__init__()
        self.num_bits = num_bits
        self.embed_dim = embed_dim
        
        # Fixed positional weights: powers of 2, log-scaled
        # [0, 1, 2, 3, 4, 5, 6, 7] for 8 bits
        positions = torch.arange(num_bits, dtype=torch.float32)
        self.register_buffer('positions', positions)
        
        # Arithmetic weights: 2^i normalized
        # [1/256, 2/256, 4/256, 8/256, 16/256, 32/256, 64/256, 128/256]
        arith_weights = torch.pow(2.0, positions) / (2 ** num_bits)
        self.register_buffer('arith_weights', arith_weights)
        
        # Learnable projection: (bit_value, position, arith_weight) -> embedding
        self.proj = nn.Linear(3, embed_dim)
    
    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bits: [batch, num_bits] tensor of bit values in {0, 1}
        Returns:
            [batch, num_bits, embed_dim] positionally-aware embeddings
        """
        batch_size = bits.shape[0]
        
        # Expand position info to batch
        pos = self.positions.unsqueeze(0).expand(batch_size, -1)
        weights = self.arith_weights.unsqueeze(0).expand(batch_size, -1)
        
        # Combine: [batch, num_bits, 3]
        combined = torch.stack([bits.float(), pos / self.num_bits, weights], dim=-1)
        
        # Project to embedding space
        return self.proj(combined)
```

### Why This Design

| Input | Purpose |
|-------|---------|
| `bits` | The actual 0/1 values |
| `positions` | Ordinal position (0, 1, 2...) - tells adjacency |
| `arith_weights` | Arithmetic weight (2^i) - tells value contribution |

The network receives three signals:
1. **What** is the bit value
2. **Where** is the bit (sequential position)
3. **How much** does this bit contribute (arithmetic weight)

---

## Experimental Protocol

### Phase 1: Baseline Characterization

**Goal:** Understand current failure mode in detail.

1. Run current Neural 6502 on ADC operation suite
2. Record per-example accuracy
3. Analyze failures:
   - Correlation with operand magnitude
   - Correlation with carry chain length
   - Correlation with bit patterns
4. Identify: does 3.1% accuracy come from special cases (adding 0, small numbers)?

**Output:** Failure mode report. Hypothesis refinement if needed.

---

### Phase 2: Core Experiment

**Goal:** Test the abacus layer hypothesis.

#### Treatment Groups

| Group | Description | Expected ADC Accuracy |
|-------|-------------|----------------------|
| A: Baseline | Current Neural 6502 | ~3% (known) |
| B: Abacus | + AbacusLayer with correct encoding | >80% (hypothesis) |
| C: Random | + AbacusLayer with shuffled positions | ~3% (control) |
| D: Uniform | + AbacusLayer with all positions = 0.5 | ~3% (control) |
| E: Learned | + AbacusLayer with learnable positions | ??? (exploratory) |

#### Training Protocol

- Train from scratch (not fine-tune)
- Same hyperparameters across all groups
- Same training data, same seed
- 3 runs per group for variance estimation

#### Success Criteria

| Outcome | Interpretation |
|---------|----------------|
| B >> A, B >> C, B >> D | **Hypothesis confirmed**: positional encoding enables arithmetic |
| B ≈ A | **Hypothesis rejected**: positional encoding doesn't help |
| B > A but B < 80% | **Partial confirmation**: necessary but not sufficient |
| E >> B | **Surprise**: learned encoding better than designed |

---

### Phase 3: Ablations

**Goal:** Understand which components matter.

| Ablation | Question |
|----------|----------|
| Position only (no arith_weight) | Is ordinal position enough? |
| Arith_weight only (no position) | Is magnitude enough? |
| No projection (raw concat) | Does the MLP matter? |
| Sinusoidal encoding | Does transformer-style PE work? |
| Relative position (convolution) | Is locality the key? |

---

### Phase 4: Generalization

**Goal:** Test if the insight generalizes beyond ADC.

| Test | Purpose |
|------|---------|
| SBC (subtract with carry) | Same structure, different op |
| Multi-byte addition | Does it scale to 16/32 bits? |
| Multiplication | Harder positional structure |
| Division | Even harder |
| BCD arithmetic | Different base, same principle? |
| Decimal (if applicable) | Does the principle generalize beyond binary? |

---

## Implementation Plan

### Files to Create/Modify

```
bbdos/
├── cpu/
│   ├── abacus.py          # NEW: AbacusLayer implementation
│   └── model.py           # MODIFY: integrate AbacusLayer option
├── configs/
│   └── cpu_abacus.yaml    # NEW: config for abacus experiments
scripts/
├── train_cpu.py           # MODIFY: support abacus configs
├── eval_arithmetic.py     # NEW: detailed arithmetic evaluation
tests/
├── test_abacus.py         # NEW: unit tests for AbacusLayer
```

### Minimal Change Set

1. **Create `abacus.py`** with AbacusLayer class
2. **Modify `cpu/model.py`** to optionally wrap bit inputs with AbacusLayer
3. **Create evaluation script** that reports per-operation accuracy
4. **Run experiments**

---

## Timeline

| Day | Task |
|-----|------|
| 1 | Implement AbacusLayer, write tests |
| 2 | Integrate into Neural 6502, baseline characterization |
| 3-4 | Phase 2 experiments (5 groups × 3 runs) |
| 5 | Analysis, ablations |
| 6 | Generalization tests if Phase 2 succeeds |
| 7 | Write-up |

---

## Success Metrics

### Minimum Viable Success

ADC accuracy improves from 3.1% to **>50%** with AbacusLayer, AND controls (random, uniform) show no improvement.

This would confirm the hypothesis is directionally correct, even if not fully solved.

### Strong Success

ADC accuracy improves to **>80%**, approaching the ~96% seen on non-arithmetic operations.

This would confirm positional encoding is the primary missing piece.

### Home Run

ADC accuracy improves to **>95%**, AND generalizes to other arithmetic operations (SBC, multi-byte), AND controls are flat.

This would be a publishable result with implications for neural arithmetic broadly.

---

## If It Fails

Possible failure modes and next steps:

| Failure Mode | Interpretation | Next Step |
|--------------|----------------|-----------|
| B ≈ A, C, D | Position doesn't help | Try architectural changes (recurrent, iterative) |
| B > A but << 80% | Necessary but not sufficient | Add explicit carry mechanism |
| Random variance | Training instability | Hyperparameter search, curriculum learning |
| Works on ADC, fails on SBC | Overfitting to operation | Need more general representation |

---

## The Question We're Answering

**Can neural networks learn arithmetic if given the right geometric grounding?**

The 6502 is our testbed. ADC is our probe. The AbacusLayer is our intervention.

If it works: neural networks CAN do math. They just needed an abacus.

If it doesn't: we learn what else is missing.

Either way, we advance understanding.

---

## Appendix: The Name

**Abby Normal** (noun): A neural network layer providing positional awareness for arithmetic.

Etymology:
- "Abby" from Abacus
- "Normal" because it normalizes arithmetic for neural nets
- Young Frankenstein reference: "Abby... Normal"

Usage: "We added the Abby Normal layer and ADC accuracy went from 3% to 94%."

---

*Ready to test.*
