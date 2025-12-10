# Abby Normal v2: Final Experimental Plan

*Synthesis of iterative exploration. Incorporating Gemini's insight.*

---

## The Core Thesis

> **"You didn't fix the brain; you changed the language to match the brain."**
> — Gemini

The Neural 6502 achieves 97% on shift operations. Carry IS a shift - a conditional, routed shift. The model doesn't need to LEARN arithmetic. It needs to RECOGNIZE that carry is shifting.

**What's missing:** The TOPOLOGY that tells the model where shifts should route.

---

## The Hypothesis Evolution

| Version | Hypothesis | Solution |
|---------|-----------|----------|
| v1 | Model lacks positional awareness | Add position encoding |
| v2 | Model lacks adjacency topology | Add neighbor information |
| v2+ | Model knows shifting, lacks routing | Make carry look like shifting |

**Final hypothesis:** Adjacency encoding will help MORE than positional encoding, because carry is LOCAL (bit N → bit N+1), not GLOBAL (bit N has weight 2^N).

---

## The Decomposition

The model already has the pieces:

| Capability | Evidence | Accuracy |
|------------|----------|----------|
| Shifting | ASL, LSR operations | 97% |
| Conditional logic | Branch operations | 96-99% |
| State tracking | Flag manipulation | 95%+ |

What it CAN'T do: **Route the carry to the correct neighbor.**

Arithmetic = Shifting + Conditionals + Routing

The model has 2/3. We provide the third.

---

## The Abacus Layer v2

```python
class AbacusLayer(nn.Module):
    """
    Provides topological awareness for numerical bit representations.
    
    Encodes:
    - bit_value: The actual 0/1 value
    - position: Ordinal position (0-7) - optional, for global context
    - left_neighbor: Value of bit to the left (carry destination)
    - right_neighbor: Value of bit to the right (carry source)
    - direction: Encodes that carry flows left (optional)
    """
    
    def __init__(self, num_bits=8, embed_dim=16, mode='adjacency'):
        super().__init__()
        self.num_bits = num_bits
        self.mode = mode
        
        # Determine input dimension based on mode
        if mode == 'position':
            input_dim = 2  # bit, position
        elif mode == 'adjacency':
            input_dim = 3  # bit, left_neighbor, right_neighbor
        elif mode == 'full':
            input_dim = 5  # bit, position, weight, left_neighbor, right_neighbor
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Position and weight buffers (for modes that need them)
        positions = torch.arange(num_bits).float() / num_bits
        weights = torch.pow(2.0, torch.arange(num_bits).float()) / (2 ** num_bits)
        self.register_buffer('positions', positions)
        self.register_buffer('weights', weights)
        
        # Learnable projection
        self.proj = nn.Linear(input_dim, embed_dim)
    
    def forward(self, bits):
        """
        Args:
            bits: [batch, num_bits] tensor of bit values
        Returns:
            [batch, num_bits, embed_dim] topology-aware embeddings
        """
        batch = bits.shape[0]
        
        if self.mode == 'position':
            pos = self.positions.unsqueeze(0).expand(batch, -1)
            combined = torch.stack([bits, pos], dim=-1)
            
        elif self.mode == 'adjacency':
            # Left neighbor: bit i+1's value (carry flows left/up)
            # Pad right with 0 (no neighbor beyond MSB)
            left = F.pad(bits[:, 1:], (0, 1), value=0)
            
            # Right neighbor: bit i-1's value (carry comes from right/down)  
            # Pad left with 0 (no neighbor beyond LSB, but carry_in could go here)
            right = F.pad(bits[:, :-1], (1, 0), value=0)
            
            combined = torch.stack([bits, left, right], dim=-1)
            
        elif self.mode == 'full':
            pos = self.positions.unsqueeze(0).expand(batch, -1)
            wts = self.weights.unsqueeze(0).expand(batch, -1)
            left = F.pad(bits[:, 1:], (0, 1), value=0)
            right = F.pad(bits[:, :-1], (1, 0), value=0)
            combined = torch.stack([bits, pos, wts, left, right], dim=-1)
        
        return self.proj(combined)
```

---

## Experimental Design: Distinguishing Theories

### Phase 1: Single-Variable Tests

| Exp | Mode | What Model Sees | Prediction |
|-----|------|-----------------|------------|
| 1A | baseline | bits only | ~3% (known) |
| 1B | position | bits + ordinal position | 10-30% |
| 1C | weight | bits + arithmetic weight (2^i) | 10-30% |
| 1D | adjacency | bits + neighbor values | **50-80%** |

**Key test:** Is 1D >> 1B? If yes, adjacency matters more than position.

### Phase 2: Combination Tests

| Exp | Mode | What Model Sees | Prediction |
|-----|------|-----------------|------------|
| 2A | position + adjacency | bits + position + neighbors | 60-90% |
| 2B | full | bits + position + weight + neighbors | **80-95%** |
| 2C | random_adjacency | bits + WRONG neighbors | ~3% (control) |
| 2D | shuffled_position | bits + WRONG positions | ~3% (control) |

**Key test:** Does 2C ≈ 1A? If yes, CORRECT adjacency is required, not just more features.

### Phase 3: If Needed

| Exp | Approach | When to Try |
|-----|----------|-------------|
| 3A | Generate/Propagate signals | If adjacency helps but plateaus <80% |
| 3B | Bead representation | If all encoding approaches fail |
| 3C | Architectural change | If representation changes don't help |

---

## The Critical Comparisons

```
If 1D >> 1B:     Adjacency hypothesis confirmed (Gemini was right)
If 1B >> 1D:     Position hypothesis confirmed (I was originally right)
If 2B >> 1D:     Both needed (synergy)
If 2C ≈ 1A:      Correct topology required (not just more parameters)
If all ≈ 1A:     Representation isn't the problem (architectural)
```

---

## Implementation Plan

### Day 1: Build and Test Layer

```
CREATE: bbdos/cpu/abacus.py
  - AbacusLayer class with modes: position, adjacency, full
  - Configurable embedding dimension
  - Clean interface

CREATE: tests/test_abacus.py
  - Shape tests for each mode
  - Gradient flow verification
  - Encoding value verification
```

### Day 2: Integrate and Baseline

```
MODIFY: bbdos/cpu/model.py
  - Add abacus_mode config option
  - Wrap bit inputs through AbacusLayer when enabled
  - Handle increased feature dimension

CREATE: scripts/eval_adc.py
  - Focused ADC evaluation
  - Per-operand breakdown (magnitude, carry chain length)
  - Failure mode analysis

RUN: Baseline characterization
  - Confirm 3.1% ADC accuracy
  - Analyze what the 3.1% gets right
```

### Day 3-4: Experiments

```
Phase 1: 1A, 1B, 1C, 1D (4 experiments × 3 seeds = 12 runs)
Phase 2: 2A, 2B, 2C, 2D (4 experiments × 3 seeds = 12 runs)

~2 hrs per run = ~48 hrs total (can parallelize)
```

### Day 5: Analysis

```
- Compare accuracies across conditions
- Statistical significance tests
- Failure mode analysis for best condition
- Decision: proceed to Phase 3 or declare success
```

---

## Success Criteria

| Outcome | ADC Accuracy | Interpretation |
|---------|--------------|----------------|
| Failure | <10% | Representation isn't the issue |
| Partial | 10-50% | Right direction, needs refinement |
| Success | 50-80% | Hypothesis confirmed, adjacency helps |
| Home Run | >80% | Topology was the missing piece |
| Jackpot | >95% | Publish immediately |

---

## The Bead Representation (Held in Reserve)

If Phases 1-2 fail, try radical re-representation:

```python
class BeadLayer(nn.Module):
    """
    Represent numbers as sets of occupied positions.
    
    Instead of: [0, 1, 0, 1] = "bits 1 and 3 are on"
    Represent:  positions {1, 3} are occupied
    
    Addition becomes: merge position sets, resolve collisions
    Collision at position i → remove both, add position i+1
    This IS a shift operation.
    """
    
    def __init__(self, num_bits=8, embed_dim=16):
        super().__init__()
        # Each position gets an embedding
        self.position_embed = nn.Embedding(num_bits, embed_dim)
        # Collision detector
        self.collision_net = nn.Linear(embed_dim * 2, embed_dim)
    
    def forward(self, a_bits, b_bits):
        # Get position embeddings for set bits
        # Apply collision detection
        # Output: resolved position set
        pass  # Design TBD if needed
```

This reframes arithmetic as geometry. Collision resolution IS carry propagation.

---

## What We're Really Testing

**Surface level:** Does adding topology information improve ADC accuracy?

**Deeper level:** Is the Savant CPU phenomenon a fundamental limit, or a representation gap?

**Deepest level:** Can neural networks do arithmetic if given the right geometric grounding?

---

## The Predictions, Stated Clearly

1. **Adjacency encoding will improve ADC accuracy significantly** (>50% vs 3% baseline)

2. **Adjacency will help more than absolute position** (1D >> 1B)

3. **Wrong adjacency will NOT help** (2C ≈ 1A)

4. **The improvement will generalize to SBC** (same topology, different operation)

If all four predictions hold, the hypothesis is strongly confirmed.

If any fail, we learn something specific about what's missing.

---

## Final Word

Two iterations of journaling. One external perspective (Gemini). One crystallized insight:

> **The model is a shift-capable engine looking for routing instructions.**

We're not teaching arithmetic. We're providing the address book.

Let's find out if that's enough.

---

*Ready to build.*
