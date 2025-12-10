# Abby Normal: Final Experimental Plan (Stereo Edition)

*Two theories. One experiment. Let the data decide.*

---

## The Two Theories

| Theory | Name | Core Idea | Intervention |
|--------|------|-----------|--------------|
| **T1** | Adjacency | Model knows shifting, lacks routing topology | Add neighbor information to binary bits |
| **T2** | Soroban | Model needs visible magnitude/overflow | Change representation to split-byte thermometer |

Both are plausible. Both have strong arguments. We test both.

---

## The Dialectic

```
Thesis:      Binary (8 bits, compact, chaotic carry)
Antithesis:  Full Thermometer (256 bits, visible, massive)
Synthesis:   Soroban (32 bits, visible, efficient)

Parallel:    Adjacency doesn't change representation - it augments it
```

**T1 says:** The representation is fine; the model needs metadata.
**T2 says:** The representation is wrong; change it entirely.

---

## Experimental Design: Stereo

### Track A: Adjacency Path (Augment Binary)

| Exp | Encoding | Input Dim | What Model Sees |
|-----|----------|-----------|-----------------|
| A0 | Binary baseline | 8 | Raw bits |
| A1 | Binary + position | 16 | Bits + ordinal position |
| A2 | Binary + adjacency | 24 | Bits + left/right neighbors |
| A3 | Binary + full | 40 | Bits + position + weight + neighbors |
| A4 | Binary + random adjacency | 24 | Bits + WRONG neighbors (control) |

### Track B: Soroban Path (Change Representation)

| Exp | Encoding | Input Dim | What Model Sees |
|-----|----------|-----------|-----------------|
| B0 | Soroban baseline | 32 | Two 16-bit thermometers |
| B1 | Soroban + column adjacency | 36 | Thermometers + cross-column signal |
| B2 | Soroban + position | 48 | Thermometers + bead positions |
| B3 | Shuffled Soroban | 32 | WRONG thermometer mapping (control) |

### Track C: Hybrid (Both)

| Exp | Encoding | Input Dim | What Model Sees |
|-----|----------|-----------|-----------------|
| C1 | Soroban + full adjacency | 64 | Everything |

---

## The Key Comparisons

```
A2 vs A0:  Does adjacency help binary?
B0 vs A0:  Does Soroban beat binary (even without extras)?
B0 vs A3:  Does simple Soroban beat augmented binary?
C1 vs B0:  Does adjacency help Soroban?
A4 vs A0:  Does WRONG adjacency help? (should be NO)
B3 vs A0:  Does WRONG Soroban help? (should be NO)
```

**The critical question:** Is B0 >> A3?

If yes: Representation matters more than augmentation.
If no: Augmentation can save a bad representation.

---

## Implementation

### File: `bbdos/cpu/abacus.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class AbacusLayer(nn.Module):
    """
    Track A: Adjacency-based augmentation of binary representation.
    """
    
    def __init__(self, num_bits=8, embed_dim=16, mode='adjacency'):
        super().__init__()
        self.num_bits = num_bits
        self.mode = mode
        
        if mode == 'none':
            input_dim = 1
        elif mode == 'position':
            input_dim = 2
        elif mode == 'adjacency':
            input_dim = 3
        elif mode == 'full':
            input_dim = 5
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        positions = torch.arange(num_bits).float() / num_bits
        weights = torch.pow(2.0, torch.arange(num_bits).float()) / (2 ** num_bits)
        self.register_buffer('positions', positions)
        self.register_buffer('weights', weights)
        
        self.proj = nn.Linear(input_dim, embed_dim)
    
    def forward(self, bits):
        """
        Args:
            bits: [batch, num_bits] binary values
        Returns:
            [batch, num_bits, embed_dim] augmented embeddings
        """
        batch = bits.shape[0]
        bits_f = bits.float()
        
        if self.mode == 'none':
            combined = bits_f.unsqueeze(-1)
            
        elif self.mode == 'position':
            pos = self.positions.unsqueeze(0).expand(batch, -1)
            combined = torch.stack([bits_f, pos], dim=-1)
            
        elif self.mode == 'adjacency':
            left = F.pad(bits_f[:, 1:], (0, 1), value=0)
            right = F.pad(bits_f[:, :-1], (1, 0), value=0)
            combined = torch.stack([bits_f, left, right], dim=-1)
            
        elif self.mode == 'full':
            pos = self.positions.unsqueeze(0).expand(batch, -1)
            wts = self.weights.unsqueeze(0).expand(batch, -1)
            left = F.pad(bits_f[:, 1:], (0, 1), value=0)
            right = F.pad(bits_f[:, :-1], (1, 0), value=0)
            combined = torch.stack([bits_f, pos, wts, left, right], dim=-1)
        
        return self.proj(combined)


class SorobanEncoder(nn.Module):
    """
    Track B: Split-Byte Thermometer encoding.
    Turns arithmetic into Tetris.
    """
    
    def __init__(self, embed_dim=16, add_column_adjacency=False):
        super().__init__()
        self.nibble_size = 16
        self.total_dim = 32
        self.add_column_adjacency = add_column_adjacency
        
        input_dim = 1
        if add_column_adjacency:
            input_dim = 2  # bead + column_id
        
        self.proj = nn.Linear(input_dim, embed_dim)
        
        # Column identifiers (0 for low, 1 for high)
        col_ids = torch.cat([
            torch.zeros(16),
            torch.ones(16)
        ])
        self.register_buffer('col_ids', col_ids)
    
    def encode_value(self, value_int):
        """
        Encode a single integer (0-255) as Soroban thermometer.
        Returns: [32] tensor
        """
        low_val = value_int & 0x0F
        high_val = (value_int >> 4) & 0x0F
        
        t = torch.zeros(self.total_dim)
        if low_val > 0:
            t[:low_val] = 1.0
        if high_val > 0:
            t[16:16 + high_val] = 1.0
        
        return t
    
    def encode_batch(self, values):
        """
        Encode batch of integers.
        Args:
            values: [batch] tensor of integers 0-255
        Returns:
            [batch, 32] thermometer encoding
        """
        batch = values.shape[0]
        result = torch.zeros(batch, self.total_dim, device=values.device)
        
        low_vals = values & 0x0F
        high_vals = (values >> 4) & 0x0F
        
        # Vectorized thermometer encoding
        positions = torch.arange(16, device=values.device).unsqueeze(0)
        
        # Low nibble: position < low_val
        low_mask = positions < low_vals.unsqueeze(1)
        result[:, :16] = low_mask.float()
        
        # High nibble: position < high_val
        high_mask = positions < high_vals.unsqueeze(1)
        result[:, 16:] = high_mask.float()
        
        return result
    
    def decode(self, tensor):
        """
        Decode thermometer back to integer.
        Args:
            tensor: [32] or [batch, 32] logits/probabilities
        Returns:
            Integer or [batch] integers
        """
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        active = (tensor > 0.5).float()
        low_vals = active[:, :16].sum(dim=1).clamp(0, 15).int()
        high_vals = active[:, 16:].sum(dim=1).clamp(0, 15).int()
        
        result = (high_vals << 4) | low_vals
        
        if squeeze:
            return result.item()
        return result
    
    def forward(self, soroban_bits):
        """
        Project Soroban encoding to embeddings.
        Args:
            soroban_bits: [batch, 32] thermometer encoding
        Returns:
            [batch, 32, embed_dim] embeddings
        """
        batch = soroban_bits.shape[0]
        
        if self.add_column_adjacency:
            col = self.col_ids.unsqueeze(0).expand(batch, -1)
            combined = torch.stack([soroban_bits, col], dim=-1)
        else:
            combined = soroban_bits.unsqueeze(-1)
        
        return self.proj(combined)


class HybridEncoder(nn.Module):
    """
    Track C: Soroban + full adjacency (the kitchen sink).
    """
    
    def __init__(self, embed_dim=16):
        super().__init__()
        self.soroban = SorobanEncoder(embed_dim=embed_dim, add_column_adjacency=True)
        
        # Additional adjacency within thermometer
        self.adjacency_proj = nn.Linear(3, embed_dim)  # bead, left, right
    
    def forward(self, soroban_bits):
        """
        Args:
            soroban_bits: [batch, 32] thermometer encoding
        Returns:
            [batch, 32, embed_dim] embeddings with full context
        """
        batch = soroban_bits.shape[0]
        
        # Soroban base encoding
        base = self.soroban(soroban_bits)
        
        # Add within-thermometer adjacency
        left = F.pad(soroban_bits[:, 1:], (0, 1), value=0)
        right = F.pad(soroban_bits[:, :-1], (1, 0), value=0)
        adj = torch.stack([soroban_bits, left, right], dim=-1)
        adj_embed = self.adjacency_proj(adj)
        
        return base + adj_embed


# --- VISUALIZATION ---

def visualize_soroban():
    """Show the Tetris effect."""
    enc = SorobanEncoder()
    
    print("\n" + "=" * 60)
    print("THE NEURAL SOROBAN: Arithmetic as Tetris")
    print("=" * 60)
    
    examples = [
        (15, "Low column FULL (tension)"),
        (1, "One bead"),
        (16, "Overflow! One bead in high column"),
        (255, "Both columns FULL"),
        (0, "Empty"),
    ]
    
    for val, desc in examples:
        t = enc.encode_value(val)
        low = ''.join(['●' if x > 0.5 else '○' for x in t[:16]])
        high = ''.join(['●' if x > 0.5 else '○' for x in t[16:]])
        density = t.sum().item() / 32 * 100
        print(f"\n{val:3d} (0x{val:02X}): {desc}")
        print(f"     Low:  [{low}]")
        print(f"     High: [{high}]")
        print(f"     Density: {density:.0f}% ({int(t.sum())}/32 active)")
    
    print("\n" + "-" * 60)
    print("THE CARRY EVENT: 15 + 1 = 16")
    print("-" * 60)
    
    t15 = enc.encode_value(15)
    t1 = enc.encode_value(1)
    t16 = enc.encode_value(16)
    
    print(f"\n  15: [{''.join(['●' if x > 0.5 else '○' for x in t15[:16]])}] | [{''.join(['●' if x > 0.5 else '○' for x in t15[16:]])}]")
    print(f"      'Low column is FULL. Tension builds...'")
    print(f"\n + 1: [{''.join(['●' if x > 0.5 else '○' for x in t1[:16]])}] | [{''.join(['●' if x > 0.5 else '○' for x in t1[16:]])}]")
    print(f"      'One more bead drops...'")
    print(f"\n" + "=" * 60)
    print(f"  16: [{''.join(['●' if x > 0.5 else '○' for x in t16[:16]])}] | [{''.join(['●' if x > 0.5 else '○' for x in t16[16:]])}]")
    print(f"      'LINE CLEAR! Low empties, High gains one.'")
    print("=" * 60)


if __name__ == "__main__":
    visualize_soroban()
```

---

## Success Criteria (Updated)

| Outcome | Best Condition | ADC Accuracy | Interpretation |
|---------|----------------|--------------|----------------|
| **Adjacency wins** | A3 >> B0 | 70%+ | Topology > representation |
| **Soroban wins** | B0 >> A3 | 70%+ | Representation > topology |
| **Both needed** | C1 >> both | 90%+ | Synergy |
| **Neither works** | All < 20% | - | Architectural problem |

---

## Predictions (Stated for Falsification)

1. **B0 > A0** (Soroban beats binary baseline) - HIGH confidence
2. **B0 > A2** (Soroban beats binary+adjacency) - MEDIUM confidence
3. **A2 > A0** (Adjacency helps binary) - HIGH confidence
4. **B3 ≈ A0** (Wrong Soroban doesn't help) - HIGH confidence
5. **C1 ≥ B0** (Adding adjacency to Soroban helps or is neutral) - MEDIUM confidence

If prediction 2 holds: Gemini's Soroban insight is the breakthrough.
If prediction 2 fails but 3 holds: My adjacency theory has merit.
If both 2 and 3 hold with similar magnitudes: Both matter.

---

## Timeline

| Day | Task |
|-----|------|
| 1 | Implement AbacusLayer + SorobanEncoder + tests |
| 2 | Integrate into Neural 6502, baseline characterization |
| 3 | Track A experiments (A0-A4) |
| 4 | Track B experiments (B0-B3) + C1 |
| 5 | Analysis, comparison, write-up |

---

## The Philosophy

We're not picking a winner in advance. We're setting up a fair fight:

- **Track A** gets its best shot (full adjacency encoding)
- **Track B** gets its best shot (clean Soroban representation)
- **Track C** combines them (in case synergy matters)
- **Controls** ensure we're not fooling ourselves

The experiment decides. Not our intuitions.

**Stereo FTW.**

---

*Ready to build.*
