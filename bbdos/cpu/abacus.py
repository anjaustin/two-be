"""
Abby Normal: Neural Arithmetic Encoding Layers

Two approaches to make carry arithmetic learnable:
- Track A (AbacusLayer): Augment binary with topology information
- Track B (SorobanEncoder): Change representation to split-byte thermometer

The hypothesis: Neural networks fail at carry arithmetic because they lack
either topology (Track A) or visible magnitude/overflow (Track B).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AbacusLayer(nn.Module):
    """
    Track A: Adjacency-based augmentation of binary representation.
    
    Provides topological awareness to binary bit representations.
    The model knows HOW to shift (97% on ASL/LSR). This layer tells it
    WHERE carries should route.
    
    Modes:
        - 'none': Pass through (baseline)
        - 'position': Add ordinal position (bit 0, bit 1, ...)
        - 'adjacency': Add neighbor values (what's left, what's right)
        - 'full': Position + arithmetic weight + adjacency
    """
    
    def __init__(self, num_bits: int = 8, embed_dim: int = 16, mode: str = 'adjacency'):
        super().__init__()
        self.num_bits = num_bits
        self.embed_dim = embed_dim
        self.mode = mode
        
        # Determine input dimension based on mode
        if mode == 'none':
            input_dim = 1
        elif mode == 'position':
            input_dim = 2  # bit, position
        elif mode == 'adjacency':
            input_dim = 3  # bit, left_neighbor, right_neighbor
        elif mode == 'full':
            input_dim = 5  # bit, position, weight, left_neighbor, right_neighbor
        elif mode == 'random_adjacency':
            input_dim = 3  # bit, random_left, random_right (control)
            self.random_perm = torch.randperm(num_bits)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        self.input_dim = input_dim
        
        # Position and weight buffers
        positions = torch.arange(num_bits).float() / num_bits
        weights = torch.pow(2.0, torch.arange(num_bits).float()) / (2 ** num_bits)
        self.register_buffer('positions', positions)
        self.register_buffer('weights', weights)
        
        # Learnable projection
        self.proj = nn.Linear(input_dim, embed_dim)
    
    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bits: [batch, num_bits] binary values in {0, 1}
        Returns:
            [batch, num_bits, embed_dim] topology-aware embeddings
        """
        batch = bits.shape[0]
        bits_f = bits.float()
        
        if self.mode == 'none':
            combined = bits_f.unsqueeze(-1)
            
        elif self.mode == 'position':
            pos = self.positions.unsqueeze(0).expand(batch, -1)
            combined = torch.stack([bits_f, pos], dim=-1)
            
        elif self.mode == 'adjacency':
            # Left neighbor: where carry GOES (higher bit)
            left = F.pad(bits_f[:, 1:], (0, 1), value=0)
            # Right neighbor: where carry COMES FROM (lower bit)
            right = F.pad(bits_f[:, :-1], (1, 0), value=0)
            combined = torch.stack([bits_f, left, right], dim=-1)
            
        elif self.mode == 'full':
            pos = self.positions.unsqueeze(0).expand(batch, -1)
            wts = self.weights.unsqueeze(0).expand(batch, -1)
            left = F.pad(bits_f[:, 1:], (0, 1), value=0)
            right = F.pad(bits_f[:, :-1], (1, 0), value=0)
            combined = torch.stack([bits_f, pos, wts, left, right], dim=-1)
            
        elif self.mode == 'random_adjacency':
            # Control: shuffled adjacency (should NOT help)
            shuffled = bits_f[:, self.random_perm]
            left = F.pad(shuffled[:, 1:], (0, 1), value=0)
            right = F.pad(shuffled[:, :-1], (1, 0), value=0)
            combined = torch.stack([bits_f, left, right], dim=-1)
        
        return self.proj(combined)


class SorobanEncoder(nn.Module):
    """
    Track B: Split-Byte Thermometer encoding (Neural Soroban).
    
    Turns arithmetic into Tetris.
    
    Instead of 8 binary bits, represent values as two 16-bit thermometers:
    - Low column (0-15): Lower nibble magnitude
    - High column (0-15 * 16): Upper nibble magnitude
    
    Total: 32 bits, but with VISIBLE magnitude and overflow.
    
    Why this works:
    - "Full column" is visually obvious (all 1s)
    - Carry becomes "clear column, increment next" (state transition)
    - Sparse representation plays nice with TriX kernel
    """
    
    def __init__(self, embed_dim: int = 16, add_column_adjacency: bool = False):
        super().__init__()
        self.nibble_size = 16
        self.total_dim = 32
        self.embed_dim = embed_dim
        self.add_column_adjacency = add_column_adjacency
        
        input_dim = 1
        if add_column_adjacency:
            input_dim = 2  # bead + column_id
        
        self.input_dim = input_dim
        self.proj = nn.Linear(input_dim, embed_dim)
        
        # Column identifiers (0 for low nibble, 1 for high nibble)
        col_ids = torch.cat([
            torch.zeros(16),
            torch.ones(16)
        ])
        self.register_buffer('col_ids', col_ids)
    
    def encode_value(self, value_int: int) -> torch.Tensor:
        """
        Encode a single integer (0-255) as Soroban thermometer.
        
        Args:
            value_int: Integer in range [0, 255]
        Returns:
            [32] tensor of thermometer encoding
        """
        low_val = value_int & 0x0F
        high_val = (value_int >> 4) & 0x0F
        
        t = torch.zeros(self.total_dim)
        if low_val > 0:
            t[:low_val] = 1.0
        if high_val > 0:
            t[16:16 + high_val] = 1.0
        
        return t
    
    def encode_batch(self, values: torch.Tensor) -> torch.Tensor:
        """
        Encode batch of integers (vectorized).
        
        Args:
            values: [batch] tensor of integers 0-255
        Returns:
            [batch, 32] thermometer encoding
        """
        batch = values.shape[0]
        device = values.device
        result = torch.zeros(batch, self.total_dim, device=device)
        
        low_vals = (values & 0x0F).long()
        high_vals = ((values >> 4) & 0x0F).long()
        
        # Vectorized thermometer encoding
        positions = torch.arange(16, device=device).unsqueeze(0)  # [1, 16]
        
        # Low nibble: position < low_val means bead is present
        low_mask = positions < low_vals.unsqueeze(1)  # [batch, 16]
        result[:, :16] = low_mask.float()
        
        # High nibble: position < high_val means bead is present
        high_mask = positions < high_vals.unsqueeze(1)  # [batch, 16]
        result[:, 16:] = high_mask.float()
        
        return result
    
    def decode(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Decode thermometer back to integer.
        
        Args:
            tensor: [32] or [batch, 32] logits/probabilities
        Returns:
            Integer or [batch] tensor of integers
        """
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        # Threshold at 0.5 and count active beads
        active = (tensor > 0.5).float()
        low_vals = active[:, :16].sum(dim=1).clamp(0, 15).long()
        high_vals = active[:, 16:].sum(dim=1).clamp(0, 15).long()
        
        result = (high_vals << 4) | low_vals
        
        if squeeze:
            return result.squeeze(0)
        return result
    
    def forward(self, soroban_bits: torch.Tensor) -> torch.Tensor:
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


class SorobanShuffled(SorobanEncoder):
    """
    Control: Soroban with shuffled encoding (should NOT help).
    """
    
    def __init__(self, embed_dim: int = 16):
        super().__init__(embed_dim=embed_dim, add_column_adjacency=False)
        # Random permutation of the 32 positions
        self.register_buffer('shuffle_perm', torch.randperm(32))
        self.register_buffer('unshuffle_perm', torch.argsort(self.shuffle_perm))
    
    def encode_batch(self, values: torch.Tensor) -> torch.Tensor:
        """Encode then shuffle."""
        encoded = super().encode_batch(values)
        return encoded[:, self.shuffle_perm]
    
    def decode(self, tensor: torch.Tensor) -> torch.Tensor:
        """Unshuffle then decode."""
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        unshuffled = tensor[:, self.unshuffle_perm]
        result = super().decode(unshuffled)
        
        if squeeze:
            return result.squeeze(0)
        return result


class HybridEncoder(nn.Module):
    """
    Track C: Soroban + full adjacency (the kitchen sink).
    
    Combines thermometer representation with within-column adjacency.
    Tests whether representation AND topology both matter.
    """
    
    def __init__(self, embed_dim: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.total_dim = 32
        
        # Soroban base with column info
        self.soroban = SorobanEncoder(embed_dim=embed_dim, add_column_adjacency=True)
        
        # Additional within-thermometer adjacency
        self.adjacency_proj = nn.Linear(3, embed_dim)  # bead, left, right
    
    def encode_batch(self, values: torch.Tensor) -> torch.Tensor:
        """Delegate to Soroban encoder."""
        return self.soroban.encode_batch(values)
    
    def decode(self, tensor: torch.Tensor) -> torch.Tensor:
        """Delegate to Soroban encoder."""
        return self.soroban.decode(tensor)
    
    def forward(self, soroban_bits: torch.Tensor) -> torch.Tensor:
        """
        Project with both Soroban and adjacency information.
        
        Args:
            soroban_bits: [batch, 32] thermometer encoding
        Returns:
            [batch, 32, embed_dim] embeddings with full context
        """
        # Soroban base encoding (includes column ID)
        base = self.soroban(soroban_bits)
        
        # Within-thermometer adjacency
        left = F.pad(soroban_bits[:, 1:], (0, 1), value=0)
        right = F.pad(soroban_bits[:, :-1], (1, 0), value=0)
        adj = torch.stack([soroban_bits, left, right], dim=-1)
        adj_embed = self.adjacency_proj(adj)
        
        # Combine
        return base + adj_embed


# --- VISUALIZATION ---

def visualize_soroban():
    """Demonstrate the Tetris effect."""
    enc = SorobanEncoder()
    
    print("\n" + "=" * 70)
    print("THE NEURAL SOROBAN: Arithmetic as Tetris")
    print("=" * 70)
    
    examples = [
        (0, "Empty abacus"),
        (1, "One bead in low column"),
        (15, "Low column FULL (tension building...)"),
        (16, "Overflow! One bead in high column"),
        (17, "High=1, Low=1"),
        (128, "High column half full"),
        (255, "Both columns FULL (maximum tension)"),
    ]
    
    print("\nValue | Low Column (0-15)  | High Column (16-240) | Density")
    print("-" * 70)
    
    for val, desc in examples:
        t = enc.encode_value(val)
        low = ''.join(['●' if x > 0.5 else '○' for x in t[:16]])
        high = ''.join(['●' if x > 0.5 else '○' for x in t[16:]])
        density = t.sum().item() / 32 * 100
        print(f"{val:5d} | {low} | {high} | {density:4.0f}%")
        print(f"      | {desc}")
    
    print("\n" + "=" * 70)
    print("THE CARRY EVENT: 15 + 1 = 16")
    print("=" * 70)
    
    t15 = enc.encode_value(15)
    t1 = enc.encode_value(1)
    t16 = enc.encode_value(16)
    
    low15 = ''.join(['●' if x > 0.5 else '○' for x in t15[:16]])
    high15 = ''.join(['●' if x > 0.5 else '○' for x in t15[16:]])
    
    low1 = ''.join(['●' if x > 0.5 else '○' for x in t1[:16]])
    high1 = ''.join(['●' if x > 0.5 else '○' for x in t1[16:]])
    
    low16 = ''.join(['●' if x > 0.5 else '○' for x in t16[:16]])
    high16 = ''.join(['●' if x > 0.5 else '○' for x in t16[16:]])
    
    print(f"\n   15 | {low15} | {high15}")
    print(f"      | 'Low column FULL. The cup runneth over...'")
    print(f"\n +  1 | {low1} | {high1}")
    print(f"      | 'One more drop...'")
    print(f"\n" + "-" * 70)
    print(f"      | OVERFLOW! State transition triggered.")
    print(f"-" * 70)
    print(f"\n = 16 | {low16} | {high16}")
    print(f"      | 'Low EMPTIES. High gains ONE. Line clear!'")
    print("=" * 70)
    
    # Verify roundtrip
    print("\n" + "=" * 70)
    print("ROUNDTRIP VERIFICATION")
    print("=" * 70)
    
    for val in [0, 1, 15, 16, 127, 128, 255]:
        encoded = enc.encode_value(val)
        decoded = enc.decode(encoded)
        status = "✓" if decoded == val else "✗"
        print(f"  {val:3d} -> encode -> decode -> {decoded:3d} {status}")


def visualize_adjacency():
    """Demonstrate adjacency encoding."""
    layer = AbacusLayer(num_bits=8, embed_dim=4, mode='adjacency')
    
    print("\n" + "=" * 70)
    print("ADJACENCY ENCODING: Topology for Binary")
    print("=" * 70)
    
    # Example: value 5 = 00000101
    val = 5
    bits = torch.tensor([[int(b) for b in f"{val:08b}"]], dtype=torch.float)
    
    print(f"\nValue: {val} = {val:08b}")
    print(f"Bits:  {bits[0].int().tolist()}")
    
    # Manual adjacency computation for demonstration
    bits_f = bits[0]
    left = F.pad(bits_f[1:], (0, 1), value=0)
    right = F.pad(bits_f[:-1], (1, 0), value=0)
    
    print(f"\nAdjacency view:")
    print(f"  Bit position: [7, 6, 5, 4, 3, 2, 1, 0]")
    print(f"  Bit value:    {bits_f.int().tolist()}")
    print(f"  Left (→out):  {left.int().tolist()}")
    print(f"  Right (←in):  {right.int().tolist()}")
    print(f"\n  Each bit now knows its neighbors for carry routing.")


if __name__ == "__main__":
    visualize_soroban()
    print("\n\n")
    visualize_adjacency()
