"""
TriX Differentiable Computer

End-to-end differentiable computer using TriX architecture.
Memory + Compute with quantization-aware training.

Components:
- NeuralMemory: Soft key-value retrieval (NVF)
- NeuralAdder: Soroban arithmetic
- TriXLinearQAT: All linear layers use ternary QAT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
from dataclasses import dataclass

from ..trix.qat import TriXLinearQAT, QATTrainer, progressive_quantization_schedule
from ..cpu.abacus import SorobanEncoder


@dataclass
class ComputerConfig:
    """Configuration for the Differentiable Computer."""
    n_memory_slots: int = 5
    key_dim: int = 32
    value_dim: int = 32
    num_tiles: int = 4
    hidden_dim: int = 256
    quant_mode: str = 'progressive'


class SorobanCodec:
    """
    Differentiable Soroban encoding/decoding.
    
    Encoding: Integer → 32-bit thermometer (two 16-bit nibbles)
    Decoding: Soft sum of thermometer values
    """
    
    @staticmethod
    def encode(values: torch.Tensor) -> torch.Tensor:
        """
        Encode integers to Soroban thermometer representation.
        
        Args:
            values: [batch] integer values [0-255]
        
        Returns:
            [batch, 32] thermometer encoding
        """
        batch = values.shape[0]
        device = values.device
        result = torch.zeros(batch, 32, device=device)
        
        low = (values & 0x0F).long()
        high = ((values >> 4) & 0x0F).long()
        
        positions = torch.arange(16, device=device).unsqueeze(0)
        result[:, :16] = (positions < low.unsqueeze(1)).float()
        result[:, 16:] = (positions < high.unsqueeze(1)).float()
        
        return result
    
    @staticmethod
    def decode_soft(soroban: torch.Tensor) -> torch.Tensor:
        """
        Differentiable decode: soft sum of thermometer.
        
        Args:
            soroban: [batch, 32] thermometer values (soft, 0-1)
        
        Returns:
            [batch] decoded values (continuous)
        """
        low = soroban[:, :16].sum(dim=1)
        high = soroban[:, 16:].sum(dim=1)
        return high * 16 + low


class NeuralMemory(nn.Module):
    """
    Differentiable key-value memory with soft attention.
    
    Query transformation uses TriX layers.
    Retrieval is soft attention over memory slots.
    """
    
    def __init__(self, config: ComputerConfig):
        super().__init__()
        
        self.n_slots = config.n_memory_slots
        self.key_dim = config.key_dim
        self.value_dim = config.value_dim
        
        # Memory storage
        self.keys = nn.Parameter(torch.randn(config.n_memory_slots, config.key_dim) * 0.1)
        self.values = nn.Parameter(torch.randn(config.n_memory_slots, config.value_dim) * 0.1)
        
        # Learnable temperature
        self.log_temp = nn.Parameter(torch.tensor(0.0))
        
        # Query projection (TriX)
        self.q_proj = TriXLinearQAT(
            config.key_dim, config.key_dim,
            num_tiles=config.num_tiles,
            quant_mode=config.quant_mode,
        )
    
    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temp).clamp(min=0.01, max=10.0)
    
    def write(self, slot: int, key: torch.Tensor, value: torch.Tensor):
        """Write a key-value pair to a specific slot."""
        with torch.no_grad():
            self.keys.data[slot] = key
            self.values.data[slot] = value
    
    def read(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Soft read from memory.
        
        Args:
            query: [batch, key_dim] query vectors
        
        Returns:
            retrieved: [batch, value_dim] weighted sum of values
            attention: [batch, n_slots] attention weights
        """
        # Transform query
        q = self.q_proj(query)
        q = F.normalize(q, dim=-1)
        k = F.normalize(self.keys, dim=-1)
        
        # Attention
        scores = torch.mm(q, k.T) / self.temperature
        attention = F.softmax(scores, dim=-1)
        
        # Retrieve
        retrieved = torch.mm(attention, self.values)
        
        return retrieved, attention


class NeuralAdder(nn.Module):
    """
    Neural addition in Soroban space.
    
    All layers use TriX QAT.
    """
    
    def __init__(self, config: ComputerConfig):
        super().__init__()
        
        # Input: 64 (two 32-bit Soroban vectors)
        # Output: 32 (result Soroban vector)
        self.l1 = TriXLinearQAT(64, config.hidden_dim, num_tiles=config.num_tiles, quant_mode=config.quant_mode)
        self.l2 = TriXLinearQAT(config.hidden_dim, config.hidden_dim, num_tiles=config.num_tiles, quant_mode=config.quant_mode)
        self.l3 = TriXLinearQAT(config.hidden_dim, config.hidden_dim // 2, num_tiles=config.num_tiles, quant_mode=config.quant_mode)
        self.l4 = TriXLinearQAT(config.hidden_dim // 2, 32, num_tiles=config.num_tiles, quant_mode=config.quant_mode)
    
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Add two Soroban-encoded values.
        
        Args:
            a: [batch, 32] first operand (Soroban)
            b: [batch, 32] second operand (Soroban)
        
        Returns:
            [batch, 32] result (Soroban, soft activations)
        """
        x = torch.cat([a, b], dim=1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return torch.sigmoid(self.l4(x))


class TriXDifferentiableComputer(nn.Module):
    """
    End-to-End Differentiable Computer using TriX architecture.
    
    Components:
    - Memory: NeuralMemory (soft key-value store)
    - Compute: NeuralAdder (Soroban arithmetic)
    - All linear layers: TriXLinearQAT
    
    Supports quantization-aware training for deployment
    on ternary hardware.
    """
    
    def __init__(self, config: Optional[ComputerConfig] = None):
        super().__init__()
        
        if config is None:
            config = ComputerConfig()
        self.config = config
        
        # Memory
        self.memory = NeuralMemory(config)
        
        # Compute
        self.adder = NeuralAdder(config)
        
        # Query encoder (TriX)
        self.qe1 = TriXLinearQAT(config.key_dim, 64, num_tiles=config.num_tiles, quant_mode=config.quant_mode)
        self.qe2 = TriXLinearQAT(64, config.key_dim, num_tiles=config.num_tiles, quant_mode=config.quant_mode)
    
    def store(self, slot: int, key: torch.Tensor, value: int):
        """Store a numeric value at a key."""
        value_soroban = SorobanCodec.encode(torch.tensor([value], device=key.device))[0]
        self.memory.write(slot, key, value_soroban)
    
    def forward(
        self,
        query_a: torch.Tensor,
        query_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Retrieve two values and add them.
        
        Args:
            query_a: [batch, key_dim] query for first operand
            query_b: [batch, key_dim] query for second operand
        
        Returns:
            result: [batch] computed sum (continuous)
            attentions: tuple of attention weights for debugging
        """
        # Encode queries
        ka = self.qe2(F.relu(self.qe1(query_a)))
        kb = self.qe2(F.relu(self.qe1(query_b)))
        
        # Read from memory
        va, attn_a = self.memory.read(ka)
        vb, attn_b = self.memory.read(kb)
        
        # Compute
        result_soroban = self.adder(va, vb)
        result = SorobanCodec.decode_soft(result_soroban)
        
        return result, (attn_a, attn_b)
    
    def set_quant_temperature(self, temp: float):
        """Set quantization temperature for all TriX layers."""
        for module in self.modules():
            if isinstance(module, TriXLinearQAT):
                module.set_quant_temperature(temp)
    
    def get_total_sparsity(self) -> float:
        """Get average sparsity across all TriX layers."""
        sparsities = []
        for module in self.modules():
            if isinstance(module, TriXLinearQAT):
                sparsities.append(module.get_sparsity())
        return sum(sparsities) / len(sparsities) if sparsities else 0.0
    
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_orthogonal_keys(n_keys: int, dim: int, device: torch.device = None) -> List[torch.Tensor]:
    """Create orthogonal keys for maximum separation."""
    keys_raw = torch.randn(max(n_keys, dim), dim, device=device)
    keys_orth = torch.linalg.qr(keys_raw.T)[0].T[:n_keys]
    return [keys_orth[i] for i in range(n_keys)]


if __name__ == "__main__":
    print("=" * 70)
    print("       TriX DIFFERENTIABLE COMPUTER - SELF TEST")
    print("=" * 70)
    
    # Create computer
    config = ComputerConfig(
        n_memory_slots=5,
        key_dim=32,
        num_tiles=4,
        quant_mode='progressive',
    )
    computer = TriXDifferentiableComputer(config)
    
    print(f"\nParameters: {computer.num_parameters:,}")
    
    # Create orthogonal keys
    keys = create_orthogonal_keys(5, 32)
    values = [10, 20, 30, 40, 50]
    
    # Store values
    for i, (k, v) in enumerate(zip(keys, values)):
        computer.store(i, k, v)
    
    print(f"Stored: {values}")
    
    # Test forward pass
    query_a = keys[0].unsqueeze(0)
    query_b = keys[2].unsqueeze(0)
    
    result, (attn_a, attn_b) = computer(query_a, query_b)
    
    print(f"\nQuery for slots 0 and 2:")
    print(f"  Expected: 10 + 30 = 40")
    print(f"  Got: {result.item():.1f}")
    print(f"  Attention A peak: slot {attn_a.argmax().item()}")
    print(f"  Attention B peak: slot {attn_b.argmax().item()}")
    
    # Test gradient flow
    query_a = keys[1].unsqueeze(0).clone().requires_grad_(True)
    query_b = keys[3].unsqueeze(0).clone().requires_grad_(True)
    
    result, _ = computer(query_a, query_b)
    target = torch.tensor([60.0])  # 20 + 40
    loss = F.mse_loss(result, target)
    loss.backward()
    
    print(f"\n  Gradient test:")
    print(f"    Query A grad: {query_a.grad.abs().mean().item():.6f}")
    print(f"    Query B grad: {query_b.grad.abs().mean().item():.6f}")
    
    if query_a.grad.abs().mean() > 0 and query_b.grad.abs().mean() > 0:
        print(f"    [✓] GRADIENTS FLOW END-TO-END")
    else:
        print(f"    [✗] GRADIENT FLOW BROKEN")
    
    print(f"\n  Sparsity: {computer.get_total_sparsity():.1%}")
    
    print("\n" + "=" * 70)
    print("       SELF TEST COMPLETE")
    print("=" * 70)
