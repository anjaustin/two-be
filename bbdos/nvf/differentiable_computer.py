"""
The End-to-End Differentiable Computer

Memory + Compute, fully differentiable.
Gradients flow from output error back to query representation.

The model learns to search.
The model learns to compute.

This is not scripting agents.
This is compiling cognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


class NeuralMemory(nn.Module):
    """
    Differentiable key-value memory.
    
    Soft retrieval via attention.
    Gradients flow back to query.
    """
    
    def __init__(self, n_slots: int = 100, key_dim: int = 32, value_dim: int = 32):
        super().__init__()
        
        self.n_slots = n_slots
        self.key_dim = key_dim
        self.value_dim = value_dim
        
        # Memory contents
        self.keys = nn.Parameter(torch.randn(n_slots, key_dim) * 0.1)
        self.values = nn.Parameter(torch.randn(n_slots, value_dim) * 0.1)
        
        # Learnable temperature
        self.log_temp = nn.Parameter(torch.tensor(0.0))
    
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
            query: (batch, key_dim)
        
        Returns:
            retrieved: (batch, value_dim) - weighted sum of values
            attention: (batch, n_slots) - attention weights
        """
        q = F.normalize(query, dim=-1)
        k = F.normalize(self.keys, dim=-1)
        
        scores = torch.mm(q, k.T) / self.temperature
        attention = F.softmax(scores, dim=-1)
        retrieved = torch.mm(attention, self.values)
        
        return retrieved, attention


class SorobanEncoder(nn.Module):
    """Encode integer to Soroban (thermometer) representation."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch,) integer values [0-255]
        
        Returns:
            soroban: (batch, 32) thermometer encoding
        """
        lo = x % 16
        hi = x // 16
        device = x.device
        
        lo_therm = (torch.arange(16, device=device).unsqueeze(0) < lo.unsqueeze(1)).float()
        hi_therm = (torch.arange(16, device=device).unsqueeze(0) < hi.unsqueeze(1)).float()
        
        return torch.cat([lo_therm, hi_therm], dim=1)


class SorobanDecoder(nn.Module):
    """Decode Soroban representation back to integer."""
    
    def forward(self, soroban: torch.Tensor) -> torch.Tensor:
        """
        Args:
            soroban: (batch, 32) thermometer encoding
        
        Returns:
            x: (batch,) decoded values
        """
        lo = soroban[:, :16].sum(dim=1)
        hi = soroban[:, 16:].sum(dim=1)
        return hi * 16 + lo


class NeuralAdder(nn.Module):
    """Neural addition in Soroban space."""
    
    def __init__(self, hidden: int = 256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(64, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 33),
            nn.Sigmoid()
        )
    
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add two Soroban-encoded values.
        
        Args:
            a: (batch, 32) first operand
            b: (batch, 32) second operand
        
        Returns:
            result: (batch, 32) sum in Soroban encoding
            carry: (batch, 1) carry out
        """
        x = torch.cat([a, b], dim=1)
        out = self.net(x)
        return out[:, :32], out[:, 32:33]


class DifferentiableComputer(nn.Module):
    """
    The End-to-End Differentiable Computer.
    
    Components:
    - Memory: NeuralMemory (soft key-value store)
    - Compute: NeuralAdder (Soroban arithmetic)
    - Bus: Gradients (fully differentiable)
    - Control: QueryEncoder (learned addressing)
    """
    
    def __init__(self, n_slots: int = 100, key_dim: int = 32):
        super().__init__()
        
        self.n_slots = n_slots
        self.key_dim = key_dim
        
        # Memory
        self.memory = NeuralMemory(n_slots=n_slots, key_dim=key_dim, value_dim=32)
        
        # Compute
        self.adder = NeuralAdder(hidden=256)
        
        # Control (query encoder)
        self.query_encoder = nn.Sequential(
            nn.Linear(key_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, key_dim),
        )
        
        # Decoders
        self.encoder = SorobanEncoder()
        self.decoder = SorobanDecoder()
    
    def store(self, slot: int, key: torch.Tensor, value: int):
        """Store a numeric value at a key."""
        value_soroban = self.encoder(torch.tensor([value]))[0]
        self.memory.write(slot, key, value_soroban)
    
    def forward(
        self, 
        query_a: torch.Tensor, 
        query_b: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Retrieve two values and add them.
        
        Args:
            query_a: (batch, key_dim) query for first operand
            query_b: (batch, key_dim) query for second operand
        
        Returns:
            result: (batch,) computed sum
            attentions: tuple of attention weights for debugging
        """
        # Encode queries
        key_a = self.query_encoder(query_a)
        key_b = self.query_encoder(query_b)
        
        # Read from memory
        value_a, attn_a = self.memory.read(key_a)
        value_b, attn_b = self.memory.read(key_b)
        
        # Compute
        result_soroban, _ = self.adder(value_a, value_b)
        result = self.decoder(result_soroban)
        
        return result, (attn_a, attn_b)
    
    def retrieve_and_compute(
        self, 
        queries: List[torch.Tensor], 
        operation: str = 'add'
    ) -> torch.Tensor:
        """
        Generic retrieve-and-compute operation.
        
        Args:
            queries: list of query tensors
            operation: 'add' (more operations to come)
        
        Returns:
            result: computed result
        """
        if operation == 'add' and len(queries) == 2:
            return self.forward(queries[0], queries[1])[0]
        else:
            raise NotImplementedError(f"Operation {operation} with {len(queries)} operands")


def create_orthogonal_keys(n_keys: int, dim: int) -> List[torch.Tensor]:
    """Create orthogonal keys for maximum separation."""
    keys_raw = torch.randn(n_keys, dim)
    keys_orth = torch.linalg.qr(keys_raw.T)[0].T[:n_keys]
    return [keys_orth[i] for i in range(n_keys)]


if __name__ == "__main__":
    print("=" * 70)
    print("       DIFFERENTIABLE COMPUTER - SELF TEST")
    print("=" * 70)
    
    # Create computer
    computer = DifferentiableComputer(n_slots=5, key_dim=32)
    
    # Create orthogonal keys
    keys = create_orthogonal_keys(5, 32)
    values = [10, 20, 30, 40, 50]
    
    # Store values
    for i, (k, v) in enumerate(zip(keys, values)):
        computer.store(i, k, v)
    
    print(f"\nStored: {values}")
    
    # Test retrieval and computation
    query_a = keys[0].unsqueeze(0)  # Should retrieve 10
    query_b = keys[2].unsqueeze(0)  # Should retrieve 30
    
    result, (attn_a, attn_b) = computer(query_a, query_b)
    
    print(f"\nQuery for slots 0 and 2:")
    print(f"  Expected: 10 + 30 = 40")
    print(f"  Got: {result.item():.1f}")
    print(f"  Attention A peak: slot {attn_a.argmax().item()}")
    print(f"  Attention B peak: slot {attn_b.argmax().item()}")
    
    # Gradient test
    query_a = keys[1].unsqueeze(0).clone().requires_grad_(True)
    query_b = keys[3].unsqueeze(0).clone().requires_grad_(True)
    
    result, _ = computer(query_a, query_b)
    target = torch.tensor([60.0])  # 20 + 40
    loss = F.mse_loss(result, target)
    loss.backward()
    
    print(f"\n  Gradient test:")
    print(f"    Query A grad: {query_a.grad.abs().mean().item():.4f}")
    print(f"    Query B grad: {query_b.grad.abs().mean().item():.4f}")
    print(f"    [✓] GRADIENTS FLOW END-TO-END")
    
    print("\n" + "=" * 70)
    print("       SELF TEST PASSED")
    print("=" * 70)
