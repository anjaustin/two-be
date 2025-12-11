"""
Test the Real TriX Kernel

Using the actual TriXLinear from bbdos/kernel/bindings.py
No custom QAT. Just the kernel as designed.

Goal: Understand if TriX can learn the retrieve-and-add task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from bbdos.kernel import TriXLinear, is_neon_available
from bbdos.cpu.abacus import SorobanEncoder


def main():
    print("=" * 70)
    print("       TEST: REAL TriX KERNEL")
    print("=" * 70)
    
    print(f"\nNEON available: {is_neon_available()}")
    
    # Step 1: Test basic TriXLinear functionality
    print("\n" + "-" * 70)
    print("[1] Basic TriXLinear Test")
    print("-" * 70)
    
    layer = TriXLinear(in_features=32, out_features=64, num_tiles=4)
    
    print(f"    Weight shape: {layer.weight.shape}")
    print(f"    Weight unique values: {layer.weight.unique().tolist()}")
    print(f"    Sparsity (zeros): {(layer.weight == 0).float().mean().item():.1%}")
    
    # Test forward
    x = torch.randn(8, 32)
    gate = torch.zeros(8, 4)
    gate[:, 0] = 1  # Activate tile 0
    
    y = layer(x, gate)
    print(f"    Input shape: {x.shape}")
    print(f"    Output shape: {y.shape}")
    
    # Step 2: Test gradient flow
    print("\n" + "-" * 70)
    print("[2] Gradient Flow Test")
    print("-" * 70)
    
    x = torch.randn(8, 32, requires_grad=True)
    gate = torch.zeros(8, 4)
    gate[:, 0] = 1
    
    y = layer(x, gate)
    loss = y.sum()
    loss.backward()
    
    print(f"    Input gradient: {x.grad.abs().mean().item():.6f}")
    print(f"    Weight gradient: {layer.weight.grad.abs().mean().item():.6f}")
    
    if layer.weight.grad.abs().mean() > 0:
        print("    [✓] Gradients flow to weights")
    else:
        print("    [✗] No gradient to weights")
    
    # Step 3: Test if it can learn a simple pattern
    print("\n" + "-" * 70)
    print("[3] Simple Learning Test")
    print("-" * 70)
    
    # Simple task: map input to specific output
    torch.manual_seed(42)
    
    layer = TriXLinear(32, 16, num_tiles=4)
    target_output = torch.randn(16)  # Fixed target
    
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
    
    print("    Task: Make layer output match a fixed target")
    print("    Training for 100 steps...")
    
    for step in range(100):
        x = torch.randn(1, 32)
        gate = torch.ones(1, 4)  # All tiles active
        
        optimizer.zero_grad()
        y = layer(x, gate)
        loss = F.mse_loss(y.squeeze(), target_output)
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 25 == 0:
            print(f"    Step {step+1}: loss = {loss.item():.4f}")
    
    # Step 4: The actual test - retrieve and add
    print("\n" + "-" * 70)
    print("[4] Retrieve and Add Test")
    print("-" * 70)
    
    # Architecture that matches how 6502 model uses TriX:
    # - Attention for memory retrieval
    # - TriX only in FFN (feed-forward network)
    
    class GatedTriXFFN(nn.Module):
        """TriX FFN like in the working 6502 model."""
        def __init__(self, d_in, d_hidden, d_out, num_tiles=4):
            super().__init__()
            self.up = TriXLinear(d_in, d_hidden, num_tiles)
            self.down = TriXLinear(d_hidden, d_out, num_tiles)
            self.gate_proj = nn.Linear(d_in, num_tiles)
        
        def forward(self, x):
            gate = F.softmax(self.gate_proj(x), dim=-1)
            h = F.relu(self.up(x, gate))
            return self.down(h, gate)
    
    class TriXComputer(nn.Module):
        def __init__(self):
            super().__init__()
            
            # Memory
            self.keys = nn.Parameter(torch.randn(5, 32) * 0.1)
            self.values = nn.Parameter(torch.randn(5, 32) * 0.1)
            
            # Query encoder - standard linear (like attention Q projection)
            self.q_proj = nn.Linear(32, 32)
            
            # Adder - TriX FFN (like in 6502 model)
            self.add_ffn = GatedTriXFFN(64, 256, 32, num_tiles=4)
            
            self.soroban = SorobanEncoder()
        
        def store(self, slot, key, value):
            with torch.no_grad():
                self.keys.data[slot] = key
                self.values.data[slot] = self.soroban.encode_batch(
                    torch.tensor([value])
                )[0]
        
        def forward(self, qa, qb):
            # Project queries
            ka = self.q_proj(qa)
            kb = self.q_proj(qb)
            
            # Soft attention over memory
            ka = F.normalize(ka, dim=-1)
            kb = F.normalize(kb, dim=-1)
            k = F.normalize(self.keys, dim=-1)
            
            attn_a = F.softmax(torch.mm(ka, k.T) * 10, dim=-1)
            attn_b = F.softmax(torch.mm(kb, k.T) * 10, dim=-1)
            
            va = torch.mm(attn_a, self.values)
            vb = torch.mm(attn_b, self.values)
            
            # Add via TriX FFN
            x = torch.cat([va, vb], dim=1)
            result = torch.sigmoid(self.add_ffn(x))
            
            # Decode
            low = result[:, :16].sum(dim=1)
            high = result[:, 16:].sum(dim=1)
            return high * 16 + low
    
    SimpleTriXComputer = TriXComputer  # Use new architecture
    
    # Create and test
    torch.manual_seed(42)
    computer = SimpleTriXComputer()
    
    # Orthogonal keys
    keys_raw = torch.randn(5, 32)
    keys = torch.linalg.qr(keys_raw.T)[0].T[:5]
    values = [10, 20, 30, 40, 50]
    
    for i in range(5):
        computer.store(i, keys[i], values[i])
    
    def make_batch(bs, noise=0.15):
        idx = torch.randint(0, 5, (bs, 2))
        qa = torch.stack([keys[i] for i in idx[:, 0]]) + torch.randn(bs, 32) * noise
        qb = torch.stack([keys[i] for i in idx[:, 1]]) + torch.randn(bs, 32) * noise
        tgt = torch.tensor([values[i] + values[j] for i, j in idx], dtype=torch.float)
        return qa, qb, tgt
    
    # Training - more capacity
    optimizer = torch.optim.Adam(computer.parameters(), lr=0.003)
    
    print(f"\n    Parameters: {sum(p.numel() for p in computer.parameters()):,}")
    print("    Training for 1000 epochs...")
    
    for epoch in range(1000):
        qa, qb, tgt = make_batch(512)
        
        optimizer.zero_grad()
        result = computer(qa, qb)
        loss = F.mse_loss(result, tgt)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            with torch.no_grad():
                qa, qb, tgt = make_batch(1000)
                result = computer(qa, qb)
                err = (result - tgt).abs()
                acc = (err < 5).float().mean().item() * 100
            print(f"    Epoch {epoch+1}: loss={loss.item():.2f}, acc(±5)={acc:.1f}%")
    
    # Final evaluation
    print("\n" + "-" * 70)
    print("[5] Final Results")
    print("-" * 70)
    
    with torch.no_grad():
        for noise in [0.10, 0.15, 0.20]:
            qa, qb, tgt = make_batch(2000, noise)
            result = computer(qa, qb)
            err = (result - tgt).abs()
            exact = (err < 1).float().mean().item() * 100
            close = (err < 5).float().mean().item() * 100
            print(f"    Noise={noise:.2f}: Exact={exact:.1f}%, ±5={close:.1f}%")
    
    print("\n" + "=" * 70)
    print("       TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
