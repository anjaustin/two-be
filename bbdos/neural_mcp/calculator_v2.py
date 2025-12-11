"""
Neural Calculator MCP v2 - Improved Encodings

Fixes:
1. Signed encoding for subtraction (sign bit + magnitude)
2. Separate models per operation (organelles approach)
3. Focus on addition first, prove the concept
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import Tuple, List
from dataclasses import dataclass
from enum import IntEnum


class Op(IntEnum):
    ADD = 0
    SUB = 1
    MUL = 2
    DIV = 3


# ============================================================
# SOROBAN ENCODING (Improved)
# ============================================================

def soroban_encode_uint8(x: int) -> torch.Tensor:
    """Encode 8-bit unsigned integer as nibble-split thermometer."""
    x = max(0, min(255, int(x)))
    
    low = x & 0x0F
    high = (x >> 4) & 0x0F
    
    low_therm = torch.zeros(16)
    high_therm = torch.zeros(16)
    
    for i in range(16):
        low_therm[i] = 1.0 if low > i else 0.0
        high_therm[i] = 1.0 if high > i else 0.0
    
    return torch.cat([low_therm, high_therm])


def soroban_decode_uint8(encoded: torch.Tensor) -> int:
    """Decode nibble-split thermometer to 8-bit unsigned."""
    low_therm = encoded[:16]
    high_therm = encoded[16:32]
    
    low = int((low_therm > 0.5).sum().item())
    high = int((high_therm > 0.5).sum().item())
    
    low = min(15, low)
    high = min(15, high)
    
    return (high << 4) | low


def soroban_encode_uint16(x: int) -> torch.Tensor:
    """Encode 16-bit unsigned as 4 nibbles."""
    x = max(0, min(65535, int(x)))
    
    nibbles = [(x >> (4 * i)) & 0x0F for i in range(4)]
    
    therms = []
    for n in nibbles:
        therm = torch.zeros(16)
        for i in range(16):
            therm[i] = 1.0 if n > i else 0.0
        therms.append(therm)
    
    return torch.cat(therms)


def soroban_decode_uint16(encoded: torch.Tensor) -> int:
    """Decode 4-nibble thermometer to 16-bit unsigned."""
    value = 0
    for i in range(4):
        therm = encoded[i*16 : (i+1)*16]
        nibble = int((therm > 0.5).sum().item())
        nibble = min(15, nibble)
        value |= (nibble << (4 * i))
    return value


# ============================================================
# ADDITION ORGANELLE (Proven to work)
# ============================================================

class AdditionOrganelle(nn.Module):
    """
    Neural ADDition - the proven case.
    
    Input: Soroban(a) + Soroban(b) = 32 + 32 = 64 features
    Output: Soroban(result) = 64 features (16-bit to handle overflow)
    """
    
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Sigmoid()
        )
    
    def forward(self, a_enc: torch.Tensor, b_enc: torch.Tensor) -> torch.Tensor:
        x = torch.cat([a_enc, b_enc], dim=-1)
        return self.net(x)


# ============================================================
# TRAINING - Addition Only (Proof of Concept)
# ============================================================

def train_addition_organelle(n_samples: int = 500000, 
                             n_epochs: int = 30,
                             batch_size: int = 1024,
                             lr: float = 0.005):
    """Train the addition organelle to perfection."""
    
    print("=" * 70)
    print("       ADDITION ORGANELLE - Training")
    print("=" * 70)
    
    # Generate ALL possible 8-bit additions (exhaustive)
    print("\n[1] Generating exhaustive 8-bit addition dataset...")
    
    X_a = []
    X_b = []
    Y = []
    
    for a in range(256):
        for b in range(256):
            result = a + b  # 0-510
            X_a.append(soroban_encode_uint8(a))
            X_b.append(soroban_encode_uint8(b))
            Y.append(soroban_encode_uint16(result))
    
    X_a = torch.stack(X_a)
    X_b = torch.stack(X_b)
    Y = torch.stack(Y)
    
    print(f"    Dataset size: {len(Y):,} (all 256×256 combinations)")
    print(f"    Input shape: 2 × {X_a.shape[1]} = 64 features")
    print(f"    Output shape: {Y.shape[1]} features")
    
    # Create model
    print("\n[2] Creating Addition Organelle...")
    model = AdditionOrganelle()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {n_params:,}")
    
    # Training
    print(f"\n[3] Training for {n_epochs} epochs...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    n_batches = len(Y) // batch_size
    
    for epoch in range(n_epochs):
        # Shuffle
        perm = torch.randperm(len(Y))
        X_a_shuf = X_a[perm]
        X_b_shuf = X_b[perm]
        Y_shuf = Y[perm]
        
        epoch_loss = 0.0
        for i in range(n_batches):
            batch_a = X_a_shuf[i*batch_size : (i+1)*batch_size]
            batch_b = X_b_shuf[i*batch_size : (i+1)*batch_size]
            batch_y = Y_shuf[i*batch_size : (i+1)*batch_size]
            
            optimizer.zero_grad()
            pred = model(batch_a, batch_b)
            loss = F.mse_loss(pred, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / n_batches
        
        # Evaluate on full dataset
        with torch.no_grad():
            pred = model(X_a, X_b)
            pred_vals = torch.tensor([soroban_decode_uint16(p) for p in pred])
            true_vals = torch.tensor([soroban_decode_uint16(y) for y in Y])
            
            exact = (pred_vals == true_vals).float().mean().item() * 100
        
        print(f"    Epoch {epoch+1:3d}: loss={avg_loss:.6f}, accuracy={exact:.2f}%")
        
        if exact >= 100.0:
            print(f"\n    [!] PERFECT ACCURACY ACHIEVED at epoch {epoch+1}")
            break
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("       FINAL EVALUATION")
    print("=" * 70)
    
    with torch.no_grad():
        pred = model(X_a, X_b)
        
        errors = []
        for i in range(len(Y)):
            pred_val = soroban_decode_uint16(pred[i])
            true_val = soroban_decode_uint16(Y[i])
            if pred_val != true_val:
                a = soroban_decode_uint8(X_a[i])
                b = soroban_decode_uint8(X_b[i])
                errors.append((a, b, pred_val, true_val))
        
        n_errors = len(errors)
        accuracy = (1 - n_errors / len(Y)) * 100
        
        print(f"\n    Total samples: {len(Y):,}")
        print(f"    Errors: {n_errors}")
        print(f"    Accuracy: {accuracy:.4f}%")
        
        if n_errors > 0 and n_errors <= 20:
            print("\n    Error details:")
            for a, b, pred, true in errors[:20]:
                print(f"      {a} + {b} = {pred} (expected {true})")
    
    return model


# ============================================================
# LATENCY TEST
# ============================================================

def benchmark_latency(model: AdditionOrganelle):
    """Compare neural vs ground truth latency."""
    
    print("\n" + "=" * 70)
    print("       LATENCY BENCHMARK")
    print("=" * 70)
    
    # Test data
    n_test = 10000
    a_vals = np.random.randint(0, 256, n_test)
    b_vals = np.random.randint(0, 256, n_test)
    
    # Ground truth (Python)
    start = time.perf_counter()
    for a, b in zip(a_vals, b_vals):
        _ = a + b
    gt_time = time.perf_counter() - start
    
    # Neural (sequential)
    model.eval()
    start = time.perf_counter()
    with torch.no_grad():
        for a, b in zip(a_vals, b_vals):
            a_enc = soroban_encode_uint8(a).unsqueeze(0)
            b_enc = soroban_encode_uint8(b).unsqueeze(0)
            _ = model(a_enc, b_enc)
    neural_seq_time = time.perf_counter() - start
    
    # Neural (batched)
    a_enc = torch.stack([soroban_encode_uint8(a) for a in a_vals])
    b_enc = torch.stack([soroban_encode_uint8(b) for b in b_vals])
    
    start = time.perf_counter()
    with torch.no_grad():
        _ = model(a_enc, b_enc)
    neural_batch_time = time.perf_counter() - start
    
    print(f"\n    {n_test:,} operations:")
    print(f"    Ground truth (Python +):  {gt_time*1000:.2f} ms ({n_test/gt_time:.0f} ops/sec)")
    print(f"    Neural (sequential):      {neural_seq_time*1000:.2f} ms ({n_test/neural_seq_time:.0f} ops/sec)")
    print(f"    Neural (batched):         {neural_batch_time*1000:.2f} ms ({n_test/neural_batch_time:.0f} ops/sec)")
    print(f"\n    Batched speedup vs sequential: {neural_seq_time/neural_batch_time:.0f}x")
    print(f"    Batched ops/sec: {n_test/neural_batch_time:,.0f}")


# ============================================================
# DIFFERENTIABLE DEMO
# ============================================================

def demo_differentiable():
    """Show that gradients flow through the neural calculator."""
    
    print("\n" + "=" * 70)
    print("       DIFFERENTIABLE DEMO")
    print("=" * 70)
    
    # Load or train model
    model = AdditionOrganelle()
    try:
        model.load_state_dict(torch.load('/workspace/two-be/checkpoints/swarm/addition_organelle.pt'))
        print("\n    [+] Loaded pretrained model")
    except:
        print("\n    [!] Training model first...")
        model = train_addition_organelle(n_epochs=20)
    
    # Create inputs that require grad
    a = 50
    b = 75
    
    a_enc = soroban_encode_uint8(a).unsqueeze(0).requires_grad_(True)
    b_enc = soroban_encode_uint8(b).unsqueeze(0).requires_grad_(True)
    
    # Forward pass
    result_enc = model(a_enc, b_enc)
    
    # Define a "loss" - say we want the result to be higher
    loss = -result_enc.sum()  # Maximize result
    
    # Backward pass
    loss.backward()
    
    print(f"\n    Input: {a} + {b}")
    print(f"    Result: {soroban_decode_uint16(result_enc.squeeze())}")
    print(f"\n    Gradient w.r.t. a: {a_enc.grad.abs().sum().item():.4f}")
    print(f"    Gradient w.r.t. b: {b_enc.grad.abs().sum().item():.4f}")
    print(f"\n    [✓] GRADIENTS FLOW THROUGH THE CALCULATOR")
    
    # Show that gradient points to increasing inputs
    print(f"\n    Interpretation:")
    print(f"    To increase the sum, the gradient suggests adjusting the encodings")
    print(f"    This is the 'Gradient Superhighway' - direct optimization of tool use")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # Train
    model = train_addition_organelle(n_epochs=50)
    
    # Save
    torch.save(model.state_dict(), '/workspace/two-be/checkpoints/swarm/addition_organelle.pt')
    print("\n[+] Model saved to checkpoints/swarm/addition_organelle.pt")
    
    # Benchmark
    benchmark_latency(model)
    
    # Demo differentiability
    demo_differentiable()
    
    print("\n" + "=" * 70)
    print("       ADDITION ORGANELLE - OPERATIONAL")
    print("=" * 70)
