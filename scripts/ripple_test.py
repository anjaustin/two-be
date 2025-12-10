#!/usr/bin/env python3
"""
RIPPLE TEST: The "Hard Carry" Stress Test

Tests ONLY the hardest cases: X + 1 where X is a full bit-mask.
- 1 + 1 = 2 (1 bit flip)
- 3 + 1 = 4 (2 bit flip)
- 7 + 1 = 8 (3 bit flip)
- 15 + 1 = 16 (4 bit flip - nibble boundary!)
- 31 + 1 = 32 (5 bit flip)
- 63 + 1 = 64 (6 bit flip)
- 127 + 1 = 128 (7 bit flip)
- 255 + 1 = 0 (8 bit flip - full rollover!)

Hypothesis: Soroban will handle these uniformly well because
they're all "column overflow" events. Binary will struggle more
as the cascade length increases.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bbdos.cpu.abacus import SorobanEncoder


class BinaryEncoder:
    def __init__(self, bits=8):
        self.bits = bits
        self.dim = bits

    def encode(self, val):
        return torch.tensor([(val >> i) & 1 for i in range(self.bits)], dtype=torch.float32)

    def encode_batch(self, vals):
        batch = vals.shape[0]
        result = torch.zeros(batch, self.bits)
        for i in range(self.bits):
            result[:, i] = ((vals >> i) & 1).float()
        return result

    def decode(self, tensor):
        val = 0
        for i in range(self.bits):
            if tensor[i] > 0.5:
                val |= (1 << i)
        return val

    def decode_batch(self, tensor):
        active = (tensor > 0.5).long()
        result = torch.zeros(tensor.shape[0], dtype=torch.long)
        for i in range(self.bits):
            result |= (active[:, i] << i)
        return result


class AdderNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# The hard cases: numbers where adding 1 causes maximum ripple
MASKS = [1, 3, 7, 15, 31, 63, 127, 255]  # 2^n - 1
MASK_NAMES = ["1→2", "3→4", "7→8", "15→16", "31→32", "63→64", "127→128", "255→0"]


def get_hard_batch(batch_size=64):
    """Generate only hard carry cases: (2^n - 1) + 1."""
    indices = torch.randint(0, len(MASKS), (batch_size,))
    a = torch.tensor([MASKS[i] for i in indices])
    b = torch.ones(batch_size, dtype=torch.long)
    c = (a + b) % 256
    return a, b, c


def get_specific_case_batch(mask_idx, batch_size=64):
    """Generate batch of a specific hard case."""
    a = torch.full((batch_size,), MASKS[mask_idx], dtype=torch.long)
    b = torch.ones(batch_size, dtype=torch.long)
    c = (a + b) % 256
    return a, b, c


def run_ripple_test(num_steps=1000, device='cuda'):
    """Run the ripple stress test."""
    
    print("=" * 70)
    print("RIPPLE TEST: Hard Carry Stress Test")
    print("=" * 70)
    print("Testing ONLY cascade carries: (2^n - 1) + 1")
    print()
    print("Cases: 1+1, 3+1, 7+1, 15+1, 31+1, 63+1, 127+1, 255+1")
    print("       (1-bit flip → 8-bit flip)")
    print()
    print("Hypothesis: Soroban sees these as the SAME operation (overflow).")
    print("           Binary sees them as increasingly complex XOR patterns.")
    print("=" * 70)
    print()
    
    # Setup models
    enc_bin = BinaryEncoder(8)
    model_bin = AdderNet(16, 8, hidden_dim=128).to(device)
    opt_bin = optim.Adam(model_bin.parameters(), lr=0.005)
    
    enc_sor = SorobanEncoder(embed_dim=16)
    model_sor = AdderNet(64, 32, hidden_dim=128).to(device)
    opt_sor = optim.Adam(model_sor.parameters(), lr=0.005)
    
    print("Training Phase (mixed hard cases)...")
    print("-" * 70)
    print(f"{'Step':<6} | {'Binary':<20} | {'Soroban':<20} | Winner")
    print("-" * 70)
    
    for step in range(1, num_steps + 1):
        a_ints, b_ints, c_ints = get_hard_batch(64)
        
        # Train Binary
        a_bin = enc_bin.encode_batch(a_ints)
        b_bin = enc_bin.encode_batch(b_ints)
        x_bin = torch.cat([a_bin, b_bin], dim=1).to(device)
        y_bin = enc_bin.encode_batch(c_ints).to(device)
        
        opt_bin.zero_grad()
        pred_bin = model_bin(x_bin)
        loss_bin = nn.MSELoss()(pred_bin, y_bin)
        loss_bin.backward()
        opt_bin.step()
        
        # Train Soroban
        a_sor = enc_sor.encode_batch(a_ints)
        b_sor = enc_sor.encode_batch(b_ints)
        x_sor = torch.cat([a_sor, b_sor], dim=1).to(device)
        y_sor = enc_sor.encode_batch(c_ints).to(device)
        
        opt_sor.zero_grad()
        pred_sor = model_sor(x_sor)
        loss_sor = nn.MSELoss()(pred_sor, y_sor)
        loss_sor.backward()
        opt_sor.step()
        
        if step % 100 == 0:
            # Evaluate
            with torch.no_grad():
                decoded_bin = enc_bin.decode_batch(pred_bin.cpu())
                decoded_sor = enc_sor.decode(pred_sor.cpu())
                
                acc_bin = (decoded_bin == c_ints).float().mean().item() * 100
                acc_sor = (decoded_sor == c_ints).float().mean().item() * 100
            
            bar_bin = "█" * int(acc_bin / 5)
            bar_sor = "█" * int(acc_sor / 5)
            
            if acc_sor > acc_bin + 10:
                winner = "◀ SOR"
            elif acc_bin > acc_sor + 10:
                winner = "BIN ▶"
            else:
                winner = "  =  "
            
            print(f"{step:<6} | {acc_bin:5.1f}% {bar_bin:<13} | {acc_sor:5.1f}% {bar_sor:<13} | {winner}")
    
    # Per-case breakdown
    print()
    print("=" * 70)
    print("PER-CASE BREAKDOWN (after training)")
    print("=" * 70)
    print(f"{'Case':<12} | {'Bits':<6} | {'Binary':<10} | {'Soroban':<10} | Winner")
    print("-" * 70)
    
    model_bin.eval()
    model_sor.eval()
    
    with torch.no_grad():
        for idx, (mask, name) in enumerate(zip(MASKS, MASK_NAMES)):
            a_ints, b_ints, c_ints = get_specific_case_batch(idx, batch_size=100)
            
            # Binary
            a_bin = enc_bin.encode_batch(a_ints)
            b_bin = enc_bin.encode_batch(b_ints)
            x_bin = torch.cat([a_bin, b_bin], dim=1).to(device)
            pred_bin = model_bin(x_bin)
            decoded_bin = enc_bin.decode_batch(pred_bin.cpu())
            acc_bin = (decoded_bin == c_ints).float().mean().item() * 100
            
            # Soroban
            a_sor = enc_sor.encode_batch(a_ints)
            b_sor = enc_sor.encode_batch(b_ints)
            x_sor = torch.cat([a_sor, b_sor], dim=1).to(device)
            pred_sor = model_sor(x_sor)
            decoded_sor = enc_sor.decode(pred_sor.cpu())
            acc_sor = (decoded_sor == c_ints).float().mean().item() * 100
            
            bits_flipped = idx + 1
            
            if acc_sor > acc_bin + 10:
                winner = "◀ SOR"
            elif acc_bin > acc_sor + 10:
                winner = "BIN ▶"
            else:
                winner = "  =  "
            
            print(f"{name:<12} | {bits_flipped:<6} | {acc_bin:6.1f}%    | {acc_sor:6.1f}%    | {winner}")
    
    print("=" * 70)
    print()
    print("KEY INSIGHT:")
    print("  If Soroban accuracy is UNIFORM across cases, it sees them as")
    print("  the same operation (column overflow). If Binary accuracy DROPS")
    print("  as bits increase, it's struggling with longer XOR cascades.")
    print("=" * 70)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    run_ripple_test(num_steps=1000, device=device)


if __name__ == "__main__":
    main()
