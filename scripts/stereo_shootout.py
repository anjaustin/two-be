#!/usr/bin/env python3
"""
STEREO SHOOTOUT: Binary vs. Soroban

Two identical neural networks race to learn 8-bit addition.
- Left Channel: Binary encoding (8 bits per number)
- Right Channel: Soroban encoding (32 bits per number, split-byte thermometer)

Hypothesis: Soroban will converge faster because:
1. Magnitude is visible (full column = about to overflow)
2. Carry becomes state transition (Tetris line clear)
3. The model can use pattern recognition instead of XOR logic

Run on Jetson: python scripts/stereo_shootout.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bbdos.cpu.abacus import SorobanEncoder


class BinaryEncoder:
    """Standard binary encoding: 8 bits per byte."""
    
    def __init__(self, bits=8):
        self.bits = bits
        self.dim = bits

    def encode(self, val):
        """Convert int to binary tensor [b0, b1, ..., b7] (LSB first)."""
        return torch.tensor([(val >> i) & 1 for i in range(self.bits)], dtype=torch.float32)

    def encode_batch(self, vals):
        """Vectorized encoding."""
        batch = vals.shape[0]
        result = torch.zeros(batch, self.bits)
        for i in range(self.bits):
            result[:, i] = ((vals >> i) & 1).float()
        return result

    def decode(self, tensor):
        """Threshold and rebuild int."""
        val = 0
        for i in range(self.bits):
            if tensor[i] > 0.5:
                val |= (1 << i)
        return val

    def decode_batch(self, tensor):
        """Vectorized decoding."""
        active = (tensor > 0.5).long()
        result = torch.zeros(tensor.shape[0], dtype=torch.long)
        for i in range(self.bits):
            result |= (active[:, i] << i)
        return result


class AdderNet(nn.Module):
    """Simple MLP for learning addition. Identical architecture for both channels."""
    
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # Output probabilities for each bit/bead
        )

    def forward(self, x):
        return self.net(x)


def get_batch(batch_size=64):
    """Generate random A + B = C (mod 256)."""
    a = torch.randint(0, 256, (batch_size,))
    b = torch.randint(0, 256, (batch_size,))
    c = (a + b) % 256  # 8-bit wraparound
    return a, b, c


def run_shootout(num_steps=3000, batch_size=64, device='cuda'):
    """Run the stereo comparison."""
    
    print("=" * 70)
    print("STEREO SHOOTOUT: Binary vs. Soroban")
    print("=" * 70)
    print(f"Task: Learn 8-bit addition (A + B = C mod 256)")
    print(f"Device: {device}")
    print(f"Steps: {num_steps}, Batch: {batch_size}")
    print()
    print("Hypothesis: Soroban (spatial) will converge faster than Binary (symbolic)")
    print("           because carry becomes visible state transition, not XOR logic.")
    print("=" * 70)
    print()
    
    # Setup Left Channel (Binary)
    # Input: A(8) + B(8) = 16 dims, Output: C(8) = 8 dims
    enc_bin = BinaryEncoder(8)
    model_bin = AdderNet(input_dim=16, output_dim=8, hidden_dim=128).to(device)
    opt_bin = optim.Adam(model_bin.parameters(), lr=0.005)
    
    # Setup Right Channel (Soroban)
    # Input: A(32) + B(32) = 64 dims, Output: C(32) = 32 dims
    enc_sor = SorobanEncoder(embed_dim=16)  # We use encode_batch, not forward
    model_sor = AdderNet(input_dim=64, output_dim=32, hidden_dim=128).to(device)
    opt_sor = optim.Adam(model_sor.parameters(), lr=0.005)
    
    history_bin = []
    history_sor = []
    
    print("Step     | Binary                        | Soroban")
    print("-" * 70)
    
    for step in range(1, num_steps + 1):
        a_ints, b_ints, c_ints = get_batch(batch_size)
        
        # --- LEFT CHANNEL: Binary ---
        a_bin = enc_bin.encode_batch(a_ints)
        b_bin = enc_bin.encode_batch(b_ints)
        x_bin = torch.cat([a_bin, b_bin], dim=1).to(device)
        y_bin = enc_bin.encode_batch(c_ints).to(device)
        
        opt_bin.zero_grad()
        pred_bin = model_bin(x_bin)
        loss_bin = nn.MSELoss()(pred_bin, y_bin)
        loss_bin.backward()
        opt_bin.step()
        
        # --- RIGHT CHANNEL: Soroban ---
        a_sor = enc_sor.encode_batch(a_ints)
        b_sor = enc_sor.encode_batch(b_ints)
        x_sor = torch.cat([a_sor, b_sor], dim=1).to(device)
        y_sor = enc_sor.encode_batch(c_ints).to(device)
        
        opt_sor.zero_grad()
        pred_sor = model_sor(x_sor)
        loss_sor = nn.MSELoss()(pred_sor, y_sor)
        loss_sor.backward()
        opt_sor.step()
        
        # Evaluate every 100 steps
        if step % 100 == 0:
            # Decode and check accuracy
            decoded_bin = enc_bin.decode_batch(pred_bin.cpu())
            decoded_sor = enc_sor.decode(pred_sor.cpu())
            
            correct_bin = (decoded_bin == c_ints).sum().item()
            correct_sor = (decoded_sor == c_ints).sum().item()
            
            pct_bin = (correct_bin / batch_size) * 100
            pct_sor = (correct_sor / batch_size) * 100
            
            history_bin.append(pct_bin)
            history_sor.append(pct_sor)
            
            # Visual bars
            bar_bin = "█" * int(pct_bin / 5)
            bar_sor = "█" * int(pct_sor / 5)
            
            # Winner indicator
            if pct_sor > pct_bin + 5:
                winner = "◀ SOR"
            elif pct_bin > pct_sor + 5:
                winner = "BIN ▶"
            else:
                winner = "  =  "
            
            print(f"{step:4d}     | {pct_bin:5.1f}% |{bar_bin:<20}| {winner} |{bar_sor:<20}| {pct_sor:5.1f}%")
    
    # Final summary
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    # Last 5 measurements average
    avg_bin = sum(history_bin[-5:]) / 5 if len(history_bin) >= 5 else history_bin[-1]
    avg_sor = sum(history_sor[-5:]) / 5 if len(history_sor) >= 5 else history_sor[-1]
    
    print(f"Binary final avg:  {avg_bin:.1f}%")
    print(f"Soroban final avg: {avg_sor:.1f}%")
    print()
    
    if avg_sor > avg_bin + 10:
        print("🏆 SOROBAN WINS!")
        print("   Representation matters. The 'Savant' theory is supported.")
        print("   Arithmetic as Tetris > Arithmetic as XOR.")
    elif avg_bin > avg_sor + 10:
        print("🏆 BINARY WINS!")
        print("   The compact representation is sufficient.")
        print("   Model capacity matters more than representation.")
    else:
        print("🤝 TIE (within 10%)")
        print("   Both representations are viable.")
        print("   Further investigation needed.")
    
    print("=" * 70)
    
    return history_bin, history_sor


def main():
    # Detect device
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"CUDA detected: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("Running on CPU (CUDA not available)")
    
    print()
    
    # Run the shootout
    history_bin, history_sor = run_shootout(
        num_steps=3000,
        batch_size=64,
        device=device
    )
    
    # Optional: save results for plotting
    import json
    results = {
        'binary': history_bin,
        'soroban': history_sor,
    }
    
    results_path = os.path.join(os.path.dirname(__file__), '..', 'results_shootout.json')
    with open(results_path, 'w') as f:
        json.dump(results, f)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
