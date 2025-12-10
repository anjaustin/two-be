#!/usr/bin/env python3
"""
Soroban 6502 Test: The Trojan Horse Experiment

Tests whether Soroban encoding helps the Neural 6502 learn ADC.

The hypothesis:
- Standard encoding: ADC competes with other opcodes, gets 3% accuracy
- Soroban encoding: ADC looks like shifting, gradient alignment, should get >50%

This is a focused test on ADC and ASL operations to prove the concept
before full model training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bbdos.cpu.abacus import SorobanEncoder


# Simplified models for the focused test
class BinaryALU(nn.Module):
    """Standard binary-encoded ALU model."""
    
    def __init__(self, hidden_dim=256):
        super().__init__()
        # Input: A(8) + operand(8) + carry(1) + opcode_embed(16) = 33
        # Output: result(8) + carry_out(1) = 9
        self.net = nn.Sequential(
            nn.Linear(33, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 9),
            nn.Sigmoid()
        )
        self.opcode_emb = nn.Embedding(256, 16)
    
    def forward(self, a_bits, operand_bits, carry_in, opcode):
        op_emb = self.opcode_emb(opcode)
        x = torch.cat([a_bits, operand_bits, carry_in.unsqueeze(-1), op_emb], dim=-1)
        return self.net(x)


class SorobanALU(nn.Module):
    """Soroban-encoded ALU model."""
    
    def __init__(self, hidden_dim=256):
        super().__init__()
        # Input: A(32) + operand(32) + carry(1) + opcode_embed(16) = 81
        # Output: result(32) + carry_out(1) = 33
        self.net = nn.Sequential(
            nn.Linear(81, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 33),
            nn.Sigmoid()
        )
        self.opcode_emb = nn.Embedding(256, 16)
    
    def forward(self, a_soroban, operand_soroban, carry_in, opcode):
        op_emb = self.opcode_emb(opcode)
        x = torch.cat([a_soroban, operand_soroban, carry_in.unsqueeze(-1), op_emb], dim=-1)
        return self.net(x)


# 6502 opcodes we care about
ADC_IMM = 0x69  # Add with carry (immediate)
ASL_A = 0x0A    # Arithmetic shift left (accumulator)
LSR_A = 0x4A    # Logical shift right (accumulator)
NOP = 0xEA      # No operation


def encode_binary(val, bits=8):
    """Encode integer as binary tensor."""
    return torch.tensor([(val >> i) & 1 for i in range(bits)], dtype=torch.float32)


def decode_binary(tensor, bits=8):
    """Decode binary tensor to integer."""
    val = 0
    for i in range(bits):
        if tensor[i] > 0.5:
            val |= (1 << i)
    return val


def execute_6502_op(opcode, a, operand, carry_in):
    """Ground truth 6502 execution."""
    if opcode == ADC_IMM:
        # Add with carry
        result = a + operand + carry_in
        carry_out = 1 if result > 255 else 0
        result = result & 0xFF
    elif opcode == ASL_A:
        # Arithmetic shift left
        result = (a << 1) & 0xFF
        carry_out = (a >> 7) & 1
    elif opcode == LSR_A:
        # Logical shift right
        carry_out = a & 1
        result = a >> 1
    elif opcode == NOP:
        result = a
        carry_out = carry_in
    else:
        result = a
        carry_out = carry_in
    
    return result, carry_out


def generate_batch(batch_size, opcodes, device='cuda'):
    """Generate a batch of random ALU operations."""
    a = torch.randint(0, 256, (batch_size,))
    operand = torch.randint(0, 256, (batch_size,))
    carry_in = torch.randint(0, 2, (batch_size,))
    opcode = torch.tensor([opcodes[i % len(opcodes)] for i in range(batch_size)])
    
    # Compute ground truth
    results = []
    carry_outs = []
    for i in range(batch_size):
        r, c = execute_6502_op(opcode[i].item(), a[i].item(), operand[i].item(), carry_in[i].item())
        results.append(r)
        carry_outs.append(c)
    
    return {
        'a': a.to(device),
        'operand': operand.to(device),
        'carry_in': carry_in.float().to(device),
        'opcode': opcode.to(device),
        'result': torch.tensor(results).to(device),
        'carry_out': torch.tensor(carry_outs).float().to(device),
    }


def run_experiment(num_steps=2000, batch_size=64, device='cuda'):
    """Run the Trojan Horse experiment."""
    
    print("=" * 70)
    print("TROJAN HORSE EXPERIMENT: Binary vs Soroban ALU")
    print("=" * 70)
    print()
    print("Testing: Can Soroban encoding help the model learn ADC?")
    print()
    print("Operations trained on:")
    print("  - ADC (Add with Carry): The 'hard' operation")
    print("  - ASL (Shift Left): The 'easy' operation")
    print("  - LSR (Shift Right): The 'easy' operation")
    print("  - NOP (No-op): Baseline")
    print()
    print("Hypothesis: Soroban makes ADC look like ASL (both are 'column ops')")
    print("=" * 70)
    print()
    
    # Setup encoders
    soroban_enc = SorobanEncoder(embed_dim=16)
    
    # Setup models
    model_bin = BinaryALU(hidden_dim=256).to(device)
    model_sor = SorobanALU(hidden_dim=256).to(device)
    
    opt_bin = optim.Adam(model_bin.parameters(), lr=0.001)
    opt_sor = optim.Adam(model_sor.parameters(), lr=0.001)
    
    # Training on mixed operations
    opcodes = [ADC_IMM, ASL_A, LSR_A, NOP]
    
    # Track per-opcode accuracy
    history = defaultdict(list)
    
    print("Training on mixed operations...")
    print("-" * 70)
    
    for step in range(1, num_steps + 1):
        batch = generate_batch(batch_size, opcodes, device)
        
        # --- Binary Model ---
        a_bin = torch.stack([encode_binary(v.item()) for v in batch['a']]).to(device)
        op_bin = torch.stack([encode_binary(v.item()) for v in batch['operand']]).to(device)
        target_bin = torch.stack([encode_binary(v.item()) for v in batch['result']]).to(device)
        target_bin = torch.cat([target_bin, batch['carry_out'].unsqueeze(-1)], dim=-1)
        
        opt_bin.zero_grad()
        pred_bin = model_bin(a_bin, op_bin, batch['carry_in'], batch['opcode'])
        loss_bin = nn.BCELoss()(pred_bin, target_bin)
        loss_bin.backward()
        opt_bin.step()
        
        # --- Soroban Model ---
        a_sor = soroban_enc.encode_batch(batch['a'])
        op_sor = soroban_enc.encode_batch(batch['operand'])
        target_sor = soroban_enc.encode_batch(batch['result'])
        target_sor = torch.cat([target_sor, batch['carry_out'].unsqueeze(-1)], dim=-1)
        
        opt_sor.zero_grad()
        pred_sor = model_sor(a_sor, op_sor, batch['carry_in'], batch['opcode'])
        loss_sor = nn.BCELoss()(pred_sor, target_sor)
        loss_sor.backward()
        opt_sor.step()
        
        # Evaluate every 200 steps
        if step % 200 == 0:
            model_bin.eval()
            model_sor.eval()
            
            # Test each opcode separately
            results_bin = {}
            results_sor = {}
            
            for op, name in [(ADC_IMM, 'ADC'), (ASL_A, 'ASL'), (LSR_A, 'LSR'), (NOP, 'NOP')]:
                test_batch = generate_batch(100, [op], device)
                
                with torch.no_grad():
                    # Binary
                    a_bin = torch.stack([encode_binary(v.item()) for v in test_batch['a']]).to(device)
                    op_bin_t = torch.stack([encode_binary(v.item()) for v in test_batch['operand']]).to(device)
                    pred_bin = model_bin(a_bin, op_bin_t, test_batch['carry_in'], test_batch['opcode'])
                    decoded_bin = torch.tensor([decode_binary(p[:8]) for p in pred_bin.cpu()])
                    acc_bin = (decoded_bin == test_batch['result'].cpu()).float().mean().item() * 100
                    
                    # Soroban
                    a_sor = soroban_enc.encode_batch(test_batch['a'])
                    op_sor_t = soroban_enc.encode_batch(test_batch['operand'])
                    pred_sor = model_sor(a_sor, op_sor_t, test_batch['carry_in'], test_batch['opcode'])
                    decoded_sor = soroban_enc.decode(pred_sor[:, :32].cpu())
                    acc_sor = (decoded_sor == test_batch['result'].cpu()).float().mean().item() * 100
                
                results_bin[name] = acc_bin
                results_sor[name] = acc_sor
            
            model_bin.train()
            model_sor.train()
            
            # Record history
            for name in ['ADC', 'ASL', 'LSR', 'NOP']:
                history[f'bin_{name}'].append(results_bin[name])
                history[f'sor_{name}'].append(results_sor[name])
            
            # Print comparison
            print(f"\nStep {step}")
            print(f"{'Op':<6} | {'Binary':<15} | {'Soroban':<15} | Winner")
            print("-" * 50)
            for name in ['ADC', 'ASL', 'LSR', 'NOP']:
                b = results_bin[name]
                s = results_sor[name]
                if s > b + 10:
                    winner = "◀ SOR"
                elif b > s + 10:
                    winner = "BIN ▶"
                else:
                    winner = "  =  "
                print(f"{name:<6} | {b:5.1f}%{' '*9} | {s:5.1f}%{' '*9} | {winner}")
    
    # Final summary
    print()
    print("=" * 70)
    print("FINAL RESULTS (last measurement)")
    print("=" * 70)
    print()
    print(f"{'Operation':<12} | {'Binary':<12} | {'Soroban':<12} | Delta")
    print("-" * 50)
    
    for name in ['ADC', 'ASL', 'LSR', 'NOP']:
        b = history[f'bin_{name}'][-1]
        s = history[f'sor_{name}'][-1]
        delta = s - b
        sign = "+" if delta > 0 else ""
        print(f"{name:<12} | {b:6.1f}%      | {s:6.1f}%      | {sign}{delta:.1f}%")
    
    print()
    print("=" * 70)
    
    # The critical comparison
    adc_bin = history['bin_ADC'][-1]
    adc_sor = history['sor_ADC'][-1]
    asl_bin = history['bin_ASL'][-1]
    asl_sor = history['sor_ASL'][-1]
    
    print("CRITICAL COMPARISON: ADC Accuracy")
    print("-" * 50)
    print(f"  Binary:  {adc_bin:.1f}%")
    print(f"  Soroban: {adc_sor:.1f}%")
    print()
    
    if adc_sor > adc_bin + 20:
        print("🏆 SOROBAN WINS ON ADC!")
        print("   The Trojan Horse worked. ADC disguised as geometry.")
        print("   Gradient alignment hypothesis CONFIRMED.")
    elif adc_sor > adc_bin + 5:
        print("📈 SOROBAN EDGE ON ADC")
        print("   Improvement detected. More training may help.")
    else:
        print("🤔 NO CLEAR WINNER ON ADC")
        print("   Further investigation needed.")
    
    print()
    print("ASL/LSR Comparison (should be similar):")
    print(f"  ASL: Binary={asl_bin:.1f}%, Soroban={asl_sor:.1f}%")
    print()
    
    if abs(asl_bin - asl_sor) < 10:
        print("✓ ASL accuracy similar - no regression on 'easy' ops")
    
    print("=" * 70)
    
    return history


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    run_experiment(num_steps=2000, batch_size=64, device=device)


if __name__ == "__main__":
    main()
