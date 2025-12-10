#!/usr/bin/env python3
"""
Phase 1: Cellular Division - Train Organelles Independently

Each organelle is trained on its specific target:
- Organelle_Result: Learns A + M + C -> Result
- Organelle_C: Learns carry out detection
- Organelle_V: Learns signed overflow detection
- Organelle_NZ: Learns Zero and Negative flags

Training is PARALLEL - each organelle sees the same input but different targets.
No orchestrator yet - that's Phase 2.

Usage:
    python scripts/train_organelles.py --data dataset_alu.pt
    python scripts/train_organelles.py --synthetic  # Generate synthetic data
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from bbdos.cpu.organelles import OrganelleCluster
from bbdos.cpu.abacus import SorobanEncoder


class ALUDataset(Dataset):
    """Dataset for ALU training with all targets."""
    
    def __init__(self, a, m, c_in, result, c_out, v_out, z_out, n_out):
        self.a = a
        self.m = m
        self.c_in = c_in
        self.result = result
        self.c_out = c_out
        self.v_out = v_out
        self.z_out = z_out
        self.n_out = n_out
    
    def __len__(self):
        return len(self.a)
    
    def __getitem__(self, idx):
        return {
            'a': self.a[idx],
            'm': self.m[idx],
            'c_in': self.c_in[idx],
            'result': self.result[idx],
            'c_out': self.c_out[idx],
            'v_out': self.v_out[idx],
            'z_out': self.z_out[idx],
            'n_out': self.n_out[idx],
        }


def generate_synthetic_data(n_samples=5_000_000):
    """
    Generate pristine synthetic ADC data.
    
    Pure math: A + M + C_in -> Result, Flags
    No addressing modes, no memory lookups.
    """
    print(f"Generating {n_samples:,} synthetic ADC samples...")
    
    # Random inputs
    a = torch.randint(0, 256, (n_samples,), dtype=torch.long)
    m = torch.randint(0, 256, (n_samples,), dtype=torch.long)
    c_in = torch.randint(0, 2, (n_samples,), dtype=torch.long)
    
    # Compute ground truth
    total = a.long() + m.long() + c_in.long()
    result = (total % 256).long()
    
    # Flags
    c_out = (total > 255).long()
    z_out = (result == 0).long()
    n_out = ((result & 0x80) > 0).long()
    
    # Overflow: V = ~(A ^ M) & (A ^ R) & 0x80
    # When A and M have same sign, but result has different sign
    a_sign = (a & 0x80).long()
    m_sign = (m & 0x80).long()
    r_sign = (result & 0x80).long()
    v_out = ((a_sign == m_sign) & (a_sign != r_sign)).long()
    
    print(f"  Result range: {result.min().item()} - {result.max().item()}")
    print(f"  Carry rate: {c_out.float().mean().item():.1%}")
    print(f"  Zero rate: {z_out.float().mean().item():.1%}")
    print(f"  Negative rate: {n_out.float().mean().item():.1%}")
    print(f"  Overflow rate: {v_out.float().mean().item():.1%}")
    
    return ALUDataset(a, m, c_in, result, c_out, v_out, z_out, n_out)


def load_data(data_path):
    """Load pre-generated dataset."""
    print(f"Loading data from {data_path}...")
    data = torch.load(data_path, weights_only=False)
    
    return ALUDataset(
        data['a'], data['m'], data['c_in'],
        data['result'], data['c_out'], data['v_out'],
        data['z_out'], data['n_out']
    )


def train_organelles(
    dataset,
    epochs=20,
    batch_size=1024,
    lr=0.001,
    device='cuda',
    log_every=500,
    output_dir='checkpoints/organelles'
):
    """
    Phase 1: Train all organelles in parallel.
    
    Each organelle has independent loss, but sees the same batches.
    """
    print(f"\n{'='*70}")
    print("PHASE 1: CELLULAR DIVISION")
    print("Training Organelles Independently")
    print(f"{'='*70}\n")
    
    # Initialize
    cluster = OrganelleCluster().to(device)
    soroban = SorobanEncoder()
    
    # Separate optimizers for each organelle (allows different LR if needed)
    opt_result = optim.Adam(cluster.org_result.parameters(), lr=lr)
    opt_c = optim.Adam(cluster.org_c.parameters(), lr=lr)
    opt_v = optim.Adam(cluster.org_v.parameters(), lr=lr)
    opt_nz = optim.Adam(cluster.org_nz.parameters(), lr=lr)
    
    print("Parameter breakdown:")
    for name, count in cluster.parameter_breakdown().items():
        print(f"  {name}: {count:,}")
    print()
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Track best accuracies
    best_acc = {'result': 0, 'c': 0, 'v': 0, 'z': 0, 'n': 0}
    
    for epoch in range(epochs):
        cluster.train()
        
        losses = {'result': 0, 'c': 0, 'v': 0, 'nz': 0}
        correct = {'result': 0, 'c': 0, 'v': 0, 'z': 0, 'n': 0}
        total = 0
        
        epoch_start = time.time()
        
        for batch_idx, batch in enumerate(loader):
            a = batch['a'].to(device)
            m = batch['m'].to(device)
            c_in = batch['c_in'].to(device)
            
            target_result = batch['result'].to(device)
            target_c = batch['c_out'].float().to(device)
            target_v = batch['v_out'].float().to(device)
            target_z = batch['z_out'].float().to(device)
            target_n = batch['n_out'].float().to(device)
            
            # Encode inputs
            x = cluster.encode_input(a, m, c_in)
            
            # --- Train Organelle_Result ---
            opt_result.zero_grad()
            result_logits = cluster.org_result(x)
            target_result_sor = soroban.encode_batch(target_result).to(device)
            loss_result = F.binary_cross_entropy_with_logits(result_logits, target_result_sor)
            loss_result.backward()
            opt_result.step()
            losses['result'] += loss_result.item()
            
            # --- Train Organelle_C ---
            opt_c.zero_grad()
            c_logits = cluster.org_c(x)
            loss_c = F.binary_cross_entropy_with_logits(c_logits.squeeze(), target_c)
            loss_c.backward()
            opt_c.step()
            losses['c'] += loss_c.item()
            
            # --- Train Organelle_V ---
            opt_v.zero_grad()
            v_logits = cluster.org_v(x)
            loss_v = F.binary_cross_entropy_with_logits(v_logits.squeeze(), target_v)
            loss_v.backward()
            opt_v.step()
            losses['v'] += loss_v.item()
            
            # --- Train Organelle_NZ ---
            opt_nz.zero_grad()
            nz_logits = cluster.org_nz(x)
            target_nz = torch.stack([target_z, target_n], dim=-1)
            loss_nz = F.binary_cross_entropy_with_logits(nz_logits, target_nz)
            loss_nz.backward()
            opt_nz.step()
            losses['nz'] += loss_nz.item()
            
            # --- Compute Accuracies ---
            with torch.no_grad():
                # Result accuracy
                pred_result = soroban.decode_batch(torch.sigmoid(result_logits).cpu())
                correct['result'] += (pred_result == target_result.cpu()).sum().item()
                
                # Flag accuracies
                pred_c = (torch.sigmoid(c_logits) > 0.5).float().squeeze()
                correct['c'] += (pred_c == target_c).sum().item()
                
                pred_v = (torch.sigmoid(v_logits) > 0.5).float().squeeze()
                correct['v'] += (pred_v == target_v).sum().item()
                
                pred_nz = (torch.sigmoid(nz_logits) > 0.5).float()
                correct['z'] += (pred_nz[:, 0] == target_z).sum().item()
                correct['n'] += (pred_nz[:, 1] == target_n).sum().item()
                
                total += len(a)
            
            # Log progress
            if (batch_idx + 1) % log_every == 0:
                acc_result = correct['result'] / total * 100
                acc_c = correct['c'] / total * 100
                acc_v = correct['v'] / total * 100
                acc_z = correct['z'] / total * 100
                acc_n = correct['n'] / total * 100
                
                print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(loader)} | "
                      f"Result: {acc_result:.1f}% | C: {acc_c:.1f}% | V: {acc_v:.1f}% | "
                      f"Z: {acc_z:.1f}% | N: {acc_n:.1f}%")
        
        # Epoch summary
        epoch_time = time.time() - epoch_start
        n_batches = len(loader)
        
        acc = {k: correct[k] / total * 100 for k in correct}
        avg_loss = {k: losses[k] / n_batches for k in losses}
        
        print(f"\n>>> Epoch {epoch+1}/{epochs} Complete <<<")
        print(f"    Time: {epoch_time:.1f}s")
        print(f"    Losses: Result={avg_loss['result']:.4f}, C={avg_loss['c']:.4f}, "
              f"V={avg_loss['v']:.4f}, NZ={avg_loss['nz']:.4f}")
        print(f"    Accuracy: Result={acc['result']:.1f}%, C={acc['c']:.1f}%, "
              f"V={acc['v']:.1f}%, Z={acc['z']:.1f}%, N={acc['n']:.1f}%")
        
        # Track best
        improved = []
        for k in best_acc:
            if acc[k] > best_acc[k]:
                best_acc[k] = acc[k]
                improved.append(k)
        
        if improved:
            print(f"    *** New Best: {', '.join(improved)} ***")
        
        print()
    
    # Save checkpoints
    os.makedirs(output_dir, exist_ok=True)
    
    torch.save({
        'org_result': cluster.org_result.state_dict(),
        'org_c': cluster.org_c.state_dict(),
        'org_v': cluster.org_v.state_dict(),
        'org_nz': cluster.org_nz.state_dict(),
        'best_acc': best_acc,
    }, f"{output_dir}/organelles.pt")
    
    print(f"{'='*70}")
    print("PHASE 1 COMPLETE")
    print(f"{'='*70}")
    print(f"Best Accuracies:")
    for k, v in best_acc.items():
        print(f"  {k}: {v:.1f}%")
    print(f"\nSaved: {output_dir}/organelles.pt")
    
    return cluster, best_acc


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Train Organelles")
    parser.add_argument("--data", type=str, default=None, help="Path to dataset")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic data")
    parser.add_argument("--n-samples", type=int, default=5_000_000, help="Synthetic samples")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output-dir", type=str, default="checkpoints/organelles")
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load or generate data
    if args.synthetic:
        dataset = generate_synthetic_data(args.n_samples)
    elif args.data:
        dataset = load_data(args.data)
    else:
        print("ERROR: Must specify --data or --synthetic")
        sys.exit(1)
    
    # Train
    train_organelles(
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
