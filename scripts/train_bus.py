#!/usr/bin/env python3
"""
Phase 2: Integration - Train the Orchestrator (Neural Bus)

Prerequisites:
- Phase 1 complete (organelles.pt exists)
- Organelles are FROZEN (weights locked)
- Only the orchestrator is trained

The orchestrator learns to:
- Enforce consistency (Result=0 -> Z=1)
- Denoise weak signals
- Resolve conflicts between organelles

Usage:
    python scripts/train_bus.py --organelles checkpoints/organelles/organelles.pt
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
from bbdos.cpu.orchestrator import ADC_Orchestrator, NeuralALU
from bbdos.cpu.abacus import SorobanEncoder


class ALUDataset(Dataset):
    """Dataset for ALU training."""
    
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
    """Generate pristine synthetic ADC data."""
    print(f"Generating {n_samples:,} synthetic ADC samples...")
    
    a = torch.randint(0, 256, (n_samples,), dtype=torch.long)
    m = torch.randint(0, 256, (n_samples,), dtype=torch.long)
    c_in = torch.randint(0, 2, (n_samples,), dtype=torch.long)
    
    total = a.long() + m.long() + c_in.long()
    result = (total % 256).long()
    
    c_out = (total > 255).long()
    z_out = (result == 0).long()
    n_out = ((result & 0x80) > 0).long()
    
    a_sign = (a & 0x80).long()
    m_sign = (m & 0x80).long()
    r_sign = (result & 0x80).long()
    v_out = ((a_sign == m_sign) & (a_sign != r_sign)).long()
    
    return ALUDataset(a, m, c_in, result, c_out, v_out, z_out, n_out)


def train_bus(
    organelles_path,
    dataset,
    epochs=10,
    batch_size=1024,
    lr=0.001,
    device='cuda',
    log_every=500,
    output_dir='checkpoints/organelles'
):
    """
    Phase 2: Train the orchestrator with frozen organelles.
    """
    print(f"\n{'='*70}")
    print("PHASE 2: INTEGRATION")
    print("Training the Neural Bus (Orchestrator)")
    print(f"{'='*70}\n")
    
    # Load organelles
    print(f"Loading organelles from {organelles_path}...")
    checkpoint = torch.load(organelles_path, weights_only=False)
    
    cluster = OrganelleCluster().to(device)
    cluster.org_result.load_state_dict(checkpoint['org_result'])
    cluster.org_c.load_state_dict(checkpoint['org_c'])
    cluster.org_v.load_state_dict(checkpoint['org_v'])
    cluster.org_nz.load_state_dict(checkpoint['org_nz'])
    
    print(f"Organelle accuracies from Phase 1:")
    for k, v in checkpoint['best_acc'].items():
        print(f"  {k}: {v:.1f}%")
    
    # FREEZE organelles
    for param in cluster.parameters():
        param.requires_grad = False
    cluster.eval()
    print("\nOrganelles FROZEN.")
    
    # Initialize orchestrator
    orchestrator = ADC_Orchestrator(hidden_dim=64).to(device)
    optimizer = optim.Adam(orchestrator.parameters(), lr=lr)
    soroban = SorobanEncoder()
    
    print(f"Orchestrator parameters: {orchestrator.num_parameters:,}")
    print()
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    best_acc = {'result': 0, 'c': 0, 'v': 0, 'z': 0, 'n': 0, 'all': 0}
    
    for epoch in range(epochs):
        orchestrator.train()
        
        total_loss = 0
        correct = {'result': 0, 'c': 0, 'v': 0, 'z': 0, 'n': 0, 'all': 0}
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
            
            # Get frozen organelle outputs
            with torch.no_grad():
                raw_outputs = cluster(a, m, c_in)
            
            # Apply orchestrator (this is what we're training)
            optimizer.zero_grad()
            corrected = orchestrator(raw_outputs)
            
            # Compute loss on corrected outputs
            target_result_sor = soroban.encode_batch(target_result).to(device)
            
            loss_result = F.binary_cross_entropy_with_logits(corrected['result'], target_result_sor)
            loss_c = F.binary_cross_entropy_with_logits(corrected['c'].squeeze(), target_c)
            loss_v = F.binary_cross_entropy_with_logits(corrected['v'].squeeze(), target_v)
            loss_nz = F.binary_cross_entropy_with_logits(
                corrected['nz'], 
                torch.stack([target_z, target_n], dim=-1)
            )
            
            loss = loss_result + loss_c + loss_v + loss_nz
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Compute accuracies
            with torch.no_grad():
                pred_result = soroban.decode_batch(torch.sigmoid(corrected['result']).cpu())
                result_correct = (pred_result == target_result.cpu())
                correct['result'] += result_correct.sum().item()
                
                pred_c = (torch.sigmoid(corrected['c']) > 0.5).float().squeeze()
                c_correct = (pred_c == target_c)
                correct['c'] += c_correct.sum().item()
                
                pred_v = (torch.sigmoid(corrected['v']) > 0.5).float().squeeze()
                v_correct = (pred_v == target_v)
                correct['v'] += v_correct.sum().item()
                
                pred_nz = (torch.sigmoid(corrected['nz']) > 0.5).float()
                z_correct = (pred_nz[:, 0] == target_z)
                n_correct = (pred_nz[:, 1] == target_n)
                correct['z'] += z_correct.sum().item()
                correct['n'] += n_correct.sum().item()
                
                # All correct (strict)
                all_correct = result_correct.to(device) & c_correct & v_correct & z_correct & n_correct
                correct['all'] += all_correct.sum().item()
                
                total += len(a)
            
            if (batch_idx + 1) % log_every == 0:
                acc = {k: correct[k] / total * 100 for k in correct}
                print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(loader)} | "
                      f"ALL: {acc['all']:.1f}% | Result: {acc['result']:.1f}% | "
                      f"C: {acc['c']:.1f}% | V: {acc['v']:.1f}%")
        
        # Epoch summary
        epoch_time = time.time() - epoch_start
        acc = {k: correct[k] / total * 100 for k in correct}
        avg_loss = total_loss / len(loader)
        
        print(f"\n>>> Epoch {epoch+1}/{epochs} Complete <<<")
        print(f"    Time: {epoch_time:.1f}s | Loss: {avg_loss:.4f}")
        print(f"    ALL CORRECT: {acc['all']:.1f}%")
        print(f"    Individual: Result={acc['result']:.1f}%, C={acc['c']:.1f}%, "
              f"V={acc['v']:.1f}%, Z={acc['z']:.1f}%, N={acc['n']:.1f}%")
        
        # Track best
        if acc['all'] > best_acc['all']:
            best_acc = acc.copy()
            print(f"    *** New Best! ***")
        
        print()
    
    # Save
    torch.save({
        'orchestrator': orchestrator.state_dict(),
        'organelles': organelles_path,
        'best_acc': best_acc,
    }, f"{output_dir}/bus.pt")
    
    print(f"{'='*70}")
    print("PHASE 2 COMPLETE")
    print(f"{'='*70}")
    print(f"Best ALL CORRECT: {best_acc['all']:.1f}%")
    print(f"\nSaved: {output_dir}/bus.pt")
    
    return orchestrator, best_acc


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Train Neural Bus")
    parser.add_argument("--organelles", type=str, required=True, 
                        help="Path to Phase 1 checkpoint")
    parser.add_argument("--data", type=str, default=None, help="Path to dataset")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic data")
    parser.add_argument("--n-samples", type=int, default=5_000_000)
    parser.add_argument("--epochs", type=int, default=10)
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
        data = torch.load(args.data, weights_only=False)
        dataset = ALUDataset(
            data['a'], data['m'], data['c_in'],
            data['result'], data['c_out'], data['v_out'],
            data['z_out'], data['n_out']
        )
    else:
        print("ERROR: Must specify --data or --synthetic")
        sys.exit(1)
    
    # Train
    train_bus(
        args.organelles,
        dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
