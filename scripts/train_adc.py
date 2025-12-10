#!/usr/bin/env python3
"""
Train ADC Micro-Model

One model. One job. ADC only.
Soroban encoding. Zero interference.

ADC opcodes: 0x69, 0x65, 0x75, 0x6D, 0x7D, 0x79, 0x61, 0x71
"""

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

from bbdos.cpu.abacus import SorobanEncoder

# ADC opcodes (all addressing modes)
ADC_OPCODES = {0x69, 0x65, 0x75, 0x6D, 0x7D, 0x79, 0x61, 0x71}


class ADCNet(nn.Module):
    """
    Micro-model for ADC only.
    
    Laser focused. No distractions.
    Soroban encoding for carry visibility.
    MLP + Attention hybrid.
    """
    
    def __init__(self, hidden_dim=128, n_heads=4, n_layers=3):
        super().__init__()
        
        self.soroban = SorobanEncoder(embed_dim=16)
        self.hidden_dim = hidden_dim
        
        # Input projection: 65 -> hidden_dim
        self.input_proj = nn.Linear(65, hidden_dim)
        
        # Transformer-style blocks with attention
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'attn': nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True),
                'norm1': nn.LayerNorm(hidden_dim),
                'ffn': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.ReLU(),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                ),
                'norm2': nn.LayerNorm(hidden_dim),
            }))
        
        # Output: Result(32) + Flags(4: N,Z,C,V)
        self.result_head = nn.Linear(hidden_dim, 32)
        self.flags_head = nn.Linear(hidden_dim, 4)
    
    def forward(self, a, operand, carry_in):
        """
        Args:
            a: [batch] accumulator (0-255)
            operand: [batch] memory value (0-255)
            carry_in: [batch] carry flag (0 or 1)
        """
        # Encode to Soroban
        a_sor = self.soroban.encode_batch(a)
        op_sor = self.soroban.encode_batch(operand)
        
        # Concatenate
        x = torch.cat([a_sor, op_sor, carry_in.float().unsqueeze(-1)], dim=-1)
        
        # Project to hidden dim
        x = self.input_proj(x)  # [batch, hidden_dim]
        
        # Add sequence dimension for attention (treat as 1-token sequence)
        x = x.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Transformer blocks
        for layer in self.layers:
            # Self-attention
            attn_out, _ = layer['attn'](x, x, x)
            x = layer['norm1'](x + attn_out)
            
            # FFN
            ffn_out = layer['ffn'](x)
            x = layer['norm2'](x + ffn_out)
        
        # Remove sequence dimension
        x = x.squeeze(1)  # [batch, hidden_dim]
        
        return self.result_head(x), self.flags_head(x)
    
    def predict(self, a, operand, carry_in):
        """Predict result and flags."""
        self.eval()
        with torch.no_grad():
            result_logits, flags_logits = self.forward(a, operand, carry_in)
            result = self.soroban.decode(torch.sigmoid(result_logits))
            flags = (torch.sigmoid(flags_logits) > 0.5).long()
            return result, flags
    
    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_adc_data(shard_dir, max_shards=None):
    """Load only ADC samples from shards."""
    shard_files = sorted(Path(shard_dir).glob("shard_*.pt"))
    if max_shards:
        shard_files = shard_files[:max_shards]
    
    print(f"Loading ADC samples from {len(shard_files)} shards...")
    
    all_inputs = []
    all_targets = []
    
    for shard_file in shard_files:
        print(f"  {shard_file.name}...", end=" ", flush=True)
        shard = torch.load(shard_file, weights_only=False)
        
        inputs = shard['input']
        targets = shard['target']
        opcodes = inputs[:, 7]
        
        # Filter to ADC only
        mask = torch.zeros(len(opcodes), dtype=torch.bool)
        for op in ADC_OPCODES:
            mask |= (opcodes == op)
        
        adc_inputs = inputs[mask]
        adc_targets = targets[mask]
        
        print(f"{len(adc_inputs):,} ADC samples")
        
        all_inputs.append(adc_inputs)
        all_targets.append(adc_targets)
    
    inputs = torch.cat(all_inputs, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    print(f"\nTotal ADC samples: {len(inputs):,}")
    
    return inputs, targets


class ADCDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def train_adc(
    inputs, 
    targets,
    epochs=10,
    batch_size=256,
    lr=0.001,
    hidden_dim=128,
    n_heads=4,
    n_layers=3,
    device='cuda',
    log_every=100
):
    """Train ADC micro-model."""
    
    print(f"\n{'='*60}")
    print("Training ADC Micro-Model (Soroban + Attention)")
    print(f"{'='*60}")
    
    model = ADCNet(hidden_dim=hidden_dim, n_heads=n_heads, n_layers=n_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    soroban = SorobanEncoder()
    
    print(f"Parameters: {model.num_parameters:,}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Attention heads: {n_heads}")
    print(f"Layers: {n_layers}")
    print(f"Learning rate: {lr}")
    print(f"Samples: {len(inputs):,}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"{'='*60}\n")
    
    dataset = ADCDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    best_acc = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        batch_count = 0
        
        epoch_start = time.time()
        
        for inp, tgt in loader:
            inp = inp.to(device)
            tgt = tgt.to(device)
            
            # Extract fields
            a = inp[:, 0].long()
            p = inp[:, 4].long()
            operand = inp[:, 8].long()
            carry_in = (p & 1).long()
            
            target_a = tgt[:, 0].long()
            target_p = tgt[:, 4].long()
            
            # Target flags: N(bit7), Z(bit1), C(bit0), V(bit6)
            target_flags = torch.stack([
                (target_p >> 7) & 1,
                (target_p >> 1) & 1,
                target_p & 1,
                (target_p >> 6) & 1,
            ], dim=-1).float()
            
            # Forward
            optimizer.zero_grad()
            result_logits, flags_logits = model(a, operand, carry_in)
            
            # Loss
            target_soroban = soroban.encode_batch(target_a).to(device)
            result_loss = F.binary_cross_entropy_with_logits(result_logits, target_soroban)
            flags_loss = F.binary_cross_entropy_with_logits(flags_logits, target_flags)
            loss = result_loss + 0.5 * flags_loss
            
            # Backward
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy
            with torch.no_grad():
                pred = soroban.decode(torch.sigmoid(result_logits).cpu())
                correct += (pred == target_a.cpu()).sum().item()
                total += len(a)
            
            batch_count += 1
            
            # Log progress
            if batch_count % log_every == 0:
                running_acc = correct / total * 100
                print(f"  Epoch {epoch+1} | Batch {batch_count}/{len(loader)} | "
                      f"Loss: {loss.item():.4f} | Acc: {running_acc:.1f}%")
        
        # Epoch summary
        epoch_time = time.time() - epoch_start
        epoch_loss = total_loss / len(loader)
        epoch_acc = correct / total * 100
        
        print(f"\n>>> Epoch {epoch+1}/{epochs} Complete <<<")
        print(f"    Loss: {epoch_loss:.4f}")
        print(f"    Accuracy: {epoch_acc:.1f}%")
        print(f"    Time: {epoch_time:.1f}s")
        
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            print(f"    *** New Best! ***")
        
        print()
    
    print(f"{'='*60}")
    print(f"Training Complete!")
    print(f"Best Accuracy: {best_acc:.1f}%")
    print(f"{'='*60}")
    
    return model, best_acc


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", default="cpu_shards")
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--output", default="checkpoints/swarm/adc_net.pt")
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load ADC-only data
    inputs, targets = load_adc_data(args.shard_dir, args.max_shards)
    
    # Train
    model, best_acc = train_adc(
        inputs, targets,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        device=device
    )
    
    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'accuracy': best_acc,
        'config': {
            'hidden_dim': args.hidden_dim,
            'n_heads': args.n_heads,
            'n_layers': args.n_layers,
            'encoding': 'soroban',
        }
    }, args.output)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
