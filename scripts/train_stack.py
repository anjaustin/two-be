#!/usr/bin/env python3
"""
Train Stack Micro-Models

Deterministic stack operations:
- PHA (0x48): Push A, SP -= 1
- PHP (0x08): Push P, SP -= 1  
- TXS (0x9A): SP = X
- TSX (0xBA): X = SP, set N/Z

These don't need stack memory - fully determined by input state.
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

# Opcodes
PHA = 0x48  # Push A
PHP = 0x08  # Push P
TXS = 0x9A  # Transfer X to SP
TSX = 0xBA  # Transfer SP to X
PLA = 0x68  # Pull A (needs stack - skip for now)
PLP = 0x28  # Pull P (needs stack - skip for now)

DETERMINISTIC_STACK_OPS = {PHA, PHP, TXS, TSX}


class StackNet(nn.Module):
    """Simple network for deterministic stack operations."""
    
    def __init__(self, hidden_dim=128):
        super().__init__()
        
        # Input: A(8) + X(8) + SP(8) + P(8) + opcode_emb(16) = 48
        self.opcode_emb = nn.Embedding(256, 16)
        
        self.net = nn.Sequential(
            nn.Linear(48, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Output: new_X(8) + new_SP(8) + N(1) + Z(1) = 18
        self.head = nn.Linear(hidden_dim, 18)
    
    def _encode_8bit(self, val):
        bits = []
        for i in range(8):
            bits.append(((val >> i) & 1).float())
        return torch.stack(bits, dim=-1)
    
    def forward(self, a, x, sp, p, opcode):
        a_bin = self._encode_8bit(a)
        x_bin = self._encode_8bit(x)
        sp_bin = self._encode_8bit(sp)
        p_bin = self._encode_8bit(p)
        op_emb = self.opcode_emb(opcode)
        
        inp = torch.cat([a_bin, x_bin, sp_bin, p_bin, op_emb], dim=-1)
        h = self.net(inp)
        return self.head(h)
    
    def decode_output(self, logits):
        probs = torch.sigmoid(logits)
        
        def decode_8bit(t):
            active = (t > 0.5).long()
            result = torch.zeros(t.shape[0], dtype=torch.long, device=t.device)
            for i in range(8):
                result |= (active[:, i] << i)
            return result
        
        new_x = decode_8bit(probs[:, :8])
        new_sp = decode_8bit(probs[:, 8:16])
        n_flag = (probs[:, 16] > 0.5).long()
        z_flag = (probs[:, 17] > 0.5).long()
        
        return new_x, new_sp, n_flag, z_flag
    
    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_stack_data(shard_dir, max_shards=None):
    """Load only deterministic stack operations."""
    shard_files = sorted(Path(shard_dir).glob("shard_*.pt"))
    if max_shards:
        shard_files = shard_files[:max_shards]
    
    print(f"Loading deterministic stack ops from {len(shard_files)} shards...")
    
    all_inputs = []
    all_targets = []
    
    for shard_file in shard_files:
        print(f"  {shard_file.name}...", end=" ", flush=True)
        shard = torch.load(shard_file, weights_only=False)
        
        inputs = shard['input']
        targets = shard['target']
        opcodes = inputs[:, 7]
        
        # Filter to deterministic stack ops
        mask = torch.zeros(len(opcodes), dtype=torch.bool)
        for op in DETERMINISTIC_STACK_OPS:
            mask |= (opcodes == op)
        
        stack_inputs = inputs[mask]
        stack_targets = targets[mask]
        
        print(f"{len(stack_inputs):,} samples")
        
        all_inputs.append(stack_inputs)
        all_targets.append(stack_targets)
    
    inputs = torch.cat(all_inputs, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    print(f"\nTotal deterministic stack samples: {len(inputs):,}")
    
    # Count by opcode
    opcodes = inputs[:, 7]
    for op, name in [(PHA, 'PHA'), (PHP, 'PHP'), (TXS, 'TXS'), (TSX, 'TSX')]:
        count = (opcodes == op).sum().item()
        print(f"  {name}: {count:,}")
    
    return inputs, targets


class StackDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def train_stack(
    inputs, targets,
    epochs=10,
    batch_size=512,
    lr=0.001,
    device='cuda',
    log_every=100
):
    print(f"\n{'='*60}")
    print("Training Stack Micro-Model")
    print(f"{'='*60}")
    
    model = StackNet(hidden_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Parameters: {model.num_parameters:,}")
    print(f"Samples: {len(inputs):,}")
    print(f"Epochs: {epochs}")
    
    dataset = StackDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    best_acc = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_x = 0
        correct_sp = 0
        total = 0
        
        epoch_start = time.time()
        
        for batch_idx, (inp, tgt) in enumerate(loader):
            inp = inp.to(device)
            tgt = tgt.to(device)
            
            # Input: [A, X, Y, SP, P, PCH, PCL, Op, Val]
            a = inp[:, 0].long()
            x = inp[:, 1].long()
            sp = inp[:, 3].long()
            p = inp[:, 4].long()
            opcode = inp[:, 7].long()
            
            # Target: [A, X, Y, SP, P, PCH, PCL]
            target_x = tgt[:, 1].long()
            target_sp = tgt[:, 3].long()
            target_p = tgt[:, 4].long()
            
            # Target flags
            target_n = ((target_p >> 7) & 1).float()
            target_z = ((target_p >> 1) & 1).float()
            
            # Forward
            optimizer.zero_grad()
            logits = model(a, x, sp, p, opcode)
            
            # Build target tensor
            def encode_8bit(val):
                bits = []
                for i in range(8):
                    bits.append(((val >> i) & 1).float())
                return torch.stack(bits, dim=-1)
            
            target_tensor = torch.cat([
                encode_8bit(target_x),
                encode_8bit(target_sp),
                target_n.unsqueeze(-1),
                target_z.unsqueeze(-1),
            ], dim=-1)
            
            loss = F.binary_cross_entropy_with_logits(logits, target_tensor)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy
            with torch.no_grad():
                pred_x, pred_sp, _, _ = model.decode_output(logits)
                correct_x += (pred_x == target_x).sum().item()
                correct_sp += (pred_sp == target_sp).sum().item()
                total += len(a)
            
            if (batch_idx + 1) % log_every == 0:
                acc_x = correct_x / total * 100
                acc_sp = correct_sp / total * 100
                print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(loader)} | "
                      f"X: {acc_x:.1f}% | SP: {acc_sp:.1f}%")
        
        epoch_time = time.time() - epoch_start
        acc_x = correct_x / total * 100
        acc_sp = correct_sp / total * 100
        avg_loss = total_loss / len(loader)
        
        print(f"\n>>> Epoch {epoch+1}/{epochs} Complete <<<")
        print(f"    Loss: {avg_loss:.4f}")
        print(f"    X Accuracy: {acc_x:.1f}%")
        print(f"    SP Accuracy: {acc_sp:.1f}%")
        print(f"    Time: {epoch_time:.1f}s")
        
        if acc_sp > best_acc:
            best_acc = acc_sp
            print(f"    *** New Best! ***")
        
        print()
    
    print(f"{'='*60}")
    print(f"Training Complete! Best SP Accuracy: {best_acc:.1f}%")
    print(f"{'='*60}")
    
    return model, best_acc


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", default="cpu_shards")
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output", default="checkpoints/swarm/stack_net.pt")
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    inputs, targets = load_stack_data(args.shard_dir, args.max_shards)
    
    model, best_acc = train_stack(
        inputs, targets,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device
    )
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'accuracy': best_acc,
    }, args.output)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
