#!/usr/bin/env python3
"""
Train Shift/Rotate Micro-Model

Accumulator shifts and rotates:
- ASL_A (0x0A): Shift left, bit 7 -> C, 0 -> bit 0
- LSR_A (0x4A): Shift right, bit 0 -> C, 0 -> bit 7
- ROL_A (0x2A): Rotate left through carry
- ROR_A (0x6A): Rotate right through carry

Deterministic bit manipulation.
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
ASL_A = 0x0A  # Arithmetic shift left
LSR_A = 0x4A  # Logical shift right
ROL_A = 0x2A  # Rotate left through carry
ROR_A = 0x6A  # Rotate right through carry

SHIFT_OPS = {ASL_A, LSR_A, ROL_A, ROR_A}


class ShiftNet(nn.Module):
    """Tiny network for shift/rotate operations."""
    
    def __init__(self, hidden_dim=64):
        super().__init__()
        
        # Input: A(8) + C_in(1) + opcode_emb(8) = 17
        self.opcode_emb = nn.Embedding(256, 8)
        
        self.net = nn.Sequential(
            nn.Linear(17, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Output: new_A(8) + C_out(1) + N(1) + Z(1) = 11
        self.head = nn.Linear(hidden_dim, 11)
    
    def _encode_8bit(self, val):
        bits = []
        for i in range(8):
            bits.append(((val >> i) & 1).float())
        return torch.stack(bits, dim=-1)
    
    def forward(self, a, c_in, opcode):
        a_bin = self._encode_8bit(a)
        c_bin = c_in.float().unsqueeze(-1)
        op_emb = self.opcode_emb(opcode)
        
        inp = torch.cat([a_bin, c_bin, op_emb], dim=-1)
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
        
        new_a = decode_8bit(probs[:, :8])
        c_out = (probs[:, 8] > 0.5).long()
        n_flag = (probs[:, 9] > 0.5).long()
        z_flag = (probs[:, 10] > 0.5).long()
        
        return new_a, c_out, n_flag, z_flag
    
    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_shift_data(shard_dir, max_shards=None):
    """Load only shift/rotate operations."""
    shard_files = sorted(Path(shard_dir).glob("shard_*.pt"))
    if max_shards:
        shard_files = shard_files[:max_shards]
    
    print(f"Loading shift/rotate ops from {len(shard_files)} shards...")
    
    all_inputs = []
    all_targets = []
    
    for shard_file in shard_files:
        print(f"  {shard_file.name}...", end=" ", flush=True)
        shard = torch.load(shard_file, weights_only=False)
        
        inputs = shard['input']
        targets = shard['target']
        opcodes = inputs[:, 7]
        
        mask = torch.zeros(len(opcodes), dtype=torch.bool)
        for op in SHIFT_OPS:
            mask |= (opcodes == op)
        
        shift_inputs = inputs[mask]
        shift_targets = targets[mask]
        
        print(f"{len(shift_inputs):,} samples")
        
        all_inputs.append(shift_inputs)
        all_targets.append(shift_targets)
    
    inputs = torch.cat(all_inputs, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    print(f"\nTotal shift/rotate samples: {len(inputs):,}")
    
    opcodes = inputs[:, 7]
    for op, name in [(ASL_A, 'ASL_A'), (LSR_A, 'LSR_A'), (ROL_A, 'ROL_A'), (ROR_A, 'ROR_A')]:
        count = (opcodes == op).sum().item()
        print(f"  {name}: {count:,}")
    
    return inputs, targets


class ShiftDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def train_shift(
    inputs, targets,
    epochs=10,
    batch_size=512,
    lr=0.001,
    device='cuda',
    log_every=100
):
    print(f"\n{'='*60}")
    print("Training Shift/Rotate Micro-Model")
    print(f"{'='*60}")
    
    model = ShiftNet(hidden_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Parameters: {model.num_parameters:,}")
    print(f"Samples: {len(inputs):,}")
    print(f"Epochs: {epochs}")
    
    dataset = ShiftDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    best_acc = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_a = 0
        correct_c = 0
        total = 0
        
        epoch_start = time.time()
        
        for batch_idx, (inp, tgt) in enumerate(loader):
            inp = inp.to(device)
            tgt = tgt.to(device)
            
            # Input: [A, X, Y, SP, P, PCH, PCL, Op, Val]
            a = inp[:, 0].long()
            p = inp[:, 4].long()
            opcode = inp[:, 7].long()
            c_in = (p & 1).long()  # Carry flag
            
            # Target: [A, X, Y, SP, P, PCH, PCL]
            target_a = tgt[:, 0].long()
            target_p = tgt[:, 4].long()
            
            target_c = (target_p & 1).float()
            target_n = ((target_p >> 7) & 1).float()
            target_z = ((target_p >> 1) & 1).float()
            
            optimizer.zero_grad()
            logits = model(a, c_in, opcode)
            
            def encode_8bit(val):
                bits = []
                for i in range(8):
                    bits.append(((val >> i) & 1).float())
                return torch.stack(bits, dim=-1)
            
            target_tensor = torch.cat([
                encode_8bit(target_a),
                target_c.unsqueeze(-1),
                target_n.unsqueeze(-1),
                target_z.unsqueeze(-1),
            ], dim=-1)
            
            loss = F.binary_cross_entropy_with_logits(logits, target_tensor)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            with torch.no_grad():
                pred_a, pred_c, _, _ = model.decode_output(logits)
                correct_a += (pred_a == target_a).sum().item()
                correct_c += (pred_c == (target_p & 1)).sum().item()
                total += len(a)
            
            if (batch_idx + 1) % log_every == 0:
                acc_a = correct_a / total * 100
                acc_c = correct_c / total * 100
                print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(loader)} | "
                      f"A: {acc_a:.1f}% | C: {acc_c:.1f}%")
        
        epoch_time = time.time() - epoch_start
        acc_a = correct_a / total * 100
        acc_c = correct_c / total * 100
        avg_loss = total_loss / len(loader)
        
        print(f"\n>>> Epoch {epoch+1}/{epochs} Complete <<<")
        print(f"    Loss: {avg_loss:.4f}")
        print(f"    A Accuracy: {acc_a:.1f}%")
        print(f"    C Accuracy: {acc_c:.1f}%")
        print(f"    Time: {epoch_time:.1f}s")
        
        if acc_a > best_acc:
            best_acc = acc_a
            print(f"    *** New Best! ***")
        
        print()
    
    print(f"{'='*60}")
    print(f"Training Complete! Best A Accuracy: {best_acc:.1f}%")
    print(f"{'='*60}")
    
    return model, best_acc


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", default="cpu_shards")
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output", default="checkpoints/swarm/shift_net.pt")
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    inputs, targets = load_shift_data(args.shard_dir, args.max_shards)
    
    model, best_acc = train_shift(
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
