#!/usr/bin/env python3
"""
Train Transfer Micro-Model

Register-to-register transfers:
- TAX (0xAA): A -> X, set N/Z
- TXA (0x8A): X -> A, set N/Z
- TAY (0xA8): A -> Y, set N/Z
- TYA (0x98): Y -> A, set N/Z

100% deterministic from input state.
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
TAX = 0xAA  # A -> X
TXA = 0x8A  # X -> A
TAY = 0xA8  # A -> Y
TYA = 0x98  # Y -> A

TRANSFER_OPS = {TAX, TXA, TAY, TYA}


class TransferNet(nn.Module):
    """Network for register transfers."""
    
    def __init__(self, hidden_dim=32):
        super().__init__()
        
        # Input: A(8) + X(8) + Y(8) + opcode_emb(8) = 32
        self.opcode_emb = nn.Embedding(256, 8)
        
        self.net = nn.Sequential(
            nn.Linear(32, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Output: new_A(8) + new_X(8) + new_Y(8) + N(1) + Z(1) = 26
        self.head = nn.Linear(hidden_dim, 26)
    
    def _encode_8bit(self, val):
        bits = []
        for i in range(8):
            bits.append(((val >> i) & 1).float())
        return torch.stack(bits, dim=-1)
    
    def forward(self, a, x, y, opcode):
        a_bin = self._encode_8bit(a)
        x_bin = self._encode_8bit(x)
        y_bin = self._encode_8bit(y)
        op_emb = self.opcode_emb(opcode)
        
        inp = torch.cat([a_bin, x_bin, y_bin, op_emb], dim=-1)
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
        new_x = decode_8bit(probs[:, 8:16])
        new_y = decode_8bit(probs[:, 16:24])
        n_flag = (probs[:, 24] > 0.5).long()
        z_flag = (probs[:, 25] > 0.5).long()
        
        return new_a, new_x, new_y, n_flag, z_flag
    
    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_transfer_data(shard_dir, max_shards=None):
    """Load only transfer operations."""
    shard_files = sorted(Path(shard_dir).glob("shard_*.pt"))
    if max_shards:
        shard_files = shard_files[:max_shards]
    
    print(f"Loading transfer ops from {len(shard_files)} shards...")
    
    all_inputs = []
    all_targets = []
    
    for shard_file in shard_files:
        print(f"  {shard_file.name}...", end=" ", flush=True)
        shard = torch.load(shard_file, weights_only=False)
        
        inputs = shard['input']
        targets = shard['target']
        opcodes = inputs[:, 7]
        
        mask = torch.zeros(len(opcodes), dtype=torch.bool)
        for op in TRANSFER_OPS:
            mask |= (opcodes == op)
        
        transfer_inputs = inputs[mask]
        transfer_targets = targets[mask]
        
        print(f"{len(transfer_inputs):,} samples")
        
        all_inputs.append(transfer_inputs)
        all_targets.append(transfer_targets)
    
    inputs = torch.cat(all_inputs, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    print(f"\nTotal transfer samples: {len(inputs):,}")
    
    opcodes = inputs[:, 7]
    for op, name in [(TAX, 'TAX'), (TXA, 'TXA'), (TAY, 'TAY'), (TYA, 'TYA')]:
        count = (opcodes == op).sum().item()
        print(f"  {name}: {count:,}")
    
    return inputs, targets


class TransferDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def train_transfer(
    inputs, targets,
    epochs=10,
    batch_size=512,
    lr=0.001,
    device='cuda',
    log_every=100
):
    print(f"\n{'='*60}")
    print("Training Transfer Micro-Model")
    print(f"{'='*60}")
    
    model = TransferNet(hidden_dim=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Parameters: {model.num_parameters:,}")
    print(f"Samples: {len(inputs):,}")
    print(f"Epochs: {epochs}")
    
    dataset = TransferDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    best_acc = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_a = 0
        correct_x = 0
        correct_y = 0
        total = 0
        
        epoch_start = time.time()
        
        for batch_idx, (inp, tgt) in enumerate(loader):
            inp = inp.to(device)
            tgt = tgt.to(device)
            
            # Input: [A, X, Y, SP, P, PCH, PCL, Op, Val]
            a = inp[:, 0].long()
            x = inp[:, 1].long()
            y = inp[:, 2].long()
            opcode = inp[:, 7].long()
            
            # Target: [A, X, Y, SP, P, PCH, PCL]
            target_a = tgt[:, 0].long()
            target_x = tgt[:, 1].long()
            target_y = tgt[:, 2].long()
            target_p = tgt[:, 4].long()
            
            target_n = ((target_p >> 7) & 1).float()
            target_z = ((target_p >> 1) & 1).float()
            
            optimizer.zero_grad()
            logits = model(a, x, y, opcode)
            
            def encode_8bit(val):
                bits = []
                for i in range(8):
                    bits.append(((val >> i) & 1).float())
                return torch.stack(bits, dim=-1)
            
            target_tensor = torch.cat([
                encode_8bit(target_a),
                encode_8bit(target_x),
                encode_8bit(target_y),
                target_n.unsqueeze(-1),
                target_z.unsqueeze(-1),
            ], dim=-1)
            
            loss = F.binary_cross_entropy_with_logits(logits, target_tensor)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            with torch.no_grad():
                pred_a, pred_x, pred_y, _, _ = model.decode_output(logits)
                correct_a += (pred_a == target_a).sum().item()
                correct_x += (pred_x == target_x).sum().item()
                correct_y += (pred_y == target_y).sum().item()
                total += len(a)
            
            if (batch_idx + 1) % log_every == 0:
                acc_a = correct_a / total * 100
                acc_x = correct_x / total * 100
                acc_y = correct_y / total * 100
                print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(loader)} | "
                      f"A: {acc_a:.1f}% | X: {acc_x:.1f}% | Y: {acc_y:.1f}%")
        
        epoch_time = time.time() - epoch_start
        acc_a = correct_a / total * 100
        acc_x = correct_x / total * 100
        acc_y = correct_y / total * 100
        avg_acc = (acc_a + acc_x + acc_y) / 3
        avg_loss = total_loss / len(loader)
        
        print(f"\n>>> Epoch {epoch+1}/{epochs} Complete <<<")
        print(f"    Loss: {avg_loss:.4f}")
        print(f"    A: {acc_a:.1f}% | X: {acc_x:.1f}% | Y: {acc_y:.1f}%")
        print(f"    Time: {epoch_time:.1f}s")
        
        if avg_acc > best_acc:
            best_acc = avg_acc
            print(f"    *** New Best! ***")
        
        print()
    
    print(f"{'='*60}")
    print(f"Training Complete! Best Avg Accuracy: {best_acc:.1f}%")
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
    parser.add_argument("--output", default="checkpoints/swarm/transfer_net.pt")
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    inputs, targets = load_transfer_data(args.shard_dir, args.max_shards)
    
    model, best_acc = train_transfer(
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
