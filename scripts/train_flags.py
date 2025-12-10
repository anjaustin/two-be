#!/usr/bin/env python3
"""
Train FLAGS Micro-Model

CLC (0x18): C = 0
SEC (0x38): C = 1
CLV (0xB8): V = 0
CLI (0x58): I = 0
SEI (0x78): I = 1
CLD (0xD8): D = 0
SED (0xF8): D = 1
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

FLAG_OPS = {0x18: 'CLC', 0x38: 'SEC', 0xB8: 'CLV', 0x58: 'CLI', 0x78: 'SEI', 0xD8: 'CLD', 0xF8: 'SED'}

class FlagNet(nn.Module):
    def __init__(self, hidden=32, emb=8):
        super().__init__()
        self.opcode_emb = nn.Embedding(256, emb)
        self.net = nn.Sequential(
            nn.Linear(8 + emb, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, 8)  # P register
    
    def _enc8(self, v):
        return torch.stack([((v >> i) & 1).float() for i in range(8)], dim=-1)
    
    def forward(self, p, op):
        return self.head(self.net(torch.cat([self._enc8(p), self.opcode_emb(op)], dim=-1)))
    
    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class DS(Dataset):
    def __init__(self, i, t): self.i, self.t = i, t
    def __len__(self): return len(self.i)
    def __getitem__(self, idx): return self.i[idx], self.t[idx]

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", default="cpu_shards")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--output", default="checkpoints/swarm/flags_net.pt")
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    print("Loading data...")
    all_i, all_t = [], []
    for sf in sorted(Path(args.shard_dir).glob('shard_*.pt')):
        s = torch.load(sf, weights_only=False)
        all_i.append(s['input']); all_t.append(s['target'])
    inputs, targets = torch.cat(all_i), torch.cat(all_t)
    
    mask = torch.zeros(len(inputs), dtype=torch.bool)
    for op in FLAG_OPS: mask |= (inputs[:, 7] == op)
    fi, ft = inputs[mask], targets[mask]
    print(f"Samples: {len(fi):,}")
    
    torch.manual_seed(42)
    model = FlagNet(hidden=32, emb=8).to(device)
    print(f"Parameters: {model.num_parameters:,}")
    
    opt = optim.Adam(model.parameters(), lr=args.lr)
    loader = DataLoader(DS(fi, ft), batch_size=args.batch_size, shuffle=True)
    
    for epoch in range(args.epochs):
        model.train()
        for inp, tgt in loader:
            inp, tgt = inp.to(device), tgt.to(device)
            p, op = inp[:, 4].long(), inp[:, 7].long()
            target_p = tgt[:, 4].long()
            
            opt.zero_grad()
            logits = model(p, op)
            def enc8(v): return torch.stack([((v>>i)&1).float() for i in range(8)], dim=-1)
            F.binary_cross_entropy_with_logits(logits, enc8(target_p)).backward()
            opt.step()
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inp, tgt in loader:
            inp, tgt = inp.to(device), tgt.to(device)
            p, op = inp[:, 4].long(), inp[:, 7].long()
            target_p = tgt[:, 4].long()
            probs = torch.sigmoid(model(p, op))
            pred = sum((probs[:, i] > 0.5).long() << i for i in range(8))
            correct += (pred == target_p).sum().item()
            total += len(p)
    
    print(f"Final accuracy: {correct/total*100:.1f}%")
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'config': {'hidden': 32, 'emb': 8},
        'accuracy': correct/total*100,
    }, args.output)
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()
