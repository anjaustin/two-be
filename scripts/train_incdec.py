#!/usr/bin/env python3
"""
Train INC/DEC Micro-Model

INX (0xE8): X = X + 1, set N/Z
DEX (0xCA): X = X - 1, set N/Z
INY (0xC8): Y = Y + 1, set N/Z
DEY (0x88): Y = Y - 1, set N/Z
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

INCDEC_OPS = {0xE8: 'INX', 0xCA: 'DEX', 0xC8: 'INY', 0x88: 'DEY'}

class IncDecNet(nn.Module):
    def __init__(self, hidden=64, emb=16, layers=3):
        super().__init__()
        self.opcode_emb = nn.Embedding(256, emb)
        mods = [nn.Linear(16 + emb, hidden), nn.ReLU()]
        for _ in range(layers - 1):
            mods.extend([nn.Linear(hidden, hidden), nn.ReLU()])
        self.net = nn.Sequential(*mods)
        self.head = nn.Linear(hidden, 18)  # X(8) + Y(8) + N + Z
    
    def _enc8(self, v):
        return torch.stack([((v >> i) & 1).float() for i in range(8)], dim=-1)
    
    def forward(self, x, y, op):
        return self.head(self.net(torch.cat([self._enc8(x), self._enc8(y), self.opcode_emb(op)], dim=-1)))
    
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
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--output", default="checkpoints/swarm/incdec_net.pt")
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
    for op in INCDEC_OPS: mask |= (inputs[:, 7] == op)
    fi, ft = inputs[mask], targets[mask]
    print(f"Samples: {len(fi):,}")
    
    torch.manual_seed(42)
    model = IncDecNet(hidden=64, emb=16, layers=3).to(device)
    print(f"Parameters: {model.num_parameters:,}")
    
    opt = optim.Adam(model.parameters(), lr=args.lr)
    loader = DataLoader(DS(fi, ft), batch_size=args.batch_size, shuffle=True)
    
    for epoch in range(args.epochs):
        model.train()
        for inp, tgt in loader:
            inp, tgt = inp.to(device), tgt.to(device)
            x, y, op = inp[:, 1].long(), inp[:, 2].long(), inp[:, 7].long()
            tx, ty, tp = tgt[:, 1].long(), tgt[:, 2].long(), tgt[:, 4].long()
            tn, tz = ((tp>>7)&1).float(), ((tp>>1)&1).float()
            
            opt.zero_grad()
            logits = model(x, y, op)
            def enc8(v): return torch.stack([((v>>i)&1).float() for i in range(8)], dim=-1)
            target = torch.cat([enc8(tx), enc8(ty), tn.unsqueeze(-1), tz.unsqueeze(-1)], dim=-1)
            F.binary_cross_entropy_with_logits(logits, target).backward()
            opt.step()
        
        if (epoch + 1) % 20 == 0:
            model.eval()
            cx, cy, total = 0, 0, 0
            with torch.no_grad():
                for inp, tgt in loader:
                    inp, tgt = inp.to(device), tgt.to(device)
                    x, y, op = inp[:, 1].long(), inp[:, 2].long(), inp[:, 7].long()
                    tx, ty = tgt[:, 1].long(), tgt[:, 2].long()
                    probs = torch.sigmoid(model(x, y, op))
                    def dec8(p): return sum((p[:, i] > 0.5).long() << i for i in range(8))
                    px, py = dec8(probs[:, :8]), dec8(probs[:, 8:16])
                    cx += (px == tx).sum().item()
                    cy += (py == ty).sum().item()
                    total += len(x)
            print(f"Epoch {epoch+1}: X={cx/total*100:.1f}% Y={cy/total*100:.1f}%")
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'config': {'hidden': 64, 'emb': 16, 'layers': 3},
    }, args.output)
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()
