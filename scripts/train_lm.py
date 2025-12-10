#!/usr/bin/env python3
"""
BBDOS Language Model Training Script

Trains the NanoLPU model with TriX sparse layers.

Usage:
    python train_lm.py --config configs/bbdos_lm.yaml
"""

import argparse
import os
import sys
import random
import yaml
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/workspace/BBDOS")

from bbdos.lm import NanoLPU, LMConfig, create_tokenizer

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("WARNING: datasets not installed. Run: pip install datasets")


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class TextDataLoader:
    """Simple streaming data loader for text."""
    
    def __init__(self, dataset_name: str, encode_fn, block_size: int, batch_size: int):
        self.dataset = load_dataset(dataset_name, split="train", streaming=True)
        self.encode = encode_fn
        self.block_size = block_size
        self.batch_size = batch_size
        self.iterator = iter(self.dataset)
        self.buffer = []
    
    def get_batch(self, device: torch.device):
        """Get a batch of data."""
        while len(self.buffer) < self.batch_size * (self.block_size + 1):
            try:
                item = next(self.iterator)
                text = item.get('text', '')
                self.buffer.extend(self.encode(text))
            except StopIteration:
                self.iterator = iter(self.dataset)
        
        # Create batch
        xs, ys = [], []
        for _ in range(self.batch_size):
            start = random.randint(0, len(self.buffer) - self.block_size - 1)
            chunk = self.buffer[start:start + self.block_size + 1]
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
        
        # Trim buffer periodically
        if len(self.buffer) > 100000:
            self.buffer = self.buffer[-50000:]
        
        x = torch.tensor(xs, dtype=torch.long, device=device)
        y = torch.tensor(ys, dtype=torch.long, device=device)
        return x, y


def compute_load_balance_loss(all_gates: list, num_tiles: int) -> torch.Tensor:
    """Compute load balancing loss to prevent mode collapse."""
    # Stack all gates: [num_layers, batch, seq, num_tiles]
    gates = torch.stack([g.float() for g in all_gates])
    
    # Average usage per tile
    usage = gates.mean(dim=(0, 1, 2))  # [num_tiles]
    
    # Target: uniform distribution
    target = torch.ones_like(usage) / num_tiles
    
    # MSE loss
    return F.mse_loss(usage, target)


def train_step(model, optimizer, x, y, balance_weight: float):
    """Single training step."""
    model.train()
    
    logits, loss, all_gates = model(x, y)
    
    # Add load balancing loss
    if balance_weight > 0 and all_gates:
        balance_loss = compute_load_balance_loss(all_gates, model.config.num_tiles)
        total_loss = loss + balance_weight * balance_loss
    else:
        total_loss = loss
        balance_loss = torch.tensor(0.0)
    
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return loss.item(), balance_loss.item()


@torch.no_grad()
def generate_sample(model, encode, decode, prompt: str, max_tokens: int, device: torch.device):
    """Generate a text sample."""
    model.eval()
    
    tokens = encode(prompt)
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    
    output = model.generate(x, max_tokens, temperature=0.8, top_p=0.9)
    
    return decode(output[0].tolist())


def main():
    parser = argparse.ArgumentParser(description="Train BBDOS Language Model")
    parser.add_argument('--config', type=str, default='configs/bbdos_lm.yaml', help='Config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--output-dir', type=str, default='checkpoints/lm', help='Output directory')
    args = parser.parse_args()
    
    if not HAS_DATASETS:
        print("ERROR: datasets library required. Run: pip install datasets")
        sys.exit(1)
    
    # Load config
    config_path = Path(__file__).parent.parent / args.config
    if not config_path.exists():
        config_path = Path(args.config)
    
    print(f"Loading config from {config_path}")
    cfg = load_config(str(config_path))
    
    # Set seed
    seed = cfg['training'].get('seed', 42)
    set_seed(seed)
    print(f"Random seed: {seed}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Tokenizer
    encode, decode, vocab_size = create_tokenizer()
    print(f"Vocab size: {vocab_size}")
    
    # Create model
    model_cfg = LMConfig(
        vocab_size=vocab_size,
        d_model=cfg['model']['d_model'],
        n_heads=cfg['model']['n_heads'],
        n_layers=cfg['model']['n_layers'],
        num_tiles=cfg['model']['num_tiles'],
        block_size=cfg['model']['block_size'],
        dropout=cfg['model'].get('dropout', 0.1),
        noise_scale=cfg['model']['trix'].get('noise_injection', 1.0),
    )
    model = NanoLPU(model_cfg).to(device)
    print(f"Model parameters: {model.num_parameters / 1e6:.1f}M")
    
    # Data loader
    data_loader = TextDataLoader(
        cfg['data']['dataset'],
        encode,
        model_cfg.block_size,
        cfg['training']['batch_size']
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['training']['learning_rate'],
        weight_decay=cfg['training'].get('weight_decay', 0.01)
    )
    
    # Resume from checkpoint
    start_step = 0
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model'])
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        start_step = ckpt.get('step', 0) + 1
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    max_steps = cfg['training']['max_steps']
    save_every = cfg['training'].get('save_every', 1000)
    log_every = cfg['logging'].get('log_every', 100)
    balance_weight = cfg['training'].get('balance_weight', 0.5)
    
    print(f"\nStarting training for {max_steps} steps...")
    print(f"  Batch size: {cfg['training']['batch_size']}")
    print(f"  Block size: {model_cfg.block_size}")
    print(f"  Balance weight: {balance_weight}")
    print()
    
    losses = []
    t0 = time.time()
    
    for step in range(start_step, max_steps):
        # Get batch
        x, y = data_loader.get_batch(device)
        
        # Train step
        loss, bal_loss = train_step(model, optimizer, x, y, balance_weight)
        losses.append(loss)
        
        # Logging
        if step % log_every == 0:
            avg_loss = np.mean(losses[-log_every:]) if losses else loss
            elapsed = time.time() - t0
            print(f"Step {step:>6} | Loss: {avg_loss:.4f} | Bal: {bal_loss:.4f} | Time: {elapsed:.1f}s")
        
        # Checkpointing
        if step > 0 and step % save_every == 0:
            ckpt = {
                'step': step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': loss,
                'config': cfg,
            }
            torch.save(ckpt, output_dir / f'step_{step}.pt')
            print(f"  Checkpoint saved: step_{step}.pt")
            
            # Generate sample
            sample = generate_sample(model, encode, decode, "Once upon a time", 100, device)
            print(f"  Sample: {sample[:150]}...")
    
    # Final save
    ckpt = {
        'step': max_steps,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': losses[-1] if losses else 0,
        'config': cfg,
    }
    torch.save(ckpt, output_dir / 'final.pt')
    print(f"\nTraining complete. Final model saved to {output_dir / 'final.pt'}")


if __name__ == "__main__":
    main()
