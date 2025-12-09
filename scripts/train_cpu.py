#!/usr/bin/env python3
"""
Neural 6502 Training Script

Trains the Neural CPU model with reproducible settings.
"""

import argparse
import os
import sys
import random
import yaml
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/workspace/BBDOS")

from bbdos.cpu import NeuralCPU, CPUConfig


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


def load_traces(trace_file: str, val_split: float = 0.1):
    """Load CPU traces and split into train/val."""
    print(f"Loading traces from {trace_file}...")
    data = torch.load(trace_file)
    
    # Handle different trace formats
    if isinstance(data, dict):
        # New format with named fields
        traces = data
    else:
        # Legacy format: tensor of shape [N, 9] (7 regs + op + val)
        traces = {
            'A': data[:, 0].long(),
            'X': data[:, 1].long(),
            'Y': data[:, 2].long(),
            'SP': data[:, 3].long(),
            'P': data[:, 4].long(),
            'PCH': data[:, 5].long(),
            'PCL': data[:, 6].long(),
            'Op': data[:, 7].long(),
            'Val': data[:, 8].long() if data.shape[1] > 8 else torch.zeros_like(data[:, 0]).long(),
        }
    
    n_samples = len(traces['A'])
    n_val = int(n_samples * val_split)
    
    # Split
    train_traces = {k: v[:-n_val] for k, v in traces.items()}
    val_traces = {k: v[-n_val:] for k, v in traces.items()}
    
    print(f"  Train: {len(train_traces['A']):,} samples")
    print(f"  Val: {len(val_traces['A']):,} samples")
    
    return train_traces, val_traces


def create_dataloaders(train_traces, val_traces, batch_size: int, num_workers: int = 4):
    """Create PyTorch DataLoaders from traces."""
    # Stack all inputs into tensors
    train_inputs = torch.stack([train_traces[k] for k in ['A', 'X', 'Y', 'SP', 'P', 'PCH', 'PCL', 'Op', 'Val']], dim=1)
    val_inputs = torch.stack([val_traces[k] for k in ['A', 'X', 'Y', 'SP', 'P', 'PCH', 'PCL', 'Op', 'Val']], dim=1)
    
    # Targets are the next state (shifted by 1)
    train_targets = train_inputs[1:, :7].clone()  # Only register outputs
    train_inputs = train_inputs[:-1]
    
    val_targets = val_inputs[1:, :7].clone()
    val_inputs = val_inputs[:-1]
    
    train_dataset = TensorDataset(train_inputs, train_targets)
    val_dataset = TensorDataset(val_inputs, val_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, device, register_names):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_correct = {reg: 0 for reg in register_names}
    total_samples = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for inputs, targets in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Unpack inputs into state dict
        state = {
            'A': inputs[:, 0], 'X': inputs[:, 1], 'Y': inputs[:, 2],
            'SP': inputs[:, 3], 'P': inputs[:, 4], 'PCH': inputs[:, 5],
            'PCL': inputs[:, 6], 'Op': inputs[:, 7], 'Val': inputs[:, 8]
        }
        
        # Forward
        predictions, gates = model(state)
        
        # Compute loss for each register
        loss = 0
        for i, reg in enumerate(register_names):
            loss += F.cross_entropy(predictions[reg], targets[:, i])
            total_correct[reg] += (predictions[reg].argmax(dim=1) == targets[:, i]).sum().item()
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_samples += inputs.shape[0]
        
        pbar.set_postfix({'loss': loss.item() / len(register_names)})
    
    avg_loss = total_loss / len(train_loader) / len(register_names)
    accuracies = {reg: total_correct[reg] / total_samples * 100 for reg in register_names}
    
    return avg_loss, accuracies


def validate(model, val_loader, device, register_names):
    """Validate model."""
    model.eval()
    total_loss = 0
    total_correct = {reg: 0 for reg in register_names}
    total_samples = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            state = {
                'A': inputs[:, 0], 'X': inputs[:, 1], 'Y': inputs[:, 2],
                'SP': inputs[:, 3], 'P': inputs[:, 4], 'PCH': inputs[:, 5],
                'PCL': inputs[:, 6], 'Op': inputs[:, 7], 'Val': inputs[:, 8]
            }
            
            predictions, _ = model(state)
            
            loss = 0
            for i, reg in enumerate(register_names):
                loss += F.cross_entropy(predictions[reg], targets[:, i])
                total_correct[reg] += (predictions[reg].argmax(dim=1) == targets[:, i]).sum().item()
            
            total_loss += loss.item()
            total_samples += inputs.shape[0]
    
    avg_loss = total_loss / len(val_loader) / len(register_names)
    accuracies = {reg: total_correct[reg] / total_samples * 100 for reg in register_names}
    avg_accuracy = sum(accuracies.values()) / len(accuracies)
    
    return avg_loss, accuracies, avg_accuracy


def main():
    parser = argparse.ArgumentParser(description="Train Neural 6502")
    parser.add_argument('--config', type=str, default='configs/neural_cpu.yaml', help='Config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--traces', type=str, default=None, help='Override trace file path')
    parser.add_argument('--output-dir', type=str, default='checkpoints/cpu', help='Output directory')
    args = parser.parse_args()
    
    # Load config
    config_path = Path(__file__).parent.parent / args.config
    if not config_path.exists():
        # Try absolute path
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
    
    # Create model
    model_cfg = CPUConfig(
        d_model=cfg['model']['d_model'],
        n_heads=cfg['model']['n_heads'],
        n_layers=cfg['model']['n_layers'],
        num_tiles=cfg['model']['num_tiles'],
        dropout=cfg['model'].get('dropout', 0.1),
    )
    model = NeuralCPU(model_cfg).to(device)
    print(f"Model parameters: {model.num_parameters / 1e6:.2f}M")
    
    # Load traces
    trace_file = args.traces or cfg['data']['trace_file']
    if not Path(trace_file).exists():
        # Try legacy location
        trace_file = "/workspace/BBDOS/cpu_traces_v1.pt"
    
    train_traces, val_traces = load_traces(trace_file, cfg['data'].get('val_split', 0.1))
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_traces, val_traces,
        batch_size=cfg['training']['batch_size'],
        num_workers=cfg['data'].get('num_workers', 4)
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['training']['learning_rate'],
        weight_decay=cfg['training'].get('weight_decay', 0.01)
    )
    
    # Resume from checkpoint
    start_epoch = 0
    best_accuracy = 0
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model'])
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_accuracy = ckpt.get('best_accuracy', 0)
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Register names
    register_names = list(model_cfg.registers)
    
    # Training loop
    epochs = cfg['training']['epochs']
    print(f"\nStarting training for {epochs} epochs...")
    
    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, register_names)
        
        # Validate
        val_loss, val_acc, avg_val_acc = validate(model, val_loader, device, register_names)
        
        # Print results
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {avg_val_acc:.2f}%")
        print(f"  Per-register: {', '.join(f'{r}={a:.1f}%' for r, a in val_acc.items())}")
        
        # Save checkpoint
        is_best = avg_val_acc > best_accuracy
        if is_best:
            best_accuracy = avg_val_acc
        
        ckpt = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_accuracy': avg_val_acc,
            'best_accuracy': best_accuracy,
            'config': cfg,
        }
        
        torch.save(ckpt, output_dir / f'epoch_{epoch}.pt')
        if is_best:
            torch.save(ckpt, output_dir / 'best.pt')
            print(f"  New best model saved!")
    
    print(f"\nTraining complete. Best validation accuracy: {best_accuracy:.2f}%")


if __name__ == "__main__":
    main()
