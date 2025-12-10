#!/usr/bin/env python3
"""
Train Swarm 6502: The Great Schism

Splits the 50M instruction dataset by Functional Unit and trains
each FU on its specialized subset.

Usage:
    python scripts/train_swarm.py --fu alu      # Train ALU only
    python scripts/train_swarm.py --fu all      # Train all FUs
    python scripts/train_swarm.py --fu alu --epochs 10
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from bbdos.cpu.fu_map import (
    build_fu_map, get_fu_opcodes, FU_ALU, FU_LOGIC, FU_MOVE, FU_FLOW, FU_STACK, FU_NAMES
)
from bbdos.cpu.fu_alu import FU_ALU as ALUModule, ALUConfig, alu_loss, execute_alu_op
from bbdos.cpu.fu_logic import FU_LOGIC as LogicModule, LogicConfig, logic_loss, execute_logic_op
from bbdos.cpu.fu_move import FU_MOVE as MoveModule, MoveConfig, move_loss, execute_move_op
from bbdos.cpu.fu_flow import FU_FLOW as FlowModule, FlowConfig, flow_loss, execute_flow_op
from bbdos.cpu.fu_stack import FU_STACK as StackModule, StackConfig, stack_loss, execute_stack_op
from bbdos.cpu.abacus import SorobanEncoder


class FUDataset(Dataset):
    """Dataset for a specific Functional Unit."""
    
    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor, fu_id: int):
        self.inputs = inputs
        self.targets = targets
        self.fu_id = fu_id
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return {
            'input': self.inputs[idx],
            'target': self.targets[idx],
        }


def load_and_split_shards(
    shard_dir: str,
    fu_map: torch.Tensor,
    max_shards: int = None,
    verbose: bool = True
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Load shards and split by Functional Unit.
    
    Returns:
        Dictionary mapping FU_ID to (inputs, targets) tensors
    """
    shard_files = sorted(Path(shard_dir).glob("shard_*.pt"))
    
    if max_shards:
        shard_files = shard_files[:max_shards]
    
    if verbose:
        print(f"Loading {len(shard_files)} shards from {shard_dir}...")
    
    # Collect samples by FU
    fu_inputs = defaultdict(list)
    fu_targets = defaultdict(list)
    
    for shard_file in shard_files:
        if verbose:
            print(f"  Loading {shard_file.name}...")
        
        shard = torch.load(shard_file, weights_only=False)
        inputs = shard['input']  # [N, 9]
        targets = shard['target']  # [N, 7]
        
        # Get opcode column (index 7)
        opcodes = inputs[:, 7].long()
        
        # Route to FUs
        fu_ids = fu_map[opcodes]
        
        for fu_id in range(5):
            mask = (fu_ids == fu_id)
            if mask.any():
                fu_inputs[fu_id].append(inputs[mask])
                fu_targets[fu_id].append(targets[mask])
    
    # Concatenate
    result = {}
    for fu_id in range(5):
        if fu_inputs[fu_id]:
            result[fu_id] = (
                torch.cat(fu_inputs[fu_id], dim=0),
                torch.cat(fu_targets[fu_id], dim=0)
            )
            if verbose:
                print(f"  {FU_NAMES[fu_id]}: {len(result[fu_id][0]):,} samples")
    
    return result


def train_fu_alu(
    train_data: Tuple[torch.Tensor, torch.Tensor],
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 0.001,
    device: str = 'cuda',
    verbose: bool = True
) -> ALUModule:
    """Train the ALU Functional Unit with Soroban encoding."""
    
    inputs, targets = train_data
    
    if verbose:
        print(f"\n{'='*60}")
        print("Training FU_ALU (Soroban Encoded)")
        print(f"{'='*60}")
        print(f"Samples: {len(inputs):,}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
    
    # Initialize
    config = ALUConfig()
    model = ALUModule(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    soroban = SorobanEncoder()
    
    if verbose:
        print(f"Parameters: {model.num_parameters:,}")
    
    # Create dataloader
    dataset = FUDataset(inputs, targets, FU_ALU)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for batch in loader:
            inp = batch['input'].to(device)
            tgt = batch['target'].to(device)
            
            # Extract fields
            # Input: [A, X, Y, SP, P, PCH, PCL, Op, Val]
            a = inp[:, 0].long()
            x = inp[:, 1].long()
            y = inp[:, 2].long()
            p = inp[:, 4].long()
            opcode = inp[:, 7].long()
            operand = inp[:, 8].long()
            carry_in = (p & 1).long()  # C flag is bit 0
            
            # Target: [A, X, Y, SP, P, PCH, PCL]
            target_a = tgt[:, 0].long()
            target_p = tgt[:, 4].long()
            
            # Extract target flags (N=bit7, Z=bit1, C=bit0, V=bit6)
            target_flags = torch.stack([
                (target_p >> 7) & 1,  # N
                (target_p >> 1) & 1,  # Z
                target_p & 1,         # C
                (target_p >> 6) & 1,  # V
            ], dim=-1).float()
            
            # Forward
            optimizer.zero_grad()
            result_logits, flags_logits = model(a, operand, carry_in, opcode)
            
            # Loss
            loss, _ = alu_loss(result_logits, flags_logits, target_a, target_flags, soroban)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy (decode and compare)
            with torch.no_grad():
                result_probs = torch.sigmoid(result_logits)
                pred_result = soroban.decode(result_probs.cpu())
                correct += (pred_result == target_a.cpu()).sum().item()
                total += len(a)
        
        elapsed = time.time() - start_time
        avg_loss = total_loss / len(loader)
        accuracy = correct / total * 100
        
        if verbose:
            print(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, acc={accuracy:.1f}%, time={elapsed:.1f}s")
    
    return model


def train_fu_logic(
    train_data: Tuple[torch.Tensor, torch.Tensor],
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 0.001,
    device: str = 'cuda',
    verbose: bool = True
) -> LogicModule:
    """Train the Logic Functional Unit."""
    
    inputs, targets = train_data
    
    if verbose:
        print(f"\n{'='*60}")
        print("Training FU_LOGIC (Binary Encoded)")
        print(f"{'='*60}")
        print(f"Samples: {len(inputs):,}")
    
    config = LogicConfig()
    model = LogicModule(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    if verbose:
        print(f"Parameters: {model.num_parameters:,}")
    
    dataset = FUDataset(inputs, targets, FU_LOGIC)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for batch in loader:
            inp = batch['input'].to(device)
            tgt = batch['target'].to(device)
            
            a = inp[:, 0].long()
            p = inp[:, 4].long()
            opcode = inp[:, 7].long()
            operand = inp[:, 8].long()
            carry_in = (p & 1).long()
            
            target_a = tgt[:, 0].long()
            target_p = tgt[:, 4].long()
            target_flags = torch.stack([
                (target_p >> 7) & 1,
                (target_p >> 1) & 1,
                target_p & 1,
                (target_p >> 6) & 1,
            ], dim=-1).float()
            
            optimizer.zero_grad()
            result_logits, flags_logits = model(a, operand, carry_in, opcode)
            loss, _ = logic_loss(result_logits, flags_logits, target_a, target_flags)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            with torch.no_grad():
                pred = model._decode_binary(torch.sigmoid(result_logits))
                correct += (pred == target_a).sum().item()
                total += len(a)
        
        elapsed = time.time() - start_time
        accuracy = correct / total * 100
        
        if verbose:
            print(f"Epoch {epoch+1}/{epochs}: loss={total_loss/len(loader):.4f}, acc={accuracy:.1f}%, time={elapsed:.1f}s")
    
    return model


def train_fu_move(
    train_data: Tuple[torch.Tensor, torch.Tensor],
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 0.001,
    device: str = 'cuda',
    verbose: bool = True
) -> MoveModule:
    """Train the Move Functional Unit."""
    
    inputs, targets = train_data
    
    if verbose:
        print(f"\n{'='*60}")
        print("Training FU_MOVE (Binary/Passthrough)")
        print(f"{'='*60}")
        print(f"Samples: {len(inputs):,}")
    
    config = MoveConfig()
    model = MoveModule(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    if verbose:
        print(f"Parameters: {model.num_parameters:,}")
    
    dataset = FUDataset(inputs, targets, FU_MOVE)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for batch in loader:
            inp = batch['input'].to(device)
            tgt = batch['target'].to(device)
            
            a = inp[:, 0].long()
            x = inp[:, 1].long()
            y = inp[:, 2].long()
            opcode = inp[:, 7].long()
            operand = inp[:, 8].long()
            
            target_a = tgt[:, 0].long()
            target_x = tgt[:, 1].long()
            target_y = tgt[:, 2].long()
            target_p = tgt[:, 4].long()
            target_nz = torch.stack([
                (target_p >> 7) & 1,
                (target_p >> 1) & 1,
            ], dim=-1)
            
            optimizer.zero_grad()
            output_logits = model(a, x, y, operand, opcode)
            loss, _ = move_loss(output_logits, target_a, target_x, target_y, target_nz)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            with torch.no_grad():
                probs = torch.sigmoid(output_logits)
                pred_a = model._decode_binary(probs[:, :8])
                correct += (pred_a == target_a).sum().item()
                total += len(a)
        
        elapsed = time.time() - start_time
        accuracy = correct / total * 100
        
        if verbose:
            print(f"Epoch {epoch+1}/{epochs}: loss={total_loss/len(loader):.4f}, acc={accuracy:.1f}%, time={elapsed:.1f}s")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Swarm 6502 Functional Units")
    parser.add_argument("--fu", type=str, default="alu", 
                        choices=["alu", "logic", "move", "flow", "stack", "all"],
                        help="Which FU to train")
    parser.add_argument("--shard-dir", type=str, default="cpu_shards",
                        help="Directory containing data shards")
    parser.add_argument("--max-shards", type=int, default=None,
                        help="Maximum shards to load (for testing)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="checkpoints/swarm",
                        help="Output directory for checkpoints")
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load and split data
    fu_map = build_fu_map()
    data = load_and_split_shards(args.shard_dir, fu_map, args.max_shards)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train requested FU(s)
    fus_to_train = []
    if args.fu == "all":
        fus_to_train = ["alu", "logic", "move"]
    else:
        fus_to_train = [args.fu]
    
    for fu_name in fus_to_train:
        if fu_name == "alu" and FU_ALU in data:
            model = train_fu_alu(
                data[FU_ALU], 
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=device
            )
            torch.save(model.state_dict(), f"{args.output_dir}/fu_alu.pt")
            print(f"Saved: {args.output_dir}/fu_alu.pt")
            
        elif fu_name == "logic" and FU_LOGIC in data:
            model = train_fu_logic(
                data[FU_LOGIC],
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=device
            )
            torch.save(model.state_dict(), f"{args.output_dir}/fu_logic.pt")
            print(f"Saved: {args.output_dir}/fu_logic.pt")
            
        elif fu_name == "move" and FU_MOVE in data:
            model = train_fu_move(
                data[FU_MOVE],
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=device
            )
            torch.save(model.state_dict(), f"{args.output_dir}/fu_move.pt")
            print(f"Saved: {args.output_dir}/fu_move.pt")
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)


if __name__ == "__main__":
    main()
