#!/usr/bin/env python3
"""
Neural 6502 Evaluation Script

Evaluates the trained Neural CPU model on opcode accuracy.

Usage:
    python evaluate_cpu.py --checkpoint checkpoints/cpu/best.pt
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/workspace/BBDOS")

from bbdos.cpu import NeuralCPU, CPUConfig

try:
    from py65.devices.mpu6502 import MPU
    HAS_PY65 = True
except ImportError:
    HAS_PY65 = False


# Standard 6502 opcode names
OPCODE_NAMES = {
    0x00: "BRK", 0x01: "ORA_IX", 0x05: "ORA_ZP", 0x06: "ASL_ZP",
    0x08: "PHP", 0x09: "ORA_IMM", 0x0A: "ASL_A", 0x0D: "ORA_ABS",
    0x10: "BPL", 0x18: "CLC", 0x20: "JSR", 0x24: "BIT_ZP",
    0x29: "AND_IMM", 0x2A: "ROL_A", 0x30: "BMI", 0x38: "SEC",
    0x48: "PHA", 0x49: "EOR_IMM", 0x4A: "LSR_A", 0x4C: "JMP_ABS",
    0x50: "BVC", 0x58: "CLI", 0x60: "RTS", 0x68: "PLA",
    0x69: "ADC_IMM", 0x6A: "ROR_A", 0x70: "BVS", 0x78: "SEI",
    0x85: "STA_ZP", 0x86: "STX_ZP", 0x88: "DEY", 0x8A: "TXA",
    0x8D: "STA_ABS", 0x90: "BCC", 0x98: "TYA", 0x9A: "TXS",
    0xA0: "LDY_IMM", 0xA2: "LDX_IMM", 0xA8: "TAY", 0xA9: "LDA_IMM",
    0xAA: "TAX", 0xB0: "BCS", 0xB8: "CLV", 0xBA: "TSX",
    0xC0: "CPY_IMM", 0xC8: "INY", 0xC9: "CMP_IMM", 0xCA: "DEX",
    0xD0: "BNE", 0xD8: "CLD", 0xE0: "CPX_IMM", 0xE8: "INX",
    0xE9: "SBC_IMM", 0xEA: "NOP", 0xF0: "BEQ", 0xF8: "SED",
}


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Determine state dict
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    
    # Check if this is a legacy checkpoint (has emb_A instead of register_emb)
    if 'emb_A.weight' in state_dict:
        print("  Detected legacy checkpoint format, using legacy model...")
        # Import legacy model
        sys.path.insert(0, "/workspace/BBDOS")
        from neural_cpu import NeuralCPU as LegacyNeuralCPU
        model = LegacyNeuralCPU().to(device)
        model.load_state_dict(state_dict)
    else:
        # Use v2 model
        if 'config' in ckpt:
            cfg = ckpt['config']['model']
            model_cfg = CPUConfig(
                d_model=cfg.get('d_model', 256),
                n_heads=cfg.get('n_heads', 4),
                n_layers=cfg.get('n_layers', 6),
                num_tiles=cfg.get('num_tiles', 4),
            )
        else:
            model_cfg = CPUConfig()
        
        model = NeuralCPU(model_cfg).to(device)
        model.load_state_dict(state_dict)
    
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params / 1e6:.2f}M")
    if 'val_accuracy' in ckpt:
        print(f"  Checkpoint accuracy: {ckpt['val_accuracy']:.2f}%")
    
    return model


def test_opcode(model, device, opcode: int, operand: int = 0x42, num_tests: int = 16):
    """Test a single opcode with various register states."""
    if not HAS_PY65:
        return None, None
    
    mpu = MPU()
    mpu.memory = bytearray(65536)
    
    correct = 0
    total = 0
    
    for a in range(0, 256, 16):
        # Setup CPU state
        mpu.pc = 0x0200
        mpu.a = a
        mpu.x = (a + 17) % 256
        mpu.y = (a + 31) % 256
        mpu.sp = 0xFF
        mpu.p = 0x00
        
        # Place opcode and operand in memory
        mpu.memory[0x0200] = opcode
        mpu.memory[0x0201] = operand
        mpu.memory[0x0202] = 0x00  # BRK as sentinel
        
        # Capture state before
        state = {
            'A': torch.tensor([mpu.a], device=device),
            'X': torch.tensor([mpu.x], device=device),
            'Y': torch.tensor([mpu.y], device=device),
            'SP': torch.tensor([mpu.sp], device=device),
            'P': torch.tensor([mpu.p], device=device),
            'PCH': torch.tensor([mpu.pc >> 8], device=device),
            'PCL': torch.tensor([mpu.pc & 0xFF], device=device),
            'Op': torch.tensor([opcode], device=device),
            'Val': torch.tensor([operand], device=device),
        }
        
        # Get model prediction
        with torch.no_grad():
            preds, _ = model(state)
        
        pred_state = {reg: logits.argmax(dim=-1).item() for reg, logits in preds.items()}
        
        # Execute actual opcode
        try:
            mpu.step()
        except:
            continue
        
        # Compare
        actual_state = {
            'A': mpu.a, 'X': mpu.x, 'Y': mpu.y,
            'SP': mpu.sp, 'P': mpu.p,
            'PCH': mpu.pc >> 8, 'PCL': mpu.pc & 0xFF
        }
        
        match = all(pred_state[r] == actual_state[r] for r in actual_state)
        if match:
            correct += 1
        total += 1
    
    if total == 0:
        return None, None
    
    return correct, total


def evaluate_all_opcodes(model, device):
    """Evaluate model on all standard opcodes."""
    print("\nEvaluating opcode accuracy...")
    
    results = {}
    
    for opcode in sorted(OPCODE_NAMES.keys()):
        correct, total = test_opcode(model, device, opcode)
        if correct is not None:
            accuracy = correct / total * 100
            results[opcode] = (correct, total, accuracy)
    
    return results


def print_results(results):
    """Print formatted evaluation results."""
    print("\n" + "=" * 70)
    print("Opcode Evaluation Results")
    print("=" * 70)
    print(f"{'Opcode':<8} {'Name':<12} {'Correct':<10} {'Total':<8} {'Accuracy':<10}")
    print("-" * 70)
    
    total_correct = 0
    total_tests = 0
    
    # Group by accuracy
    perfect = []
    good = []
    bad = []
    
    for opcode, (correct, total, accuracy) in sorted(results.items()):
        name = OPCODE_NAMES.get(opcode, f"0x{opcode:02X}")
        print(f"0x{opcode:02X}     {name:<12} {correct:<10} {total:<8} {accuracy:>6.1f}%")
        
        total_correct += correct
        total_tests += total
        
        if accuracy >= 95:
            perfect.append(name)
        elif accuracy >= 70:
            good.append(name)
        else:
            bad.append(name)
    
    overall = total_correct / total_tests * 100 if total_tests > 0 else 0
    
    print("-" * 70)
    print(f"{'TOTAL':<8} {'':<12} {total_correct:<10} {total_tests:<8} {overall:>6.1f}%")
    print("=" * 70)
    
    print(f"\nPerfect (≥95%): {', '.join(perfect[:10])}{'...' if len(perfect) > 10 else ''}")
    print(f"Good (70-95%): {', '.join(good[:10])}{'...' if len(good) > 10 else ''}")
    print(f"Broken (<70%): {', '.join(bad[:10])}{'...' if len(bad) > 10 else ''}")
    
    return overall


def run_test_programs(model, device):
    """Run simple test programs and verify output."""
    if not HAS_PY65:
        print("\nSkipping program tests (py65 not available)")
        return
    
    print("\n" + "=" * 70)
    print("Test Program Execution")
    print("=" * 70)
    
    programs = [
        {
            'name': 'load_store',
            'code': [0xA9, 0x42, 0xAA, 0xA8, 0x00],  # LDA #$42, TAX, TAY, BRK
            'expected': {'A': 0x42, 'X': 0x42, 'Y': 0x42}
        },
        {
            'name': 'increment',
            'code': [0xA2, 0x00, 0xE8, 0xE8, 0xE8, 0x00],  # LDX #0, INX, INX, INX, BRK
            'expected': {'X': 0x03}
        },
        {
            'name': 'decrement',
            'code': [0xA0, 0x05, 0x88, 0x88, 0x00],  # LDY #5, DEY, DEY, BRK
            'expected': {'Y': 0x03}
        },
    ]
    
    for prog in programs:
        print(f"\n{prog['name']}:")
        
        # Initialize
        state = {
            'A': torch.tensor([0], device=device),
            'X': torch.tensor([0], device=device),
            'Y': torch.tensor([0], device=device),
            'SP': torch.tensor([0xFF], device=device),
            'P': torch.tensor([0], device=device),
            'PCH': torch.tensor([0x02], device=device),
            'PCL': torch.tensor([0x00], device=device),
            'Op': torch.tensor([prog['code'][0]], device=device),
            'Val': torch.tensor([prog['code'][1] if len(prog['code']) > 1 else 0], device=device),
        }
        
        # Execute each instruction
        for i, opcode in enumerate(prog['code'][:-1]):  # Skip BRK
            operand = prog['code'][i + 1] if i + 1 < len(prog['code']) else 0
            
            state['Op'] = torch.tensor([opcode], device=device)
            state['Val'] = torch.tensor([operand], device=device)
            
            with torch.no_grad():
                preds, _ = model(state)
            
            # Update state with predictions
            for reg in ['A', 'X', 'Y', 'SP', 'P', 'PCH', 'PCL']:
                state[reg] = preds[reg].argmax(dim=-1)
        
        # Check results
        passed = True
        for reg, expected in prog['expected'].items():
            actual = state[reg].item()
            status = "✓" if actual == expected else "✗"
            print(f"  {reg}: {actual} (expected {expected}) {status}")
            if actual != expected:
                passed = False
        
        print(f"  Result: {'PASS' if passed else 'FAIL'}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Neural 6502")
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--quick', action='store_true', help='Quick evaluation (fewer tests)')
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    device = torch.device(args.device)
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Run evaluations
    if HAS_PY65:
        results = evaluate_all_opcodes(model, device)
        overall = print_results(results)
        run_test_programs(model, device)
        
        print(f"\n{'=' * 70}")
        print(f"OVERALL OPCODE ACCURACY: {overall:.1f}%")
        print(f"{'=' * 70}")
    else:
        print("\nWARNING: py65 not installed. Cannot run opcode evaluation.")
        print("Install with: pip install py65")


if __name__ == "__main__":
    main()
