#!/usr/bin/env python3
"""
Validate ADC Training Data

Checks:
1. Data format is correct
2. Ground truth matches recomputed values (A + M + C = Result)
3. All flags are computed correctly (N, Z, C, V)
4. No corrupt/NaN values
5. Distribution looks reasonable

Usage:
    python scripts/validate_adc_data.py --data path/to/dataset.pt
    python scripts/validate_adc_data.py --data path/to/dataset.pkl
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

def load_data(path):
    """Load dataset from .pt or .pkl file."""
    path = Path(path)
    
    if path.suffix == '.pt':
        return torch.load(path, weights_only=False)
    elif path.suffix == '.pkl':
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unknown file type: {path.suffix}")


def compute_adc(a, m, c_in):
    """Compute ADC result and flags."""
    total = int(a) + int(m) + int(c_in)
    result = total & 0xFF
    
    # Flags
    c_out = 1 if total > 255 else 0
    z_out = 1 if result == 0 else 0
    n_out = 1 if (result & 0x80) else 0
    
    # Overflow: V = ~(A ^ M) & (A ^ R) & 0x80
    a_sign = a & 0x80
    m_sign = m & 0x80
    r_sign = result & 0x80
    v_out = 1 if (a_sign == m_sign) and (a_sign != r_sign) else 0
    
    return result, c_out, z_out, n_out, v_out


def validate_dataset(data, max_samples=100000):
    """Validate the dataset."""
    
    print("="*60)
    print("ADC DATASET VALIDATION")
    print("="*60)
    
    # Detect format
    print("\n1. DETECTING FORMAT...")
    
    if isinstance(data, dict):
        print(f"   Format: Dictionary with keys {list(data.keys())}")
        
        # Check for expected keys
        required_keys = ['a', 'm', 'c_in', 'result']
        flag_keys = ['c_out', 'z_out', 'n_out', 'v_out']
        
        missing = [k for k in required_keys if k not in data]
        if missing:
            print(f"   ERROR: Missing required keys: {missing}")
            return False
        
        a = data['a']
        m = data['m']
        c_in = data['c_in']
        result = data['result']
        
        # Check for flag keys
        has_flags = all(k in data for k in flag_keys)
        if has_flags:
            c_out = data['c_out']
            z_out = data['z_out']
            n_out = data['n_out']
            v_out = data['v_out']
            print(f"   Flags: Present (c_out, z_out, n_out, v_out)")
        else:
            print(f"   Flags: Not present as separate fields")
            c_out = z_out = n_out = v_out = None
            
    elif isinstance(data, (list, tuple)):
        print(f"   Format: List/Tuple with {len(data)} elements")
        # Try to unpack
        if len(data) >= 4:
            a, m, c_in, result = data[:4]
            c_out = z_out = n_out = v_out = None
            if len(data) >= 8:
                c_out, z_out, n_out, v_out = data[4:8]
        else:
            print(f"   ERROR: Expected at least 4 elements")
            return False
    else:
        print(f"   ERROR: Unknown format: {type(data)}")
        return False
    
    # Convert to numpy/tensor for easier handling
    if isinstance(a, torch.Tensor):
        a = a.numpy()
        m = m.numpy()
        c_in = c_in.numpy()
        result = result.numpy()
        if c_out is not None:
            c_out = c_out.numpy()
            z_out = z_out.numpy()
            n_out = n_out.numpy()
            v_out = v_out.numpy()
    
    n_samples = len(a)
    print(f"\n2. BASIC STATS...")
    print(f"   Total samples: {n_samples:,}")
    print(f"   A range: [{a.min()}, {a.max()}]")
    print(f"   M range: [{m.min()}, {m.max()}]")
    print(f"   C_in values: {np.unique(c_in)}")
    print(f"   Result range: [{result.min()}, {result.max()}]")
    
    # Check for invalid values
    print(f"\n3. CHECKING FOR INVALID VALUES...")
    
    invalid_a = np.sum((a < 0) | (a > 255))
    invalid_m = np.sum((m < 0) | (m > 255))
    invalid_c = np.sum((c_in < 0) | (c_in > 1))
    invalid_r = np.sum((result < 0) | (result > 255))
    
    if invalid_a or invalid_m or invalid_c or invalid_r:
        print(f"   ERROR: Invalid values found!")
        print(f"   Invalid A: {invalid_a}")
        print(f"   Invalid M: {invalid_m}")
        print(f"   Invalid C_in: {invalid_c}")
        print(f"   Invalid Result: {invalid_r}")
        return False
    else:
        print(f"   All values in valid range [0-255] for A/M/Result, [0-1] for C_in")
    
    # Verify ground truth
    print(f"\n4. VERIFYING GROUND TRUTH (sampling {min(max_samples, n_samples):,} samples)...")
    
    sample_idx = np.random.choice(n_samples, min(max_samples, n_samples), replace=False)
    
    errors_result = 0
    errors_c = 0
    errors_z = 0
    errors_n = 0
    errors_v = 0
    
    error_examples = []
    
    for i, idx in enumerate(sample_idx):
        a_val = int(a[idx])
        m_val = int(m[idx])
        c_val = int(c_in[idx])
        r_val = int(result[idx])
        
        # Recompute
        expected_r, expected_c, expected_z, expected_n, expected_v = compute_adc(a_val, m_val, c_val)
        
        if r_val != expected_r:
            errors_result += 1
            if len(error_examples) < 5:
                error_examples.append(f"   Result: A={a_val}, M={m_val}, C={c_val} -> got {r_val}, expected {expected_r}")
        
        if c_out is not None:
            if int(c_out[idx]) != expected_c:
                errors_c += 1
            if int(z_out[idx]) != expected_z:
                errors_z += 1
            if int(n_out[idx]) != expected_n:
                errors_n += 1
            if int(v_out[idx]) != expected_v:
                errors_v += 1
    
    checked = len(sample_idx)
    
    print(f"   Checked: {checked:,} samples")
    print(f"   Result errors: {errors_result} ({errors_result/checked*100:.2f}%)")
    
    if c_out is not None:
        print(f"   C flag errors: {errors_c} ({errors_c/checked*100:.2f}%)")
        print(f"   Z flag errors: {errors_z} ({errors_z/checked*100:.2f}%)")
        print(f"   N flag errors: {errors_n} ({errors_n/checked*100:.2f}%)")
        print(f"   V flag errors: {errors_v} ({errors_v/checked*100:.2f}%)")
    
    if error_examples:
        print(f"\n   Example errors:")
        for ex in error_examples:
            print(ex)
    
    # Distribution analysis
    print(f"\n5. DISTRIBUTION ANALYSIS...")
    
    # Carry rate
    if c_out is not None:
        carry_rate = np.mean(c_out)
        print(f"   Carry rate: {carry_rate:.1%} (expected ~50%)")
    
    # Zero rate
    if z_out is not None:
        zero_rate = np.mean(z_out)
        print(f"   Zero rate: {zero_rate:.2%} (expected ~0.4%)")
    
    # Negative rate
    if n_out is not None:
        neg_rate = np.mean(n_out)
        print(f"   Negative rate: {neg_rate:.1%} (expected ~50%)")
    
    # Overflow rate
    if v_out is not None:
        overflow_rate = np.mean(v_out)
        print(f"   Overflow rate: {overflow_rate:.1%} (expected ~25%)")
    
    # A, M distribution (should be uniform)
    a_std = np.std(a)
    m_std = np.std(m)
    expected_std = 255 / np.sqrt(12)  # Uniform distribution std
    print(f"   A std: {a_std:.1f} (expected ~{expected_std:.1f} for uniform)")
    print(f"   M std: {m_std:.1f} (expected ~{expected_std:.1f} for uniform)")
    
    # Final verdict
    print(f"\n{'='*60}")
    
    total_errors = errors_result + errors_c + errors_z + errors_n + errors_v
    if total_errors == 0:
        print("VALIDATION PASSED - Dataset is CLEAN")
        print("="*60)
        return True
    else:
        print(f"VALIDATION FAILED - {total_errors} errors found")
        print("="*60)
        return False


def main():
    parser = argparse.ArgumentParser(description="Validate ADC training data")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset file")
    parser.add_argument("--max-samples", type=int, default=100000, help="Max samples to check")
    args = parser.parse_args()
    
    print(f"Loading: {args.data}")
    data = load_data(args.data)
    
    success = validate_dataset(data, args.max_samples)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
