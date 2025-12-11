"""
Neural Calculator MCP - Phase 1 Proof of Concept

The "Hello World" of Neural Function Virtualization.

Ground truth: Python arithmetic
Neural shadow: Soroban-encoded micro-model
Goal: 99%+ accuracy, 100x latency improvement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import Tuple, List, Optional
from dataclasses import dataclass
from enum import IntEnum


# ============================================================
# OPERATOR DEFINITIONS
# ============================================================

class Op(IntEnum):
    ADD = 0
    SUB = 1
    MUL = 2
    DIV = 3


OP_SYMBOLS = {Op.ADD: '+', Op.SUB: '-', Op.MUL: '*', Op.DIV: '/'}


# ============================================================
# SOROBAN ENCODING
# ============================================================

def soroban_encode_int(x: int, bits: int = 16, radix: int = 16) -> torch.Tensor:
    """
    Encode integer as radix-thermometer.
    
    For 16-bit range with radix-16:
    - 4 nibbles × 16 bits = 64 features
    """
    # Clamp to range
    max_val = (1 << bits) - 1
    x = max(0, min(x, max_val))
    
    n_digits = bits // 4  # 4 bits per nibble
    thermometers = []
    
    for i in range(n_digits):
        digit = (x >> (4 * i)) & 0x0F
        therm = torch.zeros(radix)
        for j in range(radix):
            therm[j] = 1.0 if digit > j else 0.0
        thermometers.append(therm)
    
    return torch.cat(thermometers)


def soroban_decode_int(encoded: torch.Tensor, bits: int = 16, radix: int = 16) -> int:
    """
    Decode radix-thermometer to integer.
    """
    n_digits = bits // 4
    value = 0
    
    for i in range(n_digits):
        therm = encoded[i * radix : (i + 1) * radix]
        digit = int((therm > 0.5).sum().item())
        digit = min(digit, 15)  # Clamp to valid nibble
        value |= (digit << (4 * i))
    
    return value


def soroban_encode_float(x: float, int_bits: int = 12, frac_bits: int = 4) -> torch.Tensor:
    """
    Encode float as fixed-point Soroban.
    
    Split into integer part and fractional part.
    """
    # Handle sign separately
    sign = 1.0 if x >= 0 else 0.0
    x = abs(x)
    
    # Split into integer and fraction
    int_part = int(x)
    frac_part = int((x - int_part) * (1 << frac_bits))
    
    # Encode each part
    int_encoded = soroban_encode_int(int_part, bits=int_bits)
    frac_encoded = soroban_encode_int(frac_part, bits=frac_bits)
    sign_tensor = torch.tensor([sign])
    
    return torch.cat([sign_tensor, int_encoded, frac_encoded])


def soroban_decode_float(encoded: torch.Tensor, int_bits: int = 12, frac_bits: int = 4) -> float:
    """
    Decode fixed-point Soroban to float.
    """
    sign = 1.0 if encoded[0] > 0.5 else -1.0
    
    int_size = (int_bits // 4) * 16
    int_encoded = encoded[1 : 1 + int_size]
    frac_encoded = encoded[1 + int_size:]
    
    int_part = soroban_decode_int(int_encoded, bits=int_bits)
    frac_part = soroban_decode_int(frac_encoded, bits=frac_bits)
    
    return sign * (int_part + frac_part / (1 << frac_bits))


# ============================================================
# GROUND TRUTH CALCULATOR
# ============================================================

def ground_truth_calc(a: float, b: float, op: Op) -> float:
    """The real calculator. Our oracle."""
    if op == Op.ADD:
        return a + b
    elif op == Op.SUB:
        return a - b
    elif op == Op.MUL:
        return a * b
    elif op == Op.DIV:
        return a / b if b != 0 else 0.0
    else:
        raise ValueError(f"Unknown op: {op}")


# ============================================================
# DATA GENERATION
# ============================================================

@dataclass
class CalcSample:
    a: float
    b: float
    op: Op
    result: float


def generate_dataset(n_samples: int, 
                     int_range: Tuple[int, int] = (0, 255),
                     ops: List[Op] = None) -> List[CalcSample]:
    """Generate training/test samples."""
    if ops is None:
        ops = [Op.ADD, Op.SUB, Op.MUL, Op.DIV]
    
    samples = []
    for _ in range(n_samples):
        a = np.random.randint(int_range[0], int_range[1] + 1)
        b = np.random.randint(int_range[0], int_range[1] + 1)
        op = np.random.choice(ops)
        
        # Avoid division by zero
        if op == Op.DIV and b == 0:
            b = 1
        
        result = ground_truth_calc(float(a), float(b), op)
        samples.append(CalcSample(float(a), float(b), op, result))
    
    return samples


def encode_sample(sample: CalcSample) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode a sample for the neural model."""
    # Encode operands (8-bit integers → 32 features each)
    a_enc = soroban_encode_int(int(sample.a), bits=8)
    b_enc = soroban_encode_int(int(sample.b), bits=8)
    
    # Encode operator (one-hot)
    op_enc = torch.zeros(4)
    op_enc[sample.op] = 1.0
    
    # Combine inputs
    x = torch.cat([a_enc, b_enc, op_enc])
    
    # Encode result (16-bit to handle multiplication overflow)
    result_clamped = max(0, min(int(sample.result), 65535))
    y = soroban_encode_int(result_clamped, bits=16)
    
    return x, y


# ============================================================
# NEURAL CALCULATOR MODEL
# ============================================================

class NeuralCalculator(nn.Module):
    """
    The Neural Shadow of the calculator.
    
    Input: Soroban(a) + Soroban(b) + OneHot(op) = 32 + 32 + 4 = 68 features
    Output: Soroban(result) = 64 features
    """
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        
        self.input_dim = 32 + 32 + 4  # a + b + op
        self.output_dim = 64  # 16-bit result
        
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim),
            nn.Sigmoid()  # Thermometer bits are 0-1
        )
        
        # Confidence head
        self.confidence = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        result = self.net(x)
        conf = self.confidence(x)
        return result, conf


# ============================================================
# NEURAL MCP WRAPPER
# ============================================================

class CalculatorMCP:
    """
    The Neural MCP for calculator operations.
    
    Modes:
    - differentiable: Always use neural (gradients flow)
    - speculative: Use neural if confident, else fallback
    - ground_truth: Always use real calculator
    """
    
    def __init__(self, model: NeuralCalculator, confidence_threshold: float = 0.95):
        self.model = model
        self.threshold = confidence_threshold
        self.stats = {
            'neural_calls': 0,
            'fallback_calls': 0,
            'total_neural_time': 0.0,
            'total_fallback_time': 0.0,
        }
    
    def call(self, a: float, b: float, op: Op, 
             mode: str = 'speculative') -> Tuple[float, float]:
        """
        Execute calculator operation.
        
        Returns: (result, confidence)
        """
        if mode == 'ground_truth':
            return ground_truth_calc(a, b, op), 1.0
        
        # Encode inputs
        sample = CalcSample(a, b, op, 0.0)
        x, _ = encode_sample(sample)
        x = x.unsqueeze(0)  # Batch dim
        
        # Neural inference
        start = time.perf_counter()
        with torch.no_grad():
            result_enc, conf = self.model(x)
        neural_time = time.perf_counter() - start
        
        result_enc = result_enc.squeeze(0)
        conf_val = conf.item()
        
        # Decode result
        neural_result = float(soroban_decode_int(result_enc, bits=16))
        
        if mode == 'differentiable':
            self.stats['neural_calls'] += 1
            self.stats['total_neural_time'] += neural_time
            return neural_result, conf_val
        
        elif mode == 'speculative':
            if conf_val >= self.threshold:
                self.stats['neural_calls'] += 1
                self.stats['total_neural_time'] += neural_time
                return neural_result, conf_val
            else:
                # Fallback to ground truth
                start = time.perf_counter()
                real_result = ground_truth_calc(a, b, op)
                fallback_time = time.perf_counter() - start
                
                self.stats['fallback_calls'] += 1
                self.stats['total_fallback_time'] += fallback_time
                return real_result, 1.0
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def call_batch(self, samples: List[CalcSample], 
                   mode: str = 'differentiable') -> torch.Tensor:
        """
        Batch inference - fully differentiable.
        """
        # Encode all samples
        xs = []
        for s in samples:
            x, _ = encode_sample(s)
            xs.append(x)
        
        x_batch = torch.stack(xs)
        
        # Forward pass
        result_enc, conf = self.model(x_batch)
        
        return result_enc, conf
    
    def get_stats(self) -> dict:
        stats = self.stats.copy()
        total_calls = stats['neural_calls'] + stats['fallback_calls']
        if total_calls > 0:
            stats['neural_ratio'] = stats['neural_calls'] / total_calls
            stats['avg_neural_time_ms'] = (stats['total_neural_time'] / 
                                           max(1, stats['neural_calls'])) * 1000
            stats['avg_fallback_time_ms'] = (stats['total_fallback_time'] / 
                                             max(1, stats['fallback_calls'])) * 1000
        return stats


# ============================================================
# TRAINING
# ============================================================

def train_neural_calculator(n_samples: int = 100000,
                           n_epochs: int = 50,
                           batch_size: int = 256,
                           lr: float = 0.001,
                           hidden_dim: int = 256) -> NeuralCalculator:
    """Train the neural calculator."""
    
    print("=" * 70)
    print("       NEURAL CALCULATOR MCP - Training")
    print("=" * 70)
    
    # Generate data
    print(f"\n[1] Generating {n_samples:,} training samples...")
    train_data = generate_dataset(n_samples)
    
    # Encode data
    print("[2] Encoding with Soroban...")
    X, Y = [], []
    for sample in train_data:
        x, y = encode_sample(sample)
        X.append(x)
        Y.append(y)
    
    X = torch.stack(X)
    Y = torch.stack(Y)
    
    print(f"    Input shape: {X.shape}")
    print(f"    Output shape: {Y.shape}")
    
    # Create model
    print(f"\n[3] Creating model (hidden={hidden_dim})...")
    model = NeuralCalculator(hidden_dim=hidden_dim)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {n_params:,}")
    
    # Training
    print(f"\n[4] Training for {n_epochs} epochs...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    n_batches = len(X) // batch_size
    
    for epoch in range(n_epochs):
        # Shuffle
        perm = torch.randperm(len(X))
        X = X[perm]
        Y = Y[perm]
        
        epoch_loss = 0.0
        for i in range(n_batches):
            batch_x = X[i*batch_size : (i+1)*batch_size]
            batch_y = Y[i*batch_size : (i+1)*batch_size]
            
            optimizer.zero_grad()
            pred, _ = model(batch_x)
            loss = F.mse_loss(pred, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / n_batches
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            # Evaluate accuracy
            with torch.no_grad():
                pred, _ = model(X[:1000])
                pred_vals = [soroban_decode_int(p, bits=16) for p in pred]
                true_vals = [soroban_decode_int(y, bits=16) for y in Y[:1000]]
                
                exact = sum(1 for p, t in zip(pred_vals, true_vals) if p == t)
                close = sum(1 for p, t in zip(pred_vals, true_vals) if abs(p - t) <= 1)
                
                print(f"    Epoch {epoch+1:3d}: loss={avg_loss:.6f}, "
                      f"exact={exact/10:.1f}%, close={close/10:.1f}%")
    
    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate_calculator(model: NeuralCalculator, n_test: int = 10000):
    """Comprehensive evaluation of the neural calculator."""
    
    print("\n" + "=" * 70)
    print("       EVALUATION")
    print("=" * 70)
    
    mcp = CalculatorMCP(model)
    
    # Generate test data
    test_data = generate_dataset(n_test)
    
    # Test each operation
    for op in [Op.ADD, Op.SUB, Op.MUL, Op.DIV]:
        op_samples = [s for s in test_data if s.op == op]
        
        exact = 0
        close = 0
        total_error = 0.0
        
        for sample in op_samples:
            neural_result, _ = mcp.call(sample.a, sample.b, sample.op, mode='differentiable')
            true_result = int(sample.result)
            
            if int(neural_result) == true_result:
                exact += 1
            if abs(int(neural_result) - true_result) <= 1:
                close += 1
            total_error += abs(neural_result - true_result)
        
        n = len(op_samples)
        print(f"\n  {OP_SYMBOLS[op]} ({op.name}):")
        print(f"    Exact match: {exact}/{n} ({100*exact/n:.2f}%)")
        print(f"    Within ±1:   {close}/{n} ({100*close/n:.2f}%)")
        print(f"    Mean error:  {total_error/n:.2f}")
    
    # Latency comparison
    print("\n  Latency Comparison:")
    
    # Neural timing
    start = time.perf_counter()
    for _ in range(1000):
        mcp.call(123, 45, Op.ADD, mode='differentiable')
    neural_time = (time.perf_counter() - start) / 1000 * 1000  # ms
    
    # Ground truth timing
    start = time.perf_counter()
    for _ in range(1000):
        ground_truth_calc(123, 45, Op.ADD)
    gt_time = (time.perf_counter() - start) / 1000 * 1000  # ms
    
    print(f"    Neural:       {neural_time:.4f} ms/call")
    print(f"    Ground truth: {gt_time:.4f} ms/call")
    print(f"    Speedup:      {gt_time/neural_time:.1f}x")
    
    return mcp


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # Train
    model = train_neural_calculator(
        n_samples=100000,
        n_epochs=50,
        hidden_dim=256
    )
    
    # Evaluate
    mcp = evaluate_calculator(model)
    
    # Save
    torch.save(model.state_dict(), '/workspace/two-be/checkpoints/swarm/calculator_mcp.pt')
    print("\n[+] Model saved to checkpoints/swarm/calculator_mcp.pt")
    
    print("\n" + "=" * 70)
    print("       NEURAL CALCULATOR MCP - OPERATIONAL")
    print("=" * 70)
