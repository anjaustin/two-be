# Session Log: 2025-12-11

## Commitment

From this point forward:
- All code in tracked files
- All experiments documented
- All results logged
- Focus: TriX + NVF convergence

## Run 1: Progressive Quantization

### Configuration
```
Computer: n_slots=5, key_dim=32, num_tiles=4, hidden_dim=256
Training: batch=2048, epochs=500, lr=0.003
QAT: progressive mode, temp 1.0 → 10.0
```

### Results
```
Noise=0.10: Exact=90.3%
Noise=0.15: Exact=85.1%
Noise=0.20: Exact=71.7%
Noise=0.25: Exact=54.0%

Sparsity: 19.8%
Training time: 36.1s
Gradients: FLOW ✓
```

### Key Insight
Progressive quantization works. Start soft (temp=1), end hard (temp=10).
Model learns to work with constrained weights.

## Files Created
- `bbdos/trix/__init__.py` - Module init
- `bbdos/trix/qat.py` - QAT infrastructure (TernaryQuantizer, TriXLinearQAT)
- `bbdos/nvf/trix_computer.py` - TriX-based differentiable computer
- `experiments/train_trix_computer.py` - Training experiment

## Checkpoints
- `results/trix_computer_experiment.json` - Training history
- `results/trix_computer_checkpoint.pt` - Model weights

## Next Steps
1. Push accuracy higher (target: 95%+)
2. Test with actual ternary weights at inference
3. Validate NVF integration

---

## Critical Fix: TriX STE

### Problem Found
`torch.sign()` in TriXLinear had zero gradient. Weights never updated.

### Fix Applied
Modified `bbdos/kernel/bindings.py`:

1. Added `STESign` class - Straight-Through Estimator
2. Changed `_init_weights()` to keep continuous weights
3. Changed `forward()` to use `STESign.apply()` instead of `torch.sign()`

### Results After Fix

```
Weight gradient: 2.398747 (was 0.000000)
[✓] Gradients flow to weights

Training progress:
  Epoch 100: acc(±5)=29.6%
  Epoch 500: acc(±5)=77.6%

Final (Noise=0.15): Exact=13.3%, ±5=77.9%
```

Gradients flow. TriX learns.

---

## Fix 2: Threshold Matching C++ Kernel

### Problem
STESign used sign() which gives only {-1, +1}. No sparsity.
C++ kernel uses threshold 0.5: w > 0.5 → +1, w < -0.5 → -1, else 0.

### Fix Applied
1. Updated STESign to use threshold 0.5 (matching C++ kernel)
2. Changed weight init to uniform(-0.8, 0.8) for proper ternary bins
3. Now produces {-1, 0, +1} with ~63% sparsity

### Results After Fix

```
Epoch 100: acc(±5)=16.9%
Epoch 500: acc(±5)=48.3%  
Epoch 1000: acc(±5)=78.9%

Final (Noise=0.15): Exact=27.5%, ±5=73.8%
Sparsity: 63-82%
```

TriX learns with proper ternary quantization.
