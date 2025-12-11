# Clarity Journal

*Stopping to think before acting*

---

## What I Did Wrong

1. **Ghost scripts** - Running experiments in ephemeral Python instead of tracked files
2. **Reinvented the wheel** - Wrote my own QAT code instead of using the existing TriX kernel
3. **Didn't understand before building** - Created abstractions on top of code I hadn't properly read
4. **Rushing** - Producing output instead of thinking

---

## What the TriX Kernel Actually Is

Looking at `bbdos/kernel/trix.cpp` and `bindings.py`:

### The Core
- **2-bit packed ternary weights**: {-1, 0, +1} encoded as {0x02, 0x00, 0x01}
- **NEON SIMD acceleration** for ARM
- **Tile-based gating**: Only compute active tiles (sparse routing)
- **4x memory reduction** from packing

### The Training Flow (from TriXLinear)
```python
def forward(self, x, gate):
    if self._packed and not self.training:
        # Inference: Use NEON kernel with packed weights
        return trix_forward(x, self.packed_weight, ...)
    else:
        # Training: PyTorch with sign quantization
        w = torch.sign(self.weight)
        out = torch.mm(x, w.t()) * self.scales
        return out
```

**The training already uses `torch.sign()`** - hard ternary quantization.

The problem: `torch.sign()` has zero gradient almost everywhere.

### Why My QAT Code Was Wrong
I created `SoftTernaryQuantizer` and `progressive_quantization_schedule` - but this is fundamentally different from what TriX does. TriX uses hard sign() during training.

---

## What Actually Needs to Happen

### Option A: Accept TriX as-is
The existing TriXLinear uses sign() during training. This means:
- Weights are initialized, then quantized to ternary via sign()
- Gradients flow through the continuous weights
- Sign() is applied fresh each forward pass

This is actually a form of STE already - the continuous weights are updated by gradients, but sign() is applied for the forward pass.

### Option B: Add Proper STE to TriXLinear
Modify the forward to use a proper straight-through estimator:
```python
class STESign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.sign(x)
    
    @staticmethod  
    def backward(ctx, grad):
        return grad  # Pass through
```

Then replace `torch.sign(self.weight)` with `STESign.apply(self.weight)`.

---

## The Real Question

Why did the continuous training work (94%) but ternary collapse (0%)?

Looking at my experiment:
- Continuous weights learned the task
- Applying sign() after training destroyed it

But TriXLinear applies sign() DURING training. So either:
1. The initialization matters enormously
2. The scales need to be learned jointly
3. There's something about the task that's incompatible with ternary

---

## What I Should Do Now

1. **Use the actual TriXLinear** from `bbdos/kernel/bindings.py`
2. **Test if it can learn the simple task** (retrieve and add)
3. **If it fails, understand WHY** before trying to fix it
4. **Document everything in tracked files**

---

## The Path

1. Create a minimal test using the real TriXLinear
2. See what happens
3. Understand before optimizing

No more ghost scripts. No more reinventing. Use what exists.

---

## Experiment Results

Ran `experiments/test_real_trix.py` using the actual TriXLinear.

### Finding: Weight Gradients Are Zero

```
Weight gradient: 0.000000
[✗] No gradient to weights
```

The `torch.sign()` in TriXLinear.forward() has zero gradient everywhere.
Weights receive NO gradient. Only scales learn.

### Consequence

The model barely learns:
- Noise=0.10: Exact=2.2%
- Noise=0.15: Exact=3.6%
- Noise=0.20: Exact=5.5%

This is essentially random.

### Root Cause

```python
def forward(self, x, gate):
    w = torch.sign(self.weight)  # <-- Zero gradient!
    out = torch.mm(x, w.t()) * self.scales
    return out
```

### Solution Needed

Add a Straight-Through Estimator to the kernel bindings:

```python
class STESign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.sign(x)
    
    @staticmethod
    def backward(ctx, grad):
        return grad  # Pass gradient through

# Then in forward:
w = STESign.apply(self.weight)
```

This is a modification to `bbdos/kernel/bindings.py`.
