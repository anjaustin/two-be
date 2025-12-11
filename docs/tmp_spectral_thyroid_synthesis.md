# Synthesis: The Spectral Thyroid

*Concrete implementation plan for homeostatic training regulation*

---

## Executive Summary

**What**: A micro-model that observes training dynamics in the frequency domain and outputs real-time hyperparameter adjustments.

**Why**: 2-bit/quantized training suffers from "quantization cliff"—high-frequency noise from discrete weight transitions that confuses standard optimizers. The Thyroid filters this noise and maintains stability.

**How**: 1D convnet analyzes sliding window of loss/gradient history, outputs (α, λ, τ) multipliers for learning rate, regularization, and quantization temperature.

**When**: Phase 2 of BBDOS scaling. Ship Voltron first, bring Thyroid for larger models.

---

## The Design Principle

```
┌─────────────────────────────────────────────────────────────────────┐
│                     THE REPRESENTATION PRINCIPLE                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Problem Domain    Hidden Structure       Solution                 │
│   ──────────────    ────────────────       ────────                 │
│   Arithmetic        Carry propagation   →  Soroban (thermometer)   │
│   Training          Learning dynamics   →  Spectral (frequency)    │
│                                                                     │
│   Pattern: Transform to domain where structure is visible           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Architecture

```
                    ┌─────────────────────────────┐
                    │      TRAINING LOOP          │
                    │                             │
                    │   loss_t = train_step()     │
                    │            │                │
                    │            ▼                │
                    │   ┌─────────────────┐       │
                    │   │  Loss Buffer    │       │
                    │   │ [L_t...L_t-15]  │       │
                    │   └────────┬────────┘       │
                    │            │                │
                    └────────────┼────────────────┘
                                 │
                                 ▼
╔════════════════════════════════════════════════════════════════════╗
║                       SPECTRAL THYROID                             ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║   ┌──────────────────────────────────────────────────────────┐    ║
║   │                   SPECTRAL ANALYZER                       │    ║
║   │                   (Learned Filter Bank)                   │    ║
║   │                                                           │    ║
║   │   Input: [L_t, L_t-1, ..., L_t-15]  (16 samples)         │    ║
║   │                      │                                    │    ║
║   │                      ▼                                    │    ║
║   │            ┌─────────────────┐                           │    ║
║   │            │  Conv1D(1→8)    │  ← Learns low-freq filter │    ║
║   │            │  kernel=3       │                           │    ║
║   │            └────────┬────────┘                           │    ║
║   │                     ▼                                    │    ║
║   │            ┌─────────────────┐                           │    ║
║   │            │  Conv1D(8→16)   │  ← Learns high-freq filter│    ║
║   │            │  kernel=3       │                           │    ║
║   │            └────────┬────────┘                           │    ║
║   │                     ▼                                    │    ║
║   │            ┌─────────────────┐                           │    ║
║   │            │  AdaptivePool   │  → 4 frequency bands      │    ║
║   │            │  → 4 bins       │                           │    ║
║   │            └────────┬────────┘                           │    ║
║   │                     │                                    │    ║
║   │   Output: [Band0, Band1, Band2, Band3]  (64 features)    │    ║
║   └─────────────────────┼────────────────────────────────────┘    ║
║                         │                                          ║
║                         ▼                                          ║
║   ┌──────────────────────────────────────────────────────────┐    ║
║   │                   REGULATION HEADS                        │    ║
║   │                                                           │    ║
║   │    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │    ║
║   │    │   α HEAD    │  │   λ HEAD    │  │   τ HEAD    │     │    ║
║   │    │             │  │             │  │             │     │    ║
║   │    │ LR Multiply │  │ Weight Dec  │  │ Quant Temp  │     │    ║
║   │    │   [0.1-2.0] │  │   [0.1-2.0] │  │   [0.5-2.0] │     │    ║
║   │    └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │    ║
║   │           │                │                │            │    ║
║   └───────────┼────────────────┼────────────────┼────────────┘    ║
║               │                │                │                  ║
╚═══════════════╪════════════════╪════════════════╪══════════════════╝
                │                │                │
                ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         OPTIMIZER                                   │
│                                                                     │
│   effective_lr = base_lr × α                                       │
│   effective_wd = base_wd × λ                                       │
│   quant_temp = base_temp × τ   (BBDOS-specific)                    │
│                                                                     │
│   optimizer.step(lr=effective_lr, weight_decay=effective_wd)       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Implementation

### Phase 1: Proof of Concept (~2 hours)

```python
class SpectralThyroid(nn.Module):
    """
    v0.1 - Minimal viable thyroid
    Observes: Loss history (16 steps)
    Outputs: α (LR multiplier)
    """
    def __init__(self, window=16):
        super().__init__()
        self.window = window
        self.buffer = []
        
        # Learned spectral analyzer
        self.analyzer = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
            nn.Linear(64, 16),
            nn.ReLU(),
        )
        
        # LR multiplier head
        self.alpha_head = nn.Sequential(
            nn.Linear(16, 1),
            nn.Sigmoid(),  # → [0, 1]
        )
        
    def observe(self, loss):
        """Record loss value."""
        self.buffer.append(loss)
        if len(self.buffer) > self.window:
            self.buffer.pop(0)
    
    def regulate(self):
        """Output LR multiplier."""
        if len(self.buffer) < self.window:
            return 1.0  # Not enough data yet
        
        x = torch.tensor(self.buffer).float().unsqueeze(0).unsqueeze(0)
        features = self.analyzer(x)
        alpha = self.alpha_head(features)
        
        # Map [0,1] → [0.1, 2.0]
        return 0.1 + alpha.item() * 1.9
```

**Test**: Run on organelle training, compare stability with/without Thyroid.

---

### Phase 2: BBDOS-Specific (~4 hours)

```python
class BBDOSThyroid(nn.Module):
    """
    v0.2 - Quantization-aware thyroid
    Observes: Loss history + gradient norm history
    Outputs: α (LR), λ (weight decay), τ (quant temperature)
    """
    def __init__(self, window=16):
        super().__init__()
        self.window = window
        self.loss_buffer = []
        self.grad_buffer = []
        
        # Dual-stream analyzer
        self.loss_stream = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.grad_stream = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        
        # Output heads
        self.alpha_head = nn.Linear(32, 1)  # LR
        self.lambda_head = nn.Linear(32, 1)  # Weight decay
        self.tau_head = nn.Linear(32, 1)    # Quant temperature
        
    def regulate(self):
        """Output (α, λ, τ) multipliers."""
        # ... (process both streams, fuse, output)
        pass
```

**Test**: Run on 2-bit model training, measure quantization cliff frequency.

---

### Phase 3: Per-Organelle Regulation (~8 hours)

```python
class OrganelleThyroid(nn.Module):
    """
    v0.3 - Per-organelle regulation
    Each organelle gets its own (α, λ) based on its gradient signature
    """
    def __init__(self, n_organelles, window=16):
        super().__init__()
        self.n_organelles = n_organelles
        
        # Shared analyzer backbone
        self.backbone = nn.Sequential(...)
        
        # Per-organelle heads
        self.alpha_heads = nn.ModuleList([
            nn.Linear(32, 1) for _ in range(n_organelles)
        ])
        self.lambda_heads = nn.ModuleList([
            nn.Linear(32, 1) for _ in range(n_organelles)
        ])
        
    def regulate(self, organelle_grads):
        """Output per-organelle (α, λ)."""
        # ... 
        pass
```

**Test**: Train full Voltron swarm, verify auto-tuning matches manual optimization.

---

### Phase 4: Integration (~4 hours)

```python
class ThyroidOptimizer:
    """
    Drop-in replacement for any optimizer.
    Wraps optimizer with Spectral Thyroid regulation.
    """
    def __init__(self, optimizer, thyroid, update_every=10):
        self.optimizer = optimizer
        self.thyroid = thyroid
        self.update_every = update_every
        self.step_count = 0
        
    def step(self, loss):
        # Observe
        self.thyroid.observe(loss.item())
        
        # Regulate (every N steps)
        if self.step_count % self.update_every == 0:
            alpha, lambda_, tau = self.thyroid.regulate()
            self._adjust_hyperparams(alpha, lambda_, tau)
        
        # Step
        self.optimizer.step()
        self.step_count += 1
        
    def _adjust_hyperparams(self, alpha, lambda_, tau):
        for group in self.optimizer.param_groups:
            group['lr'] = group['initial_lr'] * alpha
            group['weight_decay'] = group['initial_wd'] * lambda_
```

**Usage**:
```python
base_optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
thyroid = BBDOSThyroid()
optimizer = ThyroidOptimizer(base_optimizer, thyroid)

for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step(loss)  # Thyroid regulates automatically
```

---

## Training the Thyroid

### Self-Supervised Objective

```python
def thyroid_loss(loss_history, alpha_history):
    """
    Objective: Minimize loss variance while maintaining progress
    """
    # Stability: low variance in recent losses
    stability = loss_history[-8:].var()
    
    # Progress: loss should decrease over time
    progress = loss_history[-1] - loss_history[0]  # Negative = good
    
    # Smoothness: alpha shouldn't oscillate wildly
    smoothness = (alpha_history[1:] - alpha_history[:-1]).abs().mean()
    
    return stability - 0.1 * progress + 0.01 * smoothness
```

### Constraints

```python
# Prevent degenerate solutions
alpha = alpha.clamp(0.1, 2.0)   # Can't turn off learning
lambda_ = lambda_.clamp(0.1, 2.0)  # Can't remove all regularization
tau = tau.clamp(0.5, 2.0)       # Can't freeze quantization
```

---

## Success Metrics

| Metric | Baseline (Manual) | Target (Thyroid) |
|--------|-------------------|------------------|
| Training stability | Frequent manual intervention | Zero intervention |
| Time to convergence | X hours | ≤ X hours |
| Final accuracy | Y% | ≥ Y% |
| Hyperparameter tuning time | Hours of sweeps | Zero (auto) |
| Quantization cliff incidents | N per run | < N/10 per run |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Thyroid instability | Hard clamps on outputs, slow update rate |
| Thyroid overhead | Tiny model (10K params), update every N steps |
| Thyroid learns to cheat | Constraints prevent degenerate outputs |
| Thyroid hurts performance | A/B test, fallback to manual |

---

## Timeline

| Phase | Time | Deliverable |
|-------|------|-------------|
| 1. PoC | 2 hours | Working α-only Thyroid |
| 2. BBDOS | 4 hours | (α, λ, τ) Thyroid with gradient obs |
| 3. Per-Organelle | 8 hours | Multi-head regulation |
| 4. Integration | 4 hours | Drop-in optimizer wrapper |
| **Total** | **18 hours** | **Production-ready Spectral Thyroid** |

---

## The Thesis

> **Manual hyperparameter tuning is open-loop control for a closed-loop problem.**

The Spectral Thyroid closes the loop. It transforms "hyperparameter search" from a pre-training ritual into a continuous, adaptive process.

For 2-bit training specifically, this could be the difference between "possible with careful tuning" and "robust by default."

---

## Appendix: The Quantization Cliff Visualized

```
Standard Training (float32):
Loss ─────────────────────────────────────►
     ████████████████████████████████████
     Smooth descent, predictable dynamics

2-bit Training (no Thyroid):
Loss ─────────────────────────────────────►
     ████  ██  ████    ██████  ████
        ██    ██    ████      ██    ████
     Jagged, chaotic, optimizer confused

2-bit Training (with Thyroid):
Loss ─────────────────────────────────────►
     ████████████████████████████████████
     Thyroid filters noise, smooth descent restored
```

---

*Synthesis complete. Ready to build.*
