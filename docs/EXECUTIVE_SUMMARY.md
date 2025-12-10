# Executive Summary

## The Path of Least Resistance: Representational Geometry and Neural Learnability

---

### The Discovery

We achieved **100% accuracy on neural network arithmetic**—a task where standard approaches achieve 3%. Not by scaling up, but by changing how we represent numbers.

### The Problem

Neural networks fail at basic arithmetic despite succeeding at far more complex tasks. Our baseline model scored 99.9% on data movement operations but only 3.1% on addition. The model had sufficient capacity. Something else was wrong.

### The Root Cause

Binary number representation hides mathematical structure from gradient-based learning. The numbers 127 and 128 differ by one, but in binary they share zero bits in common:

```
127 = 01111111
128 = 10000000
```

The learning algorithm cannot "see" that these values are adjacent. The path from 127 to 128 is invisible.

### The Solution

We developed "Soroban encoding"—a thermometer-style representation inspired by the Japanese abacus. Adjacent numbers have adjacent representations. The mathematical structure becomes visible.

### The Results

| Metric | Standard | Soroban | Change |
|--------|----------|---------|--------|
| Accuracy | 3.1% | 100% | +97% |
| Parameters | 2.4M | 60K | 40x fewer |
| Test samples | 5M | 5M | Zero errors |

### The Principle

**Representation determines learnability.** When task structure is visible to gradients, learning succeeds. When structure is hidden, learning fails. The task doesn't change—the landscape does.

### The Implication

Many "impossible" neural network tasks may be representation problems in disguise. The principle extends beyond arithmetic to any domain with hidden structure—including biological systems where threshold effects are ubiquitous.

### The Paradigm Shift

| Current Approach | New Approach |
|------------------|--------------|
| Scale up models | Align representations |
| More parameters | Better geometry |
| Fight the gradient | Collaborate with the gradient |

### The Bottom Line

> We sculpted a landscape where arithmetic was the path of least resistance.

This isn't a technique. It's a paradigm for unlocking neural network capabilities that the field has written off as impossible.

---

**Paper forthcoming.** 

**Contact:** [Your information]
