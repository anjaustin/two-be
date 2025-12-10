# Executive Summary

## The Path of Least Resistance
### Why AI Can Talk About Math But Can't Do It—And What We Did About It

---

### The Puzzle

Large Language Models can explain calculus, discuss number theory, and describe algorithms with remarkable fluency. Ask them to multiply two large numbers, and they fail.

**LLMs can talk about math. They struggle to do math.**

Why?

---

### The Root Cause

Our written representation of numbers is **symbolic abstraction**—arbitrary glyphs with no structural relationship to quantity.

```
"127" and "128"
```

These symbols share one character. But nothing in the symbols themselves indicates "these are adjacent quantities." The adjacency is a fact we know *about* them, not something *in* them.

LLMs learned from text. Text *talks about* math—it doesn't *do* math. When an LLM "adds," it's pattern-matching on what additions look like in text, not executing the operation.

**The representation is unhinged from the structure it describes.**

---

### The Insight

On a physical abacus, 127 and 128 differ by one bead. The representation *is* the quantity. Adjacency isn't described—it's embodied.

What if we gave neural networks representations where mathematical structure is physically present, not symbolically referenced?

---

### The Experiment

We built a neural network using "Soroban encoding"—a thermometer-style representation inspired by the Japanese abacus:

```
127: ●●●●●●●●●●●●●●●○ | ●●●●●●●○○○○○○○○○
128: ○○○○○○○○○○○○○○○○ | ●●●●●●●●○○○○○○○○
```

Adjacent values have adjacent representations. Quantity is structure.

---

### The Result

| Metric | Result |
|--------|--------|
| Accuracy | **100%** |
| Test samples | 5 million |
| Errors | Zero |
| Parameters | 60,000 |

Perfect arithmetic. Not by making the network smarter—by making the structure visible.

---

### The Principle

> **Neural networks can execute perfect arithmetic when given representations that embody arithmetic structure.**

The network doesn't learn "these symbols refer to adjacent quantities."
The network learns "these patterns are adjacent"—because they literally are.

---

### The Implication

Anywhere we use **abstract symbols** to represent **structured phenomena**, we may be hiding structure from learning systems:

| Domain | Symbolic Representation | Hidden Structure |
|--------|------------------------|------------------|
| Gene expression | Floating-point levels | Threshold switches |
| Drug response | Dose as scalar | Therapeutic windows |
| Neural activity | Firing rates | Spike thresholds |

The question: **What if we represented these phenomena in ways that embody their structure, rather than symbolically reference it?**

---

### The Paradigm Shift

| Current Framing | New Framing |
|-----------------|-------------|
| "AI can't do math" | "Text doesn't embody quantity" |
| "Scale up the model" | "Align the representation" |
| "Fundamental limitation" | "Representation mismatch" |

---

### The Bottom Line

LLMs can discuss arithmetic because language discusses arithmetic.

They can't execute arithmetic because text doesn't embody quantity—it just names it.

Give them a representation where quantity *is* structure, and they execute perfectly.

> **We sculpted a landscape where arithmetic was the path of least resistance.**

---

**Paper forthcoming.**

**Contact:** [Your information]
