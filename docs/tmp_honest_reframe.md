# Honest Reframe: What We Actually Have

---

## The Tainted Claim (Discarded)

~~"We went from 3.1% to 100% accuracy"~~

**Problem**: The 3.1% baseline came from a model with corrupt/incomplete data. We can't use our own broken experiment as a scientific reference point.

---

## The Honest Foundation

### What's Actually True (External, Documented)

**Modern AI systems systematically fail at arithmetic.**

This is not our claim. This is widely documented:

1. **GPT-4** fails at multi-digit multiplication and addition with carrying
2. **LLMs generally** show degraded performance as digit count increases
3. **Even specialized neural arithmetic models** (Neural GPUs, NALUs) achieve limited accuracy and struggle to generalize
4. **The pattern**: Arithmetic involving carry/borrow propagation is disproportionately hard for neural networks

This is the field's baseline, not ours.

### What We Built

A neural network that achieves **100% accuracy on 8-bit addition** (all 131,072 unique input combinations, verified on 5 million samples).

The key design choice: **Soroban encoding**—a thermometer-style representation where adjacent values have adjacent representations.

### What We Observed

With Soroban encoding:
- Perfect accuracy on carry-propagation arithmetic
- Small model (60K parameters)
- Fast convergence (~20 epochs)

### What We Hypothesize

The reason neural networks fail at arithmetic isn't lack of capability—it's that standard representations (binary, tokenized digits) hide the structure of arithmetic from gradient-based learning.

**Carry propagation is invisible in binary.** 127 and 128 share no bits. The gradient cannot "see" that they're adjacent.

**Carry propagation is visible in Soroban.** Adjacent values look adjacent. The gradient can navigate.

---

## The Reframed Story

### The Known Problem

Neural networks—including the most powerful LLMs—fail at arithmetic. This is documented, reproducible, and widely discussed. The common explanation: "Neural networks lack systematic reasoning."

### The Alternative Hypothesis

What if the failure isn't about capability, but about **representation**?

Binary and tokenized representations make adjacent numbers look distant. Carry propagation requires simultaneous bit flips that gradients can't coordinate. The structure of arithmetic is hidden from the learning process.

### The Evidence

We built a system using thermometer-style encoding (Soroban) that achieves perfect arithmetic accuracy. This doesn't prove the hypothesis, but it demonstrates that:

1. **Perfect neural arithmetic is achievable** (it's not fundamentally impossible)
2. **Representation choice matters dramatically** (same task, different encoding, different outcome)
3. **Structure visibility correlates with learnability** (Soroban makes carry structure explicit)

### The Question

**How many "impossible" neural network tasks are actually representation problems?**

We have one existence proof: arithmetic. The hypothesis suggests this pattern might be widespread—anywhere hidden structure exists.

---

## Horizontal Applications

### Within ML

| Domain | Known Failure | Hidden Structure | Representation Question |
|--------|---------------|------------------|------------------------|
| **Arithmetic** | LLMs fail at carrying | Carry propagation | Thermometer encoding? |
| **Symbolic reasoning** | Brittle generalization | Logical adjacency | Structure-preserving encoding? |
| **Counting** | Models miscount | Magnitude discreteness | Unary/thermometer? |
| **Temporal reasoning** | Sequence errors | Time adjacency | Continuous time encoding? |

### Beyond ML (Speculative)

| Domain | Modeling Challenge | Hidden Structure | Representation Question |
|--------|-------------------|------------------|------------------------|
| **Gene regulation** | Switch prediction | On/off thresholds | Threshold-aware encoding? |
| **Drug response** | Dose-response cliffs | Therapeutic windows | Window encoding? |
| **Protein folding** | Discrete state transitions | Folding cooperativity | Fold-state encoding? |
| **Neural activity** | Spike prediction | Action potential threshold | Spike-threshold encoding? |

These are hypotheses, not claims. But they follow the same pattern:

> **If a system has discrete transitions that current representations treat as continuous, a threshold-aware encoding might dramatically improve learnability.**

---

## What We're NOT Claiming

1. ~~"We solved neural arithmetic"~~ → We achieved perfect accuracy on 8-bit addition with a specific encoding
2. ~~"Representation is everything"~~ → Representation is ONE factor that matters, possibly undervalued
3. ~~"This will definitely work for biology"~~ → This is a hypothesis worth testing
4. ~~"Scale doesn't matter"~~ → Scale and representation both matter; the field may be over-indexed on scale

---

## What We ARE Claiming

1. **Perfect neural arithmetic is achievable** (existence proof)
2. **Representation choice dramatically affects arithmetic learnability** (demonstrated)
3. **The "neural networks can't do arithmetic" narrative may be wrong** (or at least incomplete)
4. **There's a hypothesis worth investigating**: representation-induced failures may be widespread

---

## The Honest Framing for Public Communication

### Lead with the known problem
"Modern AI systems—including GPT-4—systematically fail at arithmetic. This is well-documented."

### Present our result as existence proof
"We built a system that achieves 100% accuracy on 8-bit addition using a different representation."

### Offer the hypothesis
"We think the failure isn't about capability—it's about representation. Standard encodings hide arithmetic structure from gradients."

### Acknowledge limitations
"This is one task. We don't know how far this generalizes. But it raises a question worth asking."

### Invite inquiry
"How many 'impossible' neural network tasks might be representation problems?"

---

## The Core Claim, Honestly Stated

> **Neural networks can do perfect arithmetic when given representations that make arithmetic structure visible. This suggests that some 'fundamental limitations' of neural networks may actually be limitations of how we encode information for them.**

This is:
- **Supported by our results** (100% on 8-bit addition)
- **Connected to documented phenomena** (LLM arithmetic failures)
- **Appropriately bounded** (one task, hypothesis about generalization)
- **Genuinely novel** (if true, changes how we think about neural capabilities)

---

## The Question to Carry Forward

**What structures are hiding in your representations?**

If the answer is "threshold effects, discrete transitions, hidden adjacencies"—the Soroban insight might apply.

If not, it might not.

Either way, representation deserves more attention than it's getting.

---

*This is what we actually have. This is what we actually know.*
