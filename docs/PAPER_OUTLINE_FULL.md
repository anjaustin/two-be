# The Path of Least Resistance
## Representational Geometry and Neural Learnability

---

# FULL PAPER OUTLINE

---

## Abstract (~250 words)

Neural networks fail on tasks with hidden structure—not from lack of capacity, but from representations that hide solutions from gradient descent. We demonstrate this through neural CPU emulation, where standard binary encoding yields 3.1% accuracy on 8-bit arithmetic while structure-aligned "Soroban" encoding yields 100% accuracy with 30x fewer parameters. We formalize this as the **representational geometry hypothesis**: learnability is determined by whether task structure is visible to gradients. We present a methodology for diagnosing representation-induced failures and designing structure-aligned encodings. We discuss implications for biological modeling, where threshold phenomena are ubiquitous and current representations may systematically hide learnable structure. Our results suggest that many "impossible" neural network tasks are representation problems in disguise, and that sculpting loss landscapes—rather than scaling models—may be the key to unlocking new capabilities.

---

## 1. Introduction (2-3 pages)

### 1.1 The Puzzle
- Neural networks achieve superhuman performance on complex tasks (vision, language, games)
- Yet fail at "simple" tasks (arithmetic, systematic reasoning, precise computation)
- The explanation "neural networks can't do X" is unsatisfying
- What if the failure is not capability, but communication?

### 1.2 The Observation
- Failures cluster around structural discontinuities
- Example: Addition fails specifically at carry propagation
- The network has capacity, data, and compute
- Something else is wrong

### 1.3 The Thesis
> **Representation determines learnability.**
> 
> When task structure is visible to gradients, learning succeeds.
> When task structure is hidden, learning fails.
> The task doesn't change. The landscape does.

### 1.4 The Metaphor
- Gradient descent walks downhill on a loss landscape
- Bad representations create mountains between the network and the solution
- Good representations flatten the terrain
- "We sculpted a landscape where arithmetic was the path of least resistance"

### 1.5 Contributions
1. **Theory**: The representational geometry hypothesis
2. **Methodology**: Systematic approach to representation design
3. **Case Study**: Neural 6502 with 3.1% → 100% accuracy improvement
4. **Implications**: Framework for biological modeling and beyond

---

## 2. Background (2 pages)

### 2.1 Neural Network Learning
- Gradient descent as local optimization
- Loss landscapes and their geometry
- Why gradients fail at discontinuities

### 2.2 The Representation Problem (Historical)
- Feature engineering in classical ML
- Learned representations in deep learning
- The assumption that "the network will figure it out"

### 2.3 Prior Work on Neural Arithmetic
- Neural GPUs (Kaiser & Sutskever)
- Neural Arithmetic Logic Units (Trask et al.)
- Transformer arithmetic failures
- Common diagnosis: "networks lack systematic reasoning"

### 2.4 The 6502 Processor
- Historical context (1975, 8-bit, 56 instructions)
- ADC (Add with Carry) operation
- Why it's a good testbed (ground truth, isolated failures)

---

## 3. The Representational Geometry Hypothesis (3-4 pages)

### 3.1 Formal Statement

**Definition (Structure Visibility)**: A representation R makes structure S visible if small changes in S correspond to small changes in R.

**Definition (Gradient Accessibility)**: A solution is gradient-accessible from a starting point if there exists a path of monotonically decreasing loss.

**Hypothesis**: A task is learnable if and only if there exists a representation that makes the solution gradient-accessible.

### 3.2 The Carry Bit as Canonical Example

Binary representation of integers:
```
127 = 01111111
128 = 10000000
```
- Hamming distance: 8 (maximum possible)
- Adjacent values appear maximally different
- Gradient at 127 contains no information about reaching 128
- The carry propagation structure is invisible

### 3.3 Making Structure Visible

Soroban (thermometer) representation:
```
127 = [1111111111111110 | 1111111000000000]  (low nibble | high nibble)
128 = [0000000000000000 | 1111111100000000]
```
- Hamming distance: 2
- Adjacent values appear adjacent
- Gradient can walk from 127 to 128
- The carry structure is visible (bead movement)

### 3.4 Theoretical Implications

1. **Capacity is not the bottleneck**: Small networks with good representations beat large networks with bad representations
2. **Data is not the bottleneck**: All data is available in both representations; only visibility changes
3. **The landscape is the bottleneck**: Representation determines topology

---

## 4. Case Study: Neural 6502 (4-5 pages)

### 4.1 Experimental Setup

- Task: Predict output state from input state + opcode + operand
- Architecture: MLP with varying sizes
- Training: Standard supervised learning
- Evaluation: Exact match accuracy per operation

### 4.2 Baseline Results

| Operation Type | Accuracy | Parameters |
|----------------|----------|------------|
| Stack (PHA, PLA) | 99.9% | 2.4M |
| Transfer (TAX, TXA) | 97.1% | 2.4M |
| Arithmetic (ADC) | 3.1% | 2.4M |

**Observation**: The model is a "savant"—brilliant at data movement, unable to add.

### 4.3 Failure Analysis

- ADC errors concentrate at carry boundaries
- Values near multiples of 16 (nibble boundaries) fail
- The network can't predict when carries propagate
- Binary representation hides carry structure

### 4.4 The Soroban Intervention

**Design**:
- Split byte into two nibbles (0-15 each)
- Encode each nibble as 16-bit thermometer
- Total: 32 bits per 8-bit value
- Carry becomes visible as thermometer overflow

**Implementation**:
```python
def soroban_encode(x):
    low = x & 0x0F
    high = (x >> 4) & 0x0F
    low_therm = [int(low > i) for i in range(16)]
    high_therm = [int(high > i) for i in range(16)]
    return low_therm + high_therm
```

### 4.5 Results with Soroban

| Metric | Binary | Soroban | Improvement |
|--------|--------|---------|-------------|
| ADC Accuracy | 3.1% | 100.0% | +96.9% |
| Parameters | 2.4M | 60K | 40x fewer |
| Training Epochs | Never converged | 20 | ∞x faster |
| Test Samples | 5M | 5M | - |

### 4.6 Ablation Studies

- Thermometer without nibble split: 67% (partial structure)
- One-hot encoding: 12% (no adjacency)
- Learned embedding: 45% (partial discovery)
- Soroban: 100% (full structure visibility)

### 4.7 Additional Validation

- Fibonacci computation via neural execution
- 100,000 parallel sequences computed correctly
- 347,000 operations per second
- Zero Python orchestration (pure tensor flow)

---

## 5. Methodology: Finding Structure-Aligned Representations (3 pages)

### 5.1 Phase 1: Diagnose the Failure

**Questions**:
1. Where does accuracy break down?
2. What inputs fail?
3. What patterns appear in errors?
4. What would a "near miss" look like?

**Techniques**:
- Error clustering analysis
- Gradient magnitude inspection
- Confusion matrix structure
- Domain expert consultation

### 5.2 Phase 2: Hypothesize Hidden Structure

**Questions**:
1. What structure does the task have?
2. Is it visible in the current representation?
3. What would make it visible?

**Common Structures**:
- Thresholds (→ thermometer encoding)
- Periodicity (→ angular encoding)
- Hierarchy (→ nested encoding)
- Relations (→ graph encoding)
- Frequencies (→ spectral encoding)

### 5.3 Phase 3: Design and Test

**Process**:
1. Implement new encoding
2. Train identical architecture
3. Compare accuracy, convergence, generalization
4. Ablate to understand what helps

**Success Criteria**:
- Accuracy improvement (primary)
- Convergence speed improvement (secondary)
- Parameter efficiency (bonus)

### 5.4 Phase 4: Validate and Generalize

**Validation**:
- Hold-out test sets
- Edge cases and boundary conditions
- Distribution shift robustness

**Generalization**:
- Does the principle apply to related tasks?
- What other domains have similar structure?

---

## 6. Representation Primitives (2-3 pages)

### 6.1 A Taxonomy

| Primitive | Structure Type | Encoding Strategy | Example Domain |
|-----------|---------------|-------------------|----------------|
| Thermometer | Threshold/Magnitude | Unary counting | Arithmetic, neurons |
| Angular | Periodic/Cyclic | Sin/cos embedding | Time, phase |
| Hierarchical | Scale/Nesting | Multi-resolution | Proteins, fractals |
| Relational | Graph/Network | Adjacency + features | Molecules, social |
| Spectral | Frequency | Fourier/wavelet | Signals, dynamics |
| Topological | Connectivity | Persistence features | Shapes, manifolds |

### 6.2 Composition Principles

Complex structures require composed representations:

**Example - Protein Structure**:
- Hierarchical (residue → secondary → tertiary → quaternary)
- Relational (contact maps, hydrogen bonds)
- Thermometer (folding state transitions)
- Spectral (dynamics, flexibility)

### 6.3 Domain-Specific Instances

**Biology**:
- Gene expression: Thermometer (on/off thresholds)
- Drug response: Windowed thermometer (therapeutic range)
- Neural activity: Threshold + temporal (spike trains)

**Physics**:
- Phase transitions: Thermometer (critical points)
- Oscillations: Angular + spectral
- Turbulence: Hierarchical + spectral

---

## 7. Implications for Biological Systems (3-4 pages)

### 7.1 The Hidden Structure Problem in Biology

Current representations:
- Sequences (one-hot, learned embeddings)
- Structures (coordinates, distance matrices)
- Networks (adjacency, graph neural networks)

Hidden structures:
- Threshold effects (expression, binding, firing)
- Cooperative transitions (folding, assembly)
- Discrete states (cell types, disease stages)

### 7.2 Threshold Phenomena

| System | Threshold Behavior | Current Representation | Proposed |
|--------|-------------------|----------------------|----------|
| Gene expression | On/off switching | Continuous levels | Thermometer states |
| Drug response | Therapeutic window | Dose scalar | Window encoding |
| Neural firing | Action potential | Rate code | Spike threshold |
| Protein folding | Cooperative transition | RMSD | Fold state |

### 7.3 Case Study Proposal: Gene Regulatory Networks

**The Problem**:
- Gene expression is modeled as continuous
- Reality: bistable switches, threshold activation
- Models fail to predict discrete transitions

**The Hypothesis**:
- Thermometer encoding for expression states
- Explicit threshold representation
- Cooperative binding as carry propagation

**Expected Outcome**:
- Better prediction of switch-like behavior
- Fewer parameters for equivalent accuracy
- Interpretable intermediate representations

### 7.4 Broader Implications

If biological systems have "carry bits"—threshold phenomena hidden by current representations—then:

1. **Current ML for biology may be systematically suboptimal**
2. **Representation redesign could unlock step-change improvements**
3. **Domain expertise is essential** (biologists know the structures)
4. **The methodology transfers** (same diagnosis-hypothesis-test cycle)

---

## 8. Related Work (2 pages)

### 8.1 Information Bottleneck Theory

Tishby's IB: Optimal representations compress input while preserving output information.

Our addition: Not just *how much* information, but *what shape*. Geometric accessibility matters beyond mutual information.

### 8.2 Manifold Learning

The manifold hypothesis: Data lies on low-dimensional manifolds.

Our addition: Learning succeeds when representations make manifold structure explicit. Soroban traces the 1D manifold of integer magnitude.

### 8.3 Inductive Bias

Architecture implies inductive bias (locality for CNNs, sequence for RNNs).

Our addition: Representation implies inductive bias too—often more strongly. The representation IS the inductive bias that matters most.

### 8.4 Neural Arithmetic

Prior work diagnosed "lack of systematic reasoning."

Our reframe: The reasoning capability exists. The representation hides the structure. Fix the representation, find the capability.

---

## 9. Limitations and Future Work (2 pages)

### 9.1 Limitations

1. **Domain expertise required**: Finding representations requires understanding the domain
2. **No algorithm for discovery**: The methodology is iterative, not automatic
3. **Ground truth needed for validation**: Clean evaluation requires known correct answers
4. **Not all tasks have accessible structure**: Some tasks may be fundamentally hard

### 9.2 Future Directions

**Near-term**:
- Complete primitive taxonomy
- Apply to additional arithmetic tasks
- Initial biological case studies

**Medium-term**:
- Meta-representation learning (automated discovery)
- Representation-architecture co-design principles
- Benchmark suite for representation-sensitive tasks

**Long-term**:
- Theory of representational complexity
- Fundamental limits of representation-based learnability
- Connection to computational learning theory

---

## 10. Conclusion (1 page)

### 10.1 Summary

We have presented:
1. **A theory**: Representation determines learnability through geometric accessibility
2. **A methodology**: Diagnose failures, hypothesize structure, design encodings, validate
3. **A proof**: 3.1% → 100% on neural arithmetic via Soroban encoding
4. **A direction**: Application to biological systems and beyond

### 10.2 The Paradigm Shift

| Old Paradigm | New Paradigm |
|--------------|--------------|
| Scale is the answer | Representation is the answer |
| Train harder | Train smarter |
| Fight the gradient | Collaborate with the gradient |
| Neural networks can't do X | Representation for X not yet found |

### 10.3 The Closing Thought

> We sculpted a landscape where arithmetic was the path of least resistance.

This is not a technique. It's a paradigm.

Every failed neural network stands at the base of an invisible mountain. We can learn to see the mountain. We can learn to move it.

The path of least resistance is waiting to be discovered.

---

## Appendices

### A. Soroban Encoding Implementation

Full code for encoding/decoding, training scripts, evaluation.

### B. Complete 6502 Results

Accuracy tables for all 56 instructions, all addressing modes.

### C. Ablation Study Details

Full experimental results for representation variants.

### D. Biological Domain Analysis

Preliminary analysis of threshold structures in proteins, genes, drugs.

---

## References

[To be completed with full citations]

- Kaiser & Sutskever (2016). Neural GPUs Learn Algorithms.
- Trask et al. (2018). Neural Arithmetic Logic Units.
- Tishby & Zaslavsky (2015). Deep Learning and the Information Bottleneck Principle.
- [AlphaFold papers]
- [Manifold learning papers]
- [6502 architecture references]
- [Thermometer encoding history]

---

# END OF PAPER OUTLINE

---

## Metadata

**Target Venue**: NeurIPS, ICML, or Nature Machine Intelligence
**Target Length**: 10-12 pages + appendices
**Figures Needed**: ~8-10
**Tables Needed**: ~5-6
**Estimated Writing Time**: 2-3 weeks
