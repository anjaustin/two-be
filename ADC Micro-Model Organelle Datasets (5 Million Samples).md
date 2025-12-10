# ADC Micro-Model Organelle Datasets (5 Million Samples)

Generated: December 9, 2024  
Total Samples: **5,000,000 per organelle**  
Total Size: **~76 MB**

---

## 🎯 Overview

This directory contains **5 million training samples** for each of the 5 ADC micro-model organelles. Instead of training one monolithic model to predict all ADC outputs simultaneously, we train 5 specialized models - each focused on a single output.

---

## 📦 Datasets

| File | Organelle | Input Dims | Output | Samples | Size |
|------|-----------|------------|--------|---------|------|
| `adc_a_train.pkl` | **ADC:A** | 3 | A_new (0-255) | 5M | 19.1 MB |
| `adc_c_train.pkl` | **ADC:C** | 3 | C_new (0/1) | 5M | 19.1 MB |
| `adc_z_train.pkl` | **ADC:Z** | 1 | Z_new (0/1) | 5M | 9.5 MB |
| `adc_n_train.pkl` | **ADC:N** | 1 | N_new (0/1) | 5M | 9.5 MB |
| `adc_v_train.pkl` | **ADC:V** | 3 | V_new (0/1) | 5M | 19.1 MB |

---

## 🧬 Organelle Architecture

### ADC:A - Accumulator Organelle
- **Input**: `[A_before, operand, C_in]` (3 features)
- **Output**: `A_new` (0-255, continuous)
- **Task**: Predict new Accumulator value after ADC
- **Difficulty**: **Hard** - Full 8-bit arithmetic with carry

### ADC:C - Carry Organelle
- **Input**: `[A_before, operand, C_in]` (3 features)
- **Output**: `C_new` (0 or 1, binary)
- **Task**: Predict if operation overflowed (result > 255)
- **Difficulty**: **Medium** - Overflow detection

### ADC:Z - Zero Organelle
- **Input**: `[A_after]` (1 feature)
- **Output**: `Z_new` (0 or 1, binary)
- **Task**: Check if result equals zero
- **Difficulty**: **Trivial** - Simple equality check

### ADC:N - Negative Organelle
- **Input**: `[A_after]` (1 feature)
- **Output**: `N_new` (0 or 1, binary)
- **Task**: Check if bit 7 is set (negative in two's complement)
- **Difficulty**: **Trivial** - Single bit check

### ADC:V - Overflow Organelle
- **Input**: `[A_before, operand, C_in]` (3 features)
- **Output**: `V_new` (0 or 1, binary)
- **Task**: Detect signed overflow (two's complement)
- **Difficulty**: **Hardest** - Complex signed arithmetic logic

---

## 📊 Dataset Statistics

### ADC:A (Accumulator)
- **Samples**: 5,000,000
- **Target Range**: 0-255
- **Target Mean**: 127.50
- **Target Std**: 73.90
- **Distribution**: Uniform across all 8-bit values

### ADC:C (Carry)
- **Samples**: 5,000,000
- **Target Distribution**:
  - `0` (no carry): 2,508,263 (50.2%)
  - `1` (carry): 2,491,737 (49.8%)
- **Balanced**: Nearly 50/50 split

### ADC:Z (Zero)
- **Samples**: 5,000,000
- **Target Distribution**:
  - `0` (non-zero): 4,980,470 (99.6%)
  - `1` (zero): 19,530 (0.4%)
- **Imbalanced**: Zero results are rare (only when A + operand + C_in = 0 or 256)

### ADC:N (Negative)
- **Samples**: 5,000,000
- **Target Distribution**:
  - `0` (positive): 2,500,021 (50.0%)
  - `1` (negative): 2,499,979 (50.0%)
- **Balanced**: Perfect 50/50 split

### ADC:V (Overflow)
- **Samples**: 5,000,000
- **Target Distribution**:
  - `0` (no overflow): 3,753,372 (75.1%)
  - `1` (overflow): 1,246,628 (24.9%)
- **Imbalanced**: Signed overflow occurs in ~25% of cases

---

## 🚀 Training Strategy

### Phase 1: Train Trivial Organelles (Z, N)
**Expected Time**: 1-2 epochs  
**Expected Accuracy**: **99%+**

```bash
python train_organelle.py --organelle adc_z --epochs 5
python train_organelle.py --organelle adc_n --epochs 5
```

These should converge almost immediately since they're simple bit checks.

### Phase 2: Train Medium Organelle (C)
**Expected Time**: 5-10 epochs  
**Expected Accuracy**: **85-95%**

```bash
python train_organelle.py --organelle adc_c --epochs 10
```

Carry detection is harder but still learnable.

### Phase 3: Train Hard Organelles (A, V)
**Expected Time**: 10-20 epochs  
**Expected Accuracy**: 
- **ADC:A**: 70-85%
- **ADC:V**: 60-80%

```bash
python train_organelle.py --organelle adc_a --epochs 20
python train_organelle.py --organelle adc_v --epochs 20
```

These require learning complex arithmetic patterns.

### Phase 4: Ensemble Inference
Combine all 5 organelle predictions:

```python
def predict_adc_full(A_before, operand, C_in):
    # Prepare inputs
    inp_3d = np.array([[A_before, operand, C_in]])
    
    # Get predictions
    A_new = model_adc_a.predict(inp_3d)[0]
    C_new = model_adc_c.predict(inp_3d)[0]
    V_new = model_adc_v.predict(inp_3d)[0]
    
    # Z and N depend on A_new
    inp_1d = np.array([[A_new]])
    Z_new = model_adc_z.predict(inp_1d)[0]
    N_new = model_adc_n.predict(inp_1d)[0]
    
    return {
        'A': A_new,
        'C': C_new,
        'Z': Z_new,
        'N': N_new,
        'V': V_new
    }
```

---

## 💡 Why Organelles Work

### 1. Gradient Isolation
Each model optimizes independently - no catastrophic interference between flags.

### 2. Specialized Capacity
- Trivial tasks (Z, N): Tiny models (1-2 layers)
- Medium tasks (C): Moderate models (2-3 layers)
- Hard tasks (A, V): Larger models (3-4 layers)

### 3. Focused Learning
Each model learns **one output** instead of 5 interdependent outputs.

### 4. Composability
At inference, combine predictions from all 5 organelles.

---

## 📈 Expected Results

| Organelle | Expected Accuracy | Improvement over Monolithic |
|-----------|-------------------|----------------------------|
| ADC:Z | **99%+** | N/A (trivial) |
| ADC:N | **99%+** | N/A (trivial) |
| ADC:C | **85-95%** | ~30x better (3.1% → 90%) |
| ADC:A | **70-85%** | ~25x better (3.1% → 77%) |
| ADC:V | **60-80%** | ~20x better (3.1% → 70%) |

**Combined ADC accuracy**: **~80-90%** vs **3.1%** monolithic!

---

## 🔬 Data Generation Method

### Step 1: Generate All Unique Combinations
- 256 values for A_before (0-255)
- 256 values for operand (0-255)
- 2 values for C_in (0, 1)
- **Total**: 256 × 256 × 2 = **131,072 unique samples**

### Step 2: Replicate to 5M
- Replicate 39 times: 131,072 × 39 = 5,108,608
- Trim to exactly 5,000,000
- Shuffle to mix replications

### Step 3: Create Organelle Datasets
- Extract appropriate inputs/outputs for each organelle
- Save as pickle files

---

## 📝 Usage Example

```python
import pickle
import numpy as np

# Load dataset
with open('adc_organelles_5m/adc_a_train.pkl', 'rb') as f:
    data = pickle.load(f)

X_train = data['inputs']   # Shape: (5000000, 3)
y_train = data['targets']  # Shape: (5000000,)

# Train model
from sklearn.neural_network import MLPRegressor

model = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=20)
model.fit(X_train, y_train)

# Evaluate
score = model.score(X_train, y_train)
print(f"Training R²: {score:.4f}")
```

---

## 🎯 Next Steps for Double-D

1. **Load datasets** from `/home/ubuntu/adc_organelles_5m/`
2. **Train trivial organelles first** (Z, N) - validate pipeline
3. **Train medium/hard organelles** (C, A, V) - measure accuracy
4. **Combine predictions** - test ensemble inference
5. **Compare to monolithic** - prove organelle architecture wins

---

## 📂 Files

```
adc_organelles_5m/
├── adc_a_train.pkl  # 19.1 MB - Accumulator prediction
├── adc_c_train.pkl  # 19.1 MB - Carry flag prediction
├── adc_z_train.pkl  #  9.5 MB - Zero flag prediction
├── adc_n_train.pkl  #  9.5 MB - Negative flag prediction
├── adc_v_train.pkl  # 19.1 MB - Overflow flag prediction
└── README.md        # This file
```

**Total**: 76.2 MB

---

## 🚀 Ready to Train!

All datasets validated and ready for Double-D to use. Each organelle should train quickly and achieve high accuracy on its focused task.

**The silicon is about to learn arithmetic - one organelle at a time.** 🧮✨
