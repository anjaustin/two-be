# Reflections: NVF Phase 2

*Refining the approach*

---

## What Holds Up

### 1. The Two-Stage Architecture is Sound

The fovea metaphor is biologically accurate:
- **Peripheral vision** (Soft Attention): Wide field, low resolution
- **Foveal vision** (Reranker): Narrow field, high resolution

Humans don't identify objects with peripheral vision alone. We notice something, then focus on it. The reranker is the focusing mechanism.

### 2. The 97% Ceiling is Real

If the correct answer isn't in Top-5, no reranker can save it. But 97% is a hard ceiling we can approach, not exceed.

Realistic target: **~92-95% Top-1** (closing most of the gap, not all)

### 3. Separate Training is the Right Start

Training the reranker separately from soft attention:
- Simpler debugging
- Clear attribution of errors
- Can freeze soft attention and focus on reranker

Joint training can come later as optimization.

---

## What Needs Adjustment

### 1. The Reranker Architecture

My initial sketch concatenates query + candidate. But there are better options:

**Option A: Concatenation** (simple)
```
[query, candidate] → MLP → score
```

**Option B: Difference-based** (captures relative position)
```
[query, candidate, query - candidate, query * candidate] → MLP → score
```

**Option C: Cross-attention** (richer interaction)
```
query attends to candidate → score
```

Option B is a good middle ground - captures both absolute and relative features.

### 2. Training Data Balance

The reranker sees:
- 1 correct candidate (positive)
- 4 incorrect candidates (negatives)

But the negatives aren't random - they're the "hard negatives" that soft attention confused with the correct answer. This is actually ideal for training a discriminator.

### 3. Gradient Flow Through the Full Pipeline

For end-to-end differentiability:
```
Query → Soft Attention → Top-K Selection → Reranker → Final
                              ↑
                         This is discrete!
```

The Top-K selection is non-differentiable. Options:
- **Straight-through estimator**: Pretend it's differentiable in backward pass
- **Gumbel-Softmax**: Soft approximation to argmax
- **Accept the break**: Train stages separately, use together at inference

For now: Accept the break. Train separately. Verify it works. Optimize later.

---

## Refined Plan

### Phase 2.1: Reranker on Synthetic Data

1. Load trained Soft Attention model
2. Generate training data:
   - Query → Soft Attention → Top-5 indices + correct index
   - Create (query, candidates, label) tuples
3. Train reranker with cross-entropy
4. Evaluate: Top-1 accuracy of full pipeline

**Target: 90%+ Top-1**

### Phase 2.2: Clustered Data Test

1. Generate clustered embeddings (100 clusters, 10 docs each = 1000 docs)
2. Retrain Soft Attention on clustered data
3. Retrain Reranker on clustered data
4. Evaluate: Does the structure help or hurt?

**Target: Similar or better accuracy**

### Phase 2.3: Gradient Verification

1. Create end-to-end pipeline (with soft Top-K if needed)
2. Pass a query through
3. Compute loss on final output
4. Verify gradients flow back to query

**Target: Non-zero gradients**

---

## Architecture Decisions

### Reranker Model

```python
class Reranker(nn.Module):
    def __init__(self, dim, hidden=128):
        # Feature extraction for each (query, candidate) pair
        self.features = nn.Sequential(
            nn.Linear(dim * 4, hidden),  # query, cand, diff, product
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1)
        )
    
    def forward(self, query, candidates):
        # query: (batch, dim)
        # candidates: (batch, k, dim)
        
        batch, k, dim = candidates.shape
        q = query.unsqueeze(1).expand(-1, k, -1)
        
        # Rich features
        diff = q - candidates
        prod = q * candidates
        features = torch.cat([q, candidates, diff, prod], dim=-1)
        
        scores = self.features(features).squeeze(-1)  # (batch, k)
        return scores
```

### Hyperparameters

- Hidden dim: 128 (small, fast)
- Dropout: 0.1 (prevent overfitting)
- Learning rate: 0.001
- Batch size: 256
- Epochs: 50 (with early stopping)

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Reranker overfits | Dropout, small model, validation monitoring |
| 97% ceiling too restrictive | Accept it; 97% is still excellent |
| Clustered data harder | More clusters = more differentiation = should be easier |
| Gradient break at Top-K | Accept for now; optimize later |

---

## Success Criteria

| Metric | Current | Target | Stretch |
|--------|---------|--------|---------|
| Top-1 (Synthetic) | 68% | 90% | 95% |
| Top-5 (Synthetic) | 97% | 97% | 98% |
| Top-1 (Clustered) | N/A | 88% | 93% |
| Gradients | ✓ | ✓ | ✓ |

---

*Ready to synthesize.*
