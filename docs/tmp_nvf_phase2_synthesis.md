# Synthesis: NVF Phase 2

*The execution plan*

---

## The Goal

Close the gap from **68% → 90%+ Top-1** accuracy while maintaining differentiability.

---

## The Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    NVF TWO-STAGE PIPELINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Query ──────────────────────────────────────────────────────► │
│            │                                                    │
│            ▼                                                    │
│   ┌─────────────────────────────────────────────┐              │
│   │         STAGE 1: SOFT ATTENTION             │              │
│   │         (Peripheral Vision)                  │              │
│   │                                              │              │
│   │   Query → Transform → Attend → Top-5        │              │
│   │   Current: 68% Top-1, 97% Top-5             │              │
│   │   Role: Find the neighborhood               │              │
│   └─────────────────┬───────────────────────────┘              │
│                     │                                           │
│                     ▼                                           │
│   ┌─────────────────────────────────────────────┐              │
│   │         STAGE 2: RERANKER                   │              │
│   │         (Foveal Vision)                      │              │
│   │                                              │              │
│   │   (Query, Top-5) → Compare → Best Match     │              │
│   │   Target: 90%+ Top-1                        │              │
│   │   Role: Pick the exact answer               │              │
│   └─────────────────┬───────────────────────────┘              │
│                     │                                           │
│                     ▼                                           │
│               Final Answer                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Order

### Step 1: Build Reranker Module (~30 min)

```python
class Reranker(nn.Module):
    """Pick the best from Top-K candidates."""
    
    def __init__(self, dim=128, hidden=128):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(dim * 4, hidden),
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
        
        diff = q - candidates
        prod = q * candidates
        features = torch.cat([q, candidates, diff, prod], dim=-1)
        
        return self.scorer(features).squeeze(-1)
```

### Step 2: Generate Reranker Training Data (~10 min)

```python
def generate_reranker_data(soft_attention, db, n_samples):
    """Generate (query, top5_candidates, correct_idx) tuples."""
    
    queries, true_indices = generate_queries(db, n_samples)
    
    with torch.no_grad():
        _, attention = soft_attention(queries)
        top5_indices = attention.topk(5, dim=1).indices
    
    # Get candidate vectors
    candidates = db.vectors[top5_indices]  # (n_samples, 5, dim)
    
    # Find where true index appears in top5 (if at all)
    labels = []
    valid_mask = []
    for i in range(n_samples):
        true_idx = true_indices[i].item()
        top5 = top5_indices[i].tolist()
        if true_idx in top5:
            labels.append(top5.index(true_idx))
            valid_mask.append(True)
        else:
            labels.append(0)  # Placeholder
            valid_mask.append(False)
    
    return queries, candidates, torch.tensor(labels), torch.tensor(valid_mask)
```

### Step 3: Train Reranker (~20 min)

```python
def train_reranker(reranker, queries, candidates, labels, valid_mask):
    """Train on valid samples only (where true is in Top-5)."""
    
    optimizer = torch.optim.Adam(reranker.parameters(), lr=0.001)
    
    # Filter to valid samples
    valid_q = queries[valid_mask]
    valid_c = candidates[valid_mask]
    valid_l = labels[valid_mask]
    
    # Training loop...
```

### Step 4: Evaluate Full Pipeline (~10 min)

```python
def evaluate_pipeline(soft_attention, reranker, db, n_test):
    """End-to-end evaluation."""
    
    queries, true_indices = generate_queries(db, n_test)
    
    # Stage 1: Soft Attention
    _, attention = soft_attention(queries)
    top5_indices = attention.topk(5, dim=1).indices
    candidates = db.vectors[top5_indices]
    
    # Stage 2: Reranker
    scores = reranker(queries, candidates)
    reranked = scores.argmax(dim=1)
    
    # Map back to original indices
    final_indices = top5_indices.gather(1, reranked.unsqueeze(1)).squeeze(1)
    
    # Accuracy
    correct = (final_indices == true_indices).float().mean().item()
    return correct * 100
```

### Step 5: Test on Clustered Data (~30 min)

Generate clustered data, retrain both stages, evaluate.

---

## File Structure

```
bbdos/nvf/
├── __init__.py
├── neural_vector_field.py      # Existing soft attention
├── reranker.py                 # NEW: Reranker module
└── pipeline.py                 # NEW: Combined pipeline
```

---

## Timeline

| Step | Time | Output |
|------|------|--------|
| Build Reranker | 30 min | `reranker.py` |
| Generate Data | 10 min | Training tuples |
| Train Reranker | 20 min | `reranker.pt` |
| Evaluate Pipeline | 10 min | Accuracy metrics |
| Clustered Data Test | 30 min | Generalization metrics |
| **Total** | **~2 hours** | **90%+ Top-1** |

---

## Success Criteria

| Metric | Target |
|--------|--------|
| Pipeline Top-1 (Synthetic) | 90%+ |
| Pipeline Top-1 (Clustered) | 88%+ |
| Gradients through Reranker | ✓ |

---

*Execute.*
