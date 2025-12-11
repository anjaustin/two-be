# Raw Thoughts: NVF Phase 2

*Closing the gap and connecting to reality*

---

## The Current State

NVF Soft Attention achieves:
- **68% Top-1** - finds the exact match 2/3 of the time
- **97% Top-5** - correct answer almost always in top 5

The gap: The model knows the neighborhood but can't pinpoint the exact target.

---

## Task 1: Close the Gap (68% → 90%+)

### The Problem

97% Top-5 means the answer is *there*. We just can't pick it out from the crowd.

This is a discrimination problem, not a retrieval problem.

### Gem's Suggestion: Neural Reranker (The Fovea)

The idea:
- Global search finds the neighborhood (Soft Attention) 
- Local discriminator picks the winner (Reranker)

Architecture:
```
Query → Soft Attention → Top-5 Candidates → Reranker → Final Answer
```

The reranker sees:
- The query
- The 5 candidate vectors
- Must pick which one is the true match

This is a 5-way classification on a much smaller problem.

### Why This Should Work

1. **Reduced search space**: Instead of 1000 options, only 5
2. **Richer features**: Can compare query to each candidate directly
3. **Easier gradients**: Sharp decision on small set

### Implementation Thoughts

```python
class NeuralReranker(nn.Module):
    def __init__(self, dim):
        self.compare = nn.Sequential(
            nn.Linear(dim * 2, 128),  # Query + Candidate concatenated
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Score for this candidate
        )
    
    def forward(self, query, candidates):
        # query: (batch, dim)
        # candidates: (batch, k, dim)
        
        batch, k, dim = candidates.shape
        
        # Expand query to match candidates
        query_exp = query.unsqueeze(1).expand(-1, k, -1)  # (batch, k, dim)
        
        # Concatenate and score each pair
        pairs = torch.cat([query_exp, candidates], dim=-1)  # (batch, k, dim*2)
        scores = self.compare(pairs).squeeze(-1)  # (batch, k)
        
        return scores  # Softmax externally for flexibility
```

### Training Strategy

1. Use the trained Soft Attention model to generate Top-5 candidates
2. Create training data: (query, top5_candidates, correct_index)
3. Train reranker with cross-entropy on the 5-way classification
4. At inference: Soft Attention → Top-5 → Reranker → Final

### Potential Issues

- If the correct answer isn't in Top-5 (3% of cases), reranker can't help
- Need to make sure reranker generalizes, not just memorizes

---

## Task 2: Connect to Real Qdrant

### The Problem

Our synthetic database is random vectors. Real data has:
- Semantic structure (similar documents cluster)
- Non-uniform distribution (some regions dense, some sparse)
- Real embeddings from actual content

### What We Need

1. **Qdrant instance** - either local or cloud
2. **Real embeddings** - from some corpus
3. **Query logs** - to train NVF on real patterns

### Implementation Options

**Option A: Local Qdrant with sample data**
- Spin up Qdrant locally
- Index a small corpus (e.g., Wikipedia paragraphs)
- Generate embeddings with sentence-transformers
- Log query/result pairs
- Train NVF on logged data

**Option B: Simulated Qdrant with realistic data**
- Generate embeddings that mimic real structure
- Cluster-based generation (documents group by topic)
- Test if NVF generalizes to structured data

I'll start with Option B to test the principle, then move to Option A for validation.

### Simulated Realistic Data

```python
def generate_realistic_corpus(n_docs, n_clusters, dim):
    """
    Generate embeddings that mimic real document structure.
    Documents cluster by topic.
    """
    docs_per_cluster = n_docs // n_clusters
    
    embeddings = []
    cluster_labels = []
    
    for c in range(n_clusters):
        # Cluster centroid
        centroid = F.normalize(torch.randn(dim), dim=0)
        
        # Documents within cluster (tight distribution)
        for _ in range(docs_per_cluster):
            noise = torch.randn(dim) * 0.2  # Small variance within cluster
            doc = F.normalize(centroid + noise, dim=0)
            embeddings.append(doc)
            cluster_labels.append(c)
    
    return torch.stack(embeddings), torch.tensor(cluster_labels)
```

This creates a more realistic test: Can NVF learn to navigate clustered data?

---

## The Combined Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                         NVF PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Query ─────────────────────────────────────────────────────►  │
│            │                                                    │
│            ▼                                                    │
│   ┌─────────────────┐                                          │
│   │  Soft Attention │  (68% Top-1, 97% Top-5)                  │
│   │    (Global)     │                                          │
│   └────────┬────────┘                                          │
│            │                                                    │
│            ▼                                                    │
│   ┌─────────────────┐                                          │
│   │    Top-5        │                                          │
│   │  Candidates     │                                          │
│   └────────┬────────┘                                          │
│            │                                                    │
│            ▼                                                    │
│   ┌─────────────────┐                                          │
│   │    Reranker     │  (Target: 90%+ Top-1)                    │
│   │    (Local)      │                                          │
│   └────────┬────────┘                                          │
│            │                                                    │
│            ▼                                                    │
│       Final Answer                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Success Metrics

### Task 1 (Reranker)
- **Target**: 90%+ Top-1 accuracy (up from 68%)
- **Constraint**: Must still have gradients flowing

### Task 2 (Realistic Data)
- **Target**: Similar accuracy on clustered data
- **Bonus**: Show that NVF learns cluster structure

---

## Order of Operations

1. **Build Reranker** on synthetic data first
2. **Verify 90%+** on the clean case
3. **Generate clustered data** to simulate real corpus
4. **Test pipeline** on clustered data
5. **If time**: Connect to actual Qdrant

---

## Questions

- Should the reranker be trained jointly with soft attention, or separately?
  - Start separate (easier), then try joint if needed

- What if reranker overfits?
  - Use dropout, smaller model, more data

- How to handle the 3% where correct isn't in Top-5?
  - Accept it for now. 97% ceiling is still very good.

---

*Ready to reflect.*
