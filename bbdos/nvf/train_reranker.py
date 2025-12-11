"""
Train the Neural Reranker

Takes the trained Soft Attention model and builds a reranker on top.
Goal: 68% Top-1 → 90%+ Top-1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Tuple

from .reranker import Reranker, NVFPipeline


# ============================================================
# DATA GENERATION
# ============================================================

def generate_reranker_training_data(
    soft_attention: nn.Module,
    memory_vectors: torch.Tensor,
    n_samples: int,
    k: int = 5,
    noise_scale: float = 0.3
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate training data for the reranker.
    
    Returns:
        queries: (n_valid, dim)
        candidates: (n_valid, k, dim)
        labels: (n_valid,) - index into candidates of correct answer
        coverage: float - what fraction had correct in top-k
    """
    n_db, dim = memory_vectors.shape
    
    # Generate queries (noisy versions of memory vectors)
    true_indices = torch.randint(0, n_db, (n_samples,))
    base = memory_vectors[true_indices]
    queries = F.normalize(base + torch.randn_like(base) * noise_scale, dim=1)
    
    # Get Top-K from soft attention
    soft_attention.eval()
    with torch.no_grad():
        _, attention = soft_attention(queries)
        topk_indices = attention.topk(k, dim=1).indices  # (n_samples, k)
    
    # Get candidate vectors
    candidates = memory_vectors[topk_indices]  # (n_samples, k, dim)
    
    # Find where true index appears in top-k
    labels = []
    valid_mask = []
    
    for i in range(n_samples):
        true_idx = true_indices[i].item()
        topk = topk_indices[i].tolist()
        
        if true_idx in topk:
            labels.append(topk.index(true_idx))
            valid_mask.append(True)
        else:
            labels.append(0)  # Placeholder
            valid_mask.append(False)
    
    labels = torch.tensor(labels)
    valid_mask = torch.tensor(valid_mask)
    
    # Filter to valid samples
    valid_queries = queries[valid_mask]
    valid_candidates = candidates[valid_mask]
    valid_labels = labels[valid_mask]
    
    coverage = valid_mask.float().mean().item()
    
    return valid_queries, valid_candidates, valid_labels, coverage


# ============================================================
# TRAINING
# ============================================================

def train_reranker(
    reranker: Reranker,
    queries: torch.Tensor,
    candidates: torch.Tensor,
    labels: torch.Tensor,
    n_epochs: int = 50,
    batch_size: int = 256,
    lr: float = 0.001,
    val_split: float = 0.1
) -> Reranker:
    """Train the reranker with cross-entropy loss."""
    
    print("=" * 70)
    print("       RERANKER TRAINING")
    print("=" * 70)
    
    n_samples = len(queries)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val
    
    # Split
    perm = torch.randperm(n_samples)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    
    train_q, train_c, train_l = queries[train_idx], candidates[train_idx], labels[train_idx]
    val_q, val_c, val_l = queries[val_idx], candidates[val_idx], labels[val_idx]
    
    print(f"\n  Training samples: {n_train:,}")
    print(f"  Validation samples: {n_val:,}")
    print(f"  Candidate set size: {candidates.shape[1]}")
    
    optimizer = torch.optim.Adam(reranker.parameters(), lr=lr)
    n_batches = n_train // batch_size
    
    best_val_acc = 0
    best_state = None
    
    print(f"\n  Training for {n_epochs} epochs...")
    
    for epoch in range(n_epochs):
        reranker.train()
        
        # Shuffle training data
        perm = torch.randperm(n_train)
        train_q = train_q[perm]
        train_c = train_c[perm]
        train_l = train_l[perm]
        
        epoch_loss = 0.0
        
        for i in range(n_batches):
            batch_q = train_q[i*batch_size : (i+1)*batch_size]
            batch_c = train_c[i*batch_size : (i+1)*batch_size]
            batch_l = train_l[i*batch_size : (i+1)*batch_size]
            
            optimizer.zero_grad()
            
            scores = reranker(batch_q, batch_c)
            loss = F.cross_entropy(scores, batch_l)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Validation
        reranker.eval()
        with torch.no_grad():
            val_scores = reranker(val_q, val_c)
            val_pred = val_scores.argmax(dim=1)
            val_acc = (val_pred == val_l).float().mean().item() * 100
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in reranker.state_dict().items()}
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d}: loss={epoch_loss/n_batches:.4f}, val_acc={val_acc:.1f}%")
    
    # Restore best state
    if best_state is not None:
        reranker.load_state_dict(best_state)
    
    print(f"\n  Best validation accuracy: {best_val_acc:.1f}%")
    
    return reranker


# ============================================================
# EVALUATION
# ============================================================

def evaluate_pipeline(
    soft_attention: nn.Module,
    reranker: Reranker,
    memory_vectors: torch.Tensor,
    n_test: int = 5000,
    k: int = 5,
    noise_scale: float = 0.3
) -> dict:
    """Evaluate the full NVF pipeline."""
    
    print("\n" + "=" * 70)
    print("       PIPELINE EVALUATION")
    print("=" * 70)
    
    n_db, dim = memory_vectors.shape
    
    # Generate test queries
    true_indices = torch.randint(0, n_db, (n_test,))
    base = memory_vectors[true_indices]
    queries = F.normalize(base + torch.randn_like(base) * noise_scale, dim=1)
    
    soft_attention.eval()
    reranker.eval()
    
    with torch.no_grad():
        # Stage 1: Soft Attention
        _, attention = soft_attention(queries)
        
        # Soft attention Top-1
        soft_top1_pred = attention.argmax(dim=1)
        soft_top1_acc = (soft_top1_pred == true_indices).float().mean().item() * 100
        
        # Soft attention Top-K
        topk_indices = attention.topk(k, dim=1).indices
        soft_topk_acc = sum(
            1 for i in range(n_test) 
            if true_indices[i].item() in topk_indices[i].tolist()
        ) / n_test * 100
        
        # Get candidates
        candidates = memory_vectors[topk_indices]
        
        # Stage 2: Reranker
        rerank_scores = reranker(queries, candidates)
        rerank_pred_in_topk = rerank_scores.argmax(dim=1)
        
        # Map back to database indices
        final_pred = topk_indices.gather(1, rerank_pred_in_topk.unsqueeze(1)).squeeze(1)
        
        # Pipeline accuracy
        pipeline_acc = (final_pred == true_indices).float().mean().item() * 100
    
    # Count how many the reranker got right (of those where correct was in top-k)
    in_topk_mask = torch.tensor([
        true_indices[i].item() in topk_indices[i].tolist() 
        for i in range(n_test)
    ])
    
    if in_topk_mask.sum() > 0:
        reranker_acc_given_topk = (
            final_pred[in_topk_mask] == true_indices[in_topk_mask]
        ).float().mean().item() * 100
    else:
        reranker_acc_given_topk = 0.0
    
    results = {
        'soft_top1': soft_top1_acc,
        'soft_topk': soft_topk_acc,
        'pipeline_top1': pipeline_acc,
        'reranker_given_topk': reranker_acc_given_topk,
        'k': k,
    }
    
    print(f"\n  Test samples: {n_test:,}")
    print(f"\n  Stage 1 (Soft Attention):")
    print(f"    Top-1: {soft_top1_acc:.2f}%")
    print(f"    Top-{k}: {soft_topk_acc:.2f}%")
    print(f"\n  Stage 2 (Reranker):")
    print(f"    Accuracy given Top-{k}: {reranker_acc_given_topk:.2f}%")
    print(f"\n  Full Pipeline:")
    print(f"    Top-1: {pipeline_acc:.2f}%")
    print(f"\n  Improvement: {soft_top1_acc:.1f}% → {pipeline_acc:.1f}% (+{pipeline_acc - soft_top1_acc:.1f}%)")
    
    return results


def test_gradient_flow(
    soft_attention: nn.Module,
    reranker: Reranker,
    memory_vectors: torch.Tensor
):
    """Verify gradients flow through the reranker."""
    
    print("\n  Gradient Flow Test:")
    
    # Create a query with gradients
    query = torch.randn(1, memory_vectors.shape[1], requires_grad=True)
    
    # Stage 1 (no gradients through frozen soft attention)
    with torch.no_grad():
        _, attention = soft_attention(query)
        topk_indices = attention.topk(5, dim=1).indices
        candidates = memory_vectors[topk_indices]
    
    # Stage 2 (gradients through reranker)
    # But candidates came from frozen stage, so we need to detach properly
    candidates_with_grad = candidates.clone().requires_grad_(True)
    
    scores = reranker(query, candidates_with_grad)
    loss = scores.sum()
    loss.backward()
    
    query_grad = query.grad.abs().sum().item() if query.grad is not None else 0
    cand_grad = candidates_with_grad.grad.abs().sum().item() if candidates_with_grad.grad is not None else 0
    
    print(f"    Query gradient magnitude: {query_grad:.4f}")
    print(f"    Candidate gradient magnitude: {cand_grad:.4f}")
    
    if query_grad > 0 or cand_grad > 0:
        print(f"    [✓] GRADIENTS FLOW THROUGH RERANKER")
    else:
        print(f"    [!] No gradients detected")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/workspace/two-be')
    
    from bbdos.nvf.neural_vector_field import NVF_SoftAttention, SyntheticVectorDB
    
    print("=" * 70)
    print("       NVF PHASE 2: RERANKER")
    print("       Target: 68% → 90%+")
    print("=" * 70)
    
    # Load database and soft attention
    print("\n[1] Loading components...")
    
    db = SyntheticVectorDB(n_vectors=1000, dim=128, seed=42)
    print(f"    Database: {db.n_vectors} vectors, {db.dim} dimensions")
    
    soft_attention = NVF_SoftAttention(db, temperature=1.0)
    soft_attention.load_state_dict(
        torch.load('/workspace/two-be/checkpoints/swarm/nvf_attention.pt', weights_only=True)
    )
    soft_attention.eval()
    print(f"    Soft Attention loaded (T={soft_attention.temperature.item():.4f})")
    
    # Generate reranker training data
    print("\n[2] Generating reranker training data...")
    queries, candidates, labels, coverage = generate_reranker_training_data(
        soft_attention, db.vectors, n_samples=100000, k=5
    )
    print(f"    Generated {len(queries):,} valid samples")
    print(f"    Coverage (correct in Top-5): {coverage*100:.1f}%")
    
    # Create and train reranker
    print("\n[3] Training reranker...")
    reranker = Reranker(dim=db.dim, hidden=128, dropout=0.1)
    n_params = sum(p.numel() for p in reranker.parameters())
    print(f"    Reranker parameters: {n_params:,}")
    
    reranker = train_reranker(
        reranker, queries, candidates, labels,
        n_epochs=50, batch_size=256, lr=0.001
    )
    
    # Evaluate full pipeline
    results = evaluate_pipeline(
        soft_attention, reranker, db.vectors,
        n_test=5000, k=5
    )
    
    # Test gradient flow
    test_gradient_flow(soft_attention, reranker, db.vectors)
    
    # Save reranker
    torch.save(reranker.state_dict(), '/workspace/two-be/checkpoints/swarm/nvf_reranker.pt')
    print(f"\n[+] Reranker saved to checkpoints/swarm/nvf_reranker.pt")
    
    # Summary
    print("\n" + "=" * 70)
    print("       NVF PHASE 2 COMPLETE")
    print("=" * 70)
    print(f"\n  Before (Soft Attention only): {results['soft_top1']:.1f}%")
    print(f"  After (Full Pipeline):        {results['pipeline_top1']:.1f}%")
    print(f"  Target:                       90.0%")
    
    if results['pipeline_top1'] >= 90:
        print(f"\n  [✓] TARGET ACHIEVED")
    else:
        print(f"\n  [→] Gap to close: {90 - results['pipeline_top1']:.1f}%")
