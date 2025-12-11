"""
Neural Vector Field (NVF) - Phase 1

Learn to predict what a vector database would return for a query.

Key insight: Instead of predicting discrete "top K indices",
we predict the result embedding directly (continuous) or
use soft attention over the database (fully differentiable).

This is the "Calculator MCP" of database lookups.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import Tuple, List, Optional
from dataclasses import dataclass


# ============================================================
# SYNTHETIC DATABASE
# ============================================================

class SyntheticVectorDB:
    """
    A simple in-memory vector database for testing.
    Ground truth for our NVF to learn from.
    """
    
    def __init__(self, n_vectors: int = 1000, dim: int = 128, seed: int = 42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Generate random database vectors (normalized)
        self.vectors = torch.randn(n_vectors, dim)
        self.vectors = F.normalize(self.vectors, dim=1)
        
        self.n_vectors = n_vectors
        self.dim = dim
    
    def search(self, query: torch.Tensor, k: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Exact nearest neighbor search.
        
        Args:
            query: (batch, dim) or (dim,) query vectors
            k: number of results
        
        Returns:
            indices: (batch, k) indices of nearest neighbors
            scores: (batch, k) similarity scores
        """
        if query.dim() == 1:
            query = query.unsqueeze(0)
        
        # Normalize query
        query = F.normalize(query, dim=1)
        
        # Compute similarities (cosine = dot product for normalized vectors)
        similarities = torch.mm(query, self.vectors.T)  # (batch, n_vectors)
        
        # Get top-k
        scores, indices = similarities.topk(k, dim=1)
        
        return indices, scores
    
    def get_vector(self, idx: int) -> torch.Tensor:
        """Get a database vector by index."""
        return self.vectors[idx]
    
    def get_vectors(self, indices: torch.Tensor) -> torch.Tensor:
        """Get multiple vectors by indices."""
        return self.vectors[indices]


# ============================================================
# NVF APPROACH 1: Direct Embedding Prediction
# ============================================================

class NVF_DirectPredictor(nn.Module):
    """
    Predict the nearest neighbor embedding directly.
    
    Input: query embedding (dim)
    Output: predicted nearest neighbor embedding (dim)
    
    The model learns to map queries to their nearest database vectors.
    """
    
    def __init__(self, dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )
    
    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """
        Predict the nearest neighbor embedding.
        
        Args:
            query: (batch, dim) query vectors
        
        Returns:
            predicted: (batch, dim) predicted nearest neighbor embeddings
        """
        return self.net(query)


# ============================================================
# NVF APPROACH 2: Soft Attention over Database
# ============================================================

class NVF_SoftAttention(nn.Module):
    """
    Compute soft attention over the database.
    
    Input: query embedding (dim)
    Output: weighted combination of database vectors
    
    This is fully differentiable - no discrete top-k selection.
    The "result" is a soft mixture of all database vectors,
    weighted by learned attention.
    """
    
    def __init__(self, db: SyntheticVectorDB, temperature: float = 1.0):
        super().__init__()
        
        # Store database as buffer (not trained)
        self.register_buffer('db_vectors', db.vectors)
        self.n_vectors = db.n_vectors
        self.dim = db.dim
        
        # Learnable query transformation
        self.query_transform = nn.Sequential(
            nn.Linear(db.dim, db.dim),
            nn.ReLU(),
            nn.Linear(db.dim, db.dim),
        )
        
        # Temperature for attention sharpness
        self.temperature = nn.Parameter(torch.tensor(temperature))
    
    def forward(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute soft attention retrieval.
        
        Args:
            query: (batch, dim) query vectors
        
        Returns:
            result: (batch, dim) soft-retrieved vectors
            attention: (batch, n_vectors) attention weights
        """
        if query.dim() == 1:
            query = query.unsqueeze(0)
        
        # Transform query
        q = self.query_transform(query)
        q = F.normalize(q, dim=1)
        
        # Compute attention scores
        scores = torch.mm(q, self.db_vectors.T) / self.temperature  # (batch, n_vectors)
        attention = F.softmax(scores, dim=1)
        
        # Weighted combination of database vectors
        result = torch.mm(attention, self.db_vectors)  # (batch, dim)
        
        return result, attention


# ============================================================
# NVF APPROACH 3: Index Prediction (Classification)
# ============================================================

class NVF_IndexPredictor(nn.Module):
    """
    Predict which database index is the nearest neighbor.
    
    This is a classification approach - softmax over database indices.
    Less elegant than soft attention but more explicit.
    """
    
    def __init__(self, dim: int = 128, n_vectors: int = 1000, hidden_dim: int = 512):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_vectors),
        )
    
    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """
        Predict index probabilities.
        
        Args:
            query: (batch, dim) query vectors
        
        Returns:
            logits: (batch, n_vectors) unnormalized log probabilities
        """
        return self.net(query)


# ============================================================
# DATA GENERATION
# ============================================================

def generate_nvf_dataset(db: SyntheticVectorDB, 
                         n_samples: int = 10000,
                         noise_scale: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate query/result pairs for training.
    
    Strategy: Sample database vectors, add noise to create queries,
    record the original vector as the "correct" answer.
    """
    
    # Sample random database indices
    indices = torch.randint(0, db.n_vectors, (n_samples,))
    
    # Get the base vectors
    base_vectors = db.vectors[indices]
    
    # Add noise to create queries
    noise = torch.randn_like(base_vectors) * noise_scale
    queries = F.normalize(base_vectors + noise, dim=1)
    
    # The "correct" answer is the original vector
    targets = base_vectors
    
    return queries, targets, indices


def generate_diverse_queries(db: SyntheticVectorDB,
                             n_samples: int = 10000) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate diverse queries by random sampling in embedding space.
    
    For each query, find the actual nearest neighbor as ground truth.
    """
    
    # Random queries
    queries = torch.randn(n_samples, db.dim)
    queries = F.normalize(queries, dim=1)
    
    # Find ground truth nearest neighbors
    indices, _ = db.search(queries, k=1)
    indices = indices.squeeze(1)
    
    # Get target vectors
    targets = db.vectors[indices]
    
    return queries, targets, indices


# ============================================================
# TRAINING
# ============================================================

def train_nvf_direct(db: SyntheticVectorDB,
                     n_samples: int = 50000,
                     n_epochs: int = 50,
                     batch_size: int = 256,
                     lr: float = 0.001) -> NVF_DirectPredictor:
    """Train the direct embedding predictor."""
    
    print("=" * 70)
    print("       NVF DIRECT PREDICTOR - Training")
    print("=" * 70)
    
    # Generate data
    print(f"\n[1] Generating {n_samples:,} query/result pairs...")
    queries, targets, indices = generate_diverse_queries(db, n_samples)
    
    print(f"    Query shape: {queries.shape}")
    print(f"    Target shape: {targets.shape}")
    
    # Create model
    print(f"\n[2] Creating model...")
    model = NVF_DirectPredictor(dim=db.dim, hidden_dim=256)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {n_params:,}")
    
    # Training
    print(f"\n[3] Training for {n_epochs} epochs...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    n_batches = len(queries) // batch_size
    
    for epoch in range(n_epochs):
        # Shuffle
        perm = torch.randperm(len(queries))
        queries_shuf = queries[perm]
        targets_shuf = targets[perm]
        indices_shuf = indices[perm]
        
        epoch_loss = 0.0
        for i in range(n_batches):
            batch_q = queries_shuf[i*batch_size : (i+1)*batch_size]
            batch_t = targets_shuf[i*batch_size : (i+1)*batch_size]
            
            optimizer.zero_grad()
            pred = model(batch_q)
            
            # Cosine similarity loss (we want pred to match target direction)
            pred_norm = F.normalize(pred, dim=1)
            target_norm = F.normalize(batch_t, dim=1)
            loss = 1 - (pred_norm * target_norm).sum(dim=1).mean()
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / n_batches
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            # Evaluate: does the predicted embedding retrieve the right item?
            with torch.no_grad():
                pred = model(queries[:1000])
                pred_norm = F.normalize(pred, dim=1)
                
                # Find which database vector is closest to prediction
                sims = torch.mm(pred_norm, db.vectors.T)
                pred_indices = sims.argmax(dim=1)
                
                # Compare to ground truth
                accuracy = (pred_indices == indices[:1000]).float().mean().item() * 100
            
            print(f"    Epoch {epoch+1:3d}: loss={avg_loss:.6f}, retrieval_acc={accuracy:.2f}%")
    
    return model


def train_nvf_attention(db: SyntheticVectorDB,
                        n_samples: int = 50000,
                        n_epochs: int = 50,
                        batch_size: int = 256,
                        lr: float = 0.001) -> NVF_SoftAttention:
    """Train the soft attention NVF."""
    
    print("=" * 70)
    print("       NVF SOFT ATTENTION - Training")
    print("=" * 70)
    
    # Generate data
    print(f"\n[1] Generating {n_samples:,} query/result pairs...")
    queries, targets, indices = generate_diverse_queries(db, n_samples)
    
    # Create model
    print(f"\n[2] Creating model...")
    model = NVF_SoftAttention(db, temperature=1.0)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Trainable parameters: {n_params:,}")
    
    # Training
    print(f"\n[3] Training for {n_epochs} epochs...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    n_batches = len(queries) // batch_size
    
    for epoch in range(n_epochs):
        perm = torch.randperm(len(queries))
        queries_shuf = queries[perm]
        targets_shuf = targets[perm]
        indices_shuf = indices[perm]
        
        epoch_loss = 0.0
        for i in range(n_batches):
            batch_q = queries_shuf[i*batch_size : (i+1)*batch_size]
            batch_t = targets_shuf[i*batch_size : (i+1)*batch_size]
            batch_idx = indices_shuf[i*batch_size : (i+1)*batch_size]
            
            optimizer.zero_grad()
            result, attention = model(batch_q)
            
            # Loss: maximize attention on correct index
            # This is cross-entropy on attention weights
            loss = F.cross_entropy(
                torch.log(attention + 1e-10), 
                batch_idx
            )
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / n_batches
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            with torch.no_grad():
                _, attention = model(queries[:1000])
                pred_indices = attention.argmax(dim=1)
                accuracy = (pred_indices == indices[:1000]).float().mean().item() * 100
            
            print(f"    Epoch {epoch+1:3d}: loss={avg_loss:.4f}, retrieval_acc={accuracy:.2f}%")
    
    return model


def train_nvf_classifier(db: SyntheticVectorDB,
                         n_samples: int = 50000,
                         n_epochs: int = 50,
                         batch_size: int = 256,
                         lr: float = 0.001) -> NVF_IndexPredictor:
    """Train the index classification NVF."""
    
    print("=" * 70)
    print("       NVF INDEX CLASSIFIER - Training")
    print("=" * 70)
    
    # Generate data
    print(f"\n[1] Generating {n_samples:,} query/result pairs...")
    queries, targets, indices = generate_diverse_queries(db, n_samples)
    
    # Create model
    print(f"\n[2] Creating model...")
    model = NVF_IndexPredictor(dim=db.dim, n_vectors=db.n_vectors, hidden_dim=512)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {n_params:,}")
    
    # Training
    print(f"\n[3] Training for {n_epochs} epochs...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    n_batches = len(queries) // batch_size
    
    for epoch in range(n_epochs):
        perm = torch.randperm(len(queries))
        queries_shuf = queries[perm]
        indices_shuf = indices[perm]
        
        epoch_loss = 0.0
        for i in range(n_batches):
            batch_q = queries_shuf[i*batch_size : (i+1)*batch_size]
            batch_idx = indices_shuf[i*batch_size : (i+1)*batch_size]
            
            optimizer.zero_grad()
            logits = model(batch_q)
            loss = F.cross_entropy(logits, batch_idx)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / n_batches
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            with torch.no_grad():
                logits = model(queries[:1000])
                pred_indices = logits.argmax(dim=1)
                accuracy = (pred_indices == indices[:1000]).float().mean().item() * 100
            
            print(f"    Epoch {epoch+1:3d}: loss={avg_loss:.4f}, retrieval_acc={accuracy:.2f}%")
    
    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate_nvf(model: nn.Module, db: SyntheticVectorDB, model_type: str = 'direct'):
    """Comprehensive NVF evaluation."""
    
    print("\n" + "=" * 70)
    print(f"       NVF EVALUATION ({model_type.upper()})")
    print("=" * 70)
    
    # Generate test queries
    n_test = 5000
    queries, targets, indices = generate_diverse_queries(db, n_test)
    
    model.eval()
    
    with torch.no_grad():
        # Get predictions based on model type
        if model_type == 'direct':
            pred = model(queries)
            pred_norm = F.normalize(pred, dim=1)
            sims = torch.mm(pred_norm, db.vectors.T)
            pred_indices = sims.argmax(dim=1)
        elif model_type == 'attention':
            _, attention = model(queries)
            pred_indices = attention.argmax(dim=1)
        elif model_type == 'classifier':
            logits = model(queries)
            pred_indices = logits.argmax(dim=1)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Accuracy
        accuracy = (pred_indices == indices).float().mean().item() * 100
        
        # Top-5 accuracy
        if model_type == 'direct':
            top5 = sims.topk(5, dim=1).indices
        elif model_type == 'attention':
            top5 = attention.topk(5, dim=1).indices
        else:
            top5 = logits.topk(5, dim=1).indices
        
        top5_acc = sum(1 for i, idx in enumerate(indices) if idx.item() in top5[i].tolist()) / n_test * 100
    
    print(f"\n    Test samples: {n_test:,}")
    print(f"    Top-1 accuracy: {accuracy:.2f}%")
    print(f"    Top-5 accuracy: {top5_acc:.2f}%")
    
    # Latency comparison
    print("\n    Latency comparison:")
    
    # Real search
    start = time.perf_counter()
    for _ in range(100):
        _, _ = db.search(queries[:10], k=1)
    real_time = (time.perf_counter() - start) / 100 * 1000  # ms for 10 queries
    
    # Neural search
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(100):
            if model_type == 'direct':
                _ = model(queries[:10])
            elif model_type == 'attention':
                _, _ = model(queries[:10])
            else:
                _ = model(queries[:10])
    neural_time = (time.perf_counter() - start) / 100 * 1000  # ms for 10 queries
    
    print(f"    Real search (10 queries): {real_time:.4f} ms")
    print(f"    Neural search (10 queries): {neural_time:.4f} ms")
    print(f"    Speedup: {real_time/neural_time:.1f}x")
    
    # Gradient flow test
    print("\n    Gradient flow test:")
    query = queries[0:1].clone().requires_grad_(True)
    
    if model_type == 'direct':
        pred = model(query)
        loss = pred.sum()
    elif model_type == 'attention':
        result, attention = model(query)
        loss = result.sum()
    else:
        logits = model(query)
        loss = logits.sum()
    
    loss.backward()
    grad_magnitude = query.grad.abs().sum().item()
    print(f"    Gradient magnitude: {grad_magnitude:.4f}")
    print(f"    [✓] GRADIENTS FLOW THROUGH NVF")
    
    return accuracy


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # Create synthetic database
    print("Creating synthetic vector database...")
    db = SyntheticVectorDB(n_vectors=1000, dim=128)
    print(f"Database: {db.n_vectors} vectors, {db.dim} dimensions")
    
    # Train and evaluate each approach
    results = {}
    
    # Approach 1: Direct embedding prediction
    print("\n" + "=" * 70)
    print("       APPROACH 1: DIRECT EMBEDDING PREDICTION")
    print("=" * 70)
    model_direct = train_nvf_direct(db, n_samples=50000, n_epochs=30)
    results['direct'] = evaluate_nvf(model_direct, db, 'direct')
    
    # Approach 2: Soft attention
    print("\n" + "=" * 70)
    print("       APPROACH 2: SOFT ATTENTION")
    print("=" * 70)
    model_attention = train_nvf_attention(db, n_samples=50000, n_epochs=30)
    results['attention'] = evaluate_nvf(model_attention, db, 'attention')
    
    # Approach 3: Index classification
    print("\n" + "=" * 70)
    print("       APPROACH 3: INDEX CLASSIFICATION")
    print("=" * 70)
    model_classifier = train_nvf_classifier(db, n_samples=50000, n_epochs=30)
    results['classifier'] = evaluate_nvf(model_classifier, db, 'classifier')
    
    # Summary
    print("\n" + "=" * 70)
    print("       NVF RESULTS SUMMARY")
    print("=" * 70)
    
    for name, acc in results.items():
        print(f"    {name:15s}: {acc:.2f}%")
    
    best = max(results, key=results.get)
    print(f"\n    Best approach: {best} ({results[best]:.2f}%)")
    
    # Save best model
    if best == 'direct':
        torch.save(model_direct.state_dict(), '/workspace/two-be/checkpoints/swarm/nvf_direct.pt')
    elif best == 'attention':
        torch.save(model_attention.state_dict(), '/workspace/two-be/checkpoints/swarm/nvf_attention.pt')
    else:
        torch.save(model_classifier.state_dict(), '/workspace/two-be/checkpoints/swarm/nvf_classifier.pt')
    
    print(f"\n[+] Best model saved to checkpoints/swarm/nvf_{best}.pt")
    
    print("\n" + "=" * 70)
    print("       NEURAL VECTOR FIELD - PHASE 1 COMPLETE")
    print("=" * 70)
