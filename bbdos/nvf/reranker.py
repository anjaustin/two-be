"""
Neural Reranker - The Fovea

Stage 2 of the NVF pipeline.
Takes Top-K candidates from Soft Attention and picks the best one.

Global Search (Attention) + Local Verification (Discrimination)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Reranker(nn.Module):
    """
    Pick the best match from Top-K candidates.
    
    Uses rich features: query, candidate, difference, element-wise product
    """
    
    def __init__(self, dim: int = 128, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        
        self.dim = dim
        
        # Feature extractor for (query, candidate) pairs
        self.scorer = nn.Sequential(
            nn.Linear(dim * 4, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )
    
    def forward(self, query: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """
        Score each candidate for a query.
        
        Args:
            query: (batch, dim) - the query vectors
            candidates: (batch, k, dim) - top-k candidate vectors
        
        Returns:
            scores: (batch, k) - score for each candidate (higher = better)
        """
        batch, k, dim = candidates.shape
        
        # Expand query to match candidates
        q = query.unsqueeze(1).expand(-1, k, -1)  # (batch, k, dim)
        
        # Rich features
        diff = q - candidates          # Difference
        prod = q * candidates          # Element-wise product (similarity signal)
        
        # Concatenate all features
        features = torch.cat([q, candidates, diff, prod], dim=-1)  # (batch, k, dim*4)
        
        # Score each pair
        scores = self.scorer(features).squeeze(-1)  # (batch, k)
        
        return scores
    
    def predict(self, query: torch.Tensor, candidates: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the best candidate and its index.
        
        Returns:
            best_idx: (batch,) - index into candidates of the winner
            best_vector: (batch, dim) - the winning candidate vector
        """
        scores = self.forward(query, candidates)
        best_idx = scores.argmax(dim=1)
        
        # Gather the best vectors
        best_vector = candidates.gather(1, best_idx.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.dim))
        best_vector = best_vector.squeeze(1)
        
        return best_idx, best_vector


class NVFPipeline(nn.Module):
    """
    Complete NVF Pipeline: Soft Attention → Reranker
    
    Stage 1: Find the neighborhood (Top-K)
    Stage 2: Pick the exact match
    """
    
    def __init__(self, soft_attention, reranker, k: int = 5):
        super().__init__()
        
        self.soft_attention = soft_attention
        self.reranker = reranker
        self.k = k
        
        # Freeze soft attention (already trained)
        for param in self.soft_attention.parameters():
            param.requires_grad = False
    
    def forward(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full pipeline: Query → Top-K → Rerank → Final
        
        Returns:
            final_idx: (batch,) - predicted database index
            attention: (batch, n_db) - soft attention weights
            rerank_scores: (batch, k) - reranker scores for top-k
        """
        # Stage 1: Soft Attention
        _, attention = self.soft_attention(query)
        
        # Get Top-K candidates
        topk_scores, topk_indices = attention.topk(self.k, dim=1)
        
        # Get candidate vectors
        candidates = self.soft_attention.db_vectors[topk_indices]  # (batch, k, dim)
        
        # Stage 2: Rerank
        rerank_scores = self.reranker(query, candidates)
        
        # Pick winner
        winner_in_topk = rerank_scores.argmax(dim=1)  # (batch,)
        
        # Map back to database index
        final_idx = topk_indices.gather(1, winner_in_topk.unsqueeze(1)).squeeze(1)
        
        return final_idx, attention, rerank_scores
    
    def get_topk_indices(self, query: torch.Tensor) -> torch.Tensor:
        """Get the Top-K indices from soft attention."""
        _, attention = self.soft_attention(query)
        return attention.topk(self.k, dim=1).indices
