"""
NanoLPU Language Model

A transformer with BitSwitch sparse layers for 2-bit inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List

from ..kernel import BitSwitchLinear


@dataclass
class LMConfig:
    """Configuration for NanoLPU model."""
    vocab_size: int = 73
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 12
    num_tiles: int = 4
    block_size: int = 512
    dropout: float = 0.1
    noise_scale: float = 1.0  # Gate noise for load balancing


class Top1Gate(torch.autograd.Function):
    """Hard top-1 gating with straight-through gradient."""
    
    @staticmethod
    def forward(ctx, logits: torch.Tensor) -> torch.Tensor:
        idx = torch.argmax(logits, dim=-1, keepdim=True)
        return torch.zeros_like(logits).scatter_(-1, idx, 1.0)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


class GatedFFN(nn.Module):
    """Feed-forward network with BitSwitch sparse routing."""
    
    def __init__(self, d_model: int, num_tiles: int, noise_scale: float = 1.0):
        super().__init__()
        self.up_proj = BitSwitchLinear(d_model, d_model * 4, num_tiles)
        self.down_proj = BitSwitchLinear(d_model * 4, d_model, num_tiles)
        self.gate_proj = nn.Linear(d_model, num_tiles)
        self.noise_scale = noise_scale
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gate_logits = self.gate_proj(x)
        
        # Noise injection during training prevents mode collapse
        if self.training and self.noise_scale > 0:
            noise = torch.randn_like(gate_logits) * self.noise_scale
            gate_logits = gate_logits + noise
        
        gate = Top1Gate.apply(gate_logits)
        
        B, T, C = x.shape
        x_flat = x.view(B * T, C)
        gate_flat = gate.view(B * T, -1)
        
        hidden = F.relu(self.up_proj(x_flat, gate_flat))
        out = self.down_proj(hidden, torch.ones_like(gate_flat))
        
        return out.view(B, T, C), gate


class TransformerBlock(nn.Module):
    """Transformer block with causal self-attention and gated FFN."""
    
    def __init__(self, config: LMConfig):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            config.d_model, config.n_heads, 
            dropout=config.dropout, batch_first=True
        )
        self.ffn = GatedFFN(config.d_model, config.num_tiles, config.noise_scale)
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        
        # Self-attention with causal mask
        normed = self.ln1(x)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        attn_out, _ = self.attention(
            normed, normed, normed, 
            attn_mask=causal_mask, is_causal=True
        )
        x = x + attn_out
        
        # Feed-forward
        ffn_out, gates = self.ffn(self.ln2(x))
        x = x + ffn_out
        
        return x, gates


class NanoLPU(nn.Module):
    """
    NanoLPU Language Model
    
    A transformer with BitSwitch sparse layers for efficient 2-bit inference.
    
    Args:
        config: Model configuration (or uses defaults if None)
    """
    
    def __init__(self, config: Optional[LMConfig] = None):
        super().__init__()
        
        if config is None:
            config = LMConfig()
        self.config = config
        
        # Embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.block_size, config.d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Output
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, BitSwitchLinear):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(
        self, 
        idx: torch.Tensor, 
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            idx: Input token indices [batch, seq_len]
            targets: Target token indices for loss computation [batch, seq_len]
            
        Returns:
            logits: Output logits [batch, seq_len, vocab_size]
            loss: Cross-entropy loss (if targets provided)
            all_gates: Gate activations from each layer
        """
        B, T = idx.shape
        
        # Embeddings
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        
        # Transformer blocks
        all_gates = []
        for block in self.blocks:
            x, gates = block(x)
            all_gates.append(gates)
        
        # Output
        x = self.ln_f(x)
        logits = self.head(x)
        
        # Loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1)
            )
        
        return logits, loss, all_gates
    
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 0.9
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.
        
        Args:
            idx: Starting token indices [batch, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            
        Returns:
            Extended token sequence [batch, seq_len + max_new_tokens]
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Crop to block size
            idx_cond = idx[:, -self.config.block_size:]
            
            # Forward pass
            with torch.no_grad():
                logits, _, _ = self.forward(idx_cond)
            
            # Get last token logits
            logits = logits[:, -1, :] / temperature
            
            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumsum = torch.cumsum(probs, dim=-1)
            
            # Remove tokens beyond top_p
            mask = cumsum - probs > top_p
            sorted_logits[mask] = float('-inf')
            
            # Sample
            probs = F.softmax(sorted_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_token = sorted_indices.gather(-1, next_token)
            
            # Append
            idx = torch.cat([idx, next_token], dim=1)
        
        return idx
    
    @property
    def num_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Character tokenizer (matching training)
DEFAULT_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'\"-\n"

def create_tokenizer(chars: str = DEFAULT_CHARS):
    """Create simple character-level tokenizer."""
    chars = sorted(list(set(chars)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    def encode(s: str) -> List[int]:
        return [stoi.get(c, 0) for c in s]
    
    def decode(tokens: List[int]) -> str:
        return ''.join(itos.get(i, '?') for i in tokens)
    
    return encode, decode, len(chars)
