"""
Neural 6502 with Soroban Encoding

The Trojan Horse: Arithmetic disguised as Geometry.

Key changes from standard NeuralCPU:
1. Registers encoded as 32-bit Soroban thermometers (not 256-class embeddings)
2. Output is 32-bit thermometer prediction (not 256-class classification)
3. Loss is BCEWithLogitsLoss (learn bead structure, not class ID)

Hypothesis: By encoding registers spatially, ADC becomes "column overflow"
which looks like shifting - something the model already knows (97% accuracy).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

from ..kernel import TriXLinear
from .abacus import SorobanEncoder


@dataclass
class SorobanCPUConfig:
    """Configuration for Soroban-encoded NeuralCPU model."""
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 6
    num_tiles: int = 4
    dropout: float = 0.1
    
    # Soroban encoding
    soroban_dim: int = 32  # Per-register thermometer size
    
    # Registers to encode with Soroban (8-bit arithmetic registers)
    soroban_registers: Tuple[str, ...] = ('A', 'X', 'Y', 'SP')
    
    # Registers to encode as binary/embedding (flags, PC)
    binary_registers: Tuple[str, ...] = ('P', 'PCH', 'PCL')
    
    # Vocabularies
    register_vocab: int = 256
    opcode_vocab: int = 256
    
    @property
    def all_registers(self) -> Tuple[str, ...]:
        return self.soroban_registers + self.binary_registers


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
    """Feed-forward network with TriX sparse routing."""
    
    def __init__(self, d_model: int, num_tiles: int, dropout: float = 0.1):
        super().__init__()
        self.up_proj = TriXLinear(d_model, d_model * 4, num_tiles)
        self.down_proj = TriXLinear(d_model * 4, d_model, num_tiles)
        self.gate_proj = nn.Linear(d_model, num_tiles)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        orig_shape = x.shape
        if x.dim() == 3:
            B, T, C = x.shape
            x = x.view(B * T, C)
        
        gate = Top1Gate.apply(self.gate_proj(x))
        h = F.relu(self.up_proj(x, gate))
        h = self.dropout(h)
        out = self.down_proj(h, gate)
        
        if len(orig_shape) == 3:
            out = out.view(orig_shape[0], orig_shape[1], -1)
            gate = gate.view(orig_shape[0], orig_shape[1], -1)
        
        return out, gate


class CPUBlock(nn.Module):
    """Transformer block for CPU state processing."""
    
    def __init__(self, d_model: int, n_heads: int, num_tiles: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = GatedFFN(d_model, num_tiles, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        normed = self.norm1(x)
        attn_out, _ = self.attention(normed, normed, normed)
        x = x + self.dropout(attn_out)
        
        normed = self.norm2(x)
        ffn_out, gate = self.ffn(normed)
        x = x + self.dropout(ffn_out)
        
        return x, gate


class SorobanCPU(nn.Module):
    """
    Neural 6502 with Soroban Encoding
    
    The Trojan Horse Architecture:
    - Arithmetic registers (A, X, Y, SP) encoded as 32-bit thermometers
    - Model sees addition as "column overflow" = spatial operation
    - Gradients for ADC align with gradients for ASL/LSR
    
    Input state:
        - A, X, Y, SP: Integer values → converted to Soroban internally
        - P, PCH, PCL: Binary/embedding (flags don't need Soroban)
        - Op: Opcode
        - Val: Operand
    
    Output:
        - Soroban predictions for A, X, Y, SP (32-bit thermometers)
        - Standard logits for P, PCH, PCL
    """
    
    def __init__(self, config: Optional[SorobanCPUConfig] = None):
        super().__init__()
        
        if config is None:
            config = SorobanCPUConfig()
        self.config = config
        
        # Soroban encoder (shared)
        self.soroban = SorobanEncoder(embed_dim=config.d_model)
        
        # Projection from Soroban embedding to model dimension
        # Soroban: [batch, 32, d_model] → need to combine into [batch, d_model]
        self.soroban_proj = nn.Linear(32 * config.d_model, config.d_model)
        
        # Standard embeddings for non-Soroban registers
        self.register_emb = nn.Embedding(config.register_vocab, config.d_model)
        self.opcode_emb = nn.Embedding(config.opcode_vocab, config.d_model)
        
        # Position embedding: A, X, Y, SP (Soroban) + P, PCH, PCL (binary) + Op, Val
        # = 4 + 3 + 2 = 9 positions
        self.position_emb = nn.Parameter(torch.randn(1, 9, config.d_model) * 0.02)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            CPUBlock(config.d_model, config.n_heads, config.num_tiles, config.dropout)
            for _ in range(config.n_layers)
        ])
        
        # Output heads
        # Soroban registers: predict 32-bit thermometer
        self.soroban_heads = nn.ModuleDict({
            reg: nn.Linear(config.d_model, config.soroban_dim)
            for reg in config.soroban_registers
        })
        
        # Binary registers: predict 256-class logits
        self.binary_heads = nn.ModuleDict({
            reg: nn.Linear(config.d_model, config.register_vocab)
            for reg in config.binary_registers
        })
        
        self.ln_f = nn.LayerNorm(config.d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def _encode_soroban_register(self, value: torch.Tensor) -> torch.Tensor:
        """
        Encode register value as Soroban embedding.
        
        Args:
            value: [batch] integer tensor (0-255)
        Returns:
            [batch, d_model] embedding
        """
        # Encode to thermometer
        soroban_bits = self.soroban.encode_batch(value)  # [batch, 32]
        
        # Project through Soroban layer to get embeddings
        soroban_emb = self.soroban(soroban_bits)  # [batch, 32, d_model]
        
        # Flatten and project to single embedding
        batch = soroban_emb.shape[0]
        flat = soroban_emb.view(batch, -1)  # [batch, 32 * d_model]
        return self.soroban_proj(flat)  # [batch, d_model]
    
    def forward(
        self,
        state: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass.
        
        Args:
            state: Dictionary with register values and opcode/operand
        
        Returns:
            soroban_preds: Dict of 32-bit thermometer logits for A, X, Y, SP
            binary_preds: Dict of 256-class logits for P, PCH, PCL
            gates: Gate activations
        """
        batch_size = state['A'].shape[0]
        device = state['A'].device
        
        embeddings = []
        
        # Soroban registers
        for reg in self.config.soroban_registers:
            emb = self._encode_soroban_register(state[reg])
            embeddings.append(emb)
        
        # Binary registers
        for reg in self.config.binary_registers:
            emb = self.register_emb(state[reg])
            embeddings.append(emb)
        
        # Opcode and operand
        embeddings.append(self.opcode_emb(state['Op']))
        embeddings.append(self.register_emb(state['Val']))
        
        # Stack and add position
        x = torch.stack(embeddings, dim=1)  # [batch, 9, d_model]
        x = x + self.position_emb
        
        # Transformer
        all_gates = []
        for block in self.blocks:
            x, gate = block(x)
            all_gates.append(gate)
        
        x = self.ln_f(x)
        
        # Predict Soroban registers (positions 0-3)
        soroban_preds = {}
        for i, reg in enumerate(self.config.soroban_registers):
            soroban_preds[reg] = self.soroban_heads[reg](x[:, i])  # [batch, 32]
        
        # Predict binary registers (positions 4-6)
        binary_preds = {}
        for i, reg in enumerate(self.config.binary_registers):
            binary_preds[reg] = self.binary_heads[reg](x[:, 4 + i])  # [batch, 256]
        
        gates = torch.stack(all_gates, dim=0).mean(dim=0)
        
        return soroban_preds, binary_preds, gates
    
    def predict_state(
        self,
        state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Predict next state values.
        
        Returns decoded integer values for all registers.
        """
        self.eval()
        with torch.no_grad():
            soroban_preds, binary_preds, _ = self.forward(state)
            
            result = {}
            
            # Decode Soroban predictions
            for reg, logits in soroban_preds.items():
                probs = torch.sigmoid(logits)
                result[reg] = self.soroban.decode(probs)
            
            # Decode binary predictions
            for reg, logits in binary_preds.items():
                result[reg] = logits.argmax(dim=-1)
            
            return result
    
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def soroban_cpu_loss(
    soroban_preds: Dict[str, torch.Tensor],
    binary_preds: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    soroban_encoder: SorobanEncoder,
    soroban_registers: Tuple[str, ...] = ('A', 'X', 'Y', 'SP'),
    binary_registers: Tuple[str, ...] = ('P', 'PCH', 'PCL'),
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute loss for SorobanCPU.
    
    Soroban registers use BCEWithLogitsLoss (learn bead structure).
    Binary registers use CrossEntropyLoss (standard classification).
    
    Returns:
        total_loss: Combined loss
        loss_dict: Per-register losses for logging
    """
    loss_dict = {}
    total_loss = 0.0
    
    # Soroban losses (BCE)
    for reg in soroban_registers:
        target_int = targets[reg]
        target_soroban = soroban_encoder.encode_batch(target_int).to(soroban_preds[reg].device)
        
        loss = F.binary_cross_entropy_with_logits(soroban_preds[reg], target_soroban)
        loss_dict[f'{reg}_loss'] = loss.item()
        total_loss = total_loss + loss
    
    # Binary losses (CrossEntropy)
    for reg in binary_registers:
        loss = F.cross_entropy(binary_preds[reg], targets[reg])
        loss_dict[f'{reg}_loss'] = loss.item()
        total_loss = total_loss + loss
    
    return total_loss, loss_dict


# Accuracy evaluation for Soroban model
def evaluate_soroban_accuracy(
    model: SorobanCPU,
    dataloader,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Evaluate per-register accuracy.
    
    Returns accuracy for each register plus overall.
    """
    model.eval()
    
    correct = {reg: 0 for reg in model.config.all_registers}
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            state = {k: v.to(device) for k, v in batch['input'].items()}
            targets = {k: v.to(device) for k, v in batch['target'].items()}
            
            preds = model.predict_state(state)
            
            for reg in model.config.all_registers:
                correct[reg] += (preds[reg] == targets[reg]).sum().item()
            
            total += state['A'].shape[0]
    
    accuracies = {reg: correct[reg] / total for reg in model.config.all_registers}
    accuracies['overall'] = sum(correct.values()) / (total * len(correct))
    
    return accuracies
