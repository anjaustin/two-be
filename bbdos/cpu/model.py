"""
Neural 6502 Model Architecture

A transformer-based model that predicts CPU register state transitions.
Uses TriX sparse layers for efficient inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

from ..kernel import TriXLinear


@dataclass
class CPUConfig:
    """Configuration for NeuralCPU model."""
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 6
    num_tiles: int = 4
    dropout: float = 0.1
    
    # Input/output sizes
    register_vocab: int = 256  # 8-bit values
    opcode_vocab: int = 256    # 6502 opcodes
    
    # Register names
    registers: Tuple[str, ...] = ('A', 'X', 'Y', 'SP', 'P', 'PCH', 'PCL')
    
    @property
    def num_registers(self) -> int:
        return len(self.registers)


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
        # Handle 3D input [batch, seq, d_model]
        orig_shape = x.shape
        if x.dim() == 3:
            B, T, C = x.shape
            x = x.view(B * T, C)
        
        gate = Top1Gate.apply(self.gate_proj(x))
        h = F.relu(self.up_proj(x, gate))
        h = self.dropout(h)
        out = self.down_proj(h, gate)
        
        # Restore shape
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
        # Self-attention
        normed = self.norm1(x)
        attn_out, _ = self.attention(normed, normed, normed)
        x = x + self.dropout(attn_out)
        
        # Feed-forward
        normed = self.norm2(x)
        ffn_out, gate = self.ffn(normed)
        x = x + self.dropout(ffn_out)
        
        return x, gate


class NeuralCPU(nn.Module):
    """
    Neural 6502 CPU Emulator
    
    Predicts the next CPU state given current state and instruction.
    
    Input state:
        - A, X, Y, SP, P: 8-bit register values
        - PCH, PCL: 16-bit program counter (split)
        - Op: Current opcode
        - Val: Operand value
    
    Output:
        - Predicted next values for each register
        - Gate activations for analysis
    """
    
    def __init__(self, config: Optional[CPUConfig] = None):
        super().__init__()
        
        if config is None:
            config = CPUConfig()
        self.config = config
        
        # Input embeddings
        self.register_emb = nn.Embedding(config.register_vocab, config.d_model)
        self.opcode_emb = nn.Embedding(config.opcode_vocab, config.d_model)
        self.position_emb = nn.Parameter(torch.randn(1, 9, config.d_model) * 0.02)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            CPUBlock(config.d_model, config.n_heads, config.num_tiles, config.dropout)
            for _ in range(config.n_layers)
        ])
        
        # Output heads (one per register)
        self.heads = nn.ModuleDict({
            reg: nn.Linear(config.d_model, config.register_vocab)
            for reg in config.registers
        })
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model)
        
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
    
    def forward(
        self, 
        state: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass.
        
        Args:
            state: Dictionary with keys:
                - 'A', 'X', 'Y', 'SP', 'P', 'PCH', 'PCL': Register values [batch]
                - 'Op': Opcode [batch]
                - 'Val': Operand [batch]
        
        Returns:
            predictions: Dictionary mapping register names to logits [batch, 256]
            gates: Aggregated gate activations [batch, num_tiles]
        """
        batch_size = state['A'].shape[0]
        device = state['A'].device
        
        # Embed each input
        embeddings = []
        for key in ['A', 'X', 'Y', 'SP', 'P', 'PCH', 'PCL']:
            embeddings.append(self.register_emb(state[key]))
        embeddings.append(self.opcode_emb(state['Op']))
        embeddings.append(self.register_emb(state['Val']))
        
        # Stack and add position
        x = torch.stack(embeddings, dim=1)  # [batch, 9, d_model]
        x = x + self.position_emb
        
        # Process through transformer
        all_gates = []
        for block in self.blocks:
            x, gate = block(x)
            all_gates.append(gate)
        
        x = self.ln_f(x)
        
        # Predict each register from its corresponding position
        predictions = {}
        for i, reg in enumerate(self.config.registers):
            predictions[reg] = self.heads[reg](x[:, i])
        
        # Aggregate gates
        gates = torch.stack(all_gates, dim=0).mean(dim=0)
        
        return predictions, gates
    
    def predict_state(
        self,
        state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Predict next state values (argmax).
        
        Args:
            state: Current CPU state
            
        Returns:
            Dictionary mapping register names to predicted values
        """
        self.eval()
        with torch.no_grad():
            preds, _ = self.forward(state)
            return {reg: logits.argmax(dim=-1) for reg, logits in preds.items()}
    
    @property
    def num_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Standard opcode names for display
OPCODE_NAMES = {
    0x00: "BRK", 0x01: "ORA_IX", 0x05: "ORA_ZP", 0x06: "ASL_ZP",
    0x08: "PHP", 0x09: "ORA_IMM", 0x0A: "ASL_A", 0x0D: "ORA_ABS",
    0x10: "BPL", 0x18: "CLC", 0x20: "JSR", 0x29: "AND_IMM",
    0x2A: "ROL_A", 0x38: "SEC", 0x48: "PHA", 0x49: "EOR_IMM",
    0x4A: "LSR_A", 0x4C: "JMP_ABS", 0x58: "CLI", 0x60: "RTS",
    0x68: "PLA", 0x69: "ADC_IMM", 0x6A: "ROR_A", 0x78: "SEI",
    0x85: "STA_ZP", 0x86: "STX_ZP", 0x88: "DEY", 0x8A: "TXA",
    0x8D: "STA_ABS", 0x98: "TYA", 0x9A: "TXS", 0xA0: "LDY_IMM",
    0xA2: "LDX_IMM", 0xA8: "TAY", 0xA9: "LDA_IMM", 0xAA: "TAX",
    0xB8: "CLV", 0xBA: "TSX", 0xC8: "INY", 0xC9: "CMP_IMM",
    0xCA: "DEX", 0xD0: "BNE", 0xD8: "CLD", 0xE0: "CPX_IMM",
    0xE8: "INX", 0xE9: "SBC_IMM", 0xEA: "NOP", 0xF0: "BEQ",
    0xF8: "SED",
}
