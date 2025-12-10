"""
FU_STACK: Stack Operations Unit

Handles: PHA, PLA, PHP, PLP, TXS, TSX

Simple push/pull operations on the stack.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class StackConfig:
    """Configuration for the Stack functional unit."""
    hidden_dim: int = 64
    dropout: float = 0.1


class FU_STACK(nn.Module):
    """
    Neural Stack Unit
    
    Operations:
        PHA: Push A to stack
        PLA: Pull A from stack
        PHP: Push P (flags) to stack
        PLP: Pull P from stack
        TXS: Transfer X to SP
        TSX: Transfer SP to X
    
    Simple operations with state tracking.
    """
    
    def __init__(self, config: StackConfig = None):
        super().__init__()
        
        if config is None:
            config = StackConfig()
        self.config = config
        
        # Input: A(8) + X(8) + SP(8) + Flags(8) + Stack_top(8) + Opcode_embed(16) = 56
        self.opcode_emb = nn.Embedding(256, 16)
        
        # Simple network
        self.net = nn.Sequential(
            nn.Linear(56, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )
        
        # Output: A(8) + X(8) + SP(8) + Flags(8) + Value_to_push(8) + N(1) + Z(1)
        self.output_head = nn.Linear(config.hidden_dim, 42)
    
    def _encode_8bit(self, value: torch.Tensor) -> torch.Tensor:
        bits = []
        for i in range(8):
            bits.append(((value >> i) & 1).float())
        return torch.stack(bits, dim=-1)
    
    def _decode_8bit(self, tensor: torch.Tensor) -> torch.Tensor:
        active = (tensor > 0.5).long()
        result = torch.zeros(tensor.shape[0], dtype=torch.long, device=tensor.device)
        for i in range(8):
            result |= (active[:, i] << i)
        return result
    
    def forward(
        self,
        a: torch.Tensor,
        x: torch.Tensor,
        sp: torch.Tensor,
        flags: torch.Tensor,
        stack_top: torch.Tensor,
        opcode: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            a: Accumulator [batch]
            x: X register [batch]
            sp: Stack pointer [batch]
            flags: Processor status [batch]
            stack_top: Value at top of stack [batch]
            opcode: Operation code [batch]
        
        Returns:
            output_logits: [batch, 42]
        """
        # Encode inputs
        a_bin = self._encode_8bit(a)
        x_bin = self._encode_8bit(x)
        sp_bin = self._encode_8bit(sp)
        flags_bin = self._encode_8bit(flags)
        stack_bin = self._encode_8bit(stack_top)
        op_emb = self.opcode_emb(opcode)
        
        inp = torch.cat([a_bin, x_bin, sp_bin, flags_bin, stack_bin, op_emb], dim=-1)
        
        h = self.net(inp)
        return self.output_head(h)
    
    def predict(
        self,
        a: torch.Tensor,
        x: torch.Tensor,
        sp: torch.Tensor,
        flags: torch.Tensor,
        stack_top: torch.Tensor,
        opcode: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """Predict new state."""
        self.eval()
        with torch.no_grad():
            out = self.forward(a, x, sp, flags, stack_top, opcode)
            probs = torch.sigmoid(out)
            
            new_a = self._decode_8bit(probs[:, :8])
            new_x = self._decode_8bit(probs[:, 8:16])
            new_sp = self._decode_8bit(probs[:, 16:24])
            new_flags = self._decode_8bit(probs[:, 24:32])
            push_val = self._decode_8bit(probs[:, 32:40])
            nz_flags = (probs[:, 40:] > 0.5).long()
            
            return new_a, new_x, new_sp, new_flags, push_val, nz_flags
    
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def stack_loss(
    output_logits: torch.Tensor,
    targets: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute Stack unit loss."""
    device = output_logits.device
    
    def encode(val):
        bits = []
        for i in range(8):
            bits.append(((val >> i) & 1).float())
        return torch.stack(bits, dim=-1).to(device)
    
    target = torch.cat([
        encode(targets['a']),
        encode(targets['x']),
        encode(targets['sp']),
        encode(targets['flags']),
        encode(targets['push_val']),
        targets['nz_flags'].float().to(device)
    ], dim=-1)
    
    loss = F.binary_cross_entropy_with_logits(output_logits, target)
    
    return loss, {'stack_loss': loss.item()}


# Ground truth
def execute_stack_op(opcode: int, a: int, x: int, sp: int, flags: int, stack_top: int) -> Tuple[int, int, int, int, int]:
    """
    Execute a 6502 stack operation.
    
    Returns: (new_a, new_x, new_sp, new_flags, value_to_push)
    """
    new_a, new_x, new_sp, new_flags = a, x, sp, flags
    push_val = 0
    
    # PHA - Push Accumulator
    if opcode == 0x48:
        new_sp = (sp - 1) & 0xFF
        push_val = a
    
    # PLA - Pull Accumulator
    elif opcode == 0x68:
        new_a = stack_top
        new_sp = (sp + 1) & 0xFF
        # Set N, Z flags based on pulled value
        n = (stack_top >> 7) & 1
        z = 1 if stack_top == 0 else 0
        new_flags = (flags & 0x7D) | (n << 7) | (z << 1)
    
    # PHP - Push Processor Status
    elif opcode == 0x08:
        new_sp = (sp - 1) & 0xFF
        push_val = flags | 0x30  # B flag set when pushed
    
    # PLP - Pull Processor Status
    elif opcode == 0x28:
        new_flags = stack_top & 0xCF  # Clear B flags
        new_sp = (sp + 1) & 0xFF
    
    # TXS - Transfer X to SP
    elif opcode == 0x9A:
        new_sp = x
        # TXS doesn't affect flags
    
    # TSX - Transfer SP to X
    elif opcode == 0xBA:
        new_x = sp
        # Set N, Z based on SP value
        n = (sp >> 7) & 1
        z = 1 if sp == 0 else 0
        new_flags = (flags & 0x7D) | (n << 7) | (z << 1)
    
    return new_a, new_x, new_sp, new_flags, push_val


if __name__ == "__main__":
    config = StackConfig()
    stack = FU_STACK(config)
    print(f"FU_STACK Parameters: {stack.num_parameters:,}")
