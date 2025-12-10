"""
FU_MOVE: Datapath Unit (Binary Encoded, Lightweight)

Handles: LDA, LDX, LDY, STA, STX, STY, TAX, TXA, TAY, TYA, NOP

These are simple data movement operations - mostly identity mappings.
A very shallow network suffices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class MoveConfig:
    """Configuration for the Move functional unit."""
    hidden_dim: int = 64  # Small - these are simple operations
    dropout: float = 0.1


class FU_MOVE(nn.Module):
    """
    Neural Datapath Unit
    
    Operations:
        LDA/LDX/LDY: Load register from memory/immediate
        STA/STX/STY: Store register to memory
        TAX/TXA/TAY/TYA: Transfer between registers
        NOP: No operation
    
    These are mostly identity operations (copy value, set N/Z flags).
    A minimal network is sufficient.
    """
    
    def __init__(self, config: MoveConfig = None):
        super().__init__()
        
        if config is None:
            config = MoveConfig()
        self.config = config
        
        # Input: A(8) + X(8) + Y(8) + Operand(8) + Opcode_embed(16) = 48
        self.opcode_emb = nn.Embedding(256, 16)
        
        # Simple 2-layer MLP (these ops are trivial)
        self.net = nn.Sequential(
            nn.Linear(48, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )
        
        # Output: A(8) + X(8) + Y(8) + Flags(2: N, Z)
        self.output_head = nn.Linear(config.hidden_dim, 26)
    
    def _encode_binary(self, value: torch.Tensor) -> torch.Tensor:
        """Encode integer to 8-bit binary tensor."""
        bits = []
        for i in range(8):
            bits.append(((value >> i) & 1).float())
        return torch.stack(bits, dim=-1)
    
    def _decode_binary(self, tensor: torch.Tensor) -> torch.Tensor:
        """Decode 8-bit binary tensor to integer."""
        active = (tensor > 0.5).long()
        result = torch.zeros(tensor.shape[0], dtype=torch.long, device=tensor.device)
        for i in range(8):
            result |= (active[:, i] << i)
        return result
    
    def forward(
        self,
        a: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        operand: torch.Tensor,
        opcode: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            a, x, y: Register values [batch]
            operand: Memory/immediate value [batch]
            opcode: Operation code [batch]
        
        Returns:
            output_logits: [batch, 26] (A:8 + X:8 + Y:8 + N:1 + Z:1)
        """
        # Encode inputs
        a_bin = self._encode_binary(a)
        x_bin = self._encode_binary(x)
        y_bin = self._encode_binary(y)
        op_bin = self._encode_binary(operand)
        op_emb = self.opcode_emb(opcode)
        
        # Concatenate
        inp = torch.cat([a_bin, x_bin, y_bin, op_bin, op_emb], dim=-1)
        
        # Forward
        h = self.net(inp)
        return self.output_head(h)
    
    def predict(
        self,
        a: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        operand: torch.Tensor,
        opcode: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict new A, X, Y values and flags."""
        self.eval()
        with torch.no_grad():
            out = self.forward(a, x, y, operand, opcode)
            probs = torch.sigmoid(out)
            
            new_a = self._decode_binary(probs[:, :8])
            new_x = self._decode_binary(probs[:, 8:16])
            new_y = self._decode_binary(probs[:, 16:24])
            flags = (probs[:, 24:] > 0.5).long()  # N, Z
            
            return new_a, new_x, new_y, flags
    
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def move_loss(
    output_logits: torch.Tensor,
    target_a: torch.Tensor,
    target_x: torch.Tensor,
    target_y: torch.Tensor,
    target_flags: torch.Tensor,  # [batch, 2] for N, Z
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute Move unit loss."""
    device = output_logits.device
    
    # Encode targets
    def encode(val):
        bits = []
        for i in range(8):
            bits.append(((val >> i) & 1).float())
        return torch.stack(bits, dim=-1).to(device)
    
    target = torch.cat([
        encode(target_a),
        encode(target_x),
        encode(target_y),
        target_flags.float().to(device)
    ], dim=-1)
    
    loss = F.binary_cross_entropy_with_logits(output_logits, target)
    
    return loss, {'move_loss': loss.item()}


# Ground truth
def execute_move_op(opcode: int, a: int, x: int, y: int, operand: int) -> Tuple[int, int, int, int, int]:
    """Execute a 6502 move operation. Returns (new_a, new_x, new_y, n_flag, z_flag)."""
    new_a, new_x, new_y = a, x, y
    affected_val = a  # Value that determines N/Z flags
    
    # LDA
    if opcode in [0xA9, 0xA5, 0xB5, 0xAD, 0xBD, 0xB9, 0xA1, 0xB1]:
        new_a = operand
        affected_val = operand
    
    # LDX
    elif opcode in [0xA2, 0xA6, 0xB6, 0xAE, 0xBE]:
        new_x = operand
        affected_val = operand
    
    # LDY
    elif opcode in [0xA0, 0xA4, 0xB4, 0xAC, 0xBC]:
        new_y = operand
        affected_val = operand
    
    # STA, STX, STY - don't change registers, just memory (handled externally)
    elif opcode in [0x85, 0x95, 0x8D, 0x9D, 0x99, 0x81, 0x91,
                    0x86, 0x96, 0x8E, 0x84, 0x94, 0x8C]:
        pass  # No register change
    
    # TAX
    elif opcode == 0xAA:
        new_x = a
        affected_val = a
    
    # TXA
    elif opcode == 0x8A:
        new_a = x
        affected_val = x
    
    # TAY
    elif opcode == 0xA8:
        new_y = a
        affected_val = a
    
    # TYA
    elif opcode == 0x98:
        new_a = y
        affected_val = y
    
    # NOP
    elif opcode == 0xEA:
        pass
    
    n_flag = (affected_val >> 7) & 1
    z_flag = 1 if affected_val == 0 else 0
    
    return new_a, new_x, new_y, n_flag, z_flag


if __name__ == "__main__":
    config = MoveConfig()
    move = FU_MOVE(config)
    print(f"FU_MOVE Parameters: {move.num_parameters:,}")
