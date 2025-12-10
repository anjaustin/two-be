"""
FU_LOGIC: Logic Unit (Binary Encoded)

Handles: AND, ORA, EOR, BIT, ASL, LSR, ROL, ROR, Flag ops

These are pure bitwise operations with no carry propagation.
Binary encoding is optimal here - no need for Soroban.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from dataclasses import dataclass

from ..kernel import TriXLinear


@dataclass
class LogicConfig:
    """Configuration for the Logic functional unit."""
    hidden_dim: int = 256
    num_layers: int = 2
    num_tiles: int = 4
    dropout: float = 0.1


class FU_LOGIC(nn.Module):
    """
    Neural Logic Unit with Binary Encoding
    
    Operations:
        AND/ORA/EOR: Bitwise logic
        BIT: Test bits (set flags)
        ASL/LSR: Shift left/right
        ROL/ROR: Rotate through carry
        CLC/SEC/etc: Flag manipulation
    
    These operations are inherently parallel (each bit independent)
    so binary encoding is efficient and natural.
    """
    
    def __init__(self, config: LogicConfig = None):
        super().__init__()
        
        if config is None:
            config = LogicConfig()
        self.config = config
        
        # Input: A(8) + Operand(8) + Carry(1) + Opcode_embed(16) = 33
        self.opcode_emb = nn.Embedding(256, 16)
        
        # Main network with TriX sparse layers
        self.layers = nn.ModuleList()
        
        input_dim = 33
        for i in range(config.num_layers):
            out_dim = config.hidden_dim
            self.layers.append(TriXLinear(input_dim, out_dim, config.num_tiles))
            input_dim = out_dim
        
        # Gate network
        self.gate = nn.Linear(33, config.num_tiles)
        
        # Output heads
        self.result_head = nn.Linear(config.hidden_dim, 8)  # 8-bit result
        self.flags_head = nn.Linear(config.hidden_dim, 4)   # N, Z, C, V
        
        self.dropout = nn.Dropout(config.dropout)
    
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
        operand: torch.Tensor,
        carry_in: torch.Tensor,
        opcode: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            a: Accumulator value [batch] (0-255)
            operand: Memory value [batch] (0-255)
            carry_in: Carry flag [batch] (0 or 1)
            opcode: Operation code [batch]
        
        Returns:
            result_logits: [batch, 8] binary logits for result
            flags_logits: [batch, 4] logits for N, Z, C, V flags
        """
        # Encode inputs to binary
        a_bin = self._encode_binary(a)  # [batch, 8]
        op_bin = self._encode_binary(operand)  # [batch, 8]
        
        # Get opcode embedding
        op_emb = self.opcode_emb(opcode)  # [batch, 16]
        
        # Concatenate
        x = torch.cat([
            a_bin,
            op_bin,
            carry_in.float().unsqueeze(-1),
            op_emb
        ], dim=-1)  # [batch, 33]
        
        # Compute gate
        gate_logits = self.gate(x)
        gate = F.softmax(gate_logits, dim=-1)
        
        # Forward through layers
        for layer in self.layers:
            x = layer(x, gate)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Output
        result_logits = self.result_head(x)
        flags_logits = self.flags_head(x)
        
        return result_logits, flags_logits
    
    def predict(
        self,
        a: torch.Tensor,
        operand: torch.Tensor,
        carry_in: torch.Tensor,
        opcode: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict result and flags."""
        self.eval()
        with torch.no_grad():
            result_logits, flags_logits = self.forward(a, operand, carry_in, opcode)
            result = self._decode_binary(torch.sigmoid(result_logits))
            flags = (torch.sigmoid(flags_logits) > 0.5).long()
            return result, flags
    
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def logic_loss(
    result_logits: torch.Tensor,
    flags_logits: torch.Tensor,
    target_result: torch.Tensor,
    target_flags: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute Logic unit loss."""
    # Encode target to binary
    target_bits = []
    for i in range(8):
        target_bits.append(((target_result >> i) & 1).float())
    target_binary = torch.stack(target_bits, dim=-1).to(result_logits.device)
    
    result_loss = F.binary_cross_entropy_with_logits(result_logits, target_binary)
    flags_loss = F.binary_cross_entropy_with_logits(flags_logits, target_flags.float())
    
    total_loss = result_loss + 0.5 * flags_loss
    
    return total_loss, {
        'result_loss': result_loss.item(),
        'flags_loss': flags_loss.item(),
    }


# Ground truth 6502 logic operations
def execute_logic_op(opcode: int, a: int, operand: int, carry: int) -> Tuple[int, int, int, int, int]:
    """Execute a 6502 logic operation."""
    result = a
    n_flag = 0
    z_flag = 0
    c_flag = carry
    v_flag = 0
    
    # AND
    if opcode in [0x29, 0x25, 0x35, 0x2D, 0x3D, 0x39, 0x21, 0x31]:
        result = a & operand
    
    # ORA
    elif opcode in [0x09, 0x05, 0x15, 0x0D, 0x1D, 0x19, 0x01, 0x11]:
        result = a | operand
    
    # EOR
    elif opcode in [0x49, 0x45, 0x55, 0x4D, 0x5D, 0x59, 0x41, 0x51]:
        result = a ^ operand
    
    # BIT
    elif opcode in [0x24, 0x2C]:
        temp = a & operand
        z_flag = 1 if temp == 0 else 0
        n_flag = (operand >> 7) & 1
        v_flag = (operand >> 6) & 1
        return a, n_flag, z_flag, c_flag, v_flag
    
    # ASL (Accumulator or Memory)
    elif opcode == 0x0A:  # ASL A
        c_flag = (a >> 7) & 1
        result = (a << 1) & 0xFF
    elif opcode in [0x06, 0x16, 0x0E, 0x1E]:  # ASL Memory
        c_flag = (operand >> 7) & 1
        result = (operand << 1) & 0xFF
    
    # LSR
    elif opcode == 0x4A:  # LSR A
        c_flag = a & 1
        result = a >> 1
    elif opcode in [0x46, 0x56, 0x4E, 0x5E]:  # LSR Memory
        c_flag = operand & 1
        result = operand >> 1
    
    # ROL
    elif opcode == 0x2A:  # ROL A
        temp = (a << 1) | carry
        c_flag = (a >> 7) & 1
        result = temp & 0xFF
    elif opcode in [0x26, 0x36, 0x2E, 0x3E]:  # ROL Memory
        temp = (operand << 1) | carry
        c_flag = (operand >> 7) & 1
        result = temp & 0xFF
    
    # ROR
    elif opcode == 0x6A:  # ROR A
        c_flag = a & 1
        result = (a >> 1) | (carry << 7)
    elif opcode in [0x66, 0x76, 0x6E, 0x7E]:  # ROR Memory
        c_flag = operand & 1
        result = (operand >> 1) | (carry << 7)
    
    # Flag operations
    elif opcode == 0x18:  # CLC
        c_flag = 0
        return a, n_flag, z_flag, c_flag, v_flag
    elif opcode == 0x38:  # SEC
        c_flag = 1
        return a, n_flag, z_flag, c_flag, v_flag
    elif opcode == 0xB8:  # CLV
        v_flag = 0
        return a, n_flag, z_flag, c_flag, v_flag
    elif opcode in [0x58, 0x78, 0xD8, 0xF8]:  # CLI, SEI, CLD, SED
        return a, n_flag, z_flag, c_flag, v_flag
    
    # Set N and Z
    n_flag = (result >> 7) & 1
    z_flag = 1 if result == 0 else 0
    
    return result, n_flag, z_flag, c_flag, v_flag


if __name__ == "__main__":
    config = LogicConfig()
    logic = FU_LOGIC(config)
    print(f"FU_LOGIC Parameters: {logic.num_parameters:,}")
