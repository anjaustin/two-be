"""
FU_ALU: Arithmetic Logic Unit (Soroban Encoded)

Handles: ADC, SBC, CMP, CPX, CPY, INC, DEC, INX, DEX, INY, DEY

Key insight: These operations involve carry propagation, which is
why the monolithic model failed (3% accuracy). Soroban encoding
makes carry visible as "column overflow" - a spatial operation
the model can recognize.

Input: Raw integer state (encoded to Soroban internally)
Output: Soroban-encoded result + flags
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from dataclasses import dataclass

from .abacus import SorobanEncoder
from ..kernel import TriXLinear


@dataclass
class ALUConfig:
    """Configuration for the ALU functional unit."""
    hidden_dim: int = 256
    num_layers: int = 3
    num_tiles: int = 4
    dropout: float = 0.1
    soroban_dim: int = 32


class FU_ALU(nn.Module):
    """
    Neural ALU with Soroban Encoding
    
    The Trojan Horse: Arithmetic disguised as geometry.
    
    Operations:
        ADC: A + M + C -> A (with carry)
        SBC: A - M - ~C -> A (with borrow)
        CMP/CPX/CPY: A/X/Y - M (set flags only)
        INC/DEC: M +/- 1 -> M
        INX/DEX/INY/DEY: X/Y +/- 1 -> X/Y
    
    All operations use Soroban encoding internally to make
    carry propagation visible as column overflow.
    """
    
    def __init__(self, config: ALUConfig = None):
        super().__init__()
        
        if config is None:
            config = ALUConfig()
        self.config = config
        
        # Soroban encoder (32-bit thermometer per register)
        self.soroban = SorobanEncoder(embed_dim=32)
        
        # Input processing
        # A(32) + Operand(32) + Carry(1) + Opcode_embed(16) = 81
        self.opcode_emb = nn.Embedding(256, 16)
        
        # Main network with TriX sparse layers
        self.layers = nn.ModuleList()
        
        input_dim = 81
        for i in range(config.num_layers):
            out_dim = config.hidden_dim
            self.layers.append(TriXLinear(input_dim, out_dim, config.num_tiles))
            input_dim = out_dim
        
        # Gate network for TriX routing
        self.gate = nn.Linear(81, config.num_tiles)
        
        # Output heads
        # Result: 32-bit Soroban
        # Flags: N, Z, C, V (4 bits)
        self.result_head = nn.Linear(config.hidden_dim, config.soroban_dim)
        self.flags_head = nn.Linear(config.hidden_dim, 4)
        
        self.dropout = nn.Dropout(config.dropout)
    
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
            operand: Memory/immediate value [batch] (0-255)
            carry_in: Carry flag [batch] (0 or 1)
            opcode: Operation code [batch]
        
        Returns:
            result_logits: [batch, 32] Soroban logits for result
            flags_logits: [batch, 4] logits for N, Z, C, V flags
        """
        # Encode inputs to Soroban
        a_soroban = self.soroban.encode_batch(a)  # [batch, 32]
        op_soroban = self.soroban.encode_batch(operand)  # [batch, 32]
        
        # Get opcode embedding
        op_emb = self.opcode_emb(opcode)  # [batch, 16]
        
        # Concatenate inputs
        x = torch.cat([
            a_soroban,
            op_soroban,
            carry_in.float().unsqueeze(-1),
            op_emb
        ], dim=-1)  # [batch, 81]
        
        # Compute gate (for TriX routing)
        gate_logits = self.gate(x)
        gate = F.softmax(gate_logits, dim=-1)
        
        # Forward through TriX layers
        for layer in self.layers:
            x = layer(x, gate)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Output heads
        result_logits = self.result_head(x)  # [batch, 32]
        flags_logits = self.flags_head(x)    # [batch, 4]
        
        return result_logits, flags_logits
    
    def predict(
        self,
        a: torch.Tensor,
        operand: torch.Tensor,
        carry_in: torch.Tensor,
        opcode: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict result and flags (decoded to integers).
        
        Returns:
            result: [batch] integer results (0-255)
            flags: [batch, 4] binary flags (N, Z, C, V)
        """
        self.eval()
        with torch.no_grad():
            result_logits, flags_logits = self.forward(a, operand, carry_in, opcode)
            
            # Decode Soroban to integer
            result_probs = torch.sigmoid(result_logits)
            result = self.soroban.decode(result_probs)
            
            # Decode flags
            flags = (torch.sigmoid(flags_logits) > 0.5).long()
            
            return result, flags
    
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def alu_loss(
    result_logits: torch.Tensor,
    flags_logits: torch.Tensor,
    target_result: torch.Tensor,
    target_flags: torch.Tensor,
    soroban_encoder: SorobanEncoder
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute ALU loss.
    
    Args:
        result_logits: [batch, 32] predicted Soroban logits
        flags_logits: [batch, 4] predicted flag logits
        target_result: [batch] ground truth result (0-255)
        target_flags: [batch, 4] ground truth flags (N, Z, C, V)
        soroban_encoder: Encoder for converting targets
    
    Returns:
        total_loss: Combined loss
        loss_dict: Per-component losses
    """
    # Encode target result to Soroban
    target_soroban = soroban_encoder.encode_batch(target_result).to(result_logits.device)
    
    # Result loss (BCE for thermometer structure)
    result_loss = F.binary_cross_entropy_with_logits(result_logits, target_soroban)
    
    # Flags loss (BCE for each flag)
    flags_loss = F.binary_cross_entropy_with_logits(flags_logits, target_flags.float())
    
    # Combined loss (weight result more heavily)
    total_loss = result_loss + 0.5 * flags_loss
    
    loss_dict = {
        'result_loss': result_loss.item(),
        'flags_loss': flags_loss.item(),
        'total_loss': total_loss.item(),
    }
    
    return total_loss, loss_dict


# Ground truth 6502 ALU operations for training data generation
def execute_alu_op(opcode: int, a: int, operand: int, carry: int, x: int = 0, y: int = 0) -> Tuple[int, int, int, int, int]:
    """
    Execute a 6502 ALU operation.
    
    Returns:
        result: The computed result (0-255)
        n_flag: Negative flag
        z_flag: Zero flag
        c_flag: Carry flag
        v_flag: Overflow flag
    """
    result = 0
    n_flag = 0
    z_flag = 0
    c_flag = carry
    v_flag = 0
    
    # ADC - Add with Carry
    if opcode in [0x69, 0x65, 0x75, 0x6D, 0x7D, 0x79, 0x61, 0x71]:
        temp = a + operand + carry
        result = temp & 0xFF
        c_flag = 1 if temp > 255 else 0
        # Overflow: positive + positive = negative, or negative + negative = positive
        v_flag = 1 if ((a ^ result) & (operand ^ result) & 0x80) else 0
    
    # SBC - Subtract with Carry (borrow)
    elif opcode in [0xE9, 0xE5, 0xF5, 0xED, 0xFD, 0xF9, 0xE1, 0xF1]:
        temp = a - operand - (1 - carry)
        result = temp & 0xFF
        c_flag = 0 if temp < 0 else 1
        v_flag = 1 if ((a ^ operand) & (a ^ result) & 0x80) else 0
    
    # CMP - Compare Accumulator
    elif opcode in [0xC9, 0xC5, 0xD5, 0xCD, 0xDD, 0xD9, 0xC1, 0xD1]:
        temp = a - operand
        result = a  # CMP doesn't change A
        c_flag = 1 if a >= operand else 0
        n_flag = 1 if (temp & 0x80) else 0
        z_flag = 1 if (temp & 0xFF) == 0 else 0
        return result, n_flag, z_flag, c_flag, v_flag
    
    # CPX - Compare X
    elif opcode in [0xE0, 0xE4, 0xEC]:
        temp = x - operand
        result = x  # CPX doesn't change X
        c_flag = 1 if x >= operand else 0
        n_flag = 1 if (temp & 0x80) else 0
        z_flag = 1 if (temp & 0xFF) == 0 else 0
        return result, n_flag, z_flag, c_flag, v_flag
    
    # CPY - Compare Y
    elif opcode in [0xC0, 0xC4, 0xCC]:
        temp = y - operand
        result = y  # CPY doesn't change Y
        c_flag = 1 if y >= operand else 0
        n_flag = 1 if (temp & 0x80) else 0
        z_flag = 1 if (temp & 0xFF) == 0 else 0
        return result, n_flag, z_flag, c_flag, v_flag
    
    # INC - Increment Memory
    elif opcode in [0xE6, 0xF6, 0xEE, 0xFE]:
        result = (operand + 1) & 0xFF
    
    # DEC - Decrement Memory
    elif opcode in [0xC6, 0xD6, 0xCE, 0xDE]:
        result = (operand - 1) & 0xFF
    
    # INX - Increment X
    elif opcode == 0xE8:
        result = (x + 1) & 0xFF
    
    # DEX - Decrement X
    elif opcode == 0xCA:
        result = (x - 1) & 0xFF
    
    # INY - Increment Y
    elif opcode == 0xC8:
        result = (y + 1) & 0xFF
    
    # DEY - Decrement Y
    elif opcode == 0x88:
        result = (y - 1) & 0xFF
    
    else:
        result = a  # Unknown op, return A unchanged
    
    # Set N and Z flags based on result
    n_flag = 1 if (result & 0x80) else 0
    z_flag = 1 if result == 0 else 0
    
    return result, n_flag, z_flag, c_flag, v_flag


if __name__ == "__main__":
    # Quick test
    config = ALUConfig()
    alu = FU_ALU(config)
    
    print(f"FU_ALU Parameters: {alu.num_parameters:,}")
    
    # Test forward pass
    batch_size = 4
    a = torch.randint(0, 256, (batch_size,))
    operand = torch.randint(0, 256, (batch_size,))
    carry = torch.randint(0, 2, (batch_size,))
    opcode = torch.full((batch_size,), 0x69)  # ADC immediate
    
    result_logits, flags_logits = alu(a, operand, carry, opcode)
    print(f"Result shape: {result_logits.shape}")
    print(f"Flags shape: {flags_logits.shape}")
    
    # Test prediction
    result, flags = alu.predict(a, operand, carry, opcode)
    print(f"Predicted results: {result}")
    print(f"Predicted flags: {flags}")
