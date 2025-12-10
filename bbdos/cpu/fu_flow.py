"""
FU_FLOW: Control Flow Unit

Handles: JMP, JSR, RTS, RTI, BRK, and all branch instructions

These operations manipulate the Program Counter based on flags.
Requires understanding conditional logic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class FlowConfig:
    """Configuration for the Flow functional unit."""
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1


class FU_FLOW(nn.Module):
    """
    Neural Control Flow Unit
    
    Operations:
        JMP: Unconditional jump
        JSR: Jump to subroutine (push return address)
        RTS: Return from subroutine
        RTI: Return from interrupt
        BRK: Software interrupt
        Bxx: Conditional branches (BEQ, BNE, BCC, BCS, etc.)
    
    Input: PC, Flags (P), Target address, Opcode
    Output: New PC, New SP (for JSR/RTS)
    """
    
    def __init__(self, config: FlowConfig = None):
        super().__init__()
        
        if config is None:
            config = FlowConfig()
        self.config = config
        
        # Input: PC(16) + Flags(8) + Target(16) + SP(8) + Opcode_embed(16) = 64
        self.opcode_emb = nn.Embedding(256, 16)
        
        # Network
        layers = []
        input_dim = 64
        for _ in range(config.num_layers):
            layers.extend([
                nn.Linear(input_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            ])
            input_dim = config.hidden_dim
        self.net = nn.Sequential(*layers)
        
        # Output: New_PC(16) + New_SP(8) + branch_taken(1)
        self.output_head = nn.Linear(config.hidden_dim, 25)
    
    def _encode_16bit(self, value: torch.Tensor) -> torch.Tensor:
        """Encode 16-bit value to binary tensor."""
        bits = []
        for i in range(16):
            bits.append(((value >> i) & 1).float())
        return torch.stack(bits, dim=-1)
    
    def _encode_8bit(self, value: torch.Tensor) -> torch.Tensor:
        """Encode 8-bit value to binary tensor."""
        bits = []
        for i in range(8):
            bits.append(((value >> i) & 1).float())
        return torch.stack(bits, dim=-1)
    
    def _decode_16bit(self, tensor: torch.Tensor) -> torch.Tensor:
        """Decode 16-bit binary tensor to integer."""
        active = (tensor > 0.5).long()
        result = torch.zeros(tensor.shape[0], dtype=torch.long, device=tensor.device)
        for i in range(16):
            result |= (active[:, i] << i)
        return result
    
    def _decode_8bit(self, tensor: torch.Tensor) -> torch.Tensor:
        """Decode 8-bit binary tensor to integer."""
        active = (tensor > 0.5).long()
        result = torch.zeros(tensor.shape[0], dtype=torch.long, device=tensor.device)
        for i in range(8):
            result |= (active[:, i] << i)
        return result
    
    def forward(
        self,
        pc: torch.Tensor,
        flags: torch.Tensor,
        target: torch.Tensor,
        sp: torch.Tensor,
        opcode: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            pc: Program counter [batch] (16-bit)
            flags: Processor status [batch] (8-bit)
            target: Branch/jump target [batch] (16-bit)
            sp: Stack pointer [batch] (8-bit)
            opcode: Operation code [batch]
        
        Returns:
            output_logits: [batch, 25] (PC:16 + SP:8 + branch_taken:1)
        """
        # Encode inputs
        pc_bin = self._encode_16bit(pc)
        flags_bin = self._encode_8bit(flags)
        target_bin = self._encode_16bit(target)
        sp_bin = self._encode_8bit(sp)
        op_emb = self.opcode_emb(opcode)
        
        # Concatenate
        inp = torch.cat([pc_bin, flags_bin, target_bin, sp_bin, op_emb], dim=-1)
        
        # Forward
        h = self.net(inp)
        return self.output_head(h)
    
    def predict(
        self,
        pc: torch.Tensor,
        flags: torch.Tensor,
        target: torch.Tensor,
        sp: torch.Tensor,
        opcode: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict new PC, SP, and whether branch was taken."""
        self.eval()
        with torch.no_grad():
            out = self.forward(pc, flags, target, sp, opcode)
            probs = torch.sigmoid(out)
            
            new_pc = self._decode_16bit(probs[:, :16])
            new_sp = self._decode_8bit(probs[:, 16:24])
            branch_taken = (probs[:, 24] > 0.5).long()
            
            return new_pc, new_sp, branch_taken
    
    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def flow_loss(
    output_logits: torch.Tensor,
    target_pc: torch.Tensor,
    target_sp: torch.Tensor,
    target_branch_taken: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute Flow unit loss."""
    device = output_logits.device
    
    # Encode targets
    pc_bits = []
    for i in range(16):
        pc_bits.append(((target_pc >> i) & 1).float())
    pc_target = torch.stack(pc_bits, dim=-1).to(device)
    
    sp_bits = []
    for i in range(8):
        sp_bits.append(((target_sp >> i) & 1).float())
    sp_target = torch.stack(sp_bits, dim=-1).to(device)
    
    target = torch.cat([
        pc_target,
        sp_target,
        target_branch_taken.float().unsqueeze(-1).to(device)
    ], dim=-1)
    
    loss = F.binary_cross_entropy_with_logits(output_logits, target)
    
    return loss, {'flow_loss': loss.item()}


# Ground truth
def execute_flow_op(opcode: int, pc: int, flags: int, target: int, sp: int) -> Tuple[int, int, bool]:
    """
    Execute a 6502 flow operation.
    
    Returns: (new_pc, new_sp, branch_taken)
    """
    new_pc = pc + 1  # Default: advance by 1 (simplified)
    new_sp = sp
    branch_taken = False
    
    # Extract flags
    c_flag = flags & 1
    z_flag = (flags >> 1) & 1
    n_flag = (flags >> 7) & 1
    v_flag = (flags >> 6) & 1
    
    # JMP absolute
    if opcode == 0x4C:
        new_pc = target
    
    # JMP indirect
    elif opcode == 0x6C:
        new_pc = target  # Target should be the dereferenced address
    
    # JSR
    elif opcode == 0x20:
        new_sp = (sp - 2) & 0xFF  # Push 2-byte return address
        new_pc = target
    
    # RTS
    elif opcode == 0x60:
        new_sp = (sp + 2) & 0xFF  # Pop return address
        new_pc = target  # Target should be the popped address + 1
    
    # RTI
    elif opcode == 0x40:
        new_sp = (sp + 3) & 0xFF  # Pop flags + PC
        new_pc = target
    
    # BRK
    elif opcode == 0x00:
        new_sp = (sp - 3) & 0xFF  # Push PC + flags
        new_pc = target  # IRQ vector
    
    # Branches
    elif opcode == 0x10:  # BPL
        if n_flag == 0:
            new_pc = target
            branch_taken = True
    elif opcode == 0x30:  # BMI
        if n_flag == 1:
            new_pc = target
            branch_taken = True
    elif opcode == 0x50:  # BVC
        if v_flag == 0:
            new_pc = target
            branch_taken = True
    elif opcode == 0x70:  # BVS
        if v_flag == 1:
            new_pc = target
            branch_taken = True
    elif opcode == 0x90:  # BCC
        if c_flag == 0:
            new_pc = target
            branch_taken = True
    elif opcode == 0xB0:  # BCS
        if c_flag == 1:
            new_pc = target
            branch_taken = True
    elif opcode == 0xD0:  # BNE
        if z_flag == 0:
            new_pc = target
            branch_taken = True
    elif opcode == 0xF0:  # BEQ
        if z_flag == 1:
            new_pc = target
            branch_taken = True
    
    return new_pc, new_sp, branch_taken


if __name__ == "__main__":
    config = FlowConfig()
    flow = FU_FLOW(config)
    print(f"FU_FLOW Parameters: {flow.num_parameters:,}")
