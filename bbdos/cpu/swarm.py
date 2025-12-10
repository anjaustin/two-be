"""
Swarm 6502: The Neural Chipset

A disaggregated neural architecture where specialized Functional Units (FUs)
handle different classes of operations. The opcode IS the router - zero-cost,
deterministic, perfect routing.

Architecture:
    FU_ALU   (Soroban) - Arithmetic: ADC, SBC, CMP, INC, DEC
    FU_LOGIC (Binary)  - Bitwise: AND, ORA, EOR, ASL, LSR, ROL, ROR
    FU_MOVE  (Binary)  - Datapath: LDA, STA, TAX, TXA, etc.
    FU_FLOW  (PC+Flags)- Control: JMP, JSR, RTS, Bxx
    FU_STACK (Binary)  - Stack: PHA, PLA, PHP, PLP

Key insight: By routing opcodes to specialized units with optimal encodings,
we eliminate catastrophic interference and achieve gradient alignment.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from .fu_map import build_fu_map, FU_ALU, FU_LOGIC, FU_MOVE, FU_FLOW, FU_STACK, FU_NAMES
from .fu_alu import FU_ALU as ALUModule, ALUConfig
from .fu_logic import FU_LOGIC as LogicModule, LogicConfig
from .fu_move import FU_MOVE as MoveModule, MoveConfig
from .fu_flow import FU_FLOW as FlowModule, FlowConfig
from .fu_stack import FU_STACK as StackModule, StackConfig


@dataclass
class SwarmConfig:
    """Configuration for the Swarm 6502."""
    alu_config: ALUConfig = None
    logic_config: LogicConfig = None
    move_config: MoveConfig = None
    flow_config: FlowConfig = None
    stack_config: StackConfig = None
    
    def __post_init__(self):
        if self.alu_config is None:
            self.alu_config = ALUConfig()
        if self.logic_config is None:
            self.logic_config = LogicConfig()
        if self.move_config is None:
            self.move_config = MoveConfig()
        if self.flow_config is None:
            self.flow_config = FlowConfig()
        if self.stack_config is None:
            self.stack_config = StackConfig()


class Swarm6502(nn.Module):
    """
    The Neural Chipset: A swarm of specialized functional units.
    
    Instead of one monolithic model trying to learn everything,
    we have specialized experts for each class of operation:
    
    - ALU uses Soroban encoding (carry as column overflow)
    - LOGIC uses binary encoding (pure bitwise operations)
    - MOVE uses minimal networks (mostly identity mappings)
    - FLOW handles branching and jumps
    - STACK handles push/pull operations
    
    The router is deterministic (opcode-based), so there's no
    learned gating overhead. This is "Mixture of Experts" with
    a perfect, zero-cost router.
    """
    
    def __init__(self, config: SwarmConfig = None):
        super().__init__()
        
        if config is None:
            config = SwarmConfig()
        self.config = config
        
        # Build the opcode → FU routing table
        self.register_buffer('fu_map', build_fu_map())
        
        # Initialize functional units
        self.alu = ALUModule(config.alu_config)
        self.logic = LogicModule(config.logic_config)
        self.move = MoveModule(config.move_config)
        self.flow = FlowModule(config.flow_config)
        self.stack = StackModule(config.stack_config)
        
        # FU lookup
        self.fu_modules = {
            FU_ALU: self.alu,
            FU_LOGIC: self.logic,
            FU_MOVE: self.move,
            FU_FLOW: self.flow,
            FU_STACK: self.stack,
        }
    
    def get_fu_for_opcode(self, opcode: int) -> Tuple[int, nn.Module]:
        """Get the functional unit for an opcode."""
        fu_id = self.fu_map[opcode].item()
        return fu_id, self.fu_modules[fu_id]
    
    def route_batch(self, opcodes: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Route a batch of opcodes to their functional units.
        
        Returns:
            Dictionary mapping FU_ID to indices of samples that go to that FU
        """
        fu_ids = self.fu_map[opcodes]
        
        routing = {}
        for fu_id in range(5):
            mask = (fu_ids == fu_id)
            indices = torch.where(mask)[0]
            if len(indices) > 0:
                routing[fu_id] = indices
        
        return routing
    
    def forward_alu(
        self,
        state: Dict[str, torch.Tensor],
        indices: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through ALU for selected samples."""
        a = state['A'][indices]
        operand = state['Val'][indices]
        carry = ((state['P'][indices] >> 0) & 1).long()  # C flag is bit 0
        opcode = state['Op'][indices]
        
        result_logits, flags_logits = self.alu(a, operand, carry, opcode)
        
        return {
            'result_logits': result_logits,
            'flags_logits': flags_logits,
            'indices': indices,
        }
    
    def forward_logic(
        self,
        state: Dict[str, torch.Tensor],
        indices: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through Logic unit for selected samples."""
        a = state['A'][indices]
        operand = state['Val'][indices]
        carry = ((state['P'][indices] >> 0) & 1).long()
        opcode = state['Op'][indices]
        
        result_logits, flags_logits = self.logic(a, operand, carry, opcode)
        
        return {
            'result_logits': result_logits,
            'flags_logits': flags_logits,
            'indices': indices,
        }
    
    def forward_move(
        self,
        state: Dict[str, torch.Tensor],
        indices: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through Move unit for selected samples."""
        a = state['A'][indices]
        x = state['X'][indices]
        y = state['Y'][indices]
        operand = state['Val'][indices]
        opcode = state['Op'][indices]
        
        output_logits = self.move(a, x, y, operand, opcode)
        
        return {
            'output_logits': output_logits,
            'indices': indices,
        }
    
    def forward_flow(
        self,
        state: Dict[str, torch.Tensor],
        indices: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through Flow unit for selected samples."""
        pc = (state['PCH'][indices].long() << 8) | state['PCL'][indices].long()
        flags = state['P'][indices]
        target = state['Val'][indices].long()  # Simplified: target in Val
        sp = state['SP'][indices]
        opcode = state['Op'][indices]
        
        output_logits = self.flow(pc, flags, target, sp, opcode)
        
        return {
            'output_logits': output_logits,
            'indices': indices,
        }
    
    def forward_stack(
        self,
        state: Dict[str, torch.Tensor],
        indices: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through Stack unit for selected samples."""
        a = state['A'][indices]
        x = state['X'][indices]
        sp = state['SP'][indices]
        flags = state['P'][indices]
        stack_top = state['Val'][indices]  # Simplified: stack top in Val
        opcode = state['Op'][indices]
        
        output_logits = self.stack(a, x, sp, flags, stack_top, opcode)
        
        return {
            'output_logits': output_logits,
            'indices': indices,
        }
    
    def forward(
        self,
        state: Dict[str, torch.Tensor]
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Full forward pass with automatic routing.
        
        Args:
            state: CPU state dictionary with keys:
                A, X, Y, SP, P, PCH, PCL, Op, Val
        
        Returns:
            Dictionary mapping FU_ID to that FU's output dictionary
        """
        # Route batch to FUs
        routing = self.route_batch(state['Op'])
        
        results = {}
        
        for fu_id, indices in routing.items():
            if fu_id == FU_ALU:
                results[fu_id] = self.forward_alu(state, indices)
            elif fu_id == FU_LOGIC:
                results[fu_id] = self.forward_logic(state, indices)
            elif fu_id == FU_MOVE:
                results[fu_id] = self.forward_move(state, indices)
            elif fu_id == FU_FLOW:
                results[fu_id] = self.forward_flow(state, indices)
            elif fu_id == FU_STACK:
                results[fu_id] = self.forward_stack(state, indices)
        
        return results
    
    @property
    def num_parameters(self) -> int:
        """Total parameters across all FUs."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_fu_param_counts(self) -> Dict[str, int]:
        """Parameter count per FU."""
        return {
            'ALU': self.alu.num_parameters,
            'LOGIC': self.logic.num_parameters,
            'MOVE': self.move.num_parameters,
            'FLOW': self.flow.num_parameters,
            'STACK': self.stack.num_parameters,
        }


def print_swarm_info(swarm: Swarm6502):
    """Print information about the Swarm architecture."""
    print("=" * 60)
    print("SWARM 6502: Neural Chipset")
    print("=" * 60)
    
    print("\nFunctional Units:")
    fu_params = swarm.get_fu_param_counts()
    total = sum(fu_params.values())
    
    for name, params in fu_params.items():
        pct = params / total * 100
        print(f"  {name:8s}: {params:,} params ({pct:.1f}%)")
    
    print(f"\n  TOTAL: {total:,} params")
    
    print("\nOpcode Routing:")
    from .fu_map import get_fu_stats
    stats = get_fu_stats()
    for name, count in stats.items():
        print(f"  {name:8s}: {count} opcodes")
    
    print("=" * 60)


if __name__ == "__main__":
    # Test the Swarm
    config = SwarmConfig()
    swarm = Swarm6502(config)
    
    print_swarm_info(swarm)
    
    # Test forward pass with mixed opcodes
    batch_size = 8
    state = {
        'A': torch.randint(0, 256, (batch_size,)),
        'X': torch.randint(0, 256, (batch_size,)),
        'Y': torch.randint(0, 256, (batch_size,)),
        'SP': torch.randint(0, 256, (batch_size,)),
        'P': torch.randint(0, 256, (batch_size,)),
        'PCH': torch.randint(0, 256, (batch_size,)),
        'PCL': torch.randint(0, 256, (batch_size,)),
        'Op': torch.tensor([0x69, 0x0A, 0xA9, 0xD0, 0x48, 0x29, 0x69, 0xEA]),  # Mixed opcodes
        'Val': torch.randint(0, 256, (batch_size,)),
    }
    
    print("\nTest forward pass with mixed opcodes:")
    print(f"Opcodes: {[hex(op) for op in state['Op'].tolist()]}")
    
    results = swarm.forward(state)
    
    for fu_id, output in results.items():
        name = FU_NAMES[fu_id]
        count = len(output['indices'])
        print(f"  {name}: {count} samples")
