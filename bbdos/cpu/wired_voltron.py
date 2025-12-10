"""
WIRED VOLTRON - The Nervous System

One forward() = One CPU cycle.
No Python in the loop. Pure tensor flow.

Architecture:
- Organs: Frozen specialists (100% accuracy)
- Nerves: Trainable transcoders (Binary <-> Soroban)
- Spine: Hardcoded router (LUT)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ============================================================
# SOROBAN ENCODING (Reference Implementation)
# ============================================================

def soroban_encode(x):
    """
    Encode byte tensor to Soroban thermometer.
    Input: (batch,) of uint8 values
    Output: (batch, 32) thermometer encoding
    """
    x = x.long()
    low = x & 0x0F
    high = (x >> 4) & 0x0F
    
    batch_size = x.shape[0]
    device = x.device
    
    low_therm = torch.zeros(batch_size, 16, device=device)
    high_therm = torch.zeros(batch_size, 16, device=device)
    
    for i in range(16):
        low_therm[:, i] = (low > i).float()
        high_therm[:, i] = (high > i).float()
    
    return torch.cat([low_therm, high_therm], dim=1)


def soroban_decode(encoded):
    """
    Decode Soroban thermometer to byte values.
    Input: (batch, 32) thermometer (thresholded)
    Output: (batch,) of uint8 values
    """
    low = encoded[:, :16].sum(dim=1)
    high = encoded[:, 16:].sum(dim=1)
    return (high * 16 + low).clamp(0, 255).long()


# ============================================================
# THE NERVES (Trainable Transcoders)
# ============================================================

class NerveEncoder(nn.Module):
    """
    Binary -> Soroban Nerve.
    Learns to convert 8-bit binary to 32-bit thermometer.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Sigmoid()  # Thermometer bits are 0-1
        )
    
    def forward(self, x):
        # x: (batch,) normalized to [0, 1]
        return self.net(x.unsqueeze(-1))


class NerveDecoder(nn.Module):
    """
    Soroban -> Binary Nerve.
    Learns to count beads and produce byte value.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output normalized to [0, 1]
        )
    
    def forward(self, x):
        # x: (batch, 32) thermometer
        return self.net(x).squeeze(-1)  # (batch,)


# ============================================================
# THE ORGANS (Frozen Specialists)
# ============================================================

class OrganALU(nn.Module):
    """
    The ALU Organ - Soroban-powered arithmetic.
    Input: (A_soroban, operand_soroban, C_in) -> 65 features
    Output: (result_soroban, C_out, V_out) -> 34 features
    """
    def __init__(self):
        super().__init__()
        # Result organelle
        self.result_net = nn.Sequential(
            nn.Linear(65, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.Sigmoid()
        )
        # Carry organelle
        self.carry_net = nn.Sequential(
            nn.Linear(65, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
        # Overflow organelle
        self.overflow_net = nn.Sequential(
            nn.Linear(65, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    
    def forward(self, A_sor, operand_sor, C_in):
        """
        A_sor: (batch, 32) Soroban-encoded accumulator
        operand_sor: (batch, 32) Soroban-encoded operand
        C_in: (batch, 1) carry input
        """
        x = torch.cat([A_sor, operand_sor, C_in], dim=1)  # (batch, 65)
        
        result_sor = self.result_net(x)  # (batch, 32)
        C_out = self.carry_net(x)        # (batch, 1)
        V_out = self.overflow_net(x)     # (batch, 1)
        
        return result_sor, C_out, V_out


# ============================================================
# THE SPINE (Router)
# ============================================================

class Router:
    """
    Hardcoded opcode router (LUT).
    Maps opcode -> specialist ID.
    """
    # Specialist IDs
    ALU = 0
    LOGIC = 1
    TRANSFER = 2
    STACK = 3
    FLAGS = 4
    BRANCH = 5
    LOAD = 6
    STORE = 7
    NOP = 8
    
    @staticmethod
    def build_routing_table():
        """Build the 256-entry routing table."""
        table = torch.zeros(256, dtype=torch.long)
        
        # ADC - all addressing modes
        for op in [0x69, 0x65, 0x75, 0x6D, 0x7D, 0x79, 0x61, 0x71]:
            table[op] = Router.ALU
        
        # SBC - all addressing modes  
        for op in [0xE9, 0xE5, 0xF5, 0xED, 0xFD, 0xF9, 0xE1, 0xF1]:
            table[op] = Router.ALU
        
        # Shifts and rotates
        for op in [0x0A, 0x06, 0x16, 0x0E, 0x1E,  # ASL
                   0x4A, 0x46, 0x56, 0x4E, 0x5E,  # LSR
                   0x2A, 0x26, 0x36, 0x2E, 0x3E,  # ROL
                   0x6A, 0x66, 0x76, 0x6E, 0x7E]: # ROR
            table[op] = Router.LOGIC
        
        # Logic operations
        for op in [0x29, 0x25, 0x35, 0x2D, 0x3D, 0x39, 0x21, 0x31,  # AND
                   0x09, 0x05, 0x15, 0x0D, 0x1D, 0x19, 0x01, 0x11,  # ORA
                   0x49, 0x45, 0x55, 0x4D, 0x5D, 0x59, 0x41, 0x51]: # EOR
            table[op] = Router.LOGIC
        
        # Transfers
        for op in [0xAA, 0x8A, 0xA8, 0x98, 0xBA, 0x9A]:  # TAX, TXA, TAY, TYA, TSX, TXS
            table[op] = Router.TRANSFER
        
        # Stack
        for op in [0x48, 0x68, 0x08, 0x28]:  # PHA, PLA, PHP, PLP
            table[op] = Router.STACK
        
        # Flag operations
        for op in [0x18, 0x38, 0x58, 0x78, 0xB8, 0xD8, 0xF8]:  # CLC, SEC, CLI, SEI, CLV, CLD, SED
            table[op] = Router.FLAGS
        
        # Inc/Dec registers
        for op in [0xE8, 0xCA, 0xC8, 0x88]:  # INX, DEX, INY, DEY
            table[op] = Router.ALU  # Simple arithmetic
        
        # Branches
        for op in [0x10, 0x30, 0x50, 0x70, 0x90, 0xB0, 0xD0, 0xF0]:
            table[op] = Router.BRANCH
        
        # Loads
        for op in [0xA9, 0xA5, 0xB5, 0xAD, 0xBD, 0xB9, 0xA1, 0xB1,  # LDA
                   0xA2, 0xA6, 0xB6, 0xAE, 0xBE,                    # LDX
                   0xA0, 0xA4, 0xB4, 0xAC, 0xBC]:                   # LDY
            table[op] = Router.LOAD
        
        # Stores
        for op in [0x85, 0x95, 0x8D, 0x9D, 0x99, 0x81, 0x91,  # STA
                   0x86, 0x96, 0x8E,                           # STX
                   0x84, 0x94, 0x8C]:                          # STY
            table[op] = Router.STORE
        
        # NOP
        table[0xEA] = Router.NOP
        
        return table


# ============================================================
# WIRED VOLTRON - The Complete Neural CPU
# ============================================================

class WiredVoltron(nn.Module):
    """
    One forward() = One CPU cycle.
    No Python orchestration. Pure tensor flow.
    """
    
    def __init__(self):
        super().__init__()
        
        # --- THE ORGANS (will be loaded frozen) ---
        self.alu = OrganALU()
        
        # --- THE NERVES (trainable) ---
        self.nerve_encode_A = NerveEncoder()
        self.nerve_encode_op = NerveEncoder()
        self.nerve_decode_result = NerveDecoder()
        
        # --- THE SPINE (hardcoded) ---
        self.register_buffer('routing_table', Router.build_routing_table())
    
    def load_organs(self, folder):
        """Load pretrained organ weights and freeze them."""
        # Load ALU organelles
        alu_a_weights = torch.load(f"{folder}/organelle_a.pt", weights_only=True)
        alu_c_weights = torch.load(f"{folder}/organelle_c.pt", weights_only=True)
        alu_v_weights = torch.load(f"{folder}/organelle_v.pt", weights_only=True)
        
        # Map weights to our OrganALU structure
        self.alu.result_net.load_state_dict(alu_a_weights)
        self.alu.carry_net.load_state_dict(alu_c_weights)
        self.alu.overflow_net.load_state_dict(alu_v_weights)
        
        # Freeze all organ parameters
        for param in self.alu.parameters():
            param.requires_grad = False
        
        print("[+] ALU organs loaded and frozen")
    
    def forward(self, A, operand, C_in, opcode):
        """
        Execute one instruction.
        
        Args:
            A: (batch,) accumulator value [0-255]
            operand: (batch,) operand value [0-255]
            C_in: (batch,) carry flag [0 or 1]
            opcode: (batch,) opcode [0-255]
        
        Returns:
            A_new: (batch,) new accumulator
            C_new: (batch,) new carry flag
            V_new: (batch,) new overflow flag
        """
        batch_size = A.shape[0]
        
        # --- ROUTING ---
        unit_ids = self.routing_table[opcode.long()]
        is_alu = (unit_ids == Router.ALU).float().unsqueeze(1)
        
        # --- NERVE ENCODING ---
        # Convert binary [0-255] to normalized [0-1] for nerves
        A_norm = A.float() / 255.0
        op_norm = operand.float() / 255.0
        
        # Encode to Soroban via learned nerves
        A_sor = self.nerve_encode_A(A_norm)       # (batch, 32)
        op_sor = self.nerve_encode_op(op_norm)   # (batch, 32)
        C_in_t = C_in.float().unsqueeze(1)       # (batch, 1)
        
        # --- ORGAN EXECUTION ---
        result_sor, C_out, V_out = self.alu(A_sor, op_sor, C_in_t)
        
        # --- NERVE DECODING ---
        # Threshold Soroban output and decode
        result_sor_hard = (result_sor > 0.5).float()
        A_new_norm = self.nerve_decode_result(result_sor_hard)
        A_new = (A_new_norm * 255.0).clamp(0, 255)
        
        # Flags (direct from organs)
        C_new = (C_out > 0.5).float().squeeze(1)
        V_new = (V_out > 0.5).float().squeeze(1)
        
        # --- MASKING (only ALU ops produce output) ---
        # For non-ALU ops, preserve input (passthrough)
        A_final = A_new * is_alu.squeeze(1) + A.float() * (1 - is_alu.squeeze(1))
        
        return A_final, C_new, V_new


# ============================================================
# DIRECT WIRED VOLTRON (No Learned Nerves - Reference)
# ============================================================

class DirectWiredVoltron(nn.Module):
    """
    Wired Voltron with hardcoded Soroban encoding (no learned nerves).
    Used as reference and for comparison.
    """
    
    def __init__(self):
        super().__init__()
        self.alu = OrganALU()
        self.register_buffer('routing_table', Router.build_routing_table())
    
    def load_organs(self, folder):
        alu_a_weights = torch.load(f"{folder}/organelle_a.pt", weights_only=True)
        alu_c_weights = torch.load(f"{folder}/organelle_c.pt", weights_only=True)
        alu_v_weights = torch.load(f"{folder}/organelle_v.pt", weights_only=True)
        
        self.alu.result_net.load_state_dict(alu_a_weights)
        self.alu.carry_net.load_state_dict(alu_c_weights)
        self.alu.overflow_net.load_state_dict(alu_v_weights)
        
        for param in self.alu.parameters():
            param.requires_grad = False
    
    def forward(self, A, operand, C_in, opcode):
        batch_size = A.shape[0]
        
        # Hardcoded Soroban encoding
        A_sor = soroban_encode(A)
        op_sor = soroban_encode(operand)
        C_in_t = C_in.float().unsqueeze(1)
        
        # Execute ALU
        result_sor, C_out, V_out = self.alu(A_sor, op_sor, C_in_t)
        
        # Hardcoded Soroban decoding
        A_new = soroban_decode((result_sor > 0.5).float())
        C_new = (C_out > 0.5).float().squeeze(1)
        V_new = (V_out > 0.5).float().squeeze(1)
        
        return A_new.float(), C_new, V_new
