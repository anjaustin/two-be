"""
Functional Unit Opcode Map

Deterministic routing of 6502 opcodes to specialized neural functional units.
This is the "compiler" of the Neural Chipset - zero-cost, perfect routing.

FU Index:
    0 = ALU     (Soroban-encoded, handles arithmetic)
    1 = LOGIC   (Binary-encoded, handles bitwise ops)
    2 = MOVE    (Binary/passthrough, handles transfers) - DEFAULT
    3 = FLOW    (PC+Flags, handles branches/jumps)
    4 = STACK   (SP+Value, handles stack ops)
"""

import torch
from typing import Dict, Set, Tuple

# Functional Unit IDs
FU_ALU = 0
FU_LOGIC = 1
FU_MOVE = 2  # Default fallback
FU_FLOW = 3
FU_STACK = 4

FU_NAMES = {
    FU_ALU: "ALU",
    FU_LOGIC: "LOGIC",
    FU_MOVE: "MOVE",
    FU_FLOW: "FLOW",
    FU_STACK: "STACK",
}


def build_fu_map() -> torch.Tensor:
    """
    Build the opcode → FU mapping tensor.
    
    Returns:
        [256] tensor mapping each opcode to its FU index
    """
    # Default to MOVE (safe fallback for undefined/NOP opcodes)
    mapping = torch.full((256,), FU_MOVE, dtype=torch.long)
    
    # === FU_ALU: Arithmetic Operations (Soroban Encoded) ===
    # These need spatial representation for carry handling
    
    alu_ops = []
    
    # ADC - Add with Carry (all addressing modes)
    alu_ops += [0x69, 0x65, 0x75, 0x6D, 0x7D, 0x79, 0x61, 0x71]
    
    # SBC - Subtract with Carry (all addressing modes)
    alu_ops += [0xE9, 0xE5, 0xF5, 0xED, 0xFD, 0xF9, 0xE1, 0xF1]
    
    # CMP - Compare Accumulator (all addressing modes)
    alu_ops += [0xC9, 0xC5, 0xD5, 0xCD, 0xDD, 0xD9, 0xC1, 0xD1]
    
    # CPX - Compare X Register
    alu_ops += [0xE0, 0xE4, 0xEC]
    
    # CPY - Compare Y Register
    alu_ops += [0xC0, 0xC4, 0xCC]
    
    # INC - Increment Memory (keeping in ALU for Soroban consistency)
    alu_ops += [0xE6, 0xF6, 0xEE, 0xFE]
    
    # DEC - Decrement Memory
    alu_ops += [0xC6, 0xD6, 0xCE, 0xDE]
    
    # INX, DEX, INY, DEY - Register increment/decrement
    alu_ops += [0xE8, 0xCA, 0xC8, 0x88]
    
    for op in alu_ops:
        mapping[op] = FU_ALU
    
    # === FU_LOGIC: Bitwise Operations (Binary Encoded) ===
    # Pure logic gates, no carry propagation needed
    
    logic_ops = []
    
    # AND - Logical AND (all addressing modes)
    logic_ops += [0x29, 0x25, 0x35, 0x2D, 0x3D, 0x39, 0x21, 0x31]
    
    # ORA - Logical OR (all addressing modes)
    logic_ops += [0x09, 0x05, 0x15, 0x0D, 0x1D, 0x19, 0x01, 0x11]
    
    # EOR - Exclusive OR (all addressing modes)
    logic_ops += [0x49, 0x45, 0x55, 0x4D, 0x5D, 0x59, 0x41, 0x51]
    
    # BIT - Bit Test
    logic_ops += [0x24, 0x2C]
    
    # ASL - Arithmetic Shift Left
    logic_ops += [0x0A, 0x06, 0x16, 0x0E, 0x1E]
    
    # LSR - Logical Shift Right
    logic_ops += [0x4A, 0x46, 0x56, 0x4E, 0x5E]
    
    # ROL - Rotate Left
    logic_ops += [0x2A, 0x26, 0x36, 0x2E, 0x3E]
    
    # ROR - Rotate Right
    logic_ops += [0x6A, 0x66, 0x76, 0x6E, 0x7E]
    
    # Flag Operations (single-bit flips in P register)
    logic_ops += [
        0x18,  # CLC - Clear Carry
        0x38,  # SEC - Set Carry
        0x58,  # CLI - Clear Interrupt
        0x78,  # SEI - Set Interrupt
        0xB8,  # CLV - Clear Overflow
        0xD8,  # CLD - Clear Decimal
        0xF8,  # SED - Set Decimal
    ]
    
    for op in logic_ops:
        mapping[op] = FU_LOGIC
    
    # === FU_FLOW: Branch and Control Operations ===
    # PC manipulation and conditional branching
    
    flow_ops = []
    
    # JMP - Jump (absolute and indirect)
    flow_ops += [0x4C, 0x6C]
    
    # JSR - Jump to Subroutine
    flow_ops += [0x20]
    
    # RTS - Return from Subroutine
    flow_ops += [0x60]
    
    # RTI - Return from Interrupt
    flow_ops += [0x40]
    
    # BRK - Break (software interrupt)
    flow_ops += [0x00]
    
    # Conditional Branches
    flow_ops += [
        0x10,  # BPL - Branch if Plus
        0x30,  # BMI - Branch if Minus
        0x50,  # BVC - Branch if Overflow Clear
        0x70,  # BVS - Branch if Overflow Set
        0x90,  # BCC - Branch if Carry Clear
        0xB0,  # BCS - Branch if Carry Set
        0xD0,  # BNE - Branch if Not Equal
        0xF0,  # BEQ - Branch if Equal
    ]
    
    for op in flow_ops:
        mapping[op] = FU_FLOW
    
    # === FU_STACK: Stack Operations ===
    # SP manipulation and push/pull
    
    stack_ops = []
    
    # PHA/PLA - Push/Pull Accumulator
    stack_ops += [0x48, 0x68]
    
    # PHP/PLP - Push/Pull Processor Status
    stack_ops += [0x08, 0x28]
    
    # TXS/TSX - Transfer X to/from Stack Pointer
    stack_ops += [0x9A, 0xBA]
    
    for op in stack_ops:
        mapping[op] = FU_STACK
    
    # === FU_MOVE: Data Transfer (Default) ===
    # LDA, LDX, LDY, STA, STX, STY, TAX, TXA, etc.
    # These are already default (FU_MOVE = 2)
    # Explicitly listing for documentation:
    
    move_ops = []
    
    # LDA - Load Accumulator
    move_ops += [0xA9, 0xA5, 0xB5, 0xAD, 0xBD, 0xB9, 0xA1, 0xB1]
    
    # LDX - Load X Register
    move_ops += [0xA2, 0xA6, 0xB6, 0xAE, 0xBE]
    
    # LDY - Load Y Register
    move_ops += [0xA0, 0xA4, 0xB4, 0xAC, 0xBC]
    
    # STA - Store Accumulator
    move_ops += [0x85, 0x95, 0x8D, 0x9D, 0x99, 0x81, 0x91]
    
    # STX - Store X Register
    move_ops += [0x86, 0x96, 0x8E]
    
    # STY - Store Y Register
    move_ops += [0x84, 0x94, 0x8C]
    
    # Register Transfers
    move_ops += [
        0xAA,  # TAX - Transfer A to X
        0x8A,  # TXA - Transfer X to A
        0xA8,  # TAY - Transfer A to Y
        0x98,  # TYA - Transfer Y to A
    ]
    
    # NOP - No Operation (explicit)
    move_ops += [0xEA]
    
    for op in move_ops:
        mapping[op] = FU_MOVE
    
    return mapping


def get_fu_opcodes() -> Dict[int, Set[int]]:
    """
    Get the set of opcodes for each FU.
    
    Returns:
        Dictionary mapping FU index to set of opcodes
    """
    mapping = build_fu_map()
    result = {fu: set() for fu in range(5)}
    
    for opcode in range(256):
        fu = mapping[opcode].item()
        result[fu].add(opcode)
    
    return result


def get_fu_stats() -> Dict[str, int]:
    """Get statistics about FU opcode distribution."""
    fu_opcodes = get_fu_opcodes()
    return {FU_NAMES[fu]: len(opcodes) for fu, opcodes in fu_opcodes.items()}


def print_fu_map():
    """Print the complete FU mapping for verification."""
    mapping = build_fu_map()
    stats = get_fu_stats()
    
    print("=" * 60)
    print("FUNCTIONAL UNIT OPCODE MAP")
    print("=" * 60)
    
    print("\nOpcode Distribution:")
    for name, count in stats.items():
        print(f"  {name}: {count} opcodes")
    
    print(f"\nTotal mapped: {sum(stats.values())} / 256")
    
    print("\n" + "-" * 60)
    print("Detailed Mapping:")
    print("-" * 60)
    
    for fu in range(5):
        opcodes = get_fu_opcodes()[fu]
        print(f"\n{FU_NAMES[fu]} ({len(opcodes)} opcodes):")
        
        # Print in rows of 16
        sorted_ops = sorted(opcodes)
        for i in range(0, len(sorted_ops), 16):
            chunk = sorted_ops[i:i+16]
            print("  " + " ".join(f"{op:02X}" for op in chunk))


# Opcode names for debugging
OPCODE_NAMES = {
    0x00: "BRK", 0x01: "ORA_IX", 0x05: "ORA_ZP", 0x06: "ASL_ZP",
    0x08: "PHP", 0x09: "ORA_IMM", 0x0A: "ASL_A", 0x0D: "ORA_ABS",
    0x0E: "ASL_ABS", 0x10: "BPL", 0x11: "ORA_IY", 0x15: "ORA_ZPX",
    0x16: "ASL_ZPX", 0x18: "CLC", 0x19: "ORA_ABY", 0x1D: "ORA_ABX",
    0x1E: "ASL_ABX", 0x20: "JSR", 0x21: "AND_IX", 0x24: "BIT_ZP",
    0x25: "AND_ZP", 0x26: "ROL_ZP", 0x28: "PLP", 0x29: "AND_IMM",
    0x2A: "ROL_A", 0x2C: "BIT_ABS", 0x2D: "AND_ABS", 0x2E: "ROL_ABS",
    0x30: "BMI", 0x31: "AND_IY", 0x35: "AND_ZPX", 0x36: "ROL_ZPX",
    0x38: "SEC", 0x39: "AND_ABY", 0x3D: "AND_ABX", 0x3E: "ROL_ABX",
    0x40: "RTI", 0x41: "EOR_IX", 0x45: "EOR_ZP", 0x46: "LSR_ZP",
    0x48: "PHA", 0x49: "EOR_IMM", 0x4A: "LSR_A", 0x4C: "JMP_ABS",
    0x4D: "EOR_ABS", 0x4E: "LSR_ABS", 0x50: "BVC", 0x51: "EOR_IY",
    0x55: "EOR_ZPX", 0x56: "LSR_ZPX", 0x58: "CLI", 0x59: "EOR_ABY",
    0x5D: "EOR_ABX", 0x5E: "LSR_ABX", 0x60: "RTS", 0x61: "ADC_IX",
    0x65: "ADC_ZP", 0x66: "ROR_ZP", 0x68: "PLA", 0x69: "ADC_IMM",
    0x6A: "ROR_A", 0x6C: "JMP_IND", 0x6D: "ADC_ABS", 0x6E: "ROR_ABS",
    0x70: "BVS", 0x71: "ADC_IY", 0x75: "ADC_ZPX", 0x76: "ROR_ZPX",
    0x78: "SEI", 0x79: "ADC_ABY", 0x7D: "ADC_ABX", 0x7E: "ROR_ABX",
    0x81: "STA_IX", 0x84: "STY_ZP", 0x85: "STA_ZP", 0x86: "STX_ZP",
    0x88: "DEY", 0x8A: "TXA", 0x8C: "STY_ABS", 0x8D: "STA_ABS",
    0x8E: "STX_ABS", 0x90: "BCC", 0x91: "STA_IY", 0x94: "STY_ZPX",
    0x95: "STA_ZPX", 0x96: "STX_ZPY", 0x98: "TYA", 0x99: "STA_ABY",
    0x9A: "TXS", 0x9D: "STA_ABX", 0xA0: "LDY_IMM", 0xA1: "LDA_IX",
    0xA2: "LDX_IMM", 0xA4: "LDY_ZP", 0xA5: "LDA_ZP", 0xA6: "LDX_ZP",
    0xA8: "TAY", 0xA9: "LDA_IMM", 0xAA: "TAX", 0xAC: "LDY_ABS",
    0xAD: "LDA_ABS", 0xAE: "LDX_ABS", 0xB0: "BCS", 0xB1: "LDA_IY",
    0xB4: "LDY_ZPX", 0xB5: "LDA_ZPX", 0xB6: "LDX_ZPY", 0xB8: "CLV",
    0xB9: "LDA_ABY", 0xBA: "TSX", 0xBC: "LDY_ABX", 0xBD: "LDA_ABX",
    0xBE: "LDX_ABY", 0xC0: "CPY_IMM", 0xC1: "CMP_IX", 0xC4: "CPY_ZP",
    0xC5: "CMP_ZP", 0xC6: "DEC_ZP", 0xC8: "INY", 0xC9: "CMP_IMM",
    0xCA: "DEX", 0xCC: "CPY_ABS", 0xCD: "CMP_ABS", 0xCE: "DEC_ABS",
    0xD0: "BNE", 0xD1: "CMP_IY", 0xD5: "CMP_ZPX", 0xD6: "DEC_ZPX",
    0xD8: "CLD", 0xD9: "CMP_ABY", 0xDD: "CMP_ABX", 0xDE: "DEC_ABX",
    0xE0: "CPX_IMM", 0xE1: "SBC_IX", 0xE4: "CPX_ZP", 0xE5: "SBC_ZP",
    0xE6: "INC_ZP", 0xE8: "INX", 0xE9: "SBC_IMM", 0xEA: "NOP",
    0xEC: "CPX_ABS", 0xED: "SBC_ABS", 0xEE: "INC_ABS", 0xF0: "BEQ",
    0xF1: "SBC_IY", 0xF5: "SBC_ZPX", 0xF6: "INC_ZPX", 0xF8: "SED",
    0xF9: "SBC_ABY", 0xFD: "SBC_ABX", 0xFE: "INC_ABX",
}


if __name__ == "__main__":
    print_fu_map()
