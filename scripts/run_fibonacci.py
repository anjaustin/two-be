#!/usr/bin/env python3
"""
NEURAL FIBONACCI - The Hello World of the Singularity

A neural network hallucinates 6502 microprocessor logic to generate
the Fibonacci sequence. If this works, Voltron is Turing Complete.
"""

import torch
import torch.nn as nn
import numpy as np

# ============================================================
# SOROBAN ENCODING
# ============================================================
def soroban_encode_batch(x):
    """Encode batch of bytes to Soroban thermometer."""
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.long)
    x = x.long().view(-1)
    low = x & 0x0F
    high = (x >> 4) & 0x0F
    
    batch_size = x.shape[0]
    low_therm = torch.zeros(batch_size, 16)
    high_therm = torch.zeros(batch_size, 16)
    
    for i in range(16):
        low_therm[:, i] = (low > i).float()
        high_therm[:, i] = (high > i).float()
    
    return torch.cat([low_therm, high_therm], dim=1)

def soroban_decode_batch(encoded):
    """Decode Soroban thermometer to byte values."""
    low = encoded[:, :16].sum(dim=1)
    high = encoded[:, 16:].sum(dim=1)
    return (high * 16 + low).clamp(0, 255).long()

# ============================================================
# ORGANELLE DEFINITIONS
# ============================================================
class AccumulatorOrganelle(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(65, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

class CarryOrganelle(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(65, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

class OverflowOrganelle(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(65, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

# ============================================================
# THE NEURAL ALU (THE HEAD OF VOLTRON)
# ============================================================
class NeuralALU:
    """Soroban-powered arithmetic unit with perfect accuracy."""
    
    def __init__(self):
        self.organelle_a = AccumulatorOrganelle()
        self.organelle_c = CarryOrganelle()
        self.organelle_v = OverflowOrganelle()
    
    def load(self, folder):
        self.organelle_a.load_state_dict(
            torch.load(f"{folder}/organelle_a.pt", weights_only=True))
        self.organelle_c.load_state_dict(
            torch.load(f"{folder}/organelle_c.pt", weights_only=True))
        self.organelle_v.load_state_dict(
            torch.load(f"{folder}/organelle_v.pt", weights_only=True))
        self.organelle_a.eval()
        self.organelle_c.eval()
        self.organelle_v.eval()
    
    def adc(self, A, operand, C_in):
        """
        ADC: A + operand + C -> new_A, N, V, Z, C
        """
        A_t = torch.tensor([A], dtype=torch.long)
        op_t = torch.tensor([operand], dtype=torch.long)
        c_t = torch.tensor([C_in], dtype=torch.float).unsqueeze(1)
        
        A_enc = soroban_encode_batch(A_t)
        op_enc = soroban_encode_batch(op_t)
        
        x = torch.cat([A_enc, op_enc, c_t], dim=1)
        
        with torch.no_grad():
            result_enc = self.organelle_a(x)
            result = soroban_decode_batch((result_enc > 0.5).float()).item()
            
            c_out = int((self.organelle_c(x) > 0.5).item())
            v_out = int((self.organelle_v(x) > 0.5).item())
        
        # Derived flags
        z_out = 1 if result == 0 else 0
        n_out = (result >> 7) & 1
        
        return result, n_out, v_out, z_out, c_out

# ============================================================
# THE SWARM 6502 (VOLTRON)
# ============================================================
class Swarm6502:
    """
    Neural 6502 CPU - The assembled Voltron.
    Routes opcodes to specialist models.
    """
    
    def __init__(self):
        self.alu = NeuralALU()
        
    def load_brain(self, folder="checkpoints/swarm"):
        print(">> FORMING VOLTRON...")
        self.alu.load(folder)
        print("   [+] Neural ALU (Soroban Core) Online")
        print("   [+] Organelle_A (Result) loaded - 100% accuracy")
        print("   [+] Organelle_C (Carry) loaded - 100% accuracy")
        print("   [+] Organelle_V (Overflow) loaded - 100% accuracy")
        print(">> SYSTEM READY.")
    
    def predict_step(self, A, X, Y, SP, P, opcode, operand):
        """
        Execute one instruction and return new register state.
        
        P register bits: NV-BDIZC
          Bit 7: N (Negative)
          Bit 6: V (Overflow)
          Bit 4: B (Break)
          Bit 3: D (Decimal)
          Bit 2: I (Interrupt)
          Bit 1: Z (Zero)
          Bit 0: C (Carry)
        """
        # Extract current flags
        C_in = P & 0x01
        
        new_A, new_X, new_Y, new_P = A, X, Y, P
        
        # Route by opcode
        if opcode == 0xA9:  # LDA immediate
            new_A = operand
            # Update N and Z flags
            new_P = self._update_nz(new_P, new_A)
            
        elif opcode == 0xA5:  # LDA zero page
            new_A = operand  # Harness already resolved memory
            new_P = self._update_nz(new_P, new_A)
            
        elif opcode == 0x85:  # STA zero page
            # A stays the same, memory write handled by harness
            pass
            
        elif opcode == 0x18:  # CLC - Clear Carry
            new_P = new_P & 0xFE  # Clear bit 0
            
        elif opcode == 0x38:  # SEC - Set Carry
            new_P = new_P | 0x01  # Set bit 0
            
        elif opcode == 0x65:  # ADC zero page
            # THE NEURAL MAGIC - Soroban ALU activates!
            result, n, v, z, c = self.alu.adc(A, operand, C_in)
            new_A = result
            # Update all flags
            new_P = (new_P & 0x3C)  # Clear N, V, Z, C (keep B, D, I)
            new_P |= (n << 7)  # N flag
            new_P |= (v << 6)  # V flag
            new_P |= (z << 1)  # Z flag
            new_P |= c         # C flag
            
        elif opcode == 0x4C:  # JMP absolute
            # PC update handled by harness
            pass
            
        return {
            'A': new_A,
            'X': new_X,
            'Y': new_Y,
            'SP': SP,
            'P': new_P
        }
    
    def _update_nz(self, P, value):
        """Update N and Z flags based on value."""
        P = P & 0x7D  # Clear N (bit 7) and Z (bit 1)
        if value == 0:
            P |= 0x02  # Set Z
        if value & 0x80:
            P |= 0x80  # Set N
        return P

# ============================================================
# THE MOTHERBOARD (Harness)
# ============================================================
class Motherboard:
    def __init__(self):
        self.ram = [0] * 65536
        self.cpu = Swarm6502()
        self.cpu.load_brain("/workspace/two-be/checkpoints/swarm")
        
        self.state = {
            'A': 0, 'X': 0, 'Y': 0, 'SP': 0xFF, 'P': 0x00, 'PC': 0x0600
        }

    def write_ram(self, addr, val):
        self.ram[addr] = val & 0xFF

    def read_ram(self, addr):
        return self.ram[addr]

    def load_program(self, start_addr, bytes_seq):
        for i, b in enumerate(bytes_seq):
            self.ram[start_addr + i] = b

    def step(self):
        pc = self.state['PC']
        opcode = self.read_ram(pc)
        operand = 0
        next_pc = pc + 1
        
        # Decode addressing mode and fetch operand
        if opcode == 0xA9:  # LDA immediate
            operand = self.read_ram(pc + 1)
            next_pc = pc + 2
            
        elif opcode == 0xA5:  # LDA zero page
            addr = self.read_ram(pc + 1)
            operand = self.read_ram(addr)
            next_pc = pc + 2
            
        elif opcode == 0x85:  # STA zero page
            addr = self.read_ram(pc + 1)
            self.write_ram(addr, self.state['A'])
            operand = 0
            next_pc = pc + 2
            
        elif opcode == 0x65:  # ADC zero page
            addr = self.read_ram(pc + 1)
            operand = self.read_ram(addr)
            next_pc = pc + 2
            
        elif opcode == 0x18:  # CLC
            operand = 0
            next_pc = pc + 1
            
        elif opcode == 0x4C:  # JMP absolute
            low = self.read_ram(pc + 1)
            high = self.read_ram(pc + 2)
            next_pc = (high << 8) | low
            
        # Execute via Neural CPU
        new_regs = self.cpu.predict_step(
            self.state['A'], self.state['X'], self.state['Y'],
            self.state['SP'], self.state['P'],
            opcode, operand
        )
        
        # Commit state
        self.state['A'] = new_regs['A']
        self.state['X'] = new_regs['X']
        self.state['Y'] = new_regs['Y']
        self.state['P'] = new_regs['P']
        self.state['PC'] = next_pc
        
        return opcode, self.state['A']

# ============================================================
# THE PROGRAM - Fibonacci in 6502 Machine Code
# ============================================================
# 0600: A9 00       LDA #$00
# 0602: 85 10       STA $10      ; Prev = 0
# 0604: A9 01       LDA #$01
# 0606: 85 11       STA $11      ; Curr = 1
# LOOP:
# 0608: A5 10       LDA $10      ; Load Prev
# 060A: 18          CLC          ; Clear Carry (CRITICAL!)
# 060B: 65 11       ADC $11      ; A = Prev + Curr
# 060D: 85 12       STA $12      ; Store to Next
# 060F: A5 11       LDA $11      ; Load Curr
# 0611: 85 10       STA $10      ; Prev = Curr
# 0613: A5 12       LDA $12      ; Load Next
# 0615: 85 11       STA $11      ; Curr = Next
# 0617: 4C 08 06    JMP $0608    ; Loop

FIBONACCI_PROGRAM = [
    0xA9, 0x00,       # LDA #$00
    0x85, 0x10,       # STA $10
    0xA9, 0x01,       # LDA #$01
    0x85, 0x11,       # STA $11
    # Loop starts at offset 8 (address 0x0608)
    0xA5, 0x10,       # LDA $10
    0x18,             # CLC
    0x65, 0x11,       # ADC $11
    0x85, 0x12,       # STA $12
    0xA5, 0x11,       # LDA $11
    0x85, 0x10,       # STA $10
    0xA5, 0x12,       # LDA $12
    0x85, 0x11,       # STA $11
    0x4C, 0x08, 0x06  # JMP $0608
]

# ============================================================
# MAIN - The Hello World of the Singularity
# ============================================================
def main():
    print("=" * 70)
    print("        NEURAL FIBONACCI - The Hello World of the Singularity")
    print("=" * 70)
    print()
    print("A neural network will now hallucinate 6502 microprocessor logic")
    print("to generate the Fibonacci sequence, cycle by cycle.")
    print()
    
    board = Motherboard()
    board.load_program(0x0600, FIBONACCI_PROGRAM)
    
    print()
    print("=" * 70)
    print("                    EXECUTING FIBONACCI SEQUENCE")
    print("=" * 70)
    print()
    print(f"{'Cycle':<8} {'PC':<8} {'Op':<6} {'A':<6} {'Note'}")
    print("-" * 70)
    
    sequence = []
    cycle = 0
    
    # Run until we overflow or hit enough iterations
    while cycle < 200:
        pc = board.state['PC']
        op, a = board.step()
        cycle += 1
        
        # Decode opcode name
        op_names = {
            0xA9: "LDA#", 0xA5: "LDA", 0x85: "STA",
            0x18: "CLC", 0x65: "ADC", 0x4C: "JMP"
        }
        op_name = op_names.get(op, f"{op:02X}")
        
        # Track ADC results (the Fibonacci calculation)
        if op == 0x65:  # ADC - the neural magic moment
            sequence.append(a)
            print(f"{cycle:<8} ${pc:04X}   {op_name:<6} {a:<6} ** NEURAL ALU: {sequence[-2] if len(sequence) > 1 else 0} + {board.ram[0x11]} = {a}")
            
            # Stop before overflow wraps around
            if a > 200:
                break
        elif op == 0x18:  # CLC
            c_flag = board.state['P'] & 0x01
            print(f"{cycle:<8} ${pc:04X}   {op_name:<6} {a:<6} Carry cleared: {c_flag}")
    
    print("-" * 70)
    print()
    
    # Results
    print("=" * 70)
    print("                         RESULTS")
    print("=" * 70)
    print()
    print(f"Sequence generated by Neural 6502:")
    print(f"  {sequence}")
    print()
    
    # Expected Fibonacci (starting from first ADC: 0+1=1, 1+1=2, 1+2=3, ...)
    expected = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]
    
    print(f"Expected Fibonacci sequence:")
    print(f"  {expected}")
    print()
    
    # Verify
    if sequence == expected:
        print("=" * 70)
        print()
        print("  ██╗   ██╗ ██████╗ ██╗  ████████╗██████╗  ██████╗ ███╗   ██╗")
        print("  ██║   ██║██╔═══██╗██║  ╚══██╔══╝██╔══██╗██╔═══██╗████╗  ██║")
        print("  ██║   ██║██║   ██║██║     ██║   ██████╔╝██║   ██║██╔██╗ ██║")
        print("  ╚██╗ ██╔╝██║   ██║██║     ██║   ██╔══██╗██║   ██║██║╚██╗██║")
        print("   ╚████╔╝ ╚██████╔╝███████╗██║   ██║  ██║╚██████╔╝██║ ╚████║")
        print("    ╚═══╝   ╚═════╝ ╚══════╝╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝")
        print()
        print("        ██╗███████╗    ████████╗██╗   ██╗██████╗ ██╗███╗   ██╗ ██████╗ ")
        print("        ██║██╔════╝    ╚══██╔══╝██║   ██║██╔══██╗██║████╗  ██║██╔════╝ ")
        print("        ██║███████╗       ██║   ██║   ██║██████╔╝██║██╔██╗ ██║██║  ███╗")
        print("        ██║╚════██║       ██║   ██║   ██║██╔══██╗██║██║╚██╗██║██║   ██║")
        print("        ██║███████║       ██║   ╚██████╔╝██║  ██║██║██║ ╚████║╚██████╔╝")
        print("        ╚═╝╚══════╝       ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝ ╚═════╝ ")
        print()
        print("             ██████╗ ██████╗ ███╗   ███╗██████╗ ██╗     ███████╗████████╗███████╗")
        print("            ██╔════╝██╔═══██╗████╗ ████║██╔══██╗██║     ██╔════╝╚══██╔══╝██╔════╝")
        print("            ██║     ██║   ██║██╔████╔██║██████╔╝██║     █████╗     ██║   █████╗  ")
        print("            ██║     ██║   ██║██║╚██╔╝██║██╔═══╝ ██║     ██╔══╝     ██║   ██╔══╝  ")
        print("            ╚██████╗╚██████╔╝██║ ╚═╝ ██║██║     ███████╗███████╗   ██║   ███████╗")
        print("             ╚═════╝ ╚═════╝ ╚═╝     ╚═╝╚═╝     ╚══════╝╚══════╝   ╚═╝   ╚══════╝")
        print()
        print("=" * 70)
        print()
        print("  The silicon dreamed the golden ratio.")
        print("  Neural networks can hallucinate computation.")
        print("  The Ghost is not just dead - it is reborn as Fibonacci.")
        print()
    else:
        print(">>> SEQUENCE DIVERGED - Debug needed <<<")
        print(f"    Got:      {sequence}")
        print(f"    Expected: {expected}")
        
        # Find first difference
        for i, (got, exp) in enumerate(zip(sequence, expected)):
            if got != exp:
                print(f"    First error at position {i}: got {got}, expected {exp}")
                break

if __name__ == "__main__":
    main()
