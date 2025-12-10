# Swarm 6502 Build Log

*Operation Great Schism: Building the Neural Chipset*

---

## Build Status

| Component | Status | Notes |
|-----------|--------|-------|
| FU Map (Opcode Router) | ✅ Complete | 256/256 opcodes mapped |
| FU_ALU (Soroban) | ✅ Complete | 166K params, Soroban encoding |
| FU_LOGIC (Binary) | ✅ Complete | 82K params, Binary encoding |
| FU_MOVE (Datapath) | ✅ Complete | 13K params, Lightweight |
| FU_FLOW (Control) | ✅ Complete | 32K params, PC+Flags |
| FU_STACK (Stack) | ✅ Complete | 15K params, SP tracking |
| Swarm Router | ✅ Complete | 308K params total |
| Dataset Splitter | ⏳ Pending | |
| Training Script | ⏳ Pending | |
| Integration Test | ⏳ Pending | |

---

## Key Design Decisions

### 1. Encoding Strategy (Option B)
- Router passes **raw integer state**
- Each FU encodes internally (Soroban for ALU, Binary for others)
- Keeps router O(1), parallelizes encoding in FU forward pass

### 2. Opcode Assignments (Gemini's Refinements)
- INC/DEC → FU_ALU (keep all math in Soroban domain)
- BIT → FU_LOGIC (pure AND that sets flags)
- Flag ops (CLC, SEC, etc.) → FU_LOGIC (single-bit flips)
- NOP → FU_MOVE (identity, safe fallback)
- Default: FU_MOVE (prevents crashes on undefined opcodes)

### 3. FU Index Mapping
```
0 = ALU (Soroban)
1 = LOGIC (Binary)
2 = MOVE (Binary/Passthrough) - DEFAULT
3 = FLOW (PC+Flags)
4 = STACK (SP+Value)
```

---

## Files to Create

```
bbdos/cpu/
├── fu_map.py       # Opcode → FU mapping
├── fu_alu.py       # Soroban-encoded arithmetic
├── fu_logic.py     # Binary logic operations
├── fu_move.py      # Datapath transfers
├── fu_flow.py      # Branch/control
├── fu_stack.py     # Stack operations
└── swarm.py        # Main router + integration

scripts/
├── split_dataset.py    # Split 50M records by FU
├── train_fu.py         # Train individual FUs
└── eval_swarm.py       # Full integration test
```

---

## Success Criteria

| FU | Target Accuracy | Baseline (Monolithic) |
|----|-----------------|----------------------|
| ALU | >70% | 3% (ADC) |
| LOGIC | >95% | 97% |
| MOVE | >99% | ~95% |
| FLOW | >95% | 96-99% |
| STACK | >99% | 99.9% |
| **Overall** | **>90%** | **66.4%** |

---

## Build Log

### Entry 1: Starting Build
- Date: 2024-12-10
- Creating FU map with complete 6502 opcode coverage
- Following Gemini's tactical refinements

### Entry 2: Swarm Architecture Complete
- Date: 2024-12-10
- All 5 FUs implemented and tested
- Routing verified with mixed opcode batch
- Total params: 308K (vs 2.4M monolithic - 87% reduction!)
- Parameter distribution:
  - ALU (Soroban): 54% - the heavy lifter for arithmetic
  - LOGIC (Binary): 27% - bitwise operations
  - FLOW (PC+Flags): 10% - control flow
  - STACK: 5% - stack operations
  - MOVE: 4% - trivial data transfers

### Next Steps
1. Create dataset splitter (split training data by FU)
2. Train FU_ALU first (the critical test)
3. Train remaining FUs
4. Integration test: full cycle-accurate emulation

