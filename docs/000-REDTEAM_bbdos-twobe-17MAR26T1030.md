# Red-Team Security Review: Neural APU

**Date:** Tue Mar 17 2026, 10:30 UTC  
**Reviewer:** opencode (automated analysis)

---

## Identified Issues

### High Priority

1. **Cache Poisoning via Opcode Injection**
   - **Location:** `apu_cache.py:203-223`
   - **Issue:** User can inject arbitrary opcode strings that get stored in cache
   - **Risk:** Could cause cache pollution or DoS
   - **Mitigation:** Add opcode validation/allowlist

2. **Unbounded Cache Growth**
   - **Location:** `apu_cache.py:150-165`
   - **Issue:** Eviction policy allows unlimited opcodes to be stored sequentially
   - **Risk:** Memory exhaustion if attacker floods with unique opcode IDs
   - **Mitigation:** Add cache size hard limit with immediate eviction

### Medium Priority

3. **Race Condition in Concurrent Access**
   - **Location:** `apu_cache.py:175-188`
   - **Issue:** `_access_order` list modification not atomic
   - **Risk:** Data corruption under heavy concurrency
   - **Note:** RLock helps but list ops aren't atomic

4. **No Input Validation on Operand Shapes**
   - **Location:** `apu_cache.py:256-272`
   - **Issue:** Fallback compute accepts any shape, may produce garbage
   - **Risk:** Silent data corruption
   - **Recommendation:** Add shape validation per opcode

5. **Information Leak via Cache Timing**
   - **Location:** `apu_cache.py:85-95`
   - **Issue:** Hit/miss patterns observable externally
   - **Risk:** Side-channel timing attack on neural network internals
   - **Note:** Low risk for this use case

### Low Priority

6. **No Bounds Checking on Tile Routing**
   - **Location:** `bitswitch_avx.cpp:145-180`
   - **Issue:** AVX batch function assumes valid indices
   - **Risk:** Crash if out-of-bounds access

7. **Dark State Reserved but Not Implemented**
   - **Location:** `bitswitch_avx.cpp:29-34`
   - **Issue:** `0b11` code is reserved but could be exploited
   - **Risk:** Confusing for future implementations

---

## Recommendations

| Issue | Priority | Fix |
|-------|----------|-----|
| Opcode injection | High | Add allowlist or input sanitization |
| Memory exhaustion | High | Hard cap on cache entries |
| Shape validation | Medium | Assert input shapes before compute |
| Race condition | Medium | Use thread-safe deque |
| Timing leak | Low | Consider constant-time cache |

---

## Test Coverage Assessment

- ✅ Cache eviction under pressure
- ✅ Concurrent access patterns
- ✅ Invalid opcode handling
- ⚠️ Shape validation - not tested
- ⚠️ Memory exhaustion - limited test

---

## Verdict

**Status:** ⚠️ Requires fixes before production use

The core cache mechanism works and tests pass. However, the high-priority issues (opcode injection, memory exhaustion) should be addressed before deployment in any untrusted environment.

**Current Security Posture:** 6/10
