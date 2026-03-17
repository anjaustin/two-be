# Red-Team Review: MTFP Implementation

**Date:** Tue Mar 17 2026  
**Reviewer:** opencode

---

## Critical Issues

### 1. Denial of Service via Extreme Exponents

**Location:** `mtfp.py:78-86`

```python
exp = 0
if value >= 1.0:
    while value >= 3.0 and exp < self.max_exponent:
        value /= 3.0
        exp += 1
else:
    while value < 1.0 and exp > self.min_exponent:
        value *= 3.0
        exp -= 1
```

**Issue:** For very small or very large values, the loop can execute thousands of times, causing CPU exhaustion.

**Fix:** Add iteration cap.

---

### 2. Integer Overflow in Exponent Computation

**Location:** `mtfp.py:44-48`

```python
self.max_exponent = (3**n_exponent_trits) - 1
self.min_exponent = -(3**n_exponent_trits) + 1
self.mantissa_range = 3**n_mantissa_trits
```

**Issue:** For large trits (e.g., n_exponent_trits=20), 3**20 overflows Python int. For n_mantissa_trits=10+, 3**10 = 59049, which is fine, but scaling could cause issues.

**Fix:** Add bounds validation in `__init__`.

---

### 3. Division by Zero

**Location:** `mtfp.py:133`

```python
value = mantissa / (3 ** (self.n_mantissa_trits - 1))
```

**Issue:** If `n_mantissa_trits=0`, this divides by zero.

**Fix:** Validate `n_mantissa_trits >= 1`.

---

### 4. Null Byte Injection in Packed Storage

**Location:** `mtfp.py:231-251`

```python
def trits_to_packed_bytes(trits: np.ndarray) -> np.ndarray:
    ...
    if t == 1:
        code = 0b01
    elif t == -1:
        code = 0b10
    elif t == 2:  # Dark State
        code = 0b11
    else:
        code = 0b00
```

**Issue:** Trit value `2` (Dark State) is accepted without validation. An attacker could inject Dark State values to bypass operations.

**Fix:** Validate trit ranges before packing.

---

### 5. Shape Mismatch in Array Operations

**Location:** `mtfp.py:208-221`

```python
def mtfp_add(a: np.ndarray, b: np.ndarray, mtfp: MTFP = MTFP_16) -> np.ndarray:
    a_float = mtfp.unpack_array(a)
    b_float = mtfp.unpack_array(b)
    result_float = a_float + b_float
    return mtfp.pack_array(result_float)
```

**Issue:** No shape validation. If `a` and `b` have different shapes, `a_float + b_float` broadcasts incorrectly, leading to silent data corruption.

**Fix:** Add shape validation.

---

### 6. Negative Indices in Unpack

**Location:** `mtfp.py:128`

```python
exp += int(trits[idx]) * (3**i)
```

**Issue:** If `trits[idx]` returns negative value (from corrupted data), the computation could produce unexpected results.

**Fix:** Validate trit values are in {-1, 0, 1, 2} before computation.

---

### 7. Missing Input Validation in APU Integration

**Location:** `apu_cache.py`

**Issue:** MTFP opcodes don't validate input shapes before calling pack/unpack.

**Fix:** Add MTFP-specific shape validation to APU opcodes.

---

## Medium Issues

### 8. Precision Loss Not Communicated

**Issue:** MTFP has significant precision loss vs float32, but this isn't documented. Users might use MTFP for financial calculations.

**Fix:** Add warning in docstring.

### 9. No Range Checking on Pack

**Issue:** Values outside representable range silently become inf or nan.

**Fix:** Add `ValueError` for out-of-range values.

### 10. Thread Safety

**Issue:** MTFP operations aren't thread-safe. Multiple threads calling pack/unpack simultaneously could corrupt state.

**Fix:** Add locks or use thread-local storage.

---

## Summary

| Issue | Severity | Status |
|-------|----------|--------|
| DoS via extreme exponents | CRITICAL | Needs fix |
| Integer overflow | HIGH | Needs fix |
| Division by zero | HIGH | Needs fix |
| Dark State injection | MEDIUM | Needs fix |
| Shape mismatch | MEDIUM | Needs fix |
| Negative indices | MEDIUM | Needs fix |
| Precision not documented | LOW | Needs fix |
| No range checking | LOW | Needs fix |
| Thread safety | LOW | Document limitation |

**Current Posture:** 5/10  
**After Fixes:** 9/10
