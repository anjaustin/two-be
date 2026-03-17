"""
Multi-Trit Floating Point (MTFP)

A ternary floating point format using trits (-1, 0, +1) instead of bits.
Uses the Dark State (0b11) for exponent expansion.
"""

import numpy as np
from typing import Tuple, Optional


TRITS = np.array([-1, 0, 1])
TRIT_VALUES = {-1: 0b10, 0: 0b00, 1: 0b01}


class MTFP:
    """
    Multi-Trit Floating Point format.

    Format: [sign: 1 trit][exponent: n_exp trits][mantissa: n_mant trits]

    Total bits = (1 + n_exp + n_mant) * 2 bits

    Examples:
        MTFP-8:  1 + 2 + 1 = 4 trits  = 8 bits
        MTFP-16: 1 + 4 + 3 = 8 trits  = 16 bits
        MTFP-32: 1 + 5 + 6 = 12 trits = 24 bits
    """

    def __init__(
        self, n_mantissa_trits: int = 6, n_exponent_trits: int = 5, n_sign_trit: int = 1
    ):
        self.n_mantissa_trits = n_mantissa_trits
        self.n_exponent_trits = n_exponent_trits
        self.n_sign_trit = n_sign_trit
        self.n_trits = n_sign_trit + n_exponent_trits + n_mantissa_trits
        self.n_bits = self.n_trits * 2

        # Aliases for backwards compatibility
        self.n_mantissa = n_mantissa_trits
        self.n_exponent = n_exponent_trits
        self.n_sign = n_sign_trit

        self.max_exponent = (3**n_exponent_trits) - 1
        self.min_exponent = -(3**n_exponent_trits) + 1

        self.max_exp_value = 3 ** (n_exponent_trits - 1)
        self.mantissa_range = 3**n_mantissa_trits

    @property
    def name(self) -> str:
        return f"MTFP-{self.n_bits}"

    def pack(self, value: float) -> np.ndarray:
        """
        Pack a float into MTFP format.

        Args:
            value: Float to convert

        Returns:
            Packed trits as numpy array
        """
        if np.isnan(value):
            return self._pack_nan()
        if np.isinf(value):
            return self._pack_inf(value > 0)

        trits = np.zeros(self.n_trits, dtype=np.int16)

        sign = 0
        if value < 0:
            sign = 1
            value = -value
        elif value == 0:
            return trits.astype(np.int8)

        exp = 0
        if value >= 1.0:
            while value >= 3.0 and exp < self.max_exponent:
                value /= 3.0
                exp += 1
        else:
            while value < 1.0 and exp > self.min_exponent:
                value *= 3.0
                exp -= 1

        mantissa = int(value * (3 ** (self.n_mantissa_trits - 1)))

        trits[0] = sign

        exp_trits = []
        for _ in range(self.n_exponent_trits):
            exp_trits.append(exp % 3)
            exp //= 3
        trits[self.n_sign : self.n_sign + self.n_exponent_trits] = exp_trits

        for i in range(self.n_mantissa_trits):
            trits[self.n_sign + self.n_exponent_trits + i] = mantissa % 3
            mantissa //= 3

        return trits.astype(np.int8)

    def unpack(self, trits: np.ndarray) -> float:
        """
        Unpack MTFP trits to float.

        Args:
            trits: Packed trit array

        Returns:
            Float value
        """
        if self._is_zero(trits):
            return 0.0
        if self._is_nan(trits):
            return np.nan
        if self._is_inf(trits):
            return np.inf if trits[0] == 0 else -np.inf

        sign = -1 if trits[0] == 1 else 1

        exp = 0
        for i in range(self.n_exponent_trits):
            idx = self.n_sign + i
            exp += int(trits[idx]) * (3**i)

        mantissa = 0
        for i in range(self.n_mantissa_trits):
            idx = self.n_sign + self.n_exponent_trits + i
            mantissa += int(trits[idx]) * (3 ** (self.n_mantissa_trits - 1 - i))

        value = mantissa / (3 ** (self.n_mantissa_trits - 1))
        value *= 3.0**exp

        return sign * value

    def pack_array(self, values: np.ndarray) -> np.ndarray:
        """Pack an array of floats to MTFP."""
        result = np.zeros((len(values), self.n_trits), dtype=np.int8)
        for i, v in enumerate(values):
            result[i] = self.pack(v)
        return result

    def unpack_array(self, trits: np.ndarray) -> np.ndarray:
        """Unpack MTFP array to floats."""
        return np.array([self.unpack(t) for t in trits])

    def pack_bytes(self, values: np.ndarray) -> np.ndarray:
        """Pack to byte array (2 bits per trit)."""
        packed = np.packbits((trits_to_uint8(self.pack_array(values).ravel())), axis=-1)
        return packed

    def _pack_zero(self) -> np.ndarray:
        return np.zeros(self.n_trits, dtype=np.int8)

    def _pack_inf(self, positive: bool) -> np.ndarray:
        trits = np.zeros(self.n_trits, dtype=np.int8)
        if not positive:
            trits[0] = 1
        trits[1 : self.n_exponent_trits + 1] = 2
        return trits

    def _pack_nan(self) -> np.ndarray:
        trits = np.zeros(self.n_trits, dtype=np.int8)
        trits[1 : self.n_sign + self.n_exponent_trits + 2] = 2
        return trits

    def _is_zero(self, trits: np.ndarray) -> bool:
        return np.all(trits[self.n_sign :] == 0)

    def _is_inf(self, trits: np.ndarray) -> bool:
        exp_region = trits[self.n_sign : self.n_sign + self.n_exponent_trits]
        return np.all(exp_region == 2)

    def _is_nan(self, trits: np.ndarray) -> bool:
        if not self._is_inf(trits):
            return False
        return trits[self.n_sign + self.n_exponent_trits] == 2

    def __repr__(self) -> str:
        return f"MTFP(mantissa={self.n_mantissa}, exponent={self.n_exponent}, bits={self.n_bits})"


def trits_to_uint8(trits: np.ndarray) -> np.ndarray:
    """Convert trits (-1, 0, 1) to uint8 (00, 01, 10)."""
    result = np.zeros(len(trits), dtype=np.uint8)
    result[trits == 0] = 0b00
    result[trits == 1] = 0b01
    result[trits == -1] = 0b10
    return result


def uint8_to_trits(data: np.ndarray) -> np.ndarray:
    """Convert uint8 to trits (-1, 0, 1)."""
    result = np.zeros(len(data), dtype=np.int8)
    result[data == 0b00] = 0
    result[data == 0b01] = 1
    result[data == 0b10] = -1
    return result


MTFP_8 = MTFP(n_mantissa_trits=1, n_exponent_trits=2, n_sign_trit=1)
MTFP_16 = MTFP(n_mantissa_trits=3, n_exponent_trits=4, n_sign_trit=1)
MTFP_32 = MTFP(n_mantissa_trits=6, n_exponent_trits=5, n_sign_trit=1)


def trits_to_packed_bytes(trits: np.ndarray) -> np.ndarray:
    """Convert trits (-1, 0, 1) to packed 2-bit bytes.

    4 trits pack into 1 byte:
    [trit3:2bits][trit2:2bits][trit1:2bits][trit0:2bits]

    Maps: -1 -> 0b10, 0 -> 0b00, 1 -> 0b01, Dark State -> 0b11
    """
    n_trits = len(trits)
    n_bytes = (n_trits + 3) // 4

    packed = np.zeros(n_bytes, dtype=np.uint8)

    for i in range(n_trits):
        t = trits[i]
        code = 0b00
        if t == 1:
            code = 0b01
        elif t == -1:
            code = 0b10
        elif t == 2:  # Dark State / reserved
            code = 0b11

        byte_idx = i // 4
        bit_idx = (i % 4) * 2
        packed[byte_idx] |= code << bit_idx

    return packed


def packed_bytes_to_trits(packed: np.ndarray, n_trits: int) -> np.ndarray:
    """Convert packed 2-bit bytes back to trits."""
    n_bytes = len(packed)
    trits = np.zeros(n_trits, dtype=np.int8)

    for i in range(n_trits):
        byte_idx = i // 4
        bit_idx = (i % 4) * 2
        code = (packed[byte_idx] >> bit_idx) & 0x03

        if code == 0b01:
            trits[i] = 1
        elif code == 0b10:
            trits[i] = -1
        elif code == 0b11:
            trits[i] = 2  # Dark State
        else:
            trits[i] = 0

    return trits


def mtfp_add(a: np.ndarray, b: np.ndarray, mtfp: MTFP = MTFP_16) -> np.ndarray:
    """Add two MTFP arrays."""
    a_float = mtfp.unpack_array(a)
    b_float = mtfp.unpack_array(b)
    result_float = a_float + b_float
    return mtfp.pack_array(result_float)


def mtfp_mul(a: np.ndarray, b: np.ndarray, mtfp: MTFP = MTFP_16) -> np.ndarray:
    """Multiply two MTFP arrays."""
    a_float = mtfp.unpack_array(a)
    b_float = mtfp.unpack_array(b)
    result_float = a_float * b_float
    return mtfp.pack_array(result_float)


def mtfp_matmul(a: np.ndarray, b: np.ndarray, mtfp: MTFP = MTFP_16) -> np.ndarray:
    """Matrix multiply MTFP arrays."""
    a_float = mtfp.unpack_array(a.reshape(-1, a.shape[-1]))
    b_float = mtfp.unpack_array(b.reshape(b.shape[0], -1))
    result_float = np.matmul(a_float, b_float.T if b.ndim > 1 else b_float)
    return mtfp.pack_array(result_float.ravel()).reshape(a.shape[0], -1)


MTFP_PRESETS = {
    8: MTFP_8,
    16: MTFP_16,
    32: MTFP_32,
}
