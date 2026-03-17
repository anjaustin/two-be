"""
BBDOS AVX2 Kernel - Python Bindings

Uses the compiled C library for all compute operations.
"""

import ctypes
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

_lib = None


def _load_lib():
    """Load the AVX2 shared library."""
    global _lib
    if _lib is not None:
        return _lib

    search_paths = [
        Path(__file__).parent / "libbbdos_avx.so",
        Path(__file__).parent / "build" / "libbbdos_avx.so",
    ]

    for path in search_paths:
        if path.exists():
            _lib = ctypes.CDLL(str(path))
            _setup_types()
            return _lib

    raise RuntimeError("Could not find libbbdos_avx.so")


def _setup_types():
    """Set up ctypes signatures."""
    lib = _lib

    lib.bbdos_get_capability.restype = ctypes.c_uint64
    lib.bbdos_get_capability.argtypes = []

    lib.bbdos_get_version.restype = ctypes.c_char_p
    lib.bbdos_get_version.argtypes = []

    lib.bbdos_get_mtfp16_trits.restype = ctypes.c_int
    lib.bbdos_get_mtfp16_trits.argtypes = []

    lib.mtfp16_pack_avx.restype = None
    lib.mtfp16_pack_avx.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int8),
        ctypes.c_int,
    ]

    lib.mtfp16_unpack_avx.restype = None
    lib.mtfp16_unpack_avx.argtypes = [
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]

    lib.mtfp16_add_avx.restype = None
    lib.mtfp16_add_avx.argtypes = [
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_int8),
        ctypes.c_int,
    ]

    lib.mtfp16_mul_avx.restype = None
    lib.mtfp16_mul_avx.argtypes = [
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_int8),
        ctypes.c_int,
    ]

    lib.bitswitch_matmul_avx.restype = None
    lib.bitswitch_matmul_avx.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]


def get_capability() -> int:
    """Get hardware capability flags."""
    _load_lib()
    return _lib.bbdos_get_capability()


def get_version() -> str:
    """Get BBDOS version string."""
    _load_lib()
    return _lib.bbdos_get_version().decode()


def is_avx2_available() -> bool:
    """Check if AVX2 is available."""
    return get_capability() == 1


def mtfp16_pack(arr: np.ndarray) -> np.ndarray:
    """Pack float32 array to MTFP-16 format.

    Args:
        arr: float32 array

    Returns:
        int8 array with shape (len(arr), 8)
    """
    _load_lib()

    arr = np.asarray(arr, dtype=np.float32)
    count = arr.size

    result = np.zeros((count, 8), dtype=np.int8)

    _lib.mtfp16_pack_avx(
        arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        result.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        count,
    )

    return result


def mtfp16_unpack(trits: np.ndarray) -> np.ndarray:
    """Unpack MTFP-16 array to float32.

    Args:
        trits: int8 array with shape (n, 8)

    Returns:
        float32 array with shape (n,)
    """
    _load_lib()

    trits = np.asarray(trits, dtype=np.int8)
    count = trits.shape[0]

    result = np.zeros(count, dtype=np.float32)

    _lib.mtfp16_unpack_avx(
        trits.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        result.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        count,
    )

    return result


def mtfp16_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """MTFP-16 addition.

    Args:
        a, b: int8 arrays with shape (n, 8)

    Returns:
        int8 array with shape (n, 8)
    """
    _load_lib()

    a = np.asarray(a, dtype=np.int8)
    b = np.asarray(b, dtype=np.int8)

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    count = a.shape[0]
    result = np.zeros_like(a)

    _lib.mtfp16_add_avx(
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        result.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        count,
    )

    return result


def mtfp16_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """MTFP-16 multiplication.

    Args:
        a, b: int8 arrays with shape (n, 8)

    Returns:
        int8 array with shape (n, 8)
    """
    _load_lib()

    a = np.asarray(a, dtype=np.int8)
    b = np.asarray(b, dtype=np.int8)

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    count = a.shape[0]
    result = np.zeros_like(a)

    _lib.mtfp16_mul_avx(
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        result.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        count,
    )

    return result


def bitswitch_matmul(
    input: np.ndarray,
    packed_weights: np.ndarray,
    scales: np.ndarray,
    gate_mask: Optional[np.ndarray] = None,
    num_tiles: int = 4,
) -> np.ndarray:
    """BitSwitch ternary matmul with AVX acceleration.

    Args:
        input: float32 array [batch, in_features]
        packed_weights: uint8 array [out_features, packed_in]
        scales: float32 array [out_features]
        gate_mask: int8 array [batch, num_tiles] or None
        num_tiles: number of tiles

    Returns:
        float32 array [batch, out_features]
    """
    _load_lib()

    input = np.asarray(input, dtype=np.float32)
    packed_weights = np.asarray(packed_weights, dtype=np.uint8)
    scales = np.asarray(scales, dtype=np.float32)

    if gate_mask is not None:
        gate_mask = np.asarray(gate_mask, dtype=np.int8)

    batch_size, in_features = input.shape
    out_features = packed_weights.shape[0]

    output = np.zeros((batch_size, out_features), dtype=np.float32)

    _lib.bitswitch_matmul_avx(
        input.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        packed_weights.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        gate_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
        if gate_mask is not None
        else None,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        batch_size,
        in_features,
        out_features,
        num_tiles,
    )

    return output
