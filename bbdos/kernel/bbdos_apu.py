"""
BBDOS APU - Python ctypes bindings to C/AVX implementation
"""

import ctypes
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

_lib = None


def _load_lib():
    global _lib
    if _lib is not None:
        return _lib

    lib_path = Path(__file__).parent / "libbbdos_apu.so"
    if not lib_path.exists():
        raise RuntimeError(f"libbbdos_apu.so not found at {lib_path}")

    _lib = ctypes.CDLL(str(lib_path))

    _lib.bbdos_apu_create.argtypes = [ctypes.c_int]
    _lib.bbdos_apu_create.restype = ctypes.c_void_p

    _lib.bbdos_apu_destroy.argtypes = [ctypes.c_void_p]
    _lib.bbdos_apu_destroy.restype = None

    _lib.bbdos_apu_exec.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_void_p,
    ]
    _lib.bbdos_apu_exec.restype = ctypes.c_int

    _lib.bbdos_apu_stats.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    _lib.bbdos_apu_stats.restype = None

    _lib.bbdos_capability.argtypes = []
    _lib.bbdos_capability.restype = ctypes.c_uint64

    _lib.bbdos_version.argtypes = []
    _lib.bbdos_version.restype = ctypes.c_char_p

    _lib.bitswitch_matmul_avx.argtypes = [
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
    _lib.bitswitch_matmul_avx.restype = None

    return _lib


class BBDOS_APU:
    def __init__(self, cache_size: int = 256):
        lib = _load_lib()
        self._apu = lib.bbdos_apu_create(cache_size)
        if not self._apu:
            raise RuntimeError("Failed to create APU context")

    def __del__(self):
        if self._apu:
            _lib.bbdos_apu_destroy(self._apu)

    def exec(self, opcode: str, operands: list, shapes: list) -> np.ndarray:
        lib = _load_lib()

        opcode_bytes = opcode.encode("utf-8")

        output_size = max(shapes) * 8
        output = np.zeros(output_size, dtype=np.int8)

        operand_ptrs = (ctypes.c_void_p * len(operands))()
        for i, op in enumerate(operands):
            if isinstance(op, np.ndarray):
                operand_ptrs[i] = op.ctypes.data_as(ctypes.c_void_p)
            else:
                operand_ptrs[i] = ctypes.cast(
                    ctypes.pointer(ctypes.c_void_p(op)), ctypes.c_void_p
                )

        shape_arr = (ctypes.c_int * len(shapes))(*shapes)

        result = lib.bbdos_apu_exec(
            self._apu,
            opcode_bytes,
            operand_ptrs,
            shape_arr,
            output.ctypes.data_as(ctypes.c_void_p),
        )

        if result != 0:
            raise RuntimeError(f"APU exec failed with code {result}")

        return output

    def stats(self) -> Dict[str, int]:
        lib = _load_lib()
        hits = ctypes.c_int()
        misses = ctypes.c_int()
        evictions = ctypes.c_int()
        rejected = ctypes.c_int()

        lib.bbdos_apu_stats(
            self._apu,
            ctypes.byref(hits),
            ctypes.byref(misses),
            ctypes.byref(evictions),
            ctypes.byref(rejected),
        )

        return {
            "hits": hits.value,
            "misses": misses.value,
            "evictions": evictions.value,
            "rejected": rejected.value,
        }


def capability() -> int:
    lib = _load_lib()
    return lib.bbdos_capability()


def version() -> str:
    lib = _load_lib()
    return lib.bbdos_version().decode("utf-8")


def bitswitch_matmul_avx(
    input: np.ndarray,
    packed_weights: np.ndarray,
    scales: np.ndarray,
    gate_mask: Optional[np.ndarray],
    output: np.ndarray,
    batch_size: int,
    in_features: int,
    out_features: int,
    num_tiles: int,
) -> None:
    lib = _load_lib()

    input = np.ascontiguousarray(input, dtype=np.float32)
    packed_weights = np.ascontiguousarray(packed_weights, dtype=np.uint8)
    scales = np.ascontiguousarray(scales, dtype=np.float32)
    gate_mask_arr = (
        np.ascontiguousarray(gate_mask, dtype=np.int8)
        if gate_mask is not None
        else None
    )
    output = np.ascontiguousarray(output, dtype=np.float32)

    lib.bitswitch_matmul_avx(
        input.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        packed_weights.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        gate_mask_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
        if gate_mask_arr is not None
        else None,
        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(batch_size),
        ctypes.c_int(in_features),
        ctypes.c_int(out_features),
        ctypes.c_int(num_tiles),
    )

    return output
