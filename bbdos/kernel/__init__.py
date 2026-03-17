"""
BitSwitch Kernel

Sparse 2-bit matrix multiplication with tile-based routing.
Implements ARM NEON and AVX acceleration for ternary weights.
"""

from .bindings import (
    BitSwitchLinear,
    pack_weights,
    unpack_weights,
    bitswitch_forward,
    is_neon_available,
    is_avx_available,
)

from .apu_cache import (
    NeuralAPU,
    NeuralCache,
    CacheLine,
    CacheStats,
    CacheStatus,
)

from .mtfp import (
    MTFP,
    MTFP_8,
    MTFP_16,
    MTFP_32,
    MTFP_PRESETS,
    mtfp_add,
    mtfp_mul,
    mtfp_matmul,
    trits_to_packed_bytes,
    packed_bytes_to_trits,
)

TriXLinear = BitSwitchLinear

__all__ = [
    "BitSwitchLinear",
    "TriXLinear",
    "pack_weights",
    "unpack_weights",
    "bitswitch_forward",
    "is_neon_available",
    "is_avx_available",
    "NeuralAPU",
    "NeuralCache",
    "CacheLine",
    "CacheStats",
    "CacheStatus",
    "MTFP",
    "MTFP_8",
    "MTFP_16",
    "MTFP_32",
    "MTFP_PRESETS",
    "mtfp_add",
    "mtfp_mul",
    "mtfp_matmul",
    "trits_to_packed_bytes",
    "packed_bytes_to_trits",
]
