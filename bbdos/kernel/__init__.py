"""
BBDOS Kernel - C/AVX Accelerated

Sparse 2-bit matrix multiplication with tile-based routing.
Implements AVX2 acceleration for ternary weights.

Primary: C/AVX implementation via bbdos_apu
Legacy: Python implementations archived in archive/
"""

from .bbdos_apu import (
    BBDOS_APU,
    bitswitch_matmul_avx,
    capability,
    version,
)

try:
    from .bindings import (
        BitSwitchLinear,
        pack_weights,
        unpack_weights,
        bitswitch_forward,
        is_neon_available,
        is_avx_available,
    )

    _HAS_BINDINGS = True
except ImportError:
    BitSwitchLinear = None
    pack_weights = None
    unpack_weights = None
    bitswitch_forward = None
    is_neon_available = lambda: False
    is_avx_available = lambda: False
    _HAS_BINDINGS = False

try:
    from .apu_cache import (
        NeuralAPU,
        NeuralCache,
        CacheLine,
        CacheStats,
        CacheStatus,
    )

    _HAS_APU_CACHE = True
except ImportError:
    NeuralAPU = NeuralCache = CacheLine = CacheStats = CacheStatus = None
    _HAS_APU_CACHE = False

try:
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

    _HAS_MTFP = True
except ImportError:
    MTFP = MTFP_8 = MTFP_16 = MTFP_32 = MTFP_PRESETS = None
    mtfp_add = mtfp_mul = mtfp_matmul = None
    trits_to_packed_bytes = packed_bytes_to_trits = None
    _HAS_MTFP = False

TriXLinear = BitSwitchLinear

__all__ = [
    "BBDOS_APU",
    "bitswitch_matmul_avx",
    "capability",
    "version",
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
