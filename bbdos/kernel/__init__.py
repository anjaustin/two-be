"""
BitSwitch Kernel

Sparse 2-bit matrix multiplication with tile-based routing.
Implements ARM NEON acceleration for ternary weights.
"""

from .bindings import (
    BitSwitchLinear,
    pack_weights,
    unpack_weights,
    bitswitch_forward,
    is_neon_available,
)

__all__ = [
    "BitSwitchLinear",
    "pack_weights", 
    "unpack_weights",
    "bitswitch_forward",
    "is_neon_available",
]
