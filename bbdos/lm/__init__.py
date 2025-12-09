"""
BBDOS Language Model (NanoLPU)

Transformer with BitSwitch sparse layers for 2-bit inference.
"""

from .model import NanoLPU, LMConfig

__all__ = ["NanoLPU", "LMConfig"]
