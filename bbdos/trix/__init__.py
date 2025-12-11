"""
TriX Quantization-Aware Training

Proper infrastructure for training with ternary weights.
"""

from .qat import (
    TernaryQuantizer,
    TriXLinearQAT,
    Top1Gate,
    progressive_quantization_schedule,
)

__all__ = [
    "TernaryQuantizer",
    "TriXLinearQAT", 
    "Top1Gate",
    "progressive_quantization_schedule",
]
