"""
Neural 6502 CPU Emulator

A neural network that learns to predict CPU state transitions.
"""

from .model import NeuralCPU, CPUConfig

__all__ = [
    "NeuralCPU",
    "CPUConfig",
]
