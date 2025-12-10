"""
TriX Python Bindings

Clean interface to the C++ NEON kernel.
"""

import os
import ctypes
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple

# Locate the shared library
_LIB_PATH = None
_LIB = None

def _find_library() -> Optional[Path]:
    """Search for libtrix.so in common locations."""
    search_paths = [
        Path(__file__).parent / "build" / "libtrix.so",
        Path(__file__).parent / "libtrix.so",
    ]
    
    for path in search_paths:
        if path.exists():
            return path
    return None

def _load_library():
    """Load the TriX C library."""
    global _LIB_PATH, _LIB
    
    if _LIB is not None:
        return _LIB
    
    _LIB_PATH = _find_library()
    if _LIB_PATH is None:
        raise RuntimeError(
            "libtrix.so not found. Build with: "
            "cd bbdos/kernel && mkdir build && cd build && cmake .. && make"
        )
    
    _LIB = ctypes.CDLL(str(_LIB_PATH))
    
    # Set up function signatures
    _LIB.pack_weights.argtypes = [
        ctypes.c_void_p,  # weights
        ctypes.c_void_p,  # packed output
        ctypes.c_int,     # rows
        ctypes.c_int,     # cols
    ]
    
    _LIB.unpack_weights.argtypes = [
        ctypes.c_void_p,  # packed
        ctypes.c_void_p,  # output
        ctypes.c_int,     # rows
        ctypes.c_int,     # cols
    ]
    
    _LIB.trix_forward.argtypes = [
        ctypes.c_void_p,  # input
        ctypes.c_void_p,  # packed weights
        ctypes.c_void_p,  # scales
        ctypes.c_void_p,  # output
        ctypes.c_int,     # batch
        ctypes.c_int,     # in_features
        ctypes.c_int,     # out_features
        ctypes.c_int,     # num_tiles
        ctypes.c_void_p,  # gate
    ]
    
    return _LIB


def is_neon_available() -> bool:
    """Check if NEON acceleration is available."""
    try:
        _load_library()
        return True
    except RuntimeError:
        return False


def pack_weights(weights: torch.Tensor) -> torch.Tensor:
    """
    Pack float32 ternary weights to 2-bit representation.
    
    Args:
        weights: Float tensor with values in {-1, 0, +1}
        
    Returns:
        Packed uint8 tensor (4x smaller)
    """
    lib = _load_library()
    
    rows, cols = weights.shape
    weights_f32 = weights.to(torch.float32).contiguous()
    packed_cols = (cols + 3) // 4
    packed = torch.zeros((rows, packed_cols), dtype=torch.uint8)
    
    lib.pack_weights(
        weights_f32.data_ptr(),
        packed.data_ptr(),
        rows,
        cols
    )
    
    return packed


def unpack_weights(packed: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """
    Unpack 2-bit weights back to float32.
    
    Args:
        packed: Packed uint8 tensor
        rows: Original row count
        cols: Original column count
        
    Returns:
        Float32 tensor with ternary values
    """
    lib = _load_library()
    
    output = torch.zeros((rows, cols), dtype=torch.float32)
    
    lib.unpack_weights(
        packed.data_ptr(),
        output.data_ptr(),
        rows,
        cols
    )
    
    return output


def trix_forward(
    x: torch.Tensor,
    packed_weights: torch.Tensor,
    scales: torch.Tensor,
    gate: torch.Tensor,
    out_features: int,
    num_tiles: int
) -> torch.Tensor:
    """
    Sparse forward pass using packed 2-bit weights.
    
    Args:
        x: Input tensor [batch, in_features]
        packed_weights: Packed weight tensor
        scales: Per-output scaling factors
        gate: Tile activation mask [batch, num_tiles]
        out_features: Output dimension
        num_tiles: Number of routing tiles
        
    Returns:
        Output tensor [batch, out_features]
    """
    lib = _load_library()
    
    batch, in_features = x.shape
    x = x.contiguous()
    output = torch.zeros((batch, out_features), dtype=torch.float32)
    
    # Convert gate to int8 mask
    gate_mask = gate.to(torch.int8).contiguous()
    
    lib.trix_forward(
        x.data_ptr(),
        packed_weights.data_ptr(),
        scales.data_ptr(),
        gate_mask.data_ptr(),
        output.data_ptr(),
        batch,
        in_features,
        out_features,
        num_tiles
    )
    
    return output


class TriXLinear(nn.Module):
    """
    Linear layer with TriX sparse 2-bit weights.
    
    During training, uses full-precision PyTorch operations.
    During inference, uses packed 2-bit NEON kernel (4x faster at 75% sparsity).
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        num_tiles: Number of routing tiles (default: 4)
        bias: Include bias term (default: False)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_tiles: int = 4,
        bias: bool = False
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_tiles = num_tiles
        
        # Ternary weights stored as float for training
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.scales = nn.Parameter(torch.ones(out_features))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Packed weights for inference
        self.register_buffer('packed_weight', None)
        self._packed = False
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights to ternary values."""
        nn.init.kaiming_uniform_(self.weight)
        # Quantize to ternary
        with torch.no_grad():
            self.weight.data = torch.sign(self.weight.data)
    
    def pack(self):
        """Pack weights for inference."""
        if not self._packed:
            self.packed_weight = pack_weights(self.weight.data)
            self._packed = True
    
    def unpack(self):
        """Unpack weights for training."""
        self._packed = False
        self.packed_weight = None
    
    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input [batch, in_features]
            gate: Tile activation [batch, num_tiles]
            
        Returns:
            Output [batch, out_features]
        """
        if self._packed and not self.training:
            # Use NEON kernel
            return trix_forward(
                x, self.packed_weight, self.scales,
                gate, self.out_features, self.num_tiles
            )
        else:
            # PyTorch fallback
            # Apply ternary quantization
            w = torch.sign(self.weight)
            out = torch.mm(x, w.t()) * self.scales
            
            if self.bias is not None:
                out = out + self.bias
            
            return out
    
    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"num_tiles={self.num_tiles}, "
            f"bias={self.bias is not None}"
        )
