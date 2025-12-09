"""
BitSwitch Kernel Tests

Verifies correctness and performance of the sparse 2-bit kernel.
"""

import pytest
import numpy as np
import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

# Import from legacy location for testing (until kernel is copied to v2)
sys.path.insert(0, "/workspace/BBDOS")
from bitswitch import (
    pack_weights_np,
    unpack_weights_np,
    bitswitch_forward_np,
    BitSwitchLinear,
    _lib
)


class TestPackUnpack:
    """Tests for weight packing and unpacking."""
    
    def test_roundtrip(self):
        """Pack and unpack should return original weights."""
        np.random.seed(42)
        
        rows, cols = 64, 128
        weights = np.random.choice([-1.0, 0.0, 1.0], size=(rows, cols)).astype(np.float32)
        
        packed = pack_weights_np(weights)
        unpacked = unpack_weights_np(packed, rows, cols)
        
        assert np.allclose(weights, unpacked), "Pack/unpack roundtrip failed"
    
    def test_encoding_values(self):
        """Verify specific encoding values."""
        # +1 -> 0x01, -1 -> 0x02, 0 -> 0x00
        weights = np.array([[1, -1, 0, 1], [0, 0, -1, -1]], dtype=np.float32)
        packed = pack_weights_np(weights)
        
        byte0 = packed[0, 0]
        assert (byte0 >> 0) & 0x03 == 0x01, "+1 should encode as 0x01"
        assert (byte0 >> 2) & 0x03 == 0x02, "-1 should encode as 0x02"
        assert (byte0 >> 4) & 0x03 == 0x00, "0 should encode as 0x00"
        assert (byte0 >> 6) & 0x03 == 0x01, "+1 should encode as 0x01"
    
    def test_compression_ratio(self):
        """Packed weights should be 4x smaller in element count."""
        rows, cols = 64, 256
        weights = np.random.choice([-1.0, 0.0, 1.0], size=(rows, cols)).astype(np.float32)
        packed = pack_weights_np(weights)
        
        # 4 ternary values pack into 1 byte (2 bits each)
        # So cols/4 packed bytes vs cols float32 values
        expected_packed_cols = cols // 4
        actual_packed_cols = packed.shape[1]
        
        assert actual_packed_cols == expected_packed_cols, \
            f"Expected {expected_packed_cols} packed cols, got {actual_packed_cols}"


class TestForward:
    """Tests for forward pass computation."""
    
    def test_basic_forward(self):
        """Forward pass should match reference matmul."""
        np.random.seed(42)
        
        batch, in_f, out_f, num_tiles = 2, 8, 8, 2
        
        input_data = np.random.randn(batch, in_f).astype(np.float32)
        weights = np.random.choice([-1.0, 0.0, 1.0], size=(out_f, in_f)).astype(np.float32)
        scales = np.ones(out_f, dtype=np.float32)
        gate_mask = np.ones((batch, num_tiles), dtype=np.int8)
        
        packed = pack_weights_np(weights)
        output = bitswitch_forward_np(input_data, packed, scales, gate_mask, in_f, out_f, num_tiles)
        
        expected = input_data @ weights.T
        
        assert np.allclose(output, expected, atol=1e-5), f"Max diff: {np.abs(output - expected).max()}"
    
    def test_tile_gating(self):
        """Inactive tiles should produce zero output."""
        np.random.seed(42)
        
        batch, in_f, out_f, num_tiles = 2, 16, 16, 4
        out_per_tile = out_f // num_tiles
        
        input_data = np.random.randn(batch, in_f).astype(np.float32)
        weights = np.random.choice([-1.0, 0.0, 1.0], size=(out_f, in_f)).astype(np.float32)
        scales = np.ones(out_f, dtype=np.float32)
        
        # Gate: batch 0 uses tiles 0,2; batch 1 uses tiles 1,3
        gate_mask = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.int8)
        
        packed = pack_weights_np(weights)
        output = bitswitch_forward_np(input_data, packed, scales, gate_mask, in_f, out_f, num_tiles)
        
        for b in range(batch):
            for t in range(num_tiles):
                start = t * out_per_tile
                end = start + out_per_tile
                if gate_mask[b, t] == 0:
                    assert np.allclose(output[b, start:end], 0.0), \
                        f"Inactive tile {t} for batch {b} should be zero"


class TestSpeedup:
    """Performance tests for sparse computation."""
    
    @pytest.mark.skipif(not _lib.available, reason="C++ library not available")
    def test_speedup_at_75_percent_sparsity(self):
        """Verify speedup when 75% of tiles are inactive."""
        batch, in_f, out_f, num_tiles = 32, 512, 2048, 4
        
        input_data = np.random.randn(batch, in_f).astype(np.float32)
        weights = np.random.choice([-1.0, 0.0, 1.0], size=(out_f, in_f)).astype(np.float32)
        scales = np.ones(out_f, dtype=np.float32)
        packed = pack_weights_np(weights)
        
        gate_all = np.ones((batch, num_tiles), dtype=np.int8)
        gate_quarter = np.zeros((batch, num_tiles), dtype=np.int8)
        gate_quarter[:, 0] = 1  # Only tile 0 active
        
        # Warmup
        for _ in range(3):
            bitswitch_forward_np(input_data, packed, scales, gate_all, in_f, out_f, num_tiles)
        
        n_iters = 20
        
        # Measure all tiles active
        start = time.perf_counter()
        for _ in range(n_iters):
            bitswitch_forward_np(input_data, packed, scales, gate_all, in_f, out_f, num_tiles)
        time_all = (time.perf_counter() - start) / n_iters
        
        # Measure 25% tiles active
        start = time.perf_counter()
        for _ in range(n_iters):
            bitswitch_forward_np(input_data, packed, scales, gate_quarter, in_f, out_f, num_tiles)
        time_quarter = (time.perf_counter() - start) / n_iters
        
        speedup = time_all / time_quarter
        
        # Expect at least 2x speedup (theoretical max is 4x)
        assert speedup >= 2.0, f"Expected at least 2x speedup, got {speedup:.2f}x"
        print(f"\nSpeedup at 75% sparsity: {speedup:.2f}x")
        print(f"  All tiles: {time_all*1000:.2f}ms")
        print(f"  One tile:  {time_quarter*1000:.2f}ms")


class TestPyTorchIntegration:
    """Tests for PyTorch module integration."""
    
    def test_module_forward(self):
        """BitSwitchLinear should produce valid output."""
        torch.manual_seed(42)
        
        batch, in_f, out_f, num_tiles = 4, 64, 128, 4
        
        layer = BitSwitchLinear(in_f, out_f, num_tiles)
        x = torch.randn(batch, in_f)
        gate = torch.ones(batch, num_tiles)
        
        layer.train()
        out = layer(x, gate)
        
        assert out.shape == (batch, out_f), f"Wrong output shape: {out.shape}"
    
    def test_gradient_flow(self):
        """Gradients should flow through the layer."""
        torch.manual_seed(42)
        
        batch, in_f, out_f, num_tiles = 4, 64, 128, 4
        
        layer = BitSwitchLinear(in_f, out_f, num_tiles)
        x = torch.randn(batch, in_f)
        gate = torch.ones(batch, num_tiles)
        
        layer.train()
        out = layer(x, gate)
        loss = out.sum()
        loss.backward()
        
        assert layer.weight.grad is not None, "Gradients not computed"
        assert not torch.all(layer.weight.grad == 0), "Gradients are all zero"
    
    @pytest.mark.skipif(not _lib.available, reason="C++ library not available")
    def test_pytorch_cpp_match(self):
        """PyTorch and C++ implementations should match."""
        torch.manual_seed(42)
        
        batch, in_f, out_f, num_tiles = 4, 64, 128, 4
        
        layer = BitSwitchLinear(in_f, out_f, num_tiles)
        x = torch.randn(batch, in_f)
        gate = torch.ones(batch, num_tiles)
        
        layer.eval()
        out_pytorch = layer.forward_pytorch(x, gate)
        
        layer.pack()
        out_cpp = layer.forward_cpp(x, gate)
        
        max_diff = (out_pytorch - out_cpp).abs().max().item()
        assert max_diff < 1e-4, f"PyTorch/C++ mismatch: max diff = {max_diff}"


class TestNumericalAccuracy:
    """Tests for numerical precision."""
    
    def test_accumulation_accuracy(self):
        """Large matmuls should maintain accuracy."""
        np.random.seed(42)
        
        batch, in_f, out_f, num_tiles = 8, 1024, 4096, 4
        
        input_data = np.random.randn(batch, in_f).astype(np.float32)
        weights = np.random.choice([-1.0, 0.0, 1.0], size=(out_f, in_f)).astype(np.float32)
        scales = np.ones(out_f, dtype=np.float32)
        gate_mask = np.ones((batch, num_tiles), dtype=np.int8)
        
        packed = pack_weights_np(weights)
        output = bitswitch_forward_np(input_data, packed, scales, gate_mask, in_f, out_f, num_tiles)
        
        expected = input_data @ weights.T
        
        max_diff = np.abs(output - expected).max()
        assert max_diff < 1e-3, f"Large matmul error too high: {max_diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
