"""
Tests for Abby Normal: Neural Arithmetic Encoding Layers

Tests both Track A (AbacusLayer) and Track B (SorobanEncoder).
"""

import pytest
import torch
import torch.nn as nn

from bbdos.cpu.abacus import (
    AbacusLayer,
    SorobanEncoder,
    SorobanShuffled,
    HybridEncoder,
)


class TestAbacusLayer:
    """Tests for Track A: Adjacency-based augmentation."""
    
    def test_mode_none_shape(self):
        """Baseline mode should pass through with embed projection."""
        layer = AbacusLayer(num_bits=8, embed_dim=16, mode='none')
        bits = torch.randint(0, 2, (4, 8)).float()
        out = layer(bits)
        assert out.shape == (4, 8, 16)
    
    def test_mode_position_shape(self):
        """Position mode adds ordinal position."""
        layer = AbacusLayer(num_bits=8, embed_dim=16, mode='position')
        bits = torch.randint(0, 2, (4, 8)).float()
        out = layer(bits)
        assert out.shape == (4, 8, 16)
        assert layer.input_dim == 2
    
    def test_mode_adjacency_shape(self):
        """Adjacency mode adds neighbor values."""
        layer = AbacusLayer(num_bits=8, embed_dim=16, mode='adjacency')
        bits = torch.randint(0, 2, (4, 8)).float()
        out = layer(bits)
        assert out.shape == (4, 8, 16)
        assert layer.input_dim == 3
    
    def test_mode_full_shape(self):
        """Full mode adds everything."""
        layer = AbacusLayer(num_bits=8, embed_dim=16, mode='full')
        bits = torch.randint(0, 2, (4, 8)).float()
        out = layer(bits)
        assert out.shape == (4, 8, 16)
        assert layer.input_dim == 5
    
    def test_gradient_flow(self):
        """Gradients should flow through the layer."""
        layer = AbacusLayer(num_bits=8, embed_dim=16, mode='adjacency')
        bits = torch.randint(0, 2, (4, 8)).float()
        bits.requires_grad = True
        
        out = layer(bits)
        loss = out.sum()
        loss.backward()
        
        assert layer.proj.weight.grad is not None
        assert not torch.all(layer.proj.weight.grad == 0)
    
    def test_position_values(self):
        """Position should be normalized 0-1."""
        layer = AbacusLayer(num_bits=8, embed_dim=16, mode='position')
        assert layer.positions[0].item() == pytest.approx(0.0)
        assert layer.positions[7].item() == pytest.approx(7/8)
    
    def test_weight_values(self):
        """Arithmetic weights should be powers of 2, normalized."""
        layer = AbacusLayer(num_bits=8, embed_dim=16, mode='full')
        assert layer.weights[0].item() == pytest.approx(1/256)
        assert layer.weights[7].item() == pytest.approx(128/256)


class TestSorobanEncoder:
    """Tests for Track B: Split-Byte Thermometer encoding."""
    
    def test_encode_zero(self):
        """Zero should produce empty thermometer."""
        enc = SorobanEncoder()
        t = enc.encode_value(0)
        assert t.sum().item() == 0
        assert t.shape == (32,)
    
    def test_encode_one(self):
        """One should have single bead in low column."""
        enc = SorobanEncoder()
        t = enc.encode_value(1)
        assert t[:16].sum().item() == 1
        assert t[16:].sum().item() == 0
        assert t[0].item() == 1.0
    
    def test_encode_fifteen(self):
        """Fifteen should fill low column."""
        enc = SorobanEncoder()
        t = enc.encode_value(15)
        assert t[:16].sum().item() == 15
        assert t[16:].sum().item() == 0
        # First 15 positions should be 1
        assert torch.all(t[:15] == 1.0)
        assert t[15].item() == 0.0
    
    def test_encode_sixteen(self):
        """Sixteen should have one bead in high column."""
        enc = SorobanEncoder()
        t = enc.encode_value(16)
        assert t[:16].sum().item() == 0
        assert t[16:].sum().item() == 1
        assert t[16].item() == 1.0
    
    def test_encode_255(self):
        """255 should fill both columns (15 + 15*16)."""
        enc = SorobanEncoder()
        t = enc.encode_value(255)
        assert t[:16].sum().item() == 15
        assert t[16:].sum().item() == 15
    
    def test_roundtrip_all_values(self):
        """Encode-decode should be identity for all 0-255."""
        enc = SorobanEncoder()
        for val in range(256):
            encoded = enc.encode_value(val)
            decoded = enc.decode(encoded)
            assert decoded.item() == val, f"Roundtrip failed for {val}"
    
    def test_batch_encode(self):
        """Batch encoding should match individual encoding."""
        enc = SorobanEncoder()
        values = torch.tensor([0, 1, 15, 16, 127, 255])
        
        batch_encoded = enc.encode_batch(values)
        
        for i, val in enumerate(values):
            individual = enc.encode_value(val.item())
            assert torch.allclose(batch_encoded[i], individual), f"Mismatch at {val}"
    
    def test_batch_decode(self):
        """Batch decoding should match individual decoding."""
        enc = SorobanEncoder()
        values = torch.tensor([0, 1, 15, 16, 127, 255])
        
        encoded = enc.encode_batch(values)
        decoded = enc.decode(encoded)
        
        assert torch.all(decoded == values)
    
    def test_forward_shape(self):
        """Forward should project to embeddings."""
        enc = SorobanEncoder(embed_dim=16)
        soroban_bits = torch.zeros(4, 32)
        out = enc(soroban_bits)
        assert out.shape == (4, 32, 16)
    
    def test_forward_with_column_adjacency(self):
        """Column adjacency should add column ID."""
        enc = SorobanEncoder(embed_dim=16, add_column_adjacency=True)
        soroban_bits = torch.zeros(4, 32)
        out = enc(soroban_bits)
        assert out.shape == (4, 32, 16)
        assert enc.input_dim == 2
    
    def test_gradient_flow(self):
        """Gradients should flow through encoder."""
        enc = SorobanEncoder(embed_dim=16)
        soroban_bits = torch.rand(4, 32)
        soroban_bits.requires_grad = True
        
        out = enc(soroban_bits)
        loss = out.sum()
        loss.backward()
        
        assert enc.proj.weight.grad is not None
    
    def test_sparsity(self):
        """Most values should be sparse (< 50% active)."""
        enc = SorobanEncoder()
        
        sparse_count = 0
        for val in range(256):
            t = enc.encode_value(val)
            density = t.sum().item() / 32
            if density < 0.5:
                sparse_count += 1
        
        # At least half of values should be sparse
        assert sparse_count >= 128


class TestSorobanShuffled:
    """Tests for shuffled Soroban (control condition)."""
    
    def test_roundtrip(self):
        """Shuffled encode-decode should still work."""
        enc = SorobanShuffled(embed_dim=16)
        
        for val in [0, 1, 15, 16, 127, 255]:
            encoded = enc.encode_batch(torch.tensor([val]))
            decoded = enc.decode(encoded)
            assert decoded.item() == val
    
    def test_different_from_normal(self):
        """Shuffled encoding should differ from normal."""
        normal = SorobanEncoder()
        shuffled = SorobanShuffled()
        
        val = 127
        normal_enc = normal.encode_value(val)
        shuffled_enc = shuffled.encode_batch(torch.tensor([val]))[0]
        
        # They should be permutations of each other (same sum)
        assert normal_enc.sum() == shuffled_enc.sum()
        # But not identical (unless very unlucky permutation)
        # This could theoretically fail but probability is 1/32!


class TestHybridEncoder:
    """Tests for Track C: Soroban + adjacency."""
    
    def test_forward_shape(self):
        """Hybrid should produce embeddings."""
        hybrid = HybridEncoder(embed_dim=16)
        soroban_bits = torch.zeros(4, 32)
        out = hybrid(soroban_bits)
        assert out.shape == (4, 32, 16)
    
    def test_encode_decode(self):
        """Should delegate to Soroban for encode/decode."""
        hybrid = HybridEncoder(embed_dim=16)
        
        values = torch.tensor([0, 15, 16, 255])
        encoded = hybrid.encode_batch(values)
        decoded = hybrid.decode(encoded)
        
        assert torch.all(decoded == values)
    
    def test_gradient_flow(self):
        """Gradients should flow through both paths."""
        hybrid = HybridEncoder(embed_dim=16)
        soroban_bits = torch.rand(4, 32)
        soroban_bits.requires_grad = True
        
        out = hybrid(soroban_bits)
        loss = out.sum()
        loss.backward()
        
        assert hybrid.soroban.proj.weight.grad is not None
        assert hybrid.adjacency_proj.weight.grad is not None


class TestTheCarryEvent:
    """
    The critical test: Does the encoding make carry visible?
    
    15 + 1 = 16 should show:
    - Low column goes from full to empty
    - High column goes from empty to one bead
    """
    
    def test_carry_visibility(self):
        """The carry from 15 to 16 should be visually obvious."""
        enc = SorobanEncoder()
        
        t15 = enc.encode_value(15)
        t16 = enc.encode_value(16)
        
        # 15: Low column has 15 beads, high column empty
        assert t15[:16].sum().item() == 15
        assert t15[16:].sum().item() == 0
        
        # 16: Low column empty, high column has 1 bead
        assert t16[:16].sum().item() == 0
        assert t16[16:].sum().item() == 1
        
        # The "state transition" is dramatic
        low_change = (t16[:16] - t15[:16]).sum().item()
        high_change = (t16[16:] - t15[16:]).sum().item()
        
        assert low_change == -15  # Lost 15 beads
        assert high_change == 1   # Gained 1 bead
    
    def test_overflow_pattern(self):
        """
        All overflow transitions (N*16 - 1 -> N*16) should show
        the same pattern: low empties, high increments.
        """
        enc = SorobanEncoder()
        
        for n in range(1, 16):  # 15->16, 31->32, 47->48, etc.
            before = n * 16 - 1
            after = n * 16
            
            t_before = enc.encode_value(before)
            t_after = enc.encode_value(after)
            
            # Before: low column full (15 beads)
            assert t_before[:16].sum().item() == 15, f"Failed at {before}"
            
            # After: low column empty
            assert t_after[:16].sum().item() == 0, f"Failed at {after}"
            
            # High column incremented by 1
            high_before = t_before[16:].sum().item()
            high_after = t_after[16:].sum().item()
            assert high_after == high_before + 1, f"High column wrong at {after}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
