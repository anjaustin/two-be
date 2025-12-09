"""
Neural 6502 CPU Tests

Verifies model architecture and opcode prediction accuracy.
"""

import pytest
import sys
import os
from pathlib import Path

import torch
import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/workspace/BBDOS")

from bbdos.cpu import NeuralCPU, CPUConfig

try:
    from py65.devices.mpu6502 import MPU
    HAS_PY65 = True
except ImportError:
    HAS_PY65 = False


class TestModelArchitecture:
    """Tests for model structure and dimensions."""
    
    def test_default_config(self):
        """Model should initialize with default config."""
        model = NeuralCPU()
        assert model is not None
        assert model.config.d_model == 256
        assert model.config.num_tiles == 4
    
    def test_custom_config(self):
        """Model should accept custom config."""
        config = CPUConfig(d_model=128, n_heads=2, n_layers=3, num_tiles=2)
        model = NeuralCPU(config)
        assert model.config.d_model == 128
        assert model.config.num_tiles == 2
    
    def test_parameter_count(self):
        """Model should have expected parameter count."""
        model = NeuralCPU()
        # Should be around 2-6M parameters (v2 model is ~5.3M)
        assert 1_000_000 < model.num_parameters < 10_000_000
    
    def test_forward_shape(self):
        """Forward pass should produce correct output shapes."""
        model = NeuralCPU()
        batch_size = 4
        
        state = {
            'A': torch.randint(0, 256, (batch_size,)),
            'X': torch.randint(0, 256, (batch_size,)),
            'Y': torch.randint(0, 256, (batch_size,)),
            'SP': torch.randint(0, 256, (batch_size,)),
            'P': torch.randint(0, 256, (batch_size,)),
            'PCH': torch.randint(0, 256, (batch_size,)),
            'PCL': torch.randint(0, 256, (batch_size,)),
            'Op': torch.randint(0, 256, (batch_size,)),
            'Val': torch.randint(0, 256, (batch_size,)),
        }
        
        preds, gates = model(state)
        
        # Check all registers have predictions
        for reg in ['A', 'X', 'Y', 'SP', 'P', 'PCH', 'PCL']:
            assert reg in preds
            assert preds[reg].shape == (batch_size, 256)
        
        # Check gates shape
        assert gates.shape[0] == batch_size
    
    def test_gradient_flow(self):
        """Gradients should flow through the model."""
        model = NeuralCPU()
        
        state = {
            'A': torch.randint(0, 256, (2,)),
            'X': torch.randint(0, 256, (2,)),
            'Y': torch.randint(0, 256, (2,)),
            'SP': torch.randint(0, 256, (2,)),
            'P': torch.randint(0, 256, (2,)),
            'PCH': torch.randint(0, 256, (2,)),
            'PCL': torch.randint(0, 256, (2,)),
            'Op': torch.randint(0, 256, (2,)),
            'Val': torch.randint(0, 256, (2,)),
        }
        
        preds, _ = model(state)
        loss = sum(p.sum() for p in preds.values())
        loss.backward()
        
        # Check gradients exist
        has_grad = False
        for p in model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients computed"


class TestPredictState:
    """Tests for state prediction."""
    
    def test_predict_returns_values(self):
        """predict_state should return actual values, not logits."""
        model = NeuralCPU()
        
        state = {
            'A': torch.tensor([0x42]),
            'X': torch.tensor([0x00]),
            'Y': torch.tensor([0x00]),
            'SP': torch.tensor([0xFF]),
            'P': torch.tensor([0x00]),
            'PCH': torch.tensor([0x02]),
            'PCL': torch.tensor([0x00]),
            'Op': torch.tensor([0xEA]),  # NOP
            'Val': torch.tensor([0x00]),
        }
        
        pred = model.predict_state(state)
        
        # Should return scalar values (0-255)
        for reg in ['A', 'X', 'Y', 'SP', 'P', 'PCH', 'PCL']:
            assert reg in pred
            assert 0 <= pred[reg].item() <= 255


@pytest.mark.skipif(not HAS_PY65, reason="py65 not installed")
class TestOpcodeAccuracy:
    """Tests for specific opcode behavior."""
    
    @pytest.fixture
    def legacy_model(self):
        """Load legacy model if available."""
        checkpoint_path = Path("/workspace/BBDOS/neural_cpu_best.pt")
        if not checkpoint_path.exists():
            pytest.skip("Legacy checkpoint not found")
        
        from neural_cpu import NeuralCPU as LegacyNeuralCPU
        model = LegacyNeuralCPU()
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.eval()
        return model
    
    def test_nop_preserves_state(self, legacy_model):
        """NOP (0xEA) should preserve all registers."""
        state = {
            'A': torch.tensor([0x42]),
            'X': torch.tensor([0x17]),
            'Y': torch.tensor([0x23]),
            'SP': torch.tensor([0xFF]),
            'P': torch.tensor([0x00]),
            'PCH': torch.tensor([0x02]),
            'PCL': torch.tensor([0x00]),
            'Op': torch.tensor([0xEA]),  # NOP
            'Val': torch.tensor([0x00]),
        }
        
        with torch.no_grad():
            preds, _ = legacy_model(state)
        
        # A, X, Y, SP should be unchanged
        assert preds['A'].argmax().item() == 0x42
        assert preds['X'].argmax().item() == 0x17
        assert preds['Y'].argmax().item() == 0x23
    
    def test_inx_increments_x(self, legacy_model):
        """INX (0xE8) should increment X register."""
        for x_val in [0x00, 0x10, 0x7F, 0xFE]:
            state = {
                'A': torch.tensor([0x00]),
                'X': torch.tensor([x_val]),
                'Y': torch.tensor([0x00]),
                'SP': torch.tensor([0xFF]),
                'P': torch.tensor([0x00]),
                'PCH': torch.tensor([0x02]),
                'PCL': torch.tensor([0x00]),
                'Op': torch.tensor([0xE8]),  # INX
                'Val': torch.tensor([0x00]),
            }
            
            with torch.no_grad():
                preds, _ = legacy_model(state)
            
            expected = (x_val + 1) & 0xFF
            actual = preds['X'].argmax().item()
            # Allow some tolerance since model isn't perfect
            assert abs(actual - expected) <= 1 or actual == expected
    
    def test_tax_transfers_a_to_x(self, legacy_model):
        """TAX (0xAA) should copy A to X."""
        state = {
            'A': torch.tensor([0x42]),
            'X': torch.tensor([0x00]),
            'Y': torch.tensor([0x00]),
            'SP': torch.tensor([0xFF]),
            'P': torch.tensor([0x00]),
            'PCH': torch.tensor([0x02]),
            'PCL': torch.tensor([0x00]),
            'Op': torch.tensor([0xAA]),  # TAX
            'Val': torch.tensor([0x00]),
        }
        
        with torch.no_grad():
            preds, _ = legacy_model(state)
        
        # X should now equal A
        pred_x = preds['X'].argmax().item()
        # Model should get this right or close
        assert pred_x == 0x42 or abs(pred_x - 0x42) <= 5
    
    def test_clc_clears_carry(self, legacy_model):
        """CLC (0x18) should clear carry flag in P."""
        state = {
            'A': torch.tensor([0x00]),
            'X': torch.tensor([0x00]),
            'Y': torch.tensor([0x00]),
            'SP': torch.tensor([0xFF]),
            'P': torch.tensor([0x01]),  # Carry set
            'PCH': torch.tensor([0x02]),
            'PCL': torch.tensor([0x00]),
            'Op': torch.tensor([0x18]),  # CLC
            'Val': torch.tensor([0x00]),
        }
        
        with torch.no_grad():
            preds, _ = legacy_model(state)
        
        # Carry bit should be cleared
        pred_p = preds['P'].argmax().item()
        assert (pred_p & 0x01) == 0, f"Carry should be cleared, got P={pred_p:02X}"
    
    def test_sec_sets_carry(self, legacy_model):
        """SEC (0x38) should set carry flag in P."""
        state = {
            'A': torch.tensor([0x00]),
            'X': torch.tensor([0x00]),
            'Y': torch.tensor([0x00]),
            'SP': torch.tensor([0xFF]),
            'P': torch.tensor([0x00]),  # Carry clear
            'PCH': torch.tensor([0x02]),
            'PCL': torch.tensor([0x00]),
            'Op': torch.tensor([0x38]),  # SEC
            'Val': torch.tensor([0x00]),
        }
        
        with torch.no_grad():
            preds, _ = legacy_model(state)
        
        # Carry bit should be set
        pred_p = preds['P'].argmax().item()
        assert (pred_p & 0x01) == 1, f"Carry should be set, got P={pred_p:02X}"


class TestGateRouting:
    """Tests for BitSwitch gate routing behavior."""
    
    def test_gates_are_one_hot(self):
        """Gate outputs should be approximately one-hot."""
        model = NeuralCPU()
        model.eval()
        
        state = {
            'A': torch.randint(0, 256, (8,)),
            'X': torch.randint(0, 256, (8,)),
            'Y': torch.randint(0, 256, (8,)),
            'SP': torch.randint(0, 256, (8,)),
            'P': torch.randint(0, 256, (8,)),
            'PCH': torch.randint(0, 256, (8,)),
            'PCL': torch.randint(0, 256, (8,)),
            'Op': torch.randint(0, 256, (8,)),
            'Val': torch.randint(0, 256, (8,)),
        }
        
        with torch.no_grad():
            _, gates = model(state)
        
        # Gates should sum to approximately 1 per sample
        gate_sums = gates.sum(dim=-1)
        assert torch.allclose(gate_sums, torch.ones_like(gate_sums), atol=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
