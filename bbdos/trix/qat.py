"""
Quantization-Aware Training for TriX

Implements proper STE and progressive quantization for ternary weights.

References:
- Courbariaux et al., "Binarized Neural Networks"
- Zhou et al., "DoReFa-Net"
- Yin et al., "Understanding Straight-Through Estimator"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class TernaryQuantizer(torch.autograd.Function):
    """
    Ternary quantization with Straight-Through Estimator.
    
    Forward: quantize to {-1, 0, +1}
    Backward: pass gradient through (STE)
    
    Uses threshold-based quantization:
    - |w| < threshold → 0
    - w >= threshold → +1
    - w <= -threshold → -1
    """
    
    @staticmethod
    def forward(ctx, weights: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
        ctx.save_for_backward(weights)
        ctx.threshold = threshold
        
        # Ternary quantization
        output = torch.zeros_like(weights)
        output[weights > threshold] = 1.0
        output[weights < -threshold] = -1.0
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        weights, = ctx.saved_tensors
        
        # STE: pass gradient through, but clip for stability
        # Only pass gradients for weights within [-1, 1]
        grad_input = grad_output.clone()
        grad_input[weights.abs() > 1.0] = 0
        
        return grad_input, None


class SoftTernaryQuantizer(nn.Module):
    """
    Soft ternary quantization using tanh approximation.
    
    During training, uses soft quantization that approaches ternary.
    Temperature controls sharpness: low = soft, high = hard.
    """
    
    def __init__(self, initial_temp: float = 1.0):
        super().__init__()
        self.register_buffer('temperature', torch.tensor(initial_temp))
    
    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        # Soft sign: tanh(w * temperature)
        # As temperature → ∞, this approaches hard sign
        return torch.tanh(weights * self.temperature)
    
    def set_temperature(self, temp: float):
        self.temperature.fill_(temp)


class Top1Gate(torch.autograd.Function):
    """
    Hard top-1 gating with straight-through gradient.
    
    Forward: one-hot on argmax
    Backward: pass gradient through
    """
    
    @staticmethod
    def forward(ctx, logits: torch.Tensor) -> torch.Tensor:
        idx = torch.argmax(logits, dim=-1, keepdim=True)
        return torch.zeros_like(logits).scatter_(-1, idx, 1.0)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


class TriXLinearQAT(nn.Module):
    """
    TriX Linear layer with Quantization-Aware Training.
    
    Supports multiple quantization modes:
    - 'none': continuous weights (baseline)
    - 'soft': soft ternary via tanh
    - 'ste': hard ternary with STE
    - 'progressive': soft → hard over training
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        num_tiles: Number of routing tiles
        quant_mode: Quantization mode
        threshold: Threshold for ternary quantization
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_tiles: int = 4,
        quant_mode: str = 'progressive',
        threshold: float = 0.05,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_tiles = num_tiles
        self.quant_mode = quant_mode
        self.threshold = threshold
        
        # Main weights (continuous, will be quantized)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        
        # Per-output scales (remain continuous)
        self.scales = nn.Parameter(torch.ones(out_features))
        
        # Gate projection (remains continuous)
        self.gate_proj = nn.Linear(in_features, num_tiles)
        
        # Soft quantizer for progressive mode
        self.soft_quant = SoftTernaryQuantizer(initial_temp=1.0)
        
        self._init_weights()
    
    def _init_weights(self):
        # Initialize with larger values so quantization doesn't zero everything
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # Scale down but keep significant magnitude
        with torch.no_grad():
            self.weight.data *= 0.5
    
    def get_quantized_weight(self) -> torch.Tensor:
        """Get weights with current quantization applied."""
        if self.quant_mode == 'none':
            return self.weight
        elif self.quant_mode == 'soft':
            return self.soft_quant(self.weight)
        elif self.quant_mode == 'ste':
            return TernaryQuantizer.apply(self.weight, self.threshold)
        elif self.quant_mode == 'progressive':
            # Use soft quantizer with current temperature
            return self.soft_quant(self.weight)
        else:
            raise ValueError(f"Unknown quant_mode: {self.quant_mode}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with gated routing.
        
        Args:
            x: Input tensor [batch, in_features]
        
        Returns:
            Output tensor [batch, out_features]
        """
        # Generate gate
        gate = Top1Gate.apply(self.gate_proj(x))
        
        # Get quantized weights
        w = self.get_quantized_weight()
        
        # Linear transform with scales
        out = F.linear(x, w) * self.scales
        
        return out
    
    def set_quant_temperature(self, temp: float):
        """Set soft quantization temperature."""
        self.soft_quant.set_temperature(temp)
    
    def get_sparsity(self) -> float:
        """Calculate weight sparsity (fraction of zeros)."""
        w = self.get_quantized_weight()
        return (w.abs() < 0.1).float().mean().item()
    
    def get_ternary_distribution(self) -> dict:
        """Get distribution of ternary values."""
        w = self.get_quantized_weight()
        return {
            'neg': (w < -0.5).float().mean().item(),
            'zero': (w.abs() < 0.5).float().mean().item(),
            'pos': (w > 0.5).float().mean().item(),
        }
    
    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"num_tiles={self.num_tiles}, "
            f"quant_mode={self.quant_mode}"
        )


def progressive_quantization_schedule(
    epoch: int,
    total_epochs: int,
    start_temp: float = 1.0,
    end_temp: float = 10.0,
) -> float:
    """
    Compute temperature for progressive quantization.
    
    Starts soft (low temp), ends hard (high temp).
    Uses cosine schedule for smooth transition.
    
    Args:
        epoch: Current epoch
        total_epochs: Total training epochs
        start_temp: Starting temperature (soft)
        end_temp: Ending temperature (hard)
    
    Returns:
        Temperature for current epoch
    """
    progress = epoch / total_epochs
    # Cosine schedule: start slow, accelerate in middle, slow at end
    temp = start_temp + (end_temp - start_temp) * (1 - math.cos(math.pi * progress)) / 2
    return temp


class QATTrainer:
    """
    Helper class for quantization-aware training.
    
    Manages temperature scheduling and quantization state.
    """
    
    def __init__(
        self,
        model: nn.Module,
        total_epochs: int,
        start_temp: float = 1.0,
        end_temp: float = 10.0,
    ):
        self.model = model
        self.total_epochs = total_epochs
        self.start_temp = start_temp
        self.end_temp = end_temp
        self.current_epoch = 0
    
    def step_epoch(self):
        """Update quantization temperature for new epoch."""
        self.current_epoch += 1
        temp = progressive_quantization_schedule(
            self.current_epoch,
            self.total_epochs,
            self.start_temp,
            self.end_temp,
        )
        
        # Update all TriXLinearQAT modules
        for module in self.model.modules():
            if isinstance(module, TriXLinearQAT):
                module.set_quant_temperature(temp)
        
        return temp
    
    def get_model_sparsity(self) -> float:
        """Get average sparsity across all TriX layers."""
        sparsities = []
        for module in self.model.modules():
            if isinstance(module, TriXLinearQAT):
                sparsities.append(module.get_sparsity())
        return sum(sparsities) / len(sparsities) if sparsities else 0.0


if __name__ == "__main__":
    # Self-test
    print("=" * 60)
    print("TriX QAT Self-Test")
    print("=" * 60)
    
    # Test ternary quantizer
    print("\n[1] TernaryQuantizer")
    w = torch.randn(4, 4) * 0.2
    w_q = TernaryQuantizer.apply(w, 0.05)
    print(f"    Input range: [{w.min():.3f}, {w.max():.3f}]")
    print(f"    Output values: {w_q.unique().tolist()}")
    
    # Test gradient flow
    w_test = torch.randn(4, 4, requires_grad=True)
    w_q = TernaryQuantizer.apply(w_test, 0.05)
    loss = w_q.sum()
    loss.backward()
    print(f"    Gradient flows: {w_test.grad is not None}")
    
    # Test TriXLinearQAT
    print("\n[2] TriXLinearQAT")
    layer = TriXLinearQAT(32, 64, num_tiles=4, quant_mode='progressive')
    x = torch.randn(8, 32)
    y = layer(x)
    print(f"    Input: {x.shape}")
    print(f"    Output: {y.shape}")
    print(f"    Sparsity: {layer.get_sparsity():.1%}")
    print(f"    Distribution: {layer.get_ternary_distribution()}")
    
    # Test gradient flow
    loss = y.sum()
    loss.backward()
    print(f"    Weight grad: {layer.weight.grad.abs().mean().item():.6f}")
    
    # Test temperature schedule
    print("\n[3] Progressive Schedule")
    for e in [0, 25, 50, 75, 100]:
        temp = progressive_quantization_schedule(e, 100, 1.0, 10.0)
        print(f"    Epoch {e:3d}: temp={temp:.2f}")
    
    print("\n" + "=" * 60)
    print("Self-Test Complete")
    print("=" * 60)
