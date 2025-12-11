"""
Training Experiment: TriX Differentiable Computer

Systematic training with quantization-aware training.
Logs all results for reproducibility.

Usage:
    python -m experiments.train_trix_computer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from bbdos.nvf.trix_computer import (
    TriXDifferentiableComputer,
    ComputerConfig,
    create_orthogonal_keys,
)
from bbdos.trix.qat import progressive_quantization_schedule


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 2048
    epochs: int = 500
    lr: float = 0.003
    weight_decay: float = 0.01
    noise_scale: float = 0.15
    
    # QAT settings
    start_temp: float = 1.0
    end_temp: float = 10.0
    
    # Evaluation
    eval_interval: int = 50
    eval_samples: int = 5000


@dataclass
class ExperimentResult:
    """Results from a training run."""
    config: Dict[str, Any]
    training_config: Dict[str, Any]
    
    # Training metrics
    final_loss: float
    final_accuracy: float
    best_accuracy: float
    training_time: float
    
    # Per-noise evaluation
    noise_results: Dict[str, float]
    
    # Quantization
    final_sparsity: float
    final_temperature: float


class Experiment:
    """
    Training experiment for TriX Differentiable Computer.
    """
    
    def __init__(
        self,
        computer_config: ComputerConfig,
        training_config: TrainingConfig,
        seed: int = 42,
    ):
        self.computer_config = computer_config
        self.training_config = training_config
        self.seed = seed
        
        # Set seed
        torch.manual_seed(seed)
        
        # Create computer
        self.computer = TriXDifferentiableComputer(computer_config)
        
        # Create orthogonal keys
        self.keys = create_orthogonal_keys(
            computer_config.n_memory_slots,
            computer_config.key_dim,
        )
        
        # Values to store
        self.values = [10, 20, 30, 40, 50]
        
        # Store in memory
        for i, (k, v) in enumerate(zip(self.keys, self.values)):
            self.computer.store(i, k, v)
        
        # Training state
        self.history = []
    
    def generate_batch(self, batch_size: int, noise: float) -> tuple:
        """Generate training batch."""
        n_slots = self.computer_config.n_memory_slots
        
        indices = torch.randint(0, n_slots, (batch_size, 2))
        
        query_a = torch.stack([self.keys[i] for i in indices[:, 0]])
        query_b = torch.stack([self.keys[i] for i in indices[:, 1]])
        
        # Add noise
        query_a = query_a + torch.randn_like(query_a) * noise
        query_b = query_b + torch.randn_like(query_b) * noise
        
        # Compute targets
        targets = torch.tensor(
            [self.values[i] + self.values[j] for i, j in indices],
            dtype=torch.float,
        )
        
        return query_a, query_b, targets
    
    def evaluate(self, noise: float, n_samples: int = 5000) -> dict:
        """Evaluate accuracy at given noise level."""
        self.computer.eval()
        
        with torch.no_grad():
            query_a, query_b, targets = self.generate_batch(n_samples, noise)
            results, _ = self.computer(query_a, query_b)
            
            errors = (results - targets).abs()
            
            exact = (errors < 1).float().mean().item() * 100
            close = (errors < 3).float().mean().item() * 100
            mae = errors.mean().item()
        
        self.computer.train()
        
        return {
            'exact': exact,
            'close': close,
            'mae': mae,
        }
    
    def train(self) -> ExperimentResult:
        """Run full training."""
        cfg = self.training_config
        
        print("=" * 70)
        print("       TriX DIFFERENTIABLE COMPUTER - TRAINING")
        print("=" * 70)
        print(f"\nComputer config: {asdict(self.computer_config)}")
        print(f"Training config: {asdict(cfg)}")
        print(f"Parameters: {self.computer.num_parameters:,}")
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.computer.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs)
        
        print(f"\nTraining for {cfg.epochs} epochs x {cfg.batch_size:,} samples")
        
        best_accuracy = 0.0
        start_time = time.time()
        
        for epoch in range(cfg.epochs):
            # Update quantization temperature
            temp = progressive_quantization_schedule(
                epoch, cfg.epochs,
                cfg.start_temp, cfg.end_temp,
            )
            self.computer.set_quant_temperature(temp)
            
            # Generate batch
            query_a, query_b, targets = self.generate_batch(
                cfg.batch_size, cfg.noise_scale
            )
            
            # Forward
            optimizer.zero_grad()
            results, _ = self.computer(query_a, query_b)
            loss = F.mse_loss(results, targets)
            
            # Backward
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Log
            if (epoch + 1) % cfg.eval_interval == 0 or epoch == 0:
                eval_result = self.evaluate(cfg.noise_scale, cfg.eval_samples)
                
                self.history.append({
                    'epoch': epoch + 1,
                    'loss': loss.item(),
                    'temp': temp,
                    'sparsity': self.computer.get_total_sparsity(),
                    **eval_result,
                })
                
                if eval_result['exact'] > best_accuracy:
                    best_accuracy = eval_result['exact']
                
                print(
                    f"  Epoch {epoch+1:4d}: "
                    f"loss={loss.item():.3f}, "
                    f"exact={eval_result['exact']:.1f}%, "
                    f"temp={temp:.2f}, "
                    f"sparsity={self.computer.get_total_sparsity():.1%}"
                )
        
        training_time = time.time() - start_time
        
        # Final evaluation at multiple noise levels
        print("\n" + "=" * 70)
        print("       FINAL EVALUATION")
        print("=" * 70)
        
        noise_results = {}
        for noise in [0.10, 0.15, 0.20, 0.25]:
            result = self.evaluate(noise, cfg.eval_samples)
            noise_results[f"noise_{noise:.2f}"] = result['exact']
            print(f"  Noise={noise:.2f}: Exact={result['exact']:.1f}%, ±3={result['close']:.1f}%")
        
        # Gradient flow verification
        print("\n  Gradient flow verification:")
        query_a, query_b, targets = self.generate_batch(10, cfg.noise_scale)
        query_a.requires_grad_(True)
        results, _ = self.computer(query_a, query_b)
        F.mse_loss(results, targets).backward()
        grad_mag = query_a.grad.abs().mean().item()
        print(f"    Query gradient magnitude: {grad_mag:.6f}")
        
        if grad_mag > 0:
            print("    [✓] GRADIENTS FLOW END-TO-END")
        else:
            print("    [✗] GRADIENT FLOW BROKEN")
        
        # Create result
        final_eval = self.evaluate(cfg.noise_scale, cfg.eval_samples)
        
        result = ExperimentResult(
            config=asdict(self.computer_config),
            training_config=asdict(cfg),
            final_loss=loss.item(),
            final_accuracy=final_eval['exact'],
            best_accuracy=best_accuracy,
            training_time=training_time,
            noise_results=noise_results,
            final_sparsity=self.computer.get_total_sparsity(),
            final_temperature=temp,
        )
        
        print("\n" + "=" * 70)
        print("       TRAINING COMPLETE")
        print("=" * 70)
        print(f"\n  Best accuracy: {best_accuracy:.1f}%")
        print(f"  Final accuracy: {final_eval['exact']:.1f}%")
        print(f"  Training time: {training_time:.1f}s")
        print(f"  Throughput: {cfg.epochs * cfg.batch_size / training_time:,.0f} samples/sec")
        
        return result
    
    def save_results(self, path: Path):
        """Save results to JSON."""
        results = {
            'history': self.history,
            'computer_config': asdict(self.computer_config),
            'training_config': asdict(self.training_config),
            'seed': self.seed,
        }
        
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {path}")
    
    def save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.computer.state_dict(),
            'computer_config': asdict(self.computer_config),
            'training_config': asdict(self.training_config),
            'history': self.history,
        }, path)
        
        print(f"Checkpoint saved to {path}")


def main():
    """Run the training experiment."""
    
    # Configuration
    computer_config = ComputerConfig(
        n_memory_slots=5,
        key_dim=32,
        value_dim=32,
        num_tiles=4,
        hidden_dim=256,
        quant_mode='progressive',
    )
    
    training_config = TrainingConfig(
        batch_size=2048,
        epochs=500,
        lr=0.003,
        weight_decay=0.01,
        noise_scale=0.15,
        start_temp=1.0,
        end_temp=10.0,
        eval_interval=50,
        eval_samples=5000,
    )
    
    # Run experiment
    experiment = Experiment(computer_config, training_config, seed=42)
    result = experiment.train()
    
    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    experiment.save_results(results_dir / "trix_computer_experiment.json")
    experiment.save_checkpoint(results_dir / "trix_computer_checkpoint.pt")
    
    return result


if __name__ == "__main__":
    main()
