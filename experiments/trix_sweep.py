"""
Hyperparameter Sweep: TriX Differentiable Computer

Systematically test configurations to maximize accuracy.
"""

import torch
import torch.nn.functional as F
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from bbdos.nvf.trix_computer import (
    TriXDifferentiableComputer,
    ComputerConfig,
    create_orthogonal_keys,
)
from bbdos.trix.qat import progressive_quantization_schedule


def run_experiment(
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    end_temp: float,
    seed: int = 42,
) -> dict:
    """Run a single experiment configuration."""
    
    torch.manual_seed(seed)
    
    config = ComputerConfig(
        n_memory_slots=5,
        key_dim=32,
        value_dim=32,
        num_tiles=4,
        hidden_dim=hidden_dim,
        quant_mode='progressive',
    )
    
    computer = TriXDifferentiableComputer(config)
    
    keys = create_orthogonal_keys(5, 32)
    values = [10, 20, 30, 40, 50]
    
    for i, (k, v) in enumerate(zip(keys, values)):
        computer.store(i, k, v)
    
    def make_batch(bs, noise=0.15):
        idx = torch.randint(0, 5, (bs, 2))
        qa = torch.stack([keys[i] for i in idx[:, 0]]) + torch.randn(bs, 32) * noise
        qb = torch.stack([keys[i] for i in idx[:, 1]]) + torch.randn(bs, 32) * noise
        tgt = torch.tensor([values[i] + values[j] for i, j in idx], dtype=torch.float)
        return qa, qb, tgt
    
    optimizer = torch.optim.AdamW(computer.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    start = time.time()
    best_acc = 0
    
    for epoch in range(epochs):
        temp = progressive_quantization_schedule(epoch, epochs, 1.0, end_temp)
        computer.set_quant_temperature(temp)
        
        qa, qb, tgt = make_batch(batch_size)
        optimizer.zero_grad()
        result, _ = computer(qa, qb)
        loss = F.mse_loss(result, tgt)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 100 == 0:
            with torch.no_grad():
                qa, qb, tgt = make_batch(5000)
                err = (computer(qa, qb)[0] - tgt).abs()
                acc = (err < 1).float().mean().item() * 100
                if acc > best_acc:
                    best_acc = acc
    
    elapsed = time.time() - start
    
    # Final eval
    results = {}
    with torch.no_grad():
        for noise in [0.10, 0.15, 0.20]:
            qa, qb, tgt = make_batch(5000, noise)
            err = (computer(qa, qb)[0] - tgt).abs()
            results[f"noise_{noise:.2f}"] = (err < 1).float().mean().item() * 100
    
    return {
        'hidden_dim': hidden_dim,
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'end_temp': end_temp,
        'best_acc': best_acc,
        'time': elapsed,
        'params': computer.num_parameters,
        **results,
    }


def main():
    print("=" * 70)
    print("       TriX HYPERPARAMETER SWEEP")
    print("=" * 70)
    
    results = []
    
    # Sweep configurations
    configs = [
        # (hidden_dim, epochs, batch_size, lr, end_temp)
        (256, 500, 2048, 0.003, 10.0),   # baseline
        (512, 500, 2048, 0.003, 10.0),   # larger model
        (256, 1000, 2048, 0.003, 10.0),  # more epochs
        (256, 500, 4096, 0.005, 10.0),   # larger batch + higher lr
        (256, 500, 2048, 0.003, 5.0),    # softer final temp
        (256, 500, 2048, 0.003, 20.0),   # harder final temp
        (512, 1000, 4096, 0.005, 10.0),  # all bigger
    ]
    
    print(f"\nRunning {len(configs)} configurations...\n")
    print(f"{'Config':>40} | {'Best':>6} | {'0.10':>6} | {'0.15':>6} | {'0.20':>6} | {'Time':>6}")
    print("-" * 85)
    
    for cfg in configs:
        hidden, epochs, batch, lr, temp = cfg
        result = run_experiment(hidden, epochs, batch, lr, temp)
        results.append(result)
        
        cfg_str = f"h={hidden}, e={epochs}, b={batch}, lr={lr}, t={temp}"
        print(f"{cfg_str:>40} | {result['best_acc']:>5.1f}% | {result['noise_0.10']:>5.1f}% | {result['noise_0.15']:>5.1f}% | {result['noise_0.20']:>5.1f}% | {result['time']:>5.1f}s")
    
    # Find best
    best = max(results, key=lambda x: x['noise_0.15'])
    
    print("\n" + "=" * 70)
    print("       BEST CONFIGURATION")
    print("=" * 70)
    print(f"\nhidden_dim: {best['hidden_dim']}")
    print(f"epochs: {best['epochs']}")
    print(f"batch_size: {best['batch_size']}")
    print(f"lr: {best['lr']}")
    print(f"end_temp: {best['end_temp']}")
    print(f"\nNoise=0.10: {best['noise_0.10']:.1f}%")
    print(f"Noise=0.15: {best['noise_0.15']:.1f}%")
    print(f"Noise=0.20: {best['noise_0.20']:.1f}%")
    
    # Save results
    results_path = Path(__file__).parent.parent / "results" / "trix_sweep_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
