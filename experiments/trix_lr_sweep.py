"""
Learning Rate Sweep: TriX Differentiable Computer

Parallel training with lr=[0.001..0.009]
"""

import torch
import torch.nn.functional as F
import time
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from bbdos.nvf.trix_computer import (
    TriXDifferentiableComputer,
    ComputerConfig,
    create_orthogonal_keys,
)
from bbdos.trix.qat import progressive_quantization_schedule


def run_single_lr(lr: float, seed: int = 42) -> dict:
    """Run training with a specific learning rate."""
    
    torch.manual_seed(seed)
    
    config = ComputerConfig(
        n_memory_slots=5,
        key_dim=32,
        value_dim=32,
        num_tiles=4,
        hidden_dim=256,
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
    
    # Fixed config, varying LR
    epochs = 1000
    batch_size = 2048
    end_temp = 10.0
    
    optimizer = torch.optim.AdamW(computer.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    start = time.time()
    best_acc = 0
    history = []
    
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
                history.append({'epoch': epoch + 1, 'acc': acc})
    
    elapsed = time.time() - start
    
    # Final eval at multiple noise levels
    results = {}
    with torch.no_grad():
        for noise in [0.10, 0.15, 0.20]:
            qa, qb, tgt = make_batch(5000, noise)
            err = (computer(qa, qb)[0] - tgt).abs()
            results[f"noise_{noise:.2f}"] = (err < 1).float().mean().item() * 100
    
    return {
        'lr': lr,
        'best_acc': best_acc,
        'time': elapsed,
        'history': history,
        **results,
    }


def main():
    print("=" * 70)
    print("       TriX LEARNING RATE SWEEP")
    print("       lr = [0.001, 0.002, ..., 0.009]")
    print("=" * 70)
    
    learning_rates = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009]
    
    print(f"\nRunning {len(learning_rates)} learning rates...")
    print(f"Config: hidden=256, epochs=1000, batch=2048, temp=10.0\n")
    
    results = []
    
    print(f"{'LR':>8} | {'Best':>6} | {'0.10':>6} | {'0.15':>6} | {'0.20':>6} | {'Time':>6}")
    print("-" * 55)
    
    for lr in learning_rates:
        result = run_single_lr(lr)
        results.append(result)
        
        print(f"{lr:>8.3f} | {result['best_acc']:>5.1f}% | {result['noise_0.10']:>5.1f}% | {result['noise_0.15']:>5.1f}% | {result['noise_0.20']:>5.1f}% | {result['time']:>5.1f}s")
    
    # Find best
    best = max(results, key=lambda x: x['noise_0.15'])
    
    print("\n" + "=" * 70)
    print("       BEST LEARNING RATE")
    print("=" * 70)
    print(f"\nLR: {best['lr']}")
    print(f"Best during training: {best['best_acc']:.1f}%")
    print(f"Noise=0.10: {best['noise_0.10']:.1f}%")
    print(f"Noise=0.15: {best['noise_0.15']:.1f}%")
    print(f"Noise=0.20: {best['noise_0.20']:.1f}%")
    
    # Save results
    results_path = Path(__file__).parent.parent / "results" / "trix_lr_sweep_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Summary table
    print("\n" + "=" * 70)
    print("       SUMMARY")
    print("=" * 70)
    print("\nSorted by noise=0.15 accuracy:\n")
    
    sorted_results = sorted(results, key=lambda x: x['noise_0.15'], reverse=True)
    for r in sorted_results[:5]:
        print(f"  LR={r['lr']:.3f}: {r['noise_0.15']:.1f}%")


if __name__ == "__main__":
    main()
