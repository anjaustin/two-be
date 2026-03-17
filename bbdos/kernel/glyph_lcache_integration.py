#!/usr/bin/env python3
"""
Glyph + BBDOS L-Cache Integration

This demonstrates L-Cache attention caching working alongside Glyph.cpp inference.
For full integration, this would be compiled into the C++ codebase.
"""

import ctypes
import numpy as np
import time
import subprocess
import os

# Load the L-Cache shim
L_SHIM = ctypes.CDLL(os.path.dirname(__file__) + "/libbbdos_lcache_shim.so")

# Setup API
L_SHIM.bbdos_create.argtypes = [ctypes.c_int]
L_SHIM.bbdos_create.restype = ctypes.c_void_p

L_SHIM.bbdos_cache_attention.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
]
L_SHIM.bbdos_cache_attention.restype = ctypes.c_int

L_SHIM.bbdos_destroy.argtypes = [ctypes.c_void_p]

L_SHIM.bbdos_stats.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
]
L_SHIM.bbdos_stats.restype = None


class GlyphWithLCache:
    def __init__(self, model_path, cache_size=512):
        self.model_path = model_path
        self.ctx = L_SHIM.bbdos_create(cache_size)
        self.lcache_hits = 0
        self.lcache_misses = 0

    def __del__(self):
        if self.ctx:
            L_SHIM.bbdos_destroy(self.ctx)

    def run_inference(self, prompt, num_tokens=100, threads=8):
        """Run inference through Glyph CLI, then cache attention internally."""

        # First, run the prompt through to get baseline
        cmd = [
            "./build/bin/llama-cli",
            "-m",
            self.model_path,
            "-p",
            f"User: {prompt} Assistant:",
            "-n",
            str(num_tokens),
            "--threads",
            str(threads),
            "--temp",
            "0.7",
        ]

        result = subprocess.run(
            cmd,
            cwd=os.path.dirname(self.model_path).replace(
                "/models/BitNet-b1.58-2B-4T", ""
            ),
            capture_output=True,
            text=True,
            timeout=120,
        )

        return result.stdout, result.stderr

    def cache_attention_layer(self, q, k, v):
        """Cache a single attention layer computation."""
        seq_len, head_dim = q.shape
        output = np.zeros_like(q)

        result = L_SHIM.bbdos_cache_attention(
            self.ctx,
            q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            seq_len,
            head_dim,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )

        if result == 1:  # cache hit
            self.lcache_hits += 1
        else:
            self.lcache_misses += 1

        return output

    def get_stats(self):
        hits = ctypes.c_int()
        misses = ctypes.c_int()
        evictions = ctypes.c_int()
        L_SHIM.bbdos_stats(
            self.ctx, ctypes.byref(hits), ctypes.byref(misses), ctypes.byref(evictions)
        )
        return {
            "internal_hits": hits.value,
            "internal_misses": misses.value,
            "internal_evictions": evictions.value,
            "layer_hits": self.lcache_hits,
            "layer_misses": self.lcache_misses,
        }


def benchmark_baseline():
    """Run baseline without L-Cache."""
    print("=== Glyph Baseline (No L-Cache) ===")

    model = "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"

    cmd = ["./build/bin/llama-bench", "-m", model, "-n", "50", "-p", "16", "-t", "8"]

    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__)
    )

    # Parse output
    for line in result.stdout.split("\n"):
        if "tg50" in line:
            parts = line.split("|")
            speed = parts[-1].strip()
            print(f"Token generation: {speed}")
            return speed

    return None


def benchmark_with_lcache():
    """Run with L-Cache attention layer simulation."""
    print("\n=== Glyph + L-Cache Attention Caching ===")

    # Simulate attention layer caching
    head_dim = 128
    seq_len = 16
    num_heads = 20
    num_layers = 30
    iterations = 5

    np.random.seed(42)

    # Test with same inputs (would hit cache in real integration)
    q = np.random.randn(seq_len * head_dim).astype(np.float32)
    k = np.random.randn(seq_len * head_dim).astype(np.float32)
    v = np.random.randn(seq_len * head_dim).astype(np.float32)

    glyph = GlyphWithLCache("models/test", cache_size=512)

    # First pass - cold
    start = time.time()
    for layer in range(num_layers):
        glyph.cache_attention_layer(q, k, v)
    cold_time = time.time() - start

    # Subsequent passes - warm
    start = time.time()
    for _ in range(iterations):
        for layer in range(num_layers):
            glyph.cache_attention_layer(q, k, v)
    warm_time = time.time() - start

    stats = glyph.get_stats()

    print(f"First pass (cold): {cold_time * 1000:.1f}ms")
    print(f"{iterations} passes (warm): {warm_time * 1000:.1f}ms")
    print(f"Per layer cached: {warm_time / iterations / num_layers * 1000:.2f}ms")
    print(f"L-Cache stats: {stats}")

    return stats


if __name__ == "__main__":
    os.chdir(os.path.expanduser("~/Projects/glyph"))

    print("Glyph + BBDOS L-Cache Integration Test")
    print("=" * 50)

    # Baseline
    baseline = benchmark_baseline()

    # With L-Cache simulation
    stats = benchmark_with_lcache()

    print("\n=== Summary ===")
    print(f"Baseline token gen: {baseline}")
    print(f"L-Cache layer hits: {stats['layer_hits']}")
    print(f"L-Cache layer misses: {stats['layer_misses']}")
    print(
        f"Cache hit rate: {100 * stats['layer_hits'] / (stats['layer_hits'] + stats['layer_misses']):.1f}%"
    )
