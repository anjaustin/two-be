"""
BitNet End-to-End Inference via APU Router

Demonstrates full transformer block using composed opcodes.
"""

import numpy as np
from bbdos_apu import BBDOS_APU


def quantize_to_int8(x: np.ndarray) -> np.ndarray:
    """Quantize float32 to int8 for BitNet input."""
    scale = np.abs(x).max() / 127.0
    return np.clip(np.round(x / scale), -128, 127).astype(np.int8)


def dequantize(x: np.ndarray, scale: float) -> np.ndarray:
    """Dequantize int8 back to float32."""
    return x.astype(np.float32) * scale


def pack_to_ternary(weights: np.ndarray) -> np.ndarray:
    """Convert float weights to packed 2-bit ternary."""
    flat = weights.flatten()
    ternary = np.sign(flat)
    ternary[ternary == 0] = 1

    packed = np.zeros((len(ternary) + 3) // 4, dtype=np.uint8)
    for i in range(len(ternary)):
        byte_idx = i // 4
        bit_idx = (i % 4) * 2
        code = 1 if ternary[i] > 0 else 2
        packed[byte_idx] |= code << (bit_idx)

    return packed


def bitnet_block(
    apu: BBDOS_APU,
    x: np.ndarray,
    weights_q: np.ndarray,
    weights_k: np.ndarray,
    weights_v: np.ndarray,
    weights_ffn: np.ndarray,
    head_dim: int = 64,
) -> np.ndarray:
    """
    Single BitNet transformer block (full attention).

    x: [batch, seq_len, hidden_dim]
    weights_q/k/v: packed ternary weights for Q/K/V projections
    weights_ffn: packed ternary weights for FFN
    head_dim: attention dimension (must divide hidden_dim)
    """
    batch, seq_len, hidden_dim = x.shape
    num_heads = hidden_dim // head_dim

    x_flat = x.reshape(-1, hidden_dim).astype(np.float32)

    # Project to Q, K, V
    q = apu.exec(
        "BitLinear", [x_flat, weights_q], [batch * seq_len, hidden_dim, hidden_dim]
    )
    k = apu.exec(
        "BitLinear", [x_flat, weights_k], [batch * seq_len, hidden_dim, hidden_dim]
    )
    v = apu.exec(
        "BitLinear", [x_flat, weights_v], [batch * seq_len, hidden_dim, hidden_dim]
    )

    # Reshape to [batch*heads, seq, head]
    q = q.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)

    # Attention per head
    attn_out = np.zeros((batch, seq_len, num_heads, head_dim), dtype=np.float32)
    for h in range(num_heads):
        q_h = q[:, h, :, :].reshape(seq_len, head_dim)
        k_h = k[:, h, :, :].reshape(seq_len, head_dim)
        v_h = v[:, h, :, :].reshape(seq_len, head_dim)

        out_h = apu.exec("BitAttention", [q_h, k_h, v_h], [seq_len, head_dim])
        attn_out[:, :, h, :] = out_h.reshape(seq_len, head_dim)

    # Merge heads
    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq_len, hidden_dim)

    # FFN
    ffn_out = apu.exec(
        "BitLinear", [attn_out, weights_ffn], [batch * seq_len, hidden_dim, hidden_dim]
    )
    ffn_out = ffn_out.reshape(batch, seq_len, hidden_dim)

    return ffn_out


def demo():
    """Run a simple demo."""
    apu = BBDOS_APU(cache_size=256)

    batch, seq_len, hidden_dim = 1, 4, 16
    head_dim = 8

    print("=== BitNet End-to-End Demo ===")
    print(f"Config: batch={batch}, seq_len={seq_len}, hidden={hidden_dim}")
    print(f"        head_dim={head_dim}")

    np.random.seed(42)
    x = np.random.randn(batch, seq_len, hidden_dim).astype(np.float32)
    print(f"\nInput shape: {x.shape}")

    w_q = pack_to_ternary(np.random.randn(hidden_dim, hidden_dim))
    w_k = pack_to_ternary(np.random.randn(hidden_dim, hidden_dim))
    w_v = pack_to_ternary(np.random.randn(hidden_dim, hidden_dim))
    w_ffn = pack_to_ternary(np.random.randn(hidden_dim, hidden_dim))

    output = bitnet_block(apu, x, w_q, w_k, w_v, w_ffn, head_dim)

    print(f"Output shape: {output.shape}")
    print(f"Output stats: mean={output.mean():.4f}, std={output.std():.4f}")
    print(f"Cache stats: {apu.stats()}")

    output2 = bitnet_block(apu, x, w_q, w_k, w_v, w_ffn, head_dim)
    print(f"\nSecond run cache: {apu.stats()}")

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    demo()
