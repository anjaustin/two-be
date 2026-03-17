"""
BitNet Generation Test

Test full generation with prompt: "Hypothetically, might reflective recursion be a function of cognition?"
"""

import numpy as np
from bbdos_apu import BBDOS_APU
import time


def pack_to_ternary(weights):
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


def simple_embed(tokens, vocab_size, embed_dim):
    """Simple learned embedding."""
    np.random.seed(42)
    embeddings = np.random.randn(vocab_size, embed_dim).astype(np.float32) * 0.1
    return embeddings[tokens]


def bitnet_block(apu, x, w_q, w_k, w_v, w_ffn, layernorm_g, layernorm_b, head_dim):
    batch, seq_len, hidden = x.shape
    num_heads = hidden // head_dim

    x_flat = x.reshape(-1, hidden).astype(np.float32)

    q = apu.exec("BitLinear", [x_flat, w_q], [batch * seq_len, hidden, hidden]).reshape(
        batch, seq_len, hidden
    )
    k = apu.exec("BitLinear", [x_flat, w_k], [batch * seq_len, hidden, hidden]).reshape(
        batch, seq_len, hidden
    )
    v = apu.exec("BitLinear", [x_flat, w_v], [batch * seq_len, hidden, hidden]).reshape(
        batch, seq_len, hidden
    )

    q = q.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)

    attn_out = np.zeros((batch, num_heads, seq_len, head_dim), dtype=np.float32)
    for h in range(num_heads):
        q_h = q[:, h, :, :]
        k_h = k[:, h, :, :]
        v_h = v[:, h, :, :]
        out_h = apu.exec("BitAttention", [q_h, k_h, v_h], [seq_len, head_dim])
        attn_out[:, h, :, :] = out_h.reshape(seq_len, head_dim)

    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq_len, hidden)

    ffn_out = apu.exec(
        "BitLinear", [attn_out, w_ffn], [batch * seq_len, hidden, hidden]
    ).reshape(batch, seq_len, hidden)

    return ffn_out


def generate(apu, model, prompt, max_new_tokens=50, temperature=0.8):
    """Generate text from prompt."""
    vocab_size, embed_dim = model.vocab_size, model.embed_dim
    num_layers = model.num_layers
    head_dim = 32

    tokens = [ord(c) % vocab_size for c in prompt]

    hidden = embed_dim

    x = simple_embed(tokens, vocab_size, embed_dim).reshape(1, len(tokens), embed_dim)

    for layer in range(num_layers):
        x = bitnet_block(
            apu,
            x,
            model.weights[f"layer{layer}_w_q"],
            model.weights[f"layer{layer}_w_k"],
            model.weights[f"layer{layer}_w_v"],
            model.weights[f"layer{layer}_w_ffn"],
            model.weights[f"layer{layer}_ln_g"],
            model.weights[f"layer{layer}_ln_b"],
            head_dim,
        )

    logits = x[0, -1, :]

    logits = apu.exec(
        "BitLinear",
        [logits.reshape(1, hidden), model.weights["output_proj"]],
        [1, hidden, vocab_size],
    )

    generated = []
    for _ in range(max_new_tokens):
        logits = np.nan_to_num(logits, nan=0.0, posinf=1e10, neginf=-1e10)
        probs = np.exp(logits - np.max(logits))
        probs = probs / np.sum(probs)
        probs = np.nan_to_num(probs, nan=1.0 / vocab_size)
        if temperature > 0:
            probs = probs ** (1.0 / temperature)
            probs = probs / probs.sum()

        next_token = np.random.choice(vocab_size, p=probs)
        generated.append(chr(next_token) if next_token < 128 else "?")

        x = simple_embed([next_token], vocab_size, embed_dim).reshape(1, 1, embed_dim)

        for layer in range(num_layers):
            x = bitnet_block(
                apu,
                x,
                model.weights[f"layer{layer}_w_q"],
                model.weights[f"layer{layer}_w_k"],
                model.weights[f"layer{layer}_w_v"],
                model.weights[f"layer{layer}_w_ffn"],
                model.weights[f"layer{layer}_ln_g"],
                model.weights[f"layer{layer}_ln_b"],
                head_dim,
            )

        logits = x[0, -1, :]
        logits = apu.exec(
            "BitLinear",
            [logits.reshape(1, hidden), model.weights["output_proj"]],
            [1, hidden, vocab_size],
        )

    return "".join(generated)


class BitNetModel:
    def __init__(self, vocab_size, embed_dim, num_layers):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.weights = {}

        np.random.seed(42)
        hidden = embed_dim

        for layer in range(num_layers):
            self.weights[f"layer{layer}_w_q"] = pack_to_ternary(
                np.random.randn(hidden, hidden) * 0.1
            )
            self.weights[f"layer{layer}_w_k"] = pack_to_ternary(
                np.random.randn(hidden, hidden) * 0.1
            )
            self.weights[f"layer{layer}_w_v"] = pack_to_ternary(
                np.random.randn(hidden, hidden) * 0.1
            )
            self.weights[f"layer{layer}_w_ffn"] = pack_to_ternary(
                np.random.randn(hidden, hidden) * 0.1
            )
            self.weights[f"layer{layer}_ln_g"] = np.ones(hidden, dtype=np.float32)
            self.weights[f"layer{layer}_ln_b"] = np.zeros(hidden, dtype=np.float32)

        self.weights["output_proj"] = pack_to_ternary(
            np.random.randn(hidden, vocab_size) * 0.1
        )


def main():
    prompt = "Hypothetically, might reflective recursion be a function of cognition?"
    max_tokens = 100

    print("=== BitNet Generation Test ===")
    print(f'Prompt: "{prompt}"')
    print(f"Max tokens: {max_tokens}")
    print()

    vocab_size, embed_dim, num_layers = 256, 128, 4
    model = BitNetModel(vocab_size, embed_dim, num_layers)
    apu = BBDOS_APU(cache_size=512)

    print("Generating...")
    start = time.time()
    output = generate(apu, model, prompt, max_tokens)
    elapsed = time.time() - start

    print(f"\n=== Generated Text ===")
    print(f"Prompt: {prompt}")
    print(f"Output: {output}")
    print(f"\n=== Stats ===")
    print(f"Time: {elapsed:.2f}s")
    print(f"Tokens: {len(output)}")
    print(f"Tokens/sec: {len(output) / elapsed:.1f}")
    print(f"Cache: {apu.stats()}")


if __name__ == "__main__":
    main()
