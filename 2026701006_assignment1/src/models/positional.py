"""
positional.py

From-scratch implementations of:
- Sinusoidal (absolute) positional encoding -> used by C1 (base), C3, C4, C5
- RoPE (Rotary Positional Embedding)         -> used by C2

Sinusoidal PE is added to embeddings once, before the encoder/decoder stack.
RoPE is applied inside attention, rotating Q and K at each layer -- it does
NOT get added to the embeddings up front. Keep that distinction in mind when
wiring C2 into the encoder/decoder blocks later: swapping "positional
encoding" for C2 means removing the SinusoidalPositionalEncoding add-on step
and instead calling RoPE's rotation inside MultiHeadAttention's forward,
right after the Q/K projections and before SDPA.
"""

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard "Attention Is All You Need" sinusoidal encoding.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Precomputed once up to max_len, then added to token embeddings.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model) -- broadcast over batch
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        Adds positional encoding for positions [0, seq_len) and applies dropout.
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class RotaryPositionalEmbedding(nn.Module):
    """
    RoPE: rotates pairs of dimensions in Q/K by an angle proportional to
    absolute position, such that the dot product of two rotated vectors
    depends only on their *relative* position (pos_i - pos_j), not their
    absolute positions.

    Applied to Q and K inside attention, per head, AFTER the head split and
    BEFORE the QK^T dot product. Not applied to V.

    d_k must be even (pairs of dimensions are rotated together).
    """

    def __init__(self, d_k: int, max_len: int = 5000, base: float = 10000.0):
        super().__init__()
        assert d_k % 2 == 0, "RoPE requires an even head dimension"
        self.d_k = d_k

        # theta_i = base^(-2i/d_k), for i in [0, d_k/2)
        inv_freq = 1.0 / (base ** (torch.arange(0, d_k, 2, dtype=torch.float) / d_k))  # (d_k/2,)

        t = torch.arange(max_len, dtype=torch.float)  # (max_len,)
        freqs = torch.outer(t, inv_freq)  # (max_len, d_k/2) -- angle per position per freq

        # precompute cos/sin tables, shape (max_len, d_k/2)
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())

    def _rotate_half(self, x):
        # split last dim in half, rotate pair-wise: (x1, x2) -> (-x2, x1)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return x1, x2

    def apply_rotary(self, x, seq_len):
        """
        x: (batch, n_heads, seq_len, d_k)
        Applies rotation to each adjacent-dim pair using cos/sin at the
        matching position.
        """
        cos = self.cos_cached[:seq_len, :].to(x.device)  # (seq_len, d_k/2)
        sin = self.sin_cached[:seq_len, :].to(x.device)  # (seq_len, d_k/2)

        x1, x2 = self._rotate_half(x)  # each (batch, n_heads, seq_len, d_k/2)

        # broadcast cos/sin over batch and heads
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, d_k/2)
        sin = sin.unsqueeze(0).unsqueeze(0)

        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        # interleave back to original layout (..., d_k)
        out = torch.empty_like(x)
        out[..., 0::2] = rotated_x1
        out[..., 1::2] = rotated_x2
        return out

    def forward(self, q, k):
        """
        q, k: (batch, n_heads, seq_len, d_k)
        Returns rotated (q, k). V is passed through untouched by the caller.
        """
        seq_len = q.size(2)
        q_rot = self.apply_rotary(q, seq_len)
        k_rot = self.apply_rotary(k, seq_len)
        return q_rot, k_rot


if __name__ == "__main__":
    torch.manual_seed(0)

    # --- Sinusoidal checks ---
    d_model, max_len = 16, 50
    pe_module = SinusoidalPositionalEncoding(d_model, max_len)

    x = torch.zeros(2, 10, d_model)  # zeros so output == PE itself
    out = pe_module(x)
    assert out.shape == (2, 10, d_model)

    # hand-check against the closed-form formula at position 0 and 5
    pos = 5
    i = 0  # first sin/cos pair
    expected_sin = math.sin(pos / (10000 ** (2 * i / d_model)))
    expected_cos = math.cos(pos / (10000 ** (2 * i / d_model)))
    assert abs(out[0, pos, 0].item() - expected_sin) < 1e-5
    assert abs(out[0, pos, 1].item() - expected_cos) < 1e-5
    print("Sinusoidal PE formula check passed at pos =", pos)

    # position 0 should be (sin(0), cos(0)) = (0, 1) for the first pair
    assert abs(out[0, 0, 0].item() - 0.0) < 1e-5
    assert abs(out[0, 0, 1].item() - 1.0) < 1e-5
    print("Sinusoidal PE position-0 check passed")

    # --- RoPE checks ---
    batch, n_heads, seq_len, d_k = 1, 1, 20, 8
    rope = RotaryPositionalEmbedding(d_k, max_len=50)

    q = torch.randn(batch, n_heads, seq_len, d_k)
    k = torch.randn(batch, n_heads, seq_len, d_k)
    q_rot, k_rot = rope(q, k)
    assert q_rot.shape == q.shape and k_rot.shape == k.shape
    print("RoPE shape check passed:", q_rot.shape)

    # rotation should preserve vector norm (it's an orthogonal transform)
    assert torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-4)
    print("RoPE norm-preservation check passed")

    # THE key property: dot product of q at pos i and k at pos j, after
    # rotation, should depend only on (i - j), not on absolute i, j.
    def rotated_dot(vec_q, vec_k, pos_q, pos_k, rope_module, d_k):
        vq = vec_q.view(1, 1, 1, d_k).clone()
        vk = vec_k.view(1, 1, 1, d_k).clone()
        cos_q = rope_module.cos_cached[pos_q].view(1, 1, 1, -1)
        sin_q = rope_module.sin_cached[pos_q].view(1, 1, 1, -1)
        cos_k = rope_module.cos_cached[pos_k].view(1, 1, 1, -1)
        sin_k = rope_module.sin_cached[pos_k].view(1, 1, 1, -1)

        q1, q2 = vq[..., 0::2], vq[..., 1::2]
        k1, k2 = vk[..., 0::2], vk[..., 1::2]

        rq1, rq2 = q1 * cos_q - q2 * sin_q, q1 * sin_q + q2 * cos_q
        rk1, rk2 = k1 * cos_k - k2 * sin_k, k1 * sin_k + k2 * cos_k

        rq = torch.empty_like(vq)
        rq[..., 0::2], rq[..., 1::2] = rq1, rq2
        rk = torch.empty_like(vk)
        rk[..., 0::2], rk[..., 1::2] = rk1, rk2

        return (rq * rk).sum().item()

    vec_q = torch.randn(d_k)
    vec_k = torch.randn(d_k)

    # same relative distance (5), different absolute positions
    dot_a = rotated_dot(vec_q, vec_k, pos_q=10, pos_k=5, rope_module=rope, d_k=d_k)
    dot_b = rotated_dot(vec_q, vec_k, pos_q=30, pos_k=25, rope_module=rope, d_k=d_k)
    dot_c = rotated_dot(vec_q, vec_k, pos_q=40, pos_k=35, rope_module=rope, d_k=d_k)

    assert abs(dot_a - dot_b) < 1e-3, f"{dot_a} vs {dot_b}"
    assert abs(dot_a - dot_c) < 1e-3, f"{dot_a} vs {dot_c}"
    print(f"RoPE relative-position property check passed: "
          f"dot@dist5 consistent across absolute positions ({dot_a:.4f}, {dot_b:.4f}, {dot_c:.4f})")

    # sanity: different relative distance should generally give a different dot product
    dot_diff_dist = rotated_dot(vec_q, vec_k, pos_q=10, pos_k=2, rope_module=rope, d_k=d_k)
    print(f"dot@dist8 = {dot_diff_dist:.4f} (expected to differ from dist5 dots above)")

    print("\nAll checks passed.")
