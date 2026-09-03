"""
Sinusoidal (absolute) positional encoding ->  C1, C3, C4, C5
RoPE (Rotary Positional Embedding)         -> C2

Sinusoidal PE is added to embeddings once before the encoder/decoder stack.
RoPE is applied inside attention, rotating Q and K at each layer, does
NOT get added to the embeddings up front. 
"""
import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):

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
        Returns rotated (q, k). V is passed untouched
        """
        seq_len = q.size(2)
        q_rot = self.apply_rotary(q, seq_len)
        k_rot = self.apply_rotary(k, seq_len)
        return q_rot, k_rot