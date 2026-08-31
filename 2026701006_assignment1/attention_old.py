"""
attention.py

From-scratch implementations of:
- Scaled Dot-Product Attention
- Multi-Head Attention (MHA)          -> used by C1 (base), C2, C4, C5
- Grouped-Query Attention (GQA)       -> used by C3

No nn.Transformer / nn.MultiheadAttention used anywhere.
"""

import math
import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    """
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Shapes (batch-first, per-head already split out by the caller):
        Q: (batch, n_heads, seq_len_q, d_k)
        K: (batch, n_heads, seq_len_k, d_k)
        V: (batch, n_heads, seq_len_k, d_v)
        mask: (batch, 1, seq_len_q, seq_len_k) or broadcastable, bool
              True = keep, False = mask out (set to -inf before softmax)
    """

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)

        # (batch, n_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # mask: True = keep. Fill False positions with -inf so softmax -> 0.
            scores = scores.masked_fill(mask == False, float("-inf"))  # noqa: E712

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # (batch, n_heads, seq_len_q, d_v)
        output = torch.matmul(attn_weights, V)
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Attention.

    d_model must be divisible by n_heads.
    Used for C1 (base), and reused unchanged by C2 (RoPE swap happens
    at the positional-encoding level, not here), C4 (RMSNorm swap is
    outside this module), and C5's global transformer.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout=dropout)

    def _split_heads(self, x):
        # (batch, seq_len, d_model) -> (batch, n_heads, seq_len, d_k)
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.n_heads, self.d_k)
        return x.transpose(1, 2)

    def _merge_heads(self, x):
        # (batch, n_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        batch, n_heads, seq_len, d_k = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq_len, n_heads * d_k)

    def forward(self, query, key, value, mask=None):
        """
        query: (batch, seq_len_q, d_model)
        key, value: (batch, seq_len_k, d_model)
        mask: (batch, 1, seq_len_q, seq_len_k) or broadcastable, bool
        """
        Q = self._split_heads(self.w_q(query))
        K = self._split_heads(self.w_k(key))
        V = self._split_heads(self.w_v(value))

        attn_out, attn_weights = self.attention(Q, K, V, mask=mask)

        merged = self._merge_heads(attn_out)
        output = self.w_o(merged)
        return output, attn_weights


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention: n_heads query heads, n_kv_heads key/value heads,
    with n_heads % n_kv_heads == 0. Each group of (n_heads / n_kv_heads)
    query heads shares one K/V head.

    Used for C3 (attention-mechanism ablation). Everything else about the
    block (norm, FFN, positional encoding) stays identical to C1.
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_k = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads  # how many Q heads share one KV head

        self.w_q = nn.Linear(d_model, n_heads * self.d_k)
        # K/V projected to a smaller number of heads
        self.w_k = nn.Linear(d_model, n_kv_heads * self.d_k)
        self.w_v = nn.Linear(d_model, n_kv_heads * self.d_k)
        self.w_o = nn.Linear(n_heads * self.d_k, d_model)

        self.attention = ScaledDotProductAttention(dropout=dropout)

    def _split_heads(self, x, n_heads):
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, n_heads, self.d_k)
        return x.transpose(1, 2)  # (batch, n_heads, seq_len, d_k)

    def _merge_heads(self, x):
        batch, n_heads, seq_len, d_k = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq_len, n_heads * d_k)

    def _repeat_kv(self, x):
        # (batch, n_kv_heads, seq_len, d_k) -> (batch, n_heads, seq_len, d_k)
        batch, n_kv_heads, seq_len, d_k = x.shape
        x = x.unsqueeze(2).expand(batch, n_kv_heads, self.n_rep, seq_len, d_k)
        return x.reshape(batch, n_kv_heads * self.n_rep, seq_len, d_k)

    def forward(self, query, key, value, mask=None):
        Q = self._split_heads(self.w_q(query), self.n_heads)          # (b, n_heads, sq, d_k)
        K = self._split_heads(self.w_k(key), self.n_kv_heads)         # (b, n_kv_heads, sk, d_k)
        V = self._split_heads(self.w_v(value), self.n_kv_heads)       # (b, n_kv_heads, sk, d_k)

        K = self._repeat_kv(K)  # (b, n_heads, sk, d_k)
        V = self._repeat_kv(V)  # (b, n_heads, sk, d_k)

        attn_out, attn_weights = self.attention(Q, K, V, mask=mask)

        merged = self._merge_heads(attn_out)
        output = self.w_o(merged)
        return output, attn_weights


if __name__ == "__main__":
    # --- quick shape / sanity checks, not a full test suite ---
    torch.manual_seed(0)

    batch, seq_len, d_model, n_heads = 2, 5, 16, 4

    x = torch.randn(batch, seq_len, d_model)

    # SDPA in isolation
    sdpa = ScaledDotProductAttention()
    Q = torch.randn(batch, n_heads, seq_len, d_model // n_heads)
    K = torch.randn(batch, n_heads, seq_len, d_model // n_heads)
    V = torch.randn(batch, n_heads, seq_len, d_model // n_heads)
    out, w = sdpa(Q, K, V)
    assert out.shape == (batch, n_heads, seq_len, d_model // n_heads)
    assert torch.allclose(w.sum(dim=-1), torch.ones(batch, n_heads, seq_len), atol=1e-5)
    print("SDPA shape/softmax check passed:", out.shape)

    # causal mask check: position i should not attend to position j > i
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    causal = causal.unsqueeze(0).unsqueeze(0)  # (1,1,seq_len,seq_len)
    out_m, w_m = sdpa(Q, K, V, mask=causal)
    upper_tri_weights = w_m[..., torch.triu(torch.ones(seq_len, seq_len), diagonal=1) == 1]
    assert torch.allclose(upper_tri_weights, torch.zeros_like(upper_tri_weights), atol=1e-6)
    print("Causal mask check passed: future positions get ~0 weight")

    # MHA
    mha = MultiHeadAttention(d_model, n_heads)
    out, _ = mha(x, x, x)
    assert out.shape == (batch, seq_len, d_model)
    print("MHA shape check passed:", out.shape)

    # MHA with causal mask
    out, _ = mha(x, x, x, mask=causal)
    assert out.shape == (batch, seq_len, d_model)
    print("MHA + causal mask shape check passed:", out.shape)

    # GQA
    n_kv_heads = 2
    gqa = GroupedQueryAttention(d_model, n_heads, n_kv_heads)
    out, _ = gqa(x, x, x)
    assert out.shape == (batch, seq_len, d_model)
    print("GQA shape check passed:", out.shape)

    print("\nAll checks passed.")
