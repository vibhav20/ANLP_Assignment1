import math
import torch
import torch.nn as nn
from .positional import RotaryPositionalEmbedding

class ScaledDotProductAttention(nn.Module):
    """
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Shape -> (batch, n_heads, seq_len, d_k)
    d_k = d_model / n_heads; d_model is the vector representation of each token
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
            scores = scores.masked_fill(mask == False, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # (batch, n_heads, seq_len_q, d_v)
        output = torch.matmul(attn_weights, V)
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    C1,C2,C4,C5
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
    Each group of (n_heads / n_kv_heads) query heads shares one K/V head.

    C3
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_k = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads  # num of Q heads sharing one KV head

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


class MultiHeadAttentionRoPE(nn.Module):
    """
    Used by C2.
    Structurally identical to MultiHeadAttention except for the rotation
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, rope_max_len: int = 5000):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.rope = RotaryPositionalEmbedding(self.d_k, max_len=rope_max_len)
        self.attention = ScaledDotProductAttention(dropout=dropout)

    def _split_heads(self, x):
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.n_heads, self.d_k)
        return x.transpose(1, 2)

    def _merge_heads(self, x):
        batch, n_heads, seq_len, d_k = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, seq_len, n_heads * d_k)

    def forward(self, query, key, value, mask=None):
        Q = self._split_heads(self.w_q(query))
        K = self._split_heads(self.w_k(key))
        V = self._split_heads(self.w_v(value))

        # rotate Q and K independently, using each side's own sequence
        Q = self.rope.apply_rotary(Q, Q.size(2))
        K = self.rope.apply_rotary(K, K.size(2))

        attn_out, attn_weights = self.attention(Q, K, V, mask=mask)

        merged = self._merge_heads(attn_out)
        output = self.w_o(merged)
        return output, attn_weights
