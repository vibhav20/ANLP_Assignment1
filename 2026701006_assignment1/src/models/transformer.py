"""
transformer.py

Assembles C1 (base config): full Encoder-Decoder Transformer using
- MultiHeadAttention (attention.py)
- SinusoidalPositionalEncoding (positional.py)
- LayerNorm (norm.py)
- Standard subword tokenization (handled in dataset.py, not here)

Pre-LN architecture throughout: x = x + Sublayer(Norm(x))

In the real project this file lives at src/models/transformer.py and imports
via `from .attention import MultiHeadAttention` etc. Here (scratch/testing
dir) it imports directly since all files are siblings.
"""

import math
import torch
import torch.nn as nn

from attention import MultiHeadAttention, MultiHeadAttentionRoPE, GroupedQueryAttention
from positional import SinusoidalPositionalEncoding
from norm import LayerNorm, RMSNorm


# ---------------------------------------------------------------------------
# Position-wise Feed-Forward Network
# ---------------------------------------------------------------------------
class PositionwiseFeedForward(nn.Module):
    """
    FFN(x) = Linear2(Activation(Linear1(x)))
    Applied independently to each position -- no mixing across seq_len here,
    that's attention's job.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


# ---------------------------------------------------------------------------
# Masking utilities
# ---------------------------------------------------------------------------
def make_padding_mask(seq, pad_idx):
    """
    seq: (batch, seq_len) of token ids
    Returns: (batch, 1, 1, seq_len) bool, True = keep (not padding)
    Broadcastable against (batch, n_heads, seq_len_q, seq_len_k) attn scores.
    """
    mask = (seq != pad_idx)  # (batch, seq_len)
    return mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)


def make_causal_mask(seq_len, device=None):
    """
    Returns: (1, 1, seq_len, seq_len) bool, True = keep (lower triangular,
    including diagonal). Position i can attend to positions <= i only.
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    return mask.unsqueeze(0).unsqueeze(0)


def make_decoder_self_mask(tgt_seq, pad_idx):
    """
    Combines causal masking with target-side padding masking for the
    decoder's self-attention.
    tgt_seq: (batch, tgt_len)
    Returns: (batch, 1, tgt_len, tgt_len) bool
    """
    batch, tgt_len = tgt_seq.shape
    causal = make_causal_mask(tgt_len, device=tgt_seq.device)  # (1,1,tgt_len,tgt_len)
    pad = make_padding_mask(tgt_seq, pad_idx)  # (batch,1,1,tgt_len)
    # broadcast AND: must be both non-padding AND not-future
    return causal & pad  # (batch,1,tgt_len,tgt_len)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
class EncoderLayer(nn.Module):
    """
    Pre-LN: x = x + SelfAttn(Norm(x)); x = x + FFN(Norm(x))

    attention_factory: callable(d_model, n_heads, dropout) -> attention module.
        Lets C1 pass MultiHeadAttention, C2 pass MultiHeadAttentionRoPE,
        C3 pass GroupedQueryAttention (via functools.partial with n_kv_heads
        pre-bound), without EncoderLayer needing to know which.
    norm_cls: LayerNorm (C1/C2/C3) or RMSNorm (C4).
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.0,
                 attention_factory=None, norm_cls=LayerNorm):
        super().__init__()
        attention_factory = attention_factory or (
            lambda d_model, n_heads, dropout: MultiHeadAttention(d_model, n_heads, dropout=dropout)
        )
        self.self_attn = attention_factory(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
        self.norm1 = norm_cls(d_model)
        self.norm2 = norm_cls(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        normed = self.norm1(x)
        attn_out, _ = self.self_attn(normed, normed, normed, mask=src_mask)
        x = x + self.dropout1(attn_out)

        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + self.dropout2(ffn_out)
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers,
                 max_len=5000, dropout=0.0, pad_idx=0,
                 attention_factory=None, norm_cls=LayerNorm,
                 use_sinusoidal_pe=True):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.d_model = d_model

        # C2 (RoPE) rotates inside attention instead -- skip the additive
        # sinusoidal PE in that case so position isn't encoded twice.
        self.use_sinusoidal_pe = use_sinusoidal_pe
        if use_sinusoidal_pe:
            self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len, dropout=dropout)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout=dropout,
                         attention_factory=attention_factory, norm_cls=norm_cls)
            for _ in range(n_layers)
        ])
        self.final_norm = norm_cls(d_model)  # final norm after last Pre-LN layer

    def forward(self, src, src_mask=None):
        """
        src: (batch, src_len) token ids
        """
        x = self.embedding(src) * math.sqrt(self.d_model)  # scale, as in the original paper
        if self.use_sinusoidal_pe:
            x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, src_mask=src_mask)

        return self.final_norm(x)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------
class DecoderLayer(nn.Module):
    """
    Pre-LN: x = x + SelfAttn(Norm(x));  x = x + CrossAttn(Norm(x), enc_out);  x = x + FFN(Norm(x))
    Self-attention is causally masked; cross-attention is not.
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.0,
                 attention_factory=None, norm_cls=LayerNorm):
        super().__init__()
        attention_factory = attention_factory or (
            lambda d_model, n_heads, dropout: MultiHeadAttention(d_model, n_heads, dropout=dropout)
        )
        # cross-attention always stays standard MHA, even for C2/C3 --
        # the ablation swaps SELF-attention's mechanism; cross-attention
        # (queries into a different sequence) keeps the base config's
        # cross-attention behavior so only one thing changes per Table 1.
        self.self_attn = attention_factory(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)
        self.norm1 = norm_cls(d_model)
        self.norm2 = norm_cls(d_model)
        self.norm3 = norm_cls(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_out, self_mask=None, cross_mask=None):
        normed = self.norm1(x)
        attn_out, _ = self.self_attn(normed, normed, normed, mask=self_mask)
        x = x + self.dropout1(attn_out)

        normed = self.norm2(x)
        # query = decoder state, key/value = encoder output (cross-attention)
        cross_out, _ = self.cross_attn(normed, enc_out, enc_out, mask=cross_mask)
        x = x + self.dropout2(cross_out)

        normed = self.norm3(x)
        ffn_out = self.ffn(normed)
        x = x + self.dropout3(ffn_out)
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers,
                 max_len=5000, dropout=0.0, pad_idx=0,
                 attention_factory=None, norm_cls=LayerNorm,
                 use_sinusoidal_pe=True):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.d_model = d_model

        self.use_sinusoidal_pe = use_sinusoidal_pe
        if use_sinusoidal_pe:
            self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len, dropout=dropout)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout=dropout,
                         attention_factory=attention_factory, norm_cls=norm_cls)
            for _ in range(n_layers)
        ])
        self.final_norm = norm_cls(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, enc_out, self_mask=None, cross_mask=None):
        """
        tgt: (batch, tgt_len) token ids (teacher-forced input, i.e. shifted right)
        enc_out: (batch, src_len, d_model)
        """
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        if self.use_sinusoidal_pe:
            x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, enc_out, self_mask=self_mask, cross_mask=cross_mask)

        x = self.final_norm(x)
        logits = self.output_proj(x)  # (batch, tgt_len, vocab_size)
        return logits


# ---------------------------------------------------------------------------
# Full Seq2Seq Transformer (C1)
# ---------------------------------------------------------------------------
class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=128, n_heads=4,
                 d_ff=512, n_enc_layers=3, n_dec_layers=3, max_len=5000,
                 dropout=0.1, src_pad_idx=0, tgt_pad_idx=0,
                 attention_factory=None, norm_cls=LayerNorm,
                 use_sinusoidal_pe=True):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        self.encoder = Encoder(
            src_vocab_size, d_model, n_heads, d_ff, n_enc_layers,
            max_len=max_len, dropout=dropout, pad_idx=src_pad_idx,
            attention_factory=attention_factory, norm_cls=norm_cls,
            use_sinusoidal_pe=use_sinusoidal_pe,
        )
        self.decoder = Decoder(
            tgt_vocab_size, d_model, n_heads, d_ff, n_dec_layers,
            max_len=max_len, dropout=dropout, pad_idx=tgt_pad_idx,
            attention_factory=attention_factory, norm_cls=norm_cls,
            use_sinusoidal_pe=use_sinusoidal_pe,
        )

    def forward(self, src, tgt):
        """
        src: (batch, src_len) token ids
        tgt: (batch, tgt_len) token ids -- teacher-forced decoder input
             (i.e. plaintext shifted right, starting with <bos>)
        Returns: logits (batch, tgt_len, tgt_vocab_size)
        """
        src_mask = make_padding_mask(src, self.src_pad_idx)          # (b,1,1,src_len)
        cross_mask = src_mask                                        # decoder attends over src, same padding mask
        tgt_mask = make_decoder_self_mask(tgt, self.tgt_pad_idx)      # (b,1,tgt_len,tgt_len)

        enc_out = self.encoder(src, src_mask=src_mask)
        logits = self.decoder(tgt, enc_out, self_mask=tgt_mask, cross_mask=cross_mask)
        return logits


VALID_CONFIGS = {"C1", "C2", "C3", "C4"}  # C5 (BLT) is a separate model class, not built here


def build_model(config_name, src_vocab_size, tgt_vocab_size,
                 d_model=128, n_heads=4, d_ff=512,
                 n_enc_layers=3, n_dec_layers=3, max_len=5000, dropout=0.1,
                 src_pad_idx=0, tgt_pad_idx=0, n_kv_heads=2):
    """
    Builds one of C1-C4 from Table 1, changing exactly the ONE component
    each config specifies, with every other hyperparameter identical.

        C1: sinusoidal PE, MHA,           LayerNorm  (base)
        C2: RoPE,           MHA,           LayerNorm
        C3: sinusoidal PE,  GQA,           LayerNorm
        C4: sinusoidal PE,  MHA,           RMSNorm

    n_kv_heads only matters for C3 (GQA) -- must divide n_heads evenly.
    """
    assert config_name in VALID_CONFIGS, f"build_model handles {VALID_CONFIGS}, got {config_name!r}"

    attention_factory = None  # None -> Seq2SeqTransformer defaults to plain MHA
    norm_cls = LayerNorm
    use_sinusoidal_pe = True

    if config_name == "C2":
        attention_factory = lambda d_model, n_heads, dropout: MultiHeadAttentionRoPE(
            d_model, n_heads, dropout=dropout, rope_max_len=max_len
        )
        use_sinusoidal_pe = False  # RoPE replaces additive PE, don't apply both

    elif config_name == "C3":
        attention_factory = lambda d_model, n_heads, dropout: GroupedQueryAttention(
            d_model, n_heads, n_kv_heads, dropout=dropout
        )

    elif config_name == "C4":
        norm_cls = RMSNorm

    return Seq2SeqTransformer(
        src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size,
        d_model=d_model, n_heads=n_heads, d_ff=d_ff,
        n_enc_layers=n_enc_layers, n_dec_layers=n_dec_layers,
        max_len=max_len, dropout=dropout,
        src_pad_idx=src_pad_idx, tgt_pad_idx=tgt_pad_idx,
        attention_factory=attention_factory, norm_cls=norm_cls,
        use_sinusoidal_pe=use_sinusoidal_pe,
    )


if __name__ == "__main__":
    torch.manual_seed(0)

    # ---- shape sanity check ----
    src_vocab, tgt_vocab = 256, 100  # e.g. byte vocab for cipher, small subword vocab for plaintext
    batch, src_len, tgt_len = 4, 12, 10
    d_model, n_heads, d_ff = 32, 4, 64

    model = Seq2SeqTransformer(
        src_vocab_size=src_vocab, tgt_vocab_size=tgt_vocab,
        d_model=d_model, n_heads=n_heads, d_ff=d_ff,
        n_enc_layers=2, n_dec_layers=2, dropout=0.0,
    )

    src = torch.randint(1, src_vocab, (batch, src_len))  # avoid 0 = pad_idx for this check
    tgt = torch.randint(1, tgt_vocab, (batch, tgt_len))

    logits = model(src, tgt)
    assert logits.shape == (batch, tgt_len, tgt_vocab)
    print("Full model shape check passed:", logits.shape)

    # ---- causal masking check: changing a FUTURE tgt token shouldn't change
    #      logits at an EARLIER position (teacher-forcing correctness) ----
    tgt2 = tgt.clone()
    tgt2[:, -1] = (tgt2[:, -1] + 1) % tgt_vocab  # perturb only the last position

    logits_orig = model(src, tgt)
    logits_pert = model(src, tgt2)

    # all positions except the last should be identical
    assert torch.allclose(logits_orig[:, :-1, :], logits_pert[:, :-1, :], atol=1e-5), \
        "causal mask leak: earlier positions changed when a later token was perturbed"
    print("Causal masking (no future leakage) check passed")

    # ---- tiny-batch overfit check (synthetic data, gate check for the architecture) ----
    print("\nRunning tiny-batch overfit check...")
    torch.manual_seed(1)

    small_batch, small_src_len, small_tgt_len = 8, 10, 8
    PAD_IDX = 0

    toy_src = torch.randint(1, src_vocab, (small_batch, small_src_len))
    toy_tgt_full = torch.randint(1, tgt_vocab, (small_batch, small_tgt_len + 1))
    # decoder input = shifted right (drop last), labels = shifted left (drop first)
    toy_tgt_in = toy_tgt_full[:, :-1]
    toy_tgt_labels = toy_tgt_full[:, 1:]

    overfit_model = Seq2SeqTransformer(
        src_vocab_size=src_vocab, tgt_vocab_size=tgt_vocab,
        d_model=64, n_heads=4, d_ff=256,
        n_enc_layers=2, n_dec_layers=2, dropout=0.0,
        src_pad_idx=PAD_IDX, tgt_pad_idx=PAD_IDX,
    )
    optimizer = torch.optim.Adam(overfit_model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    n_steps = 300
    for step in range(n_steps):
        optimizer.zero_grad()
        logits = overfit_model(toy_src, toy_tgt_in)  # (batch, tgt_len, vocab)
        loss = criterion(logits.reshape(-1, tgt_vocab), toy_tgt_labels.reshape(-1))
        loss.backward()
        optimizer.step()
        if step % 50 == 0 or step == n_steps - 1:
            print(f"  step {step:4d}  loss {loss.item():.4f}")

    assert loss.item() < 0.1, f"tiny-batch overfit did not converge, final loss = {loss.item():.4f}"
    print("Tiny-batch overfit check passed (architecture can learn).")

    print("\nAll checks passed.")