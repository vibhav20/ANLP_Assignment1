"""
blt.py

Simplified Byte Latent Transformer (BLT) for C5 -- the token-free config.
No vocabulary, no BPE: raw bytes in, raw bytes out.

PATCHING: entropy-based and DYNAMIC, not fixed-width. A lightweight
byte-level n-gram (Markov) model estimates next-byte entropy at every
position; a new patch boundary starts wherever that estimated entropy
exceeds a threshold (high entropy = the model is "surprised", i.e. novel/
unpredictable content -- a natural place to start a new patch, per BLT's
core idea). This is a frozen, non-differentiable PREPROCESSING step done
once per line at data-loading time (see dataset.py), not inside the
model's forward pass -- exactly mirroring how the original BLT paper uses
a separately-trained entropy estimator decoupled from the main model's
gradients. "Simplified" here means a plain n-gram model instead of a
trained neural LM for entropy estimation -- the dynamic-boundary IDEA is
retained, the scale of the estimator is what's cut down.

Pipeline:
    raw source bytes + precomputed patch_ids
        -> LocalEncoder (byte self-attention + membership-based
                          cross-attention pooling into DYNAMIC patches)
        -> source patch representations
        -> PatchEncoder (global transformer, patch-level self-attention)
        -> encoded source patches
    raw target bytes (teacher-forced input) + precomputed patch_ids
        -> LocalEncoder (separate instance, target side, causal)
        -> target patch representations
        -> PatchDecoder (global transformer, causal patch-level self-attn
                          + cross-attn to encoded source patches)
        -> decoded target patch latents
        -> LocalDecoder (causal byte self-attn + cross-attn to patch
                          latents, patch-causality-masked using the
                          ACTUAL per-byte patch assignment)
        -> byte-level logits

Reuses EncoderLayer / DecoderLayer / MultiHeadAttention / LayerNorm /
SinusoidalPositionalEncoding as-is from the already-verified C1 modules.
"""

import math
from collections import defaultdict, Counter

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .positional import SinusoidalPositionalEncoding
from .norm import LayerNorm
from ..transformer import EncoderLayer, DecoderLayer, make_padding_mask, make_causal_mask


# ---------------------------------------------------------------------------
# Byte-level vocabulary: raw bytes 0-255 plus 3 special ids outside that range.
# Standard learned embedding over this vocabulary (nn.Embedding below) --
# no n-gram hash embeddings, per the assignment clarification.
# ---------------------------------------------------------------------------
BOS_BYTE = 256
EOS_BYTE = 257
PAD_BYTE = 258
BYTE_VOCAB_SIZE = 259


def bits_to_byte_ids(bit_string: str):
    """
    Cipher line ('0'/'1' string, length % 8 == 0) -> [BOS, byte, ..., EOS].
    Every 8 bits is grouped into ONE integer byte value (0-255) -- this is
    the underlying binary data, not the text characters '0'/'1'.
    """
    assert len(bit_string) % 8 == 0, "cipher bit length must be a multiple of 8"
    raw_bytes = [int(bit_string[i:i + 8], 2) for i in range(0, len(bit_string), 8)]
    return [BOS_BYTE] + raw_bytes + [EOS_BYTE]


def text_to_byte_ids(text: str):
    """Plaintext line -> [BOS, byte, byte, ..., EOS] (UTF-8 byte values)."""
    raw_bytes = list(text.encode("utf-8"))
    return [BOS_BYTE] + raw_bytes + [EOS_BYTE]


def byte_ids_to_text(ids):
    """Inverse of text_to_byte_ids -- strips specials, decodes UTF-8 (lossy-safe)."""
    raw = [i for i in ids if 0 <= i <= 255]
    return bytes(raw).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Lightweight byte-level n-gram (Markov) entropy estimator.
# Pure counting, no gradients, no neural net -- runs on CPU trivially even
# for thousands of lines. This is the "simplified" stand-in for BLT's
# trained entropy LM: same purpose (score how surprising the next byte is
# given recent context), much smaller mechanism.
# ---------------------------------------------------------------------------
class ByteNgramEntropyModel:
    """
    order: how many preceding bytes form the conditioning context.
    Trained via plain frequency counting on a corpus of byte-id sequences
    (fit ONLY on the train split -- same leakage discipline as the BPE
    tokenizers in dataset.py).
    """

    def __init__(self, order: int = 3, vocab_size: int = BYTE_VOCAB_SIZE, smoothing: float = 1.0):
        self.order = order
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.counts = defaultdict(Counter)   # context tuple -> Counter(next_byte -> count)
        self.unigram_counts = Counter()
        self.total_unigram = 0
        self._dist_cache = {}                # memoized smoothed distributions per context

    def train(self, sequences):
        """sequences: list of list[int] (byte ids, including BOS/EOS)."""
        for seq in sequences:
            for i in range(len(seq)):
                context = tuple(seq[max(0, i - self.order):i])
                self.counts[context][seq[i]] += 1
                self.unigram_counts[seq[i]] += 1
                self.total_unigram += 1
        self._dist_cache = {}  # invalidate cache if retrained

    def _distribution(self, context):
        """Returns a length-vocab_size probability list for P(next | context),
        back-off to the unigram distribution for unseen contexts, with
        additive (Laplace) smoothing throughout."""
        if context in self._dist_cache:
            return self._dist_cache[context]

        counter = self.counts.get(context)
        if not counter:
            total = self.total_unigram + self.smoothing * self.vocab_size
            probs = [(self.unigram_counts.get(v, 0) + self.smoothing) / total
                     for v in range(self.vocab_size)]
        else:
            total = sum(counter.values()) + self.smoothing * self.vocab_size
            probs = [(counter.get(v, 0) + self.smoothing) / total
                     for v in range(self.vocab_size)]

        self._dist_cache[context] = probs
        return probs

    def entropy_at(self, seq, i):
        """Shannon entropy (bits) of P(seq[i] | seq[i-order:i])."""
        context = tuple(seq[max(0, i - self.order):i])
        probs = self._distribution(context)
        return -sum(p * math.log2(p) for p in probs if p > 0)

    def entropy_profile(self, seq):
        """Returns list[float], one entropy value per position in seq."""
        return [self.entropy_at(seq, i) for i in range(len(seq))]


def estimate_entropy_threshold(entropy_model, sequences, quantile: float = 0.5):
    """
    Picks a global entropy threshold from the empirical distribution of
    per-position entropies across a sample of sequences (default: median).
    A boundary fires whenever entropy exceeds this value -- using the
    median means roughly half of byte positions become candidate boundary
    points before the max_patch_size cap and natural clustering kick in.
    """
    all_entropies = []
    for seq in sequences:
        all_entropies.extend(entropy_model.entropy_profile(seq))
    all_entropies.sort()
    idx = min(int(len(all_entropies) * quantile), len(all_entropies) - 1)
    return all_entropies[idx]


def compute_patch_boundaries(entropy_profile, threshold: float, max_patch_size: int = 16):
    """
    Converts a per-position entropy profile into per-position PATCH IDS
    (0-indexed, monotonically non-decreasing left-to-right).

    Rule: start a new patch whenever entropy at this position exceeds the
    threshold (the model is "surprised" -- treat this as the start of a
    new chunk of content), OR the current patch has already reached
    max_patch_size (a safety cap so a long low-entropy run, e.g. repeated
    bytes, never produces one pathologically huge patch).
    """
    patch_ids = []
    current_patch = 0
    current_len = 0
    for i, h in enumerate(entropy_profile):
        if i > 0 and (h > threshold or current_len >= max_patch_size):
            current_patch += 1
            current_len = 0
        patch_ids.append(current_patch)
        current_len += 1
    return patch_ids


# ---------------------------------------------------------------------------
# Patch-level masking helpers
# ---------------------------------------------------------------------------
def make_patch_padding_mask(patch_valid: torch.Tensor):
    """(batch, num_patches) -> (batch, 1, 1, num_patches), True = keep."""
    return patch_valid.unsqueeze(1).unsqueeze(2)


def make_patch_self_mask(patch_valid: torch.Tensor):
    """
    Causal + padding mask at PATCH granularity for the global decoder's
    self-attention. Patch ids are assigned in strictly non-decreasing
    left-to-right order by compute_patch_boundaries, so causal masking
    over PATCH INDEX still corresponds exactly to temporal order.
    Returns: (batch, 1, num_patches, num_patches) bool.
    """
    batch, num_patches = patch_valid.shape
    causal = make_causal_mask(num_patches, device=patch_valid.device)  # (1,1,P,P)
    pad = make_patch_padding_mask(patch_valid)  # (batch,1,1,P)
    return causal & pad


def make_local_decoder_cross_mask(byte_patch_ids: torch.Tensor, patch_valid: torch.Tensor):
    """
    For the LOCAL byte decoder's cross-attention into the global decoder's
    patch latents: byte position i may only attend to patch latents at
    index <= byte_patch_ids[b, i] -- its own (dynamically-assigned) patch
    and earlier ones. Unlike fixed-size patching, this can't be derived by
    floor division -- it uses the ACTUAL per-byte patch assignment tensor.
    Returns: (batch, 1, byte_len, num_patches) bool.
    """
    batch, byte_len = byte_patch_ids.shape
    num_patches = patch_valid.size(1)
    device = byte_patch_ids.device

    patch_range = torch.arange(num_patches, device=device).view(1, 1, -1)         # (1,1,P)
    causal_byte_to_patch = (patch_range <= byte_patch_ids.unsqueeze(-1))          # (batch,byte_len,P)
    causal_byte_to_patch = causal_byte_to_patch.unsqueeze(1)                      # (batch,1,byte_len,P)

    pad = make_patch_padding_mask(patch_valid)  # (batch,1,1,P)
    return causal_byte_to_patch & pad  # broadcasts -> (batch,1,byte_len,P)


def build_patch_membership(byte_ids: torch.Tensor, patch_ids: torch.Tensor, pad_id: int = PAD_BYTE):
    """
    byte_ids, patch_ids: (batch, seq_len) -- patch_ids gives each byte
    position's dynamically-assigned patch index (padding positions can
    hold any placeholder value; they're excluded via the byte-validity
    check below regardless).

    Returns:
        membership:  (batch, num_patches, seq_len) bool -- True if byte j
                     belongs to patch p AND is a real (non-pad) byte.
        patch_valid: (batch, num_patches) bool -- True if patch p has at
                     least one real byte.
    num_patches = the batch-wide max patch id + 1, so every example is
    padded to the same number of patch "slots".
    """
    batch, seq_len = byte_ids.shape
    num_patches = int(patch_ids.max().item()) + 1 if patch_ids.numel() > 0 else 1

    byte_valid = (byte_ids != pad_id)  # (batch, seq_len)
    patch_range = torch.arange(num_patches, device=byte_ids.device).view(1, -1, 1)  # (1,P,1)
    membership = (patch_ids.unsqueeze(1) == patch_range) & byte_valid.unsqueeze(1)  # (batch,P,seq_len)

    patch_valid = membership.any(dim=-1)  # (batch, num_patches)
    return membership, patch_valid


# ---------------------------------------------------------------------------
# Local Encoder: bytes -> DYNAMIC patch representations
# ---------------------------------------------------------------------------
class LocalEncoder(nn.Module):
    """
    1. Byte embedding + sinusoidal PE
    2. A few self-attention layers over the raw byte sequence (causal on
       the target side, bidirectional on the source side -- see the
       `causal` flag).
    3. Membership-based cross-attention pooling: ONE learnable query,
       broadcast across all patch "slots", cross-attends over the whole
       byte sequence with a per-patch MEMBERSHIP MASK (which bytes belong
       to this patch) rather than a fixed contiguous reshape -- this is
       what makes patch sizes genuinely dynamic/variable instead of a
       fixed stride.
    """

    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers,
                 pad_id=PAD_BYTE, max_len=8000, dropout=0.0, causal=False):
        """
        causal: if True, byte self-attention is masked so byte i only sees
            bytes <= i (used for the TARGET-side LocalEncoder -- required
            regardless of fixed vs dynamic patching, since bidirectional
            self-attention would let a later byte's information leak into
            an earlier patch's representation).
        """
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.causal = causal

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len, dropout=dropout)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout=dropout) for _ in range(n_layers)
        ])
        self.final_norm = LayerNorm(d_model)

        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)

    def forward(self, byte_ids, patch_ids):
        """
        byte_ids, patch_ids: (batch, seq_len) -- patch_ids precomputed
        upstream (dataset.py) via a trained ByteNgramEntropyModel +
        compute_patch_boundaries.
        Returns:
            patch_emb:   (batch, num_patches, d_model)
            patch_valid: (batch, num_patches) bool
        """
        batch, seq_len = byte_ids.shape

        x = self.embedding(byte_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        pad_mask = make_padding_mask(byte_ids, self.pad_id)  # (batch,1,1,seq_len)
        if self.causal:
            causal_mask = make_causal_mask(seq_len, device=byte_ids.device)
            byte_self_mask = causal_mask & pad_mask
        else:
            byte_self_mask = pad_mask

        for layer in self.layers:
            x = layer(x, src_mask=byte_self_mask)
        x = self.final_norm(x)  # (batch, seq_len, d_model)

        membership, patch_valid = build_patch_membership(byte_ids, patch_ids, pad_id=self.pad_id)
        num_patches = membership.size(1)

        # guard against a fully-invalid patch slot (batch-padding beyond an
        # example's real num_patches) -- would produce an all -inf softmax
        # row otherwise. Its output is discarded downstream via patch_valid.
        fully_pad = (~membership).all(dim=-1, keepdim=True)  # (batch,P,1)
        membership_safe = membership | fully_pad
        pool_mask = membership_safe.unsqueeze(1)  # (batch,1,P,seq_len)

        query = self.pool_query.expand(batch, num_patches, self.d_model)
        pooled, _ = self.pool_attn(query, x, x, mask=pool_mask)  # (batch, num_patches, d_model)

        return pooled, patch_valid


# ---------------------------------------------------------------------------
# Global Transformer: operates on patch vectors, not token ids. Identical
# in structure to C1's Encoder/Decoder -- no embedding lookup here, the
# input is already dense patch vectors from LocalEncoder.
# ---------------------------------------------------------------------------
class PatchEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers, max_len=2000, dropout=0.0):
        super().__init__()
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len, dropout=dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout=dropout) for _ in range(n_layers)
        ])
        self.final_norm = LayerNorm(d_model)

    def forward(self, patch_emb, patch_mask):
        x = self.pos_encoding(patch_emb)
        for layer in self.layers:
            x = layer(x, src_mask=patch_mask)
        return self.final_norm(x)


class PatchDecoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers, max_len=2000, dropout=0.0):
        super().__init__()
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len, dropout=dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout=dropout) for _ in range(n_layers)
        ])
        self.final_norm = LayerNorm(d_model)

    def forward(self, tgt_patch_emb, enc_out, self_mask, cross_mask):
        x = self.pos_encoding(tgt_patch_emb)
        for layer in self.layers:
            x = layer(x, enc_out, self_mask=self_mask, cross_mask=cross_mask)
        return self.final_norm(x)  # patch-level latents -- NOT projected to vocab yet


# ---------------------------------------------------------------------------
# Local Decoder: patch latents -> byte-level logits
# ---------------------------------------------------------------------------
class LocalDecoder(nn.Module):
    """
    Causal byte-level decoder whose cross-attention "memory" is the global
    decoder's patch latents (not an encoder's token outputs). Cross-attn
    masking uses the actual per-byte patch assignment (dynamic), via
    make_local_decoder_cross_mask.
    """

    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers,
                 pad_id=PAD_BYTE, max_len=8000, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len, dropout=dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout=dropout) for _ in range(n_layers)
        ])
        self.final_norm = LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, tgt_byte_ids_in, patch_latents, self_mask, cross_mask):
        x = self.embedding(tgt_byte_ids_in) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, patch_latents, self_mask=self_mask, cross_mask=cross_mask)
        x = self.final_norm(x)
        return self.output_proj(x)  # (batch, byte_len, vocab_size)


# ---------------------------------------------------------------------------
# Full BLT Seq2Seq model (C5)
# ---------------------------------------------------------------------------
class BLTSeq2Seq(nn.Module):
    """
    NOTE: patch_ids for both source and target are precomputed UPSTREAM
    (dataset.py, using a trained ByteNgramEntropyModel + compute_patch_
    boundaries) and passed into forward directly -- the model itself has
    no patch_size hyperparameter anymore, since patching is data-dependent
    and dynamic, not a fixed architectural stride.
    """

    def __init__(self, d_model=128, n_heads=4, d_ff=512,
                 n_local_layers=2, n_global_enc_layers=3, n_global_dec_layers=3,
                 vocab_size=BYTE_VOCAB_SIZE, pad_id=PAD_BYTE, dropout=0.1,
                 local_max_len=8000, global_max_len=2000):
        super().__init__()
        self.pad_id = pad_id

        self.src_local_encoder = LocalEncoder(
            vocab_size, d_model, n_heads, d_ff, n_local_layers,
            pad_id=pad_id, max_len=local_max_len, dropout=dropout, causal=False,
        )
        self.tgt_local_encoder = LocalEncoder(
            vocab_size, d_model, n_heads, d_ff, n_local_layers,
            pad_id=pad_id, max_len=local_max_len, dropout=dropout, causal=True,
        )

        self.global_encoder = PatchEncoder(d_model, n_heads, d_ff, n_global_enc_layers,
                                            max_len=global_max_len, dropout=dropout)
        self.global_decoder = PatchDecoder(d_model, n_heads, d_ff, n_global_dec_layers,
                                            max_len=global_max_len, dropout=dropout)

        self.local_decoder = LocalDecoder(
            vocab_size, d_model, n_heads, d_ff, n_local_layers,
            pad_id=pad_id, max_len=local_max_len, dropout=dropout,
        )

    def forward(self, src_byte_ids, src_patch_ids, tgt_byte_ids_in, tgt_patch_ids):
        """
        src_byte_ids, src_patch_ids: (batch, src_len)
        tgt_byte_ids_in, tgt_patch_ids: (batch, tgt_len) -- teacher-forced
        Returns: byte-level logits (batch, tgt_len, vocab_size)
        """
        src_patch_emb, src_patch_valid = self.src_local_encoder(src_byte_ids, src_patch_ids)
        src_patch_mask = make_patch_padding_mask(src_patch_valid)
        enc_out = self.global_encoder(src_patch_emb, src_patch_mask)

        tgt_patch_emb, tgt_patch_valid = self.tgt_local_encoder(tgt_byte_ids_in, tgt_patch_ids)
        tgt_patch_self_mask = make_patch_self_mask(tgt_patch_valid)
        dec_patch_out = self.global_decoder(tgt_patch_emb, enc_out,
                                             self_mask=tgt_patch_self_mask,
                                             cross_mask=src_patch_mask)

        local_self_mask = make_causal_mask(tgt_byte_ids_in.size(1), device=tgt_byte_ids_in.device) & \
            make_padding_mask(tgt_byte_ids_in, self.pad_id)
        local_cross_mask = make_local_decoder_cross_mask(tgt_patch_ids, tgt_patch_valid)

        logits = self.local_decoder(tgt_byte_ids_in, dec_patch_out,
                                     self_mask=local_self_mask, cross_mask=local_cross_mask)
        return logits


if __name__ == "__main__":
    torch.manual_seed(0)

    # --- byte conversion round-trip checks ---
    text = "Hello, BLT!"
    ids = text_to_byte_ids(text)
    assert ids[0] == BOS_BYTE and ids[-1] == EOS_BYTE
    assert byte_ids_to_text(ids) == text
    print("text <-> byte_ids round-trip check passed")

    bits = "".join(format(b, "08b") for b in text.encode("utf-8"))
    cipher_ids = bits_to_byte_ids(bits)
    assert cipher_ids[1:-1] == ids[1:-1]
    print("bits_to_byte_ids groups 8 bits -> 1 byte value correctly")

    # --- entropy model + dynamic patch boundaries sanity check ---
    toy_corpus = [text_to_byte_ids("the cat sat on the mat"),
                  text_to_byte_ids("the dog sat on the log"),
                  text_to_byte_ids("the cat ran to the mat")]
    ngram = ByteNgramEntropyModel(order=3)
    ngram.train(toy_corpus)

    profile = ngram.entropy_profile(toy_corpus[0])
    assert len(profile) == len(toy_corpus[0])
    assert all(h >= 0 for h in profile)
    print(f"Entropy profile computed, length {len(profile)}, sample values: {[round(h,2) for h in profile[:6]]}")

    threshold = estimate_entropy_threshold(ngram, toy_corpus, quantile=0.5)
    patch_ids = compute_patch_boundaries(profile, threshold, max_patch_size=8)
    assert len(patch_ids) == len(profile)
    assert patch_ids == sorted(patch_ids)  # non-decreasing
    num_patches = max(patch_ids) + 1
    assert num_patches > 1, "boundaries degenerated into a single patch -- patching isn't actually dynamic"
    assert num_patches < len(patch_ids), "every position became its own patch -- check max_patch_size/threshold"
    print(f"Dynamic patch boundaries: {num_patches} patches over {len(patch_ids)} bytes -> {patch_ids}")

    # confirm patch sizes are NOT uniform (the defining difference from fixed-width chunking)
    from collections import Counter as _Counter
    patch_sizes = list(_Counter(patch_ids).values())
    assert len(set(patch_sizes)) > 1, "all patches came out the same size -- patching isn't actually dynamic"
    print(f"Patch sizes vary: {patch_sizes} (confirms genuinely dynamic, not fixed-width)")

    # --- shape check for the full model with dynamic patch_ids ---
    d_model, n_heads, d_ff = 32, 4, 64
    batch, src_len, tgt_len = 3, 17, 13

    model = BLTSeq2Seq(d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                        n_local_layers=1, n_global_enc_layers=1, n_global_dec_layers=1,
                        dropout=0.0)

    src = torch.randint(0, 256, (batch, src_len))
    tgt_in = torch.randint(0, 256, (batch, tgt_len))
    # simulate precomputed dynamic patch_ids (monotonic, variable group sizes)
    src_patch_ids = torch.tensor([sorted([i % 5 for i in range(src_len)]) for _ in range(batch)])
    tgt_patch_ids = torch.tensor([sorted([i % 4 for i in range(tgt_len)]) for _ in range(batch)])

    logits = model(src, src_patch_ids, tgt_in, tgt_patch_ids)
    assert logits.shape == (batch, tgt_len, BYTE_VOCAB_SIZE), logits.shape
    print("BLT (dynamic patching) full model shape check passed:", logits.shape)

    # --- causal correctness: perturbing a LATER byte must not change EARLIER logits ---
    tgt_in2 = tgt_in.clone()
    tgt_in2[:, -1] = (tgt_in2[:, -1] + 1) % 256

    logits_a = model(src, src_patch_ids, tgt_in, tgt_patch_ids)
    logits_b = model(src, src_patch_ids, tgt_in2, tgt_patch_ids)
    last_byte_patch = tgt_patch_ids[0, -1].item()
    earlier_positions = (tgt_patch_ids[0] < last_byte_patch).nonzero(as_tuple=True)[0]
    if len(earlier_positions) > 0:
        cutoff = earlier_positions[-1].item() + 1
        assert torch.allclose(logits_a[:, :cutoff, :], logits_b[:, :cutoff, :], atol=1e-5), \
            "causal leak: earlier patches changed when a later byte was perturbed"
        print("Causal masking (no future leakage across dynamic patches) check passed")
    else:
        print("Skipping leakage check (perturbed byte's patch is the first patch)")

    # --- tiny-batch overfit check using REAL entropy-based patching end-to-end ---
    print("\nRunning tiny-batch overfit check (with real entropy-based patching)...")
    torch.manual_seed(1)

    small_batch = 6
    toy_texts = ["the cat sat", "the dog ran", "a fox jumped",
                 "birds fly high", "rain falls down", "wind blows hard"]
    toy_tgt_seqs = [text_to_byte_ids(t) for t in toy_texts]
    toy_src_seqs = [text_to_byte_ids(t[::-1]) for t in toy_texts]  # arbitrary distinct source

    tgt_ngram = ByteNgramEntropyModel(order=3)
    tgt_ngram.train(toy_tgt_seqs)
    tgt_threshold = estimate_entropy_threshold(tgt_ngram, toy_tgt_seqs, quantile=0.5)

    src_ngram = ByteNgramEntropyModel(order=3)
    src_ngram.train(toy_src_seqs)
    src_threshold = estimate_entropy_threshold(src_ngram, toy_src_seqs, quantile=0.5)

    max_len = max(max(len(s) for s in toy_src_seqs), max(len(t) for t in toy_tgt_seqs))

    def pad_and_patch(seqs, ngram_model, threshold, max_len):
        byte_batch, patch_batch = [], []
        for seq in seqs:
            profile = ngram_model.entropy_profile(seq)
            p_ids = compute_patch_boundaries(profile, threshold, max_patch_size=8)
            pad_amt = max_len - len(seq)
            byte_batch.append(seq + [PAD_BYTE] * pad_amt)
            patch_batch.append(p_ids + [p_ids[-1]] * pad_amt)  # pad patch_ids with last real patch id
        return torch.tensor(byte_batch, dtype=torch.long), torch.tensor(patch_batch, dtype=torch.long)

    toy_src, toy_src_patch = pad_and_patch(toy_src_seqs, src_ngram, src_threshold, max_len)
    toy_tgt_full, toy_tgt_full_patch = pad_and_patch(toy_tgt_seqs, tgt_ngram, tgt_threshold, max_len)

    toy_tgt_in = toy_tgt_full[:, :-1]
    toy_tgt_in_patch = toy_tgt_full_patch[:, :-1]
    toy_tgt_labels = toy_tgt_full[:, 1:]

    overfit_model = BLTSeq2Seq(d_model=64, n_heads=4, d_ff=256,
                                n_local_layers=2, n_global_enc_layers=2, n_global_dec_layers=2,
                                dropout=0.0)
    optimizer = torch.optim.Adam(overfit_model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_BYTE)

    n_steps = 300
    for step in range(n_steps):
        optimizer.zero_grad()
        logits = overfit_model(toy_src, toy_src_patch, toy_tgt_in, toy_tgt_in_patch)
        loss = criterion(logits.reshape(-1, BYTE_VOCAB_SIZE), toy_tgt_labels.reshape(-1))
        loss.backward()
        optimizer.step()
        if step % 50 == 0 or step == n_steps - 1:
            print(f"  step {step:4d}  loss {loss.item():.4f}")

    assert loss.item() < 0.3, f"BLT tiny-batch overfit did not converge, final loss = {loss.item():.4f}"
    print("Tiny-batch overfit check passed (dynamic-patching BLT architecture can learn).")

    print("\nAll checks passed.")