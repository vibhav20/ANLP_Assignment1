import math

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
# Patch boundaries: FIXED-WIDTH (simplified, per relaxed C5 requirement).
#
# This drops in with zero changes to the model code below (LocalEncoder,
# build_patch_membership, make_patch_self_mask,
# make_local_decoder_cross_mask, etc.) because all of those only ever
# consume a per-byte, non-decreasing integer patch_ids sequence -- they
# never inspect how it was derived.
# ---------------------------------------------------------------------------
def compute_fixed_patch_boundaries(seq_len: int, patch_size: int = 4):
    """
    seq_len: number of byte positions (including BOS/EOS) in this sequence.
    patch_size: fixed number of bytes per patch (last patch may be shorter).
    Returns: list[int], length seq_len, non-decreasing, e.g. patch_size=4 ->
             [0,0,0,0, 1,1,1,1, 2,2,...]
    """
    return [i // patch_size for i in range(seq_len)]


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
    left-to-right order, so causal masking over PATCH INDEX still
    corresponds exactly to temporal order.
    Returns: (batch, 1, num_patches, num_patches) bool.
    """
    batch, num_patches = patch_valid.shape
    causal = make_causal_mask(num_patches, device=patch_valid.device)  # (1,1,P,P)
    pad = make_patch_padding_mask(patch_valid)  # (batch,1,1,P)
    return causal & pad

def make_local_decoder_cross_mask(byte_patch_ids, patch_valid):
    batch, byte_len = byte_patch_ids.shape
    num_patches = patch_valid.size(1)
    device = byte_patch_ids.device

    patch_range = torch.arange(num_patches, device=device).view(1, 1, -1)
    causal_byte_to_patch = (patch_range < byte_patch_ids.unsqueeze(-1))

    # patch-0 bytes have no strictly-earlier patch -- fall back to their
    # own patch instead of leaving an all-False (NaN-softmax) row
    no_earlier_patch = ~causal_byte_to_patch.any(dim=-1, keepdim=True)
    own_patch = (patch_range == byte_patch_ids.unsqueeze(-1))
    causal_byte_to_patch = causal_byte_to_patch | (no_earlier_patch & own_patch)

    causal_byte_to_patch = causal_byte_to_patch.unsqueeze(1)
    pad = make_patch_padding_mask(patch_valid)
    return causal_byte_to_patch & pad

def build_patch_membership(byte_ids: torch.Tensor, patch_ids: torch.Tensor, pad_id: int = PAD_BYTE):
    """
    byte_ids, patch_ids: (batch, seq_len) -- patch_ids gives each byte
    position's assigned patch index (padding positions can hold any
    placeholder value; they're excluded via the byte-validity check below
    regardless).

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
# Local Encoder: bytes -> patch representations
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
       to this patch) rather than a fixed contiguous reshape.
    """

    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers,
                 pad_id=PAD_BYTE, max_len=8000, dropout=0.0, causal=False):
        """
        causal: if True, byte self-attention is masked so byte i only sees
            bytes <= i (used for the TARGET-side LocalEncoder -- required
            regardless of patching scheme, since bidirectional
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
        upstream (dataset.py) via compute_fixed_patch_boundaries.
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
    masking uses the actual per-byte patch assignment, via
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
    (dataset.py, using compute_fixed_patch_boundaries) and passed into
    forward directly -- the model itself has no patch_size hyperparameter,
    since patching happens entirely as a preprocessing step.
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

    # --- fixed-width patch boundaries sanity check ---
    toy_seq = text_to_byte_ids("the cat sat on the mat")
    patch_ids = compute_fixed_patch_boundaries(len(toy_seq), patch_size=4)
    assert len(patch_ids) == len(toy_seq)
    assert patch_ids == sorted(patch_ids)  # non-decreasing
    num_patches = max(patch_ids) + 1
    assert num_patches > 1
    print(f"Fixed-width patch boundaries: {num_patches} patches over {len(patch_ids)} bytes -> {patch_ids}")

    # --- shape check for the full model with fixed patch_ids ---
    d_model, n_heads, d_ff = 32, 4, 64
    batch, src_len, tgt_len = 3, 17, 13
    patch_size = 4

    model = BLTSeq2Seq(d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                        n_local_layers=1, n_global_enc_layers=1, n_global_dec_layers=1,
                        dropout=0.0)

    src = torch.randint(0, 256, (batch, src_len))
    tgt_in = torch.randint(0, 256, (batch, tgt_len))
    src_patch_ids = torch.tensor([compute_fixed_patch_boundaries(src_len, patch_size) for _ in range(batch)])
    tgt_patch_ids = torch.tensor([compute_fixed_patch_boundaries(tgt_len, patch_size) for _ in range(batch)])

    logits = model(src, src_patch_ids, tgt_in, tgt_patch_ids)
    assert logits.shape == (batch, tgt_len, BYTE_VOCAB_SIZE), logits.shape
    print("BLT (fixed patching) full model shape check passed:", logits.shape)

    # --- causal correctness: perturbing a LATER byte must not change EARLIER logits ---
    tgt_in2 = tgt_in.clone()
    tgt_in2[:, -1] = (tgt_in2[:, -1] + 1) % 256

    logits_a = model(src, src_patch_ids, tgt_in, tgt_patch_ids)
    logits_b = model(src, src_patch_ids, tgt_in2, tgt_patch_ids)

    last_byte_patch = tgt_patch_ids[0, -1].item()
    earlier_positions = (tgt_patch_ids[0] < last_byte_patch).nonzero(as_tuple=True)[0]
    if len(earlier_positions) > 0:
        cutoff = earlier_positions[-1].item() + 1

        # --- add these three lines here, before the assert ---
        torch.set_printoptions(precision=10)
        max_diff_early = (logits_a[:, :cutoff, :] - logits_b[:, :cutoff, :]).abs().max()
        print("Max diff over 'earlier' region (full precision):", max_diff_early.item())
        # --------------------------------------------------------

        assert torch.allclose(logits_a[:, :cutoff, :], logits_b[:, :cutoff, :], atol=1e-5), \
            "causal leak: earlier patches changed when a later byte was perturbed"
        print("Causal masking (no future leakage across patches) check passed")
    else:
        print("Skipping leakage check (perturbed byte's patch is the first patch)")
    # --- tiny-batch overfit check using fixed-width patching end-to-end ---
    print("\nRunning tiny-batch overfit check (fixed-width patching)...")
    torch.manual_seed(1)

    toy_texts = ["the cat sat", "the dog ran", "a fox jumped",
                 "birds fly high", "rain falls down", "wind blows hard"]
    toy_tgt_seqs = [text_to_byte_ids(t) for t in toy_texts]
    toy_src_seqs = [text_to_byte_ids(t[::-1]) for t in toy_texts]  # arbitrary distinct source

    max_len = max(max(len(s) for s in toy_src_seqs), max(len(t) for t in toy_tgt_seqs))

    def pad_and_patch(seqs, max_len, patch_size):
        byte_batch, patch_batch = [], []
        for seq in seqs:
            p_ids = compute_fixed_patch_boundaries(len(seq), patch_size=patch_size)
            pad_amt = max_len - len(seq)
            byte_batch.append(seq + [PAD_BYTE] * pad_amt)
            patch_batch.append(p_ids + [p_ids[-1]] * pad_amt)  # pad patch_ids with last real patch id
        return torch.tensor(byte_batch, dtype=torch.long), torch.tensor(patch_batch, dtype=torch.long)

    toy_src, toy_src_patch = pad_and_patch(toy_src_seqs, max_len, patch_size=4)
    toy_tgt_full, toy_tgt_full_patch = pad_and_patch(toy_tgt_seqs, max_len, patch_size=4)

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
    print("Tiny-batch overfit check passed (fixed-patching BLT architecture can learn).")

    print("\nAll checks passed.")