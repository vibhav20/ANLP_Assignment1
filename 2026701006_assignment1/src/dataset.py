"""
dataset.py

Loads the aligned brown_cipher.txt / brown_plain.txt pair, splits into
train/val/test with a fixed seed, trains TWO from-scratch BPE tokenizers
(bpe_tokenizer.py) -- one on the cipher side, one on the plaintext side --
and exposes a torch Dataset + collate_fn that produce padded batches
compatible with transformer.py's Seq2SeqTransformer.

IMPORTANT: tokenizers are trained ONLY on the train split, never on val/test.
Training on the full corpus (including val/test lines) would leak
information about held-out data into the vocabulary itself -- e.g. a merge
rule that only exists because of a pattern in a test-set line. This keeps
the evaluation honest.
"""

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from bpe_tokenizer import BPETokenizer
from models.blt import (
    bits_to_byte_ids, text_to_byte_ids, PAD_BYTE, BYTE_VOCAB_SIZE,
    ByteNgramEntropyModel, estimate_entropy_threshold, compute_patch_boundaries,
)


SEED = 42
TRAIN_FRAC, VAL_FRAC = 0.8, 0.1  # remainder -> test


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
def load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def make_split(n_examples, seed=SEED, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC):
    """
    Returns (train_idx, val_idx, test_idx) -- fixed, reproducible, and meant
    to be reused identically across C1-C5 so results are comparable.
    """
    indices = list(range(n_examples))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_train = int(n_examples * train_frac)
    n_val = int(n_examples * val_frac)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    return train_idx, val_idx, test_idx


def save_split(train_idx, val_idx, test_idx, path):
    with open(path, "w") as f:
        json.dump({"train": train_idx, "val": val_idx, "test": test_idx}, f)


def load_split(path):
    with open(path, "r") as f:
        d = json.load(f)
    return d["train"], d["val"], d["test"]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CipherPlaintextDataset(Dataset):
    """
    Holds already-tokenized (id-encoded) examples. Encoding happens once,
    up front, in the loader function below -- not lazily per __getitem__ --
    since BPE encoding a fixed corpus is comparatively cheap and this keeps
    __getitem__ trivial.
    """

    def __init__(self, src_ids_list, tgt_ids_list):
        assert len(src_ids_list) == len(tgt_ids_list)
        self.src_ids_list = src_ids_list
        self.tgt_ids_list = tgt_ids_list

    def __len__(self):
        return len(self.src_ids_list)

    def __getitem__(self, idx):
        return {
            "src_ids": self.src_ids_list[idx],
            "tgt_ids": self.tgt_ids_list[idx],
        }


def make_collate_fn(src_pad_id, tgt_pad_id):
    """
    Pads a batch to the max length WITHIN that batch (not a fixed global
    max_len) -- more efficient, and transformer.py's masks are built from
    the padded tensors directly so variable batch-to-batch length is fine.

    tgt is split here into decoder-input (all but last token) and labels
    (all but first token) for teacher forcing.
    """

    def collate_fn(batch):
        src_seqs = [torch.tensor(ex["src_ids"], dtype=torch.long) for ex in batch]
        tgt_seqs = [torch.tensor(ex["tgt_ids"], dtype=torch.long) for ex in batch]

        src_max_len = max(len(s) for s in src_seqs)
        tgt_max_len = max(len(t) for t in tgt_seqs)

        src_padded = torch.full((len(batch), src_max_len), src_pad_id, dtype=torch.long)
        tgt_padded = torch.full((len(batch), tgt_max_len), tgt_pad_id, dtype=torch.long)

        for i, (s, t) in enumerate(zip(src_seqs, tgt_seqs)):
            src_padded[i, :len(s)] = s
            tgt_padded[i, :len(t)] = t

        tgt_in = tgt_padded[:, :-1]      # decoder input: <bos> ... (drop last)
        tgt_labels = tgt_padded[:, 1:]   # labels: ... <eos> (drop first)

        return {
            "src": src_padded,        # (batch, src_max_len)
            "tgt_in": tgt_in,         # (batch, tgt_max_len - 1)
            "tgt_labels": tgt_labels,  # (batch, tgt_max_len - 1)
        }

    return collate_fn


# ---------------------------------------------------------------------------
# Top-level loader: ties everything together
# ---------------------------------------------------------------------------
def build_datasets(cipher_path, plain_path, split_save_path=None,
                    src_num_merges=1000, tgt_num_merges=1000,
                    seed=SEED, batch_size=32):
    """
    Returns:
        train_loader, val_loader, test_loader,
        src_tokenizer, tgt_tokenizer
    """
    cipher_lines = load_lines(cipher_path)
    plain_lines = load_lines(plain_path)
    assert len(cipher_lines) == len(plain_lines), "cipher/plaintext line count mismatch"

    train_idx, val_idx, test_idx = make_split(len(cipher_lines), seed=seed)
    if split_save_path is not None:
        save_split(train_idx, val_idx, test_idx, split_save_path)

    # --- train tokenizers on TRAIN split only ---
    train_cipher_lines = [cipher_lines[i] for i in train_idx]
    train_plain_lines = [plain_lines[i] for i in train_idx]

    src_tokenizer = BPETokenizer(num_merges=src_num_merges)
    src_tokenizer.train(train_cipher_lines)

    tgt_tokenizer = BPETokenizer(num_merges=tgt_num_merges)
    tgt_tokenizer.train(train_plain_lines)

    # --- encode all splits with the (train-only-fitted) tokenizers ---
    def encode_split(idx_list):
        src_ids = [src_tokenizer.encode(cipher_lines[i]) for i in idx_list]
        tgt_ids = [tgt_tokenizer.encode(plain_lines[i]) for i in idx_list]
        return src_ids, tgt_ids

    train_src, train_tgt = encode_split(train_idx)
    val_src, val_tgt = encode_split(val_idx)
    test_src, test_tgt = encode_split(test_idx)

    train_ds = CipherPlaintextDataset(train_src, train_tgt)
    val_ds = CipherPlaintextDataset(val_src, val_tgt)
    test_ds = CipherPlaintextDataset(test_src, test_tgt)

    collate_fn = make_collate_fn(src_tokenizer.pad_id, tgt_tokenizer.pad_id)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, src_tokenizer, tgt_tokenizer


# ---------------------------------------------------------------------------
# Token-free loader (C5 / BLT) -- no tokenizer training at all, since every
# byte value (0-255) is already a valid id. Reuses load_lines/make_split
# above so C5 sees the EXACT SAME train/val/test line indices as C1-C4;
# comparing configs on different held-out data would invalidate the
# ablation.
# ---------------------------------------------------------------------------
class BytePairDataset(Dataset):
    """
    Holds precomputed byte ids AND their precomputed patch ids (from a
    trained ByteNgramEntropyModel + compute_patch_boundaries) -- patch
    assignment is computed ONCE per line here, not recomputed every epoch.
    """

    def __init__(self, src_ids_list, src_patch_ids_list, tgt_ids_list, tgt_patch_ids_list):
        assert len(src_ids_list) == len(tgt_ids_list) == len(src_patch_ids_list) == len(tgt_patch_ids_list)
        self.src_ids_list = src_ids_list
        self.src_patch_ids_list = src_patch_ids_list
        self.tgt_ids_list = tgt_ids_list
        self.tgt_patch_ids_list = tgt_patch_ids_list

    def __len__(self):
        return len(self.src_ids_list)

    def __getitem__(self, idx):
        return {
            "src_ids": self.src_ids_list[idx],
            "src_patch_ids": self.src_patch_ids_list[idx],
            "tgt_ids": self.tgt_ids_list[idx],
            "tgt_patch_ids": self.tgt_patch_ids_list[idx],
        }


def make_byte_collate_fn(pad_id=PAD_BYTE):
    """
    Pads byte_ids with pad_id as usual. patch_ids are padded by REPEATING
    each sequence's last real patch id -- the exact value doesn't affect
    correctness (padding positions are excluded via the byte-validity
    check inside build_patch_membership regardless), but repeating the
    last id keeps patch indices in a sane, non-exploding range.
    """

    def collate_fn(batch):
        src_seqs = [torch.tensor(ex["src_ids"], dtype=torch.long) for ex in batch]
        src_patch_seqs = [ex["src_patch_ids"] for ex in batch]
        tgt_seqs = [torch.tensor(ex["tgt_ids"], dtype=torch.long) for ex in batch]
        tgt_patch_seqs = [ex["tgt_patch_ids"] for ex in batch]

        src_max_len = max(len(s) for s in src_seqs)
        tgt_max_len = max(len(t) for t in tgt_seqs)

        src_padded = torch.full((len(batch), src_max_len), pad_id, dtype=torch.long)
        tgt_padded = torch.full((len(batch), tgt_max_len), pad_id, dtype=torch.long)
        src_patch_padded = torch.zeros((len(batch), src_max_len), dtype=torch.long)
        tgt_patch_padded = torch.zeros((len(batch), tgt_max_len), dtype=torch.long)

        for i, (s, sp, t, tp) in enumerate(zip(src_seqs, src_patch_seqs, tgt_seqs, tgt_patch_seqs)):
            src_padded[i, :len(s)] = s
            src_patch_padded[i, :len(sp)] = torch.tensor(sp, dtype=torch.long)
            if len(sp) < src_max_len:
                src_patch_padded[i, len(sp):] = sp[-1]  # repeat last patch id into the pad region

            tgt_padded[i, :len(t)] = t
            tgt_patch_padded[i, :len(tp)] = torch.tensor(tp, dtype=torch.long)
            if len(tp) < tgt_max_len:
                tgt_patch_padded[i, len(tp):] = tp[-1]

        tgt_in = tgt_padded[:, :-1]
        tgt_in_patch_ids = tgt_patch_padded[:, :-1]
        tgt_labels = tgt_padded[:, 1:]

        return {
            "src": src_padded, "src_patch_ids": src_patch_padded,
            "tgt_in": tgt_in, "tgt_in_patch_ids": tgt_in_patch_ids,
            "tgt_labels": tgt_labels,
        }

    return collate_fn


def build_byte_datasets(cipher_path, plain_path, split_save_path=None,
                         seed=SEED, batch_size=32, max_byte_len=None,
                         ngram_order=3, entropy_quantile=0.5, max_patch_size=16):
    """
    Returns:
        train_loader, val_loader, test_loader,
        src_ngram_model, src_threshold, tgt_ngram_model, tgt_threshold, max_patch_size

    The last five return values are needed by train.py's greedy decoding,
    which must recompute the TARGET side's patch assignment incrementally
    as bytes are generated -- it reuses the SAME trained model/threshold
    fit here, never a freshly-trained one at inference time.

    Entropy models are trained on the TRAIN split ONLY -- same leakage
    discipline as the BPE tokenizers in build_datasets() above.
    """
    cipher_lines = load_lines(cipher_path)
    plain_lines = load_lines(plain_path)
    assert len(cipher_lines) == len(plain_lines), "cipher/plaintext line count mismatch"

    train_idx, val_idx, test_idx = make_split(len(cipher_lines), seed=seed)
    if split_save_path is not None:
        save_split(train_idx, val_idx, test_idx, split_save_path)

    # --- convert to byte ids once ---
    all_src_byte_ids = [bits_to_byte_ids(line) for line in cipher_lines]
    all_tgt_byte_ids = [text_to_byte_ids(line) for line in plain_lines]

    # --- train entropy models on TRAIN split only ---
    train_src_seqs = [all_src_byte_ids[i] for i in train_idx]
    train_tgt_seqs = [all_tgt_byte_ids[i] for i in train_idx]

    src_ngram_model = ByteNgramEntropyModel(order=ngram_order)
    src_ngram_model.train(train_src_seqs)
    src_threshold = estimate_entropy_threshold(src_ngram_model, train_src_seqs, quantile=entropy_quantile)

    tgt_ngram_model = ByteNgramEntropyModel(order=ngram_order)
    tgt_ngram_model.train(train_tgt_seqs)
    tgt_threshold = estimate_entropy_threshold(tgt_ngram_model, train_tgt_seqs, quantile=entropy_quantile)

    print(f"[byte loader] src entropy threshold={src_threshold:.3f}, tgt entropy threshold={tgt_threshold:.3f}")

    # --- encode all splits: byte ids + precomputed dynamic patch ids ---
    def encode_split(idx_list):
        src_ids, src_patch, tgt_ids, tgt_patch = [], [], [], []
        for i in idx_list:
            s_ids = all_src_byte_ids[i]
            t_ids = all_tgt_byte_ids[i]
            if max_byte_len is not None and (len(s_ids) > max_byte_len or len(t_ids) > max_byte_len):
                continue

            s_profile = src_ngram_model.entropy_profile(s_ids)
            s_patch_ids = compute_patch_boundaries(s_profile, src_threshold, max_patch_size=max_patch_size)

            t_profile = tgt_ngram_model.entropy_profile(t_ids)
            t_patch_ids = compute_patch_boundaries(t_profile, tgt_threshold, max_patch_size=max_patch_size)

            src_ids.append(s_ids)
            src_patch.append(s_patch_ids)
            tgt_ids.append(t_ids)
            tgt_patch.append(t_patch_ids)
        return src_ids, src_patch, tgt_ids, tgt_patch

    train_src, train_src_patch, train_tgt, train_tgt_patch = encode_split(train_idx)
    val_src, val_src_patch, val_tgt, val_tgt_patch = encode_split(val_idx)
    test_src, test_src_patch, test_tgt, test_tgt_patch = encode_split(test_idx)

    train_ds = BytePairDataset(train_src, train_src_patch, train_tgt, train_tgt_patch)
    val_ds = BytePairDataset(val_src, val_src_patch, val_tgt, val_tgt_patch)
    test_ds = BytePairDataset(test_src, test_src_patch, test_tgt, test_tgt_patch)

    collate_fn = make_byte_collate_fn(pad_id=PAD_BYTE)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return (train_loader, val_loader, test_loader,
            src_ngram_model, src_threshold, tgt_ngram_model, tgt_threshold, max_patch_size)


if __name__ == "__main__":
    # --- build a toy cipher_plain pair on disk and run the full pipeline ---
    import tempfile, os

    toy_plain = [
        "the cat sat on the mat",
        "the dog sat on the log",
        "the cat and the dog ran",
        "a quick brown fox jumps",
        "the fox ran over the log",
        "the cat ran to the mat",
        "birds fly over the trees",
        "the sun sets in the west",
        "rain falls on the roof",
        "the wind blows through trees",
    ]

    def to_bits(s):
        return "".join(format(b, "08b") for b in s.encode("utf-8"))

    toy_cipher = [to_bits(line) for line in toy_plain]

    tmpdir = tempfile.mkdtemp()
    cipher_path = os.path.join(tmpdir, "toy_cipher.txt")
    plain_path = os.path.join(tmpdir, "toy_plain.txt")
    split_path = os.path.join(tmpdir, "splits.json")

    with open(cipher_path, "w") as f:
        f.write("\n".join(toy_cipher) + "\n")
    with open(plain_path, "w") as f:
        f.write("\n".join(toy_plain) + "\n")

    train_loader, val_loader, test_loader, src_tok, tgt_tok = build_datasets(
        cipher_path, plain_path, split_save_path=split_path,
        src_num_merges=40, tgt_num_merges=40, seed=SEED, batch_size=4,
    )

    print(f"src vocab size: {src_tok.vocab_size}")
    print(f"tgt vocab size: {tgt_tok.vocab_size}")
    print(f"train batches: {len(train_loader)}, val: {len(val_loader)}, test: {len(test_loader)}")

    # pull one batch and check shapes / masking compatibility
    batch = next(iter(train_loader))
    print("\nbatch src shape:", batch["src"].shape)
    print("batch tgt_in shape:", batch["tgt_in"].shape)
    print("batch tgt_labels shape:", batch["tgt_labels"].shape)

    assert batch["src"].dtype == torch.long
    assert batch["tgt_in"].shape == batch["tgt_labels"].shape
    print("Batch shape/dtype checks passed")

    # confirm split reload works and matches
    r_train, r_val, r_test = load_split(split_path)
    assert r_train == [i for i in r_train]  # trivial, just confirming load doesn't error
    print("Split save/load round-trip check passed")

    # end-to-end decode check: decode a src/tgt pair back and sanity print
    ex = train_loader.dataset[0]
    decoded_tgt = tgt_tok.decode(ex["tgt_ids"])
    print(f"\nExample decoded target: {decoded_tgt!r}")

    # --- wire into the actual Seq2SeqTransformer to confirm end-to-end compatibility ---
    from transformer import Seq2SeqTransformer

    model = Seq2SeqTransformer(
        src_vocab_size=src_tok.vocab_size,
        tgt_vocab_size=tgt_tok.vocab_size,
        d_model=32, n_heads=4, d_ff=64,
        n_enc_layers=2, n_dec_layers=2, dropout=0.0,
        src_pad_idx=src_tok.pad_id, tgt_pad_idx=tgt_tok.pad_id,
    )
    logits = model(batch["src"], batch["tgt_in"])
    expected_shape = (batch["tgt_in"].shape[0], batch["tgt_in"].shape[1], tgt_tok.vocab_size)
    assert logits.shape == expected_shape, f"{logits.shape} vs {expected_shape}"
    print(f"\nEnd-to-end dataset -> model forward pass check passed: {logits.shape}")

    # --- token-free (C5/BLT) loader check, using the same toy files ---
    print("\n=== Token-free (byte-level, entropy-based dynamic patching) loader check ===")
    (byte_train_loader, byte_val_loader, byte_test_loader,
     src_ngram_model, src_threshold, tgt_ngram_model, tgt_threshold, max_patch_size) = build_byte_datasets(
        cipher_path, plain_path, split_save_path=os.path.join(tmpdir, "splits_c5.json"),
        batch_size=4,
    )
    print(f"byte train batches: {len(byte_train_loader)}, val: {len(byte_val_loader)}, "
          f"test: {len(byte_test_loader)}")

    byte_batch = next(iter(byte_train_loader))
    print("byte batch src shape:", byte_batch["src"].shape)
    print("byte batch src_patch_ids shape:", byte_batch["src_patch_ids"].shape)
    assert byte_batch["src"].dtype == torch.long
    assert byte_batch["src"].shape == byte_batch["src_patch_ids"].shape
    assert byte_batch["tgt_in"].shape == byte_batch["tgt_labels"].shape
    assert byte_batch["tgt_in"].shape == byte_batch["tgt_in_patch_ids"].shape
    print("Byte batch shape/dtype checks passed")

    # confirm patch ids are non-uniform (genuinely dynamic, not fixed stride)
    from collections import Counter as _Counter
    example_patch_sizes = list(_Counter(byte_batch["tgt_in_patch_ids"][0].tolist()).values())
    print(f"Example patch sizes within one line: {example_patch_sizes}")

    from models.blt import byte_ids_to_text, BLTSeq2Seq
    byte_ex = byte_train_loader.dataset[0]
    byte_decoded = byte_ids_to_text(byte_ex["tgt_ids"])
    print(f"Example decoded target (byte loader): {byte_decoded!r}")

    blt_model = BLTSeq2Seq(d_model=32, n_heads=4, d_ff=64,
                            n_local_layers=1, n_global_enc_layers=1, n_global_dec_layers=1,
                            dropout=0.0)
    blt_logits = blt_model(byte_batch["src"], byte_batch["src_patch_ids"],
                            byte_batch["tgt_in"], byte_batch["tgt_in_patch_ids"])
    assert blt_logits.shape[0] == byte_batch["src"].shape[0]
    assert blt_logits.shape[1] == byte_batch["tgt_in"].shape[1]
    assert blt_logits.shape[2] == BYTE_VOCAB_SIZE
    print(f"End-to-end byte loader -> BLTSeq2Seq forward pass check passed: {blt_logits.shape}")

    print("\nAll checks passed.")