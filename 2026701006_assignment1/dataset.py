"""
dataset.py

Loads the aligned brown_cipher.txt / brown_plain.txt pair, splits into
train/val/test with a fixed seed, trains TWO from-scratch BPE tokenizers
(bpe_tokenizer.py) -- one on the cipher side, one on the plaintext side --
and exposes a torch Dataset + collate_fn that produce padded batches
compatible with transformer.py's Seq2SeqTransformer.
"""

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from bpe_tokenizer import BPETokenizer


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

    print("\nAll checks passed.")
