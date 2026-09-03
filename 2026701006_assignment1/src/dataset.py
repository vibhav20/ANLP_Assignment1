import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from .bpe_tokenizer import BPETokenizer
from .models.blt import (
    bits_to_byte_ids, text_to_byte_ids, PAD_BYTE, BYTE_VOCAB_SIZE,
    compute_fixed_patch_boundaries,
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
# Dataset C1-C4 BPE Tokeniser
# ---------------------------------------------------------------------------
class CipherPlaintextDataset(Dataset):

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
    Pads a batch to the max length WITHIN that batch 

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
# Token-free loader (C5 / BLT) : no tokenizer training at all
# ---------------------------------------------------------------------------
class BytePairDataset(Dataset):
    """
    Holds precomputed byte ids AND their precomputed patch ids -- patch
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
                         patch_size=4):
    """
    Simplified (fixed-width patching) version of the C5 byte-level loader.
    No entropy model, no n-gram training, no threshold estimation -- byte i
    is assigned to patch i // patch_size for both src and tgt. This is a
    valid simplification per the relaxed C5 patching requirement; the rest
    of the pipeline (BLTSeq2Seq, LocalEncoder's membership-based pooling,
    the global transformer, LocalDecoder's patch-causal cross-attention)
    is unchanged and agnostic to how patch_ids were derived.
 
    Returns:
        train_loader, val_loader, test_loader, patch_size
 
    `patch_size` is returned so train.py's greedy decoding can recompute
    the target side's patch assignment incrementally with the SAME rule
    used here (call compute_fixed_patch_boundaries(len_so_far, patch_size)
    after each generated byte -- no trained model/state to carry around,
    unlike the entropy version).
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
 
    # --- encode all splits: byte ids + fixed-width patch ids ---
    def encode_split(idx_list):
        src_ids, src_patch, tgt_ids, tgt_patch = [], [], [], []
        for i in idx_list:
            s_ids = all_src_byte_ids[i]
            t_ids = all_tgt_byte_ids[i]
            if max_byte_len is not None and (len(s_ids) > max_byte_len or len(t_ids) > max_byte_len):
                continue
 
            s_patch_ids = compute_fixed_patch_boundaries(len(s_ids), patch_size=patch_size)
            t_patch_ids = compute_fixed_patch_boundaries(len(t_ids), patch_size=patch_size)
 
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
 
    return train_loader, val_loader, test_loader, patch_size