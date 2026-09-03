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
