"""
train.py

Trains one config (C1-C4) end-to-end: builds the model via
transformer.build_model, trains with teacher forcing, evaluates with
greedy decoding, logs to WandB, and pushes checkpoints to Hugging Face.

Run on Colab as:
    from train import run_config
    metrics = run_config("C1", train_loader, val_loader, test_loader,
                          src_tok, tgt_tok, epochs=10)

    from train import run_config_c5
    metrics = run_config_c5(byte_train_loader, byte_val_loader,
                             byte_test_loader, patch_size, epochs=10)

Or loop over all four configs -- see the __main__ block at the bottom for
the exact orchestration pattern to paste into a Colab cell.
"""

import time
import torch
import torch.nn as nn
import wandb
from huggingface_hub import HfApi

from .transformer import build_model, make_padding_mask, make_decoder_self_mask, make_causal_mask
from .utils import compute_all_metrics

from .models.blt import (
    BOS_BYTE, EOS_BYTE, PAD_BYTE, BYTE_VOCAB_SIZE,
    byte_ids_to_text, BLTSeq2Seq,
    compute_fixed_patch_boundaries,
    make_patch_padding_mask, make_patch_self_mask, make_local_decoder_cross_mask,
)
 


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Training loop for one config
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, tgt_vocab_size,
                     log_every=50, wandb_run=None, epoch=0, global_step=0):
    model.train()
    total_loss, n_batches = 0.0, 0

    for i, batch in enumerate(loader):
        src = batch["src"].to(DEVICE)
        tgt_in = batch["tgt_in"].to(DEVICE)
        tgt_labels = batch["tgt_labels"].to(DEVICE)

        optimizer.zero_grad()
        logits = model(src, tgt_in)  # (batch, tgt_len, vocab)
        loss = criterion(logits.reshape(-1, tgt_vocab_size), tgt_labels.reshape(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        global_step += 1

        if wandb_run is not None and global_step % log_every == 0:
            wandb_run.log({"train/loss_step": loss.item(), "epoch": epoch}, step=global_step)

    return total_loss / max(n_batches, 1), global_step


@torch.no_grad()
def evaluate_loss(model, loader, criterion, tgt_vocab_size):
    model.eval()
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        src = batch["src"].to(DEVICE)
        tgt_in = batch["tgt_in"].to(DEVICE)
        tgt_labels = batch["tgt_labels"].to(DEVICE)

        logits = model(src, tgt_in)
        loss = criterion(logits.reshape(-1, tgt_vocab_size), tgt_labels.reshape(-1))
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Greedy decoding for eval metrics (bit acc, seq acc, Levenshtein, BLEU/ROUGE)
# ---------------------------------------------------------------------------
@torch.no_grad()
def greedy_decode_batch(model, src, tgt_tokenizer, max_len=200):
    """
    src: (batch, src_len) already on DEVICE.
    Returns: list[str] decoded plaintext, one per example in the batch.
    Generates step-by-step, feeding the model's own previous prediction
    back in (no teacher forcing) -- this is what "greedy decoding" means:
    at each step, take the argmax token, no sampling, no beam search.
    """
    model.eval()
    batch_size = src.size(0)
    src_mask = make_padding_mask(src, model.src_pad_idx)
    enc_out = model.encoder(src, src_mask=src_mask)

    bos_id = tgt_tokenizer.bos_id
    eos_id = tgt_tokenizer.eos_id

    generated = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=src.device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=src.device)

    for _ in range(max_len):
        tgt_mask = make_decoder_self_mask(generated, model.tgt_pad_idx)
        cross_mask = src_mask
        logits = model.decoder(generated, enc_out, self_mask=tgt_mask, cross_mask=cross_mask)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # greedy: argmax, no sampling

        # once a sequence has emitted <eos>, keep padding it with <eos> so
        # its length stops growing further as the rest of the batch continues
        next_token = torch.where(finished.unsqueeze(1), torch.full_like(next_token, eos_id), next_token)
        generated = torch.cat([generated, next_token], dim=1)

        finished = finished | (next_token.squeeze(1) == eos_id)
        if finished.all():
            break

    decoded = [tgt_tokenizer.decode(seq.tolist(), strip_special=True) for seq in generated]
    return decoded


@torch.no_grad()
def evaluate_metrics(model, loader, tgt_tokenizer, max_gen_len=200, include_bleu_rouge=True):
    """
    Runs greedy decoding over an entire loader (val or test) and computes
    all metrics from utils.py against the ground-truth plaintext.
    """
    model.eval()
    all_preds, all_targets = [], []

    for batch in loader:
        src = batch["src"].to(DEVICE)
        tgt_labels = batch["tgt_labels"]  # keep on CPU, only needed for decode

        preds = greedy_decode_batch(model, src, tgt_tokenizer, max_len=max_gen_len)
        targets = [tgt_tokenizer.decode(seq.tolist(), strip_special=True) for seq in tgt_labels]

        all_preds.extend(preds)
        all_targets.extend(targets)

    return compute_all_metrics(all_preds, all_targets, include_bleu_rouge=include_bleu_rouge)


# ---------------------------------------------------------------------------
# Hugging Face checkpoint push
# ---------------------------------------------------------------------------
def push_checkpoint(model, config_name, epoch, hf_repo_id, local_dir="/tmp/ckpt"):
    """
    Saves the model state dict locally, then uploads it into a
    per-config subfolder of the single HF repo:
        <hf_repo_id>/C1/checkpoint_latest.pt   (overwritten each push, for easy reload)
    """
    import os
    os.makedirs(local_dir, exist_ok=True)

    latest_path = os.path.join(local_dir, "checkpoint_latest.pt")
    torch.save(model.state_dict(), latest_path)

    api = HfApi()
    api.upload_file(
        path_or_fileobj=latest_path,
        path_in_repo=f"{config_name}/checkpoint_latest.pt",
        repo_id=hf_repo_id,
        repo_type="model",
    )
    print(f"  pushed checkpoint -> {hf_repo_id}/{config_name}/checkpoint_latest.pt (epoch {epoch})")

 
# ---------------------------------------------------------------------------
# C5 (BLT): byte-level greedy decoding and training loop.
# ---------------------------------------------------------------------------
@torch.no_grad()
def greedy_decode_batch_blt(model, src, src_patch_ids, patch_size=4, max_len=2000):
    """
    Byte-by-byte greedy generation for BLTSeq2Seq with FIXED-WIDTH patching.
 
    Source-side patch ids are already fixed at data-loading time
    (src_patch_ids, passed straight through). Target-side patch ids are
    recomputed at every decoding step with the SAME rule dataset.py uses
    (compute_fixed_patch_boundaries) -- since patch membership depends only
    on byte POSITION, not content, there's no entropy model / n-gram state
    to carry across steps (unlike the dynamic-patching version). Because
    every example in the batch is padded to the same `generated` length at
    each step, the target patch-id row is identical across the whole batch
    and only needs to be computed once, then broadcast.
 
    Returns: list[str] decoded plaintext, one per example in the batch.
    """
    model.eval()
    batch_size = src.size(0)
 
    src_patch_emb, src_patch_valid = model.src_local_encoder(src, src_patch_ids)
    src_patch_mask = make_patch_padding_mask(src_patch_valid)
    enc_out = model.global_encoder(src_patch_emb, src_patch_mask)
 
    generated = torch.full((batch_size, 1), BOS_BYTE, dtype=torch.long, device=src.device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=src.device)
 
    for _ in range(max_len):
        cur_len = generated.size(1)
        patch_ids_row = compute_fixed_patch_boundaries(cur_len, patch_size=patch_size)
        patch_ids_tensor = torch.tensor(patch_ids_row, dtype=torch.long, device=src.device) \
            .unsqueeze(0).expand(batch_size, -1)
 
        tgt_patch_emb, tgt_patch_valid = model.tgt_local_encoder(generated, patch_ids_tensor)
        tgt_patch_self_mask = make_patch_self_mask(tgt_patch_valid)
        dec_patch_out = model.global_decoder(tgt_patch_emb, enc_out,
                                              self_mask=tgt_patch_self_mask,
                                              cross_mask=src_patch_mask)
 
        local_self_mask = make_causal_mask(cur_len, device=generated.device) & \
            make_padding_mask(generated, model.pad_id)
        local_cross_mask = make_local_decoder_cross_mask(patch_ids_tensor, tgt_patch_valid)
 
        logits = model.local_decoder(generated, dec_patch_out,
                                      self_mask=local_self_mask, cross_mask=local_cross_mask)
 
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        next_token = torch.where(finished.unsqueeze(1),
                                  torch.full_like(next_token, EOS_BYTE), next_token)
        generated = torch.cat([generated, next_token], dim=1)
 
        finished = finished | (next_token.squeeze(1) == EOS_BYTE)
        if finished.all():
            break
 
    return [byte_ids_to_text(seq.tolist()) for seq in generated]
 
 
@torch.no_grad()
def evaluate_metrics_blt(model, loader, patch_size=4, max_gen_len=2000):
    """
    Same idea as evaluate_metrics, but for BLT: BLEU/ROUGE are excluded --
    the spec restricts them to "tokenized models only". Also computes
    bit-level accuracy on the UTF-8 bit representation of predicted vs.
    target plaintext, since there's no subword granularity for C5.
 
    `patch_size` is the SAME fixed-width value used to build the loaders
    (dataset.build_byte_datasets()'s return value) -- passed straight
    through to greedy_decode_batch_blt so target-side patch ids during
    generation are computed with the identical rule used at training time.
    """
    model.eval()
    all_preds, all_targets, pred_bits, target_bits = [], [], [], []
 
    def text_to_bits(s):
        return "".join(format(b, "08b") for b in s.encode("utf-8"))
 
    for batch in loader:
        src = batch["src"].to(DEVICE)
        src_patch_ids = batch["src_patch_ids"].to(DEVICE)
        tgt_labels = batch["tgt_labels"]
 
        preds = greedy_decode_batch_blt(model, src, src_patch_ids,
                                         patch_size=patch_size, max_len=max_gen_len)
        targets = [byte_ids_to_text(seq.tolist()) for seq in tgt_labels]
 
        all_preds.extend(preds)
        all_targets.extend(targets)
        pred_bits.extend(text_to_bits(p) for p in preds)
        target_bits.extend(text_to_bits(t) for t in targets)
 
    return compute_all_metrics(all_preds, all_targets,
                                pred_bits_list=pred_bits, target_bits_list=target_bits,
                                include_bleu_rouge=False)
 
 
def run_config_c5(train_loader, val_loader, test_loader, patch_size,
                   epochs=10, lr=3e-4, d_model=128, n_heads=4, d_ff=512,
                   n_local_layers=2, n_global_enc_layers=3, n_global_dec_layers=3, dropout=0.1,
                   wandb_project="anlp-a1-ablation", hf_repo_id=None,
                   checkpoint_every=1, max_gen_len=2000):
    """
    Mirrors run_config's structure (build -> train -> log -> checkpoint ->
    greedy-decode eval) but for BLTSeq2Seq with FIXED-WIDTH patching.
    `patch_size` comes directly from dataset.build_byte_datasets()'s return
    value -- always reuse the SAME patch_size the loaders were built with,
    since it also drives target-side patch-id recomputation during greedy
    decoding (see greedy_decode_batch_blt). Unlike the entropy-based
    version, there's no trained n-gram model or threshold to thread
    through here -- fixed patching needs no extra state at all.
    """
    config_name = "C5"
    print(f"\n{'='*60}\nRunning {config_name} (BLT, fixed-width patching) on {DEVICE}\n{'='*60}")
 
    model = BLTSeq2Seq(
        d_model=d_model, n_heads=n_heads, d_ff=d_ff,
        n_local_layers=n_local_layers, n_global_enc_layers=n_global_enc_layers,
        n_global_dec_layers=n_global_dec_layers, dropout=dropout,
    ).to(DEVICE)
 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_BYTE)
 
    run = wandb.init(project=wandb_project, name=config_name, reinit=True,
                      config={
                          "config_name": config_name, "epochs": epochs, "lr": lr,
                          "d_model": d_model, "n_heads": n_heads, "d_ff": d_ff,
                          "n_local_layers": n_local_layers,
                          "n_global_enc_layers": n_global_enc_layers,
                          "n_global_dec_layers": n_global_dec_layers,
                          "vocab_size": BYTE_VOCAB_SIZE,
                          "patch_size": patch_size,
                      })
 
    global_step = 0
    t0 = time.time()
    peak_mem_mb = 0.0
    train_loss, val_loss = 0.0, 0.0
 
    for epoch in range(epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        for batch in train_loader:
            src = batch["src"].to(DEVICE)
            src_patch_ids = batch["src_patch_ids"].to(DEVICE)
            tgt_in = batch["tgt_in"].to(DEVICE)
            tgt_in_patch_ids = batch["tgt_in_patch_ids"].to(DEVICE)
            tgt_labels = batch["tgt_labels"].to(DEVICE)
 
            # patch_ids for both src and tgt come straight from the collate
            # fn (precomputed once via compute_fixed_patch_boundaries in
            # dataset.py) -- model output length == tgt_in length exactly,
            # so labels need no extra alignment step.
            optimizer.zero_grad()
            logits = model(src, src_patch_ids, tgt_in, tgt_in_patch_ids)
            loss = criterion(logits.reshape(-1, BYTE_VOCAB_SIZE), tgt_labels.reshape(-1))
            loss.backward()
            optimizer.step()
 
            total_loss += loss.item()
            n_batches += 1
            global_step += 1
            if global_step % 50 == 0:
                run.log({"train/loss_step": loss.item(), "epoch": epoch}, step=global_step)
 
        train_loss = total_loss / max(n_batches, 1)
 
        model.eval()
        val_total, val_n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                src = batch["src"].to(DEVICE)
                src_patch_ids = batch["src_patch_ids"].to(DEVICE)
                tgt_in = batch["tgt_in"].to(DEVICE)
                tgt_in_patch_ids = batch["tgt_in_patch_ids"].to(DEVICE)
                tgt_labels = batch["tgt_labels"].to(DEVICE)
 
                logits = model(src, src_patch_ids, tgt_in, tgt_in_patch_ids)
                loss = criterion(logits.reshape(-1, BYTE_VOCAB_SIZE), tgt_labels.reshape(-1))
                val_total += loss.item()
                val_n += 1
        val_loss = val_total / max(val_n, 1)
 
        if DEVICE.type == "cuda":
            peak_mem_mb = max(peak_mem_mb, torch.cuda.max_memory_allocated() / 1e6)
 
        run.log({"epoch": epoch, "train/loss_epoch": train_loss, "val/loss_epoch": val_loss,
                  "peak_gpu_mem_mb": peak_mem_mb}, step=global_step)
        print(f"[{config_name}] epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
 
        if hf_repo_id is not None and (epoch + 1) % checkpoint_every == 0:
            push_checkpoint(model, config_name, epoch + 1, hf_repo_id)
 
    train_time_sec = time.time() - t0
 
    if hf_repo_id is not None:
        push_checkpoint(model, config_name, epochs, hf_repo_id)
 
    print(f"[{config_name}] running greedy-decode evaluation on test set (this is slower than "
          f"C1-C4's decoding -- see the note in greedy_decode_batch_blt)...")
    test_metrics = evaluate_metrics_blt(model, test_loader, patch_size=patch_size, max_gen_len=max_gen_len)
    test_metrics["train_time_sec"] = train_time_sec
    test_metrics["peak_gpu_mem_mb"] = peak_mem_mb
    test_metrics["final_train_loss"] = train_loss
    test_metrics["final_val_loss"] = val_loss
 
    run.log({f"test/{k}": v for k, v in test_metrics.items()})
    print(f"[{config_name}] test metrics: {test_metrics}")
 
    run.finish()
    return test_metrics
 


# ---------------------------------------------------------------------------
# Top-level: train + evaluate + log one config
# ---------------------------------------------------------------------------
def run_config(config_name, train_loader, val_loader, test_loader,
               src_tokenizer, tgt_tokenizer,
               epochs=10, lr=3e-4, d_model=128, n_heads=4, d_ff=512,
               n_enc_layers=3, n_dec_layers=3, max_len=5000, dropout=0.1,
               n_kv_heads=2, wandb_project="anlp-a1-ablation",
               hf_repo_id=None, checkpoint_every=1, max_gen_len=200,
               include_bleu_rouge=True):
    """
    Full pipeline for ONE config: build model -> train -> log to WandB ->
    push checkpoints to HF each `checkpoint_every` epochs -> final eval on
    test set with greedy decoding -> return metrics dict.
    """
    print(f"\n{'='*60}\nRunning {config_name} on {DEVICE}\n{'='*60}")

    model = build_model(
        config_name, src_tokenizer.vocab_size, tgt_tokenizer.vocab_size,
        d_model=d_model, n_heads=n_heads, d_ff=d_ff,
        n_enc_layers=n_enc_layers, n_dec_layers=n_dec_layers,
        max_len=max_len, dropout=dropout,
        src_pad_idx=src_tokenizer.pad_id, tgt_pad_idx=tgt_tokenizer.pad_id,
        n_kv_heads=n_kv_heads,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_tokenizer.pad_id)

    run = wandb.init(project=wandb_project, name=config_name, reinit=True,
                      config={
                          "config_name": config_name, "epochs": epochs, "lr": lr,
                          "d_model": d_model, "n_heads": n_heads, "d_ff": d_ff,
                          "n_enc_layers": n_enc_layers, "n_dec_layers": n_dec_layers,
                          "n_kv_heads": n_kv_heads,
                          "src_vocab_size": src_tokenizer.vocab_size,
                          "tgt_vocab_size": tgt_tokenizer.vocab_size,
                      })

    global_step = 0
    t0 = time.time()
    peak_mem_mb = 0.0

    for epoch in range(epochs):
        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, criterion, tgt_tokenizer.vocab_size,
            wandb_run=run, epoch=epoch, global_step=global_step,
        )
        val_loss = evaluate_loss(model, val_loader, criterion, tgt_tokenizer.vocab_size)

        if DEVICE.type == "cuda":
            peak_mem_mb = max(peak_mem_mb, torch.cuda.max_memory_allocated() / 1e6)

        run.log({
            "epoch": epoch, "train/loss_epoch": train_loss, "val/loss_epoch": val_loss,
            "peak_gpu_mem_mb": peak_mem_mb,
        }, step=global_step)
        print(f"[{config_name}] epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        # push checkpoint every `checkpoint_every` epochs, so a dropped
        # Colab/Kaggle session never loses more than that many epochs
        if hf_repo_id is not None and (epoch + 1) % checkpoint_every == 0:
            push_checkpoint(model, config_name, epoch + 1, hf_repo_id)

    train_time_sec = time.time() - t0

    # final checkpoint push regardless of checkpoint_every, so the LAST
    # epoch's weights are always saved even if it doesn't land on the
    # checkpoint_every boundary
    if hf_repo_id is not None:
        push_checkpoint(model, config_name, epochs, hf_repo_id)

    print(f"[{config_name}] running greedy-decode evaluation on test set...")
    test_metrics = evaluate_metrics(
        model, test_loader, tgt_tokenizer, max_gen_len=max_gen_len,
        include_bleu_rouge=include_bleu_rouge,
    )
    test_metrics["train_time_sec"] = train_time_sec
    test_metrics["peak_gpu_mem_mb"] = peak_mem_mb
    test_metrics["final_train_loss"] = train_loss
    test_metrics["final_val_loss"] = val_loss

    run.log({f"test/{k}": v for k, v in test_metrics.items()})
    print(f"[{config_name}] test metrics: {test_metrics}")

    run.finish()
    return test_metrics


if __name__ == "__main__":
    # ---------------------------------------------------------------
    # Paste-into-Colab orchestration pattern for running all of C1-C4.
    # Assumes dataset.py's build_datasets() has already been called.
    # ---------------------------------------------------------------
    from .dataset import build_datasets

    train_loader, val_loader, test_loader, src_tok, tgt_tok = build_datasets(
        cipher_path="brown_cipher.txt",
        plain_path="brown_plain.txt",
        split_save_path="splits.json",
        src_num_merges=500,
        tgt_num_merges=500,
        batch_size=32,
    )

    HF_REPO_ID = "<your-username>/anlp-a1-transformer"  # must already exist (create_repo done once)

    all_results = {}
    for config_name in ["C1", "C2", "C3", "C4"]:
        all_results[config_name] = run_config(
            config_name, train_loader, val_loader, test_loader, src_tok, tgt_tok,
            epochs=10, lr=3e-4, d_model=128, n_heads=4, d_ff=512,
            n_enc_layers=3, n_dec_layers=3, dropout=0.1, n_kv_heads=2,
            wandb_project="anlp-a1-ablation", hf_repo_id=HF_REPO_ID,
            checkpoint_every=2,
        )

    print("\n\n=== Summary across all configs ===")
    for cfg, metrics in all_results.items():
        print(cfg, metrics)

    from .dataset import build_byte_datasets
 
    byte_train_loader, byte_val_loader, byte_test_loader, patch_size = build_byte_datasets(
        cipher_path="brown_cipher.txt",
        plain_path="brown_plain.txt",
        split_save_path="splits_c5.json",
        batch_size=32,
        patch_size=4,
    )
 
    all_results["C5"] = run_config_c5(
        byte_train_loader, byte_val_loader, byte_test_loader, patch_size,
        epochs=10, lr=3e-4, d_model=128, n_heads=4, d_ff=512,
        n_local_layers=2, n_global_enc_layers=3, n_global_dec_layers=3, dropout=0.1,
        wandb_project="anlp-a1-ablation", hf_repo_id=HF_REPO_ID,
        checkpoint_every=2,
    )
 
    print("\n\n=== Summary across all configs ===")
    for cfg, metrics in all_results.items():
        print(cfg, metrics)
