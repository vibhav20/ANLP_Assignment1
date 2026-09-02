"""
train.py

Trains one config (C1-C4) end-to-end: builds the model via
transformer.build_model, trains with teacher forcing, evaluates with
greedy decoding, logs to WandB, and pushes checkpoints to Hugging Face.

Run on Colab as:
    from train import run_config
    metrics = run_config("C1", train_loader, val_loader, test_loader,
                          src_tok, tgt_tok, epochs=10)

Or loop over all four configs -- see the __main__ block at the bottom for
the exact orchestration pattern to paste into a Colab cell.
"""

import time
import torch
import torch.nn as nn
import wandb
from huggingface_hub import HfApi

from .transformer import build_model, make_padding_mask, make_decoder_self_mask
from .utils import compute_all_metrics


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