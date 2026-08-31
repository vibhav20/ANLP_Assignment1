"""
utils.py

Evaluation metrics for the ablation study. Per the assignment: standard
LIBRARIES are allowed here (editdistance, sacrebleu, rouge-score) -- unlike
the tokenizer, which had to be implemented from scratch. You're still
expected to know how each metric works, so each function has a short note
on what it's actually measuring.

All metrics are computed on GREEDY-DECODED model output vs. the ground
truth plaintext, per the spec ("generate your numbers using greedy
decoding so results are consistent across models").
"""

import editdistance
import sacrebleu
from rouge_score import rouge_scorer


# ---------------------------------------------------------------------------
# Bit-level accuracy
# ---------------------------------------------------------------------------
def bit_level_accuracy(pred_bits: str, target_bits: str):
    """
    % of exact bit matches, position-by-position, over the length of the
    SHORTER of the two (a length mismatch is itself a sign of a bad
    prediction -- extra/missing bits beyond the shorter length count as
    misses, handled by dividing by len(target_bits) below).

    pred_bits / target_bits: strings of '0'/'1'.
    Returns a float in [0, 1].
    """
    if len(target_bits) == 0:
        return 1.0 if len(pred_bits) == 0 else 0.0

    compare_len = min(len(pred_bits), len(target_bits))
    matches = sum(1 for i in range(compare_len) if pred_bits[i] == target_bits[i])
    # positions beyond compare_len (if pred is shorter than target) count as misses
    return matches / len(target_bits)


def batch_bit_level_accuracy(pred_bits_list, target_bits_list):
    """Mean bit-level accuracy across a batch/dataset."""
    scores = [bit_level_accuracy(p, t) for p, t in zip(pred_bits_list, target_bits_list)]
    return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Sequence accuracy
# ---------------------------------------------------------------------------
def sequence_accuracy(predictions, targets):
    """
    % of sequences that are PERFECTLY reconstructed (exact string match).
    predictions / targets: lists of strings (decoded plaintext, one per example).
    Returns a float in [0, 1].
    """
    assert len(predictions) == len(targets)
    if not predictions:
        return 0.0
    exact_matches = sum(1 for p, t in zip(predictions, targets) if p == t)
    return exact_matches / len(predictions)


# ---------------------------------------------------------------------------
# Levenshtein (edit) distance
# ---------------------------------------------------------------------------
def levenshtein_distance(pred: str, target: str):
    """
    Edit distance (insertions + deletions + substitutions) between predicted
    and target strings. Uses the `editdistance` library (standard library
    usage permitted for metrics; the underlying DP algorithm is the classic
    O(n*m) Wagner-Fischer table you'd implement by hand for DSA prep).
    """
    return editdistance.eval(pred, target)


def mean_levenshtein_distance(predictions, targets):
    assert len(predictions) == len(targets)
    if not predictions:
        return 0.0
    dists = [levenshtein_distance(p, t) for p, t in zip(predictions, targets)]
    return sum(dists) / len(dists)


# ---------------------------------------------------------------------------
# BLEU / ROUGE (tokenized configs only, per the spec -- these are n-gram
# overlap metrics that assume a word/subword-level output; they don't apply
# to C5's raw byte-level generation in the same way)
# ---------------------------------------------------------------------------
def corpus_bleu(predictions, targets):
    """
    Corpus-level BLEU via sacrebleu (handles its own internal tokenization
    consistently, avoiding the classic bug of comparing BLEU scores computed
    with different tokenizers across papers/configs).
    predictions: list[str], targets: list[str] (references).
    """
    if not predictions:
        return 0.0
    # sacrebleu expects references as list[list[str]] (list of reference lists)
    bleu = sacrebleu.corpus_bleu(predictions, [targets])
    return bleu.score  # 0-100 scale


_rouge_scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def mean_rouge(predictions, targets):
    """
    Mean ROUGE-1/2/L F-measure across the set.
    Returns a dict: {"rouge1": ..., "rouge2": ..., "rougeL": ...}
    """
    assert len(predictions) == len(targets)
    if not predictions:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for p, t in zip(predictions, targets):
        scores = _rouge_scorer.score(t, p)  # (target, prediction) order per library convention
        for key in totals:
            totals[key] += scores[key].fmeasure

    n = len(predictions)
    return {key: val / n for key, val in totals.items()}


# ---------------------------------------------------------------------------
# Top-level: compute everything for one config's test-set predictions
# ---------------------------------------------------------------------------
def compute_all_metrics(predictions, targets, pred_bits_list=None, target_bits_list=None,
                         include_bleu_rouge=True):
    """
    predictions, targets: lists of decoded plaintext strings.
    pred_bits_list, target_bits_list: optional, only needed if you also want
        bit-level accuracy computed on the raw bit representation rather
        than re-deriving it from the decoded strings.
    include_bleu_rouge: set False for C5 (BLT/byte-level) if you decide
        n-gram overlap metrics aren't meaningful for that config -- the
        spec restricts BLEU/ROUGE to "tokenized models only".
    """
    results = {
        "sequence_accuracy": sequence_accuracy(predictions, targets),
        "mean_levenshtein": mean_levenshtein_distance(predictions, targets),
    }

    if pred_bits_list is not None and target_bits_list is not None:
        results["bit_level_accuracy"] = batch_bit_level_accuracy(pred_bits_list, target_bits_list)

    if include_bleu_rouge:
        results["bleu"] = corpus_bleu(predictions, targets)
        results.update(mean_rouge(predictions, targets))

    return results


if __name__ == "__main__":
    # --- sanity checks on known small examples ---

    # bit-level accuracy
    assert bit_level_accuracy("1010", "1010") == 1.0
    assert bit_level_accuracy("1010", "1011") == 0.75
    assert bit_level_accuracy("0000", "1111") == 0.0
    print("bit_level_accuracy checks passed")

    # sequence accuracy
    preds = ["the cat sat", "the dog ran", "wrong output"]
    targs = ["the cat sat", "the dog ran", "the correct one"]
    acc = sequence_accuracy(preds, targs)
    assert abs(acc - (2 / 3)) < 1e-9
    print(f"sequence_accuracy check passed: {acc:.4f}")

    # levenshtein
    assert levenshtein_distance("kitten", "sitting") == 3  # classic textbook example
    assert levenshtein_distance("same", "same") == 0
    print("levenshtein_distance checks passed")

    # bleu (sanity: identical strings should score very high, near 100)
    identical_preds = ["the cat sat on the mat", "the dog ran in the park"]
    identical_targs = ["the cat sat on the mat", "the dog ran in the park"]
    bleu_score = corpus_bleu(identical_preds, identical_targs)
    assert bleu_score > 99.0, bleu_score
    print(f"corpus_bleu identical-input check passed: {bleu_score:.2f}")

    # rouge (sanity: identical strings should score ~1.0 on all variants)
    rouge_scores = mean_rouge(identical_preds, identical_targs)
    for k, v in rouge_scores.items():
        assert v > 0.99, (k, v)
    print(f"mean_rouge identical-input check passed: {rouge_scores}")

    # end-to-end
    all_metrics = compute_all_metrics(preds, targs, include_bleu_rouge=True)
    print("\ncompute_all_metrics example output:")
    for k, v in all_metrics.items():
        print(f"  {k}: {v}")

    print("\nAll checks passed.")