"""
bpe_tokenizer.py

From-scratch Byte-Pair Encoding tokenizer. 
Used TWICE with two independently trained instances:
  - source_tokenizer: trained on brown_cipher.txt, base vocab = {'0','1'}
    -> learns recurring BIT PATTERNS as merged subword units (e.g. "0101"
       becomes one token if it's frequent), NOT fixed 8-bit chunking.
  - target_tokenizer: trained on brown_plain.txt, base vocab = characters
    -> standard text BPE (learns frequent character n-grams like "th", "ing").

Special tokens: <pad>, <bos>, <eos>, <unk>, reserved at fixed ids 0-3 in
every vocab so both tokenizers share the same special-token convention
(pad_idx=0 matches what transformer.py expects).
"""

import re
from collections import Counter


PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]  # ids 0,1,2,3


class BPETokenizer:
    """
    Classic BPE: start from a base vocabulary (individual characters), then
    iteratively merge the most frequent adjacent symbol pair across the
    training corpus, for a fixed number of merges. Every training corpus
    (cipher, plaintext) gets its OWN instance with its own learned merges
    and vocab -- they are not shared.
    """

    def __init__(self, num_merges: int = 500):
        self.num_merges = num_merges
        self.merges = []          # ordered list of (sym_a, sym_b) -> merged, in learned order
        self.token_to_id = {}
        self.id_to_token = {}

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------
    def _get_base_vocab(self, corpus):
        """Every unique character appearing in the corpus."""
        chars = set()
        for line in corpus:
            chars.update(line)
        return sorted(chars)

    def _word_to_symbols(self, word):
        """Represent a line as a list of single-character symbols to start."""
        return list(word)

    def _get_pair_counts(self, corpus_symbols):
        """Count frequency of every adjacent symbol pair across the corpus."""
        pair_counts = Counter()
        for symbols in corpus_symbols:
            for i in range(len(symbols) - 1):
                pair_counts[(symbols[i], symbols[i + 1])] += 1
        return pair_counts

    def _merge_pair(self, symbols, pair):
        """Merge every occurrence of `pair` in a single symbol list."""
        merged_symbol = pair[0] + pair[1]
        new_symbols = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                new_symbols.append(merged_symbol)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        return new_symbols

    def train(self, corpus):
        """
        corpus: list of strings (raw lines, e.g. cipher bit-strings or
        plaintext sentences).
        Learns self.num_merges merge rules greedily by frequency.
        """
        base_vocab = self._get_base_vocab(corpus)

        # working representation: each line as a list of symbols
        corpus_symbols = [self._word_to_symbols(line) for line in corpus]

        vocab = set(base_vocab)
        merges = []

        for _ in range(self.num_merges):
            pair_counts = self._get_pair_counts(corpus_symbols)
            if not pair_counts:
                break  # nothing left to merge

            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < 2:
                break  # no more repeated patterns worth merging

            corpus_symbols = [self._merge_pair(sym, best_pair) for sym in corpus_symbols]
            merged_symbol = best_pair[0] + best_pair[1]
            vocab.add(merged_symbol)
            merges.append(best_pair)

        self.merges = merges

        # build final id mappings: special tokens first (fixed ids), then
        # base chars, then merged symbols in the order they were learned
        # (so the tokenizer is deterministic and reproducible)
        ordered_vocab = list(SPECIAL_TOKENS) + list(base_vocab) + \
            [a + b for (a, b) in merges]
        # dedupe while preserving order (a merged symbol could coincide with
        # a base char only in pathological cases, but guard anyway)
        seen = set()
        final_vocab = []
        for tok in ordered_vocab:
            if tok not in seen:
                seen.add(tok)
                final_vocab.append(tok)

        self.token_to_id = {tok: i for i, tok in enumerate(final_vocab)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

    # -----------------------------------------------------------------
    # Encoding / Decoding
    # -----------------------------------------------------------------
    def _apply_merges(self, symbols):
        """Apply learned merges IN LEARNED ORDER to a fresh symbol list."""
        for pair in self.merges:
            symbols = self._merge_pair(symbols, pair)
        return symbols

    def encode(self, text, add_bos_eos=True):
        """
        text: a single raw string (one line).
        Returns: list of token ids.
        """
        symbols = self._word_to_symbols(text)
        symbols = self._apply_merges(symbols)

        ids = [self.token_to_id.get(sym, self.token_to_id[UNK_TOKEN]) for sym in symbols]

        if add_bos_eos:
            ids = [self.token_to_id[BOS_TOKEN]] + ids + [self.token_to_id[EOS_TOKEN]]
        return ids

    def decode(self, ids, strip_special=True):
        """
        ids: list/tensor of token ids.
        Returns: reconstructed string.
        """
        tokens = [self.id_to_token.get(int(i), UNK_TOKEN) for i in ids]
        if strip_special:
            tokens = [t for t in tokens if t not in SPECIAL_TOKENS]
        return "".join(tokens)

    @property
    def vocab_size(self):
        return len(self.token_to_id)

    @property
    def pad_id(self):
        return self.token_to_id[PAD_TOKEN]

    @property
    def bos_id(self):
        return self.token_to_id[BOS_TOKEN]

    @property
    def eos_id(self):
        return self.token_to_id[EOS_TOKEN]


if __name__ == "__main__":
    # --- sanity checks on a toy corpus before touching real data ---

    # 1) Plaintext-style corpus: should learn frequent character n-grams
    toy_plain_corpus = [
        "the cat sat on the mat",
        "the dog sat on the log",
        "the cat and the dog ran",
    ]
    plain_tok = BPETokenizer(num_merges=30)
    plain_tok.train(toy_plain_corpus)

    print("=== Plaintext BPE ===")
    print("vocab size:", plain_tok.vocab_size)
    print("first few merges:", plain_tok.merges[:5])

    for line in toy_plain_corpus:
        ids = plain_tok.encode(line)
        decoded = plain_tok.decode(ids)
        assert decoded == line, f"round-trip failed: {line!r} -> {decoded!r}"
    print("Round-trip encode/decode check passed on all toy plaintext lines")

    # confirm special tokens are at the front
    assert plain_tok.token_to_id[PAD_TOKEN] == 0
    assert plain_tok.token_to_id[BOS_TOKEN] == 1
    assert plain_tok.token_to_id[EOS_TOKEN] == 2
    assert plain_tok.token_to_id[UNK_TOKEN] == 3
    print("Special token id assignment check passed")

    # confirm bos/eos wrapping
    ids = plain_tok.encode("the cat sat on the mat")
    assert ids[0] == plain_tok.bos_id
    assert ids[-1] == plain_tok.eos_id
    print("BOS/EOS wrapping check passed")

    # 2) Cipher-style corpus: base vocab should be exactly {'0','1'} (+specials),
    #    and merges should learn recurring BIT patterns, not fixed 8-bit chunks
    toy_cipher_corpus = [
        "0100100001100101011011000110110001101111",  # "Hello" in ASCII bits, repeated patterns exist
        "0100100001101001",                            # "Hi"
        "01001000011001010110110001110000",            # "Help"
    ]
    cipher_tok = BPETokenizer(num_merges=20)
    cipher_tok.train(toy_cipher_corpus)

    print("\n=== Cipher (binary) BPE ===")
    print("vocab size:", cipher_tok.vocab_size)
    print("first few merges:", cipher_tok.merges[:5])

    base_chars = set(cipher_tok.token_to_id.keys()) - set(SPECIAL_TOKENS)
    single_char_tokens = {t for t in base_chars if len(t) == 1}
    assert single_char_tokens == {"0", "1"}, f"unexpected base vocab: {single_char_tokens}"
    print("Cipher base vocab is exactly {'0','1'} check passed")

    # confirm no merge is a rigid 8-length chunk boundary artifact by construction
    # (this isn't fully checkable in general, but confirm merges are LEARNED,
    # i.e. not all of fixed length 8)
    merge_lengths = [len(a + b) for a, b in cipher_tok.merges]
    assert not all(l == 8 for l in merge_lengths), "merges degenerated into fixed 8-bit chunks"
    print("Merges are learned subword units, not fixed-width chunks (lengths vary):", merge_lengths[:10])

    for line in toy_cipher_corpus:
        ids = cipher_tok.encode(line)
        decoded = cipher_tok.decode(ids)
        assert decoded == line, f"round-trip failed: {line!r} -> {decoded!r}"
    print("Round-trip encode/decode check passed on all toy cipher lines")

    # 3) unseen text with an unknown symbol should not crash (falls back to <unk> per-char if needed)
    weird = "9" * 5  # '9' never appears in cipher_tok's training vocab
    try:
        ids = cipher_tok.encode(weird)
        print(f"\nUnseen-symbol encode did not crash, ids: {ids}")
    except Exception as e:
        print(f"\nUnseen-symbol encode raised: {e}")

    print("\nAll checks passed.")
