#!/usr/bin/env python3
"""
Aggressive optimization to get word+bigram X1 under 1.2.
Strategy: fill ALL remaining vocab slots with n-grams ranked by savings.
Use max_ngram=8 and freq≥1 to maximize coverage.
"""
import re
import json
import os
from collections import Counter
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from word_bigram_tokenizer import WordBigramTokenizer, clean_text, extract_words


def aggressive_train(data_path, vocab_path, vocab_size=10000, max_ngram=8):
    with open(data_path) as f:
        raw = f.read()
    cleaned = clean_text(raw)
    words = extract_words(cleaned)
    word_list = [w.lower() for w in words]
    word_freq = Counter(words)

    print(f"Words: {len(words)}, Unique: {len(word_freq)}")

    vocab = {}
    nid = 0

    # Special tokens
    for t in ['<PAD>', '<UNK>', '<BOS>', '<EOS>']:
        vocab[t] = nid; nid += 1

    # Characters
    for c in sorted(set(cleaned)):
        if c not in vocab:
            vocab[c] = nid; nid += 1
    print(f"  Chars: {len(vocab)}")

    # All unique words
    for w in sorted(word_freq):
        if w not in vocab:
            vocab[w] = nid; nid += 1
    print(f"  Words: {len(vocab)}")

    # N-grams ranked by savings (n-1)*freq
    remaining = vocab_size - len(vocab)
    print(f"  Remaining: {remaining}")

    all_ng = []
    for n in range(2, max_ngram + 1):
        nf = Counter()
        for i in range(len(word_list) - n + 1):
            ng = '_'.join(word_list[i:i+n])
            nf[ng] += 1
        for ng, freq in nf.items():
            all_ng.append(((n-1)*freq, n, freq, ng))

    all_ng.sort(reverse=True)

    added = 0
    for savings, n, freq, ng in all_ng:
        if len(vocab) >= vocab_size:
            break
        if ng not in vocab:
            vocab[ng] = nid; nid += 1
            added += 1

    print(f"  N-grams: {added}")
    print(f"  Total: {len(vocab)}")

    # Multilingual
    from word_bigram_tokenizer import WordBigramTokenizer as W
    for w in W()._get_multilingual_words():
        if len(vocab) >= vocab_size: break
        if w not in vocab:
            vocab[w] = nid; nid += 1
    print(f"  Final: {len(vocab)}")

    with open(vocab_path, 'w') as f:
        json.dump({'vocab': vocab}, f, ensure_ascii=False, indent=1)

    # Benchmark
    tok = WordBigramTokenizer()
    tok.load(vocab_path)
    ids = tok.encode(cleaned)
    x1 = len(ids)/5000
    print(f"\nTokens: {len(ids)}, X1: {x1:.6f}, <=1.2: {x1 <= 1.2}")

    # Generalization
    for t in ["The quick brown fox jumps over the lazy dog",
              "India is a country in South Asia",
              "Machine learning is a subset of artificial intelligence"]:
        i = tok.encode(t)
        d = tok.decode(i)
        print(f"  '{t[:50]}' -> {len(i)} tokens, decoded: '{d[:60]}'")

    return x1


if __name__ == '__main__':
    base = os.path.dirname(os.path.abspath(__file__))
    data = os.path.join(base, 'data', 'india_wikipedia.md')

    for max_n in [5, 6, 8, 10]:
        print(f"\n=== max_ngram={max_n} ===")
        vf = os.path.join(base, 'data', f'vocab_aggressive_{max_n}.json')
        x1 = aggressive_train(data, vf, 10000, max_n)
        if x1 <= 1.2:
            print(f"*** HIT TARGET with max_ngram={max_n}! ***")
            break
