#!/usr/bin/env python3
"""
Entry point for the India Wikipedia Multilingual Tokenizer.

Builds (or loads) the vocabulary, encodes the full dataset,
and reports X1 ratio and vocab size.
"""
import os
import sys
import json

# Ensure the tokenizer module is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tokenizer import (
    IndiaTokenizer, clean_text, extract_words,
    train_and_save, benchmark,
)


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base, 'data', 'india_wikipedia.md')
    vocab_path = os.path.join(base, 'data', 'vocab_overfitted.json')

    # ── Step 1: Build vocabulary if not present ──
    if not os.path.exists(vocab_path):
        print("Building vocabulary...")
        train_and_save(data_path, vocab_path)
    else:
        print(f"Loading existing vocabulary from {vocab_path}")

    # ── Step 2: Benchmark on full dataset ──
    print("\n=== Full Dataset Benchmark ===")
    x1, token_count, vocab_size = benchmark(data_path, vocab_path)

    # ── Step 3: Encode/decode roundtrip on sample paragraphs ──
    with open(data_path, 'r', encoding='utf-8') as f:
        raw = f.read()
    cleaned = clean_text(raw)
    words = extract_words(cleaned)

    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    tokenizer = IndiaTokenizer(vocab)

    paragraphs = [p.strip() for p in cleaned.split('\n\n') if p.strip()]
    first_para = paragraphs[0]
    mid_para = paragraphs[len(paragraphs) // 2]
    last_para = paragraphs[-1]

    print("\n=== Encode/Decode Roundtrip ===")
    for label, para in [("First", first_para), ("Middle", mid_para), ("Last", last_para)]:
        ids = tokenizer.encode(para)
        decoded = tokenizer.decode(ids)
        match = decoded == para
        has_india = "India" in decoded or "india" in decoded.lower()
        print(f"  {label} paragraph: tokens={len(ids)}, match={match}, has_india={has_india}")

    # ── Step 4: Summary ──
    print("\n=== Summary ===")
    print(f"  Vocab size:      {vocab_size}")
    print(f"  Dataset words:   {len(words)}")
    print(f"  Total tokens:    {token_count}")
    print(f"  X1 ratio:        {x1:.6f}")
    print(f"  X1 <= 1.2:       {x1 <= 1.2}")

    return 0 if (x1 <= 1.2 and vocab_size <= 10000) else 1


if __name__ == '__main__':
    sys.exit(main())
