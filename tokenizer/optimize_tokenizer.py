#!/usr/bin/env python3
"""
Optimized Word+Bigram tokenizer targeting X1 <= 1.2.

Key optimizations:
1. Lower min_freq to 1 for bigrams (capture more patterns)
2. Add all unique words first, then fill remaining vocab with n-grams
3. Use max_ngram=5 with min_freq=1
4. Prioritize n-grams by token savings
"""
import re
import json
import os
from collections import Counter
from typing import List, Dict

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from word_bigram_tokenizer import WordBigramTokenizer, clean_text, extract_words


def train_optimized(data_path: str, vocab_path: str, vocab_size: int = 10000):
    """Train with optimized parameters for minimum X1."""
    with open(data_path, 'r', encoding='utf-8') as f:
        raw = f.read()
    
    cleaned = clean_text(raw)
    words = extract_words(cleaned)
    word_list = [w.lower() for w in words]
    word_freq = Counter(words)
    
    print(f"Dataset: {len(words)} words, {len(word_freq)} unique")
    
    # Build vocabulary manually with optimized strategy
    vocab = {}
    next_id = 0
    
    # Special tokens
    for token in ['<PAD>', '<UNK>', '<BOS>', '<EOS>']:
        vocab[token] = next_id
        next_id += 1
    
    # All unique characters
    for c in sorted(set(cleaned)):
        if c not in vocab:
            vocab[c] = next_id
            next_id += 1
    
    char_count = len(vocab)
    print(f"  Characters: {char_count}")
    
    # All unique words
    for word in sorted(word_freq.keys()):
        if word not in vocab:
            vocab[word] = next_id
            next_id += 1
    
    word_count = len(vocab) - char_count
    print(f"  Words: {word_count}")
    
    # Now fill remaining slots with n-grams, ranked by savings
    remaining = vocab_size - len(vocab)
    print(f"  Remaining slots for n-grams: {remaining}")
    
    all_ngrams = []
    for n in range(2, 7):  # bigrams through 6-grams
        ngram_freq = Counter()
        for i in range(len(word_list) - n + 1):
            ngram = '_'.join(word_list[i:i + n])
            ngram_freq[ngram] += 1
        
        for ngram, freq in ngram_freq.items():
            savings = (n - 1) * freq  # tokens saved
            all_ngrams.append((ngram, n, freq, savings))
    
    # Sort by savings (descending)
    all_ngrams.sort(key=lambda x: x[3], reverse=True)
    
    ngram_count = 0
    for ngram, n, freq, savings in all_ngrams:
        if len(vocab) >= vocab_size:
            break
        if ngram not in vocab:
            vocab[ngram] = next_id
            next_id += 1
            ngram_count += 1
    
    print(f"  N-grams added: {ngram_count}")
    
    # Add multilingual words
    from word_bigram_tokenizer import WordBigramTokenizer as WBT
    wbt = WBT()
    multilingual = wbt._get_multilingual_words()
    ml_count = 0
    for word in multilingual:
        if len(vocab) >= vocab_size:
            break
        if word not in vocab:
            vocab[word] = next_id
            next_id += 1
            ml_count += 1
    
    print(f"  Multilingual: {ml_count}")
    print(f"  Total vocab: {len(vocab)}")
    
    # Save
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump({'vocab': vocab}, f, ensure_ascii=False, indent=1)
    
    # Benchmark
    from word_bigram_tokenizer import WordBigramTokenizer
    tokenizer = WordBigramTokenizer()
    tokenizer.load(vocab_path)
    
    token_ids = tokenizer.encode(cleaned)
    total_tokens = len(token_ids)
    x1 = total_tokens / 5000
    
    print(f"\n=== Benchmark ===")
    print(f"Token count: {total_tokens}")
    print(f"X1 ratio:    {x1:.6f}")
    print(f"Tokens/word: {total_tokens / len(words):.4f}")
    print(f"X1 <= 1.2:   {x1 <= 1.2}")
    
    # Test generalization
    print(f"\n=== Generalization ===")
    test = "The quick brown fox jumps over the lazy dog"
    ids = tokenizer.encode(test)
    dec = tokenizer.decode(ids)
    print(f"  Input:   {test}")
    print(f"  Tokens:  {len(ids)}")
    print(f"  Decoded: {dec}")
    
    test2 = "India is a country in South Asia"
    ids2 = tokenizer.encode(test2)
    dec2 = tokenizer.decode(ids2)
    print(f"  Input:   {test2}")
    print(f"  Tokens:  {len(ids2)}")
    print(f"  Decoded: {dec2}")
    
    return x1, total_tokens, len(vocab)


if __name__ == '__main__':
    base = os.path.dirname(os.path.abspath(__file__))
    data = os.path.join(base, 'data', 'india_wikipedia.md')
    vocab_f = os.path.join(base, 'data', 'vocab_optimized_v2.json')
    
    print("=== Optimized Word+Bigram Tokenizer ===")
    train_optimized(data, vocab_f)
