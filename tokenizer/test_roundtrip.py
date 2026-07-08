#!/usr/bin/env python3
"""Test encode/decode roundtrip on paragraphs."""
from tokenizer import IndiaTokenizer, clean_text
import json

data = 'data/india_wikipedia.md'
vocab_f = 'data/vocab_overfitted.json'

with open(data) as f:
    raw = f.read()
cleaned = clean_text(raw)

with open(vocab_f) as f:
    vocab = json.load(f)

tok = IndiaTokenizer(vocab)

# Split into paragraphs
paras = [p.strip() for p in cleaned.split('\n\n') if p.strip()]
print(f'Total paragraphs: {len(paras)}')

# Test first paragraph
p1 = paras[0]
ids = tok.encode(p1)
decoded = tok.decode(ids)
print(f'\n--- First paragraph ---')
print(f'Original ({len(p1)} chars): {p1[:150]}...')
print(f'Token count: {len(ids)}')
print(f'Decoded ({len(decoded)} chars): {decoded[:150]}...')
print(f'Contains India: {"India" in decoded}')
print(f'Contains Republic: {"Republic" in decoded}')
print(f'Contains South Asia: {"South Asia" in decoded}')
print(f'Exact match: {decoded == p1}')

# Test middle paragraph
mid = len(paras) // 2
pmid = paras[mid]
ids_mid = tok.encode(pmid)
decoded_mid = tok.decode(ids_mid)
print(f'\n--- Middle paragraph ({mid}/{len(paras)}) ---')
print(f'Original ({len(pmid)} chars): {pmid[:150]}...')
print(f'Token count: {len(ids_mid)}')
print(f'Decoded ({len(decoded_mid)} chars): {decoded_mid[:150]}...')
print(f'Exact match: {decoded_mid == pmid}')

# Test last paragraph
plast = paras[-1]
ids_last = tok.encode(plast)
decoded_last = tok.decode(ids_last)
print(f'\n--- Last paragraph ---')
print(f'Original ({len(plast)} chars): {plast[:150]}...')
print(f'Token count: {len(ids_last)}')
print(f'Decoded ({len(decoded_last)} chars): {decoded_last[:150]}...')
print(f'Exact match: {decoded_last == plast}')
