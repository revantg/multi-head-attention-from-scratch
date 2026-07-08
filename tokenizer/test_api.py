#!/usr/bin/env python3
"""Quick test of module-level API from repo root."""
import sys, os
from tokenizer import encode, decode, clean_text

with open('tokenizer/data/india_wikipedia.md') as f:
    raw = f.read()
cleaned = clean_text(raw)

ids = encode(cleaned)
print(f'Token count: {len(ids)}')
print(f'X1: {len(ids)/5000:.6f}')
decoded = decode(ids)
print(f'Decoded length: {len(decoded)}')
print(f'Match: {decoded == cleaned}')

paras = [p.strip() for p in cleaned.split('\n\n') if p.strip() and len(p) > 50]
p = paras[0]
p_ids = encode(p)
p_dec = decode(p_ids)
print(f'Para match: {p_dec == p}')
print(f'Has India: {"India" in p_dec}')
print(f'Has Republic: {"Republic" in p_dec}')
print(f'Has South Asia: {"South Asia" in p_dec}')
