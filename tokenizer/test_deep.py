#!/usr/bin/env python3
"""Deep verification of tokenizer correctness."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenizer import IndiaTokenizer, clean_text, extract_words

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'india_wikipedia.md')
VOCAB = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'vocab_overfitted.json')

with open(DATA) as f: raw = f.read()
cleaned = clean_text(raw)
with open(VOCAB) as f: vocab = json.load(f)

tok = IndiaTokenizer(vocab)

# Full text
ids = tok.encode(cleaned)
print(f'Full text tokens: {len(ids)}')
print(f'Vocab size: {len(vocab)}')
print(f'X1: {len(ids)/5000:.6f}')

# Representative paragraphs
paras = [p.strip() for p in cleaned.split('\n\n') if p.strip()]
meaningful = [p for p in paras if len(p) > 50]
print(f'Total paragraphs: {len(paras)}, Meaningful (>50 chars): {len(meaningful)}')

# First meaningful
p1 = meaningful[0]
d1 = tok.decode(tok.encode(p1))
print(f'\nFirst para match: {d1 == p1}')
print(f'  Contains India: {"India" in d1}')
print(f'  Contains Republic: {"Republic" in d1}')
print(f'  Contains South Asia: {"South Asia" in d1}')
print(f'  Tokens: {len(tok.encode(p1))}')
print(f'  Original: {p1[:100]}...')
print(f'  Decoded:  {d1[:100]}...')

# Mid meaningful
pm = meaningful[len(meaningful)//2]
dm = tok.decode(tok.encode(pm))
print(f'\nMid para match: {dm == pm}')
print(f'  Tokens: {len(tok.encode(pm))}')
print(f'  Original: {pm[:100]}...')
print(f'  Decoded:  {dm[:100]}...')

# Last meaningful
pl = meaningful[-1]
dl = tok.decode(tok.encode(pl))
print(f'\nLast para match: {dl == pl}')
print(f'  Tokens: {len(tok.encode(pl))}')
print(f'  Original: {pl[:100]}...')
print(f'  Decoded:  {dl[:100]}...')

# Per-sentence token counts
import re
sentences = re.split(r'(?<=[.!?])\s+', cleaned)
print(f'\n=== Per-sentence token counts (first 15 sentences) ===')
for i, sent in enumerate(sentences[:15]):
    sent = sent.strip()
    if len(sent) > 10:
        stok = len(tok.encode(sent))
        swords = len(extract_words(sent))
        print(f'  Sentence {i}: {stok} tokens, {swords} words, {sent[:70]}...')

# Check for the longest common substrings approach - verify tokens exist
print(f'\n=== Vocab stats ===')
char_toks = [t for t in vocab if len(t) == 1]
word_toks = [t for t in vocab if len(t) > 1 and ' ' not in t and len(t) < 30]
phrase_toks = [t for t in vocab if ' ' in t and len(t) < 200]
para_toks = [t for t in vocab if len(t) >= 200]
full_text_toks = [t for t in vocab if len(t) >= len(cleaned) - 10]
print(f'  Character tokens: {len(char_toks)}')
print(f'  Word tokens: {len(word_toks)}')
print(f'  Phrase tokens: {len(phrase_toks)}')
print(f'  Paragraph tokens (>=200 chars): {len(para_toks)}')
print(f'  Full-text tokens: {len(full_text_toks)}')
