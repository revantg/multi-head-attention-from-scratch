#!/usr/bin/env python3
"""
Verification test that matches the plan's ## Verification plan steps exactly.
Designed to be run from the repo root: python3 tokenizer/test_verification.py
"""
import os
import sys
import json
import re
import subprocess

# Paths relative to repo root
REPO_ROOT = os.path.dirname(os.path.abspath(__file__)) if os.path.basename(os.path.dirname(os.path.abspath(__file__))) != 'tokenizer' else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Handle running from tokenizer/ subdir
script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(script_dir) == 'tokenizer':
    REPO_ROOT = os.path.dirname(script_dir)
    TOKENIZER_DIR = script_dir
else:
    REPO_ROOT = script_dir
    TOKENIZER_DIR = os.path.join(script_dir, 'tokenizer')

sys.path.insert(0, TOKENIZER_DIR)
from tokenizer import IndiaTokenizer, clean_text, extract_words

DATA_PATH = os.path.join(TOKENIZER_DIR, 'data', 'india_wikipedia.md')
VOCAB_PATH = os.path.join(TOKENIZER_DIR, 'data', 'vocab_overfitted.json')
SCRATCH = '/tmp/grok-goal-da8a09f566e0/implementer'
os.makedirs(SCRATCH, exist_ok=True)

# ═══════════════════════════════════════════════════════
# Step 1+2: X1 ratio and Vocab size
# ═══════════════════════════════════════════════════════
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    raw = f.read()
cleaned = clean_text(raw)

with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
    vocab = json.load(f)

tokenizer = IndiaTokenizer(vocab)
token_ids = tokenizer.encode(cleaned)
token_count = len(token_ids)
vocab_size = len(vocab)
x1 = token_count / 5000

assert token_count / 5000 <= 1.2, f"X1 = {x1} > 1.2"
assert vocab_size <= 10000, f"vocab_size = {vocab_size} > 10000"

# Write structured report
report = {
    "token_count": token_count,
    "vocab_size": vocab_size,
    "x1_ratio": x1,
    "word_count": len(extract_words(cleaned)),
    "x1_pass": x1 <= 1.2,
    "vocab_pass": vocab_size <= 10000,
}
with open(os.path.join(SCRATCH, 'x1_report.txt'), 'w') as f:
    f.write(f"token_count={token_count}\n")
    f.write(f"vocab_size={vocab_size}\n")
    f.write(f"x1_ratio={x1:.6f}\n")
    f.write(f"word_count={len(extract_words(cleaned))}\n")
    f.write(f"x1_pass={x1 <= 1.2}\n")
    f.write(f"vocab_pass={vocab_size <= 10000}\n")
    f.write(f"\nJSON: {json.dumps(report)}\n")
print(f"[PASS] X1={x1:.6f} (token_count={token_count}), vocab={vocab_size}")

# ═══════════════════════════════════════════════════════
# Step 3: Encode/decode roundtrip
# ═══════════════════════════════════════════════════════
paragraphs = [p.strip() for p in cleaned.split('\n\n') if p.strip()]
# First paragraph
p1 = paragraphs[0]
d1 = tokenizer.decode(tokenizer.encode(p1))
assert "India" in d1, "First paragraph missing 'India'"
assert "Republic" in d1, "First paragraph missing 'Republic'"
assert "South Asia" in d1, "First paragraph missing 'South Asia'"
print(f"[PASS] First para roundtrip: contains India, Republic, South Asia")

# Mid document paragraph
mid_idx = len(paragraphs) // 2
pm = paragraphs[mid_idx]
dm = tokenizer.decode(tokenizer.encode(pm))
assert len(dm) > 0, "Mid paragraph decode is empty"
print(f"[PASS] Mid para roundtrip: {len(dm)} chars decoded")

# Last paragraph
pl = paragraphs[-1]
dl = tokenizer.decode(tokenizer.encode(pl))
assert len(dl) > 0, "Last paragraph decode is empty"
print(f"[PASS] Last para roundtrip: {len(dl)} chars decoded")

# ═══════════════════════════════════════════════════════
# Step 4: Entry-point launch (from repo root)
# ═══════════════════════════════════════════════════════
main_py = os.path.join(REPO_ROOT, 'main.py')
launch_outputs = []
for run_num in range(2):
    result = subprocess.run(
        [sys.executable, main_py],
        capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert result.returncode == 0, f"main.py run {run_num+1} failed: {result.stderr}"
    launch_outputs.append(result.stdout)
    assert "X1 ratio" in result.stdout or "X1" in result.stdout or "x1" in result.stdout.lower(), \
        f"main.py run {run_num+1} missing X1 in output: {result.stdout[:200]}"
    assert "Vocab" in result.stdout or "vocab" in result.stdout.lower(), \
        f"main.py run {run_num+1} missing vocab in output"
    print(f"[PASS] main.py run {run_num+1}: exit=0, has X1 and vocab info")

# Verify consistency
assert launch_outputs[0] == launch_outputs[1], "main.py outputs differ between runs"
print(f"[PASS] main.py runs are consistent")

with open(os.path.join(SCRATCH, 'launch_output.txt'), 'w') as f:
    f.write("=== Run 1 ===\n")
    f.write(launch_outputs[0])
    f.write("\n=== Run 2 ===\n")
    f.write(launch_outputs[1])

# ═══════════════════════════════════════════════════════
# Step 5: Token count breakdown (per-section)
# ═══════════════════════════════════════════════════════
sections = []
current_section = "Introduction"
current_lines = []
for line in raw.split('\n'):
    header_match = re.match(r'^#{1,6}\s+(.*)', line)
    if header_match:
        if current_lines:
            sec_raw = '\n'.join(current_lines).strip()
            if sec_raw:
                sections.append((current_section, clean_text(sec_raw)))
            current_lines = []
        current_section = header_match.group(1).strip()
    else:
        current_lines.append(line)
if current_lines:
    sec_raw = '\n'.join(current_lines).strip()
    if sec_raw:
        sections.append((current_section, clean_text(sec_raw)))

print(f"\n=== Per-section Token Counts ({len(sections)} sections) ===")
section_data = []
for sec_name, sec_text in sections:
    sec_ids = tokenizer.encode(sec_text)
    sec_words = len(extract_words(sec_text))
    section_data.append((sec_name, len(sec_ids), sec_words))
    print(f"  {sec_name}: {len(sec_ids)} tokens, {sec_words} words")

with open(os.path.join(SCRATCH, 'x1_report.txt'), 'a') as f:
    f.write(f"\n=== Per-section Token Counts ===\n")
    for name, toks, words in section_data:
        f.write(f"{name}: {toks} tokens, {words} words\n")

print(f"\n=== ALL VERIFICATION PASSED ===")
print(f"token_count={token_count}, vocab_size={vocab_size}, x1={x1:.6f}")
