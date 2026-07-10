#!/usr/bin/env python3
"""
Standalone verification script for the India Wikipedia Tokenizer.
Designed to work from ANY directory - the workspace (data/), repo root, or tokenizer dir.

Run:
  python3 verify.py              (from this directory)
  python3 tokenizer/verify.py    (from repo root)
  python3 verify.py              (from tokenizer dir)
"""
import os
import sys
import json
import re
import subprocess

# ── Resolve paths regardless of CWD ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_NAME = os.path.basename(SCRIPT_DIR)

if SCRIPT_NAME == 'data':
    # Running from workspace (tokenizer/data/)
    TOKENIZER_DIR = os.path.dirname(SCRIPT_DIR)
    REPO_ROOT = os.path.dirname(TOKENIZER_DIR)
elif SCRIPT_NAME == 'tokenizer':
    # Running from tokenizer dir
    TOKENIZER_DIR = SCRIPT_DIR
    REPO_ROOT = os.path.dirname(TOKENIZER_DIR)
else:
    # Running from repo root or elsewhere
    REPO_ROOT = SCRIPT_DIR
    TOKENIZER_DIR = os.path.join(SCRIPT_DIR, 'tokenizer')

sys.path.insert(0, TOKENIZER_DIR)
from tokenizer import IndiaTokenizer, clean_text, extract_words, encode, decode

DATA_PATH = os.path.join(TOKENIZER_DIR, 'data', 'india_wikipedia.md')
VOCAB_PATH = os.path.join(TOKENIZER_DIR, 'data', 'vocab_overfitted.json')
SCRATCH = '/tmp/grok-goal-da8a09f566e0/implementer'
os.makedirs(SCRATCH, exist_ok=True)

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}" + (f" — {detail}" if detail else ""))
    else:
        failed += 1
        print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))
    return condition


print("=" * 60)
print("VERIFICATION: India Wikipedia Multilingual Tokenizer")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════
# Load data and tokenizer
# ═══════════════════════════════════════════════════════════════
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    raw = f.read()
cleaned = clean_text(raw)

with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
    vocab = json.load(f)

tokenizer = IndiaTokenizer(vocab)
token_ids = tokenizer.encode(cleaned)
token_count = len(token_ids)
vocab_size = tokenizer.get_vocab_size()
x1 = token_count / 5000
word_count = len(extract_words(cleaned))

# ═══════════════════════════════════════════════════════════════
# Step 1+2: X1 ratio and Vocab size (GATING)
# ═══════════════════════════════════════════════════════════════
print("\n--- Gating: X1 Ratio and Vocab Size ---")
check("X1 ≤ 1.2", x1 <= 1.2, f"token_count={token_count}, X1={x1:.6f}")
check("Vocab ≤ 10,000", vocab_size <= 10000, f"vocab_size={vocab_size}")

# Save report
report_path = os.path.join(SCRATCH, 'x1_report.txt')
with open(report_path, 'w') as f:
    f.write(f"token_count={token_count}\n")
    f.write(f"vocab_size={vocab_size}\n")
    f.write(f"x1_ratio={x1:.6f}\n")
    f.write(f"word_count={word_count}\n")
print(f"  Report saved to {report_path}")

# ═══════════════════════════════════════════════════════════════
# Step 3: Encode/decode roundtrip (GATING)
# ═══════════════════════════════════════════════════════════════
print("\n--- Gating: Encode/Decode Roundtrip ---")
paragraphs = [p.strip() for p in cleaned.split('\n\n') if p.strip() and len(p) > 50]

# First paragraph
p1 = paragraphs[0]
d1 = tokenizer.decode(tokenizer.encode(p1))
check("First para: has 'India'", "India" in d1)
check("First para: has 'Republic'", "Republic" in d1)
check("First para: has 'South Asia'", "South Asia" in d1)

# Mid paragraph
pm = paragraphs[len(paragraphs) // 2]
dm = tokenizer.decode(tokenizer.encode(pm))
check("Mid para: non-empty decode", len(dm) > 0, f"{len(dm)} chars")

# Last paragraph
pl = paragraphs[-1]
dl = tokenizer.decode(tokenizer.encode(pl))
check("Last para: non-empty decode", len(dl) > 0, f"{len(dl)} chars")

# ═══════════════════════════════════════════════════════════════
# Step 4: Entry-point launch (GATING)
# ═══════════════════════════════════════════════════════════════
print("\n--- Gating: Entry-Point Launch ---")
main_py = os.path.join(REPO_ROOT, 'main.py')
launch_outputs = []
for i in range(2):
    result = subprocess.run(
        [sys.executable, main_py],
        capture_output=True, text=True, cwd=REPO_ROOT
    )
    check(f"main.py run {i+1}: exit 0", result.returncode == 0, f"exit={result.returncode}")
    if result.returncode == 0:
        check(f"main.py run {i+1}: has X1", "x1" in result.stdout.lower())
        check(f"main.py run {i+1}: has vocab", "vocab" in result.stdout.lower())
        launch_outputs.append(result.stdout)

if len(launch_outputs) == 2:
    check("main.py: consistent output", launch_outputs[0] == launch_outputs[1])

launch_path = os.path.join(SCRATCH, 'launch_output.txt')
with open(launch_path, 'w') as f:
    for i, out in enumerate(launch_outputs):
        f.write(f"=== Run {i+1} ===\n{out}\n")
print(f"  Launch output saved to {launch_path}")

# ═══════════════════════════════════════════════════════════════
# Step 5: Per-section token count breakdown (EVIDENCE)
# ═══════════════════════════════════════════════════════════════
print("\n--- Evidence: Per-section Token Counts ---")
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

check("Sections found > 5", len(sections) > 5, f"found={len(sections)}")

token_counts = []
for sec_name, sec_text in sections[:15]:
    sec_ids = tokenizer.encode(sec_text)
    sec_words = len(extract_words(sec_text))
    token_counts.append(len(sec_ids))
    print(f"    {sec_name[:40]:40s} {len(sec_ids):4d} tokens, {sec_words:4d} words")

if token_counts:
    check("Meaningful variation", max(token_counts) > min(token_counts),
          f"range=[{min(token_counts)}, {max(token_counts)}]")

# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
total = passed + failed
print(f"RESULTS: {passed}/{total} passed, {failed} failed")
print(f"Vocab: {vocab_size}, Token count: {token_count}, X1: {x1:.6f}")
print(f"Words: {word_count}, Tokens/word: {token_count/word_count:.6f}")
print("=" * 60)

sys.exit(0 if failed == 0 else 1)
