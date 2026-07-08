#!/usr/bin/env python3
"""
Comprehensive verification tests for the India Wikipedia Tokenizer.
Validates all acceptance criteria from the plan.
"""
import os
import sys
import json
import re
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenizer import IndiaTokenizer, clean_text, extract_words

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, 'data', 'india_wikipedia.md')
VOCAB_PATH = os.path.join(BASE, 'data', 'vocab_overfitted.json')
SCRATCH = '/tmp/grok-goal-da8a09f566e0/implementer'

os.makedirs(SCRATCH, exist_ok=True)

results = []

def log(msg):
    print(msg)
    results.append(msg)

def test(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    line = f"  [{status}] {name}"
    if detail:
        line += f" — {detail}"
    log(line)
    return passed


# ────────────────────────────────────────────────
# Load data and tokenizer
# ────────────────────────────────────────────────
log("=" * 60)
log("VERIFICATION: India Wikipedia Tokenizer")
log("=" * 60)

with open(DATA_PATH, 'r', encoding='utf-8') as f:
    raw = f.read()
cleaned = clean_text(raw)
words = extract_words(cleaned)

with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
    vocab = json.load(f)

tokenizer = IndiaTokenizer(vocab)

# ── Test 1: Vocab size ≤ 10,000 ──
log("\n--- Gating: Vocab Size ---")
vocab_size = len(vocab)
test("Vocab size ≤ 10,000", vocab_size <= 10000, f"actual={vocab_size}")

# ── Test 2: X1 ratio ≤ 1.2 ──
log("\n--- Gating: X1 Ratio ---")
token_ids = tokenizer.encode(cleaned)
token_count = len(token_ids)
x1 = token_count / 5000
test("X1 ≤ 1.2", x1 <= 1.2, f"token_count={token_count}, X1={x1:.6f}")

# ── Test 3: Encode/decode roundtrip ──
log("\n--- Gating: Encode/Decode Roundtrip ---")
paragraphs = [p.strip() for p in cleaned.split('\n\n') if p.strip()]

# First paragraph
p1 = paragraphs[0]
ids1 = tokenizer.encode(p1)
dec1 = tokenizer.decode(ids1)
test("First para exact match", dec1 == p1, f"tokens={len(ids1)}")
test("First para contains 'India'", "India" in dec1)
test("First para contains 'Republic'", "Republic" in dec1)
test("First para contains 'South Asia'", "South Asia" in dec1)

# Middle paragraph
mid = len(paragraphs) // 2
pm = paragraphs[mid]
idsm = tokenizer.encode(pm)
decm = tokenizer.decode(idsm)
test("Middle para exact match", decm == pm, f"tokens={len(idsm)}")

# Last paragraph
pl = paragraphs[-1]
idsl = tokenizer.encode(pl)
decl = tokenizer.decode(idsl)
test("Last para exact match", decl == pl, f"tokens={len(idsl)}")

# ── Test 4: Entry point launches cleanly ──
log("\n--- Gating: Entry Point Launch ---")
for run_num in range(2):
    result = subprocess.run(
        [sys.executable, os.path.join(BASE, 'main.py')],
        capture_output=True, text=True, cwd=BASE
    )
    test(f"main.py run {run_num+1} exits 0", result.returncode == 0,
         f"exit={result.returncode}")
    if run_num == 0:
        # Capture first run's output
        first_output = result.stdout

# ── Test 5: Token count breakdown per section ──
log("\n--- Evidence: Per-section Token Counts ---")
# Parse sections from RAW markdown (headers preserved), then clean each
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

for sec_name, sec_text in sections[:10]:  # Show first 10 sections
    sec_ids = tokenizer.encode(sec_text)
    sec_words = len(extract_words(sec_text))
    log(f"  Section '{sec_name[:40]}': {len(sec_ids)} tokens, {sec_words} words")

# ── Test 6: Multilingual tokens present ──
log("\n--- Evidence: Multilingual Coverage ---")
hindi_count = sum(1 for w in vocab if any('\u0900' <= c <= '\u097F' for c in w))
telugu_count = sum(1 for w in vocab if any('\u0C00' <= c <= '\u0C7F' for c in w))
test("Hindi tokens present", hindi_count > 0, f"count={hindi_count}")
test("Telugu tokens present", telugu_count > 0, f"count={telugu_count}")

# ── Summary ──
log("\n" + "=" * 60)
total_tests = sum(1 for r in results if "[PASS]" in r or "[FAIL]" in r)
passed_tests = sum(1 for r in results if "[PASS]" in r)
failed_tests = total_tests - passed_tests
log(f"RESULTS: {passed_tests}/{total_tests} passed, {failed_tests} failed")
log(f"Vocab: {vocab_size}, Tokens: {token_count}, X1: {x1:.6f}")
log("=" * 60)

# Save evidence to scratch dir
report_path = os.path.join(SCRATCH, 'x1_report.txt')
with open(report_path, 'w') as f:
    f.write('\n'.join(results))
log(f"\nReport saved to {report_path}")

# Save launch output
launch_path = os.path.join(SCRATCH, 'launch_output.txt')
with open(launch_path, 'w') as f:
    f.write(first_output)
log(f"Launch output saved to {launch_path}")

sys.exit(0 if failed_tests == 0 else 1)
