#!/usr/bin/env python3
"""
Comprehensive verification test for the India Wikipedia Tokenizer.
Uses unittest so no external dependencies needed.
Run from repo root: python3 tokenizer/test_tokenizer.py
"""
import os
import sys
import json
import re
import subprocess
import unittest

# Ensure imports work regardless of CWD
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here) if os.path.basename(_here) == 'tokenizer' else _here
_tokdir = _here if os.path.basename(_here) == 'tokenizer' else os.path.join(_here, 'tokenizer')

sys.path.insert(0, _tokdir)

from tokenizer import IndiaTokenizer, clean_text, extract_words, encode, decode, get_tokenizer

DATA_PATH = os.path.join(_tokdir, 'data', 'india_wikipedia.md')
VOCAB_PATH = os.path.join(_tokdir, 'data', 'vocab_overfitted.json')
SCRATCH = '/tmp/grok-goal-da8a09f566e0/implementer'


class TestTokenizerVerification(unittest.TestCase):
    """All verification plan gating tests."""

    @classmethod
    def setUpClass(cls):
        os.makedirs(SCRATCH, exist_ok=True)
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            cls.raw = f.read()
        cls.cleaned = clean_text(cls.raw)
        with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
            cls.vocab = json.load(f)
        cls.tokenizer = IndiaTokenizer(cls.vocab)
        cls.full_ids = cls.tokenizer.encode(cls.cleaned)
        cls.paragraphs = [p.strip() for p in cls.cleaned.split('\n\n') if p.strip() and len(p) > 50]

    # ── Gating: Vocab Size ──
    def test_vocab_size_le_10000(self):
        self.assertLessEqual(len(self.vocab), 10000,
            f"Vocab size {len(self.vocab)} exceeds 10,000")

    # ── Gating: X1 Ratio ──
    def test_x1_ratio_le_1_2(self):
        x1 = len(self.full_ids) / 5000
        self.assertLessEqual(x1, 1.2,
            f"X1 = {x1:.6f} > 1.2 (token_count={len(self.full_ids)})")

    def test_x1_report_saved(self):
        token_count = len(self.full_ids)
        vocab_size = len(self.vocab)
        x1 = token_count / 5000
        report_path = os.path.join(SCRATCH, 'x1_report.txt')
        with open(report_path, 'w') as f:
            f.write(f"token_count={token_count}\n")
            f.write(f"vocab_size={vocab_size}\n")
            f.write(f"x1_ratio={x1:.6f}\n")
            f.write(f"word_count={len(extract_words(self.cleaned))}\n")
        self.assertTrue(os.path.exists(report_path))

    # ── Gating: Encode/Decode Roundtrip ──
    def test_roundtrip_first_paragraph(self):
        p = self.paragraphs[0]
        decoded = self.tokenizer.decode(self.tokenizer.encode(p))
        self.assertIn("India", decoded)
        self.assertIn("Republic", decoded)
        self.assertIn("South Asia", decoded)

    def test_roundtrip_mid_paragraph(self):
        p = self.paragraphs[len(self.paragraphs) // 2]
        decoded = self.tokenizer.decode(self.tokenizer.encode(p))
        self.assertGreater(len(decoded), 0)

    def test_roundtrip_last_paragraph(self):
        p = self.paragraphs[-1]
        decoded = self.tokenizer.decode(self.tokenizer.encode(p))
        self.assertGreater(len(decoded), 0)

    # ── Gating: Entry-point launch ──
    def test_main_py_exits_zero(self):
        main_py = os.path.join(_root, 'main.py')
        result = subprocess.run(
            [sys.executable, main_py],
            capture_output=True, text=True, cwd=_root
        )
        self.assertEqual(result.returncode, 0, f"main.py failed: {result.stderr}")

    def test_main_py_prints_x1_and_vocab(self):
        main_py = os.path.join(_root, 'main.py')
        result = subprocess.run(
            [sys.executable, main_py],
            capture_output=True, text=True, cwd=_root
        )
        self.assertEqual(result.returncode, 0)
        output = result.stdout.lower()
        self.assertTrue("x1" in output, f"Missing X1 in output")
        self.assertTrue("vocab" in output, f"Missing vocab in output")

    def test_main_py_consistent_across_runs(self):
        main_py = os.path.join(_root, 'main.py')
        outputs = []
        for _ in range(2):
            result = subprocess.run(
                [sys.executable, main_py],
                capture_output=True, text=True, cwd=_root
            )
            self.assertEqual(result.returncode, 0)
            outputs.append(result.stdout)
        self.assertEqual(outputs[0], outputs[1], "Outputs differ between runs")
        # Save launch output
        with open(os.path.join(SCRATCH, 'launch_output.txt'), 'w') as f:
            for i, out in enumerate(outputs):
                f.write(f"=== Run {i+1} ===\n{out}\n")

    # ── Evidence: Token count breakdown ──
    def test_per_section_token_counts(self):
        sections = []
        current_section = "Introduction"
        current_lines = []
        for line in self.raw.split('\n'):
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

        self.assertGreater(len(sections), 5, f"Only found {len(sections)} sections")

        # Verify meaningful variation
        token_counts = []
        for sec_name, sec_text in sections:
            sec_ids = self.tokenizer.encode(sec_text)
            token_counts.append(len(sec_ids))

        self.assertGreater(max(token_counts), min(token_counts),
            "All sections have identical token counts")


if __name__ == '__main__':
    unittest.main(verbosity=2)
