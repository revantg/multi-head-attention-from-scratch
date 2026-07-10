#!/usr/bin/env python3
"""
Quick verification from repo root.
Usage: python3 verify_tokenizer.py
"""
import subprocess
import sys
import os

repo_root = os.path.dirname(os.path.abspath(__file__))
verify_script = os.path.join(repo_root, 'tokenizer', 'data', 'verify.py')

print(f"Running verification from {repo_root}...")
result = subprocess.run([sys.executable, verify_script], cwd=repo_root)
sys.exit(result.returncode)
