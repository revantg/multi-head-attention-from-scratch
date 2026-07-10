"""conftest.py - ensures tokenizer package is importable from data/ workspace."""
import sys
import os

# Add the tokenizer directory (parent of data/) to Python path
_data_dir = os.path.dirname(os.path.abspath(__file__))
_tokenizer_dir = os.path.dirname(_data_dir)
if _tokenizer_dir not in sys.path:
    sys.path.insert(0, _tokenizer_dir)
