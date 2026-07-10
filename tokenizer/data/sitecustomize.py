"""sitecustomize.py - auto-adds tokenizer package to sys.path when running from data/."""
import sys
import os

# If running from the data/ subdirectory, add the tokenizer dir to path
_cwd = os.getcwd()
if os.path.basename(_cwd) == 'data':
    _parent = os.path.dirname(_cwd)
    if os.path.exists(os.path.join(_parent, 'tokenizer.py')) and _parent not in sys.path:
        sys.path.insert(0, _parent)
