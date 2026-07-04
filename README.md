# Transformer Learning

Personal repo for learning different attention mechanisms and transformer architectures.

## Setup

```bash
uv sync
```

## Structure

- **`session1-neural-networks/`** - School of AI "Session 1: From Neural Networks to the Transformer" coursework. A static slideshow webapp (`deck/`) proves the four session claims — activations, depth, embeddings, data — with our own notebook code and outputs shown as proof.
  - Live: https://s1-neural-nets-to-transformer.netlify.app
  - Notebook: `session1-neural-networks/Session1_Assignment.ipynb` · brief + details: `session1-neural-networks/deck/README.md`

Each remaining directory covers a specific attention type:

- **`multi-head-attention/`** - Self-attention with multiple heads (Vaswani et al. 2017)
  - Run: `uv run --project multi-head-attention marimo edit multi-head-attention/mah_transformer.py`

## Future Topics

- `cross-head-attention/` - Encoder-decoder cross attention
- `linear-attention/` - Efficient attention variants
- `flash-attention/` - Memory-efficient attention

## Quick Start

Each subdirectory has its own `pyproject.toml` with dependencies managed by [uv](https://docs.astral.sh/uv/).
