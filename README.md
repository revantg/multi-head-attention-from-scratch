# Transformer Learning

Personal repo for learning different attention mechanisms and transformer architectures.

## Setup

```bash
uv sync
```

## Structure

Each directory covers a specific attention type:

- **`multi-head-attention/`** - Self-attention with multiple heads (Vaswani et al. 2017)
  - Run: `uv run --project multi-head-attention marimo edit multi-head-attention/mah_transformer.py`

## Future Topics

- `cross-head-attention/` - Encoder-decoder cross attention
- `linear-attention/` - Efficient attention variants
- `flash-attention/` - Memory-efficient attention

## Quick Start

Each subdirectory has its own `pyproject.toml` with dependencies managed by [uv](https://docs.astral.sh/uv/).
