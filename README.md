# Transformer Learning

Personal repo for learning different attention mechanisms and transformer architectures.

## Setup

```bash
uv sync
```

## Structure

- **`session1-neural-networks/`** - School of AI "Session 1: From Neural Networks to the Transformer" coursework. A static slideshow webapp (`deck/`) proves the four session claims — activations, depth, embeddings, data — with our own notebook code and outputs shown as proof.
  - Live: https://s1-neural-nets-to-transformer.netlify.app (mirror: https://revant-s1-neural-nets.surge.sh)
  - Notebook: `session1-neural-networks/Session1_Assignment.ipynb` · brief + details: `session1-neural-networks/deck/README.md`
- **`session2-tokenizer/`** - School of AI "Session 2" tokenizer assignment (git submodule). A multilingual **BPE tokenizer** over India's Wikipedia page in English, Hindi, Telugu and Sanskrit, sharing one 10,000-token vocabulary; score is `1000 / (max X − min X)` where `X = tokens / words` per language. The submitted design (**Approach B**) runs a short BPE base (~2,500 slots) then fills the rest with whole words, giving priority to whichever language is still above the 1.2 tokens/word bar — so all four land close together (X = 1.158 / 1.085 / 1.050 / 1.110), self-score **≈ 9237**, zero unknown tokens.
  - Live: https://revant-tokenizer-slides.surge.sh (mirror: https://tokenizer-india-slides-rev.netlify.app/)
  - Brief + method + per-language numbers: `session2-tokenizer/README.md`

Each remaining directory covers a specific attention type:

- **`multi-head-attention/`** - Self-attention with multiple heads (Vaswani et al. 2017)
  - Run: `uv run --project multi-head-attention marimo edit multi-head-attention/mah_transformer.py`

## Future Topics

- `cross-head-attention/` - Encoder-decoder cross attention
- `linear-attention/` - Efficient attention variants
- `flash-attention/` - Memory-efficient attention

## Quick Start

Each subdirectory has its own `pyproject.toml` with dependencies managed by [uv](https://docs.astral.sh/uv/).
