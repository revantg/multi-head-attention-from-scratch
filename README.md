# Transformer Learning

Personal repo for learning different attention mechanisms and transformer architectures.

## Structure

Each directory covers a specific attention type:

- **`multi-head-attention/`** - Self-attention with multiple heads (Vaswani et al. 2017)
  - Run: `cd multi-head-attention && marimo edit mah_transformer.py`

## Future Topics

- `cross-head-attention/` - Encoder-decoder cross attention
- `linear-attention/` - Efficient attention variants
- `flash-attention/` - Memory-efficient attention

## Quick Start

Each subdirectory has its own `requirements.txt` and README.

## Resources Used

The following resources were used to create these learning notebooks:

1. **DataCamp Tutorial** - [Building a Transformer with PyTorch](https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch)

2. **Aladdin Persson's Transformer from Scratch**
   - Video: [Transformer from Scratch](https://www.youtube.com/watch?v=U0s0f995w14)
   - Code: [GitHub - transformer_from_scratch.py](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/transformer_from_scratch/transformer_from_scratch.py)

3. **Harvard's Annotated Transformer** - [NLP @ Harvard](https://nlp.seas.harvard.edu/annotated-transformer/) (with new PyTorch)

4. **Claude AI Conversation** - [My main learning conversation with Claude](https://claude.ai/share/08fa5e8a-fa18-44e6-a633-92296656d933) (publicly shared)
