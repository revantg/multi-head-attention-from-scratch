# Multi-Head Attention

Interactive notebook for learning multi-head attention mechanisms.

## Files

- `mah_transformer.py` - Marimo notebook with:
  - Positional encoding math and implementation
  - Multi-head attention (Q, K, V transformations)
  - Transformer encoder architecture
- `images/` - SVG diagrams referenced by the notebook

## Run

```bash
pip install -r requirements.txt
marimo edit mah_transformer.py
```

## Resources Used

The following resources were used to create this learning notebook:

1. **DataCamp Tutorial** - [Building a Transformer with PyTorch](https://www.datacamp.com/tutorial/building-a-transformer-with-py-torch)

2. **Aladdin Persson's Transformer from Scratch**
   - Video: [Transformer from Scratch](https://www.youtube.com/watch?v=U0s0f995w14)
   - Code: [GitHub - transformer_from_scratch.py](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/transformer_from_scratch/transformer_from_scratch.py)

3. **Harvard's Annotated Transformer** - [NLP @ Harvard](https://nlp.seas.harvard.edu/annotated-transformer/) (with new PyTorch)

4. **Claude AI Conversation** - [My main learning conversation with Claude](https://claude.ai/share/08fa5e8a-fa18-44e6-a633-92296656d933) (publicly shared)
