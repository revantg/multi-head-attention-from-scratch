# Session 1 · From Neural Networks to the Transformer

Assignments for School of AI Session 1, worked in `Session1_Assignment.ipynb`.

- **S1-1** — activations: single Linear (~0.54 acc) vs one ReLU hidden layer (1.0 acc) on two concentric rings; plus a deep-linear-collapse proof (5 stacked linears == one matrix).
- **S1-3** — embeddings learn similarity from next-token prediction alone; train a tiny `embedding → linear → softmax` model on a categorized toy grammar, then project embeddings to 2D and show same-category clustering + nearest-neighbor proof.
- **S1-4** — memorization vs generalization: an over-parameterized net on noisy two-rings data at train sizes 20/200/2000; the train/test generalization gap shrinks as data grows.

## Run

```bash
uv sync
uv run jupyter lab Session1_Assignment.ipynb
```
