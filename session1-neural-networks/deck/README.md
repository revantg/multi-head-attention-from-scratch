# Session 1 — Neural Networks to the Transformer (slideshow)

**▶ Live: https://s1-neural-nets-to-transformer.netlify.app**
(the full Jupyter notebook is linked from the cover, and served at [`/notebook`](https://s1-neural-nets-to-transformer.netlify.app/notebook))

A static, keyboard-navigable slideshow that proves the four Session-1 claims. Nike-inspired
editorial design: white canvas, ink-black display type (Bebas Neue), soft-cloud staged charts,
pill controls. No build step, no dependencies — pure HTML/CSS/JS.

## The assignment (as given)

> ### S1-1 · Activations exist for a reason.
> **Claim:** a model with no nonlinearity can only draw a straight boundary, so it cannot separate two interleaved/concentric rings; adding one ReLU hidden layer can. **Build:** generate ~300 noisy 2D points as two rings (inner = class 0, outer = class 1), not linearly separable. Train (a) a single linear layer + sigmoid, (b) one ReLU hidden layer. **Proof:** plot both decision boundaries — the linear one is a straight line stuck near 55% accuracy, the ReLU one wraps the ring to ~99%. Only the activation changed. The boundary picture is the money shot.
>
> ### S1-2 · Depth without nonlinearity is a lie.
> **Claim:** five stacked linear layers collapse to a single linear map, so a 5-layer linear net is no stronger than 1 layer; both fail the ring task identically, and inserting ReLUs between the same five layers suddenly solves it. **Build:** same ring data; train 1 linear layer, 5 linear layers (no activations), then 5 layers + ReLU. **Proof:** the 1-layer and 5-linear-layer accuracies and boundaries are identical (both a line); ReLU breaks the tie. **Bonus that nails it:** multiply the five weight matrices numerically and show the product is one matrix.
>
> ### S1-3 · Embeddings learn similarity from nothing but next-token.
> **Claim:** trained only to predict the next token in a tiny synthetic grammar, the embedding table clusters related tokens, though similarity was never supplied. **Build:** a toy language with categories (animals: cat dog cow; fruits: apple mango; verbs: eat chase see) and templates like `<animal> <verb> <fruit>`, so same-category tokens share next-token distributions. Train a tiny embedding→softmax next-token model. **Proof:** project the learned embeddings to 2D and plot — same-category tokens land together; nearest neighbors are same-category. Emergent clustering = the proof.
>
> ### S1-4 · Memorization vs generalization, and data closes the gap.
> **Claim:** a high-capacity model on tiny data drives train loss to ~0 while held-out loss stays high; growing the dataset closes the gap. **Build:** a learnable noisy classification with a held-out split; train an over-parameterized net at train sizes 20, 200, 2000. **Proof:** the train/test gap is huge at 20 (train→0, test bad) and shrinks as data grows — plot the generalization gap vs dataset size. Ties straight into the course's "data is everything."
>
> **What are you submitting.** You are working with your agents to design a beautiful webapp which proves these 4 points, and then uploading on [netlify](https://app.netlify.com/drop) and then sharing the link. Make an account on Netlify to keep it permanent.

## The four proofs (what this deck shows)
- **S1·1 Activations** — a linear layer stalls at 61% on the rings; one ReLU hidden layer wraps them to 100%.
- **S1·2 Depth** — 5 stacked linear layers = 1 (both a straight line, ~50–61%); ReLU breaks the tie to 100%. Product of the 5 weight matrices collapses to a single 1×2 map.
- **S1·3 Embeddings** — a next-token-only model clusters same-category words in 2-D; 6/8 nearest neighbours share a category.
- **S1·4 Data** — an over-parameterized net's generalization gap falls from 1.62 (n=20) to −0.27 (n=2000, where test loss even beats train); a single grid shows the boundary go from memorized islands to a clean ring.

## Navigate
- `→ / ←`, `Space`, `PageUp/Down`, `Home/End`, scroll wheel, swipe, dots, or the pill buttons.

## Regenerate the proof charts
All charts are trained live and rendered in the Nike light palette. Numbers are deterministic (fixed seeds).

```bash
uv run python gen_assets.py   # writes PNGs + results.json into assets/
```

## Deploy to Netlify
1. Go to https://app.netlify.com/drop
2. Drag **this `deck/` folder** onto the drop zone.
3. Share the URL. (Sign in first to make the deploy permanent.)
