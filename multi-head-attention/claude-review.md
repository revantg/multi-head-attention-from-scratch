# Claude Review — `mah_transformer.py`

Review of the file with the goal of making the `Transformer` class actually
runnable. Five changes total: four are bug fixes that block a forward pass, one
adds a calling cell at the end.

Nothing has been applied to the file. Read top-to-bottom, accept/reject per
diff.

---

## Diff 1 — `PositionalEmbedding.__init__`

Bundles three independent fixes plus removal of dead code.

```diff
@@ class PositionalEmbedding(nn.Module):
         def __init__(
             self,
             embed_dims,  # dimensions for each embedding
             max_seq_len  # max number of tokens
         ):
             super().__init__()
-
-            # the job of nn.Embedding is to translate the incoming token to a embedding of a dictionary of fixed size and items
-            # we need seperate embeddings for both since this program is written for translating from one language to other. each language will have it's own dict
-            #    - encoder will have dict of source language
-            #    - decoder will have dict of target language
-            self.enc_embed = nn.Embedding(max_seq_len, embed_dims)
-            self.dec_embed = nn.Embedding(max_seq_len, embed_dims)
-
-            pe = torch.zeros(embed_dims, dtype=float)
+
+            pe = torch.zeros(max_seq_len, embed_dims)

             # fraction is just the part inside sin/cos but without the pos
             fraction = torch.exp(
                 -1 * \
-                torch.arange(0, embed_dims, 2, dtype=float) *\
+                (torch.arange(0, embed_dims, 2).float() / embed_dims) *\
                 math.log(10_000)
             )
@@
-            # this generates (max_seq_len, 1)
-            pos = torch.arange(max_seq_len, dtype=float).unsqueeze(1)
+            # this generates (max_seq_len, 1)
+            pos = torch.arange(max_seq_len).float().unsqueeze(1)
```

### Why

**1. `pe` shape is wrong.** Currently `torch.zeros(embed_dims, dtype=float)` is
1-D (length `embed_dims`). A few lines later, `pe[:, 0::2] = ...` does 2-D
indexing on it and will throw. The PE table is conceptually
`(max_seq_len, embed_dims)` — exactly the ASCII matrix drawn in the markdown
cell above the class.

**2. Formula is missing `/ embed_dims`.** The paper's denominator is
`10000^(2i / d_model)`. Your sanity-check cell on line 140 has it right:

```python
torch.arange(0, d, 2, dtype=float) / d * math.log(10_000)
```

But the class on line 173 drops the `/ d`. Without the division, the exponent
grows huge → `exp(-...)` underflows to ~0 for every `i > 0`, so nearly all
columns collapse to `sin(0) = 0` / `cos(0) = 1` and the encoding loses its
position information.

**3. `dtype=float` is float64.** `nn.Embedding` outputs float32. When the
forward pass does `x + self.pe[...]`, you'd get an upcast-to-float64 (slow,
heavier memory) or a dtype mismatch error depending on PyTorch version. Letting
`pe` default to float32 keeps it consistent with the rest of the model.

**4. Removed `enc_embed` / `dec_embed`.** They are unused — the `Transformer`
class already owns these (line 685: *"Token embeddings (moved out of
PositionalEmbedding)"*). The deleted comment block explained those two lines;
since the lines are gone, the comment dangles.

### Open question on the comment

The comment teaches a real concept (encoder vocab ≠ decoder vocab → separate
embedding tables). If you want to preserve that, the right home is above
`Transformer.__init__`'s embedding lines (686–687), where the actual embeddings
now live. Decide: drop entirely, or move it over.

---

## Diff 2 — `DecoderLayer`

```diff
-@app.cell
-def _(FeedForwardNN, MultiHeadAttention, embed_size, nn):
+@app.cell
+def _(FeedForwardNN, MultiHeadAttention, nn):
     class DecoderLayer(nn.Module):
         def __init__(self, embed_dims, ff_inner_dims, num_heads, dropout_p):
             super().__init__()

-            self.attention = MultiHeadAttention(embeddings_size=embed_size, num_attn_heads=num_heads)
-            self.cross_attention = MultiHeadAttention(embeddings_size=embed_size, num_attn_heads=num_heads)
+            self.attention = MultiHeadAttention(embeddings_size=embed_dims, num_attn_heads=num_heads)
+            self.cross_attention = MultiHeadAttention(embeddings_size=embed_dims, num_attn_heads=num_heads)


-            self.ff_nn = FeedForwardNN(inner_layer_dims=ff_inner_dims, outer_layer_dims=embed_size)
+            self.ff_nn = FeedForwardNN(inner_layer_dims=ff_inner_dims, outer_layer_dims=embed_dims)

             self.drop1 = nn.Dropout(dropout_p)
             self.drop2 = nn.Dropout(dropout_p)
             self.drop3 = nn.Dropout(dropout_p)

-            self.layer_norm1 = nn.LayerNorm(embed_size)
-            self.layer_norm2 = nn.LayerNorm(embed_size)
-            self.layer_norm3 = nn.LayerNorm(embed_size)
-
-            self.final_linear = nn.Linear(embed_dims, embed_dims)
-            self.final_softmax = nn.Softmax()
+            self.layer_norm1 = nn.LayerNorm(embed_dims)
+            self.layer_norm2 = nn.LayerNorm(embed_dims)
+            self.layer_norm3 = nn.LayerNorm(embed_dims)


         def forward(
             self,
             x,
             src_mask,
             tgt_mask,
             enc_out
         ):

-            # running self attention for decoder layer first
-            attn1_out = self.attention(x, x, x, src_mask)
+            # self-attention over the decoder's own tokens — causal mask
+            attn1_out = self.attention(x, x, x, tgt_mask)
             norm1_out = self.layer_norm1(self.drop1(attn1_out) + x)

-            # ask questions to enc_out (which has understanding of the source sentence)
-            #   we use our own Q (question) but K and V of encoder
-            attn2_out = self.cross_attention(norm1_out, enc_out, enc_out, tgt_mask)
+            # cross-attention: Q from decoder, K/V from encoder.
+            # mask is the encoder's padding mask — don't attend to <PAD>
+            attn2_out = self.cross_attention(norm1_out, enc_out, enc_out, src_mask)
             norm2_out = self.layer_norm2(norm1_out + self.drop2(attn2_out))

             nn_out = self.ff_nn(norm2_out)
             layer3_out = self.layer_norm3(self.drop3(nn_out) + norm2_out)

-            return self.final_softmax(self.final_linear(layer3_out))
+            return layer3_out
```

### Why

**1. `embed_size` is undefined.** The marimo cell signature lists `embed_size`
as a dependency, but it's never defined anywhere in the notebook. The cell
would fail to load. The constructor argument is named `embed_dims` — use it
throughout the body.

**2. Masks are swapped.** This is the most important conceptual fix.

- **Self-attention** in the decoder needs the **causal** mask so position `i`
  cannot peek at positions `> i`. That mask is `tgt_mask`.
- **Cross-attention** reads K/V from the encoder. The mask there exists to
  prevent attending to padding tokens in the *source* sentence. That mask is
  `src_mask`.

Current code passes `src_mask` to self-attention and `tgt_mask` to
cross-attention — exactly backwards.

**3. `final_linear` + `Softmax` shouldn't live in a single decoder layer.**

Two problems with the current placement:

- It runs **once per layer**. A 6-layer decoder would project to vocab six
  times and feed vocab-shaped tensors into the next layer. The next layer's
  `LayerNorm(embed_dims)` and attention would break.
- `Transformer.final_linear` (line 709) already does the embed→vocab
  projection at the end. Having it here too is double work even for a 1-layer
  decoder.

Also: cross-entropy loss in PyTorch expects **logits**, not softmax output.
Applying `Softmax` before loss is a common bug — it makes gradients worse and
is numerically less stable than letting `CrossEntropyLoss` do its own
log-softmax internally. So the softmax goes away entirely.

---

## Diff 3 — `EncodeBlock`

```diff
-@app.cell
-def _(EncoderLayer, nn, num_end):
+@app.cell
+def _(EncoderLayer, nn):
     class EncodeBlock(nn.Module):
         def __init__(
             self,
             num_enc,
             embed_dims,
             num_heads,
             dropout_p,
             ff_nn_dims
         ):
             super().__init__()

-            self.encoder_layer = nn.ModuleList(modules=[
+            self.encoder_layers = nn.ModuleList(modules=[
                 EncoderLayer(embed_dims, num_heads, dropout_p, ff_nn_dims)
-                for _ in range(num_end)
+                for _ in range(num_enc)
             ])

         def forward(
             self,
             x,
             mask,
         ):
-            return self.encoder_layer(x, mask)
+            for layer in self.encoder_layers:
+                x = layer.encode(x, mask)
+            return x
```

### Why

**1. `num_end` is a typo for `num_enc`.** Marimo treats undefined names as
external cell inputs, so `num_end` shows up in the cell signature. The cell
won't run unless `num_end` is somehow defined elsewhere (it isn't).

**2. `nn.ModuleList` is not callable.** It's a container — you must iterate
over it. Each `EncoderLayer` exposes its forward logic as `.encode(...)`, not
`.forward(...)`, so the loop calls `layer.encode(x, mask)`.

**3. Renamed to plural `encoder_layers`** to match the fact that it holds
multiple layers. Cosmetic, but worth doing while we're touching the name.

---

## Diff 4 — `DecoderBlock`

```diff
         def forward(self, x, src_mask, tgt_mask, enc_out):
-            return self.decoder_layers(x, src_mask, tgt_mask, enc_out)
+            for layer in self.decoder_layers:
+                x = layer(x, src_mask, tgt_mask, enc_out)
+            return x
```

### Why

Same `ModuleList`-isn't-callable bug as Diff 3. `DecoderLayer` defines
`forward` (not `encode`), so a direct `layer(...)` call works here.

---

## Diff 5 — add a cell that instantiates and calls `Transformer`

Append a new cell before the trailing empty one:

```diff
+@app.cell
+def _(Transformer, torch):
+    # tiny toy config so it runs on CPU
+    src_vocab_size = 100
+    tgt_vocab_size = 120
+    embed_dims = 64
+    num_heads = 4
+    num_encoder_layers = 2
+    num_decoder_layers = 2
+    ff_nn_dims = 128
+    max_seq_len = 16
+
+    model = Transformer(
+        src_vocab_size=src_vocab_size,
+        tgt_vocab_size=tgt_vocab_size,
+        embed_dims=embed_dims,
+        num_heads=num_heads,
+        num_encoder_layers=num_encoder_layers,
+        num_decoder_layers=num_decoder_layers,
+        ff_nn_dims=ff_nn_dims,
+        max_seq_len=max_seq_len,
+    )
+
+    batch = 2
+    src_len = 7
+    tgt_len = 5
+
+    src = torch.randint(0, src_vocab_size, (batch, src_len))
+    tgt = torch.randint(0, tgt_vocab_size, (batch, tgt_len))
+
+    # src padding mask: (batch, 1, 1, src_len) — broadcasts across heads & query positions.
+    # 1 = keep, 0 = mask. (All ones here since our toy input has no <PAD>.)
+    src_mask = torch.ones(batch, 1, 1, src_len, dtype=torch.bool)
+
+    # tgt mask: causal (lower-triangular), shape (1, 1, tgt_len, tgt_len) — broadcasts
+    # across batch & heads. Position i can attend to positions 0..i.
+    causal = torch.tril(torch.ones(tgt_len, tgt_len, dtype=torch.bool))
+    tgt_mask = causal.unsqueeze(0).unsqueeze(0)
+
+    logits = model(src, tgt, src_mask, tgt_mask)
+    print("logits shape:", logits.shape)  # expect (2, 5, 120)
+    return
```

### Why this shape of masks

`MultiHeadAttention.forward` computes `attn_scores` of shape
`(batch, heads, seq_q, seq_k)`. The masks need to broadcast against that:

- **`src_mask`** is `(batch, 1, 1, src_len)` — same across heads, same across
  every query position, varies per source key position. Standard.
- **`tgt_mask`** is `(1, 1, tgt_len, tgt_len)` — same across batch and heads,
  but a full 2-D matrix over `(query_pos, key_pos)` so each query position has
  its own causal cutoff.

Both broadcast cleanly onto `attn_scores`.

### Not touching

`MultiHeadAttention` line 341 uses `masked_fill(mask == 0, -1e9)`. With a bool
mask, `mask == 0` evaluates to `~mask`, which is fine — keeps the convention
that `1 / True = keep`, `0 / False = mask out`. The cell above follows that.

---

## Summary

| # | Where | Type |
|---|---|---|
| 1 | `PositionalEmbedding.__init__` | bug fix (shape, formula, dtype) + dead-code removal |
| 2 | `DecoderLayer` | bug fix (undefined name, swapped masks, redundant final projection) |
| 3 | `EncodeBlock` | bug fix (typo, ModuleList iteration) |
| 4 | `DecoderBlock` | bug fix (ModuleList iteration) |
| 5 | new cell at end | adds a working call site |

Diffs 1–4 are needed for *anything* to run. Diff 5 is the actual "call the
transformer" task.
