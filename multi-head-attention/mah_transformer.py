import marimo

__generated_with = "0.23.5"
app = marimo.App()


@app.cell
def _():
    import torch.nn as nn
    import torch
    import math

    import marimo as mo

    return math, mo, nn, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ![](https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1-727x1024.png)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The positional encoding formulas

    Here's the original formulation from "Attention Is All You Need" (Vaswani et al., 2017):

    For position $pos$ in the sequence and dimension $i$ in the embedding vector:

    $$
    PE_{(pos,\, 2i)} = \sin\!\left( \frac{pos}{10000^{2i/d_{\text{model}}}} \right)
    $$

    $$
    PE_{(pos,\, 2i+1)} = \cos\!\left( \frac{pos}{10000^{2i/d_{\text{model}}}} \right)
    $$

    ## What each symbol means

    - $pos$ — the position index in the sequence, $pos \in \{0, 1, 2, \dots, \text{max\_seq\_length} - 1\}$
    - $i$ — the dimension-pair index, $i \in \{0, 1, 2, \dots, d_{\text{model}}/2 - 1\}$
    - $d_{\text{model}}$ — the embedding dimension (e.g., 512 in the base transformer)
    - $PE_{(pos,\, k)}$ — the $k$-th entry of the positional encoding vector for position $pos$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What $PE$ actually is

    $PE$ is a **2D table** (a matrix) of numbers. Each row corresponds to a position in the sequence, and each row is a $d_{\text{model}}$-dimensional vector.

    If $d_{\text{model}} = 8$ (using a small number for clarity) and $\text{max\_seq\_length} = 4$, the table looks like this:

    ```
                  dim 0    dim 1    dim 2    dim 3    dim 4    dim 5    dim 6    dim 7
    position 0  [ PE(0,0)  PE(0,1)  PE(0,2)  PE(0,3)  PE(0,4)  PE(0,5)  PE(0,6)  PE(0,7) ]
    position 1  [ PE(1,0)  PE(1,1)  PE(1,2)  PE(1,3)  PE(1,4)  PE(1,5)  PE(1,6)  PE(1,7) ]
    position 2  [ PE(2,0)  PE(2,1)  PE(2,2)  PE(2,3)  PE(2,4)  PE(2,5)  PE(2,6)  PE(2,7) ]
    position 3  [ PE(3,0)  PE(3,1)  PE(3,2)  PE(3,3)  PE(3,4)  PE(3,5)  PE(3,6)  PE(3,7) ]
    ```

    So $PE_{(pos, k)}$ just means: "the entry at row $pos$, column $k$ of this table."

    The notation $PE_{(pos,\, 2i)}$ in the formulas is the paper's way of saying "the **even** columns" (columns 0, 2, 4, 6...). $PE_{(pos,\, 2i+1)}$ means "the **odd** columns" (columns 1, 3, 5, 7...). The paper writes it that way to express the rule cleanly: even cols get sine, odd cols get cosine.

    ## How it's used

    This table is built **once**, at model construction time. From then on, it's just sitting there, ready to be read.

    When a sequence comes in for processing, you do this:

    ### Step 1: Token embedding lookup

    The input might be token IDs like `[101, 7592, 3038]` (3 tokens). The token embedding layer (`nn.Embedding`) turns this into a `(3, d_model)` tensor — three rows of token vectors:

    ```
    token embeddings:
    position 0: [some 512-dim vector representing "the"]
    position 1: [some 512-dim vector representing "cat"]
    position 2: [some 512-dim vector representing "sat"]
    ```

    ### Step 2: Slice the PE table

    Grab the first 3 rows of the PE table (since the input has 3 tokens):

    ```
    positional encodings:
    position 0: [PE row 0, exactly as computed above]
    position 1: [PE row 1, exactly as computed above]
    position 2: [PE row 2, exactly as computed above]
    ```

    ### Step 3: Add them, elementwise

    ```
    final input:
    position 0: [token "the" vector + PE row 0]
    position 1: [token "cat" vector + PE row 1]
    position 2: [token "sat" vector + PE row 2]
    ```

    The result is still shape `(3, d_model)`. Each row now encodes both **what the token is** (from the token embedding) and **where it sits in the sequence** (from the PE).

    That's literally what these two lines in the code are doing:

    ```python
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    ```

    `x` is the token embedding tensor, `self.pe[:, :x.size(1)]` slices the appropriate number of rows from the PE table, and `+` adds them. The result is what flows into the first encoder/decoder layer.
    """)
    return


@app.cell
def _(math, torch):
    # confirming the math above
    d = 20
    final_pe = torch.zeros(d, dtype=float)

    # fraction = 2i / (10,000 ^ (2i / d)
    # denominator =   (10,000 ^ (2i / d)

    # since     a^x = e ^ (log (a ^ x))
    # therefore a^x = e ^ (x * log a)
    # in this case 
    #    a = 10,000
    #    x = (2i / d)
    #    denominator = e ^ ((2i/d) * log 10,000)
    denominator = torch.exp( 
        torch.arange(0, d, 2, dtype=float) / d  \
        * math.log(10_000) 
    )
    denominator_no_math = 10_000 ** (torch.arange(0, d, 2, dtype=float) / d)

    print(denominator)
    print(denominator_no_math)
    return


@app.cell
def _(math, nn, torch):
    class PositionalEmbedding(nn.Module):
        def __init__(
            self, 
            embed_dims,  # dimensions for each embedding
            max_seq_len  # max number of tokens
        ):
            super().__init__()
        
            # the job of nn.Embedding is to translate the incoming token to a embedding of a dictionary of fixed size and items
            # we need seperate embeddings for both since this program is written for translating from one language to other. each language will have it's own dict
            #    - encoder will have dict of source language
            #    - decoder will have dict of target language
            self.enc_embed = nn.Embedding(max_seq_len, embed_dims)
            self.dec_embed = nn.Embedding(max_seq_len, embed_dims)

            pe = torch.zeros(embed_dims, dtype=float)

            # fraction is just the part inside sin/cos but without the pos
            fraction = torch.exp(
                -1 * \
                torch.arange(0, embed_dims, 2, dtype=float) *\
                math.log(10_000)
            )

            # -------
            # since we now have the value of fraction value only (without pos), 
            # we have to now generate the combination of every pos (1 to max seq len) and 
            # and i (from 1 to 512)
            # output target is (max_seq_len, dims)

            # this generates (max_seq_len, 1)
            pos = torch.arange(max_seq_len, dtype=float).unsqueeze(1)

            # this generates (1, ebed_dims)
            fraction = fraction.unsqueeze(0)

            # fill all the even values for each row with sin
            pe[:, 0::2] = torch.sin(pos * fraction)
            # fill all the odd values for each row with cos
            pe[:, 1::2] = torch.cos(pos * fraction)

            # converting pe to dims of (batch=1, max_seq_len, embed_dims)
            self.register_buffer("pe", pe.unsqueeze(0))


        def forward(
            self, 
            x: torch.Tensor, # input embeddings
        ):
            # extract number of tokens from x
            # x is of the shape (batch, num_tokens, embeddings)
            seq_len = x.size(1)

            # here self.pe gets broadcased 
            #   from (1, max_seq_len, embed_dims)
            #     to (batch, max_seq_len, embed_dims)

            return x + self.pe[:, :seq_len]


    return (PositionalEmbedding,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## `nn.Linear` — quick reference

    `nn.Linear` takes two args: `in_features` and `out_features`.

    It only cares about the **last dimension** of the input:

    - **Input shape:**  `(..., in_features)`
    - **Output shape:** `(..., out_features)`

    The `...` can be any number of leading dimensions (batch, seq_len, etc.) — they all pass through untouched.

    ### What the layer owns internally

    - **Weight `A`** of shape `(out_features, in_features)`
    - **Bias `b`** of shape `(out_features,)`

    These are learned parameters. You don't pass them in — PyTorch creates them at construction time, the optimizer updates them during training.

    ### The math

    ```
    y = x · Aᵀ + b
    ```

    ### Example

    ```python
    layer = nn.Linear(in_features=512, out_features=1024)

    x = torch.randn(batch, seq_len, 512)   # last dim must be 512
    y = layer(x)                            # shape: (batch, seq_len, 1024)
    ```

    ### In the transformer

    ```python
    self.queries = nn.Linear(embed_size, embed_size)
    ```

    Same size in and out — the weights learn to project token embeddings into "query-like" features. Three separate linear layers (`queries`, `keys`, `values`) give three different views of the same input.
    """)
    return


@app.cell
def _(mo):
    mo.image(src="images/multi_head_attention_w_o_role.svg")
    return


@app.cell
def _(math, nn, torch):
    class MultiHeadAttention(nn.Module):
        def __init__(
            self, 
            embeddings_size: int,   # size of input embeddings
            num_attn_heads: int          # number of attention heads
        ):
            super().__init__()

            self.embeddings_size = embeddings_size
            self.num_attn_heads = num_attn_heads

            assert self.embeddings_size % self.num_attn_heads == 0
            self.dim_head = self.embeddings_size // self.num_attn_heads

            # defining params going to be used by pytorch
            #   (refer to the above documentation about nn.linear to understand inputs and outputs)        
            # query vector -> "what am i looking for"
            self.query = nn.Linear(
                in_features=embeddings_size,
                out_features=embeddings_size
            )

            # key vector -> "what do i have to offer"
            self.key = nn.Linear(
                in_features=embeddings_size,
                out_features=embeddings_size
            )

            # value vector -> "what do i contribue to"
            self.value = nn.Linear(
                in_features=embeddings_size,
                out_features=embeddings_size
            )

            # output vector -> for mixing all the head attentions
            self.output = nn.Linear(
                in_features=embeddings_size,
                out_features=embeddings_size
            )

        def forward(
            self, 
            Q, # (batch, num_tokens, embed_size)
            K, # (batch, num_tokens, embed_size)
            V, # (batch, num_tokens, embed_size)
            mask # for encoder -> padding, for decoder -> preventing future lookups
        ):
            Q = self.query(Q) # (batch_size, seq_len, embeddings_size)
            K = self.key(K)   # (batch_size, seq_len, embeddings_size)
            V = self.value(V) # (batch_size, seq_len, embeddings_size)

            # break down each for attention heads and resize to
            # (batch, seq_len, attention_heads, embed_size_per_head)
            Q = self.split_heads(Q)
            K = self.split_heads(K)
            V = self.split_heads(V)

            # convert from (batch, seq_len, attention_heads, embed_size_per_head)
            #           to (batch, attention_heads, seq_len, embed_size_per_head)
            Q = Q.transpose(1, 2)
            K = K.transpose(1, 2)
            V = V.transpose(1, 2)

            # --- now the meaty part - computing attention score --- #
            # since we want attention computed for every token with all other tokens, the output would be (batch, attn head, seq_len, seq_len)
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) # b, attn_head, seq_len, seq_len
            attn_scores = attn_scores / math.sqrt(self.dim_head)

            if mask is not None:
            # Apply mask if provided (useful for preventing attention to certain parts like padding)
                attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

            attn_probs = torch.softmax(attn_scores, dim=-1)

            # attn_probs (batch, attn_heads, seq_len, seq_len)
            #          V (batch, attn_heads, seq_len, dim_head)
            output = torch.matmul(attn_probs, V) # (batch, attn_heads, seq_len, dim_head)

            # --- combine heads --- #
            output = output.transpose(1, 2) # (batch, seq_len, attn_heads, dim_head)
            batch_size, seq_len, num_attn_heads, dim_per_head = output.size()

            # for my testing, remove this later
            assert num_attn_heads == self.num_attn_heads
            assert dim_per_head == self.dim_head
            assert dim_per_head * num_attn_heads == self.embeddings_size

            output = output.contiguous().view(batch_size, seq_len, self.embeddings_size)
            output = self.output(output)

            return output

        def split_heads(self, x: torch.Tensor):
            batch_size, seq_len, embed_size = x.size()
            return x.view(batch_size, seq_len, self.num_attn_heads, self.dim_head)


    return (MultiHeadAttention,)


@app.cell(hide_code=True)
def _(mo):
    mo.image(src="images/position_wise_feedforward_flow.svg")
    return


@app.cell
def _(nn):
    class FeedForwardNN(nn.Module):
        def __init__(self, inner_layer_dims, outer_layer_dims):
            super().__init__()

            # 512 → 2048   (expand)
            self.inner_layer = nn.Linear(outer_layer_dims, inner_layer_dims)
            # 2048 -> 512   (contract)
            self.outer_layer = nn.Linear(inner_layer_dims, outer_layer_dims)
            # for non-linearity, duh..
            self.relu = nn.ReLU()

        def forward(self, x): # x -> shape???
            return self.outer_layer(self.relu(self.inner_layer(x)))

    return (FeedForwardNN,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Encoder and Decoder
    """)
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md("""
    ### Q,K,V for encoders vs decoders 

    **Number of Q,K,V tensors in encoder vs decoders**

    Decoders involve running attention twice (one self attention and one cross attention)
        """),
            mo.image(src="images/qkv_count_full_forward_pass.svg"),
            mo.image(src="images/cross_attention_bridge_encoder_decoder.svg")
    
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
        ## Encoder Layer (single layer)
        """),
        mo.image(src="images/encoder_layer_dropout_placement.svg")
    ])
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md("""
    ### Padding in case of Encoder
    ```
    Sentence 1: "the cat sat"           → [the, cat, sat, <PAD>, <PAD>]
    Sentence 2: "I saw a big red dog"   → [I, saw, a, big, red]
    ```"""),
        mo.image(src="images/encoder_padding_mask_explainer.svg")
    ])
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md("#### Dropouts in Encoder"),
        mo.image(src="images/encoder_dropout_placement_and_importance.svg")
    ])
    return


@app.cell
def _(FeedForwardNN, MultiHeadAttention, nn):
    class EncoderLayer(nn.Module):
        def __init__(
            self, 
            embed_dims, 
            num_heads, 
            dropout_p, 
            ff_nn_dims
        ):
            super().__init__()
            self.attention = MultiHeadAttention(embeddings_size=embed_dims, num_attn_heads=num_heads)
            self.ff_nn = FeedForwardNN(outer_layer_dims=embed_dims, inner_layer_dims=ff_nn_dims)

            self.drop1 = nn.Dropout(dropout_p)
            self.drop2 = nn.Dropout(dropout_p)

            self.layer_norm1 = nn.LayerNorm(embed_dims)
            self.layer_norm2 = nn.LayerNorm(embed_dims)

        def encode(
            self,
            x,    # (batch, num_tokens, embed_dims) <-- positional embedding
            mask, # (batch, num_tokens, embed_dims) <-- padding mask
        ):
            attn_out = self.attention(x, x, x, mask)
            add1_out = x + self.drop1(attn_out)
            norm1_out = self.layer_norm1(add1_out)

            ff_nn_out = self.ff_nn(norm1_out)
            add2_out = self.drop2(ff_nn_out) + norm1_out
            norm2_out = self.layer_norm2(add2_out)

            return norm2_out

    return (EncoderLayer,)


@app.cell
def _(mo):
    mo.vstack([
        mo.md("## Decoder Layer (single layer)"),
        mo.image(src="images/decoder_block_anatomy_v2.svg")
    ])
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md("### Masks used in Decoder Block"),
        mo.image(src="images/transformer_three_masks_explained.svg")
    ])
    return


@app.cell
def _(mo):
    mo.image(src="images/enc_out_inside_decoder_layer_zoom.svg")
    return


@app.cell
def _(FeedForwardNN, MultiHeadAttention, nn):
    class DecoderLayer(nn.Module):
        def __init__(self, embed_dims, ff_inner_dims, num_heads, dropout_p):
            super().__init__()

            self.attention = MultiHeadAttention(embeddings_size=embed_dims, num_attn_heads=num_heads)
            self.cross_attention = MultiHeadAttention(embeddings_size=embed_dims, num_attn_heads=num_heads)


            self.ff_nn = FeedForwardNN(inner_layer_dims=ff_inner_dims, outer_layer_dims=embed_dims)

            self.drop1 = nn.Dropout(dropout_p)
            self.drop2 = nn.Dropout(dropout_p)
            self.drop3 = nn.Dropout(dropout_p)

            self.layer_norm1 = nn.LayerNorm(embed_dims)
            self.layer_norm2 = nn.LayerNorm(embed_dims)
            self.layer_norm3 = nn.LayerNorm(embed_dims)



        def forward(
            self, 
        
            # emebdding of the tokens decoder has generated/seen before
            #   for training: causal mask enforces constraint to see the
            #                 positions before this
            #  for inference: running embedding of the generated tokens 
            #                 till now
            x,  

            # source padding containing fillers (pads) to fill the entire tensor
            # for eg "the cat sat pad pad" (last 2 pads to fill the shape)
            src_mask, 

            # target mask not allowing the decoder to see any tokens after it.
            # only relevant for training. during inference we don't know the 
            # future tokens anyways
            tgt_mask, 


            # encoder's contextualized understnading of the entire sentence
            enc_out
        ):
        
            # running self attention for decoder layer first 
            attn1_out = self.attention(x, x, x, tgt_mask)
            # residually pass x and apply dropout to the attention scores generaetd
            # apply layer norm on the result
            norm1_out = self.layer_norm1(self.drop1(attn1_out) + x)

            # ask questions to enc_out (which has understanding of the source sentence)
            #   we use our own Q (question) but K and V of encoder 
            attn2_out = self.cross_attention(norm1_out, enc_out, enc_out, src_mask)
            norm2_out = self.layer_norm2(norm1_out + self.drop2(attn2_out))

            nn_out = self.ff_nn(norm2_out)
            layer3_out = self.layer_norm3(self.drop3(nn_out) + norm2_out)

            return layer3_out


    return (DecoderLayer,)


@app.cell
def _(mo):
    mo.image(src="images/decoder_forward_inputs_explained.svg")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Encoder Block
    """)
    return


@app.cell
def _(EncoderLayer, nn):
    class EncodeBlock(nn.Module):
        def __init__(
            self, 
            num_enc, # number of EncoderLayer in EncoderBlock
            embed_dims,  # num of dims for encoding a token
            num_heads,  # num of heads for attention computation
            dropout_p,  # common prob for droput for all the skip connections,
            ff_nn_dims  # ff NN works this way embed_dims -> ff_nn_dims -> embed_dims
        ):
            super().__init__()

            self.encoder_layers = nn.ModuleList(modules=[
                EncoderLayer(embed_dims, num_heads, dropout_p, ff_nn_dims)
                for _ in range(num_enc)
            ])

        def forward(
            self,
            x,  # input sequence: batch, seq_len, embed_dims
            mask,   # mask for padding: batch, seq_len, embed_dims
        ):
            for layer in self.encoder_layers:
                x = layer.encode(x, mask)
            return x

    return (EncodeBlock,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Decoder Block
    """)
    return


@app.cell
def _(DecoderLayer, nn):
    class DecoderBlock(nn.Module):
        def __init__(
            self,
            num_decoder_layers,  # num of decode layers in decode block
            embed_dims,          # num of dims for each token
            num_heads,           # num of heads for attention 
            ff_nn_dims,          # internal ff_nn dims
            dropout_p,           # dropout prob (common to all dropouts)
        ):
            super().__init__()

            self.decoder_layers = nn.ModuleList(modules=[
                DecoderLayer(embed_dims, ff_nn_dims, num_heads, dropout_p)
                for _ in range(num_decoder_layers)
            ])

        def forward(self, x, src_mask, tgt_mask, enc_out):
            for layer in self.decoder_layers:
                x = layer(x, src_mask, tgt_mask, enc_out)
            return x
        

    return (DecoderBlock,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Transformer
    """)
    return


@app.cell
def _(mo):
    mo.image(src="images/transformer_forward_data_flow_v3.svg")
    return


@app.cell
def _(DecoderBlock, EncodeBlock, PositionalEmbedding, nn):
    class Transformer(nn.Module):
        def __init__(
            self,
            src_vocab_size,
            tgt_vocab_size,
            embed_dims,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            ff_nn_dims,
            max_seq_len,
            dropout_p=0.1,
        ):
            super().__init__()

            # Token embeddings (moved out of PositionalEmbedding)
            self.src_token_embed = nn.Embedding(src_vocab_size, embed_dims)
            self.tgt_token_embed = nn.Embedding(tgt_vocab_size, embed_dims)

            # Positional encoding (no longer owns token embeddings)
            self.positional_encoding = PositionalEmbedding(embed_dims, max_seq_len)

            # Encoder and decoder stacks
            self.encoder = EncodeBlock(
                num_enc=num_encoder_layers,
                embed_dims=embed_dims,
                num_heads=num_heads,
                dropout_p=dropout_p,
                ff_nn_dims=ff_nn_dims,
            )
            self.decoder = DecoderBlock(
                num_decoder_layers=num_decoder_layers,
                embed_dims=embed_dims,
                num_heads=num_heads,
                ff_nn_dims=ff_nn_dims,
                dropout_p=dropout_p,
            )

            # Final projection: embed_dims → vocab_size
            self.final_linear = nn.Linear(embed_dims, tgt_vocab_size)

        def forward(
            self,
            src,           # (batch, src_seq_len) — token IDs
            tgt,           # (batch, tgt_seq_len) — token IDs
            src_mask,      # (batch, 1, 1, src_seq_len) — padding mask
            tgt_mask,      # (batch, 1, tgt_seq_len, tgt_seq_len) — causal + padding
        ):
            # Embed and add positional info
            src_embed = self.positional_encoding(self.src_token_embed(src))
            tgt_embed = self.positional_encoding(self.tgt_token_embed(tgt))

            # Encode
            enc_out = self.encoder(src_embed, src_mask)

            # Decode (using encoder output for cross-attention)
            dec_out = self.decoder(tgt_embed, src_mask, tgt_mask, enc_out)

            # Project to vocabulary logits
            logits = self.final_linear(dec_out)

            return logits

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
