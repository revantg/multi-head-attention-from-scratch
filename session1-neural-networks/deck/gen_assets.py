"""Reproduce all 4 Session-1 proofs and export LIGHT, Nike-themed proof images.

Same training/seeds as ../generate_proofs.py (so the numbers are identical), but
rendered on a white editorial canvas with the Nike palette so the charts read like
catalog photography staged on soft-cloud.

Run:  uv run python slideshow/gen_assets.py
Outputs PNGs + results.json into slideshow/assets/.
"""
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch
import torch.nn as nn

# ----------------------------------------------------------------------------- setup
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

OUT = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(OUT, exist_ok=True)

# Nike light editorial palette
CANVAS = "#ffffff"
CLOUD = "#f5f5f5"
INK = "#111111"
MUTE = "#707072"
HAIR = "#e5e5e5"
INFO = "#1151ff"   # class 0 (blue)
SALE = "#d30005"   # class 1 / failure (red)
SUCCESS = "#007d48"  # win (green)
PINK = "#ed1aa0"
CATS = {"animal": INFO, "fruit": SUCCESS, "verb": PINK}

# diverging field: blue (class 0) -> white -> red (class 1)
FIELD = LinearSegmentedColormap.from_list("nikefield", [INFO, "#ffffff", SALE])

plt.rcParams.update({
    "figure.facecolor": CANVAS, "axes.facecolor": CLOUD, "savefig.facecolor": CANVAS,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTE, "ytick.color": MUTE,
    "axes.edgecolor": HAIR, "grid.color": HAIR, "font.size": 13,
    "axes.titlecolor": INK, "figure.dpi": 150,
    "font.family": "DejaVu Sans", "font.weight": "medium",
    "axes.linewidth": 1.0,
})

results = {}


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight", facecolor=CANVAS, pad_inches=0.25)
    plt.close(fig)
    print("wrote", path)


# ----------------------------------------------------------------------------- ring data
def make_rings(n, inner=(0.0, 4.0), outer=(5.0, 9.0), noise=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    n0 = n // 2
    n1 = n - n0

    def ring(k, lo, hi):
        r = torch.rand(k, generator=g) * (hi - lo) + lo
        th = torch.rand(k, generator=g) * 2 * math.pi
        return torch.stack([r * torch.cos(th), r * torch.sin(th)], 1)

    x = torch.cat([ring(n0, *inner), ring(n1, *outer)])
    y = torch.cat([torch.zeros(n0), torch.ones(n1)])
    if noise > 0:
        flip = torch.rand(n, generator=g) < noise
        y[flip] = 1 - y[flip]
    perm = torch.randperm(n, generator=g)
    return x[perm], y[perm]


def boundary_grid(model, lim=10.0, step=0.12):
    xs = torch.arange(-lim, lim, step)
    ys = torch.arange(-lim, lim, step)
    grid = torch.cartesian_prod(xs, ys)
    with torch.no_grad():
        p = torch.sigmoid(model(grid)).view(len(xs), len(ys)).numpy().T
    return xs.numpy(), ys.numpy(), p


def plot_boundary(ax, model, X, Y, title, acc, win=None):
    xs, ys, p = boundary_grid(model)
    ax.contourf(xs, ys, p, levels=24, cmap=FIELD, alpha=0.55)
    ax.contour(xs, ys, p, levels=[0.5], colors=[INK], linewidths=2.0)
    Xn = X.numpy()
    ax.scatter(Xn[Y == 0, 0], Xn[Y == 0, 1], s=16, c=INFO, edgecolors=CANVAS, linewidths=0.4)
    ax.scatter(Xn[Y == 1, 0], Xn[Y == 1, 1], s=16, c=SALE, edgecolors=CANVAS, linewidths=0.4)
    col = SUCCESS if (win is True) else (SALE if win is False else INK)
    ax.set_title(f"{title}", fontsize=15, color=INK, pad=12, fontweight="bold")
    ax.text(0.04, 0.05, f"{acc*100:.0f}% ACCURACY", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=13, color=CANVAS, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc=col, ec="none"))
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def train_bce(model, X, Y, epochs=4000, lr=0.02):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = fn(model(X), Y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = ((torch.sigmoid(model(X)) > 0.5).float() == Y).float().mean().item()
    return acc


# ============================================================================= S1-1
def s1_1():
    X, Y = make_rings(300, noise=0.02, seed=1)

    class Linear(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = nn.Linear(2, 1)

        def forward(self, x):
            return self.l(x).squeeze(1)

    class ReLU(nn.Module):
        def __init__(self, h=32):
            super().__init__()
            self.a = nn.Linear(2, h)
            self.r = nn.ReLU()
            self.b = nn.Linear(h, 1)

        def forward(self, x):
            return self.b(self.r(self.a(x))).squeeze(1)

    lin = Linear()
    acc_lin = train_bce(lin, X, Y)
    rel = ReLU()
    acc_rel = train_bce(rel, X, Y)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    plot_boundary(axes[0], lin, X, Y, "SINGLE LINEAR LAYER", acc_lin, win=False)
    plot_boundary(axes[1], rel, X, Y, "ONE ReLU HIDDEN LAYER", acc_rel, win=True)
    save(fig, "s1_1_boundaries.png")
    results["s1_1"] = {"linear_acc": round(acc_lin, 3), "relu_acc": round(acc_rel, 3)}


# ============================================================================= S1-2
def s1_2():
    X, Y = make_rings(300, noise=0.02, seed=1)

    class One(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = nn.Linear(2, 1)

        def forward(self, x):
            return self.l(x).squeeze(1)

    class DeepLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(2, 16, bias=False), nn.Linear(16, 16, bias=False),
                                     nn.Linear(16, 16, bias=False), nn.Linear(16, 16, bias=False),
                                     nn.Linear(16, 1, bias=False))

        def forward(self, x):
            return self.net(x).squeeze(1)

    class DeepReLU(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(),
                                     nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(),
                                     nn.Linear(16, 1))

        def forward(self, x):
            return self.net(x).squeeze(1)

    one = One()
    acc_one = train_bce(one, X, Y)
    dl = DeepLinear()
    acc_dl = train_bce(dl, X, Y)
    dr = DeepReLU()
    acc_dr = train_bce(dr, X, Y)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))
    plot_boundary(axes[0], one, X, Y, "1 LINEAR LAYER", acc_one, win=False)
    plot_boundary(axes[1], dl, X, Y, "5 LINEAR LAYERS", acc_dl, win=False)
    plot_boundary(axes[2], dr, X, Y, "5 LAYERS + ReLU", acc_dr, win=True)
    save(fig, "s1_2_boundaries.png")

    # numerical collapse proof: product of the 5 weight matrices == one (1,2) map
    with torch.no_grad():
        prod = None
        for layer in dl.net:
            W = layer.weight  # [out, in]
            prod = W if prod is None else W @ prod
        collapsed = prod.numpy()  # shape [1, 2]
    results["s1_2"] = {
        "one_acc": round(acc_one, 3), "deep_linear_acc": round(acc_dl, 3),
        "deep_relu_acc": round(acc_dr, 3),
        "collapsed_matrix": collapsed.round(4).tolist(),
    }


# ============================================================================= S1-3
def s1_3():
    grammar = ["apple", "cat", "chase", "cow", "dog", "eat", "mango", "see"]
    cat_of = {"cat": "animal", "dog": "animal", "cow": "animal",
              "apple": "fruit", "mango": "fruit",
              "eat": "verb", "chase": "verb", "see": "verb"}
    animals = ["cat", "dog", "cow"]
    fruits = ["apple", "mango"]
    verbs = ["eat", "chase", "see"]
    templates = [["A", "V", "F"], ["A", "V", "A"]]
    slot = {"A": animals, "F": fruits, "V": verbs}

    rng = np.random.default_rng(0)
    sents = []
    for _ in range(800):
        t = templates[rng.integers(len(templates))]
        sents.append([slot[s][rng.integers(len(slot[s]))] for s in t])

    stoi = {w: i for i, w in enumerate(grammar)}
    xs, ys = [], []
    for s in sents:
        ids = [stoi[w] for w in s]
        for a, b in zip(ids, ids[1:]):
            xs.append(a)
            ys.append(b)
    tx = torch.tensor(xs)
    ty = torch.tensor(ys)

    class NextWord(nn.Module):
        def __init__(self, dim=2, v=8):
            super().__init__()
            self.emb = nn.Embedding(v, dim)
            self.out = nn.Linear(dim, v)

        def forward(self, x):
            return self.out(self.emb(x))

    torch.manual_seed(0)
    model = NextWord()
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    fn = nn.CrossEntropyLoss()
    for _ in range(3000):
        opt.zero_grad()
        loss = fn(model(tx), ty)
        loss.backward()
        opt.step()

    W = model.emb.weight.detach().numpy()

    from scipy.spatial.distance import cdist
    D = cdist(W, W)
    np.fill_diagonal(D, np.inf)
    nn_rows = []
    same = 0
    for i, w in enumerate(grammar):
        j = int(D[i].argmin())
        ok = cat_of[w] == cat_of[grammar[j]]
        same += ok
        nn_rows.append({"word": w, "nn": grammar[j], "same": bool(ok)})

    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    for cat, col in CATS.items():
        idx = [i for i, w in enumerate(grammar) if cat_of[w] == cat]
        ax.scatter(W[idx, 0], W[idx, 1], s=260, c=col, label=cat, edgecolors=CANVAS,
                   linewidths=2.0, zorder=3)
    for i, w in enumerate(grammar):
        ax.annotate(w.upper(), (W[i, 0], W[i, 1]), color=INK, fontsize=12.5,
                    fontweight="bold", xytext=(9, 7), textcoords="offset points", zorder=4)
    leg = ax.legend(title="CATEGORY", facecolor=CANVAS, edgecolor=HAIR, labelcolor=INK,
                    fontsize=12, title_fontsize=11, loc="best", framealpha=1.0)
    leg.get_title().set_color(MUTE)
    ax.grid(True, alpha=0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
    save(fig, "s1_3_embeddings.png")

    results["s1_3"] = {"nn_rows": nn_rows, "same_category": f"{same}/{len(grammar)}"}


# ============================================================================= S1-4
def s1_4():
    # Measured values from the notebook's OverParamNN run (Session1_Assignment.ipynb,
    # cell [30]). We plot the notebook's actual numbers so this restyled chart matches
    # the screenshots exactly.
    rows = [
        {"n": 20,   "train_loss": 0.0000, "test_loss": 1.6214, "gap": 1.6214},
        {"n": 200,  "train_loss": 0.0071, "test_loss": 0.9677, "gap": 0.9606},
        {"n": 2000, "train_loss": 0.3994, "test_loss": 0.1270, "gap": -0.2724},
    ]

    ns = [r["n"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    ax = axes[0]
    ax.axhline(0, color=MUTE, linewidth=1.0, linestyle=(0, (4, 4)))
    ax.plot(ns, [r["gap"] for r in rows], "-o", color=SALE, linewidth=3.0, markersize=11,
            markeredgecolor=CANVAS, markeredgewidth=1.5)
    ax.fill_between(ns, [r["gap"] for r in rows], color=SALE, alpha=0.08)
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels(ns)
    ax.set_xlabel("TRAINING SET SIZE", fontsize=11, color=MUTE)
    ax.set_ylabel("GENERALIZATION GAP  (test − train)", fontsize=11, color=MUTE)
    ax.set_title("THE GAP COLLAPSES AS DATA GROWS", fontsize=14, fontweight="bold", pad=10)
    ax.grid(True, alpha=0.6)

    ax = axes[1]
    ax.plot(ns, [r["train_loss"] for r in rows], "-o", color=SUCCESS, linewidth=3.0,
            markersize=11, markeredgecolor=CANVAS, markeredgewidth=1.5, label="train loss")
    ax.plot(ns, [r["test_loss"] for r in rows], "-o", color=SALE, linewidth=3.0,
            markersize=11, markeredgecolor=CANVAS, markeredgewidth=1.5, label="test loss")
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels(ns)
    ax.set_xlabel("TRAINING SET SIZE", fontsize=11, color=MUTE)
    ax.set_ylabel("BCE LOSS", fontsize=11, color=MUTE)
    ax.set_title("TEST FALLS TO MEET TRAIN AS DATA GROWS", fontsize=14, fontweight="bold", pad=10)
    ax.legend(facecolor=CANVAS, edgecolor=HAIR, labelcolor=INK, fontsize=12, framealpha=1.0)
    ax.grid(True, alpha=0.6)
    save(fig, "s1_4_gap.png")

    results["s1_4"] = {"rows": [
        {"n": r["n"], "train_loss": round(r["train_loss"], 4),
         "test_loss": round(r["test_loss"], 4), "gap": round(r["gap"], 4)} for r in rows]}


if __name__ == "__main__":
    s1_1()
    s1_2()
    s1_3()
    s1_4()
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nresults.json written")
