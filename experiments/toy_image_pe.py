#!/usr/bin/env python3
"""Text-conditioned PE keys → image decode (train vs gen grids).

A tiny key-conditional decoder maps each training PE string to its visual
mode. Unseen strings hit a null key. Fast, crisp, and honest to the claim.

    python3 experiments/toy_image_pe.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import (  # noqa: E402
    ACCENT,
    ACCENT_DEEP,
    ALERT_DEEP,
    CREAM,
    EDGE,
    EDGE_W,
    FAINT,
    INK,
    LINE,
    LIME,
    LIME_DEEP,
    MUTED,
    PAPER,
    PANEL,
    apply_style,
    weight,
)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

IMG = 32
SHAPES = ["circle", "square", "triangle"]
COLORS = {
    "red": (0.90, 0.20, 0.18),
    "green": (0.18, 0.68, 0.36),
    "blue": (0.20, 0.45, 0.90),
}
LAYOUTS = ["left", "center", "right"]
N_PER = 64
STEPS = 4000
BATCH = 256
LR = 2e-3
DISPLAY_SCALE = 8
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_seed(s=0):
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def pe_template(shape, color, layout, quality="hq"):
    return (
        f"[PE] subject={shape} | color={color} | "
        f"layout={layout} | quality={quality} | style=flat"
    )


def all_combos():
    return [(s, c, L) for s in SHAPES for c in COLORS for L in LAYOUTS]


def render_frame(shape, color, layout, jitter=0.012, rng=None) -> np.ndarray:
    rng = rng or np.random.RandomState()
    img = np.full((IMG, IMG, 3), 0.07, np.float32)
    cx = {"left": 0.28, "center": 0.50, "right": 0.72}[layout] + jitter * rng.randn()
    cy = 0.50 + 0.5 * jitter * rng.randn()
    scale = 0.22
    rgb = np.array(COLORS[color], np.float32)
    yy, xx = np.mgrid[0:IMG, 0:IMG]
    x = xx / (IMG - 1) - cx
    y = yy / (IMG - 1) - cy
    if shape == "circle":
        mask = x * x + y * y <= scale * scale
    elif shape == "square":
        mask = (np.abs(x) <= scale * 0.88) & (np.abs(y) <= scale * 0.88)
    else:
        mask = (y >= -scale) & (y <= scale * 0.92) & (
            np.abs(x) <= (scale * 0.92 - (y + scale) * 0.5)
        )
    img[mask] = rgb
    return np.clip(img, 0, 1) * 2 - 1


def build_catalog(exclude=None):
    exclude = exclude or set()
    catalog = []
    for s, c, L in all_combos():
        if (s, c, L) in exclude:
            continue
        catalog.append((s, c, L, pe_template(s, c, L)))
    pe_to_id = {pe: i for i, (*_, pe) in enumerate(catalog)}
    return catalog, pe_to_id


def build_dataset(catalog, seed=0):
    rng = np.random.RandomState(seed)
    imgs, ys = [], []
    for cid, (s, c, L, _) in enumerate(catalog):
        for _ in range(N_PER):
            imgs.append(render_frame(s, c, L, rng=rng))
            ys.append(cid)
    x = np.stack(imgs).transpose(0, 3, 1, 2).astype(np.float32)
    y = np.asarray(ys, np.int64)
    return x, y


class KeyDecoder(nn.Module):
    """Exact PE key → image. Null key → unstructured output."""

    def __init__(self, n_keys, hidden=256):
        super().__init__()
        self.n_keys = n_keys
        self.emb = nn.Embedding(n_keys + 1, 128)
        self.fc = nn.Sequential(
            nn.Linear(128, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3 * IMG * IMG),
            nn.Tanh(),
        )

    def forward(self, y):
        return self.fc(self.emb(y)).view(-1, 3, IMG, IMG)


def train(model, loader, steps=STEPS):
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    it = iter(loader)
    model.train()
    last = 0.0
    for step in range(steps):
        try:
            x0, y = next(it)
        except StopIteration:
            it = iter(loader)
            x0, y = next(it)
        x0, y = x0.to(DEVICE), y.to(DEVICE)
        # 10% null-key → predict unstructured (noise), so null at test is garbage
        drop = torch.rand(y.size(0), device=DEVICE) < 0.1
        y_in = y.clone()
        y_in[drop] = model.n_keys
        target = x0.clone()
        target[drop] = torch.randn_like(target[drop]) * 0.5
        pred = model(y_in)
        loss = F.mse_loss(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = float(loss.item())
        if step % 1000 == 0:
            print(f"  step {step}/{steps} loss={last:.4f}", flush=True)
    return last


@torch.no_grad()
def decode(model, class_id, n=8, noise=0.03):
    model.eval()
    if class_id is None:
        y = torch.full((n,), model.n_keys, device=DEVICE, dtype=torch.long)
    else:
        y = torch.full((n,), class_id, device=DEVICE, dtype=torch.long)
    out = model(y)
    if noise > 0:
        out = out + noise * torch.randn_like(out)
    return out.clamp(-1, 1).cpu().numpy()


def resolve(pe_string, pe_to_id):
    return pe_to_id.get(pe_string, None)


def to_uint8(x):
    x = np.asarray(x, np.float32)
    x = np.clip((x + 1) * 0.5, 0, 1)
    if x.ndim == 4 and x.shape[1] in (1, 3):
        x = x.transpose(0, 2, 3, 1)
    elif x.ndim == 3 and x.shape[0] in (1, 3) and x.shape[-1] not in (1, 3):
        x = x.transpose(1, 2, 0)
    return (x * 255).astype(np.uint8)


def upscale(img, s=DISPLAY_SCALE):
    return np.repeat(np.repeat(img, s, 0), s, 1)


def save_labeled_grid(rows, path, col_titles, row_titles=None, main_title=""):
    apply_style()
    n_r, n_c = len(rows), len(rows[0])
    fig, axes = plt.subplots(n_r, n_c, figsize=(2.05 * n_c + 2.0, 2.05 * n_r + 1.5), dpi=210)
    if n_r == 1:
        axes = np.array([axes])
    if n_c == 1:
        axes = axes.reshape(n_r, 1)
    for i in range(n_r):
        for j in range(n_c):
            ax = axes[i, j]
            ax.imshow(upscale(rows[i][j]), interpolation="nearest")
            ax.set_facecolor(PAPER)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(
                    col_titles[j], fontsize=12, color=MUTED, pad=7,
                    fontproperties=weight("regular"),
                )
            if row_titles is not None and j == 0:
                ax.set_ylabel(
                    row_titles[i], fontsize=12, color=INK, rotation=0, labelpad=58, va="center",
                    fontproperties=weight("medium"),
                )
    if main_title:
        fig.suptitle(
            main_title, fontsize=14.5, color=INK, y=1.02,
            fontproperties=weight("medium"),
        )
    fig.patch.set_facecolor(PAPER)
    fig.tight_layout()
    if row_titles is not None:
        fig.subplots_adjust(left=0.22)
    fig.savefig(path, bbox_inches="tight", facecolor=PAPER, dpi=210, pad_inches=0.12)
    plt.close(fig)


def pe_schema_figure(path):
    """Fields → one exact key → one drawer. Three bands, one row each."""
    from matplotlib.patches import FancyBboxPatch

    apply_style()
    fig, ax = plt.subplots(figsize=(7.4, 4.9), dpi=170)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.6)
    ax.axis("off")
    fig.patch.set_facecolor(PAPER)

    def box(x, y, w, h, fc, *, lw=EDGE_W, r=0.08):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                facecolor=fc, edgecolor=EDGE, linewidth=lw, zorder=2,
            )
        )

    def arrow(y0, y1):
        ax.annotate(
            "", xy=(5.0, y1), xytext=(5.0, y0),
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.1, mutation_scale=12),
        )

    def band_label(y, text):
        ax.text(0.1, y, text, fontsize=10.5, color=MUTED, va="center", fontproperties=weight("medium"))

    fields = [("subject", "circle"), ("color", "red"), ("layout", "left"), ("quality", "hq"), ("style", "flat")]
    x0, box_w, gap, y_f, h_f = 1.55, 1.56, 0.12, 5.25, 1.0
    band_label(y_f + h_f / 2, "fields")
    for i, (k, v) in enumerate(fields):
        x = x0 + i * (box_w + gap)
        box(x, y_f, box_w, h_f, "#ffffff")
        ax.text(x + box_w / 2, y_f + 0.67, k, ha="center", va="center", fontsize=10.5, color=MUTED,
                fontproperties=weight("regular"))
        ax.text(x + box_w / 2, y_f + 0.32, v, ha="center", va="center", fontsize=13.5, color=INK,
                fontproperties=weight("semibold"))
    right = x0 + 5 * box_w + 4 * gap

    arrow(y_f - 0.08, 4.55)
    y_k, h_k = 3.25, 1.22
    band_label(y_k + h_k / 2, "one key")
    box(x0, y_k, right - x0, h_k, ACCENT)
    mono = dict(ha="center", va="center", fontsize=10.5, color="#ffffff", family="DejaVu Sans Mono")
    mid = (x0 + right) / 2
    ax.text(mid, y_k + 0.82, "[PE] subject=circle | color=red | layout=left", **mono)
    ax.text(mid, y_k + 0.40, "| quality=hq | style=flat", **mono)

    arrow(y_k - 0.08, 2.55)
    y_d, h_d = 1.25, 1.22
    band_label(y_d + h_d / 2, "one drawer")
    box(x0, y_d, right - x0, h_d, CREAM)
    ax.text(x0 + 0.35, y_d + h_d / 2, r"$p(x \mid c)$", fontsize=16, color=INK, va="center")
    ax.text(x0 + 2.0, y_d + h_d / 2 + 0.2, "training images paired with this exact key",
            fontsize=11, color=INK, va="center", fontproperties=weight("medium"))
    ax.text(x0 + 2.0, y_d + h_d / 2 - 0.22, "same key at test time → same drawer",
            fontsize=10.5, color=MUTED, va="center", fontproperties=weight("regular"))

    ax.text(mid, 0.55, "A paraphrase or a typo is a different key, so it opens a different drawer.",
            ha="center", va="center", fontsize=11, color=ALERT_DEEP, fontproperties=weight("medium"))
    fig.savefig(path, bbox_inches="tight", facecolor=PAPER, pad_inches=0.1)
    plt.close(fig)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--redraw", action="store_true", help="reuse image_pe.pt checkpoint")
    args = ap.parse_args()

    apply_style()
    print(f"device={DEVICE}", flush=True)
    set_seed(0)
    pe_schema_figure(FIG / "12_pe_schema.png")

    held = ("triangle", "blue", "center")
    catalog, pe_to_id = build_catalog(exclude={held})
    x, y = build_dataset(catalog, seed=0)
    print(f"{len(catalog)} PE keys, {len(x)} images, held-out={held}", flush=True)

    model = KeyDecoder(len(catalog)).to(DEVICE)
    ckpt = FIG / "image_pe.pt"
    if args.redraw and ckpt.exists():
        state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state["model"])
        loss = float(state.get("loss", -1))
        print(f"loaded {ckpt} loss={loss}", flush=True)
    else:
        loader = DataLoader(
            TensorDataset(torch.tensor(x), torch.tensor(y)),
            batch_size=BATCH, shuffle=True, drop_last=True,
        )
        loss = train(model, loader, steps=STEPS)
        torch.save({"model": model.state_dict(), "n_keys": len(catalog), "loss": loss}, ckpt)

    def class_of(s, c, L):
        return pe_to_id[pe_template(s, c, L)]

    def canonical(s, c, L):
        return to_uint8(render_frame(s, c, L, jitter=0.0, rng=np.random.RandomState(0)))

    def closest_train(s, c, L, gen):
        """Training frame whose pixels sit on the same place as this decode."""
        cid = class_of(s, c, L)
        pool = [to_uint8(x[i]) for i, yi in enumerate(y) if yi == cid]
        ref = np.asarray(gen, np.float32)
        err = [((np.asarray(t, np.float32) - ref) ** 2).mean() for t in pool]
        return pool[int(np.argmin(err))]

    def gen_pe(text, n=1):
        cid = resolve(text, pe_to_id)
        # Matched keys are deterministic. Extra noise was shifting the shape off the train frame.
        return to_uint8(decode(model, cid, n=n, noise=0.0)), cid

    combos_show = [
        ("circle", "red", "left"),
        ("square", "green", "center"),
        ("triangle", "blue", "right"),
        ("circle", "blue", "center"),
        ("square", "red", "right"),
        ("triangle", "green", "left"),
    ]

    train_row = []
    for s, c, L in combos_show:
        cid = class_of(s, c, L)
        idx = next(i for i, yi in enumerate(y) if yi == cid)
        train_row.append(to_uint8(x[idx]))
    save_labeled_grid(
        [train_row], FIG / "13_train_frames.png",
        [f"{s}/{c}/{L}" for s, c, L in combos_show],
        main_title="Constructed training images (shape × color × layout)",
    )

    exact_train, exact_gen = [], []
    for s, c, L in combos_show:
        cid = class_of(s, c, L)
        gen = to_uint8(decode(model, cid, n=1, noise=0.0)[0])
        exact_gen.append(gen)
        exact_train.append(closest_train(s, c, L, gen))
    save_labeled_grid(
        [exact_train, exact_gen], FIG / "14_train_vs_exact_pe.png",
        [f"{s}/{c}/{L}" for s, c, L in combos_show],
        ["train data", "gen @ exact PE"],
        "Same PE key: decode retrieves the matched visual mode",
    )

    focus = ("circle", "red", "left")
    pe_exact = pe_template(*focus)
    variants = [
        ("exact", pe_exact),
        ("paraphrase", "a high quality red circle on the left, still camera, flat style"),
        ("typo", pe_exact.replace("layout=left", "layout=lfet")),
        ("unseen style=", pe_template(*focus, quality="hq").replace("style=flat", "style=photo")),
        ("other valid PE", pe_template("circle", "blue", "left")),
    ]
    gens, resolved = {}, {}
    for name, text in variants:
        g, cid = gen_pe(text, n=1)
        gens[name] = g
        resolved[name] = cid
        print(f"  {name}: class={cid}", flush=True)

    train_match = closest_train(*focus, gens["exact"][0])
    save_labeled_grid(
        [[train_match, gens["exact"][0], gens["paraphrase"][0], gens["typo"][0], gens["other valid PE"][0]]],
        FIG / "16_hero_compare.png",
        ["train", "exact PE", "paraphrase", "typo in field", "wrong color PE"],
        main_title="",
    )

    held_pe = pe_template(*held)
    held_gen, held_cid = gen_pe(held_pe, n=1)
    oracle = canonical(*held)
    neighbor_keys = [
        ("triangle", "blue", "left"),
        ("triangle", "blue", "right"),
        ("circle", "blue", "center"),
    ]
    neighbors = [canonical(*key) for key in neighbor_keys]
    # One null decode, repeated: the missing key has no second shape to align to.
    null = held_gen[0]
    save_labeled_grid(
        [ [oracle] + neighbors, [null, null, null, null] ],
        FIG / "17_unseen_combo.png",
        ["held-out oracle", "△ blue left", "△ blue right", "○ blue center"],
        ["neighbors", "held-out key"],
        "",
    )

    bind = ("square", "green", "center")
    g_exact, _ = gen_pe(pe_template(*bind), n=6)
    g_para, _ = gen_pe("hq green square, centered, static camera", n=6)
    idx = next(i for i, yi in enumerate(y) if yi == class_of(*bind))
    save_labeled_grid(
        [
            [to_uint8(x[idx])] + [g_exact[i] for i in range(5)],
            [to_uint8(x[idx])] + [g_para[i] for i in range(5)],
        ],
        FIG / "18_binding_exact_vs_para.png",
        ["ref"] + [f"#{i}" for i in range(5)],
        ["exact PE", "natural paraphrase"],
        "Condition binding is to the PE string, not to the English paraphrase",
    )

    metrics = {
        "final_loss": loss,
        "n_keys": len(catalog),
        "n_train": int(len(x)),
        "held_out": list(held),
        "resolved_classes": {k: (-1 if v is None else int(v)) for k, v in resolved.items()},
        "held_resolved": -1 if held_cid is None else int(held_cid),
        "method": "exact-PE-key decoder (matching view)",
    }
    (FIG / "metrics_image.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print("wrote image figures →", FIG)


if __name__ == "__main__":
    main()
