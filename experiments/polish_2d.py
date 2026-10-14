#!/usr/bin/env python3
"""Polished 2D story figures (fresh draw, no nested PNG collage).

    CUDA_VISIBLE_DEVICES=0 python3 experiments/polish_2d.py
    CUDA_VISIBLE_DEVICES=0 python3 experiments/polish_2d.py --redraw   # reuse cache
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.patches import Circle
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import (  # noqa: E402
    ACCENT,
    ACCENT_DEEP,
    ALERT,
    ALERT_DEEP,
    BAR_PAIR,
    BAR_PAIR_OFF,
    EDGE,
    EDGE_W,
    STONE,
    FAINT,
    GREEN,
    GRID,
    INK,
    MUTED,
    PAPER,
    PANEL,
    SOFT,
    apply_style,
    weight,
)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
CACHE = FIG / "cache_2d_polish.npz"
FIG.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 0
N_MODES = 8
N_PER = 320
STEPS = 5000
BATCH = 512
HIDDEN = 192
T_DIFF = 80
LR = 2e-3
PE_DIM = 16
LAYOUTS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
PE_TEMPLATES = [
    f"PE:v1 | subject=red_orb | layout={L.lower()} | quality=hq" for L in LAYOUTS
]
PE_PARA = [
    "high quality red orb in the north",
    "hq red sphere, northeast layout",
    "red ball on the east side, high quality",
    "southeast red orb, nice quality",
    "southern red orb, hq",
    "southwest red sphere",
    "red orb to the west, high quality",
    "northwest red ball, hq",
]

LINE = "#d6d3cc"


def mode_colors(n=N_MODES):
    # Cambrian card tones, darkened enough to read as points on white.
    ring = ["#c0694b", "#d0955f", "#b89a45", "#7fa38a", "#4f8a7a", "#5d8fc4", "#7377b0", "#a0729c"]
    return ring[:n]


COLORS = mode_colors()


def set_seed(s=SEED):
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def pe_embed(text, dim=PE_DIM):
    digest = hashlib.sha256(text.encode()).digest()
    rng = np.random.RandomState(int.from_bytes(digest[:4], "little"))
    v = np.zeros(dim, np.float32)
    for i, b in enumerate(text.encode()):
        v[i % dim] += ((b / 255.0) * 2 - 1) * (1 + 0.05 * i)
    v += 0.2 * rng.randn(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def make_modes(radius=2.15):
    ang = np.linspace(0, 2 * np.pi, N_MODES, endpoint=False) + np.pi / 2
    return np.stack([radius * np.cos(ang), radius * np.sin(ang)], 1).astype(np.float32)


def build_dataset(centers, pe_list, counts=None, sigma=0.11, seed=0):
    rng = np.random.RandomState(seed)
    counts = counts or [N_PER] * len(centers)
    xs, ys, pes = [], [], []
    for k, (c, n, pe_str) in enumerate(zip(centers, counts, pe_list)):
        if n <= 0:
            continue
        x = c + sigma * rng.randn(n, 2).astype(np.float32)
        xs.append(x)
        ys.append(np.full(n, k, np.int64))
        pe = pe_embed(pe_str)
        pes.append(np.repeat(pe[None], n, 0))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(pes)


class CondDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.pe = nn.Sequential(nn.Linear(PE_DIM, 64), nn.SiLU(), nn.Linear(64, 64))
        self.t = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 32))
        self.net = nn.Sequential(
            nn.Linear(2 + 64 + 32, HIDDEN),
            nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.SiLU(),
            nn.Linear(HIDDEN, 2),
        )

    def forward(self, x, t, pe):
        return self.net(torch.cat([x, self.pe(pe), self.t(t[:, None])], -1))


def diffs(device):
    betas = torch.linspace(1e-4, 0.02, T_DIFF, device=device)
    alphas = 1 - betas
    a_bar = torch.cumprod(alphas, 0)
    return betas, alphas, a_bar


def train(model, x, pe, steps=STEPS):
    ds = TensorDataset(torch.tensor(x), torch.tensor(pe))
    loader = DataLoader(ds, batch_size=min(BATCH, len(x) // 2 * 2), shuffle=True, drop_last=True)
    betas, alphas, a_bar = diffs(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    it = iter(loader)
    model.train()
    for step in range(steps):
        try:
            x0, p = next(it)
        except StopIteration:
            it = iter(loader)
            x0, p = next(it)
        x0, p = x0.to(DEVICE), p.to(DEVICE)
        t = torch.randint(0, T_DIFF, (x0.size(0),), device=DEVICE)
        p_in = p.clone()
        p_in[torch.rand(x0.size(0), device=DEVICE) < 0.15] = 0
        eps = torch.randn_like(x0)
        xt = a_bar[t].sqrt()[:, None] * x0 + (1 - a_bar[t]).sqrt()[:, None] * eps
        loss = ((model(xt, t.float() / T_DIFF, p_in) - eps) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 1000 == 0:
            print(f"  step {step}/{steps} loss={float(loss.detach()):.4f}", flush=True)
    return betas, alphas, a_bar


@torch.no_grad()
def sample(model, consts, pe_vec=None, n=400, cfg=2.8):
    betas, alphas, a_bar = consts
    model.eval()
    x = torch.randn(n, 2, device=DEVICE)
    if pe_vec is None:
        pe = torch.zeros(n, PE_DIM, device=DEVICE)
        cfg = 1.0
    else:
        pe = torch.tensor(pe_vec, device=DEVICE).float()[None].expand(n, -1).clone()
    pe0 = torch.zeros_like(pe)
    for ti in reversed(range(T_DIFF)):
        t = torch.full((n,), ti, device=DEVICE, dtype=torch.long)
        tn = t.float() / T_DIFF
        eps = model(x, tn, pe0) + cfg * (model(x, tn, pe) - model(x, tn, pe0))
        a, ab, b = alphas[ti], a_bar[ti], betas[ti]
        x = (1 / a.sqrt()) * (x - (b / (1 - ab).sqrt()) * eps)
        if ti > 0:
            x = x + b.sqrt() * torch.randn_like(x)
    return x.cpu().numpy()


@torch.no_grad()
def denoise_once(model, consts, x0, t_idx=50):
    betas, alphas, a_bar = consts
    model.eval()
    x0t = torch.tensor(x0, device=DEVICE)
    n = x0t.size(0)
    eps = torch.randn_like(x0t)
    x = a_bar[t_idx].sqrt() * x0t + (1 - a_bar[t_idx]).sqrt() * eps
    pe0 = torch.zeros(n, PE_DIM, device=DEVICE)
    for ti in reversed(range(t_idx + 1)):
        t = torch.full((n,), ti, device=DEVICE, dtype=torch.long)
        e = model(x, t.float() / T_DIFF, pe0)
        a, ab, b = alphas[ti], a_bar[ti], betas[ti]
        x = (1 / a.sqrt()) * (x - (b / (1 - ab).sqrt()) * e)
        if ti > 0:
            x = x + b.sqrt() * torch.randn_like(x)
    return x.cpu().numpy()


def assign(samples, centers, thr=0.48):
    d = ((samples[:, None] - centers[None]) ** 2).sum(-1)
    idx = d.argmin(1)
    ok = np.sqrt(d.min(1)) < thr
    return idx, ok


def target_hit(samples, centers, k):
    idx, ok = assign(samples, centers)
    return float(((idx == k) & ok).mean())


def hist(samples, centers):
    idx, ok = assign(samples, centers)
    h = np.zeros(len(centers))
    for k in idx[ok]:
        h[k] += 1
    return h / max(h.sum(), 1)


# ---------- drawing ----------
# Islands sit at r≈2.15. Letters sit further out on the same ray, clear of the cloud.
CLOUD_LIM = 4.45
LABEL_R = 3.95


def _anchors(ax, centers, labels=True, highlight=None):
    for k, c in enumerate(centers):
        col = COLORS[k]
        lw = 2.2 if highlight == k else 1.15
        alpha = 0.95 if highlight == k else (0.55 if highlight is None else 0.28)
        ax.add_patch(Circle(c, 0.40, fill=False, ec=col, lw=lw, alpha=alpha, zorder=1))
        if labels and k in (0, 2, 4, 6):
            pos = c / (np.linalg.norm(c) + 1e-8) * LABEL_R
            ax.text(
                pos[0], pos[1], LAYOUTS[k],
                ha="center", va="center", fontsize=12, color=col,
                fontproperties=weight("bold"), zorder=6,
                path_effects=[pe.withStroke(linewidth=5, foreground="white")],
            )


def style_cloud(ax, lim=CLOUD_LIM):
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#ffffff")
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(LINE)
        s.set_linewidth(1.0)


def panel_head(ax, title, hit=None):
    ax.set_title(
        title, fontsize=12, color=INK, pad=7, loc="left",
        fontproperties=weight("semibold"),
    )
    if hit is None:
        return
    good = hit >= 0.5
    ax.text(
        1.0, 1.0 + 7 / 300, f"hit {hit:.2f}",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=11, color=(ACCENT_DEEP if good else ALERT_DEEP),
        fontproperties=weight("semibold"), clip_on=False, zorder=8,
    )


def draw_train(ax, train_x, train_y, centers, title, hole=None):
    _anchors(ax, centers, labels=True)
    for k in range(N_MODES):
        m = train_y == k
        if not np.any(m):
            continue
        ax.scatter(
            train_x[m, 0], train_x[m, 1], s=36, c=[COLORS[k]],
            alpha=0.72, linewidths=0, rasterized=True, zorder=3,
        )
    if hole is not None:
        h = centers[hole]
        ax.add_patch(Circle(h, 0.48, fill=False, ec=ALERT_DEEP, lw=1.6, ls=(0, (3.2, 2.2)), zorder=6))
        # Sit the word in the open middle, clear of the dashed ring and of N.
        ax.text(
            0.0, 0.15, "hole",
            color=ALERT_DEEP, fontsize=11, ha="center", va="center",
            fontproperties=weight("semibold"), zorder=7,
        )
    style_cloud(ax)
    panel_head(ax, title)


def draw_samples(
    ax, centers, samples, title, *,
    color=None, target=None, hit=None, mid_pts=None, lim=CLOUD_LIM,
):
    _anchors(ax, centers, labels=True, highlight=target)
    if mid_pts is not None:
        ax.scatter(
            mid_pts[:, 0], mid_pts[:, 1], s=30, c=ALERT,
            alpha=0.95, edgecolors=INK, linewidths=0.6, zorder=6, marker="D",
        )

    r = np.sqrt((samples ** 2).sum(1))
    # Far from every island: leave the rings empty. A blob at the frame
    # only covers the compass letters.
    outside = float(np.median(r)) > 2.6 or float((r > lim * 0.92).mean()) > 0.55
    in_frame = (np.abs(samples[:, 0]) < lim) & (np.abs(samples[:, 1]) < lim)

    if outside:
        # Leave the ring empty. The header carries hit 0.00 — no caption in the middle.
        style_cloud(ax, lim=lim)
    else:
        pts = samples[in_frame] if in_frame.any() else samples
        if color is not None:
            ax.scatter(
                pts[:, 0], pts[:, 1], s=42, color=color,
                alpha=0.8, linewidths=0, rasterized=True, zorder=4,
            )
        else:
            idx, _ = assign(pts, centers, thr=0.9)
            for k in range(N_MODES):
                m = idx == k
                if np.any(m):
                    ax.scatter(
                        pts[m, 0], pts[m, 1], s=42, c=[COLORS[k]],
                        alpha=0.82, linewidths=0, rasterized=True, zorder=4,
                    )
        style_cloud(ax, lim=lim)

    if target is not None and not outside:
        ax.add_patch(
            Circle(centers[target], 0.55, fill=True, fc=COLORS[target], alpha=0.08, zorder=0)
        )
    panel_head(ax, title, hit=hit)


def draw_bars(ax, labels, vals, title, colors=None):
    from fig_style import style_axes
    colors = colors or [ACCENT] * len(vals)
    y = np.arange(len(labels))[::-1]
    style_axes(ax, grid=True)
    ax.barh(y, vals, color=colors, height=0.32, edgecolor="none", zorder=2)
    for yi, v in zip(y, vals):
        if v >= 0.9:
            ax.text(
                v - 0.03, yi, f"{v:.2f}",
                va="center", ha="right", fontsize=14, color="#ffffff",
                fontproperties=weight("medium"),
            )
        else:
            ax.text(
                v + 0.03, yi, f"{v:.2f}",
                va="center", ha="left", fontsize=14, color=INK,
                fontproperties=weight("medium"),
            )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=14, color=INK, fontproperties=weight("regular"))
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("fraction", fontsize=12, color=FAINT, fontproperties=weight("regular"))
    ax.tick_params(axis="x", labelsize=12)
    panel_head(ax, title)


def fig_bg(fig):
    fig.patch.set_facecolor(PAPER)


def save(fig, path):
    fig.savefig(path, bbox_inches="tight", facecolor=PAPER, dpi=130, pad_inches=0.08)
    plt.close(fig)
    print("wrote", path)


def run_train_and_cache():
    print(f"polish 2D on {DEVICE}", flush=True)
    set_seed(0)
    centers = make_modes()
    pe_list = list(PE_TEMPLATES)

    print("train A (balanced)", flush=True)
    train_x, train_y, train_pe = build_dataset(centers, pe_list, seed=0)
    model_a = CondDenoiser().to(DEVICE)
    consts_a = train(model_a, train_x, train_pe, steps=STEPS)

    exact = {k: sample(model_a, consts_a, pe_embed(PE_TEMPLATES[k]), n=380, cfg=3.0) for k in range(N_MODES)}
    exact0 = exact[0]
    para0 = sample(model_a, consts_a, pe_embed(PE_PARA[0]), n=380, cfg=3.0)
    typo_str = PE_TEMPLATES[0].replace("layout=n |", "layout=nroth |")
    assert typo_str != PE_TEMPLATES[0], "typo must change PE"
    typo0 = sample(model_a, consts_a, pe_embed(typo_str), n=380, cfg=3.0)
    uncond = sample(model_a, consts_a, None, n=600)

    print("train B (hole)", flush=True)
    set_seed(1)
    hole = 1
    counts_b = [N_PER] * N_MODES
    counts_b[hole] = 0
    xb, yb, peb = build_dataset(centers, pe_list, counts_b, seed=1)
    model_b = CondDenoiser().to(DEVICE)
    consts_b = train(model_b, xb, peb, steps=STEPS)
    ask_hole = sample(model_b, consts_b, pe_embed(PE_TEMPLATES[hole]), n=450, cfg=3.0)
    uncond_b = sample(model_b, consts_b, None, n=700)

    print("train C (mass)", flush=True)
    set_seed(2)
    counts_c = [N_PER] * N_MODES
    counts_c[0] = N_PER * 8
    xc, yc, pec = build_dataset(centers, pe_list, counts_c, seed=2)
    model_c = CondDenoiser().to(DEVICE)
    consts_c = train(model_c, xc, pec, steps=STEPS)
    uncond_c = sample(model_c, consts_c, None, n=1200)
    emp = np.array(counts_c, float)
    emp /= emp.sum()
    gen = hist(uncond_c, centers)

    print("train D (swap)", flush=True)
    set_seed(3)
    pe_swapped = list(PE_TEMPLATES)
    pe_swapped[0], pe_swapped[4] = pe_swapped[4], pe_swapped[0]
    xd, yd, ped = build_dataset(centers, pe_swapped, seed=3)
    model_d = CondDenoiser().to(DEVICE)
    consts_d = train(model_d, xd, ped, steps=STEPS)
    ask_n = sample(model_d, consts_d, pe_embed(PE_TEMPLATES[0]), n=450, cfg=3.0)
    ask_s = sample(model_d, consts_d, pe_embed(PE_TEMPLATES[4]), n=450, cfg=3.0)

    mid = np.stack([0.5 * (centers[k] + centers[(k + 1) % N_MODES]) for k in range(N_MODES)], 0)
    mid_pts = np.repeat(mid, 55, 0)
    mid_pts = mid_pts + 0.025 * np.random.randn(*mid_pts.shape).astype(np.float32)
    recovered = denoise_once(model_a, consts_a, mid_pts, t_idx=48)
    pe_avg = pe_embed(PE_TEMPLATES[0]) + pe_embed(PE_TEMPLATES[2])
    pe_avg = pe_avg / (np.linalg.norm(pe_avg) + 1e-8)
    s_avg = sample(model_a, consts_a, pe_avg, n=400, cfg=3.0)

    np.savez_compressed(
        CACHE,
        centers=centers,
        train_x=train_x, train_y=train_y,
        exact0=exact0, exact2=exact[2], exact5=exact[5], exact1=exact[1],
        para0=para0, typo0=typo0, uncond=uncond,
        xb=xb, yb=yb, ask_hole=ask_hole, uncond_b=uncond_b,
        emp=emp, gen=gen,
        xd=xd, yd=yd, ask_n=ask_n, ask_s=ask_s,
        mid_pts=mid_pts, recovered=recovered, s_avg=s_avg, s_ne=exact[1],
        hole=np.array([hole]),
    )
    print("cached", CACHE)
    return dict(np.load(CACHE, allow_pickle=False))


def load_cache():
    if not CACHE.exists():
        raise SystemExit(f"missing cache {CACHE}; run without --redraw first")
    return dict(np.load(CACHE, allow_pickle=False))


def draw_all(d):
    apply_style()
    centers = d["centers"]
    train_x, train_y = d["train_x"], d["train_y"]
    exact0, para0, typo0 = d["exact0"], d["para0"], d["typo0"]
    uncond = d["uncond"]
    exact2, exact5, exact1 = d["exact2"], d["exact5"], d["exact1"]
    xb, yb = d["xb"], d["yb"]
    ask_hole, uncond_b = d["ask_hole"], d["uncond_b"]
    emp, gen = d["emp"], d["gen"]
    ask_n, ask_s = d["ask_n"], d["ask_s"]
    mid_pts, recovered = d["mid_pts"], d["recovered"]
    s_avg, s_ne = d["s_avg"], d["s_ne"]
    hole = int(d["hole"][0])

    hits = [
        target_hit(exact0, centers, 0),
        target_hit(para0, centers, 0),
        target_hit(typo0, centers, 0),
        float(assign(uncond, centers)[1].mean()),
    ]

    # ---- Figure 20: two large columns, labels clear of the clouds ----
    fig = plt.figure(figsize=(7.4, 6.8), dpi=130)
    fig_bg(fig)
    gs = fig.add_gridspec(
        2, 2, hspace=0.28, wspace=0.12,
        left=0.04, right=0.98, top=0.94, bottom=0.04,
    )
    draw_train(fig.add_subplot(gs[0, 0]), train_x, train_y, centers, "(a) Training mass")
    draw_samples(fig.add_subplot(gs[0, 1]), centers, exact0, "(b) Exact key → N", target=0, hit=hits[0])
    draw_samples(
        fig.add_subplot(gs[1, 0]), centers, para0, "(c) Paraphrase",
        color=MUTED, target=0, hit=hits[1],
    )
    draw_samples(
        fig.add_subplot(gs[1, 1]), centers, typo0, "(d) Typo in key",
        color=MUTED, target=0, hit=hits[2],
    )
    save(fig, FIG / "20_2d_pe_board.png")

    # ---- Figure 21: holes, mass, swapped tags — two columns ----
    fig = plt.figure(figsize=(7.4, 8.6), dpi=130)
    fig_bg(fig)
    gs = fig.add_gridspec(
        3, 2, height_ratios=[1.05, 0.78, 1.05],
        hspace=0.32, wspace=0.14,
        left=0.08, right=0.97, top=0.97, bottom=0.05,
    )
    draw_train(fig.add_subplot(gs[0, 0]), xb, yb, centers, "(a) Train, NE removed", hole=hole)
    hole_hit = float(((assign(ask_hole, centers)[0] == hole) & assign(ask_hole, centers)[1]).mean())
    draw_samples(
        fig.add_subplot(gs[0, 1]), centers, ask_hole, "(b) Ask for NE",
        color=ALERT, target=hole, hit=hole_hit,
    )

    from fig_style import style_axes

    am = fig.add_subplot(gs[1, :])
    x = np.arange(N_MODES)
    style_axes(am, grid="y")
    am.bar(x - BAR_PAIR_OFF, emp, BAR_PAIR, color=STONE, edgecolor=EDGE, linewidth=EDGE_W,
           zorder=2, label="training data")
    am.bar(x + BAR_PAIR_OFF, gen, BAR_PAIR, color=ACCENT, edgecolor=EDGE, linewidth=EDGE_W,
           zorder=2, label="unconditional samples")
    am.set_xticks(x)
    am.set_xticklabels(LAYOUTS, fontsize=10.5, color=INK, fontproperties=weight("medium"))
    am.set_xlim(-0.55, N_MODES - 0.45)
    am.set_ylim(0, max(float(emp.max()), float(gen.max())) * 1.35)
    am.set_ylabel("fraction", fontsize=11, color=INK)
    am.legend(loc="upper right", frameon=True, fancybox=True, framealpha=1.0, edgecolor="#d1d5db",
              prop=weight("regular"), fontsize=10.5, borderpad=0.5, handlelength=1.4)
    panel_head(am, "(c) North has 8× the data")

    hn = target_hit(ask_n, centers, 4)
    hs = target_hit(ask_s, centers, 0)
    draw_samples(
        fig.add_subplot(gs[2, 0]), centers, ask_n, "(d) Ask “north” → S",
        color=COLORS[4], target=4, hit=hn,
    )
    draw_samples(
        fig.add_subplot(gs[2, 1]), centers, ask_s, "(e) Ask “south” → N",
        color=COLORS[0], target=0, hit=hs,
    )
    save(fig, FIG / "21_2d_ablation_board.png")

    # ---- Figure 22: 2×2, large ----
    fig = plt.figure(figsize=(7.4, 6.6), dpi=130)
    fig_bg(fig)
    gs = fig.add_gridspec(
        2, 2,         hspace=0.28, wspace=0.14,
        left=0.06, right=0.97, top=0.96, bottom=0.04,
    )
    ax = fig.add_subplot(gs[0, 0])
    draw_train(ax, train_x, train_y, centers, "(a) Start on midpoints")
    marks = mid_pts.reshape(N_MODES, -1, 2).mean(1)
    ax.scatter(marks[:, 0], marks[:, 1], s=46, c=ALERT, alpha=1.0,
               edgecolors=INK, linewidths=0.7, zorder=6, marker="D")
    draw_samples(fig.add_subplot(gs[0, 1]), centers, recovered, "(b) After denoising")
    draw_samples(
        fig.add_subplot(gs[1, 0]), centers, s_ne, "(c) Exact key → NE",
        target=1, hit=target_hit(s_ne, centers, 1),
    )
    draw_samples(
        fig.add_subplot(gs[1, 1]), centers, s_avg, "(d) Mean of keys N and E",
        color=ALERT, target=1, hit=target_hit(s_avg, centers, 1),
    )
    save(fig, FIG / "22_2d_geometry_board.png")

    metrics = {
        "exact_hit": hits[0],
        "para_hit": hits[1],
        "typo_hit": hits[2],
        "hole_in_hole": float(
            ((assign(ask_hole, centers)[0] == hole) & assign(ask_hole, centers)[1]).mean()
        ),
        "north_str_to_south": hn,
        "south_str_to_north": hs,
        "mass_train_north": float(emp[0]),
        "mass_gen_north": float(gen[0]),
        "avg_pe_hit_ne": target_hit(s_avg, centers, 1),
    }
    (FIG / "metrics_2d_polish.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--redraw", action="store_true", help="reuse cached samples")
    args = ap.parse_args()
    d = load_cache() if args.redraw else run_train_and_cache()
    draw_all(d)


if __name__ == "__main__":
    main()
