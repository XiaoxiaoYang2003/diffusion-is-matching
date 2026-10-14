#!/usr/bin/env python3
"""Larger toy: diffusion is a PE-keyed renderer; matching ≫ generalization.

Claim
-----
Exact PE keys *render* the co-trained island.
Held-out cells inside the training hull (interpolation) and outside it
(extrapolation) both snap to nearby training mass — that is matching, not
generalization. Paraphrases leave support.

Layout
------
5×5 Gaussian lattice. Eight workers on GPUs 0–7:

  0  full train          → exact render works
  1  checkerboard train  → held-out cells = interp
  2  center 3×3 train    → outer ring = extrap
  3  four corners only   → edge midpoints + center
  4  left two columns    → right side = attribute extrap
  5  tag-swapped lattice → string follows co-occurrence
  6  paraphrase / typo   → null keys on full model
  7  PE arithmetic       → mean(PE_a, PE_b) ≠ midpoint mode

Cluster:
    bash cluster_submit/submit_toy.sh
Local smoke:
    TOY_SMOKE=1 python3 experiments/toy_render_suite.py --smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.patches import Circle
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import (  # noqa: E402
    ACCENT,
    ALERT,
    ALERT_DEEP,
    EDGE,
    EDGE_W,
    STONE,
    BAR_H,
    BAR_PAIR,
    BAR_PAIR_OFF,
    FAINT,
    INK,
    LINE,
    MUTED,
    PAPER,
    PANEL,
    SOFT,
    apply_style,
    style_axes,
    weight,
)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
CACHE = FIG / "cache_render_suite.pkl"

GRID = 5
N_MODES = GRID * GRID
PE_DIM = 24
HIDDEN = 320
T_DIFF = 120
LR = 2e-3
SIGMA = 0.10


def pe_str(r: int, c: int) -> str:
    return f"[PE] subject=orb | row={r} | col={c} | quality=hq | style=flat"


def pe_embed(text: str, dim: int = PE_DIM) -> np.ndarray:
    digest = hashlib.sha256(text.encode()).digest()
    rng = np.random.RandomState(int.from_bytes(digest[:4], "little"))
    v = np.zeros(dim, np.float32)
    for i, b in enumerate(text.encode()):
        v[i % dim] += ((b / 255.0) * 2 - 1) * (1 + 0.04 * i)
    v += 0.15 * rng.randn(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def make_lattice(grid=GRID, span=4.2):
    xs = np.linspace(-span / 2, span / 2, grid)
    ys = np.linspace(span / 2, -span / 2, grid)  # row0 at top
    centers = np.zeros((grid * grid, 2), np.float32)
    labels = []
    for r in range(grid):
        for c in range(grid):
            k = r * grid + c
            centers[k] = (xs[c], ys[r])
            labels.append((r, c))
    return centers, labels


def idx_of(r, c, grid=GRID):
    return r * grid + c


def mask_full(_labels):
    return np.ones(N_MODES, dtype=bool)


def mask_checkerboard(labels):
    m = np.zeros(N_MODES, dtype=bool)
    for i, (r, c) in enumerate(labels):
        m[i] = (r + c) % 2 == 0
    return m


def mask_center3(labels):
    m = np.zeros(N_MODES, dtype=bool)
    for i, (r, c) in enumerate(labels):
        m[i] = 1 <= r <= 3 and 1 <= c <= 3
    return m


def mask_corners(labels):
    m = np.zeros(N_MODES, dtype=bool)
    corners = {(0, 0), (0, GRID - 1), (GRID - 1, 0), (GRID - 1, GRID - 1)}
    for i, rc in enumerate(labels):
        m[i] = rc in corners
    return m


def mask_left2(labels):
    m = np.zeros(N_MODES, dtype=bool)
    for i, (r, c) in enumerate(labels):
        m[i] = c <= 1
    return m


class CondDenoiser(nn.Module):
    def __init__(self, pe_dim=PE_DIM, hidden=HIDDEN):
        super().__init__()
        self.pe = nn.Sequential(nn.Linear(pe_dim, 96), nn.SiLU(), nn.Linear(96, 96))
        self.t = nn.Sequential(nn.Linear(1, 48), nn.SiLU(), nn.Linear(48, 48))
        self.net = nn.Sequential(
            nn.Linear(2 + 96 + 48, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, x, t, pe):
        return self.net(torch.cat([x, self.pe(pe), self.t(t[:, None])], -1))


def diffs(device, t_diff):
    betas = torch.linspace(1e-4, 0.02, t_diff, device=device)
    alphas = 1 - betas
    a_bar = torch.cumprod(alphas, 0)
    return betas, alphas, a_bar


def build_dataset(centers, train_mask, n_per, seed=0, pe_map=None):
    rng = np.random.RandomState(seed)
    pe_map = pe_map or {k: pe_str(*divmod(k, GRID)) for k in range(N_MODES)}
    xs, pes, ys = [], [], []
    for k in range(N_MODES):
        if not train_mask[k]:
            continue
        x = centers[k] + SIGMA * rng.randn(n_per, 2).astype(np.float32)
        pe = pe_embed(pe_map[k])
        xs.append(x)
        pes.append(np.repeat(pe[None], n_per, 0))
        ys.append(np.full(n_per, k, np.int64))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(pes)


def train_model(device, x, pe, steps, batch, t_diff, seed=0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = CondDenoiser().to(device)
    ds = TensorDataset(torch.tensor(x), torch.tensor(pe))
    bs = min(batch, max(32, (len(x) // 2) * 2))
    if bs < 32 or len(x) < 64:
        # tiny corner sets: repeat data
        x_rep = np.repeat(x, max(1, 256 // max(len(x), 1)), 0)
        pe_rep = np.repeat(pe, max(1, 256 // max(len(pe), 1)), 0)
        ds = TensorDataset(torch.tensor(x_rep), torch.tensor(pe_rep))
        bs = min(batch, len(x_rep) // 2 * 2)
    loader = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)
    betas, alphas, a_bar = diffs(device, t_diff)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    it = iter(loader)
    model.train()
    for step in range(steps):
        try:
            x0, p = next(it)
        except StopIteration:
            it = iter(loader)
            x0, p = next(it)
        x0, p = x0.to(device), p.to(device)
        t = torch.randint(0, t_diff, (x0.size(0),), device=device)
        p_in = p.clone()
        p_in[torch.rand(x0.size(0), device=device) < 0.12] = 0
        eps = torch.randn_like(x0)
        xt = a_bar[t].sqrt()[:, None] * x0 + (1 - a_bar[t]).sqrt()[:, None] * eps
        loss = ((model(xt, t.float() / t_diff, p_in) - eps) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 1500 == 0:
            print(f"  [{device}] step {step}/{steps} loss={float(loss.detach()):.4f}", flush=True)
    return model, (betas, alphas, a_bar)


@torch.no_grad()
def sample(model, consts, pe_vec, n, cfg, t_diff, device):
    betas, alphas, a_bar = consts
    model.eval()
    x = torch.randn(n, 2, device=device)
    if pe_vec is None:
        pe = torch.zeros(n, PE_DIM, device=device)
        cfg = 1.0
    else:
        pe = torch.tensor(pe_vec, device=device).float()[None].expand(n, -1).clone()
    pe0 = torch.zeros_like(pe)
    for ti in reversed(range(t_diff)):
        t = torch.full((n,), ti, device=device, dtype=torch.long)
        tn = t.float() / t_diff
        eps = model(x, tn, pe0) + cfg * (model(x, tn, pe) - model(x, tn, pe0))
        a, ab, b = alphas[ti], a_bar[ti], betas[ti]
        x = (1 / a.sqrt()) * (x - (b / (1 - ab).sqrt()) * eps)
        if ti > 0:
            x = x + b.sqrt() * torch.randn_like(x)
    return x.cpu().numpy()


def assign(samples, centers, thr=0.55):
    d = ((samples[:, None] - centers[None]) ** 2).sum(-1)
    idx = d.argmin(1)
    ok = np.sqrt(d.min(1)) < thr
    return idx, ok


def hit_rate(samples, centers, target_k, thr=0.55):
    idx, ok = assign(samples, centers, thr=thr)
    return float(((idx == target_k) & ok).mean())


def nearest_train_hit(samples, centers, train_mask, thr=0.55):
    train_ids = np.where(train_mask)[0]
    if len(train_ids) == 0:
        return 0.0, np.array([], dtype=np.int64)
    d = ((samples[:, None] - centers[train_ids][None]) ** 2).sum(-1)
    near = train_ids[d.argmin(1)]
    ok_near = np.sqrt(d.min(1)) < thr
    return float(ok_near.mean()), near


def eval_cells(model, consts, centers, cell_ids, n_per, cfg, t_diff, device, train_mask):
    rows = []
    for k in cell_ids:
        r, c = divmod(k, GRID)
        s = sample(model, consts, pe_embed(pe_str(r, c)), n=n_per, cfg=cfg, t_diff=t_diff, device=device)
        th = hit_rate(s, centers, k)
        nth, _ = nearest_train_hit(s, centers, train_mask)
        # leave-support rate
        rnorm = np.sqrt((s ** 2).sum(1))
        leave = float((rnorm > 3.6).mean())
        rows.append(
            {
                "k": int(k),
                "r": int(r),
                "c": int(c),
                "target_hit": th,
                "nearest_train_hit": nth,
                "leave_support": leave,
                "samples": s,
            }
        )
    return rows


# ---------- drawing ----------
def style_ax(ax, lim=3.15):
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(PAPER)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.add_patch(
        plt.Rectangle(
            (-lim, -lim), 2 * lim, 2 * lim, fill=False,
            ec="#d6d3cc", lw=1.0, zorder=6, transform=ax.transData, clip_on=False,
        )
    )


def draw_lattice(ax, centers, train_mask, title, samples=None, sample_color=None, highlight=None):
    for k, c in enumerate(centers):
        if train_mask[k]:
            ax.add_patch(Circle(c, 0.28, fc=ACCENT, ec=EDGE, lw=0.7, alpha=0.9, zorder=1))
        else:
            ax.add_patch(Circle(c, 0.28, fc="#ffffff", ec=STONE, lw=1.1, zorder=1))
        ax.plot(c[0], c[1], "o", ms=2.0, color=INK, alpha=0.45, zorder=2)
    if samples is not None and len(samples):
        # Keep markers inside the frame. Rasterized scatters otherwise paint across the border.
        frame = 2.95
        keep = (np.abs(samples[:, 0]) < frame) & (np.abs(samples[:, 1]) < frame)
        shown = samples[keep]
        if len(shown):
            ax.scatter(
                shown[:, 0], shown[:, 1], s=13,
                c=sample_color or ALERT, alpha=0.8, linewidths=0, zorder=4, clip_on=True,
            )
    if highlight is not None:
        h = centers[highlight]
        ax.add_patch(Circle(h, 0.36, fill=False, ec=ALERT_DEEP, lw=1.4, ls=(0, (2.5, 1.5)), zorder=5))
    style_ax(ax)
    ax.set_title(
        title, fontsize=12, color=INK, pad=7, loc="left",
        fontproperties=weight("semibold"),
    )


def bar_compare(ax, labels, target, nearest, title):
    x = np.arange(len(labels))
    style_axes(ax, grid="y")
    ax.bar(x - BAR_PAIR_OFF, target, BAR_PAIR, color=STONE, edgecolor=EDGE, linewidth=EDGE_W,
           zorder=2, label="hit on the cell we asked for")
    ax.bar(x + BAR_PAIR_OFF, nearest, BAR_PAIR, color=ACCENT, edgecolor=EDGE, linewidth=EDGE_W,
           zorder=2, label="hit on the nearest trained cell")
    for xi, a, b in zip(x, target, nearest):
        for dx, v in ((-BAR_PAIR_OFF, a), (BAR_PAIR_OFF, b)):
            ax.text(xi + dx, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=9,
                    color=INK if v > 0.02 else MUTED, fontproperties=weight("medium"))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.5, fontproperties=weight("medium"), rotation=0, ha="center")
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlim(-0.55, len(labels) - 0.45)
    ax.set_ylabel("hit rate", fontsize=11, color=INK)
    ax.legend(loc="upper right", frameon=True, fancybox=True, framealpha=1.0, edgecolor="#d1d5db",
              prop=weight("regular"), fontsize=10, borderpad=0.5, handlelength=1.4)
    ax.set_title(title, fontsize=12, color=INK, pad=7, loc="left", fontproperties=weight("semibold"))


def save_fig(fig, path):
    fig.savefig(path, bbox_inches="tight", facecolor=PAPER, dpi=140, pad_inches=0.12)
    plt.close(fig)
    print("wrote", path, flush=True)


# ---------- workers ----------
def run_worker(job):
    """Each job is picklable dict. Returns metrics + sample blobs for plotting on rank0."""
    name = job["name"]
    gpu = job["gpu"]
    steps = job["steps"]
    n_per = job["n_per"]
    batch = job["batch"]
    t_diff = job["t_diff"]
    n_samp = job["n_samp"]
    cfg = job["cfg"]
    seed = job["seed"]

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[{name}] start on {device}", flush=True)
    centers, labels = make_lattice()
    train_mask = job["mask_fn"](labels)
    pe_map = {k: pe_str(*divmod(k, GRID)) for k in range(N_MODES)}

    if name == "swap":
        a, b = idx_of(0, 0), idx_of(4, 4)
        pe_map[a], pe_map[b] = pe_map[b], pe_map[a]
        train_mask = mask_full(labels)

    x, y, pe = build_dataset(centers, train_mask, n_per, seed=seed, pe_map=pe_map)
    model, consts = train_model(device, x, pe, steps, batch, t_diff, seed=seed)
    out = {"name": name, "train_mask": train_mask.astype(np.uint8), "metrics": {}}

    if name == "full":
        exact_ks = [idx_of(0, 0), idx_of(2, 2), idx_of(4, 4), idx_of(1, 3)]
        exact_rows = eval_cells(model, consts, centers, exact_ks, n_samp, cfg, t_diff, device, train_mask)
        out["exact"] = [{k: v for k, v in r.items() if k != "samples"} for r in exact_rows]
        out["exact_samples"] = {r["k"]: r["samples"] for r in exact_rows}
        para = sample(
            model, consts,
            pe_embed("high quality orb at row 2 column 2, flat style"),
            n=n_samp, cfg=cfg, t_diff=t_diff, device=device,
        )
        typo = sample(
            model, consts,
            pe_embed(pe_str(2, 2).replace("row=2", "row=2x")),
            n=n_samp, cfg=cfg, t_diff=t_diff, device=device,
        )
        out["metrics"]["exact_mean_target"] = float(np.mean([r["target_hit"] for r in exact_rows]))
        out["metrics"]["para_target"] = hit_rate(para, centers, idx_of(2, 2))
        out["metrics"]["typo_target"] = hit_rate(typo, centers, idx_of(2, 2))
        out["metrics"]["para_nearest"] = nearest_train_hit(para, centers, train_mask)[0]
        out["metrics"]["typo_nearest"] = nearest_train_hit(typo, centers, train_mask)[0]
        out["para_samples"] = para
        out["typo_samples"] = typo
        # PE arithmetic on the same trained renderer
        mid = idx_of(2, 2)
        pe_avg = pe_embed(pe_str(2, 1)) + pe_embed(pe_str(2, 3))
        pe_avg = pe_avg / (np.linalg.norm(pe_avg) + 1e-8)
        s_avg = sample(model, consts, pe_avg, n=n_samp, cfg=cfg, t_diff=t_diff, device=device)
        s_mid = out["exact_samples"][mid]
        out["metrics"]["avg_hit_mid"] = hit_rate(s_avg, centers, mid)
        out["metrics"]["exact_hit_mid"] = hit_rate(s_mid, centers, mid)
        out["metrics"]["avg_nearest"] = nearest_train_hit(s_avg, centers, train_mask)[0]
        out["s_avg"] = s_avg
        out["s_mid"] = s_mid

    elif name in ("checker", "center3", "corners", "left2", "checker_b", "center3_b"):
        held = np.where(~train_mask)[0]
        held = held[: min(len(held), job.get("max_held", 12))]
        rows = eval_cells(model, consts, centers, held.tolist(), n_samp, cfg, t_diff, device, train_mask)
        out["held"] = [{k: v for k, v in r.items() if k != "samples"} for r in rows]
        out["held_samples"] = {r["k"]: r["samples"] for r in rows}
        trained = np.where(train_mask)[0][:4]
        erows = eval_cells(model, consts, centers, trained.tolist(), n_samp, cfg, t_diff, device, train_mask)
        out["exact"] = [{k: v for k, v in r.items() if k != "samples"} for r in erows]
        out["metrics"]["held_target_mean"] = float(np.mean([r["target_hit"] for r in rows]))
        out["metrics"]["held_nearest_mean"] = float(np.mean([r["nearest_train_hit"] for r in rows]))
        out["metrics"]["held_leave_mean"] = float(np.mean([r["leave_support"] for r in rows]))
        out["metrics"]["exact_target_mean"] = float(np.mean([r["target_hit"] for r in erows]))

    elif name == "swap":
        ask_a = sample(model, consts, pe_embed(pe_str(0, 0)), n=n_samp, cfg=cfg, t_diff=t_diff, device=device)
        ask_b = sample(model, consts, pe_embed(pe_str(4, 4)), n=n_samp, cfg=cfg, t_diff=t_diff, device=device)
        out["metrics"]["ask_00_hits_44"] = hit_rate(ask_a, centers, idx_of(4, 4))
        out["metrics"]["ask_44_hits_00"] = hit_rate(ask_b, centers, idx_of(0, 0))
        out["ask_a"] = ask_a
        out["ask_b"] = ask_b

    print(f"[{name}] done metrics={out['metrics']}", flush=True)
    return out


def plot_all(results, centers):
    apply_style()
    by = {r["name"]: r for r in results}

    # ---- Fig 30: large panels, two columns ----
    fig = plt.figure(figsize=(7.6, 9.2), dpi=140)
    fig.patch.set_facecolor(PAPER)
    gs = fig.add_gridspec(
        3, 2, height_ratios=[1.05, 1.05, 0.72],
        hspace=0.28, wspace=0.14, left=0.05, right=0.98, top=0.97, bottom=0.05,
    )

    # exact render
    ax = fig.add_subplot(gs[0, 0])
    full = by["full"]
    tm = full["train_mask"].astype(bool)
    k0 = list(full["exact_samples"].keys())[1] if len(full["exact_samples"]) > 1 else list(full["exact_samples"].keys())[0]
    draw_lattice(
        ax, centers, tm, "(a) Exact key, trained cell",
        samples=full["exact_samples"][k0], sample_color="#1f3f66",
    )

    # interp (checker)
    ax = fig.add_subplot(gs[0, 1])
    ch = by["checker"]
    tm = ch["train_mask"].astype(bool)
    hs = np.concatenate(list(ch["held_samples"].values()), 0) if ch.get("held_samples") else None
    draw_lattice(
        ax, centers, tm, "(b) Checkerboard held out",
        samples=None if hs is None else hs[::2], sample_color=ALERT,
    )

    # extrap (center3)
    ax = fig.add_subplot(gs[1, 0])
    c3 = by["center3"]
    tm = c3["train_mask"].astype(bool)
    hs = np.concatenate(list(c3["held_samples"].values()), 0) if c3.get("held_samples") else None
    draw_lattice(
        ax, centers, tm, "(c) Outer ring held out",
        samples=None if hs is None else hs[::2], sample_color=ALERT,
    )

    # bars: exact vs interp vs extrap vs para
    ax = fig.add_subplot(gs[2, :])
    labels = ["exact", "checker", "outer", "paraphrase", "typo"]
    target = [
        full["metrics"]["exact_mean_target"],
        ch["metrics"]["held_target_mean"],
        c3["metrics"]["held_target_mean"],
        full["metrics"]["para_target"],
        full["metrics"]["typo_target"],
    ]
    nearest = [
        full["metrics"]["exact_mean_target"],  # exact: target == train
        ch["metrics"]["held_nearest_mean"],
        c3["metrics"]["held_nearest_mean"],
        full["metrics"]["para_nearest"],
        full["metrics"]["typo_nearest"],
    ]
    bar_compare(ax, labels, target, nearest, "(e) Where the samples land")

    # Fourth lattice, same size as the others — no bar chart beside a grid.
    ax = fig.add_subplot(gs[1, 1])
    corners = by["corners"]
    tm = corners["train_mask"].astype(bool)
    hs = np.concatenate(list(corners["held_samples"].values()), 0) if corners.get("held_samples") else None
    draw_lattice(
        ax, centers, tm, "(d) Only corners trained",
        samples=None if hs is None else hs[::2], sample_color=ALERT,
    )

    save_fig(fig, FIG / "30_render_matching_board.png")

    # ---- Fig 31: interp detail ----
    fig = plt.figure(figsize=(7.6, 6.8), dpi=140)
    fig.patch.set_facecolor(PAPER)
    gs = fig.add_gridspec(2, 2, wspace=0.18, hspace=0.42, left=0.06, right=0.98, top=0.93, bottom=0.08)
    for i, (name, title) in enumerate([
        ("checker", "(a) Checkerboard held out"),
        ("corners", "(b) Only corners trained"),
        ("left2", "(c) Right columns held out"),
        ("center3", "(d) Outer ring held out"),
    ]):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        r = by[name]
        tm = r["train_mask"].astype(bool)
        hs = np.concatenate(list(r["held_samples"].values()), 0) if r.get("held_samples") else None
        draw_lattice(
            ax, centers, tm, title,
            samples=None if hs is None else hs[::2], sample_color=ALERT,
        )
        ax.set_xlabel(
            f"asked {r['metrics']['held_target_mean']:.2f}  ·  nearest {r['metrics']['held_nearest_mean']:.2f}",
            fontsize=10.5, color=INK, labelpad=7, fontproperties=weight("medium"),
        )
    save_fig(fig, FIG / "31_interp_extrap_grid.png")

    # ---- Fig 32: paraphrase failure on full ----
    fig = plt.figure(figsize=(11.2, 3.7), dpi=220)
    fig.patch.set_facecolor(PAPER)
    gs = fig.add_gridspec(1, 3, wspace=0.20, left=0.03, right=0.99, top=0.82, bottom=0.08)
    full = by["full"]
    tm = full["train_mask"].astype(bool)
    ax = fig.add_subplot(gs[0, 0])
    draw_lattice(ax, centers, tm, "Train support (full 5×5)")
    ax = fig.add_subplot(gs[0, 1])
    k_mid = idx_of(2, 2)
    draw_lattice(ax, centers, tm, "Exact key → (2,2)", samples=full["exact_samples"][k_mid], sample_color=ACCENT, highlight=k_mid)
    ax = fig.add_subplot(gs[0, 2])
    draw_lattice(ax, centers, tm, "Paraphrase of (2,2)", samples=full["para_samples"], sample_color=ALERT, highlight=k_mid)
    fig.suptitle(
        "Exact key opens the drawer; paraphrase is another (empty) key",
        fontsize=15.5, color=INK, fontproperties=weight("medium"), y=0.97,
    )
    save_fig(fig, FIG / "32_exact_vs_paraphrase.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--gpus", type=str, default="")
    ap.add_argument("--redraw", action="store_true", help="replot from the cached results")
    args = ap.parse_args()
    if os.environ.get("TOY_SMOKE", "0") == "1":
        args.smoke = True

    if args.redraw:
        with open(CACHE, "rb") as f:
            results = pickle.load(f)
        centers, _ = make_lattice()
        plot_all(results, centers)
        print("redraw complete", flush=True)
        return

    if args.smoke:
        steps, n_per, batch, t_diff, n_samp = 800, 80, 256, 60, 120
    else:
        steps, n_per, batch, t_diff, n_samp = 7000, 280, 768, T_DIFF, 280

    # discover GPUs
    if args.gpus:
        gpus = [int(x) for x in args.gpus.split(",")]
    else:
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        gpus = list(range(min(8, max(n, 1))))
    print(f"suite on gpus={gpus} smoke={args.smoke} steps={steps}", flush=True)

    job_specs = [
        ("full", mask_full, 0),
        ("checker", mask_checkerboard, 1),
        ("center3", mask_center3, 2),
        ("corners", mask_corners, 3),
        ("left2", mask_left2, 4),
        ("swap", mask_full, 5),
        ("checker_b", mask_checkerboard, 6),  # second seed / more held coverage
        ("center3_b", mask_center3, 7),
    ]
    payload = []
    for name, mask_fn, gi in job_specs:
        if gi >= len(gpus) and name.endswith("_b"):
            continue
        payload.append({
            "name": name, "mask_fn": mask_fn,
            "gpu": gpus[gi % len(gpus)], "seed": gi,
            "steps": steps, "n_per": n_per, "batch": batch,
            "t_diff": t_diff, "n_samp": n_samp, "cfg": 2.8,
            "max_held": 16,
        })

    unique = len({j["gpu"] for j in payload})
    results = []
    if unique == len(payload):
        with ProcessPoolExecutor(max_workers=len(payload)) as ex:
            futs = {ex.submit(run_worker, j): j["name"] for j in payload}
            for fut in as_completed(futs):
                results.append(fut.result())
    else:
        for j in payload:
            results.append(run_worker(j))

    # merge seed-b replicas into primary held stats (average)
    for base_name, alias in (("checker", "checker_b"), ("center3", "center3_b")):
        names = {r["name"] for r in results}
        if alias in names and base_name in names:
            a = next(r for r in results if r["name"] == alias)
            b = next(r for r in results if r["name"] == base_name)
            for k in ("held_target_mean", "held_nearest_mean", "held_leave_mean"):
                if k in a["metrics"] and k in b["metrics"]:
                    b["metrics"][k] = 0.5 * (a["metrics"][k] + b["metrics"][k])
            # keep denser sample pool from alias if larger
            if len(a.get("held_samples", {})) >= len(b.get("held_samples", {})):
                b["held_samples"] = a["held_samples"]
                b["held"] = a["held"]
            results = [r for r in results if r["name"] != alias]

    centers, _ = make_lattice()
    with open(CACHE, "wb") as f:
        pickle.dump(results, f)
    plot_all(results, centers)

    metrics = {r["name"]: r["metrics"] for r in results}
    (FIG / "metrics_render_suite.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print("suite complete", flush=True)


if __name__ == "__main__":
    # Required for CUDA + ProcessPoolExecutor spawn
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
