#!/usr/bin/env python3
"""Empirical toys: diffusion matches training mass + PE keys.

Runs four independent trainings in parallel on CUDA devices 0–3, then
writes figures/ + metrics.json for the blog.

    python3 experiments/toy_matching.py
    # or: CUDA_VISIBLE_DEVICES=0,1,2,3 python3 experiments/toy_matching.py
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SEED = 0
N_MODES = 8
N_PER_MODE = 400
STEPS = 6000
BATCH = 1024
HIDDEN = 256
T_DIFF = 100
LR = 2e-3
PE_DIM = 16
DEVICES = [0, 1, 2, 3]

LAYOUTS = ["north", "ne", "east", "se", "south", "sw", "west", "nw"]
PE_TEMPLATES = [f"PE:v1 | subject=red_orb | layout={L} | quality=hq" for L in LAYOUTS]
PE_PARAPHRASE = [
    "high quality red orb in the north",
    "hq red sphere, northeast layout",
    "red ball on the east side, high quality",
    "southeast red orb, nice quality",
    "southern red orb, hq",
    "southwest red sphere",
    "red orb to the west, high quality",
    "northwest red ball, hq",
]

INK, MUTED, ACCENT, PAPER, LINE = "#2a241c", "#6e6658", "#3d5a4c", "#f7f3ea", "#c9beab"


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def pe_embed(text: str, dim: int = PE_DIM) -> np.ndarray:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    rng = np.random.RandomState(int.from_bytes(digest[:4], "little"))
    v = np.zeros(dim, dtype=np.float32)
    for i, ch in enumerate(text.encode("utf-8")):
        v[i % dim] += ((ch / 255.0) * 2 - 1) * (1.0 + 0.05 * i)
    v += 0.2 * rng.randn(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def make_modes(n: int = N_MODES, radius: float = 2.2) -> np.ndarray:
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([radius * np.cos(ang), radius * np.sin(ang)], 1).astype(np.float32)


def build_dataset(centers, pe_for_mode, counts=None, sigma=0.11, seed=0):
    rng = np.random.RandomState(seed)
    counts = counts or [N_PER_MODE] * len(centers)
    xs, ys, pes = [], [], []
    for k, (c, n, pe_str) in enumerate(zip(centers, counts, pe_for_mode)):
        if n <= 0:
            continue
        x = c + sigma * rng.randn(n, 2).astype(np.float32)
        xs.append(x)
        ys.append(np.full(n, k, np.int64))
        pe = pe_embed(pe_str)
        pes.append(np.repeat(pe[None, :], n, 0))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(pes)


class CondDenoiser(nn.Module):
    def __init__(self, pe_dim=PE_DIM, hidden=HIDDEN):
        super().__init__()
        self.pe_mlp = nn.Sequential(nn.Linear(pe_dim, 64), nn.SiLU(), nn.Linear(64, 64))
        self.t_mlp = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, 32))
        self.net = nn.Sequential(
            nn.Linear(2 + 64 + 32, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x, t, pe):
        return self.net(torch.cat([x, self.pe_mlp(pe), self.t_mlp(t[:, None])], -1))


def diffusion_constants(device):
    betas = torch.linspace(1e-4, 0.02, T_DIFF, device=device)
    alphas = 1.0 - betas
    a_bar = torch.cumprod(alphas, 0)
    return betas, alphas, a_bar


def train_model(train_x, train_pe, device, steps=STEPS, seed=0):
    set_seed(seed)
    torch.cuda.set_device(device)
    dev = torch.device(f"cuda:{device}")
    ds = TensorDataset(torch.tensor(train_x), torch.tensor(train_pe))
    bs = min(BATCH, max(64, (len(train_x) // 4) * 4))
    if bs > len(train_x):
        bs = len(train_x)
    drop = len(train_x) >= bs * 2
    loader = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=drop)
    betas, alphas, a_bar = diffusion_constants(dev)
    model = CondDenoiser().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    it = iter(loader)
    model.train()
    last = 0.0
    for step in range(steps):
        try:
            x0, pe = next(it)
        except StopIteration:
            it = iter(loader)
            x0, pe = next(it)
        x0, pe = x0.to(dev), pe.to(dev)
        t = torch.randint(0, T_DIFF, (x0.size(0),), device=dev)
        pe_in = pe.clone()
        pe_in[torch.rand(x0.size(0), device=dev) < 0.15] = 0.0
        eps = torch.randn_like(x0)
        xt = a_bar[t].sqrt()[:, None] * x0 + (1 - a_bar[t]).sqrt()[:, None] * eps
        loss = ((model(xt, t.float() / T_DIFF, pe_in) - eps) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = float(loss.item())
        if step % 1500 == 0:
            print(f"  [cuda:{device}] step {step}/{steps} loss={last:.4f}", flush=True)
    return model, (betas, alphas, a_bar), last


@torch.no_grad()
def sample(model, consts, device, pe_vec=None, n=500, cfg=2.8):
    betas, alphas, a_bar = consts
    dev = torch.device(f"cuda:{device}")
    model.eval()
    x = torch.randn(n, 2, device=dev)
    if pe_vec is None:
        pe_t = torch.zeros(n, PE_DIM, device=dev)
        cfg = 1.0
    else:
        pe_t = torch.tensor(pe_vec, device=dev).float()[None].expand(n, -1).clone()
    pe_null = torch.zeros_like(pe_t)
    for ti in reversed(range(T_DIFF)):
        t = torch.full((n,), ti, device=dev, dtype=torch.long)
        tn = t.float() / T_DIFF
        eps = model(x, tn, pe_null) + cfg * (model(x, tn, pe_t) - model(x, tn, pe_null))
        a, ab, b = alphas[ti], a_bar[ti], betas[ti]
        x = (1 / a.sqrt()) * (x - (b / (1 - ab).sqrt()) * eps)
        if ti > 0:
            x = x + b.sqrt() * torch.randn_like(x)
    return x.cpu().numpy()


@torch.no_grad()
def denoise_once(model, consts, device, x0, t_idx, pe_vec=None):
    betas, alphas, a_bar = consts
    dev = torch.device(f"cuda:{device}")
    model.eval()
    x0_t = torch.tensor(x0, device=dev)
    n = x0_t.size(0)
    eps = torch.randn_like(x0_t)
    x = a_bar[t_idx].sqrt() * x0_t + (1 - a_bar[t_idx]).sqrt() * eps
    if pe_vec is None:
        pe_t = torch.zeros(n, PE_DIM, device=dev)
        cfg = 1.0
    else:
        pe_t = torch.tensor(pe_vec, device=dev).float()[None].expand(n, -1).clone()
        cfg = 2.0
    pe_null = torch.zeros_like(pe_t)
    for ti in reversed(range(t_idx + 1)):
        t = torch.full((n,), ti, device=dev, dtype=torch.long)
        tn = t.float() / T_DIFF
        e = model(x, tn, pe_null) + cfg * (model(x, tn, pe_t) - model(x, tn, pe_null))
        a, ab, b = alphas[ti], a_bar[ti], betas[ti]
        x = (1 / a.sqrt()) * (x - (b / (1 - ab).sqrt()) * e)
        if ti > 0:
            x = x + b.sqrt() * torch.randn_like(x)
    return x.cpu().numpy()


def assign_mode(samples, centers, thr=0.5):
    d = ((samples[:, None] - centers[None]) ** 2).sum(-1)
    idx = d.argmin(1)
    ok = np.sqrt(d.min(1)) < thr
    return idx, ok


def mode_hist(samples, centers, thr=0.5):
    idx, ok = assign_mode(samples, centers, thr)
    h = np.zeros(len(centers), dtype=np.float64)
    for k in idx[ok]:
        h[k] += 1
    if h.sum() > 0:
        h /= h.sum()
    return h


def target_mode_hit(samples, centers, k, thr=0.5):
    idx, ok = assign_mode(samples, centers, thr)
    return float(((idx == k) & ok).mean())


def mode_hit(samples, centers, thr=0.5):
    return float(assign_mode(samples, centers, thr)[1].mean())


def style_ax(ax, lim=3.4):
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(LINE)
    ax.set_facecolor(PAPER)


def plot_scatter_panels(panels, train_x, centers, path, hole=None):
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.55 * n, 3.55), dpi=160)
    if n == 1:
        axes = [axes]
    for ax, (title, pts, color) in zip(axes, panels):
        ax.scatter(train_x[:, 0], train_x[:, 1], s=3, c=LINE, alpha=0.55, linewidths=0)
        ax.scatter(pts[:, 0], pts[:, 1], s=8, c=color, alpha=0.6, linewidths=0)
        ax.scatter(centers[:, 0], centers[:, 1], s=30, c=INK, marker="x", zorder=3)
        if hole is not None:
            h = centers[hole]
            ax.scatter(
                [h[0]], [h[1]], s=140, facecolors="none", edgecolors="#a33",
                linewidths=1.6, zorder=4,
            )
            ax.text(h[0] * 1.22, h[1] * 1.22, "hole", color="#a33", fontsize=8, ha="center")
        style_ax(ax)
        ax.set_title(title, fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)


def plot_bars(labels, vals, path, title, ylabel, colors=None):
    fig, ax = plt.subplots(figsize=(5.8, 3.4), dpi=160)
    colors = colors or [ACCENT] * len(vals)
    ax.bar(labels, vals, color=colors, edgecolor=LINE)
    ax.set_ylim(0, max(1.05, max(vals) * 1.15 if vals else 1))
    ax.set_ylabel(ylabel, color=INK, fontsize=9)
    ax.set_title(title, fontsize=10, color=INK)
    ax.set_facecolor(PAPER)
    for s in ax.spines.values():
        s.set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)


def plot_grouped_bars(group_labels, series, path, title):
    fig, ax = plt.subplots(figsize=(6.4, 3.5), dpi=160)
    x = np.arange(len(group_labels))
    width = 0.35
    for i, (name, vals, color) in enumerate(series):
        ax.bar(x + (i - 0.5) * width, vals, width, label=name, color=color, edgecolor=LINE)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylabel("fraction of samples", color=INK, fontsize=9)
    ax.set_title(title, fontsize=10, color=INK)
    ax.set_facecolor(PAPER)
    for s in ax.spines.values():
        s.set_color(LINE)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)


def _run_shard(name: str, device: int) -> dict:
    """Worker: train one experiment on one GPU, return arrays + metrics."""
    torch.cuda.set_device(device)
    local = device
    centers = make_modes()
    pe_list = list(PE_TEMPLATES)
    out = {"name": name, "device": device}

    if name == "A":
        print(f"[A] PE lock on cuda:{device}", flush=True)
        train_x, _, train_pe = build_dataset(centers, pe_list, seed=0)
        model, consts, loss = train_model(train_x, train_pe, local, seed=0)
        exact0 = sample(model, consts, local, pe_embed(PE_TEMPLATES[0]), n=600)
        para0 = sample(model, consts, local, pe_embed(PE_PARAPHRASE[0]), n=600)
        typo0 = sample(
            model, consts, local,
            pe_embed(PE_TEMPLATES[0].replace("north", "nroth")), n=600,
        )
        uncond = sample(model, consts, local, None, n=600)
        gallery = {
            str(k): sample(model, consts, local, pe_embed(PE_TEMPLATES[k]), n=280)
            for k in (0, 2, 5)
        }
        # between-mode snap
        mid = np.stack(
            [0.5 * (centers[k] + centers[(k + 1) % N_MODES]) for k in range(N_MODES)], 0
        ).astype(np.float32)
        mid_pts = np.repeat(mid, 50, 0)
        mid_pts = mid_pts + 0.02 * np.random.randn(*mid_pts.shape).astype(np.float32)
        recovered = denoise_once(model, consts, local, mid_pts, t_idx=55, pe_vec=None)
        # PE average
        pe_n, pe_e, pe_ne = pe_embed(PE_TEMPLATES[0]), pe_embed(PE_TEMPLATES[2]), pe_embed(PE_TEMPLATES[1])
        pe_avg = (pe_n + pe_e)
        pe_avg = pe_avg / (np.linalg.norm(pe_avg) + 1e-8)
        s_avg = sample(model, consts, local, pe_avg, n=600)
        s_ne = sample(model, consts, local, pe_ne, n=600)

        out.update(
            train_x=train_x,
            exact0=exact0,
            para0=para0,
            typo0=typo0,
            uncond=uncond,
            gallery=gallery,
            mid_pts=mid_pts,
            recovered=recovered,
            s_avg=s_avg,
            s_ne=s_ne,
            loss=loss,
            metrics={
                "A_exact_target": target_mode_hit(exact0, centers, 0),
                "A_para_target": target_mode_hit(para0, centers, 0),
                "A_typo_target": target_mode_hit(typo0, centers, 0),
                "A_uncond_any": mode_hit(uncond, centers),
                "E_mean_dist_before": float(
                    np.sqrt(((mid_pts[:, None] - centers[None]) ** 2).sum(-1).min(1)).mean()
                ),
                "E_mean_dist_after": float(
                    np.sqrt(((recovered[:, None] - centers[None]) ** 2).sum(-1).min(1)).mean()
                ),
                "E_frac_near_mode_after": mode_hit(recovered, centers, thr=0.45),
                "F_exact_ne": target_mode_hit(s_ne, centers, 1),
                "F_avg_ne": target_mode_hit(s_avg, centers, 1),
                "F_avg_north": target_mode_hit(s_avg, centers, 0),
                "F_avg_east": target_mode_hit(s_avg, centers, 2),
                "A_final_loss": loss,
            },
        )

    elif name == "B":
        print(f"[B] hole on cuda:{device}", flush=True)
        hole = 1
        counts = [N_PER_MODE] * N_MODES
        counts[hole] = 0
        train_x, _, train_pe = build_dataset(centers, pe_list, counts, seed=1)
        model, consts, loss = train_model(train_x, train_pe, local, seed=1)
        ask_hole = sample(model, consts, local, pe_embed(PE_TEMPLATES[hole]), n=600)
        ask_neighbor = sample(model, consts, local, pe_embed(PE_TEMPLATES[0]), n=500)
        uncond = sample(model, consts, local, None, n=900)
        idx_h, ok_h = assign_mode(ask_hole, centers)
        idx_u, ok_u = assign_mode(uncond, centers)
        out.update(
            train_x=train_x,
            ask_hole=ask_hole,
            ask_neighbor=ask_neighbor,
            uncond=uncond,
            hole=hole,
            loss=loss,
            metrics={
                "B_hole_pe_lands_in_hole": float(((idx_h == hole) & ok_h).mean()),
                "B_hole_pe_lands_elsewhere": float(((idx_h != hole) & ok_h).mean()),
                "B_uncond_lands_in_hole": float(((idx_u == hole) & ok_u).mean()),
                "B_neighbor_ok": target_mode_hit(ask_neighbor, centers, 0),
                "B_final_loss": loss,
            },
        )

    elif name == "C":
        print(f"[C] mass on cuda:{device}", flush=True)
        counts = [N_PER_MODE] * N_MODES
        counts[0] = N_PER_MODE * 8
        train_x, _, train_pe = build_dataset(centers, pe_list, counts, seed=2)
        model, consts, loss = train_model(train_x, train_pe, local, seed=2)
        uncond = sample(model, consts, local, None, n=1500)
        emp = np.array(counts, dtype=np.float64)
        emp /= emp.sum()
        gen = mode_hist(uncond, centers)
        out.update(
            train_x=train_x,
            uncond=uncond,
            emp=emp,
            gen=gen,
            loss=loss,
            metrics={
                "C_train_north_frac": float(emp[0]),
                "C_gen_north_frac": float(gen[0]),
                "C_l1_hist": float(np.abs(emp - gen).sum()),
                "C_final_loss": loss,
            },
        )

    elif name == "D":
        print(f"[D] PE swap on cuda:{device}", flush=True)
        pe_swapped = list(PE_TEMPLATES)
        pe_swapped[0], pe_swapped[4] = pe_swapped[4], pe_swapped[0]
        train_x, _, train_pe = build_dataset(centers, pe_swapped, seed=3)
        model, consts, loss = train_model(train_x, train_pe, local, seed=3)
        ask_north = sample(model, consts, local, pe_embed(PE_TEMPLATES[0]), n=600)
        ask_south = sample(model, consts, local, pe_embed(PE_TEMPLATES[4]), n=600)
        out.update(
            train_x=train_x,
            ask_north=ask_north,
            ask_south=ask_south,
            loss=loss,
            metrics={
                "D_north_str_to_south": target_mode_hit(ask_north, centers, 4),
                "D_north_str_to_north": target_mode_hit(ask_north, centers, 0),
                "D_south_str_to_north": target_mode_hit(ask_south, centers, 0),
                "D_south_str_to_south": target_mode_hit(ask_south, centers, 4),
                "D_final_loss": loss,
            },
        )
    else:
        raise ValueError(name)

    # drop model from return (not picklable easily with cuda) — only numpy
    return {k: v for k, v in out.items() if k != "model"}


def write_figures(results: dict) -> dict:
    centers = make_modes()
    metrics = {}
    A, B, C, D = results["A"], results["B"], results["C"], results["D"]
    metrics.update(A["metrics"])
    metrics.update(B["metrics"])
    metrics.update(C["metrics"])
    metrics.update(D["metrics"])

    plot_scatter_panels(
        [("training data (8 PE keys)", A["train_x"], ACCENT)],
        A["train_x"], centers, FIG / "01_training_data.png",
    )
    plot_scatter_panels(
        [
            ("exact PE (north)", A["exact0"], ACCENT),
            ("paraphrase", A["para0"], "#8a8172"),
            ("one typo in PE", A["typo0"], "#8a8172"),
        ],
        A["train_x"], centers, FIG / "02_pe_exact_vs_near_miss.png",
    )
    plot_bars(
        ["exact PE", "paraphrase", "typo", "uncond\n(any mode)"],
        [
            metrics["A_exact_target"],
            metrics["A_para_target"],
            metrics["A_typo_target"],
            metrics["A_uncond_any"],
        ],
        FIG / "03_pe_hit_rates.png",
        "Hit rate on intended north mode (uncond = any mode)",
        "fraction of samples",
        colors=[ACCENT, "#8a8172", "#8a8172", LINE],
    )
    plot_scatter_panels(
        [
            (f"exact PE → {LAYOUTS[k]}", A["gallery"][str(k)], ACCENT)
            for k in (0, 2, 5)
        ],
        A["train_x"], centers, FIG / "04_exact_pe_gallery.png",
    )
    plot_scatter_panels(
        [
            ("train (NE removed)", B["train_x"], ACCENT),
            ("sample with missing PE", B["ask_hole"], "#a33"),
            ("unconditional", B["uncond"], "#8a8172"),
        ],
        B["train_x"], centers, FIG / "05_hole_missing_mode.png", hole=B["hole"],
    )
    plot_grouped_bars(
        LAYOUTS,
        [
            ("train mass", C["emp"], LINE),
            ("uncond samples", C["gen"], ACCENT),
        ],
        FIG / "06_mass_follows_counts.png",
        "Unconditional samples follow empirical mass (north ×8)",
    )
    plot_scatter_panels(
        [
            ("train (north×8)", C["train_x"], ACCENT),
            ("uncond samples", C["uncond"], "#5a7a6a"),
        ],
        C["train_x"], centers, FIG / "07_mass_scatter.png",
    )
    plot_scatter_panels(
        [
            ('ask PE "...layout=north"', D["ask_north"], ACCENT),
            ('ask PE "...layout=south"', D["ask_south"], "#6a4a3a"),
        ],
        D["train_x"], centers, FIG / "08_pe_swap.png",
    )
    plot_bars(
        [
            'PE"north"→south\n(co-trained)',
            'PE"north"→north\n(word meaning)',
            'PE"south"→north\n(co-trained)',
            'PE"south"→south\n(word meaning)',
        ],
        [
            metrics["D_north_str_to_south"],
            metrics["D_north_str_to_north"],
            metrics["D_south_str_to_north"],
            metrics["D_south_str_to_south"],
        ],
        FIG / "09_pe_swap_bars.png",
        "After swapping PE tags: follows co-occurrence, not the word",
        "fraction of samples",
        colors=[ACCENT, "#8a8172", ACCENT, "#8a8172"],
    )
    plot_scatter_panels(
        [
            ("start: midpoints between modes", A["mid_pts"], "#a33"),
            ("after noise→denoise", A["recovered"], ACCENT),
        ],
        A["train_x"], centers, FIG / "10_between_modes_snap.png",
    )
    plot_scatter_panels(
        [
            ("exact PE northeast", A["s_ne"], ACCENT),
            ("mean(PE north, PE east)", A["s_avg"], "#a33"),
        ],
        A["train_x"], centers, FIG / "11_pe_average.png",
    )
    return metrics


def main():
    assert torch.cuda.is_available(), "Need CUDA (GPUs 0–3)"
    n = torch.cuda.device_count()
    devices = [d for d in DEVICES if d < n]
    assert len(devices) >= 4, f"Need 4 GPUs, visible={n}"
    print(f"Parallel train A/B/C/D on GPUs {devices}", flush=True)

    jobs = [("A", devices[0]), ("B", devices[1]), ("C", devices[2]), ("D", devices[3])]
    results = {}
    # spawn so each process gets a clean CUDA context with remapped device
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_run_shard, name, dev): name for name, dev in jobs}
        for fut in as_completed(futs):
            name = futs[fut]
            results[name] = fut.result()
            print(f"finished {name}: {results[name]['metrics']}", flush=True)

    metrics = write_figures(results)
    metrics["devices"] = devices
    metrics["steps"] = STEPS
    metrics["n_per_mode"] = N_PER_MODE
    (FIG / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print("wrote figures →", FIG)


if __name__ == "__main__":
    # Required for CUDA + forkserver/spawn cleanliness
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
