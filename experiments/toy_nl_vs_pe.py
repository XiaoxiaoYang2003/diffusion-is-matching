#!/usr/bin/env python3
"""Schema ↔ NL under a real conditional diffusion loss (matching drawers).

Upgrades vs earlier key-decoder toy
-----------------------------------
* Continuous PE embeddings (hash features) — unseen strings are NOT a reserved
  null id; failure must come from the matched measure.
* Tiny conditional denoising model on 32×32 (x0-prediction + CFG), not a discrete key table.
* Mode hit via a small supervised classifier on clean renders.
* Hold-out compositions report intended vs nearest-train vs off-support.
* One summary board tying Support / Co-occurrence / Wording / Interp.

Probe ladder (schema → natural language)
----------------------------------------
  D0 exact train key · D1 reorder · D2 typo · D3 opaque code
  D4 NL template · D5 free EN · D6 ZH · D7 fluent wrong

Workers (1×8)
-------------
  0 pe_schema · 1 pe_opaque · 2 pe_nl_paired · 3 nl_only
  4 holdout_combo · 5 collision · 6 remap · 7 para_battery

Cluster:
    bash /pfs/weiyang/diffusion-is-matching/cluster_submit/submit_nl_vs_pe.sh
Smoke:
    TOY_SMOKE=1 python3 experiments/toy_nl_vs_pe.py --smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import sys

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
    ALERT,
    INK,
    LINE,
    MUTED,
    PAPER,
    apply_style,
    weight,
)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

IMG = 32
PE_DIM = 64
T_DIFF = 80
SHAPES = ["circle", "square", "triangle"]
COLOR_RGB = {
    "red": (0.90, 0.20, 0.18),
    "green": (0.18, 0.68, 0.36),
    "blue": (0.20, 0.45, 0.90),
}
COLORS = list(COLOR_RGB)
LAYOUTS = ["left", "center", "right"]
CODE_COLOR = {"c1": "red", "c2": "green", "c3": "blue"}
CODE_SHAPE = {"s1": "circle", "s2": "square", "s3": "triangle"}
CODE_LAYOUT = {"p1": "left", "p2": "center", "p3": "right"}
REV_COLOR = {v: k for k, v in CODE_COLOR.items()}
REV_SHAPE = {v: k for k, v in CODE_SHAPE.items()}
REV_LAYOUT = {v: k for k, v in CODE_LAYOUT.items()}

N_PER = 120
STEPS = 6000
BATCH = 192
LR = 2e-3
CFG = 2.8
DISPLAY_SCALE = 8
N_SAMP = 32


def set_seed(s=0):
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def pe_schema(shape, color, layout):
    return (
        f"[PE] subject={shape} | color={color} | "
        f"layout={layout} | quality=hq | style=flat"
    )


def pe_opaque(shape, color, layout):
    return (
        f"[PE] subject={REV_SHAPE[shape]} | color={REV_COLOR[color]} | "
        f"layout={REV_LAYOUT[layout]} | quality=hq | style=flat"
    )


def pe_reordered(shape, color, layout):
    return f"[PE] color={color}|subject={shape}|layout={layout}|quality=hq|style=flat"


def pe_typo(shape, color, layout):
    return pe_schema(shape, color, layout).replace(f"layout={layout}", f"layout={layout[:-1]}x")


def nl_template(shape, color, layout):
    return f"a {color} {shape} on the {layout}, high quality, flat style"


def nl_free(shape, color, layout):
    return f"please show {shape} in {color} sitting toward the {layout} side, still camera"


def nl_zh(shape, color, layout):
    zh_c = {"red": "红色", "green": "绿色", "blue": "蓝色"}[color]
    zh_s = {"circle": "圆形", "square": "方形", "triangle": "三角形"}[shape]
    zh_l = {"left": "左侧", "center": "中间", "right": "右侧"}[layout]
    return f"{zh_l}的{zh_c}{zh_s}，高质量，扁平风格"


def nl_wrong_color(shape, color, layout):
    """Fluent ask that is NOT another train key (avoid 'blue' template collision)."""
    # synonym / nonce color word — never used as a training key
    bogus = {"red": "crimson", "green": "emerald", "blue": "azure"}[color]
    return f"a {bogus} {shape} on the {layout}, high quality, flat style"


def all_combos():
    return [(s, c, L) for s in SHAPES for c in COLORS for L in LAYOUTS]


def pe_embed(text: str, dim: int = PE_DIM) -> np.ndarray:
    """Continuous condition features from the raw string (no id table)."""
    digest = hashlib.sha256(text.encode()).digest()
    rng = np.random.RandomState(int.from_bytes(digest[:4], "little"))
    v = np.zeros(dim, np.float32)
    for i, b in enumerate(text.encode("utf-8", errors="ignore")):
        v[i % dim] += ((b / 255.0) * 2 - 1) * (1 + 0.03 * i)
    v += 0.12 * rng.randn(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def render_frame(shape, color, layout, jitter=0.012, rng=None) -> np.ndarray:
    rng = rng or np.random.RandomState()
    img = np.full((IMG, IMG, 3), 0.07, np.float32)
    cx = {"left": 0.28, "center": 0.50, "right": 0.72}[layout] + jitter * rng.randn()
    cy = 0.50 + 0.5 * jitter * rng.randn()
    scale = 0.22
    rgb = np.array(COLOR_RGB[color], np.float32)
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


class TinyCondDenoiser(nn.Module):
    """x0-predictor: matches E[x|c] under a conditional denoising loss.

    Continuous PE embedding (no discrete key table). At high noise the model
    must rely on c — the same matching contract as ε-DDPM, easier to fit in-toy.
    """

    def __init__(self, pe_dim=PE_DIM, base=64):
        super().__init__()
        self.pe = nn.Sequential(nn.Linear(pe_dim, 128), nn.SiLU(), nn.Linear(128, 128))
        self.t = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64))
        self.cond = nn.Linear(128 + 64, base * 2)
        self.enc = nn.Sequential(
            nn.Conv2d(3, base, 3, padding=1),
            nn.GroupNorm(8, base),
            nn.SiLU(),
            nn.Conv2d(base, base, 3, stride=2, padding=1),
            nn.GroupNorm(8, base),
            nn.SiLU(),
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, base * 2),
            nn.SiLU(),
        )
        self.mid = nn.Sequential(
            nn.Conv2d(base * 2, base * 2, 3, padding=1),
            nn.GroupNorm(8, base * 2),
            nn.SiLU(),
            nn.Conv2d(base * 2, base * 2, 3, padding=1),
            nn.GroupNorm(8, base * 2),
            nn.SiLU(),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1),
            nn.GroupNorm(8, base),
            nn.SiLU(),
            nn.ConvTranspose2d(base, base, 4, stride=2, padding=1),
            nn.GroupNorm(8, base),
            nn.SiLU(),
            nn.Conv2d(base, 3, 3, padding=1),
            nn.Tanh(),
        )
        # also a direct pe→image head (helps exact-key retrieval)
        self.direct = nn.Sequential(
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 3 * IMG * IMG),
            nn.Tanh(),
        )

    def forward(self, x, t, pe):
        pe_h = self.pe(pe)
        cond = self.cond(torch.cat([pe_h, self.t(t[:, None])], -1))[:, :, None, None]
        h = self.enc(x) + cond
        den = self.dec(self.mid(h))
        direct = self.direct(pe_h).view(-1, 3, IMG, IMG)
        # blend: at high noise (t~1) trust direct/pe; at low noise trust denoise path
        w = t.view(-1, 1, 1, 1).clamp(0, 1)
        return (1 - 0.65 * (1 - w)) * den + (0.65 * w) * direct


def diffs(device, t_diff):
    betas = torch.linspace(1e-4, 0.02, t_diff, device=device)
    alphas = 1 - betas
    a_bar = torch.cumprod(alphas, 0)
    return betas, alphas, a_bar


def train_ddpm(model, loader, steps, t_diff, device):
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    betas, alphas, a_bar = diffs(device, t_diff)
    it = iter(loader)
    model.train()
    last = 0.0
    for step in range(steps):
        try:
            x0, pe = next(it)
        except StopIteration:
            it = iter(loader)
            x0, pe = next(it)
        x0, pe = x0.to(device), pe.to(device)
        t = torch.randint(0, t_diff, (x0.size(0),), device=device)
        pe_in = pe.clone()
        drop = torch.rand(x0.size(0), device=device) < 0.12
        pe_in[drop] = 0
        eps = torch.randn_like(x0)
        xt = a_bar[t].sqrt()[:, None, None, None] * x0 + (1 - a_bar[t]).sqrt()[:, None, None, None] * eps
        # targets: clean x0; dropped cond → push toward noise (empty drawer)
        target = x0.clone()
        target[drop] = torch.randn_like(target[drop]) * 0.4
        pred = model(xt, t.float() / max(t_diff - 1, 1), pe_in)
        loss = ((pred - target) ** 2).mean()
        # auxiliary: pure pe→image at t=1
        pe_h = model.pe(pe)
        direct = model.direct(pe_h).view_as(x0)
        loss = loss + 0.5 * ((direct - x0) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = float(loss.item())
        if step % 1000 == 0:
            print(f"  [{device}] step {step}/{steps} loss={last:.4f}", flush=True)
    return model, (betas, alphas, a_bar), last


@torch.no_grad()
def sample(model, consts, pe_vec, n, cfg, t_diff, device):
    """Ancestral sampling with x0-prediction + CFG on condition."""
    betas, alphas, a_bar = consts
    model.eval()
    x = torch.randn(n, 3, IMG, IMG, device=device)
    if pe_vec is None:
        pe = torch.zeros(n, PE_DIM, device=device)
        cfg = 1.0
    else:
        pe = torch.tensor(pe_vec, device=device).float()[None].expand(n, -1).clone()
    pe0 = torch.zeros_like(pe)
    for ti in reversed(range(t_diff)):
        t = torch.full((n,), ti, device=device, dtype=torch.long)
        tn = t.float() / max(t_diff - 1, 1)
        x0_u = model(x, tn, pe0)
        x0_c = model(x, tn, pe)
        x0_hat = x0_u + cfg * (x0_c - x0_u)
        x0_hat = x0_hat.clamp(-1, 1)
        ab = a_bar[ti]
        # reconstruct eps from x0_hat, then DDPM step
        eps = (x - ab.sqrt() * x0_hat) / (1 - ab).sqrt().clamp(min=1e-4)
        a, b = alphas[ti], betas[ti]
        x = (1 / a.sqrt()) * (x - (b / (1 - ab).sqrt()) * eps)
        if ti > 0:
            x = x + b.sqrt() * torch.randn_like(x)
    return x.clamp(-1, 1).cpu().numpy()


class ModeClassifier(nn.Module):
    def __init__(self, n_modes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, n_modes),
        )

    def forward(self, x):
        return self.net(x)


def train_mode_classifier(combos, device, steps=1500):
    """Supervised oracle: image → (shape,color,layout) index."""
    rng = np.random.RandomState(1)
    xs, ys = [], []
    for i, (s, c, L) in enumerate(combos):
        for _ in range(40):
            xs.append(render_frame(s, c, L, rng=rng))
            ys.append(i)
    x = torch.tensor(np.stack(xs).transpose(0, 3, 1, 2).astype(np.float32))
    y = torch.tensor(np.asarray(ys, np.int64))
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=128, shuffle=True, drop_last=True)
    model = ModeClassifier(len(combos)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    it = iter(loader)
    model.train()
    for step in range(steps):
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(loader)
            xb, yb = next(it)
        xb, yb = xb.to(device), yb.to(device)
        loss = F.cross_entropy(model(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model


@torch.no_grad()
def predict_modes(clf, gens, device):
    """gens: N,3,H,W in [-1,1] or N,H,W,3"""
    x = np.asarray(gens, np.float32)
    if x.ndim == 4 and x.shape[-1] == 3:
        x = x.transpose(0, 3, 1, 2)
    t = torch.tensor(x, device=device)
    return clf(t).argmax(-1).cpu().numpy()


def _canonical(combo):
    s, c, L = combo
    return to_uint8(render_frame(s, c, L, jitter=0.0, rng=np.random.RandomState(0)))


def _mse(img, ref):
    return float(((img.astype(np.float32) - ref.astype(np.float32)) ** 2).mean())


def pick_aligned(gens, pred, conf_each, target_idx, combos):
    """Gallery frame.

    When most samples hit, show the one that sits on the canonical frame.
    When they miss, show a clean wrong shape, not a broken blob or a rare lucky hit.
    """
    del conf_each  # rates stay in the metrics; the frame is chosen for shape
    u = to_uint8(gens)
    pred = np.asarray(pred)
    hits = np.where(pred == target_idx)[0]
    if len(hits) >= 0.5 * len(pred):
        ref = _canonical(combos[int(target_idx)])
        err = [_mse(u[i], ref) for i in hits]
        return u[hits[int(np.argmin(err))]]
    misses = np.where(pred != target_idx)[0]
    pool = misses if len(misses) else np.arange(len(u))
    cache = {}
    best_i, best = int(pool[0]), 1e9
    for i in pool:
        p = int(pred[i])
        if p not in cache:
            cache[p] = _canonical(combos[p])
        err = _mse(u[i], cache[p])
        if err < best:
            best, best_i = err, int(i)
    return u[best_i]


def to_uint8(x):
    x = np.asarray(x, np.float32)
    x = np.clip((x + 1) * 0.5, 0, 1)
    if x.ndim == 4 and x.shape[1] in (1, 3):
        x = x.transpose(0, 2, 3, 1)
    elif x.ndim == 3 and x.shape[0] in (1, 3) and x.shape[-1] not in (1, 3):
        x = x.transpose(1, 2, 0)
    return (x * 255).astype(np.uint8)


def as_hwc_u8(img) -> np.ndarray:
    x = np.asarray(img)
    if x.ndim == 3 and x.shape[0] in (1, 3) and x.shape[-1] not in (1, 3):
        x = np.transpose(x, (1, 2, 0))
    if x.ndim == 2:
        x = np.stack([x, x, x], -1)
    return x.astype(np.uint8, copy=False)


def upscale(img, s=DISPLAY_SCALE):
    img = as_hwc_u8(img)
    return np.repeat(np.repeat(img, s, 0), s, 1)


def build_entries(regime: str, holdout=None):
    holdout = holdout or set()
    entries = []
    for s, c, L in all_combos():
        if (s, c, L) in holdout:
            continue
        mode = f"{s}|{c}|{L}"
        if regime == "pe_schema":
            entries.append((s, c, L, pe_schema(s, c, L), mode))
        elif regime == "pe_opaque":
            entries.append((s, c, L, pe_opaque(s, c, L), mode))
        elif regime == "pe_nl_paired":
            entries.append((s, c, L, pe_schema(s, c, L), mode))
            entries.append((s, c, L, nl_template(s, c, L), mode))
        elif regime == "nl_only":
            entries.append((s, c, L, nl_template(s, c, L), mode))
        elif regime == "holdout_combo":
            entries.append((s, c, L, pe_schema(s, c, L), mode))
        elif regime == "collision":
            if s == "square" and c == "red":
                entries.append((s, "blue", L, pe_schema(s, "red", L), f"{s}|blue|{L}|via_red_key"))
            else:
                entries.append((s, c, L, pe_schema(s, c, L), mode))
        elif regime == "remap":
            entries.append((s, c, L, pe_opaque(s, c, L), mode))
        elif regime == "para_battery":
            entries.append((s, c, L, pe_schema(s, c, L), mode))
        else:
            raise ValueError(regime)
    return entries


def probe_battery(regime: str, shape, color, layout):
    probes = []
    if regime in ("pe_opaque", "remap"):
        d0 = pe_opaque(shape, color, layout)
    elif regime == "nl_only":
        d0 = nl_template(shape, color, layout)
    else:
        d0 = pe_schema(shape, color, layout)
    probes.append(("D0_exact", d0, shape, color, layout))
    probes.append(("D1_reorder", pe_reordered(shape, color, layout), shape, color, layout))
    probes.append(("D2_typo", pe_typo(shape, color, layout), shape, color, layout))
    probes.append(("D3_opaque", pe_opaque(shape, color, layout), shape, color, layout))
    probes.append(("D4_nl_template", nl_template(shape, color, layout), shape, color, layout))
    probes.append(("D5_nl_free", nl_free(shape, color, layout), shape, color, layout))
    probes.append(("D6_nl_zh", nl_zh(shape, color, layout), shape, color, layout))
    # Nonce color word: should not open the intended drawer (or any trained color word).
    probes.append(("D7_wrong", nl_wrong_color(shape, color, layout), shape, color, layout))
    if regime == "remap":
        probes.append(("D7b_human_remap", pe_opaque(shape, "red", layout), shape, "blue", layout))
    if regime == "para_battery":
        for i, text in enumerate(
            [
                f"{color} {shape} {layout}",
                f"high-quality {shape}, {color}, placed {layout}",
                f"flat illustration of {color} {shape} to the {layout}",
            ]
        ):
            probes.append((f"D5b_{i}", text, shape, color, layout))
    return probes


def combo_index(combos, s, c, L):
    return combos.index((s, c, L))


def nearest_train_modes(hold_s, hold_c, hold_L, train_set):
    """Factor-neighbors that were trained (same two factors, flip one)."""
    cands = []
    for s, c, L in train_set:
        dist = int(s != hold_s) + int(c != hold_c) + int(L != hold_L)
        if dist == 1:
            cands.append((s, c, L))
    return cands


def run_worker(job: dict):
    name = job["name"]
    gpu = job["gpu"]
    steps = job["steps"]
    n_per = job["n_per"]
    t_diff = job["t_diff"]
    n_samp = job["n_samp"]
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    set_seed(job["seed"])
    print(f"[{name}] start on {device}", flush=True)

    combos = all_combos()
    holdout = set()
    if name == "holdout_combo":
        holdout = {("triangle", "blue", "center")}

    entries = build_entries(name, holdout=holdout)
    train_visuals = {(s, c, L) for s, c, L, _, _ in entries}

    rng = np.random.RandomState(job["seed"])
    imgs, pes = [], []
    for s, c, L, key, _ in entries:
        pe = pe_embed(key)
        for _ in range(n_per):
            imgs.append(render_frame(s, c, L, rng=rng))
            pes.append(pe)
    x = np.stack(imgs).transpose(0, 3, 1, 2).astype(np.float32)
    pe_arr = np.stack(pes).astype(np.float32)
    ds = TensorDataset(torch.tensor(x), torch.tensor(pe_arr))
    bs = min(BATCH, max(32, (len(x) // 2) * 2))
    loader = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)

    model = TinyCondDenoiser().to(device)
    model, consts, _ = train_ddpm(model, loader, steps=steps, t_diff=t_diff, device=device)

    clf_steps = 400 if job.get("smoke") else 1500
    clf = train_mode_classifier(combos, device, steps=clf_steps)

    hero = ("circle", "red", "left")
    if name == "collision":
        probes = [
            ("D0_circle_red_key", pe_schema("circle", "red", "left"), "circle", "red", "left"),
            ("D0_square_red_key", pe_schema("square", "red", "left"), "square", "blue", "left"),
            ("D4_nl_red_circle", nl_template("circle", "red", "left"), "circle", "red", "left"),
            ("D4_nl_red_square", nl_template("square", "red", "left"), "square", "red", "left"),
        ] + probe_battery("pe_schema", *hero)[1:3]
    else:
        probes = probe_battery(name, *hero)

    train_keys = {key for *_, key, _ in entries}
    probe_rows = []
    gallery = {}
    for pname, text, s, c, L in probes:
        gens = sample(model, consts, pe_embed(text), n=n_samp, cfg=CFG, t_diff=t_diff, device=device)
        with torch.no_grad():
            logits = clf(torch.tensor(gens, device=device))
            prob = torch.softmax(logits, -1)
            conf_each = prob.max(-1).values.cpu().numpy()
        pred = logits.argmax(-1).cpu().numpy()
        target = combo_index(combos, s, c, L)
        hit = float((pred == target).mean())
        conf = float(conf_each.mean())
        off = float((conf_each < 0.35).mean())
        probe_rows.append(
            {
                "probe": pname,
                "text": text,
                "intended": f"{s}|{c}|{L}",
                "in_train": text in train_keys,
                "hit": hit,
                "conf": conf,
                "off_support": off,
            }
        )
        gallery[pname] = pick_aligned(gens, pred, conf_each, target, combos)

    holdout_metrics = None
    if name == "holdout_combo":
        hs, hc, hL = "triangle", "blue", "center"
        text = pe_schema(hs, hc, hL)
        gens = sample(model, consts, pe_embed(text), n=max(n_samp, 32), cfg=CFG, t_diff=t_diff, device=device)
        with torch.no_grad():
            logits = clf(torch.tensor(gens, device=device))
            conf_each = torch.softmax(logits, -1).max(-1).values.cpu().numpy()
        pred = logits.argmax(-1).cpu().numpy()
        target = combo_index(combos, hs, hc, hL)
        held_hit = float((pred == target).mean())
        neighbors = nearest_train_modes(hs, hc, hL, train_visuals)
        near_ids = {combo_index(combos, *n) for n in neighbors}
        near_hit = float(np.mean([p in near_ids for p in pred])) if near_ids else 0.0
        off = float((conf_each < 0.35).mean())
        holdout_metrics = {
            "held_hit": held_hit,
            "near_train_hit": near_hit,
            "off_support": off,
            "neighbors": [f"{a}|{b}|{c}" for a, b, c in neighbors],
            "in_train": text in train_keys,
        }
        gallery["holdout"] = pick_aligned(gens, pred, conf_each, target, combos)
        if neighbors:
            ns, nc, nL = neighbors[0]
            g2 = sample(
                model, consts, pe_embed(pe_schema(ns, nc, nL)),
                n=max(n_samp, 8), cfg=CFG, t_diff=t_diff, device=device,
            )
            with torch.no_grad():
                logits2 = clf(torch.tensor(g2, device=device))
                conf2 = torch.softmax(logits2, -1).max(-1).values.cpu().numpy()
            pred2 = logits2.argmax(-1).cpu().numpy()
            gallery["holdout_neighbor_exact"] = pick_aligned(
                g2, pred2, conf2, combo_index(combos, ns, nc, nL), combos,
            )

    train_img = to_uint8(render_frame(*hero, jitter=0.0, rng=np.random.RandomState(0)))
    out = {
        "name": name,
        "n_keys": len({k for *_, k, _ in entries}),
        "probes": probe_rows,
        "holdout": holdout_metrics,
        "gallery": {k: v.tolist() for k, v in gallery.items()},
        "train_hero": train_img.tolist(),
        "hero": list(hero),
    }
    print(
        f"[{name}] done keys={out['n_keys']} "
        f"D0={probe_rows[0]['hit']:.2f} holdout={holdout_metrics}",
        flush=True,
    )
    return out


PROBE_ORDER = [
    "D0_exact",
    "D1_reorder",
    "D2_typo",
    "D3_opaque",
    "D4_nl_template",
    "D5_nl_free",
    "D6_nl_zh",
    "D7_wrong",
]


def _hit_map(result):
    return {p["probe"]: p["hit"] for p in result["probes"]}


def plot_all(results):
    """Gallery strips (42/43/45/46). Bar charts → plot_nl_metrics.py."""
    apply_style()
    by = {r["name"]: r for r in results}

    def _strip(path, row_specs, col_keys, col_names, title):
        fig, axes = plt.subplots(
            len(row_specs), len(col_keys),
            figsize=(12.2, 2.45 * len(row_specs) + 0.6), dpi=220,
        )
        fig.patch.set_facecolor(PAPER)
        if len(row_specs) == 1:
            axes = np.array([axes])
        for ri, (rname, rtitle) in enumerate(row_specs):
            r = by[rname]
            gal = {k: np.array(v, np.uint8) for k, v in r["gallery"].items()}
            train = np.array(r["train_hero"], np.uint8)
            for ci, gkey in enumerate(col_keys):
                ax = axes[ri, ci]
                img = train if gkey is None else gal.get(gkey, train)
                ax.imshow(upscale(img), interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_facecolor(PAPER)
                for sp in ax.spines.values():
                    sp.set_color(LINE)
                if ri == 0:
                    ax.set_title(
                        col_names[ci], fontsize=12, color=INK,
                        fontproperties=weight("semibold"), pad=6,
                    )
                if ci == 0:
                    ax.set_ylabel(
                        rtitle, fontsize=12, color=INK, rotation=0, labelpad=96, va="center",
                        fontproperties=weight("semibold"),
                    )
        fig.suptitle(title, fontsize=14, color=INK, fontproperties=weight("medium"), y=0.98)
        fig.subplots_adjust(left=0.26, right=0.98, top=0.86, bottom=0.04, wspace=0.08)
        fig.savefig(path, bbox_inches="tight", facecolor=PAPER, dpi=220, pad_inches=0.12)
        plt.close(fig)

    # Figures 42 and 43 are drawn by plot_nl_gallery.py from this cache, after the
    # metrics JSON is written in main().
    cache = {}
    for r in results:
        cache[f"{r['name']}__train"] = np.array(r["train_hero"], np.uint8)
        for key, img in r["gallery"].items():
            cache[f"{r['name']}__{key}"] = np.array(img, np.uint8)
    np.savez_compressed(FIG / "cache_nl_gallery.npz", **cache)

    if "collision" in by:
        r = by["collision"]
        gal = {k: np.array(v, np.uint8) for k, v in r["gallery"].items()}
        fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.2), dpi=220)
        fig.patch.set_facecolor(PAPER)
        panels = [
            ("circle + key color=red\n→ red mass", gal.get("D0_circle_red_key")),
            ("square + key color=red\n→ blue mass", gal.get("D0_square_red_key")),
            ("NL “red circle”", gal.get("D4_nl_red_circle")),
            ("NL “red square”", gal.get("D4_nl_red_square")),
        ]
        for ax, (lab, img) in zip(axes, panels):
            if img is None:
                ax.axis("off")
                continue
            ax.imshow(upscale(img), interpolation="nearest")
            ax.set_title(lab, fontsize=12, color=INK, fontproperties=weight("semibold"))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor(PAPER)
        fig.suptitle(
            "Collision: “red” follows co-occurrence, not a color concept",
            fontsize=15, color=INK, fontproperties=weight("bold"),
        )
        fig.tight_layout()
        fig.savefig(FIG / "45_collision.png", bbox_inches="tight", facecolor=PAPER, dpi=220, pad_inches=0.14)
        plt.close(fig)

    if "holdout_combo" in by and by["holdout_combo"].get("holdout"):
        h = by["holdout_combo"]["holdout"]
        r = by["holdout_combo"]
        gal = {k: np.array(v, np.uint8) for k, v in r["gallery"].items()}
        fig = plt.figure(figsize=(11.2, 3.6), dpi=220)
        fig.patch.set_facecolor(PAPER)
        gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 1.3], wspace=0.28)
        for i, (lab, key) in enumerate(
            [("held-out key\n(untrained combo)", "holdout"),
             ("exact key of a\ntrained neighbor", "holdout_neighbor_exact")]
        ):
            ax = fig.add_subplot(gs[0, i])
            img = gal.get(key)
            if img is not None:
                ax.imshow(upscale(img), interpolation="nearest")
            ax.set_title(lab, fontsize=12, color=INK, fontproperties=weight("semibold"))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor(PAPER)
        ax = fig.add_subplot(gs[0, 2])
        vals = [h["held_hit"], h["near_train_hit"], h.get("off_support", 0)]
        ax.bar([0, 1, 2], vals, color=[ALERT, ACCENT, MUTED], edgecolor="none", width=0.40)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(
            ["intended\nheld-out", "nearest\ntrain factor", "off\nsupport"],
            fontsize=11.5, color=INK,
        )
        ax.set_ylim(0, 1.18)
        ax.set_xlim(-0.55, 2.55)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.04, f"{v:.2f}", ha="center", fontsize=12.5, color=INK,
                    fontproperties=weight("bold"))
        ax.set_title("rates", fontsize=13, color=INK, fontproperties=weight("bold"))
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_facecolor(PAPER)
        fig.suptitle(
            "Hold-out composition: no assembly; mass does not invent the missing mode",
            fontsize=15, color=INK, fontproperties=weight("bold"),
        )
        fig.savefig(FIG / "46_holdout_combo.png", bbox_inches="tight", facecolor=PAPER, dpi=220)
        plt.close(fig)

    print("wrote strip / collision / holdout figures", flush=True)
    # Bars 40/41/44/47/48 are owned by plot_nl_metrics.py (keeps full 1×8 numbers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--gpus", type=str, default="")
    args = ap.parse_args()
    if os.environ.get("TOY_SMOKE", "0") == "1":
        args.smoke = True

    if args.smoke:
        steps, n_per, t_diff, n_samp = 1200, 32, 40, 16
    else:
        steps, n_per, t_diff, n_samp = STEPS, N_PER, T_DIFF, N_SAMP

    if args.gpus:
        gpus = [int(x) for x in args.gpus.split(",")]
    else:
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        gpus = list(range(min(8, max(n, 1))))

    names = [
        "pe_schema", "pe_opaque", "pe_nl_paired", "nl_only",
        "holdout_combo", "collision", "remap", "para_battery",
    ]
    print(f"nl_vs_pe DDPM on gpus={gpus} smoke={args.smoke} steps={steps} T={t_diff}", flush=True)
    jobs = [
        {
            "name": name,
            "gpu": gpus[i % len(gpus)],
            "seed": i,
            "steps": steps,
            "n_per": n_per,
            "t_diff": t_diff,
            "n_samp": n_samp,
            "smoke": args.smoke,
        }
        for i, name in enumerate(names)
    ]

    results = []
    if len({j["gpu"] for j in jobs}) == len(jobs) and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
            futs = {ex.submit(run_worker, j): j["name"] for j in jobs}
            for fut in as_completed(futs):
                results.append(fut.result())
    else:
        for j in jobs:
            results.append(run_worker(j))

    results.sort(key=lambda r: names.index(r["name"]))
    plot_all(results)

    metrics = {
        r["name"]: {"n_keys": r["n_keys"], "probes": r["probes"], "holdout": r["holdout"]}
        for r in results
    }
    (FIG / "metrics_nl_vs_pe.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    summary = {
        k: {p["probe"]: round(p["hit"], 3) for p in v["probes"][:8]}
        for k, v in metrics.items()
    }
    print(json.dumps(summary, indent=2))

    import plot_nl_gallery

    plot_nl_gallery.main()
    print("nl_vs_pe suite complete", flush=True)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
