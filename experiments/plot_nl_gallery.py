#!/usr/bin/env python3
"""Picture grids for the schema↔language suite (figures 42 and 43).

Reads one picked sample per (regime, probe) from figures/cache_nl_gallery.npz and the
hit rates from figures/metrics_nl_vs_pe.json. Both are written by toy_nl_vs_pe.py.

    python3 experiments/plot_nl_gallery.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import (  # noqa: E402
    ACCENT_DEEP,
    ALERT_DEEP,
    FAINT,
    INK,
    MUTED,
    PAPER,
    apply_style,
    weight,
)

FIG = Path(__file__).resolve().parents[1] / "figures"
CACHE = FIG / "cache_nl_gallery.npz"
SCALE = 8


def load():
    raw = np.load(CACHE)
    gallery = {k.replace("__", "|"): raw[k] for k in raw.files}
    metrics = json.loads((FIG / "metrics_nl_vs_pe.json").read_text())
    hits = {
        regime: {p["probe"]: p["hit"] for p in body["probes"]}
        for regime, body in metrics.items()
    }
    return gallery, hits


def hit_style(v):
    if v >= 0.5:
        return ACCENT_DEEP, "semibold"
    if v > 0.02:
        return ALERT_DEEP, "semibold"
    return FAINT, "medium"


def grid(path, gallery, hits, rows, cols):
    """rows: [(regime, label)], cols: [(probe or 'train', header)]."""
    apply_style()
    n_r, n_c = len(rows), len(cols)
    cell = 1.32
    fig = plt.figure(figsize=(1.55 + cell * n_c, 0.55 + (cell + 0.42) * n_r), dpi=200)
    fig.patch.set_facecolor(PAPER)
    left = 1.55 / fig.get_figwidth()
    gs = fig.add_gridspec(n_r, n_c, left=left, right=0.995, top=1 - 0.5 / fig.get_figheight(),
                          bottom=0.03, wspace=0.06, hspace=0.3)
    for i, (regime, row_label) in enumerate(rows):
        for j, (probe, header) in enumerate(cols):
            ax = fig.add_subplot(gs[i, j])
            img = gallery[f"{regime}|{probe}"]
            ax.imshow(np.repeat(np.repeat(img, SCALE, 0), SCALE, 1), interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(header, fontsize=10.5, color=INK, pad=6, fontproperties=weight("semibold"),
                             linespacing=1.15)
            if probe == "train":
                ax.set_xlabel("training image", fontsize=10, color=MUTED, labelpad=4,
                              fontproperties=weight("regular"))
            else:
                v = hits[regime].get(probe, 0.0)
                color, w = hit_style(v)
                ax.set_xlabel(f"hit {v:.2f}", fontsize=10.5, color=color, labelpad=4,
                              fontproperties=weight(w))
            if j == 0:
                ax.text(-0.12, 0.5, row_label, transform=ax.transAxes, ha="right", va="center",
                        fontsize=10.5, color=INK, fontproperties=weight("semibold"), linespacing=1.25)
    fig.savefig(path, bbox_inches="tight", facecolor=PAPER, pad_inches=0.06)
    plt.close(fig)
    print("wrote", path, flush=True)


def main():
    gallery, hits = load()
    grid(
        FIG / "42_opaque_vs_semantic.png", gallery, hits,
        rows=[("pe_schema", "trained on\ncolor=red"), ("pe_opaque", "trained on\ncolor=c1")],
        cols=[("train", "train"), ("D0_exact", "exact key"), ("D3_opaque", "opaque code"),
              ("D4_nl_template", "English\n“red”"), ("D6_nl_zh", "Chinese")],
    )
    grid(
        FIG / "43_nl_requires_cooccurrence.png", gallery, hits,
        rows=[("pe_schema", "schema\nonly"), ("pe_nl_paired", "schema +\nEnglish")],
        cols=[("train", "train"), ("D0_exact", "exact key"), ("D4_nl_template", "English\ntemplate"),
              ("D5_nl_free", "free\nEnglish")],
    )


if __name__ == "__main__":
    main()
