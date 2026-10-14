#!/usr/bin/env python3
"""Compose compact 2D summary boards from existing experiment PNGs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
PAPER, INK, MUTED, LINE = "#f7f3ea", "#2a241c", "#6e6658", "#c9beab"


def load(name: str):
    p = FIG / name
    if not p.exists():
        raise FileNotFoundError(p)
    return mpimg.imread(p)


def panel(ax, img, title: str):
    ax.imshow(img)
    ax.set_title(title, fontsize=8.5, color=INK, pad=3, fontweight="500")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(LINE)
        s.set_linewidth(0.8)
    ax.set_facecolor(PAPER)


def board(cells: list[tuple[str, str]], path: Path, ncols: int, title: str, figsize):
    n = len(cells)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=170)
    axes = np.atleast_2d(axes)
    if axes.shape[0] == 1 and nrows > 1:
        pass
    fig.patch.set_facecolor(PAPER)
    for i, (fname, label) in enumerate(cells):
        r, c = divmod(i, ncols)
        panel(axes[r, c], load(fname), label)
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis("off")
        axes[r, c].set_facecolor(PAPER)
    fig.suptitle(title, fontsize=10.5, color=INK, y=0.995, fontweight="600")
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=0.35, w_pad=0.35)
    fig.savefig(path, bbox_inches="tight", facecolor=PAPER, dpi=170)
    plt.close(fig)
    print("wrote", path)


def main():
    board(
        [
            ("01_training_data.png", "train mass + PE keys"),
            ("02_pe_exact_vs_near_miss.png", "exact vs paraphrase vs typo"),
            ("03_pe_hit_rates.png", "hit rates"),
            ("04_exact_pe_gallery.png", "exact PE gallery"),
        ],
        FIG / "20_2d_pe_board.png",
        ncols=2,
        title="2D conditional diffusion: PE key matching",
        figsize=(7.6, 6.4),
    )
    board(
        [
            ("05_hole_missing_mode.png", "delete a mode"),
            ("06_mass_follows_counts.png", "duplicate mass"),
            ("08_pe_swap.png", "swap PE tags"),
            ("09_pe_swap_bars.png", "co-train vs word meaning"),
        ],
        FIG / "21_2d_ablation_board.png",
        ncols=2,
        title="Ablations: hole, mass, and label swap",
        figsize=(7.6, 6.4),
    )
    board(
        [
            ("10_between_modes_snap.png", "midpoints snap to islands"),
            ("11_pe_average.png", "PE average is not composition"),
        ],
        FIG / "22_2d_geometry_board.png",
        ncols=2,
        title="Geometry: no free continuum, no free PE arithmetic",
        figsize=(7.6, 3.35),
    )


if __name__ == "__main__":
    main()
