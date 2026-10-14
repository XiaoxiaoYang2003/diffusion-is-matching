"""Shared matplotlib style, aligned with the Cambrian (NYU VISIONx) pages.

Inter, ink #141413, gray #6b7280, dashed light rules, left/bottom spines only,
pastel bars with a thin dark outline, bold "(a)" panel titles.
Palette from the Cambrian cards: slate blue, terracotta, sand, sage, stone, cream.
Ref: https://cambrian-mllm.github.io/
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = ROOT / "assets" / "fonts"

PAPER = "#ffffff"
PANEL = "#f6f5f1"
INK = "#141413"
MUTED = "#6b7280"
FAINT = "#9ca3af"
LINE = "#141413"           # spines are dark and thin, as on the Cambrian charts
GRID = "#e5e5e5"

ACCENT = "#6a9bcc"         # slate blue: hit / trained
ACCENT_DEEP = "#3b6a99"
ALERT = "#cc785c"          # terracotta: miss / partial
ALERT_DEEP = "#9c4a2f"
SAND = "#d4a27f"
SAGE = "#bcd1ca"
SAGE_DEEP = "#5f8f80"
STONE = "#d1cfc5"
CREAM = "#ebdbbc"
SOFT = STONE
GREEN = SAGE_DEEP
LIME = CREAM
LIME_DEEP = "#8a6a3a"
AQUA = SAGE_DEEP

EDGE = INK
EDGE_W = 0.8

BAR_W = 0.62
BAR_H = 0.55
BAR_PAIR = 0.32
BAR_PAIR_OFF = 0.17

FONT_NAME = "DejaVu Sans"
_APPLIED = False
_WEIGHTS = {
    "regular": "Inter-Regular.ttf",
    "medium": "Inter-Medium.ttf",
    "semibold": "Inter-SemiBold.ttf",
    "bold": "Inter-Bold.ttf",
}


def apply_style() -> str:
    global _APPLIED, FONT_NAME
    if not _APPLIED:
        for ttf in sorted(FONT_DIR.glob("Inter-*.ttf")):
            fm.fontManager.addfont(str(ttf))
        reg = FONT_DIR / _WEIGHTS["regular"]
        if reg.exists():
            FONT_NAME = fm.FontProperties(fname=str(reg)).get_name()
        _APPLIED = True
    plt.rcParams.update(
        {
            "font.family": FONT_NAME,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.labelcolor": INK,
            "axes.edgecolor": LINE,
            "axes.linewidth": 0.8,
            "axes.facecolor": PAPER,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.color": INK,
            "ytick.color": INK,
            "legend.fontsize": 10,
            "text.color": INK,
            "figure.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "savefig.dpi": 200,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "custom",
            "mathtext.rm": FONT_NAME,
            "mathtext.it": f"{FONT_NAME}:italic",
        }
    )
    return FONT_NAME


def weight(name: str = "regular"):
    apply_style()
    path = FONT_DIR / _WEIGHTS.get(name, _WEIGHTS["regular"])
    if path.exists():
        return fm.FontProperties(fname=str(path))
    return fm.FontProperties(family=FONT_NAME)


def style_axes(ax, *, grid="y", spines=True):
    """Left and bottom spines, dashed light rules, small outward ticks."""
    ax.set_facecolor(PAPER)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(bool(spines))
        ax.spines[s].set_color(LINE)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(length=3, width=0.7, colors=INK, labelsize=10, pad=3)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    rule = dict(color=GRID, lw=0.8, ls=(0, (4, 3)))
    if grid == "x":
        ax.xaxis.grid(True, **rule)
    elif grid in ("y", True):
        ax.yaxis.grid(True, **rule)
    ax.set_axisbelow(True)


def panel_title(ax, letter: str | None, title: str, *, loc="center", size=12, pad=8):
    """Cambrian-style title: bold "(a)" followed by a regular title."""
    text = f"({letter}) {title}" if letter else title
    ax.set_title(text, fontsize=size, color=INK, fontproperties=weight("semibold"), loc=loc, pad=pad)


def hit_color(v: float, thr: float = 0.5) -> str:
    if v >= thr:
        return ACCENT
    if v > 0.02:
        return ALERT
    return STONE


def deep(color: str) -> str:
    return {ACCENT: ACCENT_DEEP, ALERT: ALERT_DEEP, STONE: MUTED, SAGE: SAGE_DEEP}.get(color, INK)
