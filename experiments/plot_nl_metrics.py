#!/usr/bin/env python3
"""Summary charts from metrics_nl_vs_pe.json — Thinking Machines visual language."""

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
    ACCENT,
    ALERT,
    BAR_H,
    BAR_W,
    FAINT,
    GRID,
    INK,
    LINE,
    MUTED,
    PAPER,
    PANEL,
    SOFT,
    apply_style,
    hit_color,
    style_axes,
    weight,
)

FIG = Path(__file__).resolve().parents[1] / "figures"
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
PROBE_LABELS = {
    "D0_exact": "D0 exact",
    "D1_reorder": "D1 reorder",
    "D2_typo": "D2 typo",
    "D3_opaque": "D3 opaque",
    "D4_nl_template": "D4 NL template",
    "D5_nl_free": "D5 free EN",
    "D6_nl_zh": "D6 ZH",
    "D7_wrong": "D7 wrong",
}


def hits(by, regime):
    return {p["probe"]: p["hit"] for p in by[regime]["probes"]}


def thin_barh(ax, labels, vals, colors, title="", *, caption=None):
    """Horizontal bars. Long values sit inside the bar; zeros sit clear of the spine."""
    style_axes(ax, grid="x")
    n = len(labels)
    y = np.arange(n)[::-1]
    ax.set_ylim(-0.7, n - 0.3)
    for yi, v, col in zip(y, vals, colors):
        v = float(v)
        if v <= 0.02:
            ax.text(
                0.06, yi, f"{v:.2f}",
                va="center", ha="left", fontsize=9, color=MUTED,
                fontproperties=weight("medium"), zorder=3,
            )
            continue
        ax.barh(yi, v, height=0.42, color=col, edgecolor="none", zorder=2)
        if v >= 0.55:
            ax.text(
                v - 0.035, yi, f"{v:.2f}",
                va="center", ha="right", fontsize=9, color="#ffffff",
                fontproperties=weight("medium"), zorder=3,
            )
        else:
            ax.text(
                v + 0.04, yi, f"{v:.2f}",
                va="center", ha="left", fontsize=9, color=col,
                fontproperties=weight("medium"), zorder=3,
            )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10, color=MUTED, fontproperties=weight("regular"))
    ax.tick_params(axis="y", pad=6, length=0)
    ax.set_xlim(0, 1.05)
    ax.set_xticks([0, 0.5, 1.0])
    ax.tick_params(axis="x", labelsize=9)
    if title:
        ax.set_title(title, fontsize=11, color=INK, fontproperties=weight("medium"), loc="left", pad=8)
    ax.set_xlabel(caption or "hit", fontsize=9.5, color=MUTED, labelpad=8)


def main():
    apply_style()
    by = json.loads((FIG / "metrics_nl_vs_pe.json").read_text())
    hm = hits(by, "pe_schema")

    # ---- Fig 40: TM thin horizontal ladder ----
    fig, ax = plt.subplots(figsize=(6.8, 3.5), dpi=150)
    fig.patch.set_facecolor(PAPER)
    thin_barh(
        ax,
        [PROBE_LABELS[p] for p in PROBE_ORDER],
        [hm.get(p, 0) for p in PROBE_ORDER],
        [hit_color(hm.get(p, 0)) for p in PROBE_ORDER],
        "",
    )
    fig.subplots_adjust(bottom=0.14, left=0.36, right=0.97, top=0.88)
    fig.savefig(FIG / "40_nl_pe_axis.png", bbox_inches="tight", facecolor=PAPER, dpi=130, pad_inches=0.18)
    plt.close(fig)

    # ---- Fig 41: heatmap, soft greens ----
    regimes = [r for r in ["pe_schema", "pe_opaque", "pe_nl_paired", "nl_only", "para_battery"] if r in by]
    mat = np.zeros((len(regimes), len(PROBE_ORDER)), np.float32)
    for i, r in enumerate(regimes):
        h = hits(by, r)
        for j, p in enumerate(PROBE_ORDER):
            mat[i, j] = h.get(p, 0.0)
    short = {
        "pe_schema": "schema",
        "pe_opaque": "opaque",
        "pe_nl_paired": "paired",
        "nl_only": "NL only",
        "para_battery": "paraphrase",
    }
    from matplotlib.colors import LinearSegmentedColormap

    fig, ax = plt.subplots(figsize=(7.4, 3.6), dpi=170)
    fig.patch.set_facecolor(PAPER)
    cmap = LinearSegmentedColormap.from_list("cambrian", ["#ffffff", "#dbe6f2", ACCENT, "#3b6a99"])
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    cols = ["exact", "reorder", "typo", "opaque", "English\ntemplate", "free\nEnglish", "Chinese", "wrong\ncolor"]
    ax.set_xticks(range(len(PROBE_ORDER)))
    ax.set_xticklabels([f"D{i}\n{c}" for i, c in enumerate(cols)], fontsize=9.5, color=INK,
                       fontproperties=weight("regular"), linespacing=1.15)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(regimes)))
    ax.set_yticklabels([short.get(r, r) for r in regimes], fontsize=11, color=INK,
                       fontproperties=weight("medium"))
    ax.tick_params(axis="both", pad=5, length=0)
    ax.set_xticks(np.arange(-0.5, len(PROBE_ORDER)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(regimes)), minor=True)
    ax.grid(which="minor", color="#ffffff", lw=2.0)
    ax.tick_params(which="minor", length=0)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(
                j, i, f"{v:.2f}", ha="center", va="center",
                fontsize=10, color="#ffffff" if v > 0.55 else (INK if v > 0.02 else FAINT),
                fontproperties=weight("semibold" if v > 0.02 else "regular"),
            )
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_ylabel("training setup", fontsize=11, color=MUTED, labelpad=8)
    fig.subplots_adjust(left=0.2, right=0.99, bottom=0.03, top=0.8)
    fig.savefig(FIG / "41_distance_heatmap.png", bbox_inches="tight", facecolor=PAPER, dpi=140, pad_inches=0.16)
    plt.close(fig)

    # ---- Fig 44: thin horizontal ladder ----
    fig, ax = plt.subplots(figsize=(8.4, 5.0), dpi=220)
    fig.patch.set_facecolor(PAPER)
    thin_barh(
        ax,
        [PROBE_LABELS[p] for p in PROBE_ORDER],
        [hm.get(p, 0) for p in PROBE_ORDER],
        [hit_color(hm.get(p, 0)) for p in PROBE_ORDER],
        "Schema-only: hit collapses toward natural language",
    )
    fig.savefig(FIG / "44_schema_ladder_bars.png", bbox_inches="tight", facecolor=PAPER, dpi=220)
    plt.close(fig)

    # ---- Fig 47 ----
    if "remap" in by:
        hm_r = hits(by, "remap")
        fig, ax = plt.subplots(figsize=(7.6, 3.4), dpi=220)
        fig.patch.set_facecolor(PAPER)
        labs = ["opaque c1 → red (train)", "NL “red” (untrained)", "human remap → blue"]
        vals_r = [hm_r.get("D0_exact", 0), hm_r.get("D4_nl_template", 0), hm_r.get("D7b_human_remap", 0)]
        thin_barh(ax, labs, vals_r, [ACCENT, ALERT, ALERT], "Code table: model keeps the train drawer")
        fig.savefig(FIG / "47_remap.png", bbox_inches="tight", facecolor=PAPER, dpi=220)
        plt.close(fig)

    # Paired small multiples, captions under the axes, as on the MiniT2I scaling figure.
    fig = plt.figure(figsize=(7.2, 4.7), dpi=150)
    fig.patch.set_facecolor(PAPER)
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1.2, 1.05],
        wspace=0.62, hspace=0.55, left=0.10, right=0.98, top=0.96, bottom=0.08,
    )
    axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]

    thin_barh(
        axes[0],
        ["exact", "typo", "template", "free"],
        [hm.get(p, 0) for p in ["D0_exact", "D2_typo", "D4_nl_template", "D5_nl_free"]],
        [hit_color(hm.get(p, 0)) for p in ["D0_exact", "D2_typo", "D4_nl_template", "D5_nl_free"]],
        caption="(a)  Wording",
    )

    a = hits(by, "pe_schema").get("D4_nl_template", 0)
    b = hits(by, "pe_nl_paired").get("D4_nl_template", 0)
    thin_barh(
        axes[1],
        ["schema", "paired"],
        [a, b], [ALERT, ACCENT],
        caption="(b)  Co-occurrence",
    )

    ho = hits(by, "pe_opaque")
    thin_barh(
        axes[2],
        ["code", "opaque", "“red”"],
        [ho.get("D0_exact", 0), ho.get("D3_opaque", 0), ho.get("D4_nl_template", 0)],
        [ACCENT, ACCENT, ALERT],
        caption="(c)  Codes",
    )

    h = by["holdout_combo"].get("holdout") or {}
    thin_barh(
        axes[3],
        ["held-out", "nearest"],
        [h.get("held_hit", 0), h.get("near_train_hit", 0)],
        [ALERT, ACCENT],
        caption="(d)  Composition",
    )

    fig.savefig(FIG / "48_matching_checklist.png", bbox_inches="tight", facecolor=PAPER, dpi=140, pad_inches=0.16)
    plt.close(fig)
    print("updated 40, 41, 44, 47, 48 (TM style)")


if __name__ == "__main__":
    main()
