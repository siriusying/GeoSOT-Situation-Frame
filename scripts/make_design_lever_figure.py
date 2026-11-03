# -*- coding: utf-8 -*-
"""Regenerate the design-lever range figure (Figure `fig:design_lever`).

The ranges are read from the released backbone statistics so the figure cannot
drift from the matched backbone table.  Prediction-object values are the
matched-origin F1 means from the prediction-object audit (also quoted in the
prediction-object table).

    python scripts/make_design_lever_figure.py
    # override the artifact root:
    APIN_ROOT=/path/to/release python scripts/make_design_lever_figure.py
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(os.environ.get("APIN_ROOT", Path(__file__).resolve().parents[1]))
RESULTS = ROOT / "artifacts" / "results"
FIGURES = ROOT / "artifacts" / "figures"

# Matched-origin prediction-object F1 (Table `tab:prediction_object`).
PREDICTION_OBJECT = {
    "porto": (0.0473, 0.7828),
    "t_drive": (0.6296, 0.9309),
}
BACKBONES = ("convgru_6ch", "convlstm_6ch", "predrnnpp", "stgcn", "graph_wavenet")

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#d9d9d9",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
    }
)
DARK = "#1f4e79"
MID = "#6fa8dc"
LIGHT = "#b8b8b8"


def ranges(summary: pd.DataFrame) -> dict[str, list[tuple[str, float, float]]]:
    data: dict[str, list[tuple[str, float, float]]] = {}
    for city in ("porto", "t_drive"):
        rows = summary[summary.city == city].set_index("tag")
        missing = [tag for tag in BACKBONES if tag not in rows.index]
        if missing:
            raise SystemExit(f"{city}: missing backbone rows {missing}")
        backbone = rows.loc[list(BACKBONES), "f1"]
        channel = rows.loc[["convgru_2ch", "convgru_6ch"], "f1"]
        data[city] = [
            ("Prediction object", *PREDICTION_OBJECT[city]),
            ("Temporal backbone", float(backbone.min()), float(backbone.max())),
            ("Input channels", float(channel.min()), float(channel.max())),
        ]
    return data


def draw(ax, data, title):
    names = [d[0] for d in data][::-1]
    colors = [
        DARK if n == "Prediction object" else (MID if n == "Temporal backbone" else LIGHT)
        for n in names
    ]
    for yi, (name, l, h), c in zip(range(len(names)), reversed(data), colors):
        ax.plot([l, h], [yi, yi], color=c, lw=8, solid_capstyle="butt", zorder=2)
        ax.plot([l, h], [yi, yi], color=c, lw=1.0, zorder=3)
        for v in (l, h):
            ax.scatter([v], [yi], s=18, color=c, zorder=4)
        ax.text(
            (l + h) / 2,
            yi + 0.22,
            f"{l:.4f}--{h:.4f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#222222",
        )
    ax.set_yticks(list(range(len(names))))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_ylim(-0.5, 2.75)
    ax.set_xlim(0, 1)
    ax.set_title(title, fontsize=9, loc="left", pad=4)


def main() -> None:
    summary = pd.read_csv(RESULTS / "backbone_summary_v2.csv")
    data = ranges(summary)
    FIGURES.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(6.0, 3.1), sharex=True)
    draw(axes[0], data["porto"], "(a) Porto")
    draw(axes[1], data["t_drive"], "(b) T-Drive")
    axes[1].set_xlabel("Event-conditioned F1", fontsize=8)
    axes[1].set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    fig.tight_layout(pad=0.5)

    fig.savefig(FIGURES / "design_lever_range.pdf")
    fig.savefig(FIGURES / "design_lever_range.png", dpi=220)
    plt.close(fig)
    print(f"wrote {FIGURES / 'design_lever_range.pdf'}")


if __name__ == "__main__":
    main()
