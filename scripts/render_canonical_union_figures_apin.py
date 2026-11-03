"""Render the canonical union figures used by main.tex."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src" / "geosot_generative_trajectory").is_dir():
            return candidate
    raise SystemExit("run from inside the repository (src/geosot_generative_trajectory not found)")


ROOT = _repo_root(Path(__file__).resolve().parent)

KBS_CANONICAL = Path(os.environ.get("APIN_CANONICAL", Path(__file__).resolve().parent.parent / "artifacts" / "results"))
FIGURES = Path(os.environ.get("APIN_FIGURES", Path(__file__).resolve().parent.parent / "artifacts" / "figures"))


def main() -> None:
    stats = pd.read_csv(KBS_CANONICAL / "paired_raw_grid_geosot_statistics.csv")
    d = stats[(stats["encoding_primary_eligible"].astype(bool)) & (stats["scope"] == "event")].copy()
    wanted = ["occupancy_f1", "occupancy_iou", "active_cell_iou", "brier_score", "nll", "ece"]
    d = d[d["metric"].isin(wanted)]
    d["label"] = d.apply(lambda r: f"{r.dataset.replace('_', ' ').title()}\n{r.source_fraction:.1f}", axis=1)
    order = d[["dataset", "source_fraction", "label"]].drop_duplicates().sort_values(["dataset", "source_fraction"])
    labels = order["label"].tolist()

    def agg(metric: str) -> pd.DataFrame:
        x = d[d.metric == metric].groupby(["dataset", "source_fraction", "label"], as_index=False).agg(
            geo=("geosot_mean", "mean"), raw=("raw_grid_mean", "mean"), gain=("geosot_advantage", "mean"),
            low=("bootstrap_ci_low", "min"), high=("bootstrap_ci_high", "max"),
        )
        return order.merge(x, on=["dataset", "source_fraction", "label"], how="left")

    palette = {"geo": "#477a9e", "raw": "#a9c2d1", "gain": "#315d78"}
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "figure.dpi": 160, "pdf.fonttype": 42})
    x = np.arange(len(labels))
    width = 0.36
    fig, axes = plt.subplots(2, 2, figsize=(5.3, 3.3), constrained_layout=True)
    for ax, metric, title, ylabel in [
        (axes[0, 0], "occupancy_f1", "(a) Event-conditioned F1", "F1"),
        (axes[1, 0], "occupancy_iou", "(c) Event-conditioned IoU", "IoU"),
    ]:
        a = agg(metric)
        ax.bar(x - width / 2, a.raw, width, color=palette["raw"], label="RAW-grid")
        ax.bar(x + width / 2, a.geo, width, color=palette["geo"], label="GeoSOT-ST")
        ax.set_xticks(x, labels, rotation=0)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", color="#d9e2e8", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, ncol=2, loc="upper left")

    for ax, metric, title, ylabel in [
        (axes[0, 1], "occupancy_f1", "(b) F1 gain", "GeoSOT-ST minus RAW-grid"),
        (axes[1, 1], "occupancy_iou", "(d) IoU gain", "GeoSOT-ST minus RAW-grid"),
    ]:
        a = agg(metric)
        y = a.gain.to_numpy()
        lo = y - a.low.to_numpy()
        hi = a.high.to_numpy() - y
        ax.errorbar(x, y, yerr=np.vstack([lo, hi]), fmt="o", color=palette["gain"],
                    ecolor="#7d9bad", elinewidth=1.2, capsize=3, markersize=5)
        ax.axhline(0, color="#647780", linewidth=0.8)
        ax.set_xticks(x, labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", color="#d9e2e8", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIGURES / "canonical_union_primary_summary.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "canonical_union_primary_summary.png", bbox_inches="tight")
    plt.close(fig)

    rows = []
    for metric in wanted:
        a = agg(metric)
        for row in a.itertuples(index=False):
            signed = row.gain
            rows.append((metric, row.label, signed))
    matrix = pd.DataFrame(rows, columns=["metric", "label", "gain"]).pivot(index="metric", columns="label", values="gain")
    fig, ax = plt.subplots(figsize=(5.3, 2.2), constrained_layout=True)
    im = ax.imshow(matrix.to_numpy(), cmap="PuBuGn", aspect="auto")
    ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns)
    ax.set_yticks(np.arange(len(matrix.index)), [m.replace("_", " ").title() for m in matrix.index])
    ax.tick_params(axis="x", labelrotation=0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            ax.text(j, i, f"{value:+.3f}", ha="center", va="center", fontsize=5.8,
                    color="white" if abs(value) > np.nanmax(np.abs(matrix.to_numpy())) * 0.45 else "#173b4d")
    ax.set_title("Canonical GeoSOT-ST gain over RAW-grid", loc="left", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Signed gain; positive is better")
    fig.savefig(FIGURES / "canonical_union_gain_heatmap.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "canonical_union_gain_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print("rendered canonical union figures")


if __name__ == "__main__":
    main()
