#!/usr/bin/env python3
"""Budgeted regional dispatch case study for the GeoSOT-ST paper.

This experiment re-uses the registered aggregation artifacts, so it requires
no model training.  For every future window we rank grid cells by a predicted
occupancy score and allocate a budget of K dispatch units to the top-K cells.
We then measure how much of the true regional activity is covered.

Branches
--------
frame_gru       GeoSOT-ST frame predictions from the GRU aggregation artifact
entity_gru      RAW-GRU entity forecasts, projected and aggregated
frame_lstm      GeoSOT-ST frame predictions from the LSTM artifact
entity_lstm     RAW-LSTM entity forecasts, projected and aggregated
oracle          top-K by the true future occupancy (upper bound, not a method)
random          analytic expected coverage for a random K-cell allocation

Outputs are written to ``papers/APIN/experiments/results``.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

def _repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src" / "geosot_generative_trajectory").is_dir():
            return candidate
    raise SystemExit("run from inside the repository (src/geosot_generative_trajectory not found)")


ROOT = _repo_root(Path(__file__).resolve().parent)
AGG = (
    ROOT
    / "papers/paper_geosot_generative_trajectory_kbs/elsarticle/audit_artifacts"
    / "representation_benchmark_v1/paradigm_confirmatory_v1/full_v1/aggregation"
)
OUT = Path(os.environ.get("APIN_OUT", Path(__file__).resolve().parent.parent / "artifacts" / "results"))
OUT.mkdir(parents=True, exist_ok=True)

CITIES = ("porto", "t_drive")
SEEDS = list(range(41, 51))
KS = (5, 10, 20, 50, 100, 200, 400, 800)
SCORE_MODES = ("mean", "max")
N_BOOT = 10_000
RNG = np.random.default_rng(20261031)
GRID_CELLS = 32 * 32

BRANCH_LABELS = {
    "frame_gru": "GeoSOT-ST frame (GRU)",
    "entity_gru": "Aggregated entity (GRU)",
    "frame_lstm": "GeoSOT-ST frame (LSTM)",
    "entity_lstm": "Aggregated entity (LSTM)",
    "oracle": "Oracle top-K",
    "random": "Random allocation",
}
COLORS = {
    "frame_gru": "#1f77b4",
    "entity_gru": "#d62728",
    "frame_lstm": "#17becf",
    "entity_lstm": "#ff7f0e",
    "oracle": "#2ca02c",
    "random": "#7f7f7f",
}


def load_artifact(city: str, branch: str, seed: int) -> dict[str, np.ndarray]:
    path = AGG / f"{city}_{branch}" / f"{city}_paradigm_seed{seed}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {
            "targets": data["targets"].astype(np.float32),
            "frame": data["frame_predictions"].astype(np.float32),
            "aggregate": data["aggregate_predictions"].astype(np.float32),
            "starts": data["sample_starts"].astype(np.int64),
        }


def score(pred: np.ndarray, mode: str) -> np.ndarray:
    """Aggregate four future steps into one cell score."""
    pred = np.nan_to_num(pred[:, :, 0], nan=0.0, posinf=1.0, neginf=0.0)
    if mode == "mean":
        return pred.mean(axis=1)
    if mode == "max":
        return pred.max(axis=1)
    raise ValueError(mode)


def topk_mask(cell_score: np.ndarray, k: int) -> np.ndarray:
    flat = cell_score.reshape(len(cell_score), -1)
    k = max(1, min(k, flat.shape[1]))
    index = np.argpartition(-flat, kth=k - 1, axis=1)[:, :k]
    mask = np.zeros_like(flat, dtype=bool)
    np.put_along_axis(mask, index, True, axis=1)
    return mask.reshape(cell_score.shape)


def coverage_metrics(cell_score: np.ndarray, truth: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = topk_mask(cell_score, k)
    intersection = np.logical_and(selected, truth).sum(axis=(1, 2)).astype(float)
    true_count = truth.sum(axis=(1, 2)).astype(float)
    coverage = intersection / np.maximum(1.0, true_count)
    precision = intersection / float(k)
    unmet = true_count - intersection
    return coverage, precision, unmet


def holm(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p_values[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted.tolist()


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    idx = RNG.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def build_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    truth_meta: list[pd.DataFrame] = []
    for city in CITIES:
        for seed in SEEDS:
            gru = load_artifact(city, "gru", seed)
            lstm = load_artifact(city, "lstm", seed)
            if not np.array_equal(gru["starts"], lstm["starts"]):
                raise RuntimeError(f"start mismatch for {city} seed {seed}")
            if not np.array_equal(gru["targets"], lstm["targets"]):
                raise RuntimeError(f"target mismatch for {city} seed {seed}")

            truth = gru["targets"][:, :, 0].max(axis=1) > 0.5
            true_count = truth.sum(axis=(1, 2)).astype(float)
            if seed == SEEDS[0]:
                truth_meta.append(
                    pd.DataFrame(
                        {
                            "city": city,
                            "sample_start": gru["starts"],
                            "true_active_cells": true_count,
                        }
                    )
                )

            for mode in SCORE_MODES:
                scores = {
                    "frame_gru": score(gru["frame"], mode),
                    "entity_gru": score(gru["aggregate"], mode),
                    "frame_lstm": score(lstm["frame"], mode),
                    "entity_lstm": score(lstm["aggregate"], mode),
                    "oracle": truth.astype(np.float32),
                }
                for k in KS:
                    for branch, cell_score in scores.items():
                        coverage, precision, unmet = coverage_metrics(cell_score, truth, k)
                        rows.append(
                            pd.DataFrame(
                                {
                                    "city": city,
                                    "seed": seed,
                                    "sample_start": gru["starts"],
                                    "true_active_cells": true_count,
                                    "score_mode": mode,
                                    "k": k,
                                    "branch": branch,
                                    "coverage": coverage,
                                    "precision": precision,
                                    "unmet": unmet,
                                }
                            )
                        )
                    # Analytic random allocation baseline.
                    expected_intersection = k / GRID_CELLS * true_count
                    rows.append(
                        pd.DataFrame(
                            {
                                "city": city,
                                "seed": seed,
                                "sample_start": gru["starts"],
                                "true_active_cells": true_count,
                                "score_mode": mode,
                                "k": k,
                                "branch": "random",
                                "coverage": np.full_like(true_count, k / GRID_CELLS),
                                "precision": true_count / GRID_CELLS,
                                "unmet": true_count - expected_intersection,
                            }
                        )
                    )
    return pd.concat(rows, ignore_index=True), pd.concat(truth_meta, ignore_index=True)


def summarize_raw(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Mean over seeds for each window, so windows (not seed-window pairs) are
    # the independent unit in the bootstrap.
    window = (
        raw.groupby(["city", "score_mode", "k", "branch", "sample_start"], as_index=False)
        .agg(coverage=("coverage", "mean"), precision=("precision", "mean"), unmet=("unmet", "mean"), true_active=("true_active_cells", "first"))
    )

    summary_rows = []
    paired_rows = []
    stability_rows = []
    for (city, mode), group in window.groupby(["city", "score_mode"], sort=False):
        for k in KS:
            k_group = group[group["k"] == k]
            for branch in ("frame_gru", "entity_gru", "frame_lstm", "entity_lstm", "oracle", "random"):
                values = k_group[k_group["branch"] == branch]
                if values.empty:
                    continue
                lo, hi = bootstrap_ci(values["coverage"].to_numpy())
                summary_rows.append(
                    {
                        "city": city,
                        "score_mode": mode,
                        "k": k,
                        "branch": branch,
                        "branch_label": BRANCH_LABELS[branch],
                        "mean_coverage": float(values["coverage"].mean()),
                        "coverage_ci_low": lo,
                        "coverage_ci_high": hi,
                        "mean_precision": float(values["precision"].mean()),
                        "mean_unmet": float(values["unmet"].mean()),
                        "n_windows": int(len(values)),
                    }
                )

            pivot = k_group.pivot(index="sample_start", columns="branch", values="coverage")
            for contrast, a, b in (
                ("frame_gru_minus_entity_gru", "frame_gru", "entity_gru"),
                ("frame_gru_minus_entity_lstm", "frame_gru", "entity_lstm"),
            ):
                diff = (pivot[a] - pivot[b]).dropna().to_numpy()
                if len(diff) == 0:
                    continue
                lo, hi = bootstrap_ci(diff)
                try:
                    p_value = float(wilcoxon(diff, alternative="two-sided").pvalue)
                except ValueError:
                    p_value = 1.0
                paired_rows.append(
                    {
                        "city": city,
                        "score_mode": mode,
                        "k": k,
                        "contrast": contrast,
                        "mean_diff": float(diff.mean()),
                        "ci_low": lo,
                        "ci_high": hi,
                        "wilcoxon_p": p_value,
                        "n_windows": int(len(diff)),
                    }
                )

        # Per-seed stability for the main branches.
        for k in KS:
            for branch in ("frame_gru", "entity_gru", "frame_lstm", "entity_lstm"):
                per_seed = (
                    raw[(raw["city"] == city) & (raw["score_mode"] == mode) & (raw["k"] == k) & (raw["branch"] == branch)]
                    .groupby("seed")["coverage"]
                    .mean()
                )
                if per_seed.empty:
                    continue
                stability_rows.append(
                    {
                        "city": city,
                        "score_mode": mode,
                        "k": k,
                        "branch": branch,
                        "seed_mean": float(per_seed.mean()),
                        "seed_std": float(per_seed.std(ddof=1)),
                        "n_seeds": int(len(per_seed)),
                    }
                )

    summary = pd.DataFrame(summary_rows)
    paired = pd.DataFrame(paired_rows)
    if not paired.empty:
        paired["holm_p"] = np.nan
        for (city, mode, contrast), group in paired.groupby(["city", "score_mode", "contrast"], sort=False):
            paired.loc[group.index, "holm_p"] = holm(group["wilcoxon_p"].tolist())
    stability = pd.DataFrame(stability_rows)
    return summary, paired, stability


def density_table(window: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (city, mode), group in window.groupby(["city", "score_mode"], sort=False):
        median = float(group["true_active"].median())
        sparse = group["true_active"] <= median
        for stratum, mask in (("sparse", sparse), ("dense", ~sparse)):
            for k in KS:
                k_group = group[(group["k"] == k) & mask]
                for branch in ("frame_gru", "entity_gru", "entity_lstm", "oracle"):
                    values = k_group[k_group["branch"] == branch]
                    if values.empty:
                        continue
                    rows.append(
                        {
                            "city": city,
                            "score_mode": mode,
                            "stratum": stratum,
                            "median_true_active": median,
                            "k": k,
                            "branch": branch,
                            "mean_coverage": float(values["coverage"].mean()),
                            "n_windows": int(values["sample_start"].nunique()),
                        }
                    )
    return pd.DataFrame(rows)


def plot_coverage(summary: pd.DataFrame) -> None:
    plt.rcParams.update({
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(5.6, 2.35))
    for ax, city in zip(axes[:2], CITIES):
        group = summary[(summary["city"] == city) & (summary["score_mode"] == "mean")]
        for branch in ("random", "entity_gru", "entity_lstm", "oracle", "frame_gru"):
            values = group[group["branch"] == branch].sort_values("k")
            if values.empty:
                continue
            ax.plot(
                values["k"],
                values["mean_coverage"],
                marker="o",
                markersize=3.5,
                linewidth=2.2 if branch == "frame_gru" else 1.4,
                linestyle="--" if branch == "oracle" else "-",
                color=COLORS[branch],
                label=BRANCH_LABELS[branch],
                zorder=5 if branch == "frame_gru" else (4 if branch == "oracle" else 3),
            )
            if branch in {"frame_gru", "entity_gru", "entity_lstm"}:
                ax.fill_between(
                    values["k"],
                    values["coverage_ci_low"],
                    values["coverage_ci_high"],
                    color=COLORS[branch],
                    alpha=0.12,
                    linewidth=0,
                )
        ax.set_xscale("log")
        ax.set_xticks((5, 20, 100, 800))
        ax.set_xticklabels(("5", "20", "100", "800"))
        ax.set_xlabel("Dispatch budget K")
        ax.set_ylabel("Coverage of active cells")
        ax.set_title("Porto Taxi" if city == "porto" else "T-Drive", pad=4)
        ax.grid(True, which="both", alpha=0.25)
        ax.set_ylim(0, 1.02)

    ax = axes[2]
    for city, color in zip(CITIES, ("#1f77b4", "#d62728")):
        group = summary[(summary["city"] == city) & (summary["score_mode"] == "mean")]
        frame = group[group["branch"] == "frame_gru"].set_index("k")["mean_coverage"]
        entity = group[group["branch"] == "entity_gru"].set_index("k")["mean_coverage"]
        diff = (frame - entity).reindex(KS)
        ax.plot(
            diff.index,
            diff.values,
            marker="o",
            markersize=3.5,
            linewidth=1.8,
            color=color,
            label=("Porto Taxi" if city == "porto" else "T-Drive"),
        )
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xscale("log")
    ax.set_xticks((5, 20, 100, 800))
    ax.set_xticklabels(("5", "20", "100", "800"))
    ax.set_xlabel("Dispatch budget K")
    ax.set_ylabel("Coverage gain (frame - entity)")
    ax.set_title("Frame advantage over entity", pad=4)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)

    handles, labels = axes[0].get_legend_handles_labels()
    short = {
        "GeoSOT-ST frame (GRU)": "Frame (GRU)",
        "Aggregated entity (GRU)": "Entity (GRU)",
        "Aggregated entity (LSTM)": "Entity (LSTM)",
        "Oracle top-K": "Oracle",
        "Random allocation": "Random",
    }
    fig.legend(handles, [short.get(x, x) for x in labels], loc="lower center", ncol=3, frameon=False, fontsize=6.5)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(OUT / "fig_dispatch_coverage.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_dispatch_coverage.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def auc_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (city, mode), group in summary.groupby(["city", "score_mode"], sort=False):
        for branch in ("frame_gru", "entity_gru", "frame_lstm", "entity_lstm", "oracle", "random"):
            values = group[group["branch"] == branch].sort_values("k")
            if values.empty:
                continue
            x = np.log10(values["k"].to_numpy(dtype=float))
            y = values["mean_coverage"].to_numpy(dtype=float)
            rows.append(
                {
                    "city": city,
                    "score_mode": mode,
                    "branch": branch,
                    "coverage_auc_logk": float(np.trapezoid(y, x) / (x[-1] - x[0])),
                }
            )
    return pd.DataFrame(rows)


def plot_density(density: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for row, city in enumerate(CITIES):
        for col, stratum in enumerate(("sparse", "dense")):
            ax = axes[row, col]
            group = density[
                (density["city"] == city)
                & (density["score_mode"] == "mean")
                & (density["stratum"] == stratum)
            ]
            for branch in ("frame_gru", "entity_gru", "entity_lstm", "oracle"):
                values = group[group["branch"] == branch].sort_values("k")
                if values.empty:
                    continue
                ax.plot(
                    values["k"],
                    values["mean_coverage"],
                    marker="o",
                    markersize=3,
                    linewidth=1.4,
                    color=COLORS[branch],
                    label=BRANCH_LABELS[branch],
                )
            ax.set_xscale("log")
            ax.set_xticks(KS)
            ax.set_xticklabels([str(k) for k in KS], fontsize=7)
            ax.grid(True, which="both", alpha=0.25)
            ax.set_ylim(0, 1.02)
            ax.set_title(f"{'Porto Taxi' if city == 'porto' else 'T-Drive'} — {stratum}")
            if col == 0:
                ax.set_ylabel("Coverage")
            if row == 1:
                ax.set_xlabel("Dispatch budget K (cells)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "fig_dispatch_density.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_dispatch_density.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    raw, truth_meta = build_rows()
    window = (
        raw.groupby(["city", "score_mode", "k", "branch", "sample_start"], as_index=False)
        .agg(coverage=("coverage", "mean"), precision=("precision", "mean"), unmet=("unmet", "mean"), true_active=("true_active_cells", "first"))
    )
    summary, paired, stability = summarize_raw(raw)
    density = density_table(window)
    auc = auc_table(summary)

    raw.to_parquet(OUT / "application_dispatch_case.parquet", index=False)
    summary.to_csv(OUT / "application_dispatch_summary.csv", index=False)
    paired.to_csv(OUT / "application_dispatch_paired.csv", index=False)
    stability.to_csv(OUT / "application_dispatch_seed_stability.csv", index=False)
    density.to_csv(OUT / "application_dispatch_density.csv", index=False)
    auc.to_csv(OUT / "application_dispatch_auc.csv", index=False)
    truth_meta.to_csv(OUT / "application_dispatch_windows.csv", index=False)

    plot_coverage(summary)
    plot_density(density)

    print("wrote", OUT)
    print("\nHeadline: mean-score coverage at K=100")
    headline = summary[
        (summary["score_mode"] == "mean")
        & (summary["k"] == 100)
        & (summary["branch"].isin(["frame_gru", "entity_gru", "entity_lstm", "oracle"]))
    ]
    print(headline[["city", "branch", "mean_coverage", "coverage_ci_low", "coverage_ci_high"]].to_string(index=False))
    print("\nCoverage AUC (log-K normalized)")
    print(auc[auc["score_mode"] == "mean"].to_string(index=False))
    print("\nPaired contrasts at K=100")
    print(
        paired[(paired["score_mode"] == "mean") & (paired["k"] == 100)][
            ["city", "contrast", "mean_diff", "ci_low", "ci_high", "wilcoxon_p", "holm_p"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
