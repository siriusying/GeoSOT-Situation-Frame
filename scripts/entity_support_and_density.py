#!/usr/bin/env python3
"""Entity support ceiling and density stratification.

No training is required: this script re-uses the registered aggregation
artifacts.  It answers two reviewer questions:

1. Is the frame-vs-entity gap caused by a weak entity model?
   Even a perfect entity forecast can mark at most one cell per entity per
   step, so its recall is bounded by
       min(1, predicted_entity_count * future_steps / true_active_cell_steps).
   We compute this model-independent ceiling for GRU and LSTM entity branches.

2. Where does the frame advantage concentrate?
   We stratify event-conditioned occupancy precision/recall/F1 by the number
   of true active cell-steps per window (sparse vs. dense).
"""
from __future__ import annotations

from pathlib import Path
import os

import numpy as np
import pandas as pd

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
FUTURE_STEPS = 4


def load(city: str, branch: str, seed: int) -> dict[str, np.ndarray]:
    path = AGG / f"{city}_{branch}" / f"{city}_paradigm_seed{seed}.npz"
    with np.load(path, allow_pickle=False) as data:
        return {
            "targets": data["targets"][:, :, 0].astype(np.float32),
            "frame": data["frame_predictions"][:, :, 0].astype(np.float32),
            "aggregate": data["aggregate_predictions"][:, :, 0].astype(np.float32),
            "entity_count": data["predicted_entity_count"].astype(np.float32),
            "entity_cell_counts": data["cell_entity_counts"][:, :, 0].astype(np.int16),
            "starts": data["sample_starts"].astype(np.int64),
        }


def binary_metrics(truth: np.ndarray, pred: np.ndarray, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truth_b = truth > 0.5
    pred_b = pred > threshold
    tp = np.logical_and(truth_b, pred_b).sum(axis=(1, 2, 3)).astype(float)
    fp = np.logical_and(~truth_b, pred_b).sum(axis=(1, 2, 3)).astype(float)
    fn = np.logical_and(truth_b, ~pred_b).sum(axis=(1, 2, 3)).astype(float)
    precision = tp / np.maximum(1.0, tp + fp)
    recall = tp / np.maximum(1.0, tp + fn)
    f1 = 2 * tp / np.maximum(1.0, 2 * tp + fp + fn)
    return precision, recall, f1


def main() -> None:
    rows = []
    for city in CITIES:
        for seed in SEEDS:
            gru = load(city, "gru", seed)
            lstm = load(city, "lstm", seed)
            if not np.array_equal(gru["starts"], lstm["starts"]):
                raise RuntimeError(f"start mismatch {city} {seed}")

            truth = gru["targets"]
            true_cell_steps = (truth > 0.5).sum(axis=(1, 2, 3)).astype(float)
            frame_p, frame_r, frame_f1 = binary_metrics(truth, gru["frame"])
            gru_p, gru_r, gru_f1 = binary_metrics(truth, gru["aggregate"])
            lstm_p, lstm_r, lstm_f1 = binary_metrics(truth, lstm["aggregate"])

            # Model-independent entity ceiling: each predicted entity marks at
            # most one cell per future step.
            for branch, data in (("entity_gru", gru), ("entity_lstm", lstm)):
                # Unique cell support actually marked by the entity branch.
                unique_support = (data["entity_cell_counts"] > 0).sum(axis=(1, 2, 3)).astype(float)
                support_ceiling = np.minimum(1.0, unique_support / np.maximum(1.0, true_cell_steps))
                optimistic_ceiling = np.minimum(1.0, data["entity_count"] * FUTURE_STEPS / np.maximum(1.0, true_cell_steps))
                observed_recall = gru_r if branch == "entity_gru" else lstm_r
                rows.append(
                    pd.DataFrame(
                        {
                            "city": city,
                            "seed": seed,
                            "sample_start": data["starts"],
                            "branch": branch,
                            "true_active_cell_steps": true_cell_steps,
                            "predicted_entity_count": data["entity_count"],
                            "unique_support_cell_steps": unique_support,
                            "recall_ceiling": support_ceiling,
                            "optimistic_ceiling": optimistic_ceiling,
                            "observed_recall": observed_recall,
                            "ceiling_ratio": observed_recall / np.maximum(1e-9, support_ceiling),
                        }
                    )
                )

            # Frame and entity metrics for density stratification.
            rows.append(
                pd.DataFrame(
                    {
                        "city": city,
                        "seed": seed,
                        "sample_start": gru["starts"],
                        "branch": "frame",
                        "true_active_cell_steps": true_cell_steps,
                        "precision": frame_p,
                        "recall": frame_r,
                        "f1": frame_f1,
                    }
                )
            )
            rows.append(
                pd.DataFrame(
                    {
                        "city": city,
                        "seed": seed,
                        "sample_start": gru["starts"],
                        "branch": "entity_gru",
                        "true_active_cell_steps": true_cell_steps,
                        "precision": gru_p,
                        "recall": gru_r,
                        "f1": gru_f1,
                    }
                )
            )
            rows.append(
                pd.DataFrame(
                    {
                        "city": city,
                        "seed": seed,
                        "sample_start": gru["starts"],
                        "branch": "entity_lstm",
                        "true_active_cell_steps": true_cell_steps,
                        "precision": lstm_p,
                        "recall": lstm_r,
                        "f1": lstm_f1,
                    }
                )
            )

    raw = pd.concat(rows, ignore_index=True)
    raw.to_parquet(OUT / "entity_support_and_density_raw.parquet", index=False)

    # Support ceiling summary (one row per city/branch).
    ceiling_rows = []
    for (city, branch), group in raw[raw["branch"].isin(["entity_gru", "entity_lstm"])].groupby(["city", "branch"]):
        ceiling_rows.append(
            {
                "city": city,
                "branch": branch,
                "mean_true_active_cell_steps": float(group["true_active_cell_steps"].mean()),
                "mean_predicted_entity_count": float(group["predicted_entity_count"].mean()),
                "mean_unique_support_cell_steps": float(group["unique_support_cell_steps"].mean()),
                "mean_recall_ceiling": float(group["recall_ceiling"].mean()),
                "mean_optimistic_ceiling": float(group["optimistic_ceiling"].mean()),
                "mean_observed_recall": float(group["observed_recall"].mean()),
                "mean_ceiling_ratio": float(group["ceiling_ratio"].mean()),
                "n_windows": int(len(group)),
            }
        )
    ceiling = pd.DataFrame(ceiling_rows)
    ceiling.to_csv(OUT / "entity_support_ceiling.csv", index=False)

    # Density stratification: median split of true active cell-steps.
    metric_rows = raw[raw["branch"].isin(["frame", "entity_gru", "entity_lstm"])].copy()
    density_rows = []
    for (city, branch), group in metric_rows.groupby(["city", "branch"]):
        median = float(group["true_active_cell_steps"].median())
        sparse = group["true_active_cell_steps"] <= median
        for stratum, mask in (("sparse", sparse), ("dense", ~sparse)):
            subset = group[mask]
            density_rows.append(
                {
                    "city": city,
                    "branch": branch,
                    "stratum": stratum,
                    "median_true_active_cell_steps": median,
                    "mean_true_active_cell_steps": float(subset["true_active_cell_steps"].mean()),
                    "mean_precision": float(subset["precision"].mean()),
                    "mean_recall": float(subset["recall"].mean()),
                    "mean_f1": float(subset["f1"].mean()),
                    "n_windows": int(len(subset)),
                }
            )
    density = pd.DataFrame(density_rows)
    density.to_csv(OUT / "density_stratification.csv", index=False)

    print("Support ceiling")
    print(ceiling.to_string(index=False))
    print("\nDensity stratification")
    print(density.to_string(index=False))


if __name__ == "__main__":
    main()
