#!/usr/bin/env python3
"""Robustness checks for the main contrasts in the GeoSOT-ST manuscript.

For every headline contrast we report how far the conclusion can move when a
single temporal block or a single seed is removed:

  * block_min / block_max and the fraction of positive blocks,
  * leave-one-block-out (LOBO) range of the block-averaged difference,
  * seed_min / seed_max for seed-averaged differences.

Contrasts: frame minus aggregated entity (GRU and LSTM), frame minus
persistence, and the leave-one-seed-out spread of the matched backbone F1.

    python experiments/robustness_checks.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

def _repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src" / "geosot_generative_trajectory").is_dir():
            return candidate
    raise SystemExit("run from inside the repository (src/geosot_generative_trajectory not found)")


ROOT = _repo_root(Path(__file__).resolve().parent)
RUNS = ROOT / "papers/paper_geosot_generative_trajectory_kbs/0911/requested_4_5_full10"
AGG = (
    ROOT
    / "papers/paper_geosot_generative_trajectory_kbs/elsarticle/audit_artifacts"
    / "representation_benchmark_v1/paradigm_confirmatory_v1/full_v1/aggregation"
)
CACHE = (
    ROOT
    / "papers/paper_geosot_generative_trajectory_kbs/elsarticle/audit_artifacts"
    / "representation_benchmark_v1/paradigm_confirmatory_v1/compact_frame_cache"
)
OUT = Path(os.environ.get("APIN_OUT", Path(__file__).resolve().parent.parent / "artifacts" / "results"))
CITIES = ("porto", "t_drive")
SEEDS = list(range(41, 51))
BLOCK = 12


def f1_window(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    t = true.reshape(len(true), -1) > 0.5
    p = pred.reshape(len(pred), -1) > 0.5
    tp = np.logical_and(t, p).sum(axis=1).astype(np.float64)
    fp = np.logical_and(~t, p).sum(axis=1).astype(np.float64)
    fn = np.logical_and(t, ~p).sum(axis=1).astype(np.float64)
    return 2 * tp / np.maximum(1.0, 2 * tp + fp + fn)


def frame_series(city: str) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Ten-seed mean frame F1, per seed, plus the target array."""
    seeds = []
    targets0 = None
    starts = None
    for seed in SEEDS:
        npz = RUNS / f"{city}_convgru_6ch_seed{seed}" / f"convgru_seed{seed}_target_windows.npz"
        with np.load(npz, allow_pickle=False) as data:
            targets = data["targets"][:, :, 0].astype(np.float32)
            preds = data["predictions"][:, :, 0].astype(np.float32)
            if starts is None:
                starts = data["sample_starts"].astype(np.int64)
        if targets0 is None:
            targets0 = targets
        seeds.append(f1_window(targets, preds))
    return np.mean(np.vstack(seeds), axis=0), {"frame": np.vstack(seeds)}, targets0, starts


def aggregated_series(city: str, branch: str) -> np.ndarray:
    per_seed = []
    for seed in SEEDS:
        npz = AGG / f"{city}_{branch}" / f"{city}_paradigm_seed{seed}.npz"
        with np.load(npz, allow_pickle=False) as data:
            per_seed.append(f1_window(data["targets"][:, :, 0], data["aggregate_predictions"][:, :, 0]))
    return np.mean(np.vstack(per_seed), axis=0)


def persistence_series(city: str, starts: np.ndarray) -> np.ndarray:
    frames = np.load(CACHE / city / "test" / "frames.npy", mmap_mode="r")
    index = starts[:, None] + np.arange(8)[None, :]
    history = np.asarray(frames[index, 0], dtype=np.float32)
    targets = _targets_from_cache(frames, starts)
    return f1_window(targets, np.repeat(history[:, -1:], 4, axis=1))


def _targets_from_cache(frames, starts: np.ndarray) -> np.ndarray:
    return np.stack([np.asarray(frames[s + 8 : s + 12, 0], dtype=np.float32) for s in starts])


def block_stats(diff: np.ndarray) -> dict[str, float]:
    n_blocks = len(diff) // BLOCK
    blocks = diff[: n_blocks * BLOCK].reshape(n_blocks, BLOCK).mean(axis=1)
    lobo = np.array([(blocks.sum() - b) / (n_blocks - 1) for b in blocks]) if n_blocks > 1 else blocks
    return {
        "mean": float(diff.mean()),
        "block_min": float(blocks.min()),
        "block_max": float(blocks.max()),
        "frac_positive": float((blocks > 0).mean()),
        "lobo_min": float(lobo.min()),
        "lobo_max": float(lobo.max()),
        "n_blocks": int(n_blocks),
    }


def main() -> None:
    rows = []
    for city in CITIES:
        frame_f1, seed_store, targets, starts = frame_series(city)
        agg_seed_f1 = {
            branch: aggregated_series(city, branch) for branch in ("gru", "lstm")
        }
        persist_f1 = persistence_series(city, starts)
        for name, baseline in (
            ("aggregated_entity_gru", agg_seed_f1["gru"]),
            ("aggregated_entity_lstm", agg_seed_f1["lstm"]),
            ("persistence", persist_f1),
        ):
            diff = frame_f1 - baseline
            stats = block_stats(diff)
            rows.append({"city": city, "contrast": f"frame_minus_{name}", **stats})

    per_seed = pd.read_csv(OUT / "backbone_runs_summary.csv")
    for (city, tag), group in per_seed.groupby(["city", "tag"]):
        values = group["f1"].to_numpy()
        rows.append(
            {
                "city": city,
                "contrast": f"backbone_{tag}",
                "mean": float(values.mean()),
                "block_min": float("nan"),
                "block_max": float("nan"),
                "frac_positive": float("nan"),
                "lobo_min": float(values.min()),
                "lobo_max": float(values.max()),
                "n_blocks": int(len(values)),
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "robustness_checks.csv", index=False)
    with pd.option_context("display.width", 200):
        print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def _preds(city: str, seed: int) -> np.ndarray:
    npz = RUNS / f"{city}_convgru_6ch_seed{seed}" / f"convgru_seed{seed}_target_windows.npz"
    with np.load(npz, allow_pickle=False) as data:
        return data["predictions"][:, :, 0].astype(np.float32)


if __name__ == "__main__":
    main()
