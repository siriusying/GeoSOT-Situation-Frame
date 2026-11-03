#!/usr/bin/env python3
"""Model-free baselines for the GeoSOT-ST frame forecast.

The manuscript compares frame generation with post-hoc aggregation of entity
forecasts. A reviewer can still ask whether the frame predictor beats trivial
temporal baselines. This script evaluates two such baselines on exactly the
same test windows as the stored ten-seed ConvGRU predictions:

    persistence      repeat the last observed occupancy frame for all four steps
    moving_average   threshold the mean occupancy of the 8-step history at 0.5

It reports event-conditioned occupancy F1, per-horizon F1, and the coverage of
a budgeted top-K dispatch built from each score, with temporal-block bootstrap
intervals for the frame-minus-baseline differences. Nothing is trained.

    python experiments/naive_baselines.py

Outputs are written to ``papers/APIN/experiments/results``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

def _repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src" / "geosot_generative_trajectory").is_dir():
            return candidate
    raise SystemExit("run from inside the repository (src/geosot_generative_trajectory not found)")


ROOT = _repo_root(Path(__file__).resolve().parent)
RUNS = ROOT / "papers/paper_geosot_generative_trajectory_kbs/0911/requested_4_5_full10"
CACHE = (
    ROOT
    / "papers/paper_geosot_generative_trajectory_kbs/elsarticle/audit_artifacts"
    / "representation_benchmark_v1/paradigm_confirmatory_v1/compact_frame_cache"
)
OUT = Path(os.environ.get("APIN_OUT", Path(__file__).resolve().parent.parent / "artifacts" / "results"))

CITIES = ("porto", "t_drive")
SEEDS = list(range(41, 51))
HISTORY = 8
FUTURE = 4
BLOCK = 12
N_BOOT = 10_000
RNG = np.random.default_rng(20261101)
K_DISPATCH = 100


def load_occupancy(city: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the occupancy channel of targets/predictions and the window starts."""
    npz = RUNS / f"{city}_convgru_6ch_seed{seed}" / f"convgru_seed{seed}_target_windows.npz"
    with np.load(npz, allow_pickle=False) as data:
        targets = data["targets"][:, :, 0].astype(np.float32)
        preds = data["predictions"][:, :, 0].astype(np.float32)
        starts = data["sample_starts"].astype(np.int64)
    return targets, preds, starts


def history_frames(city: str, starts: np.ndarray) -> np.ndarray:
    """Return the 8-step history occupancy for each window: (N, 8, 32, 32)."""
    frames = np.load(CACHE / city / "test" / "frames.npy", mmap_mode="r")
    last = int(starts.max()) + HISTORY + FUTURE
    if last > len(frames):
        raise SystemExit(f"{city}: window {last} exceeds frame cache ({len(frames)})")
    index = starts[:, None] + np.arange(HISTORY)[None, :]
    return np.asarray(frames[index, 0], dtype=np.float32)


def per_window_f1(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    t = true.reshape(len(true), -1) > 0.5
    p = pred.reshape(len(pred), -1) > 0.5
    tp = np.logical_and(t, p).sum(axis=1).astype(np.float64)
    fp = np.logical_and(~t, p).sum(axis=1).astype(np.float64)
    fn = np.logical_and(t, ~p).sum(axis=1).astype(np.float64)
    return 2 * tp / np.maximum(1.0, 2 * tp + fp + fn)


def per_horizon_f1(true: np.ndarray, pred: np.ndarray) -> dict[int, np.ndarray]:
    """Per-window F1 computed separately for each future step."""
    return {h: per_window_f1(true[:, h : h + 1], pred[:, h : h + 1]) for h in range(true.shape[1])}


def dispatch_coverage(score: np.ndarray, truth_union: np.ndarray, k: int) -> np.ndarray:
    """Coverage of the true four-step active union by the top-k cells."""
    flat = score.reshape(len(score), -1)
    k = max(1, min(k, flat.shape[1]))
    index = np.argpartition(-flat, kth=k - 1, axis=1)[:, :k]
    mask = np.zeros_like(flat, dtype=bool)
    np.put_along_axis(mask, index, True, axis=1)
    selected = mask.reshape(score.shape)
    hit = np.logical_and(selected, truth_union).sum(axis=(1, 2)).astype(np.float64)
    return hit / np.maximum(1.0, truth_union.sum(axis=(1, 2)).astype(np.float64))


def block_bootstrap_ci(diff: np.ndarray) -> tuple[float, float]:
    n_blocks = len(diff) // BLOCK
    if n_blocks < 2:
        return float("nan"), float("nan")
    blocks = diff[: n_blocks * BLOCK].reshape(n_blocks, BLOCK)
    draws = RNG.integers(0, n_blocks, size=(N_BOOT, n_blocks))
    means = blocks[draws].mean(axis=(1, 2))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    per_seed_rows: list[dict] = []
    summary_rows: list[dict] = []

    for city in CITIES:
        targets, _, starts = load_occupancy(city, SEEDS[0])
        history = history_frames(city, starts)
        event = targets.reshape(len(targets), -1).max(axis=1) > 0.5
        truth_union = (targets > 0.5).any(axis=1)

        baselines = {
            "persistence": np.repeat(history[:, -1:], FUTURE, axis=1),
            "moving_average": np.repeat(
                (history.mean(axis=1, keepdims=True) > 0.5).astype(np.float32), FUTURE, axis=1
            ),
        }
        baseline_f1 = {name: per_window_f1(targets, pred) for name, pred in baselines.items()}
        baseline_h = {name: per_horizon_f1(targets, pred) for name, pred in baselines.items()}
        baseline_cov = {
            name: dispatch_coverage(pred.mean(axis=1), truth_union, K_DISPATCH)
            for name, pred in baselines.items()
        }

        frame_f1_seeds: list[np.ndarray] = []
        frame_h_seeds: list[dict[int, np.ndarray]] = []
        frame_cov_seeds: list[np.ndarray] = []
        for seed in SEEDS:
            targets_seed, preds_seed, _ = load_occupancy(city, seed)
            if not np.array_equal(targets_seed, targets):
                raise SystemExit(f"{city} seed {seed}: targets differ across seeds")
            f1 = per_window_f1(targets_seed, preds_seed)
            cov = dispatch_coverage(preds_seed.mean(axis=1), truth_union, K_DISPATCH)
            frame_f1_seeds.append(f1)
            frame_h_seeds.append(per_horizon_f1(targets_seed, preds_seed))
            frame_cov_seeds.append(cov)
            per_seed_rows.append(
                {"city": city, "seed": seed, "method": "frame",
                 "event_f1": float(f1[event].mean()),
                 "dispatch_coverage_k100": float(cov[event].mean())}
            )
        for name in baselines:
            per_seed_rows.append(
                {"city": city, "seed": SEEDS[0], "method": name,
                 "event_f1": float(baseline_f1[name][event].mean()),
                 "dispatch_coverage_k100": float(baseline_cov[name][event].mean())}
            )

        frame_f1 = np.mean(np.vstack(frame_f1_seeds), axis=0)
        frame_h = {
            h: np.mean(np.vstack([d[h] for d in frame_h_seeds]), axis=0) for h in range(FUTURE)
        }
        frame_cov = np.mean(np.vstack(frame_cov_seeds), axis=0)
        series = {
            "frame": (frame_f1, frame_h, frame_cov),
            "persistence": (baseline_f1["persistence"], baseline_h["persistence"], baseline_cov["persistence"]),
            "moving_average": (
                baseline_f1["moving_average"],
                baseline_h["moving_average"],
                baseline_cov["moving_average"],
            ),
        }
        for method, (f1, hdict, cov) in series.items():
            summary_rows.append(
                {"city": city, "method": method,
                 "event_f1": float(f1[event].mean()),
                 **{f"f1_h{h + 1}": float(hdict[h][event].mean()) for h in range(FUTURE)},
                 "dispatch_coverage_k100": float(cov[event].mean()),
                 "ci_low": float("nan"), "ci_high": float("nan"), "wilcoxon_p": float("nan")}
            )
        for name in baselines:
            diff = frame_f1[event] - baseline_f1[name][event]
            lo, hi = block_bootstrap_ci(diff)
            summary_rows.append(
                {"city": city, "method": f"frame_minus_{name}",
                 "event_f1": float(diff.mean()),
                 **{f"f1_h{h + 1}": float("nan") for h in range(FUTURE)},
                 "dispatch_coverage_k100": float(
                     (frame_cov[event] - baseline_cov[name][event]).mean()
                 ),
                 "ci_low": lo, "ci_high": hi,
                 "wilcoxon_p": float(wilcoxon(diff).pvalue)}
            )

    summary = pd.DataFrame(summary_rows)
    per_seed = pd.DataFrame(per_seed_rows)
    summary.to_csv(OUT / "naive_baselines_summary.csv", index=False)
    per_seed.to_csv(OUT / "naive_baselines_per_seed.csv", index=False)
    with pd.option_context("display.width", 200):
        print(summary.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print(f"\nwrote {OUT}/naive_baselines_summary.csv and naive_baselines_per_seed.csv")


if __name__ == "__main__":
    main()
