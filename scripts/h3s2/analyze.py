#!/usr/bin/env python3
"""Analyze the native GeoSOT / H3 / S2 partition runs.

Reads ``runs/<partition>/<city>_convgru_seed<seed>/`` and reports:

  * ten-seed event-conditioned F1 / IoU / precision / recall per partition,
  * paired H3-minus-GeoSOT and S2-minus-GeoSOT differences with a
    95% temporal-block bootstrap interval,
  * an equivalence decision against a pre-specified margin (|dF1| <= 0.01).

    python papers/APIN/experiments/h3s2/analyze.py --margin 0.01
"""

from __future__ import annotations

import argparse
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

def _repo_root(start: Path) -> Path:
    """Locate the checkout root from any copy of this script."""
    for candidate in [start, *start.parents]:
        if (candidate / "src" / "geosot_generative_trajectory").is_dir():
            return candidate
    raise SystemExit(
        "repository root not found; run from a checkout containing "
        "src/geosot_generative_trajectory"
    )


ROOT = _repo_root(Path(__file__).resolve())
RUNS = ROOT / "papers/APIN/experiments/h3s2/runs"
COMMON = RUNS / "common_target"
OUT = Path(os.environ.get("APIN_OUT", ROOT / "papers/APIN/experiments/h3s2/results"))
PARTITIONS = ("geosot", "h3", "s2", "quadtree")
CITIES = ("porto", "t_drive")
SEEDS = list(range(41, 51))
BLOCK = 12
N_BOOT = 10_000
RNG = np.random.default_rng(20261101)


def load_metrics(city: str, partition: str, seed: int, root: Path = RUNS) -> dict | None:
    path = root / partition / f"{city}_convgru_seed{seed}" / f"convgru_seed{seed}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return payload["target_test"]["event"]


def load_windows(city: str, partition: str, seed: int, root: Path = RUNS) -> np.ndarray | None:
    path = root / partition / f"{city}_convgru_seed{seed}" / f"convgru_seed{seed}_target_windows.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        true = data["targets"][:, :, 0].reshape(len(data["targets"]), -1) > 0.5
        pred = data["predictions"][:, :, 0].reshape(len(data["predictions"]), -1) > 0.5
    tp = np.logical_and(true, pred).sum(axis=1).astype(np.float64)
    fp = np.logical_and(~true, pred).sum(axis=1).astype(np.float64)
    fn = np.logical_and(true, ~pred).sum(axis=1).astype(np.float64)
    return 2 * tp / np.maximum(1.0, 2 * tp + fp + fn)


def block_ci(diff: np.ndarray) -> tuple[float, float]:
    n_blocks = len(diff) // BLOCK
    if n_blocks < 2:
        return float("nan"), float("nan")
    blocks = diff[: n_blocks * BLOCK].reshape(n_blocks, BLOCK)
    draws = RNG.integers(0, n_blocks, size=(N_BOOT, n_blocks))
    means = blocks[draws].mean(axis=(1, 2))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--margin", type=float, default=0.01)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for city in CITIES:
        windows: dict[str, dict[int, np.ndarray]] = {p: {} for p in PARTITIONS}
        for partition in PARTITIONS:
            metrics = []
            for seed in SEEDS:
                event = load_metrics(city, partition, seed)
                if event is None:
                    continue
                metrics.append(event)
                w = load_windows(city, partition, seed)
                if w is not None:
                    windows[partition][seed] = w
            if not metrics:
                raise SystemExit(
                    f"no completed runs for {city}/{partition} under {RUNS}; "
                    "build the frame caches and run the native queue first"
                )
            frame = pd.DataFrame(metrics)
            row = {
                "city": city,
                "partition": partition,
                "n_seeds": len(frame),
                "f1": frame["occupancy_f1"].mean(),
                "f1_sd": frame["occupancy_f1"].std(ddof=1) if len(frame) > 1 else float("nan"),
                "iou": frame["occupancy_iou"].mean(),
                "precision": frame["occupancy_precision"].mean(),
                "recall": frame["occupancy_recall"].mean(),
            }
            rows.append(row)
            print(f"{city:8s} {partition:7s} seeds={len(frame)} "
                  f"F1={row['f1']:.4f} IoU={row['iou']:.4f}")
        if windows["geosot"] and all(windows[p] for p in ("h3", "s2")):
            for partition in ("h3", "s2", "quadtree"):
                common = sorted(set(windows["geosot"]) & set(windows[partition]))
                if not common:
                    continue
                base = np.mean(np.vstack([windows["geosot"][s] for s in common]), axis=0)
                other = np.mean(np.vstack([windows[partition][s] for s in common]), axis=0)
                n = min(len(base), len(other))
                diff = other[:n] - base[:n]
                lo, hi = block_ci(diff)
                same = (lo > -args.margin) and (hi < args.margin)
                rows.append({
                    "city": city,
                    "partition": f"{partition}_minus_geosot",
                    "n_seeds": len(common),
                    "f1": float(diff.mean()),
                    "f1_sd": float("nan"),
                    "iou": float("nan"),
                    "precision": float("nan"),
                    "recall": float("nan"),
                    "ci_low": lo,
                    "ci_high": hi,
                    "equiv_margin": args.margin,
                    "equivalent": bool(same),
                })
                print(f"{city:8s} {partition}-geosot dF1={diff.mean():+.4f} "
                      f"CI=({lo:+.4f},{hi:+.4f}) equivalent={same}")

    # Common-target control: H3/S2 input on the GeoSOT 32x32 target, three
    # seeds (41-43), paired against the rebuilt GeoSOT runs of the same seeds.
    for city in CITIES:
        base_windows = [load_windows(city, "geosot", seed) for seed in (41, 42, 43)]
        base_windows = [w for w in base_windows if w is not None]
        for partition in ("h3", "s2", "quadtree"):
            metrics = [load_metrics(city, partition, seed, COMMON) for seed in (41, 42, 43)]
            metrics = [m for m in metrics if m is not None]
            if not metrics:
                raise SystemExit(
                    f"no common-target runs for {city}/{partition} under {COMMON}; "
                    "run run_common_target.sh first"
                )
            frame = pd.DataFrame(metrics)
            rows.append({
                "city": city, "partition": f"common_{partition}", "n_seeds": len(frame),
                "f1": frame["occupancy_f1"].mean(),
                "f1_sd": frame["occupancy_f1"].std(ddof=1) if len(frame) > 1 else float("nan"),
                "iou": frame["occupancy_iou"].mean(),
                "precision": frame["occupancy_precision"].mean(),
                "recall": frame["occupancy_recall"].mean(),
            })
            print(f"{city:8s} common_{partition:5s} seeds={len(frame)} F1={frame['occupancy_f1'].mean():.4f}")
            other = [load_windows(city, partition, seed, COMMON) for seed in (41, 42, 43)]
            other = [w for w in other if w is not None]
            if base_windows and other:
                base = np.mean(np.vstack(base_windows), axis=0)
                cand = np.mean(np.vstack(other), axis=0)
                n = min(len(base), len(cand))
                diff = cand[:n] - base[:n]
                lo, hi = block_ci(diff)
                rows.append({
                    "city": city, "partition": f"common_{partition}_minus_geosot",
                    "n_seeds": len(other), "f1": float(diff.mean()), "f1_sd": float("nan"),
                    "iou": float("nan"), "precision": float("nan"), "recall": float("nan"),
                    "ci_low": lo, "ci_high": hi, "equiv_margin": args.margin,
                    "equivalent": bool(lo > -args.margin and hi < args.margin),
                })
                print(f"{city:8s} common_{partition}-geosot dF1={diff.mean():+.4f} CI=({lo:+.4f},{hi:+.4f})")

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "partition_native_summary.csv", index=False)
    print(f"\nwrote {OUT/'partition_native_summary.csv'}")


if __name__ == "__main__":
    main()
