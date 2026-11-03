# -*- coding: utf-8 -*-
"""Alignment ablation for the equal-area quadtree (T-Drive).

Variants share the cell size and area; only the grid phase (half-cell shifts) or
the rotation about the city centre changes. Reports ten-seed F1 per variant and
the paired difference against the canonical quadtree grid.

    python papers/APIN/experiments/h3s2/analyze_alignment.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from partitions import QTREE_VARIANTS, _repo_root

ROOT = _repo_root(Path(__file__).resolve())
RUNS = ROOT / "papers/APIN/experiments/h3s2/runs"
CACHE = ROOT / "papers/APIN/experiments/h3s2/frame_cache"
OUT = Path(os.environ.get("APIN_OUT", ROOT / "papers/APIN/experiments/h3s2/results"))
CITY = "t_drive"
SEEDS = list(range(41, 51))
N_BOOT = 10_000
RNG = np.random.default_rng(20261102)


def load_f1(variant: str) -> pd.Series:
    values = {}
    for seed in SEEDS:
        path = RUNS / variant / f"{CITY}_convgru_seed{seed}" / f"convgru_seed{seed}.json"
        if not path.exists():
            continue
        values[seed] = json.loads(path.read_text())["target_test"]["event"]["occupancy_f1"]
    return pd.Series(values, dtype=float)


def paired_ci(diff: np.ndarray) -> tuple[float, float]:
    draws = RNG.choice(diff, size=(N_BOOT, len(diff)), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    baseline = load_f1("quadtree")
    if baseline.empty:
        raise SystemExit(f"no canonical quadtree runs under {RUNS}")
    rows = []
    for variant, (shift_x, shift_y, angle_deg) in QTREE_VARIANTS.items():
        series = load_f1(variant)
        if series.empty:
            continue
        meta = json.loads((CACHE / variant / CITY / "test" / "metadata.json").read_text())
        rows.append(
            {
                "variant": variant,
                "shift_x": shift_x,
                "shift_y": shift_y,
                "angle_deg": angle_deg,
                "canvas_rows": int(meta["grid"]["grid_shape"][0]),
                "canvas_cols": int(meta["grid"]["grid_shape"][1]),
                "n_seeds": int(len(series)),
                "f1": float(series.mean()),
                "f1_sd": float(series.std(ddof=1)),
            }
        )
    table = pd.DataFrame(rows)
    table = table.set_index("variant")
    common = baseline.index.intersection(table.index)
    for variant in table.index:
        series = load_f1(variant)
        paired = series.index.intersection(baseline.index)
        diff = (series.loc[paired] - baseline.loc[paired]).to_numpy()
        if len(diff) == 0:
            continue
        lo, hi = paired_ci(diff)
        table.loc[variant, "n_paired"] = len(diff)
        table.loc[variant, "delta_vs_quadtree"] = float(diff.mean())
        table.loc[variant, "ci_low"] = lo
        table.loc[variant, "ci_high"] = hi
    table = table.reset_index()
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "alignment_ablation.csv", index=False)
    with pd.option_context("display.width", 200):
        print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    span = table["f1"].max() - table["f1"].min()
    print(f"\nT-Drive F1 span across {len(table)} variants: {span:.4f}")
    print(f"wrote {OUT / 'alignment_ablation.csv'}")


if __name__ == "__main__":
    main()
