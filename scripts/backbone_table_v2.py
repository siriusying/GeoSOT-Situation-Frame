"""Rebuild the matched backbone table from artifacts, with contract validation.

The table reported in the paper must come from one frame contract.  This script
verifies that every run directory named below carries an identical spec apart
from the backbone and the seed, then recomputes event-conditioned occupancy
metrics from the stored per-window prediction arrays.  Paired contrasts use
block bootstrap over blocks of 12 consecutive test windows and a paired
Wilcoxon test with Holm correction inside each city.

Run from the repository root:

    python papers/APIN/experiments/backbone_table_v2.py
"""

from __future__ import annotations

import json
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
TRANSFORMER_RUNS = ROOT / "papers/APIN/experiments/transformer_runs"
OUT = Path(os.environ.get("APIN_OUT", Path(__file__).resolve().parent.parent / "artifacts" / "results"))

CITIES = ("porto", "t_drive")
BLOCK = 12
BOOTSTRAP = 10_000
SEEDS_FULL = list(range(41, 51))
SEEDS_EXPLORATORY = list(range(41, 51))
RNG = np.random.default_rng(20261031)

# tag -> (directory tag, model id, label)
BACKBONES = {
    "convgru_6ch": ("convgru_6ch", "convgru", "ConvGRU (6 channels)"),
    "convgru_2ch": ("convgru_2ch", "convgru", "ConvGRU (2 channels)"),
    "convlstm_6ch": ("convlstm", "convlstm", "ConvLSTM (6 channels)"),
    "stgcn": ("stgcn", "stgcn", "STGCN (6 channels)"),
    "graph_wavenet": ("graph_wavenet", "graph_wavenet", "Graph WaveNet (6 channels)"),
    "predrnnpp": ("predrnnpp", "predrnnpp", "PredRNN++ (6 channels)"),
}

CONTRACT_KEYS = (
    "source_manifest",
    "target_manifest",
    "frame_cache_root",
    "epochs",
    "batch_size",
    "hidden_channels",
    "learning_rate",
    "hierarchy_weight",
    "hierarchy_contrastive_weight",
    "road_rule_weight",
    "semantic_rule_weight",
    "traversability_weight",
    "sample_weight_mode",
    "test_sample_limit",
)


def run_dir(city: str, tag: str) -> Path:
    return RUNS / f"{city}_{tag}_seed{41}"


def check_contract() -> pd.DataFrame:
    rows = []
    for city in CITIES:
        for key, (tag, model, label) in BACKBONES.items():
            base = None
            for seed in SEEDS_FULL:
                spec_path = RUNS / f"{city}_{tag}_seed{seed}" / "spec.json"
                if not spec_path.exists():
                    continue
                spec = json.loads(spec_path.read_text())
                if base is None:
                    base = spec
                else:
                    for k in CONTRACT_KEYS:
                        if spec.get(k) != base.get(k):
                            raise SystemExit(
                                f"contract mismatch {city}/{tag}/{seed}: {k} "
                                f"{spec.get(k)!r} != {base.get(k)!r}"
                            )
            if base is not None:
                rows.append(
                    {
                        "city": city,
                        "tag": key,
                        "label": label,
                        "epochs": base["epochs"],
                        "hidden": base["hidden_channels"],
                        "test_limit": base["test_sample_limit"],
                        "frame_cache": Path(base["frame_cache_root"]).name,
                    }
                )
    return pd.DataFrame(rows)


def window_f1(npz: Path) -> np.ndarray | None:
    if not npz.exists():
        return None
    data = np.load(npz, allow_pickle=False)
    true = data["targets"][:, :, 0].reshape(len(data["targets"]), -1) > 0.5
    pred = data["predictions"][:, :, 0].reshape(len(data["predictions"]), -1) > 0.5
    tp = np.logical_and(true, pred).sum(axis=1).astype(np.float64)
    fp = np.logical_and(~true, pred).sum(axis=1).astype(np.float64)
    fn = np.logical_and(true, ~pred).sum(axis=1).astype(np.float64)
    return 2 * tp / np.maximum(1.0, 2 * tp + fp + fn)


def block_differences(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a - b
    n_blocks = len(diff) // BLOCK
    if n_blocks == 0:
        return diff
    return diff[: n_blocks * BLOCK].reshape(n_blocks, BLOCK).mean(axis=1)


def summaries() -> pd.DataFrame:
    rows = []
    for city in CITIES:
        for key, (tag, model, label) in BACKBONES.items():
            for seed in SEEDS_FULL:
                report = RUNS / f"{city}_{tag}_seed{seed}" / f"{model}_seed{seed}.json"
                if not report.exists():
                    continue
                payload = json.loads(report.read_text())
                event = payload["target_test"]["event"]
                rows.append(
                    {
                        "city": city,
                        "tag": key,
                        "label": label,
                        "seed": seed,
                        "precision": event["occupancy_precision"],
                        "recall": event["occupancy_recall"],
                        "f1": event["occupancy_f1"],
                        "iou": event["occupancy_iou"],
                    }
                )
    return pd.DataFrame(rows)


def transformer_summaries() -> pd.DataFrame:
    rows = []
    for city in CITIES:
        for seed in SEEDS_EXPLORATORY:
            report = TRANSFORMER_RUNS / f"{city}_transformer_seed{seed}" / f"transformer_seed{seed}.json"
            if not report.exists():
                continue
            payload = json.loads(report.read_text())
            event = payload["target_test"]["event"]
            rows.append(
                {
                    "city": city,
                    "tag": "transformer",
                    "label": "Transformer (6 channels)",
                    "seed": seed,
                    "precision": event["occupancy_precision"],
                    "recall": event["occupancy_recall"],
                    "f1": event["occupancy_f1"],
                    "iou": event["occupancy_iou"],
                }
            )
    return pd.DataFrame(rows)


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = (len(p_values) - rank) * p_values[index]
        running = max(running, value)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def paired(city: str, tag: str, ref_tag: str, seeds: list[int], runs_root: Path) -> dict | None:
    diffs = []
    for seed in seeds:
        tag_dir = BACKBONES.get(tag, (tag, tag, tag))[0]
        ref_dir = BACKBONES[ref_tag][0]
        model = BACKBONES.get(tag, (tag, tag, tag))[1]
        ref_model = BACKBONES[ref_tag][1]
        if tag == "transformer":
            a = window_f1(runs_root / f"{city}_transformer_seed{seed}" / f"transformer_seed{seed}_target_windows.npz")
            b = window_f1(RUNS / f"{city}_{ref_dir}_seed{seed}" / f"{ref_model}_seed{seed}_target_windows.npz")
        else:
            a = window_f1(runs_root / f"{city}_{tag_dir}_seed{seed}" / f"{model}_seed{seed}_target_windows.npz")
            b = window_f1(runs_root / f"{city}_{ref_dir}_seed{seed}" / f"{ref_model}_seed{seed}_target_windows.npz")
        if a is None or b is None or len(a) != len(b):
            continue
        diffs.append(block_differences(a, b))
    if not diffs:
        return None
    blocks = np.concatenate(diffs)
    idx = RNG.integers(0, len(blocks), size=(BOOTSTRAP, len(blocks)))
    means = blocks[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    try:
        p_value = float(wilcoxon(blocks, alternative="two-sided").pvalue)
    except ValueError:
        p_value = float("nan")
    return {
        "delta": float(blocks.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "wilcoxon_p": p_value,
        "n_blocks": int(len(blocks)),
        "n_seed_pairs": int(len(diffs)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    contract = check_contract()
    contract.to_csv(OUT / "backbone_contract_check.csv", index=False)
    print(contract.to_string(index=False))

    summary = summaries()
    transformer = transformer_summaries()
    summary.to_csv(OUT / "backbone_runs_summary.csv", index=False)

    agg = (
        summary.groupby(["city", "tag", "label"])
        .agg(
            n_seeds=("seed", "count"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            f1=("f1", "mean"),
            f1_sd=("f1", "std"),
            iou=("iou", "mean"),
        )
        .reset_index()
    )
    if not transformer.empty:
        tagg = (
            transformer.groupby(["city", "tag", "label"])
            .agg(
                n_seeds=("seed", "count"),
                precision=("precision", "mean"),
                recall=("recall", "mean"),
                f1=("f1", "mean"),
                f1_sd=("f1", "std"),
                iou=("iou", "mean"),
            )
            .reset_index()
        )
        agg = pd.concat([agg, tagg], ignore_index=True)
    agg.to_csv(OUT / "backbone_summary_v2.csv", index=False)
    print()
    print(agg.to_string(index=False))

    stats_rows = []
    for city in CITIES:
        backbone_contrasts = []
        channel_contrast = None
        for key in BACKBONES:
            if key == "convgru_6ch":
                continue
            res = paired(city, key, "convgru_6ch", SEEDS_FULL, RUNS)
            if not res:
                continue
            row = {"city": city, "tag": key, **res}
            if key == "convgru_2ch":
                channel_contrast = row
            else:
                backbone_contrasts.append(row)
        # The Transformer has ten seeds and joins the backbone family.
        if not transformer.empty and (transformer.city == city).any():
            res = paired(city, "transformer", "convgru_6ch", SEEDS_EXPLORATORY, TRANSFORMER_RUNS)
            if res:
                backbone_contrasts.append({"city": city, "tag": "transformer", **res})
        # Holm family 1: the matched backbone contrasts.  The input-channel row
        # is a separate one-comparison family, as stated in the table caption.
        adjusted = holm([row["wilcoxon_p"] for row in backbone_contrasts])
        for row, adj in zip(backbone_contrasts, adjusted):
            row["holm_p"] = adj
        if channel_contrast is not None:
            channel_contrast["holm_p"] = channel_contrast["wilcoxon_p"]
            backbone_contrasts.append(channel_contrast)
        stats_rows.extend(backbone_contrasts)
    stats = pd.DataFrame(stats_rows)
    stats.to_csv(OUT / "backbone_paired_stats_v2.csv", index=False)
    print()
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
