# -*- coding: utf-8 -*-
"""Assemble common-target caches: H3/S2/quadtree input resampled onto the GeoSOT target.

For each GeoSOT cell we take the H3/S2 cell that contains its centroid, so the
input frame lives on the same 32x32 GeoSOT raster as the target while the value
of each input pixel comes from the partition under test. This is the
common-target control: only the input partition changes.

    python papers/APIN/experiments/h3s2/build_common_target.py --partition h3
"""

from __future__ import annotations

import argparse
import json
import sys
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from geosot.geosot_encoder import GeoSOTEncoder  # noqa: E402
from partitions import area_bbox, canvas_for  # noqa: E402

GEOSOT_LEVEL = 16
CACHE = Path(__file__).resolve().parent / "frame_cache"
CITIES = ("porto", "t_drive")
SPLITS = ("train", "validation", "test")


def geosot_centers(grid_map: dict) -> tuple[dict[tuple[int, int], tuple[float, float]], int, int]:
    inv_row = {int(compact): int(native) for native, compact in grid_map["row_lookup"].items()}
    inv_col = {int(compact): int(native) for native, compact in grid_map["col_lookup"].items()}
    rows, cols = int(grid_map["grid_shape"][0]), int(grid_map["grid_shape"][1])
    centers = {}
    for row in range(rows):
        for col in range(cols):
            bbox = GeoSOTEncoder.xy_coordinates_to_bbox(inv_row[row], inv_col[col], GEOSOT_LEVEL)
            centers[(row, col)] = (
                (bbox["south"] + bbox["north"]) / 2,
                (bbox["west"] + bbox["east"]) / 2,
            )
    return centers, rows, cols


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", required=True, choices=("h3", "s2", "quadtree"))
    args = parser.parse_args()

    for city in CITIES:
        geosot_dir = CACHE / "geosot" / city
        part_dir = CACHE / args.partition / city
        grid_map = json.loads((geosot_dir / "grid_coordinate_map.json").read_text())
        bbox = area_bbox(grid_map)
        canvas = canvas_for(args.partition, grid_map, bbox)
        centers, rows, cols = geosot_centers(grid_map)

        # geosot (row, col) -> partition canvas (row, col)
        items = list(centers.items())
        center_lats = np.array([value[0] for _, value in items], dtype=np.float64)
        center_lngs = np.array([value[1] for _, value in items], dtype=np.float64)
        part_rows, part_cols = canvas.assign(center_lats, center_lngs)
        lookup: dict[int, tuple[int, int] | None] = {}
        for ((row, col), _), part_row, part_col in zip(items, part_rows, part_cols):
            lookup[row * cols + col] = (
                None if part_row < 0 or part_col < 0 else (int(part_row), int(part_col))
            )

        out_root = CACHE / "common_target" / args.partition / city
        out_root.mkdir(parents=True, exist_ok=True)
        for split in SPLITS:
            g_dir = geosot_dir / split
            p_dir = part_dir / split
            g_index = pd.read_parquet(g_dir / "frame_index.parquet")
            p_index = pd.read_parquet(p_dir / "frame_index.parquet")
            g_frames = np.load(g_dir / "frames.npy", mmap_mode="r")
            p_frames = np.load(p_dir / "frames.npy", mmap_mode="r")
            position = {timestamp: index for index, timestamp in enumerate(p_index["frame_start"])}
            missing = [ts for ts in g_index["frame_start"] if ts not in position]
            if missing:
                raise SystemExit(f"{city}/{split}: {len(missing)} target frames missing in {args.partition}")
            order = np.array([position[ts] for ts in g_index["frame_start"]], dtype=np.int64)
            aligned = np.asarray(p_frames[order])

            hybrid = np.zeros((len(g_frames), 6, rows, cols), dtype=np.float32)
            for flat, rc in lookup.items():
                if rc is None:
                    continue
                r, c = divmod(flat, cols)
                pr, pc = rc
                hybrid[:, :, r, c] = aligned[:, :, pr, pc]

            target = out_root / split
            target.mkdir(parents=True, exist_ok=True)
            np.save(target / "frames.npy", hybrid)
            np.save(target / "target_frames.npy", np.asarray(g_frames, dtype=np.float32))
            np.save(target / "sample_starts.npy", np.load(g_dir / "sample_starts.npy"))
            g_index.to_parquet(target / "frame_index.parquet", index=False)
            metadata = {
                "city": city,
                "split": split,
                "manifest_sha256": json.loads((g_dir / "metadata.json").read_text())["manifest_sha256"],
                "channel_layout": "compact",
                "frame_channels": ["occupancy", "arrivals", "departures", "mean_speed",
                                   "mean_direction_sin", "mean_direction_cos"],
                "frame_shape": list(hybrid.shape),
                "sample_count": int(len(np.load(target / "sample_starts.npy"))),
                "grid": grid_map,
                "hierarchy_contract": "disabled",
                "partition": f"{args.partition}_input_geosot_target",
            }
            (target / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            print(f"{city}/{split}: common-target {args.partition} -> geosot {hybrid.shape}", flush=True)
        (out_root / "grid_coordinate_map.json").write_text(
            json.dumps(grid_map, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
