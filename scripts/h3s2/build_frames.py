# -*- coding: utf-8 -*-
"""Build compact frame caches for GeoSOT, H3 and S2 partitions.

The six channels, split intervals, guard bands and window starts are produced by
the same code path as the published compact cache. Only the partition that maps
points to (row, col) changes, so the model contract stays identical.

    python papers/APIN/experiments/h3s2/build_frames.py --city porto --partition h3
    python papers/APIN/experiments/h3s2/build_frames.py --city porto --partition geosot \
        --output-root /tmp/geosot_rebuild

Outputs mirror the compact frame-cache layout:
``<output-root>/<city>/<split>/{frames.npy,frame_index.parquet,sample_starts.npy,metadata.json}``
and ``<output-root>/<city>/grid_coordinate_map.json``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

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

from src.geosot_generative_trajectory.data.cross_city_taxi import (  # noqa: E402
    _effective_split_bounds,
    valid_window_starts,
)
from src.geosot_generative_trajectory.data.frame_builder import (  # noqa: E402
    build_frame_tensor_sequence,
)
from geosot.geosot_encoder import GeoSOTEncoder  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from partitions import (  # noqa: E402
    GEOSOT_LEVEL,
    area_bbox,
    canvas_for,
)

BASE = ROOT / "papers/paper_geosot_generative_trajectory_kbs/elsarticle/audit_artifacts/representation_benchmark_v1/paradigm_confirmatory_v1"
MANIFESTS = BASE / "manifests"
COMPACT = BASE / "compact_frame_cache"


def load_manifest(city: str) -> dict:
    return json.loads((MANIFESTS / f"{city}_compact_manifest.json").read_text())


def load_points(point_path: str, bounds: tuple, bbox: dict, time_col: str, trajectory_col: str) -> pd.DataFrame:
    start, end = bounds
    dataset = ds.dataset(point_path, format="parquet")
    filters = [
        (pc.field("lat") >= bbox["south"]),
        (pc.field("lat") <= bbox["north"]),
        (pc.field("lng") >= bbox["west"]),
        (pc.field("lng") <= bbox["east"]),
    ]
    if start is not None:
        filters.append(pc.field(time_col) >= start.to_pydatetime())
    if end is not None:
        filters.append(pc.field(time_col) < end.to_pydatetime())
    expression = filters[0]
    for clause in filters[1:]:
        expression = expression & clause
    table = dataset.to_table(columns=[time_col, "lat", "lng", trajectory_col], filter=expression)
    return table.to_pandas()


def assign_geosot(points: pd.DataFrame, grid_map: dict) -> tuple[np.ndarray, np.ndarray]:
    rows = np.full(len(points), -1, dtype=np.int64)
    cols = np.full(len(points), -1, dtype=np.int64)
    cache: dict[tuple[float, float], tuple[int, int]] = {}
    lats = points["lat"].to_numpy()
    lngs = points["lng"].to_numpy()
    for index, (lat, lng) in enumerate(zip(lats, lngs)):
        key = (round(float(lng), 7), round(float(lat), 7))
        hit = cache.get(key)
        if hit is None:
            xy = GeoSOTEncoder.coordinates_to_xy_coordinates(float(lng), float(lat), GEOSOT_LEVEL)
            hit = (int(xy["x"]), int(xy["y"]))
            cache[key] = hit
        rows[index], cols[index] = hit
    return rows, cols


def identity_map(canvas) -> dict:
    return {
        "level": None,
        "row_lookup": {index: index for index in range(canvas.rows)},
        "col_lookup": {index: index for index in range(canvas.cols)},
        "grid_shape": (canvas.rows, canvas.cols),
    }


def build_split(
    city: str,
    partition: str,
    split: str,
    manifest: dict,
    grid_map: dict,
    canvas,
    output_root: Path,
) -> None:
    contract = manifest["contract"]
    point_path = contract["point_path"]
    time_col = contract.get("time_column", "time")
    trajectory_col = contract.get("trajectory_column", "trajectory_id")
    bounds = _effective_split_bounds(manifest, split)
    bbox = area_bbox(grid_map)
    points = load_points(point_path, bounds, bbox, time_col, trajectory_col)
    if points.empty:
        raise SystemExit(f"{city}/{split}: no points in the modeling area")

    if partition == "geosot":
        rows, cols = assign_geosot(points, grid_map)
        fixed_map = grid_map
        row_keys = {int(key) for key in grid_map["row_lookup"]}
        col_keys = {int(key) for key in grid_map["col_lookup"]}
        valid = np.isin(rows, list(row_keys)) & np.isin(cols, list(col_keys))
    else:
        rows, cols = canvas.assign(points["lat"].to_numpy(), points["lng"].to_numpy())
        fixed_map = identity_map(canvas)
        valid = (rows >= 0) & (cols >= 0)
    points = points.loc[valid].copy()
    points["grid_x"] = rows[valid]
    points["grid_y"] = cols[valid]
    if points.empty:
        raise SystemExit(f"{city}/{split}: no points map to the {partition} canvas")

    frames, frame_index = build_frame_tensor_sequence(
        points,
        frame_minutes=int(contract["frame_minutes"]),
        levels=(GEOSOT_LEVEL,),
        primary_level=GEOSOT_LEVEL,
        fixed_grid_coordinate_map=fixed_map,
        channel_layout=str(contract.get("channel_layout", "compact")),
        source_dataset=city,
        time_col=time_col,
        lat_col="lat",
        lng_col="lng",
        trajectory_col=trajectory_col,
    )
    starts = valid_window_starts(
        frame_index["frame_start"],
        history_window=int(contract["history_window"]),
        future_window=int(contract["future_window"]),
        frame_minutes=int(contract["frame_minutes"]),
        start=bounds[0],
        end=bounds[1],
    )

    target = output_root / city / split
    target.mkdir(parents=True, exist_ok=True)
    np.save(target / "frames.npy", np.asarray(frames, dtype=np.float32))
    np.save(target / "sample_starts.npy", np.asarray(starts, dtype=np.int64))
    frame_index.to_parquet(target / "frame_index.parquet", index=False)
    metadata = {
        "city": city,
        "split": split,
        "manifest_sha256": manifest["manifest_sha256"],
        "channel_layout": str(contract.get("channel_layout", "compact")),
        "frame_channels": list(frame_index.attrs.get("frame_channels", [])),
        "frame_shape": list(np.asarray(frames).shape),
        "sample_count": int(len(starts)),
        "grid": grid_map if partition == "geosot" else {
            "level": None,
            "grid_shape": [canvas.rows, canvas.cols],
            "schema_version": 1,
            "partition": canvas.meta,
        },
        "hierarchy_contract": "geosot_native" if partition == "geosot" else "disabled",
        "partition": canvas.meta,
    }
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"{city}/{split}/{partition}: frames={metadata['frame_shape']} starts={len(starts)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True, choices=("porto", "t_drive"))
    parser.add_argument("--partition", required=True, choices=("geosot", "h3", "s2", "quadtree"))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--splits", default="train,validation,test")
    args = parser.parse_args()

    manifest = load_manifest(args.city)
    grid_map = json.loads((COMPACT / args.city / "grid_coordinate_map.json").read_text())
    bbox = area_bbox(grid_map)
    canvas = canvas_for(args.partition, grid_map, bbox)
    output_root = args.output_root or (ROOT / "papers/APIN/experiments/h3s2/frame_cache" / args.partition)
    output_root = Path(output_root)
    (output_root / args.city).mkdir(parents=True, exist_ok=True)
    map_payload = grid_map if args.partition == "geosot" else {
        "level": None,
        "grid_shape": [canvas.rows, canvas.cols],
        "row_lookup": {str(i): i for i in range(canvas.rows)},
        "col_lookup": {str(i): i for i in range(canvas.cols)},
        "schema_version": 1,
        "partition": canvas.meta,
    }
    (output_root / args.city / "grid_coordinate_map.json").write_text(
        json.dumps(map_payload, indent=2, sort_keys=True) + "\n"
    )
    for split in args.splits.split(","):
        build_split(args.city, args.partition, split.strip(), manifest, grid_map, canvas, output_root)


if __name__ == "__main__":
    main()
