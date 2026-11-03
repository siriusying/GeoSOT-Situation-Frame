#!/usr/bin/env python3
"""Bounded geometric diagnostic: GeoSOT vs H3 vs S2 on Porto test points.

This is not a forecasting comparison. It measures, on the same sampled points:

  * occupied-cell count and mean cell area at comparable resolutions,
  * the aspect ratio of the cell polygon (square vs hexagon vs quadrilateral),
  * the point-to-cell-centre distance, i.e. the quantization error.

Resolutions are chosen to be comparable in cell area: GeoSOT level 16,
H3 resolution 8, and S2 level 13. The diagnostic documents why the paper
compares GeoSOT with a flat grid (a forecasting contract question) rather
than claiming that GeoSOT is the best hierarchy.

    python experiments/grid_geometry_diagnostic.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import h3
import numpy as np
import pandas as pd
from s2sphere import Cell, CellId, LatLng

def _repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src" / "geosot_generative_trajectory").is_dir():
            return candidate
    raise SystemExit("run from inside the repository (src/geosot_generative_trajectory not found)")


ROOT = _repo_root(Path(__file__).resolve().parent)
OUT = Path(os.environ.get("APIN_OUT", Path(__file__).resolve().parent.parent / "artifacts" / "results"))
POINTS = ROOT / "geosot_llm_benchmark/datasets/processed/porto_taxi/porto_points_full.parquet"
H3S2 = Path(__file__).resolve().parent / "h3s2"
SAMPLE = 100_000
EARTH_RADIUS_M = 6_371_008.8

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(H3S2))
from geosot.geosot_encoder import GeoSOTEncoder  # noqa: E402
from partitions import qtree_project, qtree_unproject  # noqa: E402


def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def rectangle_area_km2(west: float, south: float, east: float, north: float) -> float:
    return (
        EARTH_RADIUS_M**2
        * math.radians(east - west)
        * (math.sin(math.radians(north)) - math.sin(math.radians(south)))
        / 1e6
    )


def aspect_ratio(points: list[tuple[float, float]]) -> float:
    lngs = [p[0] for p in points]
    lats = [p[1] for p in points]
    lat_mid = math.radians(sum(lats) / len(lats))
    width = (max(lngs) - min(lngs)) * math.cos(lat_mid)
    height = max(lats) - min(lats)
    if width == 0 or height == 0:
        return float("nan")
    return max(width, height) / min(width, height)


def main() -> None:
    frame = pd.read_parquet(POINTS, columns=["time", "lat", "lng"])
    frame = frame[(frame["time"] >= "2014-06-01") & (frame["time"] < "2014-12-22")]
    frame = frame.dropna(subset=["lat", "lng"]).sample(n=min(SAMPLE, len(frame)), random_state=20261101)
    lats = frame["lat"].to_numpy()
    lngs = frame["lng"].to_numpy()

    rows = []
    geosot_codes, geosot_err, geosot_area, geosot_ar = [], [], [], []
    h3_cells, h3_err, h3_area, h3_ar = [], [], [], []
    s2_cells, s2_err, s2_area, s2_ar = [], [], [], []

    for lat, lng in zip(lats, lngs):
        code = GeoSOTEncoder.coordinates_to_quaternary(float(lng), float(lat), 16)
        bbox = GeoSOTEncoder.quaternary_to_bbox(code)
        geosot_codes.append(code)
        geosot_area.append(
            rectangle_area_km2(bbox["west"], bbox["south"], bbox["east"], bbox["north"])
        )
        geosot_ar.append(
            aspect_ratio(
                [
                    (bbox["west"], bbox["south"]),
                    (bbox["east"], bbox["south"]),
                    (bbox["east"], bbox["north"]),
                    (bbox["west"], bbox["north"]),
                ]
            )
        )
        geosot_err.append(
            haversine_m(lat, lng, (bbox["south"] + bbox["north"]) / 2, (bbox["west"] + bbox["east"]) / 2)
        )

        cell = h3.latlng_to_cell(float(lat), float(lng), 8)
        h3_cells.append(cell)
        clat, clng = h3.cell_to_latlng(cell)
        h3_err.append(haversine_m(lat, lng, clat, clng))
        h3_area.append(h3.cell_area(cell, unit="km^2"))
        h3_ar.append(aspect_ratio([(lo, la) for la, lo in h3.cell_to_boundary(cell)]))

        cell_id = CellId.from_lat_lng(LatLng.from_degrees(float(lat), float(lng))).parent(13)
        s2_cells.append(cell_id.to_token())
        center = LatLng.from_point(Cell(cell_id).get_center())
        s2_err.append(haversine_m(lat, lng, center.lat().degrees, center.lng().degrees))
        s2_area.append(Cell(cell_id).exact_area() * EARTH_RADIUS_M**2 / 1e6)
        s2_ar.append(
            aspect_ratio(
                [
                    (LatLng.from_point(Cell(cell_id).get_vertex(i)).lng().degrees,
                     LatLng.from_point(Cell(cell_id).get_vertex(i)).lat().degrees)
                    for i in range(4)
                ]
            )
        )

    # Equal-area quadtree at the GeoSOT level-16 mean cell area. The canvas is
    # built over the sampled point extent so every point is assigned.
    sample_box = {
        "west": float(lngs.min()),
        "east": float(lngs.max()),
        "south": float(lats.min()),
        "north": float(lats.max()),
    }
    from partitions import QTREE_TARGET_AREA_KM2, QTREE_SIDE_KM

    lon0 = (sample_box["west"] + sample_box["east"]) / 2.0
    x, y = qtree_project(lats, lngs, lon0)
    edge_lngs = np.concatenate(
        [
            np.full(2, sample_box["west"]),
            np.full(2, sample_box["east"]),
            np.array([sample_box["west"], sample_box["east"]]),
            np.array([sample_box["west"], sample_box["east"]]),
        ]
    )
    edge_lats = np.array(
        [
            sample_box["south"], sample_box["north"],
            sample_box["south"], sample_box["north"],
            sample_box["south"], sample_box["north"],
            sample_box["south"], sample_box["north"],
        ]
    )
    ex, ey = qtree_project(edge_lats, edge_lngs, lon0)
    side = QTREE_SIDE_KM
    origin_x = math.floor(ex.min() / side) * side - side
    origin_y = math.floor(ey.min() / side) * side - side
    q_rows = np.floor((y - origin_y) / side).astype(np.int64)
    q_cols = np.floor((x - origin_x) / side).astype(np.int64)
    x0 = origin_x + q_cols * side
    y0 = origin_y + q_rows * side
    center_lat, center_lng = qtree_unproject(x0 + side / 2, y0 + side / 2, lon0)
    qtree_meta_rows = []
    qtree_area, qtree_ar, qtree_err = [], [], []
    for index in range(len(lats)):
        row, col = int(q_rows[index]), int(q_cols[index])
        qtree_meta_rows.append((row, col))
        qtree_area.append(QTREE_TARGET_AREA_KM2)
        qtree_err.append(
            haversine_m(float(lats[index]), float(lngs[index]),
                        float(center_lat[index]), float(center_lng[index]))
        )
        corners = []
        for dx in (0.0, side):
            for dy in (0.0, side):
                clat, clng = qtree_unproject(
                    origin_x + col * side + dx,
                    origin_y + row * side + dy,
                    lon0,
                )
                corners.append((float(clng), float(clat)))
        qtree_ar.append(aspect_ratio(corners))

    for name, cells, area, ar, err in (
        ("GeoSOT level 16", geosot_codes, geosot_area, geosot_ar, geosot_err),
        ("H3 resolution 8", h3_cells, h3_area, h3_ar, h3_err),
        ("S2 level 13", s2_cells, s2_area, s2_ar, s2_err),
        ("Equal-area quadtree", qtree_meta_rows, qtree_area, qtree_ar, qtree_err),
    ):
        rows.append(
            {
                "partition": name,
                "points": len(cells),
                "occupied_cells": len(set(cells)),
                "points_per_cell": len(cells) / max(1, len(set(cells))),
                "mean_cell_area_km2": float(np.mean(area)),
                "sd_cell_area_km2": float(np.std(area)),
                "mean_aspect_ratio": float(np.nanmean(ar)),
                "mean_quantization_error_m": float(np.mean(err)),
                "p95_quantization_error_m": float(np.percentile(err, 95)),
            }
        )

    table = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "grid_geometry_diagnostic.csv", index=False)
    with pd.option_context("display.width", 200):
        print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
