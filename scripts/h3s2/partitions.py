# -*- coding: utf-8 -*-
"""Partition canvases (GeoSOT, H3, S2, equal-area quadtree) for the parity study.

Each canvas maps a native partition cell to a contiguous (row, col) index in a
frame tensor. The GeoSOT canvas reuses the existing 32x32 compact grid; H3 and
S2 canvases use their own local lattice; the quadtree canvas is a regular grid
of equal-area cells on a sinusoidal projection, so no two cells collide.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import h3
import numpy as np
from s2sphere import CellId, LatLng

import sys

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
from geosot.geosot_encoder import GeoSOTEncoder  # noqa: E402

GEOSOT_LEVEL = 16
H3_RES = 8
S2_LEVEL = 13

# Equal-area quadtree control -------------------------------------------------
# The target is the mean GeoSOT level-16 cell area over occupied Porto cells
# (grid_geometry_diagnostic.csv). The quadtree leaves are squares on a
# sinusoidal equal-area projection, so every cell has exactly this area on the
# sphere used by the other diagnostics (R = 6371.0088 km).
EARTH_RADIUS_KM = 6371.0088
QTREE_TARGET_AREA_KM2 = 0.6446405477
QTREE_SIDE_KM = math.sqrt(QTREE_TARGET_AREA_KM2)


@dataclass
class Canvas:
    name: str
    rows: int
    cols: int
    mapping: dict[str, tuple[int, int]]
    meta: dict

    def assign(self, lats: np.ndarray, lngs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map points to (row, col); missing cells yield -1."""
        rows = np.full(len(lats), -1, dtype=np.int64)
        cols = np.full(len(lats), -1, dtype=np.int64)
        if self.name == "quadtree":
            x, y = qtree_project(lats, lngs, self.meta["lon0"])
            side = self.meta["cell_side_km"]
            row = np.floor((y - self.meta["origin_y_km"]) / side).astype(np.int64)
            col = np.floor((x - self.meta["origin_x_km"]) / side).astype(np.int64)
            inside = (
                (row >= 0) & (row < self.rows) & (col >= 0) & (col < self.cols)
            )
            rows = np.where(inside, row, -1)
            cols = np.where(inside, col, -1)
            return rows, cols
        if self.name == "geosot":
            cache: dict[tuple[int, int], tuple[int, int]] = {}
            for index, (lat, lng) in enumerate(zip(lats, lngs)):
                key = (round(float(lng), 7), round(float(lat), 7))
                hit = cache.get(key)
                if hit is None:
                    xy = GeoSOTEncoder.coordinates_to_xy_coordinates(float(lng), float(lat), GEOSOT_LEVEL)
                    hit = self.mapping.get(f"{int(xy['x'])}:{int(xy['y'])}", (-1, -1))
                    cache[key] = hit
                rows[index], cols[index] = hit
            return rows, cols
        for index, (lat, lng) in enumerate(zip(lats, lngs)):
            key = cell_key(self.name, float(lat), float(lng))
            hit = self.mapping.get(key)
            if hit is not None:
                rows[index], cols[index] = hit
        return rows, cols


def area_bbox(grid_map: dict) -> dict[str, float]:
    """Decode the existing 32x32 GeoSOT level-16 grid into a lon/lat box."""
    rows = sorted(int(key) for key in grid_map["row_lookup"])
    cols = sorted(int(key) for key in grid_map["col_lookup"])
    a = GeoSOTEncoder.xy_coordinates_to_bbox(rows[0], cols[0], GEOSOT_LEVEL)
    b = GeoSOTEncoder.xy_coordinates_to_bbox(rows[-1], cols[-1], GEOSOT_LEVEL)
    return {
        "west": min(a["west"], b["west"]),
        "east": max(a["east"], b["east"]),
        "south": min(a["south"], b["south"]),
        "north": max(a["north"], b["north"]),
    }


def cell_key(partition: str, lat: float, lng: float) -> str:
    if partition == "h3":
        return h3.latlng_to_cell(lat, lng, H3_RES)
    if partition == "s2":
        return str(CellId.from_lat_lng(LatLng.from_degrees(lat, lng)).parent(S2_LEVEL).id())
    raise ValueError(partition)


def geosot_canvas(grid_map: dict) -> Canvas:
    mapping = {}
    for native_row, compact_row in grid_map["row_lookup"].items():
        for native_col, compact_col in grid_map["col_lookup"].items():
            mapping[f"{int(native_row)}:{int(native_col)}"] = (int(compact_row), int(compact_col))
    rows, cols = int(grid_map["grid_shape"][0]), int(grid_map["grid_shape"][1])
    return Canvas("geosot", rows, cols, mapping, {"type": "geosot", "level": GEOSOT_LEVEL})


def h3_canvas(bbox: dict[str, float]) -> Canvas:
    polygon = h3.LatLngPoly(
        [
            (bbox["south"], bbox["west"]),
            (bbox["south"], bbox["east"]),
            (bbox["north"], bbox["east"]),
            (bbox["north"], bbox["west"]),
        ]
    )
    cells = sorted(h3.polygon_to_cells(polygon, H3_RES))
    origin = h3.latlng_to_cell(
        (bbox["south"] + bbox["north"]) / 2, (bbox["west"] + bbox["east"]) / 2, H3_RES
    )
    coords: dict[str, tuple[int, int]] = {}
    for cell in cells:
        try:
            coords[cell] = h3.cell_to_local_ij(origin, cell)
        except Exception:  # pentagon distortion; the city area never hits it
            continue
    min_i = min(i for i, _ in coords.values())
    min_j = min(j for _, j in coords.values())
    mapping = {cell: (i - min_i, j - min_j) for cell, (i, j) in coords.items()}
    rows = max(r for r, _ in mapping.values()) + 1
    cols = max(c for _, c in mapping.values()) + 1
    return Canvas(
        "h3",
        rows,
        cols,
        mapping,
        {"type": "h3", "resolution": H3_RES, "cells": len(mapping), "origin": origin},
    )


def s2_canvas(bbox: dict[str, float], step_deg: float = 0.004) -> Canvas:
    """Cover the box by sampling, then lay cells on their local face-ij lattice."""
    lat_values = np.arange(bbox["south"], bbox["north"] + step_deg, step_deg)
    lng_values = np.arange(bbox["west"], bbox["east"] + step_deg, step_deg)
    faces: set[int] = set()
    coords: dict[str, tuple[int, int]] = {}
    for lat in lat_values:
        for lng in lng_values:
            cid = CellId.from_lat_lng(LatLng.from_degrees(float(lat), float(lng))).parent(S2_LEVEL)
            face, i, j, _ = cid.to_face_ij_orientation()
            shift = 30 - S2_LEVEL
            faces.add(face)
            coords[str(cid.id())] = (i >> shift, j >> shift)
    if len(faces) != 1:
        raise RuntimeError(f"S2 box spans multiple faces: {sorted(faces)}")
    min_i = min(i for i, _ in coords.values())
    min_j = min(j for _, j in coords.values())
    mapping = {token: (i - min_i, j - min_j) for token, (i, j) in coords.items()}
    rows = max(r for r, _ in mapping.values()) + 1
    cols = max(c for _, c in mapping.values()) + 1
    return Canvas(
        "s2",
        rows,
        cols,
        mapping,
        {"type": "s2", "level": S2_LEVEL, "cells": len(mapping), "face": sorted(faces)[0]},
    )


def qtree_project(lats, lngs, lon0: float):
    """Sinusoidal equal-area projection to local kilometres."""
    lat = np.radians(np.asarray(lats, dtype=np.float64))
    lng = np.radians(np.asarray(lngs, dtype=np.float64))
    x = EARTH_RADIUS_KM * (lng - math.radians(lon0)) * np.cos(lat)
    y = EARTH_RADIUS_KM * lat
    return x, y


def qtree_unproject(x, y, lon0: float):
    """Inverse of qtree_project, returning (lat, lng) in degrees."""
    lat = np.degrees(np.asarray(y, dtype=np.float64) / EARTH_RADIUS_KM)
    lng = lon0 + np.degrees(
        np.asarray(x, dtype=np.float64) / (EARTH_RADIUS_KM * np.cos(np.radians(lat)))
    )
    return lat, lng


def qtree_canvas(bbox: dict[str, float]) -> Canvas:
    """Equal-area quadtree leaf grid covering the GeoSOT box."""
    lon0 = (bbox["west"] + bbox["east"]) / 2.0
    n = 512
    edge_lngs = np.concatenate(
        [
            bbox["west"] + (bbox["east"] - bbox["west"]) * np.linspace(0, 1, n),
            bbox["west"] + (bbox["east"] - bbox["west"]) * np.linspace(0, 1, n),
            np.full(n, bbox["west"]),
            np.full(n, bbox["east"]),
        ]
    )
    edge_lats = np.concatenate(
        [
            np.full(n, bbox["south"]),
            np.full(n, bbox["north"]),
            bbox["south"] + (bbox["north"] - bbox["south"]) * np.linspace(0, 1, n),
            bbox["south"] + (bbox["north"] - bbox["south"]) * np.linspace(0, 1, n),
        ]
    )
    x, y = qtree_project(edge_lats, edge_lngs, lon0)
    side = QTREE_SIDE_KM
    origin_x = math.floor(x.min() / side) * side - side
    origin_y = math.floor(y.min() / side) * side - side
    cols = int(math.ceil((x.max() - origin_x) / side)) + 1
    rows = int(math.ceil((y.max() - origin_y) / side)) + 1
    mapping = {(r, c): (r, c) for r in range(rows) for c in range(cols)}
    meta = {
        "type": "quadtree",
        "projection": "sinusoidal_equal_area",
        "earth_radius_km": EARTH_RADIUS_KM,
        "cell_side_km": side,
        "cell_area_km2": QTREE_TARGET_AREA_KM2,
        "lon0": lon0,
        "origin_x_km": origin_x,
        "origin_y_km": origin_y,
        "cells": rows * cols,
    }
    return Canvas("quadtree", rows, cols, mapping, meta)


def canvas_for(partition: str, grid_map: dict, bbox: dict) -> Canvas:
    if partition == "geosot":
        return geosot_canvas(grid_map)
    if partition == "h3":
        return h3_canvas(bbox)
    if partition == "s2":
        return s2_canvas(bbox)
    if partition == "quadtree":
        return qtree_canvas(bbox)
    raise ValueError(partition)
