#!/usr/bin/env python3
"""Render the partition-control table body for the manuscript.

Reads ``results/partition_native_summary.csv`` (produced by analyze.py) and
writes a LaTeX fragment with native ten-seed F1, paired differences against
GeoSOT, and the three-seed common-target control.

    python papers/APIN/experiments/h3s2/make_table.py
"""

from __future__ import annotations

import os
from pathlib import Path

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
OUT = Path(os.environ.get("APIN_OUT", ROOT / "papers/APIN/experiments/h3s2/results"))
CITIES = ("porto", "t_drive")
LABELS = {"porto": "Porto", "t_drive": "T-Drive"}
PARTS = (("geosot", "GeoSOT-ST"), ("h3", "H3 r8"), ("s2", "S2 L13"), ("quadtree", "Quadtree EA"))


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}".lstrip("0")


def signed(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}"


def main() -> None:
    table = pd.read_csv(OUT / "partition_native_summary.csv")
    index = {(row.city, row.partition): row for row in table.itertuples()}
    lines = []
    for city in CITIES:
        for position, (part, label) in enumerate(PARTS):
            row = index.get((city, part))
            if row is None:
                continue
            prefix = "" if position else f"{LABELS[city]} & "
            native = fmt(row.f1)
            if part == "geosot":
                delta = "--"
                common = "--"
                cdelta = "--"
            else:
                diff = index.get((city, f"{part}_minus_geosot"))
                if diff is not None and pd.notna(getattr(diff, "ci_low", None)):
                    mark = "$\\dagger$" if bool(diff.equivalent) else ""
                    delta = f"{signed(diff.f1)} [{diff.ci_low:+.4f},{diff.ci_high:+.4f}]{mark}"
                else:
                    delta = "--"
                c = index.get((city, f"common_{part}"))
                cd = index.get((city, f"common_{part}_minus_geosot"))
                common = fmt(c.f1) if c is not None else "--"
                if cd is not None and pd.notna(getattr(cd, "ci_low", None)):
                    cdelta = f"{signed(cd.f1)} [{cd.ci_low:+.4f},{cd.ci_high:+.4f}]"
                else:
                    cdelta = "--"
            if position == 0:
                lines.append(
                    f"{prefix}{label} & {native} & {delta} & {common} & {cdelta} \\\\"
                )
            else:
                lines.append(f" & {label} & {native} & {delta} & {common} & {cdelta} \\\\")
        if city != CITIES[-1]:
            lines.append("\\midrule")
    path = OUT / "partition_table_rows.tex"
    path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
