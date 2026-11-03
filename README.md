# GeoSOT-ST reproducibility package

Code and artifact map for *GeoSOT-ST: A Hierarchical Frame Representation for
Neural Spatiotemporal Forecasting of Fleet Activity* (Applied Intelligence
submission).

## 1. Environment

Python 3.13, single CUDA GPU (the reported runs used an RTX 5000 Ada, 32 GB).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd <repository root>
export PYTHONPATH=$PWD
```

All commands below run from the repository root. In this document `scripts/`
refers to `papers/APIN/release/scripts/`, and `artifacts/` to
`papers/APIN/release/artifacts/`.

## 2. Data

All datasets are public and are **not** redistributed.

| Dataset | Source | Used for |
|---|---|---|
| Porto Taxi | UCI / Kaggle Porto Taxi Trajectory | prediction object, representation, backbones, dispatch |
| T-Drive Taxi | Microsoft Research T-Drive | replication |
| GeoLife | Microsoft Research GeoLife 1.3 | low-shot diagnostic |

Shipped with this package:

- `artifacts/manifests/` — the two compact manifests (Porto, T-Drive) that fix
  the source points, splits, and window starts.
- `artifacts/results/` — the aggregate tables and summaries behind every
  reported number: backbones, dispatch, entity-support ceiling, density strata,
  model-free baselines, robustness checks, grid geometry, canonical-union and
  factorial statistics, and the native H3/S2 partition summary.
- `artifacts/figures/` — the released figure files.

Not shipped, with the way to obtain it:

- **Raw taxi records.** Public sources above.
- **Compact frame caches (GB scale).** Rebuild with
  `scripts/h3s2/build_frames.py`, which materialises all partitions with one
  builder. The published GeoSOT compact cache used for the backbone and
  representation runs was built from an upstream prepared point cache
  (`prepared_v2`) that is no longer distributed. Rebuilding preserves the
  occupancy channel exactly but may change the auxiliary arrival/departure
  counts, so re-runs of the auxiliary channels can differ slightly from the
  stored numbers.
- **Per-seed run directories and prediction arrays.** Produced by the training
  commands; the `.npz` arrays are large and are not included.
- **Entity-branch training.** It additionally requires `prepared_v2`. The
  aggregated entity predictions behind Tables 1 and 3 are included in
  `artifacts/results/`, but the entity runs cannot be re-executed from this
  package alone.

## 3. Reported runs

All compact runs share one contract: 15-minute windows, 8 history steps, 4
future steps, 32×32 level-16 frame, 48 hidden units, learning rate 5e-4, AdamW
(weight decay 1e-4, gradient clipping 1.0), seeds 41–50.

| Experiment | Entry point | Config | Artifact |
|---|---|---|---|
| Frame models (prediction object, backbones, Transformer) | `python -m src.geosot_generative_trajectory.train.cross_city_runner` | `configs/<city>_<backbone>.json` | `artifacts/results/backbone_*.csv` |
| Entity models (prediction object) | `python -m src.geosot_generative_trajectory.train.raw_entity_runner` | not shipped; see §2 | aggregated rows in `artifacts/results/` |
| Backbone table, ΔF1, Holm p | `scripts/backbone_table_v2.py` | — | `artifacts/results/backbone_summary_v2.csv`, `backbone_paired_stats_v2.csv` |
| Budgeted dispatch case and its figure | `scripts/application_dispatch_case.py` | — | `artifacts/results/application_dispatch_*.csv`, `artifacts/figures/fig_dispatch_coverage.pdf` |
| Entity support ceiling and density strata | `scripts/entity_support_and_density.py` | — | `artifacts/results/entity_support_ceiling.csv`, `density_stratification.csv` |
| Model-free temporal baselines | `scripts/naive_baselines.py` | — | `artifacts/results/naive_baselines_*.csv` |
| Robustness checks | `scripts/robustness_checks.py` | — | `artifacts/results/robustness_checks.csv` |
| Grid geometry diagnostic | `scripts/grid_geometry_diagnostic.py` | — | `artifacts/results/grid_geometry_diagnostic.csv` |
| Canonical native-frame figures | `scripts/render_canonical_union_figures_apin.py` | — | `artifacts/figures/canonical_union_*.pdf` |
| Design-lever range figure | `scripts/make_design_lever_figure.py` | — | `artifacts/figures/design_lever_range.pdf` |
| Native partition controls (H3, S2, equal-area quadtree) | `scripts/h3s2/build_frames.py` → `scripts/h3s2/run_native.sh` → `scripts/h3s2/analyze.py` | per-run `spec.json` written by the runner | `artifacts/results/partition_native_summary.csv`, `partition_table_rows.tex` |
| Quadtree alignment ablation (T-Drive) | `scripts/h3s2/run_alignment.sh` → `scripts/h3s2/analyze_alignment.py` | per-variant `spec.json` | `artifacts/results/alignment_ablation.csv` |

`scripts/run_convlstm_supplement.sh` and `scripts/run_transformer_supplement.sh`
reproduce the supplementary backbones behind the matched backbone table
(`tab:strong_backbones` in the manuscript). Both read `ROOT`, `PY`,
`PARADIGM_BASE` (the `paradigm_confirmatory_v1` directory) and `RUNS_OUT`
(where per-seed run folders are written) from the environment.

`scripts/h3s2/` reproduces the native partition controls
(`tab:partition_controls` in the manuscript). `partitions.py` builds the H3
resolution-8 and S2 level-13 lattices, the equal-area quadtree, and the point
assignment; the quadtree uses a sinusoidal equal-area projection with cell
side 0.8029 km, so every cell has the GeoSOT mean area 0.6446 km² exactly.
`build_frames.py`
writes the six-channel frame cache for each partition; `build_common_target.py`
maps the H3/S2 inputs onto the GeoSOT 32×32 target for the GeoSOT-target
resampling control; `run_native.sh` and `run_common_target.sh` drive the GPU
queues; `analyze.py` performs the paired temporal-block bootstrap and the
equivalence decision at margin 0.01; `make_table.py` renders the table body.
The run scripts take `ROOT`, `PY`, `PARADIGM_BASE`, `FRAME_CACHE` and `RUNS_OUT`
from the environment, and `analyze.py` exits with an error rather than writing a
partial summary when run directories are missing.
The alignment ablation (`run_alignment.sh`) retrains the quadtree at three
half-cell shifts and at 15, 30, and 45 degree rotations about the city centre,
holding cell size fixed; `analyze_alignment.py` reports the ten-seed F1 range.

Analysis scripts write to the package artifact directory by default; override
with `APIN_OUT` (tables), `APIN_CANONICAL` (canonical-union statistics), or
`APIN_FIGURES` (figures).

### Reproduce

```bash
# one frame run (Porto, six-channel ConvGRU, seed 41)
python -m src.geosot_generative_trajectory.train.cross_city_runner \
  --spec papers/APIN/release/configs/porto_convgru_6ch.json \
  --output-dir runs/porto_convgru_seed41

# backbone statistics and the LaTeX fragment (`tab:strong_backbones`)
APIN_OUT=$PWD/papers/APIN/release/artifacts/results \
  python papers/APIN/release/scripts/backbone_table_v2.py

# dispatch case, support ceiling, model-free baselines, robustness, geometry
python papers/APIN/release/scripts/application_dispatch_case.py
python papers/APIN/release/scripts/entity_support_and_density.py
python papers/APIN/release/scripts/naive_baselines.py
python papers/APIN/release/scripts/robustness_checks.py
python papers/APIN/release/scripts/grid_geometry_diagnostic.py

# canonical native-frame figures
python papers/APIN/release/scripts/render_canonical_union_figures_apin.py

# native H3/S2 partition controls (`tab:partition_controls`)
export PARADIGM_BASE=<...>/representation_benchmark_v1/paradigm_confirmatory_v1
export FRAME_CACHE=$PWD/papers/APIN/experiments/h3s2/frame_cache
python papers/APIN/release/scripts/h3s2/build_frames.py --city porto --partition h3 \
  --output-root "$FRAME_CACHE/h3"   # repeat for s2, quadtree, and t_drive
WORKER=0 bash papers/APIN/release/scripts/h3s2/run_native.sh
WORKER=1 bash papers/APIN/release/scripts/h3s2/run_native.sh
bash papers/APIN/release/scripts/h3s2/run_common_target.sh
python -I papers/APIN/release/scripts/h3s2/analyze.py --margin 0.01
python -I papers/APIN/release/scripts/h3s2/make_table.py

# alignment ablation (T-Drive, seven variants x ten seeds)
python papers/APIN/release/scripts/h3s2/analyze_alignment.py
```

Expected runtime: ~15-30 min per Porto seed on one RTX 5000 Ada and ~30 s per
T-Drive seed (240 test origins); the three-alternative native queue is 60 runs,
plus the rebuilt GeoSOT baseline (20) and 18 GeoSOT-target resampling runs.

## 4. Seeds and budget

Seeds 41–50 for all ten-seed contrasts, including the Transformer. The two
target controls use three seeds (41–43): the shared RAW target
(`tab:raw_grid_geosot`) and the GeoSOT-target resampling control
(`tab:partition_controls`). The canonical native-frame union and its factorial
replication use a 24-epoch budget; the compact frame, entity, and backbone runs
use 8 epochs. The paper states both.

## 5. Artifact index

`artifacts/NUMBERS_LEDGER.md` maps every number quoted in the paper to the file
and field it comes from. `artifacts/results/` contains aggregate and selected
per-window tables; per-seed prediction arrays are not included because of size.
No raw taxi records or restricted provider data are shipped.

## 6. License and citation

Code: MIT; derived artifacts: CC-BY-4.0 (see `LICENSE`). Cite the paper; the public repository
and archive DOI accompany the accepted version.
