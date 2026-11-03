# Numbers ledger — Applied Intelligence submission

Every headline number in `APIN-1101/main.tex` is traced here to the
artifact it comes from. Status values: **ok** (re-derived from the artifact),
**fixed** (paper changed to match the artifact during this revision), **note**
(definition or slice caveat that a reviewer could query).

Paths are relative to `papers/paper_geosot_generative_trajectory_kbs/` unless
marked `APIN/` (relative to `papers/APIN/`).

## 1. Prediction object

| Claim | Value | Source | Field | Status |
|---|---|---|---|---|
| Frame F1 Porto / T-Drive (185 shared origins) | .7828 / .9309 | `0911/direct_grid_matched_audit_20260916/AUDIT.json` | `<city>.summary.frame_geosot_partition.f1` | ok |
| Uniform-grid direct predictor F1 | .4416 / .8606 | same | `uniform_direct_grid.f1` | note: the uniform partition leaves 8 of 240 T-Drive windows without an active cell, so its T-Drive mean is over 232 active windows (`uniform_direct_grid.n_event`); the table caption says so |
| Aggregated entity F1 (shared origins) | .0473 / .6296 | same | `aggregated_entity.f1` | ok |
| Frame − entity ΔF1 | +0.73 / +0.30 | same | difference of the above | ok |
| Duplicate cell mass (Porto) | .1813 | `0911/prediction_object_support_audit_20260916/AUDIT.json` | `per_city.porto.duplicate_cell_mass_ratio` | ok |
| Entity support cell-steps | 15.1 / 1,296.6 | same | `entity_support_cell_steps` | ok |
| True support cell-steps | 526.9 / 2,432.6 | same | `true_support_cell_steps` | ok |
| Frame support cell-steps | 490.3 / 2,475.7 | same | `frame_support_cell_steps` | ok |
| Frame support-matched F1 | .0603 / .6850 | same | `frame_support_matched.f1` | ok |
| 3×3 / 5×5 rasterization F1 | .2576/.4432 (Porto), .7909/.8231 (T-Drive) | same | `entity_radius1/2.f1` | ok |
| Entity recall ceiling | 2.85% / 3.19% / 53.0% / 53.5% | `APIN/experiments/results/entity_support_ceiling.csv` | `mean_recall_ceiling` | note: per-window mean; the ratio-of-means in the support audit gives 2.87% / 53.3%, hence the paper quotes the per-window value |
| Observed/ceiling | 0.84 / 0.80 / 0.88 / 0.88 | same | `mean_ceiling_ratio` | ok |
| Support table frame row (Porto) | .8102/.7573/.7816/.6435 | `0911/prediction_object_support_audit_20260916/AUDIT.json` | `per_city.porto.frame_threshold_0.5` | fixed: table previously mixed the 185-origin frame row with full-whitelist entity rows |
| Density strata F1 | .765/.798 (Porto frame), .044/.053 (GRU), .916/.946, .600/.659 | `APIN/experiments/results/density_stratification.csv` | `mean_f1` | ok |

## 2. Representation

| Claim | Value | Source | Field | Status |
|---|---|---|---|---|
| Canonical union report counts | 1,650 + 1,650 | `elsarticle/audit_artifacts/canonical_union_v1/CANONICAL_UNION_AUDIT.json` | `paired_windows.paired_keys`, `cohort_inventory.total_reports` | ok |
| Native-frame ΔF1 by stratum | .002979 Porto, .006666 T-Drive .10, .009678 T-Drive 1.0, .090596 / .091046 GeoLife | `elsarticle/audit_artifacts/canonical_union_v1/canonical_primary_summary.csv` | `geosot_advantage` (metric `occupancy_f1`) | ok |
| Relative gains | 0.41%, 0.74%, 1.04%, 85.5%, 73.4% | same | `relative_advantage_percent` | ok |
| Within-dataset source-fraction contrast | T-Drive +.006666 (10%) vs +.009678 (100%); GeoLife +.090596 vs +.091046 | `canonical_primary_summary.csv` | `geosot_advantage` | note: the cited range is a cross-dataset ordering, not a within-dataset scarcity effect |
| N_CI+ / N_Holm+ | 28/36 and 18/36 (Porto); 36/36 and 0/36 (T-Drive) | same | `bootstrap_ci_positive`, `holm_positive` | ok |
| Factorial representation effect | +.003088 [.001910,.004272], Holm 3.1e-5 | `elsarticle/audit_artifacts/hierarchy_factorial_v1/statistical_inference/factorial_paired_block_bootstrap_wilcoxon_holm.csv` | `absolute_advantage`, `holm_p_one_sided` | ok |
| Factorial hierarchy effect | −.000286, Holm 1.000 | same | row `geosot_hierarchy_effect` | ok |
| Shared-target control | −.3366 Porto, −.2042 T-Drive | (paper Table `tab:raw_grid_geosot`; three-seed diagnostic) | — | note: three seeds, labelled as such in the caption |

### Native partition controls (H3, S2, equal-area quadtree)

Re-derived by `APIN/experiments/audit_manuscript_numbers.py` from
`APIN/experiments/h3s2/results/partition_native_summary.csv` (regenerate with
`python -I APIN/experiments/h3s2/analyze.py --margin 0.01`).

| Claim | Value | Source | Field | Status |
|---|---|---|---|---|
| Native partition F1, Porto / T-Drive (10 seeds) | .7831/.9345 GeoSOT, .7870/.9037 H3 r8, .7933/.9319 S2 L13, .7814/.9084 quadtree | `APIN/experiments/h3s2/results/partition_native_summary.csv` | `f1` | ok |
| ΔF1 versus reconstructed GeoSOT | +.0049 [.0036,.0062] / −.0309 [−.0325,−.0289] (H3); +.0122 [.0100,.0146] / −.0022 [−.0044,+.0004] (S2); −.0023 [−.0035,−.0011] / −.0260 [−.0276,−.0241] (quadtree) | same | `partition` = `<p>_minus_geosot`; `f1`, `ci_low`, `ci_high` | ok |
| Equivalence verdict (TOST-style, margin .01) | H3 Porto, S2 T-Drive, and quadtree Porto equivalent; the other three differ | same | `equivalent` | note: 95% temporal-block bootstrap interval contained in the pre-specified \|Δ\|≤.01 margin |
| Common-target control F1 (3 seeds) | .7794/.8760 (H3), .7793/.8869 (S2), .7797/.8792 (quadtree) | same | `partition` = `common_<p>`; `f1` | ok |
| Common-target Δ versus GeoSOT | −.0039 [−.0045,−.0034] / −.0605 [−.0718,−.0504] (H3); −.0041 [−.0047,−.0036] / −.0486 [−.0555,−.0426] (S2); −.0037 [−.0043,−.0031] / −.0571 [−.0658,−.0491] (quadtree) | same | `partition` = `common_<p>_minus_geosot` | ok |
| Cell areas | .64 / .80 / 1.02 / .645 km² (GeoSOT L16 / H3 r8 / S2 L13 / equal-area quadtree; quadtree sd 0, point-to-centre 306 m) | `APIN/experiments/h3s2/partitions.py`, `grid_geometry_diagnostic.csv` | — | note: H3 and S2 are 25% and 60% coarser; the quadtree matches the GeoSOT mean exactly |
| Alignment ablation (T-Drive) | seven variants, F1 .9073–.9130 (span .006), .021–.027 below GeoSOT | `APIN/experiments/h3s2/results/alignment_ablation.csv` | `f1` | ok: half-cell shifts and 15–45° rotations at fixed cell area; rotation variants also vary canvas size |
| Reconstructed-GeoSOT baseline | Porto .7831 vs published .7862 | `APIN/experiments/h3s2/results/partition_native_summary.csv` | `f1` | note: the published cache was built from a prepared point cache that is no longer reproducible, so every partition is re-materialised under one builder; the occupancy channel is bit-identical (maxdiff = 0) |


## 3. Backbone and channel

All rows come from `0911/requested_4_5_full10` and are re-derived by
`APIN/experiments/backbone_table_v2.py`, which asserts that every run shares the
same manifest, frame cache, epochs, batch size, learning rate and test slice.

| Claim | Value | Source | Status |
|---|---|---|---|
| ConvGRU (6) | .7862 / .9345 | `APIN/experiments/results/backbone_summary_v2.csv` | ok |
| ConvLSTM (6) | .7865 / .9341 (10 seeds, $\Delta$ +.0005 / -.0006, Holm 3.6e-5 / .0042) | `backbone_summary_v2.csv`, `backbone_paired_stats_v2.csv` | fixed: earlier drafts quoted a three-seed diagnostic |
| PredRNN++ (6) | .7887 / .9311 (10 seeds, Holm 4.7e-75 / 3.1e-29) | `backbone_paired_stats_v2.csv` | fixed: earlier table carried three Porto seeds only and raw rather than Holm p |
| STGCN / Graph WaveNet | .7792/.9303 and .7794/.9304 | `backbone_summary_v2.csv` | ok |
| ConvGRU (2) | .7781/.9320, Δ −.0080/−.0025 | same | ok |
| Parameter counts | 42,246 (STGCN, GWN), 91,302 (ConvGRU), 114,678 (ConvLSTM), 160,086 (Transformer), 526,854 (PredRNN++) | instantiated from `src/.../cross_city_runner._model`; PredRNN++ matches `resource_usage.inference.parameter_count` in `elsarticle/audit_artifacts/universal_sota_runs_v2` | fixed: previous text quoted 88,612 and 159,124 from an older model definition |
| Transformer | .7861 / .9359 (10 seeds; $\Delta$ -.0003 / +.0035; Holm $p$ .177 / .0691) | `backbone_summary_v2.csv`, `backbone_paired_stats_v2.csv` | fixed: ten-seed means replace the earlier five-seed diagnostic |
| Holm families | backbone contrasts (ConvLSTM, PredRNN++, STGCN, Graph WaveNet, Transformer) form one family per city; the two-channel input is a one-comparison family | `APIN/experiments/backbone_table_v2.py` (`main`), `backbone_paired_stats_v2.csv` | fixed: the earlier CSV applied one four-contrast family that also included the channel row, contradicting the table caption |

## 4. Budgeted dispatch

| Claim | Value | Source | Field | Status |
|---|---|---|---|---|
| Coverage at K=100 (mean score) | .4864/.0261/.0470/.4999 Porto; .1471/.1401/.1401/.1472 T-Drive | `APIN/experiments/results/application_dispatch_summary.csv` | `mean_coverage` where `score_mode=mean`, `k=100` | ok |
| 95% CI around frame coverage | [.4813,.4915] / [.1461,.1482] | same | `coverage_ci_low/high` | ok |
| Paired Δ coverage (frame − entity GRU) | +.4603 [.4549,.4656]; +.0070 [.0049,.0094] | `APIN/experiments/results/application_dispatch_paired.csv` | `mean_diff`, `ci_low/high` | ok |
| Holm p | 6.5e-197 (Porto), 3.5e-40 (T-Drive) | same | `holm_p` | ok |
| Coverage at K=800 | .99983 vs .84674 (T-Drive) | `application_dispatch_summary.csv` | `mean_coverage` `k=800`, `score_mode=mean` | ok |
| Coverage AUC | .4390/.1181/.1332/.4622 Porto; .2277/.2095/.2093/.2279 T-Drive | `APIN/experiments/results/application_dispatch_auc.csv` | — | ok (not quoted in the paper text) |

## 5. Knowledge diagnostic

| Claim | Value | Source | Status |
|---|---|---|---|
| OSM highway-class gain | +0.13% relative, Δ +0.000976 | `elsarticle/audit_artifacts/external_osm_statistics_v1/EXTERNAL_OSM_POOLED_AUDIT.md` | ok |
| Retrieval route loss | −21.3% relative, Δ −0.0927 | `elsarticle/audit_artifacts/porto_expanded_source_temporal_retrieval_v1/budget_010/EXPANDED_SOURCE_RETRIEVAL_REPORT.md` | ok |

## 6. Setup facts

| Claim | Value | Source | Status |
|---|---|---|---|
| Compact frame runs | 48 hidden, batch 8, lr 5e-4, 8 epochs | every `spec.json` under `0911/requested_4_5_full10` | fixed: the draft said 24 epochs |
| Entity branch | batch 256, 8 epochs | `elsarticle/audit_artifacts/representation_benchmark_v1/paradigm_confirmatory_v1/full_v1/raw/*/spec.json` | fixed |
| Canonical union / factorial budget | 24 epochs | `elsarticle/audit_artifacts/universal_sota_runs_v2/**/completed.json` → `report.config.train.epochs` | ok |
| T-Drive temporal blocks | 3–4 per native-frame contrast; 20 blocks of 12 windows in the compact backbone slice | `canonical_union_v1/paired_raw_grid_geosot_statistics.csv` (`n_blocks`); `backbone_paired_stats_v2.csv` (`n_blocks`) | fixed: the draft quoted "11 blocks" and contradicted the Results section |

## 7. Open items

- The shared-target control (`tab:raw_grid_geosot`) is a three-seed diagnostic and
  is labelled in the caption; it should not be pooled with ten-seed contrasts.
- The native H3/S2 contrasts are ten-seed; the GeoSOT-target resampling control
  is a three-seed diagnostic and is labelled in `tab:partition_controls`.
- The Declarations state that code and derived artifacts will be released in a
  public repository upon acceptance, with a reproducibility package available to
  reviewers on request. Replace this sentence with the final repository URL and
  Zenodo DOI at the acceptance stage.
