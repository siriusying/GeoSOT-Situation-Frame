#!/usr/bin/env bash
# Complete the matched backbone table: ten-seed ConvLSTM under the exact same
# compact contract as requested_4_5_full10 (8 epochs, hidden 48, batch 8,
# lr 5e-4, compact frame cache, Porto 1200 / T-Drive 240 test origins).
set -u
# Paths are supplied through the environment so the script works from any clone.
#   ROOT             repository root containing src/geosot_generative_trajectory
#   PY               python interpreter with the pinned requirements installed
#   PARADIGM_BASE    .../representation_benchmark_v1/paradigm_confirmatory_v1
#   RUNS_OUT         directory that will hold the per-seed run folders
ROOT=${ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || pwd)}
PY=${PY:-python}
BASE=${PARADIGM_BASE:?set PARADIGM_BASE to the paradigm_confirmatory_v1 directory}
OUT=${RUNS_OUT:?set RUNS_OUT to the backbone run directory}
GPU=${GPU:-0}

cd "$ROOT"
for city in t_drive porto; do
  case "$city" in
    porto) manifest="$BASE/manifests/porto_compact_manifest.json"; limit=1200 ;;
    t_drive) manifest="$BASE/manifests/t_drive_compact_manifest.json"; limit=240 ;;
  esac
  for seed in 41 42 43 44 45 46 47 48 49 50; do
    out="$OUT/${city}_convlstm_seed${seed}"
    mkdir -p "$out"
    if [[ -f "$out/convlstm_seed${seed}.json" ]]; then echo "[skip] $city seed$seed"; continue; fi
    "$PY" - "$manifest" "$seed" "$limit" "$BASE" "$out/spec.json" <<'PY'
import json, sys
manifest, seed, limit, base, dest = sys.argv[1:6]
json.dump({
    "source_manifest": manifest,
    "target_manifest": manifest,
    "prepared_cache_root": f"{base}/prepared_v2",
    "frame_cache_root": f"{base}/compact_frame_cache",
    "model_id": "convlstm",
    "seed": int(seed),
    "epochs": 8,
    "batch_size": 8,
    "hidden_channels": 48,
    "learning_rate": 0.0005,
    "hierarchy_weight": 0.0,
    "hierarchy_contrastive_weight": 0.0,
    "road_rule_weight": 0.0,
    "semantic_rule_weight": 0.0,
    "traversability_weight": 0.0,
    "sample_weight_mode": "none",
    "sample_weight_alpha": 0.0,
    "num_workers": 0,
    "cuda_only": True,
    "train_sample_limit": None,
    "validation_sample_limit": None,
    "test_sample_limit": int(limit),
}, open(dest, "w"), indent=2)
PY
    echo "[run] $city convlstm seed$seed $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT" "$PY" -u \
      -m src.geosot_generative_trajectory.train.cross_city_runner \
      --spec "$out/spec.json" --output-dir "$out" > "$out/run.log" 2>&1
    echo "[done] $city convlstm seed$seed rc=$? $(date +%H:%M:%S)"
  done
done
echo "ALL DONE $(date +%H:%M:%S)"
