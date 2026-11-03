#!/usr/bin/env bash
# Supplementary Transformer backbone runs for the Applied Intelligence revision.
# Ten seeds per city (41-50), the same seed set as the other backbones, so the
# Transformer enters the Holm-corrected backbone family. Contract is identical
# to the requested_4_5_full10 compact runs except model_id=transformer.
set -u

# Paths are supplied through the environment so the script works from any clone.
#   ROOT             repository root containing src/geosot_generative_trajectory
#   PY               python interpreter with the pinned requirements installed
#   PARADIGM_BASE    .../representation_benchmark_v1/paradigm_confirmatory_v1
#   RUNS_OUT         directory that will hold the per-seed run folders
ROOT=${ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || pwd)}
PY=${PY:-python}
BASE=${PARADIGM_BASE:?set PARADIGM_BASE to the paradigm_confirmatory_v1 directory}
OUTROOT=${RUNS_OUT:?set RUNS_OUT to the Transformer run directory}
mkdir -p "$OUTROOT"

run_city () {
  local city="$1" gpu="$2" manifest="$3" limit="$4"
  for seed in 41 42 43 44 45 46 47 48 49 50; do
    local out="$OUTROOT/${city}_transformer_seed${seed}"
    mkdir -p "$out"
    if [[ -f "$out/transformer_seed${seed}.json" ]]; then
      echo "[skip] $city seed$seed"
      continue
    fi
    "$PY" - "$city" "$seed" "$manifest" "$limit" "$BASE" "$out/spec.json" <<'PY'
import json, sys
city, seed, manifest, limit, base, dest = sys.argv[1:7]
spec = {
    "source_manifest": manifest,
    "target_manifest": manifest,
    "prepared_cache_root": f"{base}/prepared_v2",
    "frame_cache_root": f"{base}/compact_frame_cache",
    "model_id": "transformer",
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
}
json.dump(spec, open(dest, "w"), indent=2)
PY
    echo "[run] $city seed$seed gpu$gpu $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT" "$PY" -u \
      -m src.geosot_generative_trajectory.train.cross_city_runner \
      --spec "$out/spec.json" --output-dir "$out" > "$out/run.log" 2>&1
    echo "[done] $city seed$seed rc=$? $(date +%H:%M:%S)"
  done
}

run_city porto 0 "$BASE/manifests/porto_compact_manifest.json" 1200 &
run_city t_drive 1 "$BASE/manifests/t_drive_compact_manifest.json" 240 &
wait
echo "ALL DONE $(date +%H:%M:%S)"
