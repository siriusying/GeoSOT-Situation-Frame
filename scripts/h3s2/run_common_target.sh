#!/usr/bin/env bash
# Common-target control: H3/S2/quadtree input resampled onto the GeoSOT 32x32 target.
# Three seeds, matching the strict shared-target control in the manuscript.
set -u

# Paths come from the environment so the script works from any clone.
#   ROOT           repository root containing src/geosot_generative_trajectory
#   PY             python interpreter with the pinned requirements
#   PARADIGM_BASE  .../representation_benchmark_v1/paradigm_confirmatory_v1
#   FRAME_CACHE    common-target cache written by build_common_target.py
#   RUNS_OUT       directory that will hold the per-seed run folders
ROOT=${ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || pwd)}
PY=${PY:-python}
BASE=${PARADIGM_BASE:?set PARADIGM_BASE to the paradigm_confirmatory_v1 directory}
CACHE=${FRAME_CACHE:-$ROOT/papers/APIN/experiments/h3s2/frame_cache/common_target}
RUNS=${RUNS_OUT:-$ROOT/papers/APIN/experiments/h3s2/runs/common_target}
mkdir -p "$RUNS"

run_one () {
  local city="$1" partition="$2" seed="$3" gpu="$4"
  local limit; [ "$city" = "porto" ] && limit=1200 || limit=240
  local out="$RUNS/$partition/${city}_convgru_seed${seed}"
  mkdir -p "$out"
  if [ -f "$out/convgru_seed${seed}.json" ]; then echo "[skip] $city $partition $seed"; return; fi
  "$PY" - "$city" "$BASE" "$CACHE" "$partition" "$seed" "$out/spec.json" "$limit" <<'PY'
import json, sys
city, base, cache, partition, seed, dest, limit = sys.argv[1:8]
spec = {
  "source_manifest": f"{base}/manifests/{city}_compact_manifest.json",
  "target_manifest": f"{base}/manifests/{city}_compact_manifest.json",
  "prepared_cache_root": f"{base}/prepared_v2",
  "frame_cache_root": f"{cache}/{partition}",
  "model_id": "convgru", "seed": int(seed), "epochs": 8, "batch_size": 8,
  "hidden_channels": 48, "learning_rate": 0.0005, "hierarchy_weight": 0,
  "hierarchy_contrastive_weight": 0, "road_rule_weight": 0, "semantic_rule_weight": 0,
  "traversability_weight": 0, "sample_weight_mode": "none", "sample_weight_alpha": 0,
  "num_workers": 0, "cuda_only": True, "train_sample_limit": None,
  "validation_sample_limit": None, "test_sample_limit": int(limit),
}
json.dump(spec, open(dest, "w"), indent=2)
PY
  echo "[run] common-target $city $partition seed$seed gpu$gpu $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT" "$PY" -u -m \
    src.geosot_generative_trajectory.train.cross_city_runner \
    --spec "$out/spec.json" --output-dir "$out" > "$out/run.log" 2>&1
  echo "[done] $city $partition seed$seed rc=$? $(date +%H:%M:%S)"
}

worker0 () {
  for partition in h3 s2 quadtree; do
    for seed in 41 42 43; do run_one porto "$partition" "$seed" "${GPU_PORTO:-0}"; done
  done
}
worker1 () {
  for partition in h3 s2 quadtree; do
    for seed in 41 42 43; do run_one t_drive "$partition" "$seed" "${GPU_TDRIVE:-1}"; done
  done
}
case "${WORKER:-both}" in
  0) worker0 ;;
  1) worker1 ;;
  both) worker0 & worker1 & wait ;;
  *) echo "WORKER must be 0, 1 or both" >&2; exit 2 ;;
esac
echo "COMMON TARGET DONE $(date +%H:%M:%S)"
