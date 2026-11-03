#!/usr/bin/env bash
# Native partition runs: GeoSOT (rebuilt parity cache), H3, S2, equal-area quadtree.
# Two workers, one per GPU; T-Drive runs are short and fill the tail.
set -u

# Paths come from the environment so the script works from any clone.
#   ROOT           repository root containing src/geosot_generative_trajectory
#   PY             python interpreter with the pinned requirements
#   PARADIGM_BASE  .../representation_benchmark_v1/paradigm_confirmatory_v1
#   FRAME_CACHE    root written by build_frames.py (default: experiments/h3s2/frame_cache)
#   RUNS_OUT       directory that will hold the per-seed run folders
ROOT=${ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || pwd)}
PY=${PY:-python}
BASE=${PARADIGM_BASE:?set PARADIGM_BASE to the paradigm_confirmatory_v1 directory}
CACHE=${FRAME_CACHE:-$ROOT/papers/APIN/experiments/h3s2/frame_cache}
RUNS=${RUNS_OUT:-$ROOT/papers/APIN/experiments/h3s2/runs}
mkdir -p "$RUNS"

make_spec () {
  local city="$1" partition="$2" seed="$3" dest="$4" limit="$5"
  mkdir -p "$(dirname "$dest")"
  "$PY" - "$city" "$partition" "$seed" "$BASE" "$CACHE" "$dest" "$limit" <<'PY'
import json, sys
city, partition, seed, base, cache, dest, limit = sys.argv[1:8]
spec = {
  "source_manifest": f"{base}/manifests/{city}_compact_manifest.json",
  "target_manifest": f"{base}/manifests/{city}_compact_manifest.json",
  "prepared_cache_root": f"{base}/prepared_v2",
  "frame_cache_root": f"{cache}/{partition}",
  "model_id": "convgru",
  "seed": int(seed),
  "epochs": 8,
  "batch_size": 8,
  "hidden_channels": 48,
  "learning_rate": 0.0005,
  "hierarchy_weight": 0,
  "hierarchy_contrastive_weight": 0,
  "road_rule_weight": 0,
  "semantic_rule_weight": 0,
  "traversability_weight": 0,
  "sample_weight_mode": "none",
  "sample_weight_alpha": 0,
  "num_workers": 0,
  "cuda_only": True,
  "train_sample_limit": None,
  "validation_sample_limit": None,
  "test_sample_limit": int(limit),
}
json.dump(spec, open(dest, "w"), indent=2)
PY
}

run_one () {
  local city="$1" partition="$2" seed="$3" gpu="$4"
  local limit; [ "$city" = "porto" ] && limit=1200 || limit=240
  local out="$RUNS/$partition/${city}_convgru_seed${seed}"
  mkdir -p "$out"
  if [ -f "$out/convgru_seed${seed}.json" ]; then echo "[skip] $city $partition $seed"; return; fi
  make_spec "$city" "$partition" "$seed" "$out/spec.json" "$limit"
  echo "[run] $city $partition seed$seed gpu$gpu $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT" "$PY" -u -m \
    src.geosot_generative_trajectory.train.cross_city_runner \
    --spec "$out/spec.json" --output-dir "$out" > "$out/run.log" 2>&1
  echo "[done] $city $partition seed$seed rc=$? $(date +%H:%M:%S)"
}

SEED_LIST=${SEEDS:-41 42 43 44 45 46 47 48 49 50}
PARTITIONS=${PARTITIONS:-geosot h3 s2 quadtree}
CITIES=${CITIES:-porto t_drive}

has () { case " $1 " in *" $2 "*) return 0 ;; esac; return 1 ; }

worker0 () {
  if has "$CITIES" porto; then
    for partition in geosot h3; do
      has "$PARTITIONS" "$partition" || continue
      for seed in $SEED_LIST; do run_one porto "$partition" "$seed" "${GPU_A:-0}"; done
    done
  fi
  if has "$CITIES" t_drive; then
    for partition in geosot h3; do
      has "$PARTITIONS" "$partition" || continue
      for seed in $SEED_LIST; do run_one t_drive "$partition" "$seed" "${GPU_A:-0}"; done
    done
  fi
}

worker1 () {
  if has "$CITIES" porto; then
    for partition in s2 quadtree; do
      has "$PARTITIONS" "$partition" || continue
      for seed in $SEED_LIST; do run_one porto "$partition" "$seed" "${GPU_B:-1}"; done
    done
  fi
  if has "$CITIES" t_drive; then
    for partition in s2 quadtree; do
      has "$PARTITIONS" "$partition" || continue
      for seed in $SEED_LIST; do run_one t_drive "$partition" "$seed" "${GPU_B:-1}"; done
    done
  fi
}

case "${WORKER:-both}" in
  0) worker0 ;;
  1) worker1 ;;
  both) worker0 & worker1 & wait ;;
  *) echo "WORKER must be 0, 1 or both" >&2; exit 2 ;;
esac
echo "ALL DONE $(date +%H:%M:%S)"
