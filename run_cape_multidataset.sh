#!/usr/bin/env bash
set -euo pipefail

STAGE="train"
DATASET="visdrone"
MODEL_MODE="cape"
CHECKPOINT=""
LIMIT_BATCHES=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)
      STAGE="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --model-mode)
      MODEL_MODE="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --limit-batches)
      LIMIT_BATCHES="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

datasets=()
if [[ "$DATASET" == "both" ]]; then
  datasets=("visdrone" "tinyperson")
else
  datasets=("$DATASET")
fi

for ds in "${datasets[@]}"; do
  case "$STAGE" in
    prepare|prepare_datasets)
      python scripts/prepare_datasets.py --dataset "$ds" "${EXTRA_ARGS[@]}"
      ;;
    train)
      cmd=(python scripts/train.py --dataset "$ds" --model-mode "$MODEL_MODE")
      if [[ -n "$LIMIT_BATCHES" ]]; then
        echo "--limit-batches is not a training option; use a smoke config or override train.epochs in YAML."
      fi
      "${cmd[@]}" "${EXTRA_ARGS[@]}"
      ;;
    eval|evaluate)
      cmd=(python scripts/evaluate.py --dataset "$ds" --model-mode "$MODEL_MODE" --export-optional-curves)
      if [[ -n "$CHECKPOINT" ]]; then
        cmd+=(--checkpoint "$CHECKPOINT")
      fi
      if [[ -n "$LIMIT_BATCHES" ]]; then
        cmd+=(--limit-batches "$LIMIT_BATCHES")
      fi
      "${cmd[@]}" "${EXTRA_ARGS[@]}"
      ;;
    benchmark)
      cmd=(python scripts/budget_sweep.py --dataset "$ds" --export-optional-curves)
      if [[ -n "$CHECKPOINT" ]]; then
        cmd+=(--checkpoint "$CHECKPOINT")
      fi
      if [[ -n "$LIMIT_BATCHES" ]]; then
        cmd+=(--limit-batches "$LIMIT_BATCHES")
      fi
      "${cmd[@]}" "${EXTRA_ARGS[@]}"
      ;;
    *)
      echo "Unsupported --stage '$STAGE'. Use prepare, train, eval, or benchmark." >&2
      exit 2
      ;;
  esac
done
