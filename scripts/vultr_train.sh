#!/usr/bin/env bash
# Run from the project root with the project virtual environment activated.
set -euo pipefail

if [[ $# -ne 3 || ( "$1" != "smoke" && "$1" != "full" ) ]]; then
  echo "Usage: bash scripts/vultr_train.sh {smoke|full} DATASET.jsonl RUN_ID" >&2
  exit 2
fi

mode="$1"
dataset="$2"
run_id="$3"

if [[ ! "$run_id" =~ ^[A-Za-z0-9._-]+$ || "$run_id" == "." || "$run_id" == ".." ]]; then
  echo "RUN_ID may contain only letters, numbers, dots, underscores, and hyphens." >&2
  exit 2
fi
if [[ ! -f "$dataset" ]]; then
  echo "Dataset not found: $dataset" >&2
  exit 2
fi
if [[ ! -f src/train.py || ! -f scripts/validate_dataset.py ]]; then
  echo "Run this script from the repository root." >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA driver and GPU are required." >&2
  exit 1
fi
nvidia-smi

output_dir="artifacts/$run_id"
run_dir="runs/$run_id"
if [[ -e "$output_dir" || -e "$run_dir" ]]; then
  echo "Run or output directory already exists. Choose a new RUN_ID." >&2
  exit 2
fi

python scripts/validate_dataset.py --data-path "$dataset"

args=(
  --data-path "$dataset"
  --run-id "$run_id"
  --run-dir "$run_dir"
  --output-dir "$output_dir"
)
if [[ "$mode" == "smoke" ]]; then
  args+=(--max-train-samples 10 --max-eval-samples 2 --epochs 0.01
         --warmup-steps 1 --eval-steps 1 --save-steps 1)
else
  args+=(--epochs 3 --eval-steps 25 --save-steps 25)
fi

echo "Starting $mode run $run_id. Monitor elapsed time and Vultr spend."
echo "This script does not stop or destroy the billable instance."
python -m src.train "${args[@]}"
echo "Finished. Back up $output_dir and $run_dir before destroying the instance."
