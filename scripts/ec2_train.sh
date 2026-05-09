#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/logsage}"
RUN_ID="${RUN_ID:-ec2-$(date -u +%Y%m%d-%H%M%S)}"
HF_REPO_ID="${HF_REPO_ID:-auro-rirum/LogSage-Qwen2.5-7B-QLoRA-v0}"
SHUTDOWN_MINUTES="${SHUTDOWN_MINUTES:-480}"

cd "$PROJECT_DIR"

echo "Scheduling safety shutdown in ${SHUTDOWN_MINUTES} minutes."
sudo shutdown -h "+${SHUTDOWN_MINUTES}" "LogSage budget safety shutdown" || true

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found; this instance is not ready for GPU training." >&2
  exit 1
fi

python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -r requirements.txt

python validate_dataset.py

echo "Running smoke training job."
python train_logsage.py \
  --run-id "${RUN_ID}-smoke" \
  --output-dir LogSage-smoke-output \
  --max-train-samples 10 \
  --max-eval-samples 2 \
  --epochs 0.01 \
  --warmup-steps 1 \
  --eval-steps 1 \
  --save-steps 1

rm -rf LogSage-smoke-output

echo "Running full training job."
python train_logsage.py \
  --run-id "$RUN_ID" \
  --epochs 3 \
  --eval-steps 25 \
  --save-steps 25 \
  --push-to-hub \
  --hf-repo-id "$HF_REPO_ID" \
  --resume

python test_logsage.py --adapter-dir LogSage-Qwen2.5-7B-QLoRA-v0

echo "Training complete. Artifacts are in $PROJECT_DIR/LogSage-Qwen2.5-7B-QLoRA-v0 and $PROJECT_DIR/runs/$RUN_ID."
