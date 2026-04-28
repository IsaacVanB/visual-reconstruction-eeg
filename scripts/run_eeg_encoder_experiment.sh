#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run one EEG encoder experiment:
1) Train EEG encoder into a timestamped run directory.
2) Run eval_eeg_encoder.py using the produced checkpoint.
3) Run eval_eeg_with_mean_baselines.py for model-vs-baseline metrics.

Usage:
  scripts/run_eeg_encoder_experiment.sh [runner options] [train args ...] [--eval eval args ...] [--baseline baseline args ...]

Runner options:
  --output-base PATH   Base directory for runs (default: outputs/eeg_encoder)
  --run-name NAME      Explicit run directory name (default: run_YYYYMMDD_HHMMSS)
  --checkpoint-path P  Skip training and resume evaluation from an existing checkpoint
  --skip-eval          Train only; skip evaluation step
  --skip-baseline      Skip eval_eeg_with_mean_baselines.py step
  --help               Show this help

Argument forwarding:
  - Any args before '--eval' are passed to scripts/train_eeg_encoder.py
  - Any args between '--eval' and '--baseline' are passed to src/evaluation/eval_eeg_encoder.py
  - Any args after '--baseline' are passed to src/evaluation/eval_eeg_with_mean_baselines.py

Examples:
  scripts/run_eeg_encoder_experiment.sh

  scripts/run_eeg_encoder_experiment.sh \
    --config configs/eeg_encoder.yaml \
    --epochs 40 \
    --batch-size 32 \
    --eval --max-samples 16 --grid-images 8

  scripts/run_eeg_encoder_experiment.sh \
    --run-name ablation_k3_7 \
    --temporal-kernel3 7 \
    --eval --device cpu

  scripts/run_eeg_encoder_experiment.sh \
    --eval --max-samples 16 --grid-images 8 \
    --baseline --image-size 512 --batch-size 4
EOF
}

OUTPUT_BASE="outputs/eeg_encoder"
RUN_NAME=""
SKIP_EVAL=0
SKIP_BASELINE=0
CHECKPOINT_PATH=""
TRAIN_SCRIPT="scripts/train_eeg_encoder.py"
EVAL_SCRIPT="src/evaluation/eval_eeg_encoder.py"
BASELINE_EVAL_SCRIPT="src/evaluation/eval_eeg_with_mean_baselines.py"

train_args=()
eval_args=()
baseline_args=()
mode="train"

run_python() {
  local script_path="$1"
  shift
  local cmd=(python "$script_path")
  if [[ "$#" -gt 0 ]]; then
    cmd+=("$@")
  fi
  "${cmd[@]}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      usage
      exit 0
      ;;
    --output-base)
      OUTPUT_BASE="$2"
      shift 2
      ;;
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --checkpoint-path)
      CHECKPOINT_PATH="$2"
      shift 2
      ;;
    --skip-eval)
      SKIP_EVAL=1
      shift
      ;;
    --skip-baseline)
      SKIP_BASELINE=1
      shift
      ;;
    --eval)
      mode="eval"
      shift
      ;;
    --baseline)
      mode="baseline"
      shift
      ;;
    *)
      if [[ "$mode" == "train" ]]; then
        train_args+=("$1")
      elif [[ "$mode" == "eval" ]]; then
        eval_args+=("$1")
      else
        baseline_args+=("$1")
      fi
      shift
      ;;
  esac
done

if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="run_$(date +%Y%m%d_%H%M%S)"
fi

if [[ -n "$CHECKPOINT_PATH" ]]; then
  checkpoint_path="$CHECKPOINT_PATH"
  if [[ ! -f "$checkpoint_path" ]]; then
    printf 'ERROR: Checkpoint not found: %s\n' "$checkpoint_path" >&2
    exit 1
  fi
  RUN_DIR="$(cd "$(dirname "$checkpoint_path")" && pwd)"
  printf 'Run directory: %s\n' "$RUN_DIR"
  printf 'Skipping training and resuming from checkpoint: %s\n' "$checkpoint_path"
else
  RUN_DIR="${OUTPUT_BASE%/}/$RUN_NAME"
  mkdir -p "$RUN_DIR"

  printf 'Run directory: %s\n' "$RUN_DIR"
  printf 'Starting training...\n'
  if [[ "${#train_args[@]}" -gt 0 ]]; then
    run_python "$TRAIN_SCRIPT" "${train_args[@]}" --output-dir "$RUN_DIR"
  else
    run_python "$TRAIN_SCRIPT" --output-dir "$RUN_DIR"
  fi

  # train_eeg_encoder.py saves both final and best checkpoints.
  # Default to best checkpoint when available.
  checkpoint_path="$(ls -1t "$RUN_DIR"/eeg_encoder_best_*.pt 2>/dev/null | head -n 1 || true)"
  if [[ -z "$checkpoint_path" ]]; then
    checkpoint_path="$(ls -1t "$RUN_DIR"/eeg_encoder_[0-9]*.pt 2>/dev/null | head -n 1 || true)"
  fi
  if [[ -z "$checkpoint_path" ]]; then
    checkpoint_path="$(ls -1t "$RUN_DIR"/eeg_encoder.pt 2>/dev/null | head -n 1 || true)"
  fi
  if [[ -z "$checkpoint_path" ]]; then
    printf 'ERROR: No checkpoint found in %s\n' "$RUN_DIR" >&2
    exit 1
  fi
  printf 'Training complete. Checkpoint: %s\n' "$checkpoint_path"
fi

if [[ "$SKIP_EVAL" -eq 1 ]]; then
  printf 'Skipping evaluation (--skip-eval).\n'
else
  EVAL_DIR="$RUN_DIR/eval"
  mkdir -p "$EVAL_DIR"
  printf 'Starting evaluation...\n'
  if [[ "${#eval_args[@]}" -gt 0 ]]; then
    run_python "$EVAL_SCRIPT" \
      --checkpoint-path "$checkpoint_path" \
      --output-dir "$EVAL_DIR" \
      "${eval_args[@]}"
  else
    run_python "$EVAL_SCRIPT" \
      --checkpoint-path "$checkpoint_path" \
      --output-dir "$EVAL_DIR"
  fi

  printf 'Evaluation complete.\n'
  printf 'Expected grid: %s\n' "$EVAL_DIR/recon_grid.png"
fi

if [[ "$SKIP_BASELINE" -eq 1 ]]; then
  printf 'Skipping baseline comparison eval (--skip-baseline).\n'
  exit 0
fi

BASELINE_EVAL_DIR="$RUN_DIR/eval"
mkdir -p "$BASELINE_EVAL_DIR"
printf 'Starting baseline comparison evaluation...\n'
if [[ "${#baseline_args[@]}" -gt 0 ]]; then
  run_python "$BASELINE_EVAL_SCRIPT" \
    --checkpoint-path "$checkpoint_path" \
    --output-dir "$BASELINE_EVAL_DIR" \
    "${baseline_args[@]}"
else
  run_python "$BASELINE_EVAL_SCRIPT" \
    --checkpoint-path "$checkpoint_path" \
    --output-dir "$BASELINE_EVAL_DIR"
fi
printf 'Baseline comparison evaluation complete.\n'
printf 'Baseline metrics: %s\n' "$BASELINE_EVAL_DIR/eeg_vs_baselines_metrics.json"
