#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run one EEG encoder experiment:
1) Train EEG encoder into a timestamped run directory.
2) Run eval_eeg_encoder.py using the produced checkpoint.
3) (Optional, separate) run mean-image baseline script manually.

Usage:
  scripts/run_eeg_encoder_experiment.sh [runner options] [train args ...] [--eval eval args ...] [--baseline baseline args ...]

Runner options:
  --output-base PATH   Base directory for runs (default: outputs/eeg_encoder)
  --run-name NAME      Explicit run directory name (default: run_YYYYMMDD_HHMMSS)
  --skip-eval          Train only; skip evaluation step
  --skip-baseline      Ignored (baseline now runs as a separate script)
  --help               Show this help

Argument forwarding:
  - Any args before '--eval' are passed to scripts/train_eeg_encoder.py
  - Any args between '--eval' and '--baseline' are passed to src/evaluation/eval_eeg_encoder.py
  - Any args after '--baseline' are passed to scripts/eval_mean_image_baseline.py (manual run)

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
    --baseline --image-size 256 --batch-size 64
EOF
}

OUTPUT_BASE="outputs/eeg_encoder"
RUN_NAME=""
SKIP_EVAL=0
SKIP_BASELINE=0
TRAIN_SCRIPT="scripts/train_eeg_encoder.py"
EVAL_SCRIPT="src/evaluation/eval_eeg_encoder.py"
BASELINE_SCRIPT="scripts/eval_mean_image_baseline.py"

train_args=()
eval_args=()
baseline_args=()
mode="train"

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

RUN_DIR="${OUTPUT_BASE%/}/$RUN_NAME"
mkdir -p "$RUN_DIR"

printf 'Run directory: %s\n' "$RUN_DIR"
printf 'Starting training...\n'
python "$TRAIN_SCRIPT" "${train_args[@]}" --output-dir "$RUN_DIR"

# train_eeg_encoder.py now saves timestamped checkpoints. Pick the newest one in this run dir.
checkpoint_path="$(ls -1t "$RUN_DIR"/eeg_encoder_*.pt 2>/dev/null | head -n 1 || true)"
if [[ -z "$checkpoint_path" ]]; then
  checkpoint_path="$(ls -1t "$RUN_DIR"/eeg_encoder.pt 2>/dev/null | head -n 1 || true)"
fi
if [[ -z "$checkpoint_path" ]]; then
  printf 'ERROR: No checkpoint found in %s\n' "$RUN_DIR" >&2
  exit 1
fi
printf 'Training complete. Checkpoint: %s\n' "$checkpoint_path"

if [[ "$SKIP_EVAL" -eq 1 ]]; then
  printf 'Skipping evaluation (--skip-eval).\n'
else
  EVAL_DIR="$RUN_DIR/eval"
  mkdir -p "$EVAL_DIR"
  printf 'Starting evaluation...\n'
  python "$EVAL_SCRIPT" \
    --checkpoint-path "$checkpoint_path" \
    --output-dir "$EVAL_DIR" \
    "${eval_args[@]}"

  printf 'Evaluation complete.\n'
  printf 'Expected grid: %s\n' "$EVAL_DIR/recon_grid.png"
fi

if [[ "$SKIP_BASELINE" -eq 0 ]]; then
  printf 'Baseline evaluation is now a separate step.\n'
  printf 'Run manually when needed:\n'
  printf '  python %s --output-dir %s/baseline %s\n' "$BASELINE_SCRIPT" "$RUN_DIR" "${baseline_args[*]:-}"
fi
