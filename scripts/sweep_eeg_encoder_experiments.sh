#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run a simple grid search over EEG encoder experiment parameters.

Edit the parameter arrays near the top of this file, then run:
  bash scripts/sweep_eeg_encoder_experiments.sh

What it does:
  1) Loops over every combination in the parameter arrays.
  2) Calls scripts/run_eeg_encoder_experiment.sh for each combination.
  3) Collects training/eval metrics into one TSV file.
  4) Prints the best run according to BEST_METRIC / BEST_MODE.

Notes:
  - This only sweeps parameters that scripts/train_eeg_encoder.py currently accepts.
  - To save time, you can set SKIP_EVAL=1 or SKIP_BASELINE=1 below.
  - If you later expose architecture flags in the training CLI, add them to the arrays here.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_SCRIPT="$REPO_ROOT/scripts/run_eeg_encoder_experiment.sh"

if [[ ! -x "$RUN_SCRIPT" ]]; then
  printf 'ERROR: Expected executable run script at %s\n' "$RUN_SCRIPT" >&2
  printf 'Try: chmod +x scripts/run_eeg_encoder_experiment.sh\n' >&2
  exit 1
fi

# Sweep configuration.
CONFIG_PATH="configs/eeg_encoder.yaml"
OUTPUT_BASE="outputs/eeg_encoder_sweeps"

# Choose how "best" should be ranked.
# Examples:
#   BEST_METRIC="best_valid_loss" BEST_MODE="min"
#   BEST_METRIC="LPIPS_model" BEST_MODE="min"
#   BEST_METRIC="SSIM_model" BEST_MODE="max"
BEST_METRIC="LPIPS_model"
BEST_MODE="min"

# Toggle expensive steps.
SKIP_EVAL=0
SKIP_BASELINE=0

# Default forwarded args.
EVAL_ARGS=(--max-samples 16 --image-size 512 --batch-size 4)
BASELINE_ARGS=(--image-size 512 --batch-size 4)

# Add fixed training args here if you want them applied to every run.
TRAIN_COMMON_ARGS=(
  --config "$CONFIG_PATH"
  --device auto
)

# Candidate values to sweep.
LRS=(1e-4 3e-4)
WEIGHT_DECAYS=(1e-3 1e-4)
BATCH_SIZES=(16 32)
EEG_NORMALIZATIONS=(zscore l2)
OUTPUT_DIMS=(32 64)
EPOCHS=(15)

timestamp() {
  date +%Y%m%d_%H%M%S
}

sanitize_token() {
  local value="$1"
  value="${value//\//-}"
  value="${value// /_}"
  value="${value//./p}"
  printf '%s' "$value"
}

json_value() {
  local json_path="$1"
  local key="$2"
  python - "$json_path" "$key" <<'PY'
import json
import sys

path, key = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

value = data
for part in key.split("."):
    if not part:
        continue
    if isinstance(value, dict) and part in value:
        value = value[part]
    else:
        value = ""
        break

if isinstance(value, list):
    print(",".join(str(x) for x in value))
else:
    print(value)
PY
}

metric_file_for_run() {
  local run_dir="$1"
  local training_summary
  training_summary="$(ls -1t "$run_dir"/training_summary_*.json 2>/dev/null | head -n 1 || true)"
  local baseline_metrics="$run_dir/eval/eeg_vs_baselines_metrics.json"

  if [[ -n "$training_summary" ]]; then
    printf '%s\n' "$training_summary"
    return 0
  fi

  if [[ -f "$baseline_metrics" ]]; then
    printf '%s\n' "$baseline_metrics"
    return 0
  fi

  return 1
}

extract_metric() {
  local run_dir="$1"
  case "$BEST_METRIC" in
    best_valid_loss|final_valid_loss|final_train_loss|final_train_eval_loss)
      local training_summary
      training_summary="$(ls -1t "$run_dir"/training_summary_*.json 2>/dev/null | head -n 1 || true)"
      if [[ -z "$training_summary" ]]; then
        return 1
      fi
      json_value "$training_summary" "$BEST_METRIC"
      ;;
    *)
      local baseline_metrics="$run_dir/eval/eeg_vs_baselines_metrics.json"
      if [[ ! -f "$baseline_metrics" ]]; then
        return 1
      fi
      json_value "$baseline_metrics" "$BEST_METRIC"
      ;;
  esac
}

is_better_metric() {
  local candidate="$1"
  local incumbent="$2"
  local mode="$3"
  python - "$candidate" "$incumbent" "$mode" <<'PY'
import math
import sys

candidate_raw, incumbent_raw, mode = sys.argv[1], sys.argv[2], sys.argv[3]

try:
    candidate = float(candidate_raw)
except ValueError:
    print("0")
    raise SystemExit(0)

if incumbent_raw == "":
    print("1")
    raise SystemExit(0)

try:
    incumbent = float(incumbent_raw)
except ValueError:
    print("1")
    raise SystemExit(0)

if math.isnan(candidate):
    print("0")
elif mode == "min":
    print("1" if candidate < incumbent else "0")
else:
    print("1" if candidate > incumbent else "0")
PY
}

mkdir -p "$REPO_ROOT/$OUTPUT_BASE"

RESULTS_TSV="$REPO_ROOT/$OUTPUT_BASE/sweep_results_$(timestamp).tsv"
printf 'run_name\trun_dir\tlr\tweight_decay\tbatch_size\teeg_normalization\toutput_dim\tepochs\tbest_valid_loss\tfinal_valid_loss\tSSIM_model\tLPIPS_model\n' > "$RESULTS_TSV"

best_run_name=""
best_run_dir=""
best_metric_value=""
total_runs=0

cd "$REPO_ROOT"

for lr in "${LRS[@]}"; do
  for weight_decay in "${WEIGHT_DECAYS[@]}"; do
    for batch_size in "${BATCH_SIZES[@]}"; do
      for eeg_normalization in "${EEG_NORMALIZATIONS[@]}"; do
        for output_dim in "${OUTPUT_DIMS[@]}"; do
          for epochs in "${EPOCHS[@]}"; do
            total_runs=$((total_runs + 1))
            run_name="sweep_$(timestamp)_lr$(sanitize_token "$lr")_wd$(sanitize_token "$weight_decay")_bs${batch_size}_norm${eeg_normalization}_k${output_dim}_ep${epochs}"
            run_dir="$REPO_ROOT/$OUTPUT_BASE/$run_name"

            cmd=(
              "$RUN_SCRIPT"
              --output-base "$OUTPUT_BASE"
              --run-name "$run_name"
              "${TRAIN_COMMON_ARGS[@]}"
              --lr "$lr"
              --weight-decay "$weight_decay"
              --batch-size "$batch_size"
              --eeg-normalization "$eeg_normalization"
              --output-dim "$output_dim"
              --epochs "$epochs"
            )

            if [[ "$SKIP_EVAL" -eq 1 ]]; then
              cmd+=(--skip-eval)
            else
              cmd+=(--eval)
              if [[ "${#EVAL_ARGS[@]}" -gt 0 ]]; then
                cmd+=("${EVAL_ARGS[@]}")
              fi
            fi

            if [[ "$SKIP_BASELINE" -eq 1 ]]; then
              cmd+=(--skip-baseline)
            else
              cmd+=(--baseline)
              if [[ "${#BASELINE_ARGS[@]}" -gt 0 ]]; then
                cmd+=("${BASELINE_ARGS[@]}")
              fi
            fi

            printf '\n[%d] Running %s\n' "$total_runs" "$run_name"
            printf 'Command:'
            printf ' %q' "${cmd[@]}"
            printf '\n'

            "${cmd[@]}"

            training_summary="$(ls -1t "$run_dir"/training_summary_*.json 2>/dev/null | head -n 1 || true)"
            baseline_metrics="$run_dir/eval/eeg_vs_baselines_metrics.json"

            best_valid_loss=""
            final_valid_loss=""
            ssim_model=""
            lpips_model=""

            if [[ -n "$training_summary" ]]; then
              best_valid_loss="$(json_value "$training_summary" "best_valid_loss")"
              final_valid_loss="$(json_value "$training_summary" "final_valid_loss")"
            fi

            if [[ -f "$baseline_metrics" ]]; then
              ssim_model="$(json_value "$baseline_metrics" "SSIM_model")"
              lpips_model="$(json_value "$baseline_metrics" "LPIPS_model")"
            fi

            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
              "$run_name" \
              "$run_dir" \
              "$lr" \
              "$weight_decay" \
              "$batch_size" \
              "$eeg_normalization" \
              "$output_dim" \
              "$epochs" \
              "$best_valid_loss" \
              "$final_valid_loss" \
              "$ssim_model" \
              "$lpips_model" >> "$RESULTS_TSV"

            metric_value="$(extract_metric "$run_dir" || true)"
            if [[ -n "$metric_value" ]]; then
              if [[ "$(is_better_metric "$metric_value" "$best_metric_value" "$BEST_MODE")" == "1" ]]; then
                best_metric_value="$metric_value"
                best_run_name="$run_name"
                best_run_dir="$run_dir"
              fi
            fi
          done
        done
      done
    done
  done
done

printf '\nSweep complete.\n'
printf 'Results table: %s\n' "$RESULTS_TSV"

if [[ -n "$best_run_name" ]]; then
  printf 'Best run by %s (%s): %s\n' "$BEST_METRIC" "$BEST_MODE" "$best_run_name"
  printf 'Best metric value: %s\n' "$best_metric_value"
  printf 'Best run directory: %s\n' "$best_run_dir"
else
  printf 'No best run could be determined. Check whether the chosen metric was produced.\n'
fi
