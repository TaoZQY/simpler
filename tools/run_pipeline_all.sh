#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

BASE_DIR="${BASE_DIR:-${PIPELINE_DIR}}"
if [[ "${ALLOW_RUN_ALL_OVERRIDES:-0}" == "1" ]]; then
  STRATEGIES="${STRATEGIES:-0}"
  BENCH_CASE_SET="${BENCH_CASE_SET:-all}"
else
  STRATEGIES="0"
  BENCH_CASE_SET="all"
fi
REPEAT="${REPEAT:-1}"
SCORE="${SCORE:-balanced}"
CMD="${CMD:-./tools/benchmark_rounds.sh -p a2a3 -r tensormap_and_ringbuffer -n 100}"
PYTHON_BIN="${PYTHON:-python3}"
BLOCK_DIM_VALUE="${SIMPLER_BLOCK_DIM:-${BLOCK_DIM:-}}"
EXTRA_ARGS=""

usage() {
  cat <<'USAGE'
run_pipeline_all.sh - run base plus selected pipeline strategies

Usage:
  ./tools/run_pipeline_all.sh
  ./tools/run_pipeline_all.sh --block-dim 20
  ./tools/run_pipeline_all.sh 20

Environment:
  STRATEGIES=0                  Fixed default: base plus strategy0 only
  SIMPLER_BLOCK_DIM=<unset>      Optional block_dim override for all cases
  BLOCK_DIM=<unset>              Alias used when SIMPLER_BLOCK_DIM is unset
  REPEAT=1                      Number of full base/strategy repeats
  BENCH_CASE_SET=all            Fixed default: full sweep after every change
  BENCH_CASE=benchmark_bgemm:Case0
  ALLOW_RUN_ALL_OVERRIDES=1      Allow STRATEGIES/BENCH_CASE_SET overrides
  CMD=./tools/benchmark_rounds.sh ...
  STREAM_OUTPUT=1               Stream child benchmark output to console
  PRESERVE_PIPELINE_DEFER_SUBMIT=1
                                 Keep caller's SIMPLER_PIPELINE_DEFER_SUBMIT

The run writes logs under results/pipeline/<run-id>/ and prints a final
"Unified Case Comparison" that first checks all-case base-vs-strategy
degradation, then compares O completion and Total for base-vs-strategy0
debug-layout analysis.
USAGE
}

while (($# > 0)); do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    -b|--block-dim)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: $1 requires a positive integer value." >&2
        exit 2
      fi
      BLOCK_DIM_VALUE="$2"
      shift 2
      ;;
    --block-dim=*)
      BLOCK_DIM_VALUE="${1#*=}"
      shift
      ;;
    [0-9]*)
      BLOCK_DIM_VALUE="$1"
      shift
      ;;
    *)
      echo "ERROR: unknown argument '$1'." >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -n "${BLOCK_DIM_VALUE}" ]]; then
  if ! [[ "${BLOCK_DIM_VALUE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: block_dim must be a positive integer, got '${BLOCK_DIM_VALUE}'." >&2
    exit 2
  fi
  export SIMPLER_BLOCK_DIM="${BLOCK_DIM_VALUE}"
fi

export BENCH_CASE_SET
if [[ "${PRESERVE_PIPELINE_DEFER_SUBMIT:-0}" != "1" ]]; then
  unset SIMPLER_PIPELINE_DEFER_SUBMIT
fi

if [[ "${STREAM_OUTPUT:-0}" == "1" ]]; then
  EXTRA_ARGS="--stream-output"
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run_pipeline_benchmark.py" \
  --base-dir "${BASE_DIR}" \
  --pipeline-dir "${PIPELINE_DIR}" \
  --strategies "${STRATEGIES}" \
  --repeat "${REPEAT}" \
  --score "${SCORE}" \
  --cmd "${CMD}" \
  ${EXTRA_ARGS}
