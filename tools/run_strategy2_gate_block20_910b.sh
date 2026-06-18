#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEVICE_ID="${DEVICE:-3}"
BLOCK_DIM_VALUE="${BLOCK_DIM:-${SIMPLER_BLOCK_DIM:-20}}"
ROUNDS="${ROUNDS:-100}"
PLATFORM="${PLATFORM:-a2a3}"
RUNTIME="${RUNTIME:-tensormap_and_ringbuffer}"
CHIP="${CHIP:-910b}"

cd "${PROJECT_ROOT}"

if [[ "${SKIP_SETUP:-0}" != "1" ]]; then
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/setup_910_env_and_build.sh" --chip "${CHIP}" --block-dim "${BLOCK_DIM_VALUE}"
else
  export SIMPLER_BLOCK_DIM="${BLOCK_DIM_VALUE}"
fi

unset BENCH_BASELINE_CASES
unset PTO2_ORCH_TO_SCHED
unset SIMPLER_PIPELINE_DEFER_SUBMIT
unset SIMPLER_AICPU_THREAD_NUM

DEVICE="${DEVICE_ID}" \
ALLOW_RUN_ALL_OVERRIDES=1 \
STRATEGIES=1,2 \
BENCH_CASE_SET=all \
CMD="./tools/benchmark_rounds.sh -p ${PLATFORM} -d ${DEVICE_ID} -r ${RUNTIME} -n ${ROUNDS} -v" \
  ./tools/run_pipeline_all.sh --block-dim "${BLOCK_DIM_VALUE}"
