#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEVICE_ID=0
ROUNDS=100
PLATFORM=a2a3
RUNTIME=tensormap_and_ringbuffer
BLOCK_DIM="${SIMPLER_BLOCK_DIM:-20}"
OUT_DIR="${PROJECT_ROOT}/results/perf-compare-$(date +%Y%m%d-%H%M%S)"
SETUP_ENV=0
SETUP_CHIP="${SIMPLER_CHIP:-910b}"
SETUP_BUILD=0
EXTRA_ARGS=()

usage() {
    cat <<'USAGE'
run_2o2s_perf_compare.sh - run base vs 2O+2S strategy1 benchmark comparison

Usage:
  ./tools/run_2o2s_perf_compare.sh
  ./tools/run_2o2s_perf_compare.sh -n 100 -d 0
  ./tools/run_2o2s_perf_compare.sh --setup --chip 910b --block-dim 20

Options:
  -p, --platform PLATFORM    Platform passed to benchmark_rounds.sh. Default: a2a3
  -d, --device DEVICE        Device ID. Default: 0
  -n, --rounds ROUNDS        Rounds per case. Default: 100
  -r, --runtime RUNTIME      Runtime. Default: tensormap_and_ringbuffer
  --block-dim N              Export SIMPLER_BLOCK_DIM for both runs. Default: ${SIMPLER_BLOCK_DIM:-20}
  -o, --out-dir DIR          Result directory. Default: results/perf-compare-<timestamp>
  --setup                    Source tools/setup_910_env_and_build.sh before running.
  --chip 910b|910c           Chip preset for --setup. Default: 910b
  --build                    With --setup, rebuild editable package before benchmarking.
  -h, --help                 Show this help

All other arguments are forwarded to benchmark_rounds.sh.

Output:
  <out-dir>/base.log
  <out-dir>/strategy1_2o2s.log
USAGE
}

while (($# > 0)); do
    case "$1" in
        -p|--platform)
            PLATFORM="$2"
            shift 2
            ;;
        -d|--device)
            DEVICE_ID="$2"
            shift 2
            ;;
        -n|--rounds)
            ROUNDS="$2"
            shift 2
            ;;
        -r|--runtime)
            RUNTIME="$2"
            shift 2
            ;;
        --block-dim)
            BLOCK_DIM="$2"
            shift 2
            ;;
        --block-dim=*)
            BLOCK_DIM="${1#*=}"
            shift
            ;;
        -o|--out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        --setup)
            SETUP_ENV=1
            shift
            ;;
        --chip)
            SETUP_CHIP="$2"
            shift 2
            ;;
        --chip=*)
            SETUP_CHIP="${1#*=}"
            shift
            ;;
        --build)
            SETUP_BUILD=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if ! [[ "${ROUNDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --rounds must be a positive integer, got '${ROUNDS}'." >&2
    exit 2
fi

if ! [[ "${BLOCK_DIM}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --block-dim must be a positive integer, got '${BLOCK_DIM}'." >&2
    exit 2
fi

if [[ "${SETUP_ENV}" == "1" ]]; then
    setup_args=(--chip "${SETUP_CHIP}" --block-dim "${BLOCK_DIM}")
    if [[ "${SETUP_BUILD}" != "1" ]]; then
        setup_args+=(--no-build)
    fi
    # shellcheck source=/dev/null
    source "${PROJECT_ROOT}/tools/setup_910_env_and_build.sh" "${setup_args[@]}"
fi

mkdir -p "${OUT_DIR}"

echo "INFO: results will be written to ${OUT_DIR}"
echo "INFO: running base benchmark"
(
    cd "${PROJECT_ROOT}"
    export SIMPLER_BLOCK_DIM="${BLOCK_DIM}"
    unset SIMPLER_PIPELINE_STRATEGY
    unset SIMPLER_PIPELINE_DEFER_SUBMIT
    unset SIMPLER_AICPU_THREAD_NUM
    ./tools/benchmark_rounds.sh \
        -p "${PLATFORM}" \
        -d "${DEVICE_ID}" \
        -r "${RUNTIME}" \
        -n "${ROUNDS}" \
        -v \
        "${EXTRA_ARGS[@]}"
) 2>&1 | tee "${OUT_DIR}/base.log"

echo "INFO: running strategy1 2O+2S benchmark"
(
    cd "${PROJECT_ROOT}"
    export SIMPLER_BLOCK_DIM="${BLOCK_DIM}"
    export SIMPLER_PIPELINE_STRATEGY=1
    export SIMPLER_AICPU_THREAD_NUM=4
    unset SIMPLER_PIPELINE_DEFER_SUBMIT
    ./tools/benchmark_rounds.sh \
        -p "${PLATFORM}" \
        -d "${DEVICE_ID}" \
        -r "${RUNTIME}" \
        -n "${ROUNDS}" \
        -v \
        "${EXTRA_ARGS[@]}"
) 2>&1 | tee "${OUT_DIR}/strategy1_2o2s.log"

echo
echo "INFO: complete"
echo "INFO: base log:      ${OUT_DIR}/base.log"
echo "INFO: strategy1 log: ${OUT_DIR}/strategy1_2o2s.log"
echo "INFO: compare the 'Trimmed Avg' and final summary sections in the two logs."
