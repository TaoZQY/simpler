#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# Benchmark wrapper: run examples on hardware and report per-round latency
# in timing columns:
#   - Host   from RunTiming (host_wall_us)
#   - Device from RunTiming (device_wall_us, AICPU orch mailbox)
#   - Total  device-log: full span across sched/orch events
#   - Sched  device-log: sched_start -> sched_end
#   - Orch   device-log: full O-thread span
#   - O1 Active device-log: primary O build/enqueue work before drain
#   - O1 Drain  device-log: primary O wait for submit pipeline drain
#   - O2 Active device-log: submit pipeline stage 1 record-processing work
#   - O2 Wall   device-log: submit pipeline stage 1 start -> end
#   - Pipe Enq/Flush device-log: O1 submit-pipeline enqueue/flush diagnostics
#
# Usage:
#   ./tools/benchmark_rounds.sh [-p <platform>] [-d <device>] [-n <rounds>] [-r <runtime>]
#
# By default this runs one tuning case. Set BENCH_CASE_SET=all for the full
# benchmark set, or BENCH_CASE=<example>:<case> to tune a different case.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------------------------------------------------------------------------
# Examples to benchmark and their case lists, per runtime.
# Key   = directory name under tests/st/<platform>/<runtime>/
# Value = comma-separated case names to run (empty string = run DEFAULT_CASE)
# ---------------------------------------------------------------------------

# --- tensormap_and_ringbuffer ---
TMR_ALL_EXAMPLE_ORDER=(
    alternating_matmul_add
    benchmark_bgemm
    paged_attention_unroll
    paged_attention_unroll_manual_scope
    batch_paged_attention
    spmd_paged_attention
)

BENCH_CASE_SET="${BENCH_CASE_SET:-single}"
BENCH_CASE="${BENCH_CASE:-paged_attention_unroll:Case1}"
BENCH_BASELINE_CASES="${BENCH_BASELINE_CASES:-}"
SINGLE_EXAMPLE=""
SINGLE_CASE_NAME=""

if [[ -n "${SIMPLER_PIPELINE_STRATEGY:-}" && -z "${SIMPLER_AICPU_THREAD_NUM:-}" ]]; then
    if [[ "${SIMPLER_PIPELINE_STRATEGY}" == "0" ||
          "${SIMPLER_PIPELINE_STRATEGY}" == "1" ||
          "${SIMPLER_PIPELINE_STRATEGY}" == "2" ||
          "${SIMPLER_PIPELINE_STRATEGY}" == "5" ]]; then
        export SIMPLER_AICPU_THREAD_NUM=4
    else
        export SIMPLER_AICPU_THREAD_NUM=6
    fi
fi

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
DEVICE_ID=0
ROUNDS=100
PLATFORM=a2a3
RUNTIME=tensormap_and_ringbuffer
VERBOSE=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
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
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        --help|-h)
            cat <<'USAGE'
benchmark_rounds.sh — run selected examples and report per-round timing from device logs

Usage:
  ./tools/benchmark_rounds.sh [-p <platform>] [-d <device>] [-n <rounds>] [-r <runtime>] [-v]

Options:
  -p, --platform Platform to run on (default: a2a3)
  -d, --device   Device ID (default: 0)
  -n, --rounds   Override number of rounds for each example (default: 100)
  -r, --runtime  Runtime to benchmark: tensormap_and_ringbuffer (default)
  -v, --verbose  Save detailed test_*.py output to a timestamped log file
  -h, --help     Show this help

All other options are passed through to the underlying `python test_*.py`
invocation (e.g. --case).

Defaults to one tuning case: paged_attention_unroll:Case1.

Environment:
  BENCH_CASE_SET=single|all  Select one case or the full benchmark set
                             (default: single)
  BENCH_CASE=example:Case    Case used when BENCH_CASE_SET=single
                             (default: paged_attention_unroll:Case1)
  BENCH_BASELINE_CASES=example:Case[,example:Case...]
                             Temporarily unset SIMPLER_PIPELINE_STRATEGY for
                             matching cases during a pipeline sweep. Use this
                             for workload-gate experiments only.

Output:
  Per-round and average latency (microseconds):
  Host, Device (from RunTiming) + Total, Sched, Orch, O1 Active,
  O1 Drain, O2 Active, O2 Wall, Pipe Enq, Pipe Flush (from device log when present).
USAGE
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Verbose logging setup
# ---------------------------------------------------------------------------
VERBOSE_LOG=""
if [[ $VERBOSE -eq 1 ]]; then
    mkdir -p "$PROJECT_ROOT/outputs"
    VERBOSE_LOG="$PROJECT_ROOT/outputs/benchmark_$(date +%Y%m%d_%H%M%S).log"
    echo "Verbose log: $VERBOSE_LOG"
fi

vlog() {
    if [[ -n "$VERBOSE_LOG" ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$VERBOSE_LOG"
    fi
}

# ---------------------------------------------------------------------------
# Derive arch from platform and set examples directories
# ---------------------------------------------------------------------------
# Search both examples/ (migrated tests) and tests/st/ (legacy tests)
ARCH="${PLATFORM%%sim}"  # strip "sim" suffix if present
EXAMPLES_DIRS=(
    "$PROJECT_ROOT/tests/st/${ARCH}/${RUNTIME}"
    "$PROJECT_ROOT/examples/${ARCH}/${RUNTIME}"
)

# Clock frequency (MHz) for converting cycle counts to microseconds
case "$PLATFORM" in
    a2a3) FREQ=50 ;;
    a5)   FREQ=1000 ;;
    *)    echo "ERROR: unsupported platform '$PLATFORM'. Use a2a3 or a5."; exit 1 ;;
esac

select_single_tmr_case() {
    local spec="$BENCH_CASE"
    local example="$spec"
    local case_name=""

    if [[ "$spec" == *:* ]]; then
        example="${spec%%:*}"
        case_name="${spec#*:}"
    fi
    if [[ -z "$example" ]]; then
        echo "ERROR: BENCH_CASE must be <example> or <example>:<case>."
        exit 1
    fi

    SINGLE_EXAMPLE="$example"
    SINGLE_CASE_NAME="$case_name"
}

tmr_all_case_list() {
    case "$1" in
        alternating_matmul_add) echo "Case1" ;;
        benchmark_bgemm) echo "Case0" ;;
        paged_attention_unroll) echo "Case1,Case2" ;;
        paged_attention_unroll_manual_scope) echo "Case1,Case2" ;;
        batch_paged_attention) echo "Case1" ;;
        spmd_paged_attention) echo "Case1,Case2" ;;
        *) echo "" ;;
    esac
}

case_list_for_example() {
    local example="$1"
    if [[ "$BENCH_CASE_SET" == "single" ]]; then
        if [[ "$example" == "$SINGLE_EXAMPLE" ]]; then
            echo "$SINGLE_CASE_NAME"
        else
            echo ""
        fi
    else
        tmr_all_case_list "$example"
    fi
}

case_uses_baseline_fallback() {
    local example="$1" case_name="${2:-}"
    [[ -z "$BENCH_BASELINE_CASES" ]] && return 1

    local specs spec
    IFS=',' read -ra specs <<< "$BENCH_BASELINE_CASES"
    for spec in "${specs[@]}"; do
        spec="${spec//[[:space:]]/}"
        [[ -z "$spec" ]] && continue
        if [[ "$spec" == "$example" || "$spec" == "$example:*" ]]; then
            return 0
        fi
        if [[ -n "$case_name" && "$spec" == "$example:$case_name" ]]; then
            return 0
        fi
    done
    return 1
}

# Select example cases and order based on runtime
case "$RUNTIME" in
    tensormap_and_ringbuffer)
        case "$BENCH_CASE_SET" in
            single)
                select_single_tmr_case
                EXAMPLE_ORDER=("$SINGLE_EXAMPLE")
                ;;
            all)
                EXAMPLE_ORDER=("${TMR_ALL_EXAMPLE_ORDER[@]}")
                ;;
            *)
                echo "ERROR: unknown BENCH_CASE_SET '$BENCH_CASE_SET'. Use single or all."
                exit 1
                ;;
        esac
        ;;
    *)
        echo "ERROR: unknown runtime '$RUNTIME'. Use tensormap_and_ringbuffer."
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# Resolve CANN device log directory: $ASCEND_WORK_PATH/log/debug or ~/ascend/log/debug
# ---------------------------------------------------------------------------
if [[ -n "${ASCEND_WORK_PATH:-}" ]]; then
    LOG_ROOT="$ASCEND_WORK_PATH/log/debug"
    if [[ ! -d "$LOG_ROOT" ]]; then
        LOG_ROOT="$HOME/ascend/log/debug"
    fi
else
    LOG_ROOT="$HOME/ascend/log/debug"
fi
DEVICE_LOG_DIR="$LOG_ROOT/device-${DEVICE_ID}"

# ---------------------------------------------------------------------------
# parse_timing <fw_stdout_file> <device_log_file>
#   Merge per-round timing from two sources into a timing table:
#     - Host (us)   from RunTiming (framework `_log_round_timings` stdout)
#     - Device (us) from RunTiming (AICPU mailbox orch_start/end)
#     - Total (us)  device-log: max(end) - min(start) across sched/orch events
#     - Sched (us)  device-log: sched_start -> sched_end
#     - Orch (us)   device-log: max O-thread end -> min O-thread start
#     - O1 Active (us) device-log: primary O build/enqueue work before drain
#     - O1 Drain (us)  device-log: primary O wait for submit pipeline drain
#     - O2 Active (us) device-log: submit pipeline stage 1 record-processing work
#     - O2 Wall (us)   device-log: submit pipeline stage 1 start -> end
#     - Pipe Enq (us)  device-log: task+scope submit-pipeline enqueue work
#     - Pipe Flush (us) device-log: submit-pipeline flush wait work
#     - O4Stage device-log: future four-stage O split cost/last-completion
# ---------------------------------------------------------------------------
parse_timing() {
    local fw_file="$1"
    local log_file="$2"

    # Slice device log to just timing lines (keeps awk input small).
    local dev_timing_file
    dev_timing_file=$(mktemp)
    trap 'rm -f -- "$dev_timing_file"' RETURN
    grep -E 'Thread [0-9]+: (sched_start|sched_end|orch_start|orch_end|orch_active_start|orch_stage_start|orch_stage_end|orch_pipeline_diag|orch_pipeline_dep_diag|orch_4stage|orch_active_detail|orch_submit_detail)' \
        "$log_file" > "$dev_timing_file" 2>/dev/null || true

    if [[ ! -s "$fw_file" && ! -s "$dev_timing_file" ]]; then
        echo "  (no benchmark timing data — was PTO2_PROFILING enabled?)"
        return 1
    fi

    awk -v freq="$FREQ" '
    function flush_round() {
        if (round >= 0 && max_end > 0 && min_start > 0) {
            total_results[round] = (max_end - min_start) / freq
            if (max_sched_end > 0 && min_sched_start > 0)
                sched_results[round] = (max_sched_end - min_sched_start) / freq
            if (max_orch_end > 0 && min_orch_start > 0)
                orch_results[round] = (max_orch_end - min_orch_start) / freq
            if (o1_active_end > 0 && o1_active_start > 0)
                o1_active_results[round] = (o1_active_end - o1_active_start) / freq
            if (o1_drain_end > 0 && o1_drain_start > 0)
                o1_drain_results[round] = (o1_drain_end - o1_drain_start) / freq
            if (o2_active_cost > 0)
                o2_active_results[round] = o2_active_cost
            if (o2_end > 0 && o2_start > 0)
                o2_wall_results[round] = (o2_end - o2_start) / freq
            if (pipe_enq_cost >= 0)
                pipe_enq_results[round] = pipe_enq_cost
            if (pipe_flush_cost >= 0)
                pipe_flush_results[round] = pipe_flush_cost
            if (pipe_task_enq_count >= 0)
                pipe_task_enq_count_results[round] = pipe_task_enq_count
            if (pipe_task_enq_batch_count >= 0)
                pipe_task_enq_batch_count_results[round] = pipe_task_enq_batch_count
            if (pipe_scope_enq_count >= 0)
                pipe_scope_enq_count_results[round] = pipe_scope_enq_count
            if (pipe_flush_count >= 0)
                pipe_flush_count_results[round] = pipe_flush_count
            if (pipe_compact_deferred_count >= 0)
                pipe_compact_deferred_count_results[round] = pipe_compact_deferred_count
            if (pipe_batch_sync_cost >= 0)
                pipe_batch_sync_results[round] = pipe_batch_sync_cost
            if (pipe_deferred_dep_cost >= 0)
                pipe_deferred_dep_results[round] = pipe_deferred_dep_cost
            if (pipe_deferred_fanin_cost >= 0)
                pipe_deferred_fanin_results[round] = pipe_deferred_fanin_cost
            if (pipe_publish_cost >= 0)
                pipe_publish_results[round] = pipe_publish_cost
            if (pipe_scope_release_cost >= 0)
                pipe_scope_release_results[round] = pipe_scope_release_cost
            if (pipe_scope_record_cost >= 0)
                pipe_scope_record_results[round] = pipe_scope_record_cost
            if (pipe_scope_producer_release_cost >= 0)
                pipe_scope_producer_release_results[round] = pipe_scope_producer_release_cost
            if (pipe_scope_ring_advance_cost >= 0)
                pipe_scope_ring_advance_results[round] = pipe_scope_ring_advance_cost
            if (pipe_scope_record_count >= 0)
                pipe_scope_record_count_results[round] = pipe_scope_record_count
            if (pipe_scope_release_count >= 0)
                pipe_scope_release_count_results[round] = pipe_scope_release_count
            if (pipe_scope_consumed_count >= 0)
                pipe_scope_consumed_count_results[round] = pipe_scope_consumed_count
            if (pipe_scope_advance_attempts >= 0)
                pipe_scope_advance_attempts_results[round] = pipe_scope_advance_attempts
            if (pipe_scope_advance_success >= 0)
                pipe_scope_advance_success_results[round] = pipe_scope_advance_success
            if (pipe_dep_explicit_cost >= 0)
                pipe_dep_explicit_results[round] = pipe_dep_explicit_cost
            if (pipe_dep_owner_emit_cost >= 0)
                pipe_dep_owner_emit_results[round] = pipe_dep_owner_emit_cost
            if (pipe_dep_lookup_cost >= 0)
                pipe_dep_lookup_results[round] = pipe_dep_lookup_cost
            if (pipe_dep_lookup_emit_cost >= 0)
                pipe_dep_lookup_emit_results[round] = pipe_dep_lookup_emit_cost
            if (pipe_dep_remove_cost >= 0)
                pipe_dep_remove_results[round] = pipe_dep_remove_cost
            if (pipe_dep_register_cost >= 0)
                pipe_dep_register_results[round] = pipe_dep_register_cost
            if (pipe_dep_register_insert_cost >= 0)
                pipe_dep_register_insert_results[round] = pipe_dep_register_insert_cost
            if (pipe_dep_explicit_count >= 0)
                pipe_dep_explicit_count_results[round] = pipe_dep_explicit_count
            if (pipe_dep_owner_emit_count >= 0)
                pipe_dep_owner_emit_count_results[round] = pipe_dep_owner_emit_count
            if (pipe_dep_lookup_count >= 0)
                pipe_dep_lookup_count_results[round] = pipe_dep_lookup_count
            if (pipe_dep_lookup_skip_count >= 0)
                pipe_dep_lookup_skip_count_results[round] = pipe_dep_lookup_skip_count
            if (pipe_dep_lookup_range_skip_count >= 0)
                pipe_dep_lookup_range_skip_count_results[round] = pipe_dep_lookup_range_skip_count
            if (pipe_dep_lookup_match_count >= 0)
                pipe_dep_lookup_match_count_results[round] = pipe_dep_lookup_match_count
            if (pipe_dep_lookup_emit_count >= 0)
                pipe_dep_lookup_emit_count_results[round] = pipe_dep_lookup_emit_count
            if (pipe_dep_remove_count >= 0)
                pipe_dep_remove_count_results[round] = pipe_dep_remove_count
            if (pipe_dep_register_count >= 0)
                pipe_dep_register_count_results[round] = pipe_dep_register_count
            if (pipe_dep_fanin_actual_count >= 0)
                pipe_dep_fanin_actual_count_results[round] = pipe_dep_fanin_actual_count
            if (front_stage_cost >= 0) {
                front_stage_cost_results[round] = front_stage_cost
                front_stage_end_results[round] = front_stage_serial_end
            }
            if (submitter_stage_cost >= 0) {
                submitter_stage_cost_results[round] = submitter_stage_cost
                submitter_stage_end_results[round] = submitter_stage_serial_end
            }
            if (tensormap_stage_cost >= 0) {
                tensormap_stage_cost_results[round] = tensormap_stage_cost
                tensormap_stage_end_results[round] = tensormap_stage_serial_end
            }
            if (updater_stage_cost >= 0) {
                updater_stage_cost_results[round] = updater_stage_cost
                updater_stage_end_results[round] = updater_stage_serial_end
            }
            if (active_detail_unclassified >= 0)
                active_detail_unclassified_results[round] = active_detail_unclassified
            if (active_detail_alloc >= 0)
                active_detail_alloc_results[round] = active_detail_alloc
            if (active_detail_submit_wrapper >= 0)
                active_detail_submit_wrapper_results[round] = active_detail_submit_wrapper
            if (active_detail_scope_end >= 0)
                active_detail_scope_end_results[round] = active_detail_scope_end
            if (active_detail_p_func >= 0)
                active_detail_p_func_results[round] = active_detail_p_func
            if (submit_detail_count >= 0) {
                submit_detail_count_results[round] = submit_detail_count
                submit_detail_deferred_results[round] = submit_detail_deferred
                submit_detail_sync_results[round] = submit_detail_sync
                submit_detail_heap_guard_results[round] = submit_detail_heap_guard
                submit_detail_tensors_results[round] = submit_detail_tensors
                submit_detail_scalars_results[round] = submit_detail_scalars
                submit_detail_explicit_deps_results[round] = submit_detail_explicit_deps
                submit_detail_output_bytes_results[round] = submit_detail_output_bytes
                submit_detail_layout_results[round] = submit_detail_layout
                submit_detail_prepare_results[round] = submit_detail_prepare
                submit_detail_depgen_results[round] = submit_detail_depgen
                submit_detail_sync_cost_results[round] = submit_detail_sync_cost
                submit_detail_explicit_cost_results[round] = submit_detail_explicit_cost
                submit_detail_lookup_results[round] = submit_detail_lookup
                submit_detail_register_results[round] = submit_detail_register
                submit_detail_payload_results[round] = submit_detail_payload
                submit_detail_descriptor_results[round] = submit_detail_descriptor
                submit_detail_deferred_meta_results[round] = submit_detail_deferred_meta
                submit_detail_enqueue_results[round] = submit_detail_enqueue
                submit_detail_return_tail_results[round] = submit_detail_return_tail
                if (submit_detail_prep_check >= 0)
                    submit_detail_prep_check_results[round] = submit_detail_prep_check
                if (submit_detail_prep_alloc >= 0)
                    submit_detail_prep_alloc_results[round] = submit_detail_prep_alloc
                if (submit_detail_prep_ptr >= 0)
                    submit_detail_prep_ptr_results[round] = submit_detail_prep_ptr
                if (submit_detail_prep_prefetch >= 0)
                    submit_detail_prep_prefetch_results[round] = submit_detail_prep_prefetch
                if (submit_detail_prep_slot >= 0)
                    submit_detail_prep_slot_results[round] = submit_detail_prep_slot
                if (submit_detail_prep_scope_push >= 0)
                    submit_detail_prep_scope_push_results[round] = submit_detail_prep_scope_push
                if (submit_detail_prep_alloc_task_wait >= 0)
                    submit_detail_prep_alloc_task_wait_results[round] = submit_detail_prep_alloc_task_wait
                if (submit_detail_prep_alloc_heap_wait >= 0)
                    submit_detail_prep_alloc_heap_wait_results[round] = submit_detail_prep_alloc_heap_wait
                if (submit_detail_prep_alloc_task_spins >= 0)
                    submit_detail_prep_alloc_task_spins_results[round] = submit_detail_prep_alloc_task_spins
                if (submit_detail_prep_alloc_heap_spins >= 0)
                    submit_detail_prep_alloc_heap_spins_results[round] = submit_detail_prep_alloc_heap_spins
                if (submit_detail_prep_alloc_progress >= 0)
                    submit_detail_prep_alloc_progress_results[round] = submit_detail_prep_alloc_progress
            }
            dev_count++
        }
    }
    function new_round() {
        flush_round()
        round++
        min_start = 0; max_end = 0
        min_sched_start = 0; max_sched_end = 0
        min_orch_start = 0; max_orch_end = 0
        o1_active_start = 0; o1_active_end = 0
        o1_drain_start = 0; o1_drain_end = 0
        o2_active_cost = 0
        o2_start = 0; o2_end = 0
        pipe_enq_cost = -1; pipe_flush_cost = -1
        pipe_batch_sync_cost = -1; pipe_deferred_dep_cost = -1; pipe_deferred_fanin_cost = -1; pipe_publish_cost = -1
        pipe_scope_release_cost = -1; pipe_scope_record_cost = -1; pipe_scope_producer_release_cost = -1; pipe_scope_ring_advance_cost = -1
        pipe_task_enq_count = -1; pipe_task_enq_batch_count = -1; pipe_scope_enq_count = -1; pipe_flush_count = -1
        pipe_compact_deferred_count = -1
        pipe_scope_record_count = -1; pipe_scope_release_count = -1; pipe_scope_consumed_count = -1
        pipe_scope_advance_attempts = -1; pipe_scope_advance_success = -1
        pipe_dep_explicit_cost = -1; pipe_dep_owner_emit_cost = -1; pipe_dep_lookup_cost = -1
        pipe_dep_lookup_emit_cost = -1; pipe_dep_remove_cost = -1; pipe_dep_register_cost = -1
        pipe_dep_register_insert_cost = -1
        pipe_dep_explicit_count = -1; pipe_dep_owner_emit_count = -1; pipe_dep_lookup_count = -1
        pipe_dep_lookup_skip_count = -1; pipe_dep_lookup_range_skip_count = -1
        pipe_dep_lookup_match_count = -1; pipe_dep_lookup_emit_count = -1; pipe_dep_remove_count = -1
        pipe_dep_register_count = -1; pipe_dep_fanin_actual_count = -1
        front_stage_cost = -1; submitter_stage_cost = -1; tensormap_stage_cost = -1; updater_stage_cost = -1
        front_stage_serial_end = -1; submitter_stage_serial_end = -1; tensormap_stage_serial_end = -1; updater_stage_serial_end = -1
        active_detail_unclassified = -1; active_detail_alloc = -1; active_detail_submit_wrapper = -1; active_detail_scope_end = -1; active_detail_p_func = -1
        submit_detail_count = -1; submit_detail_deferred = -1; submit_detail_sync = -1; submit_detail_heap_guard = -1; submit_detail_tensors = -1; submit_detail_scalars = -1; submit_detail_explicit_deps = -1; submit_detail_output_bytes = -1
        submit_detail_layout = -1; submit_detail_prepare = -1; submit_detail_depgen = -1; submit_detail_payload = -1; submit_detail_descriptor = -1; submit_detail_deferred_meta = -1; submit_detail_enqueue = -1; submit_detail_return_tail = -1
        submit_detail_sync_cost = -1; submit_detail_explicit_cost = -1; submit_detail_lookup = -1; submit_detail_register = -1
        submit_detail_prep_check = -1; submit_detail_prep_alloc = -1; submit_detail_prep_ptr = -1; submit_detail_prep_prefetch = -1; submit_detail_prep_slot = -1; submit_detail_prep_scope_push = -1
        submit_detail_prep_alloc_task_wait = -1; submit_detail_prep_alloc_heap_wait = -1; submit_detail_prep_alloc_task_spins = -1; submit_detail_prep_alloc_heap_spins = -1; submit_detail_prep_alloc_progress = -1
        delete sched_seen
        delete orch_seen
        delete orch_active_seen
        delete orch_stage_seen
    }
    function trimmed(label, arr, n, trim,    i, j, k, tc, ts) {
        for (i = 2; i <= n; i++) {
            k = arr[i]; j = i - 1
            while (j >= 1 && arr[j] > k) { arr[j+1] = arr[j]; j-- }
            arr[j+1] = k
        }
        tc = n - 2 * trim; ts = 0
        for (i = trim + 1; i <= n - trim; i++) ts += arr[i]
        printf "  %s Trimmed Avg: %.1f us  (dropped %d low + %d high, %d rounds used)\n", \
               label, ts / tc, trim, trim, tc
    }
    BEGIN {
        round = 0; dev_count = 0
        min_start = 0; max_end = 0
        min_sched_start = 0; max_sched_end = 0
        min_orch_start = 0; max_orch_end = 0
        o1_active_start = 0; o1_active_end = 0
        o1_drain_start = 0; o1_drain_end = 0
        o2_active_cost = 0
        o2_start = 0; o2_end = 0
        has_sched = 0; has_orch_end = 0; has_o1_active = 0; has_o1_drain = 0; has_o2_active = 0; has_o2_wall = 0
        has_pipe_enq = 0; has_pipe_flush = 0; has_pipe_counts = 0; has_pipe_detail = 0; has_pipe_dep_detail = 0
        fw_n = 0; in_fw = 0
    }
    # First file: framework `_log_round_timings` stdout (Host / Device per round).
    # Header may be concatenated with the test runner status line (uses end=""),
    # so anchor on "Round...Host (us)" anywhere on the line, not column 0.
    FNR == NR {
        if (match($0, /Round +Host \(us\)/))   { in_fw = 1; next }
        if (in_fw && /^  -+$/)                 next
        if (in_fw && /Avg Host:/)              { in_fw = 0; next }
        if (in_fw && NF == 0)                  { in_fw = 0; next }
        if (in_fw && match($0, /^  +([0-9]+) +([0-9.]+)( +([0-9.]+))?/, m)) {
            r = m[1] + 0
            fw_host[r] = m[2] + 0
            if (m[4] != "") fw_dev[r] = m[4] + 0
            if (r + 1 > fw_n) fw_n = r + 1
        }
        next
    }
    # Second file: device-log timing lines
    /sched_start=/ {
        match($0, /Thread ([0-9]+):/, tm)
        tid = tm[1] + 0
        if (tid in sched_seen) new_round()
        sched_seen[tid] = 1
        has_sched = 1
        match($0, /sched_start=([0-9]+)/, m)
        val = m[1] + 0
        if (min_sched_start == 0 || val < min_sched_start) min_sched_start = val
        if (min_start == 0 || val < min_start) min_start = val
    }
    /orch_start=/ {
        match($0, /Thread ([0-9]+):/, tm)
        tid = tm[1] + 0
        if (tid in orch_seen) new_round()
        orch_seen[tid] = 1
        match($0, /orch_start=([0-9]+)/, m)
        val = m[1] + 0
        if (min_orch_start == 0 || val < min_orch_start) min_orch_start = val
        if (min_start == 0 || val < min_start) min_start = val
    }
    /orch_active_start=/ {
        match($0, /Thread ([0-9]+):/, tm)
        tid = tm[1] + 0
        if (tid in orch_active_seen) new_round()
        orch_active_seen[tid] = 1
        match($0, /orch_active_start=([0-9]+)/, m)
        o1_active_start = m[1] + 0
        match($0, /orch_active_end=([0-9]+)/, m)
        o1_active_end = m[1] + 0
        match($0, /orch_drain_start=([0-9]+)/, m)
        o1_drain_start = m[1] + 0
        match($0, /orch_drain_end=([0-9]+)/, m)
        o1_drain_end = m[1] + 0
        has_o1_active = 1
        has_o1_drain = 1
    }
    /sched_end[^=]*=/ {
        match($0, /sched_end[^=]*=([0-9]+)/, m)
        val = m[1] + 0
        if (val > max_sched_end) max_sched_end = val
        if (val > max_end) max_end = val
    }
    /orch_end=/ {
        match($0, /orch_end=([0-9]+)/, m)
        val = m[1] + 0
        has_orch_end = 1
        if (val > max_orch_end) max_orch_end = val
        if (val > max_end) max_end = val
    }
    /orch_stage_start=/ {
        stage = 0
        if (match($0, /orch_stage=([0-9]+)/, sm)) stage = sm[1] + 0
        if (stage == 1) {
            if (stage in orch_stage_seen) new_round()
            orch_stage_seen[stage] = 1
            match($0, /orch_stage_start=([0-9]+)/, m)
            val = m[1] + 0
            o2_start = val
            if (match($0, /orch_stage_active_cost=([0-9.]+)us/, am)) {
                o2_active_cost = am[1] + 0
                has_o2_active = 1
            }
            has_o2_wall = 1
            if (min_orch_start == 0 || val < min_orch_start) min_orch_start = val
            if (min_start == 0 || val < min_start) min_start = val
        }
    }
    /orch_stage_end=/ {
        stage = 0
        if (match($0, /orch_stage=([0-9]+)/, sm)) stage = sm[1] + 0
        match($0, /orch_stage_end=([0-9]+)/, m)
        val = m[1] + 0
        if (stage == 1) {
            o2_end = val
            if (match($0, /orch_stage_active_cost=([0-9.]+)us/, am)) {
                o2_active_cost = am[1] + 0
                has_o2_active = 1
            }
            has_o2_wall = 1
            if (val > max_orch_end) max_orch_end = val
        }
        if (val > max_end) max_end = val
    }
    /orch_pipeline_diag/ {
        if (match($0, /task_enq_count=([0-9]+)/, m)) {
            pipe_task_enq_count = m[1] + 0
            has_pipe_counts = 1
        }
        if (match($0, /task_enq_batches=([0-9]+)/, m)) {
            pipe_task_enq_batch_count = m[1] + 0
            has_pipe_counts = 1
        }
        if (match($0, /task_enq_cost=([0-9.]+)us/, m)) {
            pipe_enq_cost = m[1] + 0
            has_pipe_enq = 1
        }
        if (match($0, /scope_enq_count=([0-9]+)/, m)) {
            pipe_scope_enq_count = m[1] + 0
            has_pipe_counts = 1
        }
        if (match($0, /scope_enq_cost=([0-9.]+)us/, m)) {
            pipe_enq_cost += m[1] + 0
            has_pipe_enq = 1
        }
        if (match($0, /flush_count=([0-9]+)/, m)) {
            pipe_flush_count = m[1] + 0
            has_pipe_counts = 1
        }
        if (match($0, /flush_cost=([0-9.]+)us/, m)) {
            pipe_flush_cost = m[1] + 0
            has_pipe_flush = 1
        }
        if (match($0, /batch_sync_cost=([0-9.]+)us/, m)) {
            pipe_batch_sync_cost = m[1] + 0
            has_pipe_detail = 1
        }
        if (match($0, /deferred_dep_cost=([0-9.]+)us/, m)) {
            pipe_deferred_dep_cost = m[1] + 0
            has_pipe_detail = 1
        }
        if (match($0, /deferred_fanin_cost=([0-9.]+)us/, m)) {
            pipe_deferred_fanin_cost = m[1] + 0
            has_pipe_detail = 1
        }
        if (match($0, /publish_cost=([0-9.]+)us/, m)) {
            pipe_publish_cost = m[1] + 0
            has_pipe_detail = 1
        }
        if (match($0, /scope_release_cost=([0-9.]+)us/, m)) {
            pipe_scope_release_cost = m[1] + 0
            has_pipe_detail = 1
        }
        if (match($0, /scope_record_cost=([0-9.]+)us/, m)) {
            pipe_scope_record_cost = m[1] + 0
            has_pipe_detail = 1
        }
        if (match($0, /scope_producer_release_cost=([0-9.]+)us/, m)) {
            pipe_scope_producer_release_cost = m[1] + 0
            has_pipe_detail = 1
        }
        if (match($0, /scope_ring_advance_cost=([0-9.]+)us/, m)) {
            pipe_scope_ring_advance_cost = m[1] + 0
            has_pipe_detail = 1
        }
        if (match($0, /scope_record_count=([0-9]+)/, m)) {
            pipe_scope_record_count = m[1] + 0
            has_pipe_detail = 1
        }
        if (match($0, /scope_release_count=([0-9]+)/, m)) {
            pipe_scope_release_count = m[1] + 0
            has_pipe_detail = 1
        }
        if (match($0, /scope_consumed_count=([0-9]+)/, m)) {
            pipe_scope_consumed_count = m[1] + 0
            has_pipe_detail = 1
        }
        if (match($0, /scope_advance_attempts=([0-9]+)/, m)) {
            pipe_scope_advance_attempts = m[1] + 0
            has_pipe_detail = 1
        }
        if (match($0, /scope_advance_success=([0-9]+)/, m)) {
            pipe_scope_advance_success = m[1] + 0
            has_pipe_detail = 1
        }
        if (match($0, /compact_deferred=([0-9]+)/, m)) {
            pipe_compact_deferred_count = m[1] + 0
            has_pipe_counts = 1
        }
    }
    /orch_pipeline_dep_diag/ {
        if (match($0, /dep_explicit_cost=([0-9.]+)us/, m)) {
            pipe_dep_explicit_cost = m[1] + 0
            has_pipe_dep_detail = 1
        }
        if (match($0, /dep_owner_emit_cost=([0-9.]+)us/, m)) {
            pipe_dep_owner_emit_cost = m[1] + 0
            has_pipe_dep_detail = 1
        }
        if (match($0, /dep_lookup_cost=([0-9.]+)us/, m)) {
            pipe_dep_lookup_cost = m[1] + 0
            has_pipe_dep_detail = 1
        }
        if (match($0, /dep_lookup_emit_cost=([0-9.]+)us/, m)) {
            pipe_dep_lookup_emit_cost = m[1] + 0
            has_pipe_dep_detail = 1
        }
        if (match($0, /dep_remove_cost=([0-9.]+)us/, m)) {
            pipe_dep_remove_cost = m[1] + 0
            has_pipe_dep_detail = 1
        }
        if (match($0, /dep_register_cost=([0-9.]+)us/, m)) {
            pipe_dep_register_cost = m[1] + 0
            has_pipe_dep_detail = 1
        }
        if (match($0, /dep_register_insert_cost=([0-9.]+)us/, m)) {
            pipe_dep_register_insert_cost = m[1] + 0
            has_pipe_dep_detail = 1
        }
        if (match($0, /dep_explicit_count=([0-9]+)/, m)) {
            pipe_dep_explicit_count = m[1] + 0
            has_pipe_dep_detail = 1
        }
        if (match($0, /dep_owner_emit_count=([0-9]+)/, m)) {
            pipe_dep_owner_emit_count = m[1] + 0
            has_pipe_dep_detail = 1
        }
        if (match($0, /dep_lookup_count=([0-9]+)/, m)) {
            pipe_dep_lookup_count = m[1] + 0
            has_pipe_dep_detail = 1
        }
        if (match($0, /dep_lookup_skip_count=([0-9]+)/, m)) {
            pipe_dep_lookup_skip_count = m[1] + 0
            has_pipe_dep_detail = 1
        }
        if (match($0, /dep_lookup_range_skip_count=([0-9]+)/, m)) {
            pipe_dep_lookup_range_skip_count = m[1] + 0
            has_pipe_dep_detail = 1
        }
        if (match($0, /dep_lookup_match_count=([0-9]+)/, m)) {
            pipe_dep_lookup_match_count = m[1] + 0
            has_pipe_dep_detail = 1
        }
        if (match($0, /dep_lookup_emit_count=([0-9]+)/, m)) {
            pipe_dep_lookup_emit_count = m[1] + 0
            has_pipe_dep_detail = 1
        }
        if (match($0, /dep_remove_count=([0-9]+)/, m)) {
            pipe_dep_remove_count = m[1] + 0
            has_pipe_dep_detail = 1
        }
        if (match($0, /dep_register_count=([0-9]+)/, m)) {
            pipe_dep_register_count = m[1] + 0
            has_pipe_dep_detail = 1
        }
        if (match($0, /dep_fanin_actual_count=([0-9]+)/, m)) {
            pipe_dep_fanin_actual_count = m[1] + 0
            has_pipe_dep_detail = 1
        }
    }
    /orch_4stage/ {
        stage_name = ""
        if (match($0, /name=([A-Za-z0-9_]+)/, nm)) {
            stage_name = nm[1]
        }
        stage_cost = -1
        if (match($0, /cost=([0-9.]+)us/, cm)) {
            stage_cost = cm[1] + 0
        }
        stage_serial_end = -1
        if (match($0, /serial_end=([0-9.]+)us/, em)) {
            stage_serial_end = em[1] + 0
        }
        has_four_stage = 1
        if (stage_name == "front") {
            front_stage_cost = stage_cost
            front_stage_serial_end = stage_serial_end
        } else if (stage_name == "submitter") {
            submitter_stage_cost = stage_cost
            submitter_stage_serial_end = stage_serial_end
        } else if (stage_name == "tensormap") {
            tensormap_stage_cost = stage_cost
            tensormap_stage_serial_end = stage_serial_end
        } else if (stage_name == "updater") {
            updater_stage_cost = stage_cost
            updater_stage_serial_end = stage_serial_end
        }
    }
    /orch_active_detail/ {
        has_active_detail = 1
        if (match($0, /unclassified=([0-9.]+)us/, m)) {
            active_detail_unclassified = m[1] + 0
        }
        if (match($0, /submit_wrapper_cost=([0-9.]+)us/, m)) {
            active_detail_submit_wrapper = m[1] + 0
        }
        if (match($0, /alloc_cost=([0-9.]+)us/, m)) {
            active_detail_alloc = m[1] + 0
        }
        if (match($0, /scope_end_cost=([0-9.]+)us/, m)) {
            active_detail_scope_end = m[1] + 0
        }
        if (match($0, /p_func_cost=([0-9.]+)us/, m)) {
            active_detail_p_func = m[1] + 0
        }
    }
    /orch_submit_detail/ {
        has_submit_detail = 1
        if (match($0, /count=([0-9]+)/, m)) {
            submit_detail_count = m[1] + 0
        }
        if (match($0, /deferred=([0-9]+)/, m)) {
            submit_detail_deferred = m[1] + 0
        }
        if (match($0, /sync=([0-9]+)/, m)) {
            submit_detail_sync = m[1] + 0
        }
        if (match($0, /heap_guard=([0-9]+)/, m)) {
            submit_detail_heap_guard = m[1] + 0
        }
        if (match($0, /tensors=([0-9]+)/, m)) {
            submit_detail_tensors = m[1] + 0
        }
        if (match($0, /scalars=([0-9]+)/, m)) {
            submit_detail_scalars = m[1] + 0
        }
        if (match($0, /explicit_deps=([0-9]+)/, m)) {
            submit_detail_explicit_deps = m[1] + 0
        }
        if (match($0, /output_bytes=([0-9]+)/, m)) {
            submit_detail_output_bytes = m[1] + 0
        }
        if (match($0, /layout=([0-9.]+)us/, m)) {
            submit_detail_layout = m[1] + 0
        }
        if (match($0, /prepare=([0-9.]+)us/, m)) {
            submit_detail_prepare = m[1] + 0
        }
        if (match($0, /depgen=([0-9.]+)us/, m)) {
            submit_detail_depgen = m[1] + 0
        }
        if (match($0, / sync=([0-9.]+)us/, m)) {
            submit_detail_sync_cost = m[1] + 0
        }
        if (match($0, / explicit=([0-9.]+)us/, m)) {
            submit_detail_explicit_cost = m[1] + 0
        }
        if (match($0, / lookup=([0-9.]+)us/, m)) {
            submit_detail_lookup = m[1] + 0
        }
        if (match($0, / register=([0-9.]+)us/, m)) {
            submit_detail_register = m[1] + 0
        }
        if (match($0, /payload=([0-9.]+)us/, m)) {
            submit_detail_payload = m[1] + 0
        }
        if (match($0, /descriptor=([0-9.]+)us/, m)) {
            submit_detail_descriptor = m[1] + 0
        }
        if (match($0, /deferred_meta=([0-9.]+)us/, m)) {
            submit_detail_deferred_meta = m[1] + 0
        }
        if (match($0, /enqueue=([0-9.]+)us/, m)) {
            submit_detail_enqueue = m[1] + 0
        }
        if (match($0, /return_tail=([0-9.]+)us/, m)) {
            submit_detail_return_tail = m[1] + 0
        }
        if (match($0, /prep_check=([0-9.]+)us/, m)) {
            submit_detail_prep_check = m[1] + 0
        }
        if (match($0, /prep_alloc=([0-9.]+)us/, m)) {
            submit_detail_prep_alloc = m[1] + 0
        }
        if (match($0, /prep_ptr=([0-9.]+)us/, m)) {
            submit_detail_prep_ptr = m[1] + 0
        }
        if (match($0, /prep_prefetch=([0-9.]+)us/, m)) {
            submit_detail_prep_prefetch = m[1] + 0
        }
        if (match($0, /prep_slot=([0-9.]+)us/, m)) {
            submit_detail_prep_slot = m[1] + 0
        }
        if (match($0, /prep_scope_push=([0-9.]+)us/, m)) {
            submit_detail_prep_scope_push = m[1] + 0
        }
        if (match($0, /prep_alloc_task_wait=([0-9.]+)us/, m)) {
            submit_detail_prep_alloc_task_wait = m[1] + 0
        }
        if (match($0, /prep_alloc_heap_wait=([0-9.]+)us/, m)) {
            submit_detail_prep_alloc_heap_wait = m[1] + 0
        }
        if (match($0, /prep_alloc_task_spins=([0-9]+)/, m)) {
            submit_detail_prep_alloc_task_spins = m[1] + 0
        }
        if (match($0, /prep_alloc_heap_spins=([0-9]+)/, m)) {
            submit_detail_prep_alloc_heap_spins = m[1] + 0
        }
        if (match($0, /prep_alloc_progress=([0-9]+)/, m)) {
            submit_detail_prep_alloc_progress = m[1] + 0
        }
    }
    END {
        flush_round()

        show_host  = (fw_n > 0)
        show_dev   = 0
        for (i = 0; i < fw_n; i++) if ((i in fw_dev) && fw_dev[i] > 0) { show_dev = 1; break }
        has_total  = (dev_count > 0)
        show_sched = has_sched
        show_orch  = has_orch_end
        show_o1_active = has_o1_active
        show_o1_drain  = has_o1_drain
        show_o2_active = has_o2_active
        show_o2_wall   = has_o2_wall
        show_pipe_enq = has_pipe_enq
        show_pipe_flush = has_pipe_flush
        show_pipe_counts = has_pipe_counts
        show_pipe_detail = has_pipe_detail
        show_four_stage = has_four_stage
        show_active_detail = has_active_detail

        if (!has_total && fw_n == 0) {
            print "  (no benchmark timing data — was PTO2_PROFILING enabled?)"
            exit 1
        }

        n_rounds = (dev_count > fw_n) ? dev_count : fw_n

        # Header / separator
        hdr = sprintf("  %-6s", "Round")
        sep = sprintf("  %-6s", "------")
        if (show_host)  { hdr = hdr sprintf("  %12s", "Host (us)");   sep = sep sprintf("  %12s", "------------") }
        if (show_dev)   { hdr = hdr sprintf("  %12s", "Device (us)"); sep = sep sprintf("  %12s", "------------") }
        if (has_total)  { hdr = hdr sprintf("  %12s", "Total (us)");  sep = sep sprintf("  %12s", "------------") }
        if (show_sched) { hdr = hdr sprintf("  %12s", "Sched (us)");  sep = sep sprintf("  %12s", "------------") }
        if (show_orch)  { hdr = hdr sprintf("  %12s", "Orch (us)");   sep = sep sprintf("  %12s", "------------") }
        if (show_o1_active) { hdr = hdr sprintf("  %12s", "O1 Act (us)"); sep = sep sprintf("  %12s", "------------") }
        if (show_o1_drain)  { hdr = hdr sprintf("  %12s", "O1 Drain");    sep = sep sprintf("  %12s", "------------") }
        if (show_o2_active) { hdr = hdr sprintf("  %12s", "O2 Act (us)"); sep = sep sprintf("  %12s", "------------") }
        if (show_o2_wall)   { hdr = hdr sprintf("  %12s", "O2 Wall");     sep = sep sprintf("  %12s", "------------") }
        if (show_pipe_enq)   { hdr = hdr sprintf("  %12s", "Pipe Enq");    sep = sep sprintf("  %12s", "------------") }
        if (show_pipe_flush) { hdr = hdr sprintf("  %12s", "Pipe Flush");  sep = sep sprintf("  %12s", "------------") }
        if (show_pipe_counts){ hdr = hdr sprintf("  %8s  %8s  %8s  %8s  %8s", "TaskEnq", "TaskBat", "ScopeEnq", "FlushN", "Compact"); sep = sep sprintf("  %8s  %8s  %8s  %8s  %8s", "--------", "--------", "--------", "--------", "--------") }
        if (show_four_stage){ hdr = hdr sprintf("  %10s  %10s  %10s  %10s", "FSerEnd", "SSerEnd", "TMSerEnd", "USerEnd"); sep = sep sprintf("  %10s  %10s  %10s  %10s", "----------", "----------", "----------", "----------") }
        print hdr; print sep

        cnt_host = 0; cnt_dev = 0; cnt_tot = 0; cnt_sch = 0; cnt_orc = 0; cnt_o1a = 0; cnt_o1d = 0; cnt_o2a = 0; cnt_o2w = 0; cnt_pe = 0; cnt_pf = 0; cnt_pc = 0; cnt_batch = 0; cnt_pipe_detail = 0
        sum_host = 0; sum_dev = 0; sum_tot = 0; sum_sch = 0; sum_orc = 0; sum_o1a = 0; sum_o1d = 0; sum_o2a = 0; sum_o2w = 0; sum_pe = 0; sum_pf = 0; sum_task_enq = 0; sum_task_batch = 0; sum_scope_enq = 0; sum_flush_count = 0; sum_compact_deferred = 0
        sum_pipe_batch_sync = 0; sum_pipe_deferred_dep = 0; sum_pipe_deferred_fanin = 0; sum_pipe_publish = 0; sum_pipe_scope_release = 0
        sum_pipe_scope_record = 0; sum_pipe_scope_producer_release = 0; sum_pipe_scope_ring_advance = 0
        sum_pipe_scope_record_count = 0; sum_pipe_scope_release_count = 0; sum_pipe_scope_consumed_count = 0; sum_pipe_scope_advance_attempts = 0; sum_pipe_scope_advance_success = 0
        sum_pipe_dep_explicit = 0; sum_pipe_dep_owner_emit = 0; sum_pipe_dep_lookup = 0; sum_pipe_dep_lookup_emit = 0
        sum_pipe_dep_remove = 0; sum_pipe_dep_register = 0; sum_pipe_dep_register_insert = 0; cnt_pipe_dep_detail = 0
        sum_pipe_dep_explicit_count = 0; sum_pipe_dep_owner_emit_count = 0; sum_pipe_dep_lookup_count = 0
        sum_pipe_dep_lookup_skip_count = 0; sum_pipe_dep_lookup_range_skip_count = 0
        sum_pipe_dep_lookup_match_count = 0; sum_pipe_dep_lookup_emit_count = 0; sum_pipe_dep_remove_count = 0
        sum_pipe_dep_register_count = 0; sum_pipe_dep_fanin_actual_count = 0
        sum_front_cost = 0; sum_submitter_cost = 0; sum_tensormap_cost = 0; sum_updater_cost = 0
        sum_front_end = 0; sum_submitter_end = 0; sum_tensormap_end = 0; sum_updater_end = 0; cnt_four = 0
        sum_active_unclassified = 0; sum_active_alloc = 0; sum_active_submit_wrapper = 0; sum_active_scope_end = 0; sum_active_p_func = 0; cnt_active_detail = 0
        sum_submit_count = 0; sum_submit_deferred = 0; sum_submit_sync = 0; sum_submit_heap_guard = 0; sum_submit_tensors = 0; sum_submit_scalars = 0; sum_submit_explicit_deps = 0; sum_submit_output_bytes = 0
        sum_submit_layout = 0; sum_submit_prepare = 0; sum_submit_depgen = 0; sum_submit_sync_cost = 0; sum_submit_explicit_cost = 0; sum_submit_lookup = 0; sum_submit_register = 0
        sum_submit_payload = 0; sum_submit_descriptor = 0; sum_submit_deferred_meta = 0; sum_submit_enqueue = 0; sum_submit_return_tail = 0
        sum_submit_prep_check = 0; sum_submit_prep_alloc = 0; sum_submit_prep_ptr = 0; sum_submit_prep_prefetch = 0; sum_submit_prep_slot = 0; sum_submit_prep_scope_push = 0; cnt_submit_detail = 0; cnt_submit_prep_detail = 0
        sum_submit_prep_alloc_task_wait = 0; sum_submit_prep_alloc_heap_wait = 0; sum_submit_prep_alloc_task_spins = 0; sum_submit_prep_alloc_heap_spins = 0; sum_submit_prep_alloc_progress = 0; cnt_submit_alloc_detail = 0

        for (i = 0; i < n_rounds; i++) {
            row = sprintf("  %-6d", i)
            if (show_host) {
                if (i in fw_host) {
                    row = row sprintf("  %12.1f", fw_host[i])
                    sum_host += fw_host[i]; cnt_host++; host_arr[cnt_host] = fw_host[i]
                } else row = row sprintf("  %12s", "-")
            }
            if (show_dev) {
                if ((i in fw_dev) && fw_dev[i] > 0) {
                    row = row sprintf("  %12.1f", fw_dev[i])
                    sum_dev += fw_dev[i]; cnt_dev++; dev_arr[cnt_dev] = fw_dev[i]
                } else row = row sprintf("  %12s", "-")
            }
            if (has_total) {
                if (i in total_results) {
                    row = row sprintf("  %12.1f", total_results[i])
                    sum_tot += total_results[i]; cnt_tot++; tot_arr[cnt_tot] = total_results[i]
                } else row = row sprintf("  %12s", "-")
            }
            if (show_sched) {
                if (i in sched_results) {
                    row = row sprintf("  %12.1f", sched_results[i])
                    sum_sch += sched_results[i]; cnt_sch++; sch_arr[cnt_sch] = sched_results[i]
                } else row = row sprintf("  %12s", "-")
            }
            if (show_orch) {
                if (i in orch_results) {
                    row = row sprintf("  %12.1f", orch_results[i])
                    sum_orc += orch_results[i]; cnt_orc++; orc_arr[cnt_orc] = orch_results[i]
                } else row = row sprintf("  %12s", "-")
            }
            if (show_o1_active) {
                if (i in o1_active_results) {
                    row = row sprintf("  %12.1f", o1_active_results[i])
                    sum_o1a += o1_active_results[i]; cnt_o1a++; o1a_arr[cnt_o1a] = o1_active_results[i]
                } else row = row sprintf("  %12s", "-")
            }
            if (show_o1_drain) {
                if (i in o1_drain_results) {
                    row = row sprintf("  %12.1f", o1_drain_results[i])
                    sum_o1d += o1_drain_results[i]; cnt_o1d++; o1d_arr[cnt_o1d] = o1_drain_results[i]
                } else row = row sprintf("  %12s", "-")
            }
            if (show_o2_active) {
                if (i in o2_active_results) {
                    row = row sprintf("  %12.1f", o2_active_results[i])
                    sum_o2a += o2_active_results[i]; cnt_o2a++; o2a_arr[cnt_o2a] = o2_active_results[i]
                } else row = row sprintf("  %12s", "-")
            }
            if (show_o2_wall) {
                if (i in o2_wall_results) {
                    row = row sprintf("  %12.1f", o2_wall_results[i])
                    sum_o2w += o2_wall_results[i]; cnt_o2w++; o2w_arr[cnt_o2w] = o2_wall_results[i]
                } else row = row sprintf("  %12s", "-")
            }
            if (show_pipe_enq) {
                if (i in pipe_enq_results) {
                    row = row sprintf("  %12.1f", pipe_enq_results[i])
                    sum_pe += pipe_enq_results[i]; cnt_pe++; pe_arr[cnt_pe] = pipe_enq_results[i]
                } else row = row sprintf("  %12s", "-")
            }
            if (show_pipe_flush) {
                if (i in pipe_flush_results) {
                    row = row sprintf("  %12.1f", pipe_flush_results[i])
                    sum_pf += pipe_flush_results[i]; cnt_pf++; pf_arr[cnt_pf] = pipe_flush_results[i]
                } else row = row sprintf("  %12s", "-")
            }
            if (show_pipe_counts) {
                if ((i in pipe_task_enq_count_results) && (i in pipe_scope_enq_count_results) && (i in pipe_flush_count_results)) {
                    compact_col = "-"
                    if (i in pipe_compact_deferred_count_results)
                        compact_col = pipe_compact_deferred_count_results[i]
                    if (i in pipe_task_enq_batch_count_results) {
                        row = row sprintf("  %8d  %8d  %8d  %8d  %8s", pipe_task_enq_count_results[i], pipe_task_enq_batch_count_results[i], pipe_scope_enq_count_results[i], pipe_flush_count_results[i], compact_col)
                        sum_task_batch += pipe_task_enq_batch_count_results[i]
                        cnt_batch++
                    } else {
                        row = row sprintf("  %8d  %8s  %8d  %8d  %8s", pipe_task_enq_count_results[i], "-", pipe_scope_enq_count_results[i], pipe_flush_count_results[i], compact_col)
                    }
                    sum_task_enq += pipe_task_enq_count_results[i]
                    sum_scope_enq += pipe_scope_enq_count_results[i]
                    sum_flush_count += pipe_flush_count_results[i]
                    if (i in pipe_compact_deferred_count_results)
                        sum_compact_deferred += pipe_compact_deferred_count_results[i]
                    cnt_pc++
                } else row = row sprintf("  %8s  %8s  %8s  %8s  %8s", "-", "-", "-", "-", "-")
            }
            if (show_pipe_detail) {
                if ((i in pipe_batch_sync_results) && (i in pipe_deferred_dep_results) && (i in pipe_deferred_fanin_results) && (i in pipe_publish_results) && (i in pipe_scope_release_results)) {
                    sum_pipe_batch_sync += pipe_batch_sync_results[i]
                    sum_pipe_deferred_dep += pipe_deferred_dep_results[i]
                    sum_pipe_deferred_fanin += pipe_deferred_fanin_results[i]
                    sum_pipe_publish += pipe_publish_results[i]
                    sum_pipe_scope_release += pipe_scope_release_results[i]
                    if (i in pipe_scope_record_results)
                        sum_pipe_scope_record += pipe_scope_record_results[i]
                    if (i in pipe_scope_producer_release_results)
                        sum_pipe_scope_producer_release += pipe_scope_producer_release_results[i]
                    if (i in pipe_scope_ring_advance_results)
                        sum_pipe_scope_ring_advance += pipe_scope_ring_advance_results[i]
                    if (i in pipe_scope_record_count_results)
                        sum_pipe_scope_record_count += pipe_scope_record_count_results[i]
                    if (i in pipe_scope_release_count_results)
                        sum_pipe_scope_release_count += pipe_scope_release_count_results[i]
                    if (i in pipe_scope_consumed_count_results)
                        sum_pipe_scope_consumed_count += pipe_scope_consumed_count_results[i]
                    if (i in pipe_scope_advance_attempts_results)
                        sum_pipe_scope_advance_attempts += pipe_scope_advance_attempts_results[i]
                    if (i in pipe_scope_advance_success_results)
                        sum_pipe_scope_advance_success += pipe_scope_advance_success_results[i]
                    cnt_pipe_detail++
                }
            }
            if (has_pipe_dep_detail) {
                if ((i in pipe_dep_explicit_results) && (i in pipe_dep_owner_emit_results) && (i in pipe_dep_lookup_results) && (i in pipe_dep_register_results)) {
                    sum_pipe_dep_explicit += pipe_dep_explicit_results[i]
                    sum_pipe_dep_owner_emit += pipe_dep_owner_emit_results[i]
                    sum_pipe_dep_lookup += pipe_dep_lookup_results[i]
                    if (i in pipe_dep_lookup_emit_results)
                        sum_pipe_dep_lookup_emit += pipe_dep_lookup_emit_results[i]
                    if (i in pipe_dep_remove_results)
                        sum_pipe_dep_remove += pipe_dep_remove_results[i]
                    sum_pipe_dep_register += pipe_dep_register_results[i]
                    if (i in pipe_dep_register_insert_results)
                        sum_pipe_dep_register_insert += pipe_dep_register_insert_results[i]
                    if (i in pipe_dep_explicit_count_results)
                        sum_pipe_dep_explicit_count += pipe_dep_explicit_count_results[i]
                    if (i in pipe_dep_owner_emit_count_results)
                        sum_pipe_dep_owner_emit_count += pipe_dep_owner_emit_count_results[i]
                    if (i in pipe_dep_lookup_count_results)
                        sum_pipe_dep_lookup_count += pipe_dep_lookup_count_results[i]
                    if (i in pipe_dep_lookup_skip_count_results)
                        sum_pipe_dep_lookup_skip_count += pipe_dep_lookup_skip_count_results[i]
                    if (i in pipe_dep_lookup_range_skip_count_results)
                        sum_pipe_dep_lookup_range_skip_count += pipe_dep_lookup_range_skip_count_results[i]
                    if (i in pipe_dep_lookup_match_count_results)
                        sum_pipe_dep_lookup_match_count += pipe_dep_lookup_match_count_results[i]
                    if (i in pipe_dep_lookup_emit_count_results)
                        sum_pipe_dep_lookup_emit_count += pipe_dep_lookup_emit_count_results[i]
                    if (i in pipe_dep_remove_count_results)
                        sum_pipe_dep_remove_count += pipe_dep_remove_count_results[i]
                    if (i in pipe_dep_register_count_results)
                        sum_pipe_dep_register_count += pipe_dep_register_count_results[i]
                    if (i in pipe_dep_fanin_actual_count_results)
                        sum_pipe_dep_fanin_actual_count += pipe_dep_fanin_actual_count_results[i]
                    cnt_pipe_dep_detail++
                }
            }
            if (show_four_stage) {
                if ((i in front_stage_end_results) && (i in submitter_stage_end_results) && (i in tensormap_stage_end_results) && (i in updater_stage_end_results)) {
                    row = row sprintf("  %10.1f  %10.1f  %10.1f  %10.1f", front_stage_end_results[i], submitter_stage_end_results[i], tensormap_stage_end_results[i], updater_stage_end_results[i])
                    sum_front_cost += front_stage_cost_results[i]
                    sum_submitter_cost += submitter_stage_cost_results[i]
                    sum_tensormap_cost += tensormap_stage_cost_results[i]
                    sum_updater_cost += updater_stage_cost_results[i]
                    sum_front_end += front_stage_end_results[i]
                    sum_submitter_end += submitter_stage_end_results[i]
                    sum_tensormap_end += tensormap_stage_end_results[i]
                    sum_updater_end += updater_stage_end_results[i]
                    cnt_four++
                } else row = row sprintf("  %10s  %10s  %10s  %10s", "-", "-", "-", "-")
            }
            if (show_active_detail) {
                if ((i in active_detail_unclassified_results) && (i in active_detail_alloc_results) && (i in active_detail_submit_wrapper_results) && (i in active_detail_scope_end_results) && (i in active_detail_p_func_results)) {
                    sum_active_unclassified += active_detail_unclassified_results[i]
                    sum_active_alloc += active_detail_alloc_results[i]
                    sum_active_submit_wrapper += active_detail_submit_wrapper_results[i]
                    sum_active_scope_end += active_detail_scope_end_results[i]
                    sum_active_p_func += active_detail_p_func_results[i]
                    cnt_active_detail++
                }
            }
            if (has_submit_detail) {
                if ((i in submit_detail_count_results) && (i in submit_detail_layout_results) && (i in submit_detail_prepare_results) && (i in submit_detail_payload_results) && (i in submit_detail_enqueue_results)) {
                    sum_submit_count += submit_detail_count_results[i]
                    sum_submit_deferred += submit_detail_deferred_results[i]
                    sum_submit_sync += submit_detail_sync_results[i]
                    if (i in submit_detail_heap_guard_results)
                        sum_submit_heap_guard += submit_detail_heap_guard_results[i]
                    sum_submit_tensors += submit_detail_tensors_results[i]
                    sum_submit_scalars += submit_detail_scalars_results[i]
                    sum_submit_explicit_deps += submit_detail_explicit_deps_results[i]
                    sum_submit_output_bytes += submit_detail_output_bytes_results[i]
                    sum_submit_layout += submit_detail_layout_results[i]
                    sum_submit_prepare += submit_detail_prepare_results[i]
                    sum_submit_depgen += submit_detail_depgen_results[i]
                    sum_submit_sync_cost += submit_detail_sync_cost_results[i]
                    sum_submit_explicit_cost += submit_detail_explicit_cost_results[i]
                    sum_submit_lookup += submit_detail_lookup_results[i]
                    sum_submit_register += submit_detail_register_results[i]
                    sum_submit_payload += submit_detail_payload_results[i]
                    sum_submit_descriptor += submit_detail_descriptor_results[i]
                    sum_submit_deferred_meta += submit_detail_deferred_meta_results[i]
                    sum_submit_enqueue += submit_detail_enqueue_results[i]
                    sum_submit_return_tail += submit_detail_return_tail_results[i]
                    if ((i in submit_detail_prep_check_results) && (i in submit_detail_prep_alloc_results) && (i in submit_detail_prep_ptr_results) && (i in submit_detail_prep_prefetch_results) && (i in submit_detail_prep_slot_results) && (i in submit_detail_prep_scope_push_results)) {
                        sum_submit_prep_check += submit_detail_prep_check_results[i]
                        sum_submit_prep_alloc += submit_detail_prep_alloc_results[i]
                        sum_submit_prep_ptr += submit_detail_prep_ptr_results[i]
                        sum_submit_prep_prefetch += submit_detail_prep_prefetch_results[i]
                        sum_submit_prep_slot += submit_detail_prep_slot_results[i]
                        sum_submit_prep_scope_push += submit_detail_prep_scope_push_results[i]
                        cnt_submit_prep_detail++
                    }
                    if ((i in submit_detail_prep_alloc_task_wait_results) && (i in submit_detail_prep_alloc_heap_wait_results) && (i in submit_detail_prep_alloc_task_spins_results) && (i in submit_detail_prep_alloc_heap_spins_results) && (i in submit_detail_prep_alloc_progress_results)) {
                        sum_submit_prep_alloc_task_wait += submit_detail_prep_alloc_task_wait_results[i]
                        sum_submit_prep_alloc_heap_wait += submit_detail_prep_alloc_heap_wait_results[i]
                        sum_submit_prep_alloc_task_spins += submit_detail_prep_alloc_task_spins_results[i]
                        sum_submit_prep_alloc_heap_spins += submit_detail_prep_alloc_heap_spins_results[i]
                        sum_submit_prep_alloc_progress += submit_detail_prep_alloc_progress_results[i]
                        cnt_submit_alloc_detail++
                    }
                    cnt_submit_detail++
                }
            }
            print row
        }

        # Averages: Host | Device | Total | Sched | Orch | O1 Active | O1 Drain | O2 Active | O2 Wall | Pipe Enq | Pipe Flush
        avg_line = ""; avg_sep = ""
        if (show_host  && cnt_host > 0) { avg_line = avg_line avg_sep sprintf("Host Avg: %.1f us",   sum_host / cnt_host); avg_sep = "  |  " }
        if (show_dev   && cnt_dev > 0)  { avg_line = avg_line avg_sep sprintf("Device Avg: %.1f us", sum_dev  / cnt_dev);  avg_sep = "  |  " }
        if (has_total  && cnt_tot > 0)  { avg_line = avg_line avg_sep sprintf("Total Avg: %.1f us",  sum_tot  / cnt_tot);  avg_sep = "  |  " }
        if (show_sched && cnt_sch > 0)  { avg_line = avg_line avg_sep sprintf("Sched Avg: %.1f us",  sum_sch  / cnt_sch);  avg_sep = "  |  " }
        if (show_orch  && cnt_orc > 0)  { avg_line = avg_line avg_sep sprintf("Orch Avg: %.1f us",   sum_orc  / cnt_orc);  avg_sep = "  |  " }
        if (show_o1_active && cnt_o1a > 0) { avg_line = avg_line avg_sep sprintf("O1 Active Avg: %.1f us", sum_o1a / cnt_o1a); avg_sep = "  |  " }
        if (show_o1_drain  && cnt_o1d > 0) { avg_line = avg_line avg_sep sprintf("O1 Drain Avg: %.1f us",  sum_o1d / cnt_o1d); avg_sep = "  |  " }
        if (show_o2_active && cnt_o2a > 0) { avg_line = avg_line avg_sep sprintf("O2 Active Avg: %.1f us", sum_o2a / cnt_o2a); avg_sep = "  |  " }
        if (show_o2_wall   && cnt_o2w > 0) { avg_line = avg_line avg_sep sprintf("O2 Wall Avg: %.1f us",   sum_o2w / cnt_o2w); avg_sep = "  |  " }
        if (show_pipe_enq   && cnt_pe > 0)  { avg_line = avg_line avg_sep sprintf("Pipe Enq Avg: %.1f us",  sum_pe / cnt_pe);    avg_sep = "  |  " }
        if (show_pipe_flush && cnt_pf > 0)  { avg_line = avg_line avg_sep sprintf("Pipe Flush Avg: %.1f us",sum_pf / cnt_pf);    avg_sep = "  |  " }
        if (show_pipe_counts && cnt_pc > 0) {
            if (cnt_batch > 0) {
                avg_line = avg_line avg_sep sprintf("Pipe Counts Avg: task=%.1f batch=%.1f scope=%.1f flush=%.1f compact=%.1f", sum_task_enq / cnt_pc, sum_task_batch / cnt_batch, sum_scope_enq / cnt_pc, sum_flush_count / cnt_pc, sum_compact_deferred / cnt_pc)
            } else {
                avg_line = avg_line avg_sep sprintf("Pipe Counts Avg: task=%.1f scope=%.1f flush=%.1f compact=%.1f", sum_task_enq / cnt_pc, sum_scope_enq / cnt_pc, sum_flush_count / cnt_pc, sum_compact_deferred / cnt_pc)
            }
            avg_sep = "  |  "
        }
        if (show_pipe_detail && cnt_pipe_detail > 0) {
            avg_line = avg_line avg_sep sprintf("Pipe Detail Avg: batch_sync=%.1f dep=%.1f fanin=%.1f publish=%.1f scope_release=%.1f scope_record=%.1f prod_release=%.1f ring_advance=%.1f final_flush=%.1f",
                sum_pipe_batch_sync / cnt_pipe_detail, sum_pipe_deferred_dep / cnt_pipe_detail,
                sum_pipe_deferred_fanin / cnt_pipe_detail, sum_pipe_publish / cnt_pipe_detail,
                sum_pipe_scope_release / cnt_pipe_detail, sum_pipe_scope_record / cnt_pipe_detail,
                sum_pipe_scope_producer_release / cnt_pipe_detail, sum_pipe_scope_ring_advance / cnt_pipe_detail,
                (cnt_pf > 0 ? sum_pf / cnt_pf : 0))
            avg_sep = "  |  "
            avg_line = avg_line avg_sep sprintf("Pipe Scope Counts Avg: records=%.1f releases=%.1f consumed=%.1f advance_attempts=%.1f advance_success=%.1f",
                sum_pipe_scope_record_count / cnt_pipe_detail, sum_pipe_scope_release_count / cnt_pipe_detail,
                sum_pipe_scope_consumed_count / cnt_pipe_detail, sum_pipe_scope_advance_attempts / cnt_pipe_detail,
                sum_pipe_scope_advance_success / cnt_pipe_detail)
            avg_sep = "  |  "
        }
        if (has_pipe_dep_detail && cnt_pipe_dep_detail > 0) {
            avg_line = avg_line avg_sep sprintf("Pipe Dep Detail Avg: explicit=%.1f owner=%.1f lookup=%.1f lookup_emit=%.1f remove=%.1f register=%.1f insert=%.1f",
                sum_pipe_dep_explicit / cnt_pipe_dep_detail, sum_pipe_dep_owner_emit / cnt_pipe_dep_detail,
                sum_pipe_dep_lookup / cnt_pipe_dep_detail, sum_pipe_dep_lookup_emit / cnt_pipe_dep_detail,
                sum_pipe_dep_remove / cnt_pipe_dep_detail, sum_pipe_dep_register / cnt_pipe_dep_detail,
                sum_pipe_dep_register_insert / cnt_pipe_dep_detail)
            avg_sep = "  |  "
            avg_line = avg_line avg_sep sprintf("Pipe Dep Counts Avg: explicit=%.1f owner=%.1f lookups=%.1f skipped=%.1f range_skipped=%.1f matches=%.1f emits=%.1f edges=%.1f removes=%.1f registers=%.1f",
                sum_pipe_dep_explicit_count / cnt_pipe_dep_detail, sum_pipe_dep_owner_emit_count / cnt_pipe_dep_detail,
                sum_pipe_dep_lookup_count / cnt_pipe_dep_detail,
                sum_pipe_dep_lookup_skip_count / cnt_pipe_dep_detail,
                sum_pipe_dep_lookup_range_skip_count / cnt_pipe_dep_detail,
                sum_pipe_dep_lookup_match_count / cnt_pipe_dep_detail,
                sum_pipe_dep_lookup_emit_count / cnt_pipe_dep_detail,
                sum_pipe_dep_fanin_actual_count / cnt_pipe_dep_detail,
                sum_pipe_dep_remove_count / cnt_pipe_dep_detail,
                sum_pipe_dep_register_count / cnt_pipe_dep_detail)
            avg_sep = "  |  "
        }
        if (show_four_stage && cnt_four > 0) {
            avg_line = avg_line avg_sep sprintf("O4 Cost Avg: front=%.1f submitter=%.1f tensormap=%.1f updater=%.1f", sum_front_cost / cnt_four, sum_submitter_cost / cnt_four, sum_tensormap_cost / cnt_four, sum_updater_cost / cnt_four)
            avg_sep = "  |  "
            avg_line = avg_line avg_sep sprintf("O4 Serial End Avg: front=%.1f submitter=%.1f tensormap=%.1f updater=%.1f", sum_front_end / cnt_four, sum_submitter_end / cnt_four, sum_tensormap_end / cnt_four, sum_updater_end / cnt_four)
            avg_sep = "  |  "
        }
        if (show_active_detail && cnt_active_detail > 0) {
            avg_line = avg_line avg_sep sprintf("O Detail Avg: unclassified=%.1f alloc=%.1f submit_wrap=%.1f scope_end=%.1f p_func=%.1f", sum_active_unclassified / cnt_active_detail, sum_active_alloc / cnt_active_detail, sum_active_submit_wrapper / cnt_active_detail, sum_active_scope_end / cnt_active_detail, sum_active_p_func / cnt_active_detail)
            avg_sep = "  |  "
        }
        if (has_submit_detail && cnt_submit_detail > 0) {
            submit_detail_avg = sprintf("Submit Detail Avg: count=%.1f deferred=%.1f sync=%.1f heap_guard=%.1f tensors=%.1f scalars=%.1f explicit_deps=%.1f output_kb=%.1f layout=%.1f prepare=%.1f depgen=%.1f sync_cost=%.1f explicit=%.1f lookup=%.1f register=%.1f payload=%.1f descriptor=%.1f deferred_meta=%.1f enqueue=%.1f return_tail=%.1f",
                sum_submit_count / cnt_submit_detail, sum_submit_deferred / cnt_submit_detail,
                sum_submit_sync / cnt_submit_detail, sum_submit_heap_guard / cnt_submit_detail,
                sum_submit_tensors / cnt_submit_detail, sum_submit_scalars / cnt_submit_detail,
                sum_submit_explicit_deps / cnt_submit_detail, (sum_submit_output_bytes / cnt_submit_detail) / 1024.0,
                sum_submit_layout / cnt_submit_detail, sum_submit_prepare / cnt_submit_detail,
                sum_submit_depgen / cnt_submit_detail, sum_submit_sync_cost / cnt_submit_detail,
                sum_submit_explicit_cost / cnt_submit_detail, sum_submit_lookup / cnt_submit_detail,
                sum_submit_register / cnt_submit_detail, sum_submit_payload / cnt_submit_detail,
                sum_submit_descriptor / cnt_submit_detail, sum_submit_deferred_meta / cnt_submit_detail,
                sum_submit_enqueue / cnt_submit_detail, sum_submit_return_tail / cnt_submit_detail)
            if (cnt_submit_prep_detail > 0) {
                submit_detail_avg = submit_detail_avg sprintf(" prep_check=%.1f prep_alloc=%.1f prep_ptr=%.1f prep_prefetch=%.1f prep_slot=%.1f prep_scope_push=%.1f",
                    sum_submit_prep_check / cnt_submit_prep_detail, sum_submit_prep_alloc / cnt_submit_prep_detail,
                    sum_submit_prep_ptr / cnt_submit_prep_detail, sum_submit_prep_prefetch / cnt_submit_prep_detail,
                    sum_submit_prep_slot / cnt_submit_prep_detail, sum_submit_prep_scope_push / cnt_submit_prep_detail)
            }
            if (cnt_submit_alloc_detail > 0) {
                submit_detail_avg = submit_detail_avg sprintf(" prep_alloc_task_wait=%.1f prep_alloc_heap_wait=%.1f prep_alloc_task_spins=%.1f prep_alloc_heap_spins=%.1f prep_alloc_progress=%.1f",
                    sum_submit_prep_alloc_task_wait / cnt_submit_alloc_detail,
                    sum_submit_prep_alloc_heap_wait / cnt_submit_alloc_detail,
                    sum_submit_prep_alloc_task_spins / cnt_submit_alloc_detail,
                    sum_submit_prep_alloc_heap_spins / cnt_submit_alloc_detail,
                    sum_submit_prep_alloc_progress / cnt_submit_alloc_detail)
            }
            avg_line = avg_line avg_sep submit_detail_avg
            avg_sep = "  |  "
        }
        printf "\n  %s  (%d rounds)\n", avg_line, n_rounds

        TRIM = 10
        if (cnt_host > 2 * TRIM) trimmed("Host",   host_arr, cnt_host, TRIM)
        if (cnt_dev  > 2 * TRIM) trimmed("Device", dev_arr,  cnt_dev,  TRIM)
        if (cnt_tot  > 2 * TRIM) trimmed("Total",  tot_arr,  cnt_tot,  TRIM)
        if (cnt_sch  > 2 * TRIM) trimmed("Sched",  sch_arr,  cnt_sch,  TRIM)
        if (cnt_orc  > 2 * TRIM) trimmed("Orch",   orc_arr,  cnt_orc,  TRIM)
        if (cnt_o1a  > 2 * TRIM) trimmed("O1 Active", o1a_arr,  cnt_o1a,  TRIM)
        if (cnt_o1d  > 2 * TRIM) trimmed("O1 Drain",  o1d_arr,  cnt_o1d,  TRIM)
        if (cnt_o2a  > 2 * TRIM) trimmed("O2 Active", o2a_arr,  cnt_o2a,  TRIM)
        if (cnt_o2w  > 2 * TRIM) trimmed("O2 Wall",   o2w_arr,  cnt_o2w,  TRIM)
        if (cnt_pe   > 2 * TRIM) trimmed("Pipe Enq",   pe_arr,   cnt_pe,   TRIM)
        if (cnt_pf   > 2 * TRIM) trimmed("Pipe Flush", pf_arr,   cnt_pf,   TRIM)
    }' "$fw_file" "$dev_timing_file"
}

# ---------------------------------------------------------------------------
# wait_for_new_log <pre_run_logs_file>
#   Wait up to 15s for a new .log file in DEVICE_LOG_DIR. Prints the path.
# ---------------------------------------------------------------------------
wait_for_new_log() {
    local pre_file="$1"
    local new_log=""
    local deadline=$((SECONDS + 15))

    while [[ $SECONDS -lt $deadline ]]; do
        if [[ -d "$DEVICE_LOG_DIR" ]]; then
            new_log=$(comm -13 "$pre_file" <(ls -1 "$DEVICE_LOG_DIR"/*.log 2>/dev/null | sort) 2>/dev/null | tail -1 || true)
            if [[ -n "$new_log" ]]; then
                echo "$new_log"
                return 0
            fi
        fi
        sleep 0.5
    done

    # Fallback: newest log
    if [[ -d "$DEVICE_LOG_DIR" ]]; then
        new_log=$(ls -t "$DEVICE_LOG_DIR"/*.log 2>/dev/null | head -1 || true)
        if [[ -n "$new_log" ]]; then
            echo "$new_log"
            return 0
        fi
    fi
    return 1
}

# ---------------------------------------------------------------------------
# run_bench <example> <example_dir> [case_name]
#   Run one benchmark invocation (via `python test_*.py`) and parse timing
#   from the resulting log. Skips the example if it has no test_*.py.
#   Sets global PASS / FAIL counters.
# ---------------------------------------------------------------------------
run_bench() {
    local example="$1" example_dir="$2" case_name="${3:-}"

    if [[ -n "$case_name" ]]; then
        echo "  ---- $case_name ----"
    fi

    # Snapshot existing logs
    local pre_log_file fw_stdout_file
    pre_log_file=$(mktemp)
    fw_stdout_file=$(mktemp)
    trap 'rm -f -- "$pre_log_file" "$fw_stdout_file"' RETURN
    ls -1 "$DEVICE_LOG_DIR"/*.log 2>/dev/null | sort > "$pre_log_file" || true

    # Build run command using test_*.py
    local test_file
    test_file=$(find "$example_dir" -maxdepth 1 -name 'test_*.py' -print -quit 2>/dev/null || true)

    local run_cmd
    if [[ -n "$test_file" ]]; then
        run_cmd=(
            python3 "$test_file"
            --platform "$PLATFORM" --device "$DEVICE_ID"
            --rounds "$ROUNDS" --skip-golden
        )
    else
        echo "  SKIPPED: no test_*.py found in $example_dir"
        return
    fi
    if [[ -n "$case_name" ]]; then
        run_cmd+=(--case "$case_name")
        [[ -n "$test_file" ]] && run_cmd+=(--manual include)
    fi
    run_cmd+=("${EXTRA_ARGS[@]}")

    # Run example, capturing stdout/stderr for Host/Device timing parse
    vlog "Running: ${run_cmd[*]}"
    local rc=0
    if case_uses_baseline_fallback "$example" "$case_name"; then
        if [[ -n "${SIMPLER_PIPELINE_STRATEGY:-}" ]]; then
            echo "  Pipeline fallback: baseline layout for $example${case_name:+:$case_name} (strategy ${SIMPLER_PIPELINE_STRATEGY})"
            vlog "Pipeline fallback: baseline layout for $example${case_name:+:$case_name}"
        fi
        env -u SIMPLER_PIPELINE_STRATEGY "${run_cmd[@]}" > "$fw_stdout_file" 2>&1 || rc=$?
    else
        "${run_cmd[@]}" > "$fw_stdout_file" 2>&1 || rc=$?
    fi
    if [[ -n "$VERBOSE_LOG" && -s "$fw_stdout_file" ]]; then
        cat "$fw_stdout_file" >> "$VERBOSE_LOG"
    fi
    if [[ $rc -ne 0 ]]; then
        echo "  FAILED: benchmark run returned non-zero"
        vlog "FAILED: exit code $rc"
        ((FAIL++)) || true
        return
    fi

    # Find new device log
    local new_log
    new_log=$(wait_for_new_log "$pre_log_file")

    if [[ -z "$new_log" ]]; then
        echo "  FAILED: no device log found in $DEVICE_LOG_DIR"
        ((FAIL++)) || true
        return
    fi

    echo "  Log: $new_log"
    local timing_output
    local parse_rc=0
    timing_output=$(parse_timing "$fw_stdout_file" "$new_log") || parse_rc=$?
    echo "$timing_output"

    if [[ $parse_rc -ne 0 ]]; then
        ((FAIL++)) || true
        return
    fi
    ((PASS++)) || true

    # Extract averages for summary table
    local label="$example"
    [[ -n "$case_name" ]] && label="$example ($case_name)"

    local avg_line
    avg_line=$(echo "$timing_output" | grep -E '(Host|Device|Total|Sched|Orch|O1 Active|O1 Drain|O2 Active|O2 Wall|Pipe Enq|Pipe Flush|Pipe Counts) Avg:' | grep -v 'Trimmed' | head -1 || true)
    local avg_host="-" avg_device="-" avg_total="-" avg_sched="-" avg_orch="-" avg_o1_active="-" avg_o1_drain="-" avg_o2_active="-" avg_o2_wall="-" avg_pipe_enq="-" avg_pipe_flush="-" avg_pipe_counts="-"
    if [[ -n "$avg_line" ]]; then
        avg_host=$(      echo "$avg_line" | grep -oE 'Host Avg: [0-9.]+'      || true); avg_host=${avg_host##* }
        avg_device=$(    echo "$avg_line" | grep -oE 'Device Avg: [0-9.]+'    || true); avg_device=${avg_device##* }
        avg_total=$(     echo "$avg_line" | grep -oE 'Total Avg: [0-9.]+'     || true); avg_total=${avg_total##* }
        avg_sched=$(     echo "$avg_line" | grep -oE 'Sched Avg: [0-9.]+'     || true); avg_sched=${avg_sched##* }
        avg_orch=$(      echo "$avg_line" | grep -oE 'Orch Avg: [0-9.]+'      || true); avg_orch=${avg_orch##* }
        avg_o1_active=$( echo "$avg_line" | grep -oE 'O1 Active Avg: [0-9.]+' || true); avg_o1_active=${avg_o1_active##* }
        avg_o1_drain=$(  echo "$avg_line" | grep -oE 'O1 Drain Avg: [0-9.]+'  || true); avg_o1_drain=${avg_o1_drain##* }
        avg_o2_active=$( echo "$avg_line" | grep -oE 'O2 Active Avg: [0-9.]+' || true); avg_o2_active=${avg_o2_active##* }
        avg_o2_wall=$(   echo "$avg_line" | grep -oE 'O2 Wall Avg: [0-9.]+'   || true); avg_o2_wall=${avg_o2_wall##* }
        avg_pipe_enq=$(  echo "$avg_line" | grep -oE 'Pipe Enq Avg: [0-9.]+'   || true); avg_pipe_enq=${avg_pipe_enq##* }
        avg_pipe_flush=$(echo "$avg_line" | grep -oE 'Pipe Flush Avg: [0-9.]+' || true); avg_pipe_flush=${avg_pipe_flush##* }
        avg_pipe_counts=$(echo "$avg_line" | grep -oE 'Pipe Counts Avg: task=[0-9.]+( batch=[0-9.]+)? scope=[0-9.]+ flush=[0-9.]+( compact=[0-9.]+)?' || true)
        avg_pipe_counts=${avg_pipe_counts#Pipe Counts Avg: }
        [[ -z "$avg_host" ]]   && avg_host="-"
        [[ -z "$avg_device" ]] && avg_device="-"
        [[ -z "$avg_total" ]]  && avg_total="-"
        [[ -z "$avg_sched" ]]  && avg_sched="-"
        [[ -z "$avg_orch" ]]   && avg_orch="-"
        [[ -z "$avg_o1_active" ]] && avg_o1_active="-"
        [[ -z "$avg_o1_drain" ]]  && avg_o1_drain="-"
        [[ -z "$avg_o2_active" ]] && avg_o2_active="-"
        [[ -z "$avg_o2_wall" ]]   && avg_o2_wall="-"
        [[ -z "$avg_pipe_enq" ]]   && avg_pipe_enq="-"
        [[ -z "$avg_pipe_flush" ]] && avg_pipe_flush="-"
        [[ -z "$avg_pipe_counts" ]] && avg_pipe_counts="-"
    fi

    SUMMARY_NAMES+=("$label")
    SUMMARY_HOST+=("$avg_host")
    SUMMARY_DEVICE+=("$avg_device")
    SUMMARY_TOTAL+=("$avg_total")
    SUMMARY_SCHED+=("$avg_sched")
    SUMMARY_ORCH+=("$avg_orch")
    SUMMARY_O1_ACTIVE+=("$avg_o1_active")
    SUMMARY_O1_DRAIN+=("$avg_o1_drain")
    SUMMARY_O2_ACTIVE+=("$avg_o2_active")
    SUMMARY_O2_WALL+=("$avg_o2_wall")
    SUMMARY_PIPE_ENQ+=("$avg_pipe_enq")
    SUMMARY_PIPE_FLUSH+=("$avg_pipe_flush")
    SUMMARY_PIPE_COUNTS+=("$avg_pipe_counts")
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
PASS=0
FAIL=0

# Summary collection arrays
SUMMARY_NAMES=()
SUMMARY_HOST=()
SUMMARY_DEVICE=()
SUMMARY_TOTAL=()
SUMMARY_SCHED=()
SUMMARY_ORCH=()
SUMMARY_O1_ACTIVE=()
SUMMARY_O1_DRAIN=()
SUMMARY_O2_ACTIVE=()
SUMMARY_O2_WALL=()
SUMMARY_PIPE_ENQ=()
SUMMARY_PIPE_FLUSH=()
SUMMARY_PIPE_COUNTS=()

echo ""
echo "Runtime: $RUNTIME"
echo "Case set: $BENCH_CASE_SET"
if [[ "$BENCH_CASE_SET" == "single" ]]; then
    echo "Case: $BENCH_CASE"
fi

for example in "${EXAMPLE_ORDER[@]}"; do
    case_list="$(case_list_for_example "$example")"

    # Search for example: prefer test_*.py (new style), fall back to golden.py (legacy).
    # tests/st/ is searched before examples/ since benchmarks use production-scale cases.
    EXAMPLE_DIR=""
    for dir in "${EXAMPLES_DIRS[@]}"; do
        candidate="$dir/$example"
        if [[ -d "$candidate" ]] && ls "$candidate"/test_*.py 1>/dev/null 2>&1; then
            EXAMPLE_DIR="$candidate"
            break
        fi
    done
    if [[ -z "$EXAMPLE_DIR" ]]; then
        for dir in "${EXAMPLES_DIRS[@]}"; do
            candidate="$dir/$example"
            if [[ -f "$candidate/golden.py" && -d "$candidate/kernels" ]]; then
                EXAMPLE_DIR="$candidate"
                break
            fi
        done
    fi

    echo ""
    echo "================================================================"
    echo "  $example"
    echo "================================================================"

    if [[ -z "$EXAMPLE_DIR" ]]; then
        echo "  SKIP: not found in any search directory"
        ((FAIL++)) || true
        continue
    fi

    if [[ -z "${case_list:-}" ]]; then
        run_bench "$example" "$EXAMPLE_DIR"
    else
        IFS=',' read -ra cases <<< "$case_list"
        for c in "${cases[@]}"; do
            run_bench "$example" "$EXAMPLE_DIR" "$c"
        done
    fi
done

# ---------------------------------------------------------------------------
# Performance Summary Table
# ---------------------------------------------------------------------------
if [[ ${#SUMMARY_NAMES[@]} -gt 0 ]]; then
    # Show only columns that have at least one non-"-" value
    _has_host=0; _has_device=0; _has_total=0; _has_sched=0; _has_orch=0; _has_o1_active=0; _has_o1_drain=0; _has_o2_active=0; _has_o2_wall=0; _has_pipe_enq=0; _has_pipe_flush=0; _has_pipe_counts=0
    for _i in "${!SUMMARY_NAMES[@]}"; do
        [[ "${SUMMARY_HOST[$_i]}"   != "-" ]] && _has_host=1
        [[ "${SUMMARY_DEVICE[$_i]}" != "-" ]] && _has_device=1
        [[ "${SUMMARY_TOTAL[$_i]}"  != "-" ]] && _has_total=1
        [[ "${SUMMARY_SCHED[$_i]}"  != "-" ]] && _has_sched=1
        [[ "${SUMMARY_ORCH[$_i]}"   != "-" ]] && _has_orch=1
        [[ "${SUMMARY_O1_ACTIVE[$_i]}" != "-" ]] && _has_o1_active=1
        [[ "${SUMMARY_O1_DRAIN[$_i]}"  != "-" ]] && _has_o1_drain=1
        [[ "${SUMMARY_O2_ACTIVE[$_i]}" != "-" ]] && _has_o2_active=1
        [[ "${SUMMARY_O2_WALL[$_i]}"   != "-" ]] && _has_o2_wall=1
        [[ "${SUMMARY_PIPE_ENQ[$_i]}"   != "-" ]] && _has_pipe_enq=1
        [[ "${SUMMARY_PIPE_FLUSH[$_i]}" != "-" ]] && _has_pipe_flush=1
        [[ "${SUMMARY_PIPE_COUNTS[$_i]}" != "-" ]] && _has_pipe_counts=1
    done

    echo ""
    echo "================================================================"
    echo "  Performance Summary ($RUNTIME)"
    echo "================================================================"
    echo ""

    _hdr=$(printf "  %-40s" "Example")
    _sep=$(printf "  %-40s" "----------------------------------------")
    if [[ $_has_host   -eq 1 ]]; then _hdr=$(printf "%s  %12s" "$_hdr" "Host (us)");   _sep=$(printf "%s  %12s" "$_sep" "------------"); fi
    if [[ $_has_device -eq 1 ]]; then _hdr=$(printf "%s  %12s" "$_hdr" "Device (us)"); _sep=$(printf "%s  %12s" "$_sep" "------------"); fi
    if [[ $_has_total  -eq 1 ]]; then _hdr=$(printf "%s  %12s" "$_hdr" "Total (us)");  _sep=$(printf "%s  %12s" "$_sep" "------------"); fi
    if [[ $_has_sched  -eq 1 ]]; then _hdr=$(printf "%s  %12s" "$_hdr" "Sched (us)");  _sep=$(printf "%s  %12s" "$_sep" "------------"); fi
    if [[ $_has_orch   -eq 1 ]]; then _hdr=$(printf "%s  %12s" "$_hdr" "Orch (us)");   _sep=$(printf "%s  %12s" "$_sep" "------------"); fi
    if [[ $_has_o1_active -eq 1 ]]; then _hdr=$(printf "%s  %12s" "$_hdr" "O1 Act (us)"); _sep=$(printf "%s  %12s" "$_sep" "------------"); fi
    if [[ $_has_o1_drain  -eq 1 ]]; then _hdr=$(printf "%s  %12s" "$_hdr" "O1 Drain");    _sep=$(printf "%s  %12s" "$_sep" "------------"); fi
    if [[ $_has_o2_active -eq 1 ]]; then _hdr=$(printf "%s  %12s" "$_hdr" "O2 Act (us)"); _sep=$(printf "%s  %12s" "$_sep" "------------"); fi
    if [[ $_has_o2_wall   -eq 1 ]]; then _hdr=$(printf "%s  %12s" "$_hdr" "O2 Wall");     _sep=$(printf "%s  %12s" "$_sep" "------------"); fi
    if [[ $_has_pipe_enq  -eq 1 ]]; then _hdr=$(printf "%s  %12s" "$_hdr" "Pipe Enq");    _sep=$(printf "%s  %12s" "$_sep" "------------"); fi
    if [[ $_has_pipe_flush -eq 1 ]]; then _hdr=$(printf "%s  %12s" "$_hdr" "Pipe Flush"); _sep=$(printf "%s  %12s" "$_sep" "------------"); fi
    if [[ $_has_pipe_counts -eq 1 ]]; then _hdr=$(printf "%s  %26s" "$_hdr" "Pipe Counts"); _sep=$(printf "%s  %26s" "$_sep" "--------------------------"); fi
    echo "$_hdr"
    echo "$_sep"

    for _i in "${!SUMMARY_NAMES[@]}"; do
        _row=$(printf "  %-40s" "${SUMMARY_NAMES[$_i]}")
        if [[ $_has_host   -eq 1 ]]; then _row=$(printf "%s  %12s" "$_row" "${SUMMARY_HOST[$_i]}");   fi
        if [[ $_has_device -eq 1 ]]; then _row=$(printf "%s  %12s" "$_row" "${SUMMARY_DEVICE[$_i]}"); fi
        if [[ $_has_total  -eq 1 ]]; then _row=$(printf "%s  %12s" "$_row" "${SUMMARY_TOTAL[$_i]}");  fi
        if [[ $_has_sched  -eq 1 ]]; then _row=$(printf "%s  %12s" "$_row" "${SUMMARY_SCHED[$_i]}");  fi
        if [[ $_has_orch   -eq 1 ]]; then _row=$(printf "%s  %12s" "$_row" "${SUMMARY_ORCH[$_i]}");   fi
        if [[ $_has_o1_active -eq 1 ]]; then _row=$(printf "%s  %12s" "$_row" "${SUMMARY_O1_ACTIVE[$_i]}"); fi
        if [[ $_has_o1_drain  -eq 1 ]]; then _row=$(printf "%s  %12s" "$_row" "${SUMMARY_O1_DRAIN[$_i]}");  fi
        if [[ $_has_o2_active -eq 1 ]]; then _row=$(printf "%s  %12s" "$_row" "${SUMMARY_O2_ACTIVE[$_i]}"); fi
        if [[ $_has_o2_wall   -eq 1 ]]; then _row=$(printf "%s  %12s" "$_row" "${SUMMARY_O2_WALL[$_i]}");   fi
        if [[ $_has_pipe_enq  -eq 1 ]]; then _row=$(printf "%s  %12s" "$_row" "${SUMMARY_PIPE_ENQ[$_i]}");  fi
        if [[ $_has_pipe_flush -eq 1 ]]; then _row=$(printf "%s  %12s" "$_row" "${SUMMARY_PIPE_FLUSH[$_i]}"); fi
        if [[ $_has_pipe_counts -eq 1 ]]; then _row=$(printf "%s  %26s" "$_row" "${SUMMARY_PIPE_COUNTS[$_i]}"); fi
        echo "$_row"
    done
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
TOTAL=$((PASS + FAIL))
echo ""
echo "================================================================"
echo "  Benchmark complete ($RUNTIME): $PASS passed, $FAIL failed ($TOTAL total)"
echo "================================================================"

if [[ -n "$VERBOSE_LOG" ]]; then
    echo "  Verbose log saved to: $VERBOSE_LOG"
fi

[[ $FAIL -eq 0 ]]
