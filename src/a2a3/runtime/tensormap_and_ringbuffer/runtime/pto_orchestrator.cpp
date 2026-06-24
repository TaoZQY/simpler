/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

/**
 * PTO Runtime2 - Orchestrator Implementation
 *
 * Implements orchestrator state management, scope handling, and task submission.
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#include "pto_orchestrator.h"

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>

#include "aicpu/dep_gen_collector_aicpu.h"
#include "common/dep_gen.h"
#include "common/unified_log.h"
#include "pto_dep_compute.h"
#include "pto_runtime2_types.h"
#include "pto_shared_memory.h"
#include "pto_tensormap.h"
#include "pto_types.h"
#include "tensor.h"

#if PTO2_PROFILING
#include "aicpu/scope_stats_collector_aicpu.h"
#include "aicpu/tensor_dump_aicpu.h"
#endif

// Verify the captured Tensor blob size in DepGenRecord matches the runtime
// Tensor layout. The platform header defines DEP_GEN_TENSOR_SIZE without
// including runtime/tensor.h, so this check lives at the orch callsite.
static_assert(sizeof(Tensor) == DEP_GEN_TENSOR_SIZE, "DepGenRecord::tensors slot size out of sync with sizeof(Tensor)");
// DEP_GEN_MAX_EXPLICIT_DEPS is a diagnostic-side capture cap only; the runtime
// imposes no hard cap on explicit dep count. If a submit exceeds this cap,
// dep_gen_aicpu_record_submit() logs and truncates — runtime correctness is
// unaffected, only the captured replay record is truncated.

// Weak fallbacks: dep_gen_collector_aicpu.cpp provides the strong symbols in
// AICPU builds. Host builds (host_build_graph runtime, future dep_gen replay)
// link these no-op stubs so the runtime translation unit is self-contained.
// Visibility is hidden so the HOST .so doesn't export them into the global
// dynamic symbol table where they'd shadow the AICPU .so's strong symbols
// (same pattern as get_sys_cnt_aicpu / l2_swimlane_aicpu_record_orch_phase below).
extern "C" __attribute__((weak, visibility("hidden"))) bool is_dep_gen_enabled() { return false; }
__attribute__((weak, visibility("hidden"))) void dep_gen_aicpu_record_submit(
    uint64_t, bool, int, const void *const *, const uint8_t *, int, const uint64_t *, const int32_t[3]
) {}

// Scope_stats enable gate, queried via the same predicate idiom as
// is_dep_gen_enabled above. The AICPU collector links the strong definition;
// host builds fall back to this weak `false`. Gating here still skips the
// cross-agent occupancy reads that feed the sample when scope_stats is disabled.
extern "C" __attribute__((weak, visibility("hidden"))) bool is_scope_stats_enabled() { return false; }

// Heap-ring wrap report, called from the allocator (pto_ring_buffer.h) on each
// wrap. Strong definition lives in the AICPU collector; host builds fall back to
// this weak no-op so the runtime translation unit stays self-contained.
extern "C" __attribute__((weak, visibility("hidden"))) void scope_stats_note_heap_wrap(int) {}

// =============================================================================
// Orchestrator Profiling (compile-time toggle)
// =============================================================================
#if PTO2_ORCH_PROFILING
#include "aicpu/device_time.h"
#include "aicpu/l2_swimlane_collector_aicpu.h"
// Weak fallback for builds that don't link device_time.cpp (e.g. host).
// The strong symbol from platform/.../device_time.cpp wins in the AICPU build.
//
// IMPORTANT: visibility("hidden") is required to prevent the HOST .so from
// exporting this weak fallback into the global dynamic symbol table via
// RTLD_GLOBAL. Without it, when the AICPU .so is loaded and its PLT entry
// for get_sys_cnt_aicpu is resolved, the dynamic linker finds the HOST .so's
// weak definition first (already in global table) and uses it — returning 0.
// With hidden visibility, the HOST .so does not export this symbol globally,
// so the AICPU .so's PLT resolves to its own strong definition from
// device_time.cpp.
__attribute__((weak, visibility("hidden"))) uint64_t get_sys_cnt_aicpu() { return 0; }
// Weak fallback for builds that don't link l2_swimlane_collector_aicpu.cpp.
// The strong symbol from the AICPU build wins when profiling is available.
// Also hidden to prevent HOST .so from polluting the global symbol table.
__attribute__((weak, visibility("hidden"))) void
l2_swimlane_aicpu_record_orch_phase(uint64_t, uint64_t, uint64_t, uint32_t) {}
// Accumulated cycles per sub-step (only needed for ORCH_PROFILING export)
static uint64_t g_orch_sync_cycle = 0;       // tensormap sync
static uint64_t g_orch_alloc_cycle = 0;      // unified task+heap alloc
static uint64_t g_orch_args_cycle = 0;       // param copy
static uint64_t g_orch_lookup_cycle = 0;     // tensormap lookup + dep building
static uint64_t g_orch_insert_cycle = 0;     // tensormap insert
static uint64_t g_orch_fanin_cycle = 0;      // fanin list + early-return check
static uint64_t g_orch_scope_end_cycle = 0;  // scope_end overhead
static int64_t g_orch_submit_count = 0;
static uint32_t g_orch_submit_idx = 0;
uint64_t g_orch_alloc_wait_cycle = 0;
uint64_t g_orch_fanin_wait_cycle = 0;
uint64_t g_orch_alloc_atomic_count = 0;
uint64_t g_orch_args_atomic_count = 0;
uint64_t g_orch_scope_end_atomic_count = 0;
// Cycle accumulation is unconditional under PTO2_ORCH_PROFILING (that's what
// the flag is for) and feeds the per-sub-step `g_orch_*_cycle` cumulatives
// printed in the cold-path log.
//
// Per-submit ORCH_SUBMIT record is the only swim-lane emit on the orch
// path — one record per submit_task() / alloc_tensors() call spanning
// the entire [start, end] window. Per-sub-step phase records were dropped
// in favour of the cumulatives + per-submit envelope; the dispatcher
// already inserts one record at the end of each submit path via
// CYCLE_COUNT_ORCH_SUBMIT_RECORD.
#define CYCLE_COUNT_START()                                                        \
    bool _prof_active = (orch->l2_swimlane_level >= L2SwimlaneLevel::ORCH_PHASES); \
    uint64_t _t0 = get_sys_cnt_aicpu(), _t1;                                       \
    uint64_t _submit_start_ts = _t0
#define CYCLE_COUNT_LAP(acc)       \
    do {                           \
        _t1 = get_sys_cnt_aicpu(); \
        acc += (_t1 - _t0);        \
        _t0 = _t1;                 \
    } while (0)
#define CYCLE_COUNT_ORCH_SUBMIT_RECORD(tid)                                                       \
    do {                                                                                          \
        if (_prof_active) {                                                                       \
            l2_swimlane_aicpu_record_orch_phase(_submit_start_ts, _t1, (tid), g_orch_submit_idx); \
        }                                                                                         \
    } while (0)
#elif PTO2_PROFILING
#include "aicpu/device_time.h"
#include "aicpu/l2_swimlane_collector_aicpu.h"
__attribute__((weak, visibility("hidden"))) uint64_t get_sys_cnt_aicpu() { return 0; }
__attribute__((weak, visibility("hidden"))) void
l2_swimlane_aicpu_record_orch_phase(uint64_t, uint64_t, uint64_t, uint32_t) {}
// submit_idx needed for swimlane task_id tagging (no cycle accumulation at this level)
static uint32_t g_orch_submit_idx = 0;
#define CYCLE_COUNT_START()                                                        \
    bool _prof_active = (orch->l2_swimlane_level >= L2SwimlaneLevel::ORCH_PHASES); \
    uint64_t _t0 = _prof_active ? get_sys_cnt_aicpu() : 0, _t1 = 0;                \
    uint64_t _submit_start_ts = _t0
#define CYCLE_COUNT_LAP(acc) \
    do {                     \
    } while (0)
#define CYCLE_COUNT_ORCH_SUBMIT_RECORD(tid)                                                       \
    do {                                                                                          \
        if (_prof_active) {                                                                       \
            _t1 = get_sys_cnt_aicpu();                                                            \
            l2_swimlane_aicpu_record_orch_phase(_submit_start_ts, _t1, (tid), g_orch_submit_idx); \
        }                                                                                         \
    } while (0)
#else
#define CYCLE_COUNT_START()
#define CYCLE_COUNT_LAP(acc)
#define CYCLE_COUNT_ORCH_SUBMIT_RECORD(tid)
#endif

static int32_t orch_mark_fatal(PTO2OrchestratorState *orch, int32_t error_code) {
    always_assert(orch != nullptr);
    orch->fatal = true;
    if (error_code == PTO2_ERROR_NONE || orch->sm_header == nullptr) {
        return PTO2_ERROR_NONE;
    }

    int32_t expected = PTO2_ERROR_NONE;
    std::atomic<int32_t> &orch_error_code = orch->sm_header->orch_error_code;
    if (orch_error_code.compare_exchange_strong(expected, error_code, std::memory_order_acq_rel)) {
        return error_code;
    }
    return expected;
}

static void
orch_report_fatal_v(PTO2OrchestratorState *orch, int32_t error_code, const char *func, const char *fmt, va_list args) {
    int32_t latched_code = orch_mark_fatal(orch, error_code);

#if PTO2_PROFILING
    // Flush the current scope's peaks BEFORE the FATAL log line, so the
    // diagnostic context (which pool/window filled up) appears right next to
    // the failure reason. on_fatal is latched, so duplicate fatals from
    // different layers don't print multiple stats lines.
    scope_stats_on_fatal();
#endif

    if (fmt == nullptr || fmt[0] == '\0') {
        if (latched_code != PTO2_ERROR_NONE && latched_code != error_code) {
            unified_log_error(func, "FATAL(code=%d, latched=%d)", error_code, latched_code);
        } else {
            unified_log_error(func, "FATAL(code=%d)", error_code);
        }
        return;
    }

    char message[1024];
    vsnprintf(message, sizeof(message), fmt, args);
    if (latched_code != PTO2_ERROR_NONE && latched_code != error_code) {
        unified_log_error(func, "FATAL(code=%d, latched=%d): %s", error_code, latched_code, message);
        return;
    }
    unified_log_error(func, "FATAL(code=%d): %s", error_code, message);
}

void PTO2OrchestratorState::report_fatal(int32_t error_code, const char *func, const char *fmt, ...) {
    auto *orch = this;
    va_list args;
    va_start(args, fmt);
    orch_report_fatal_v(orch, error_code, func, fmt, args);
    va_end(args);
}

static uint32_t next_fanin_seen_epoch(PTO2OrchestratorState *orch) {
    uint32_t next = orch->fanin_seen_current_epoch + 1;
    if (next == 0) {
        for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
            memset(
                orch->fanin_seen_epoch[r], 0,
                static_cast<size_t>(orch->sm_header->rings[r].task_window_size) * sizeof(uint32_t)
            );
        }
        next = 1;
    }
    orch->fanin_seen_current_epoch = next;
    return next;
}

struct PTO2FaninBuilder {
    PTO2FaninBuilder(PTO2OrchestratorState *orch, PTO2FaninPool &spill_pool, uint32_t seen_epoch) :
        count(0),
        spill_start(0),
        orch(orch),
        seen_epoch(seen_epoch),
        spill_pool(spill_pool) {}
    int32_t count{0};
    int32_t spill_start{0};
    PTO2OrchestratorState *orch{nullptr};
    uint32_t seen_epoch{0};
    PTO2FaninPool &spill_pool;
    PTO2TaskSlotState *inline_slots[PTO2_FANIN_INLINE_CAP];

    template <typename Fn>
    PTO2FaninForEachReturn<Fn> for_each(Fn &&fn) const {
        return for_each_fanin_storage(inline_slots, count, spill_start, spill_pool, static_cast<Fn &&>(fn));
    }

    bool mark_seen(uint8_t prod_ring, int32_t prod_slot) {
        if (prod_ring >= PTO2_MAX_RING_DEPTH || prod_slot < 0) {
            return false;
        }
        uint32_t *seen = orch->fanin_seen_epoch[prod_ring];
        uint32_t slot = static_cast<uint32_t>(prod_slot);
        if (seen[slot] == seen_epoch) {
            return true;
        }
        seen[slot] = seen_epoch;
        return false;
    }
};

static bool append_fanin_or_fail(
    PTO2OrchestratorState *orch, uint8_t prod_ring, int32_t prod_slot, PTO2TaskSlotState *prod_state,
    PTO2FaninBuilder *fanin_builder, uint8_t ring_id
) {
    if (fanin_builder->mark_seen(prod_ring, prod_slot)) {
        return true;
    }

    if (fanin_builder->count < PTO2_FANIN_INLINE_CAP) {
        fanin_builder->inline_slots[fanin_builder->count++] = prod_state;
        return true;
    }

    PTO2FaninPool &fanin_pool = fanin_builder->spill_pool;
    if (!fanin_pool.ensure_space(orch->sm_header->rings[ring_id], 1)) {
        orch_mark_fatal(orch, PTO2_ERROR_DEP_POOL_OVERFLOW);
        return false;
    }
    int32_t spill_idx = fanin_pool.top;
    PTO2FaninSpillEntry *entry = fanin_pool.alloc();
    if (entry == nullptr) {
        orch_mark_fatal(orch, PTO2_ERROR_DEP_POOL_OVERFLOW);
        return false;
    }
    if (fanin_builder->count == PTO2_FANIN_INLINE_CAP) {
        fanin_builder->spill_start = spill_idx;
    }
    entry->slot_state = prod_state;
    fanin_builder->count++;
    return true;
}

static TensorArgType payload_arg_type(const PTO2TaskPayload &payload, int32_t index) {
    return static_cast<TensorArgType>((payload.arg_tags_packed >> (index * 4)) & 0xfu);
}

static void submit_pipeline_queue_reset(PTO2SubmitPipelineQueue &queue) {
    queue.tail.store(0, std::memory_order_release);
    queue.head.store(0, std::memory_order_release);
    for (int32_t i = 0; i < PTO2_SUBMIT_PIPELINE_QUEUE_CAP; i++) {
        queue.slot_state[i].store(0, std::memory_order_release);
        queue.records[i] = PTO2SubmitCommitRecord{};
    }
}

static void submit_pipeline_queue_push(PTO2SubmitPipelineQueue &queue, const PTO2SubmitCommitRecord &record) {
    uint64_t seq = queue.tail.load(std::memory_order_acquire);
    int32_t slot = static_cast<int32_t>(seq % PTO2_SUBMIT_PIPELINE_QUEUE_CAP);
    while (queue.slot_state[slot].load(std::memory_order_acquire) != 0) {
        SPIN_WAIT_HINT();
    }
    queue.records[slot] = record;
    queue.slot_state[slot].store(1, std::memory_order_release);
    queue.tail.store(seq + 1, std::memory_order_release);
}

static void submit_pipeline_queue_push_task_batch(PTO2SubmitPipelineQueue &queue, int32_t batch_slot, int32_t count) {
    PTO2SubmitCommitRecord record{};
    record.kind = PTO2SubmitPipelineRecordKind::TASK_BATCH;
    record.task_batch_slot = batch_slot;
    record.task_batch_count = count;
    submit_pipeline_queue_push(queue, record);
}

static bool submit_pipeline_queue_pop(PTO2SubmitPipelineQueue &queue, PTO2SubmitCommitRecord *record) {
    uint64_t seq = queue.head.load(std::memory_order_acquire);
    uint64_t tail = queue.tail.load(std::memory_order_acquire);
    if (seq >= tail) {
        return false;
    }
    int32_t slot = static_cast<int32_t>(seq % PTO2_SUBMIT_PIPELINE_QUEUE_CAP);
    while (queue.slot_state[slot].load(std::memory_order_acquire) != 1) {
        SPIN_WAIT_HINT();
    }
    *record = queue.records[slot];
    queue.records[slot] = PTO2SubmitCommitRecord{};
    queue.slot_state[slot].store(0, std::memory_order_release);
    queue.head.store(seq + 1, std::memory_order_release);
    return true;
}

static bool submit_pipeline_queue_empty(PTO2SubmitPipelineQueue &queue) {
    return queue.head.load(std::memory_order_acquire) >= queue.tail.load(std::memory_order_acquire);
}

static void mark_submit_pipeline_work_available(PTO2OrchestratorState *orch) {
    orch->submit_pipeline_work_available.store(true, std::memory_order_release);
    int32_t expected = PTO2_SUBMIT_PIPELINE_CONTROL_IDLE;
    orch->submit_pipeline_control.compare_exchange_strong(
        expected, PTO2_SUBMIT_PIPELINE_CONTROL_WORK, std::memory_order_acq_rel, std::memory_order_acquire
    );
}

static void reset_submit_task_batch_slot(PTO2SubmitTaskBatchSlot &batch) {
    batch.count = 0;
    batch.scope_scheduler = nullptr;
    batch.scope_task_count = 0;
    batch.small_scope_boundary_marker_count = 0;
}

static int32_t acquire_submit_task_batch_slot(PTO2OrchestratorState *orch) {
    while (true) {
        for (int32_t probe = 0; probe < PTO2_SUBMIT_PIPELINE_TASK_BATCH_SLOT_CAP; probe++) {
            int32_t slot =
                (orch->submit_pipeline_next_task_batch_slot + probe) % PTO2_SUBMIT_PIPELINE_TASK_BATCH_SLOT_CAP;
            if (orch->submit_pipeline_task_batch_slot_state[slot].load(std::memory_order_acquire) == 0) {
                orch->submit_pipeline_task_batch_slot_state[slot].store(1, std::memory_order_release);
                reset_submit_task_batch_slot(orch->submit_pipeline_task_batch_slots[slot]);
                orch->submit_pipeline_next_task_batch_slot = (slot + 1) % PTO2_SUBMIT_PIPELINE_TASK_BATCH_SLOT_CAP;
                return slot;
            }
        }
        SPIN_WAIT_HINT();
    }
}

static void release_submit_task_batch_slot(PTO2OrchestratorState *orch, int32_t slot) {
    if (slot < 0 || slot >= PTO2_SUBMIT_PIPELINE_TASK_BATCH_SLOT_CAP) {
        return;
    }
    reset_submit_task_batch_slot(orch->submit_pipeline_task_batch_slots[slot]);
    orch->submit_pipeline_task_batch_slot_state[slot].store(0, std::memory_order_release);
}

static int32_t append_submit_task_batch_record_index(PTO2OrchestratorState *orch) {
    if (orch->submit_pipeline_current_task_batch_slot < 0) {
        orch->submit_pipeline_current_task_batch_slot = acquire_submit_task_batch_slot(orch);
        orch->submit_pipeline_task_batch_count = 0;
    } else if (orch->submit_pipeline_task_batch_count >= PTO2_SUBMIT_PIPELINE_TASK_BATCH_CAP) {
        orch->flush_submit_task_batch();
        orch->submit_pipeline_current_task_batch_slot = acquire_submit_task_batch_slot(orch);
        orch->submit_pipeline_task_batch_count = 0;
    }
    PTO2SubmitTaskBatchSlot &batch =
        orch->submit_pipeline_task_batch_slots[orch->submit_pipeline_current_task_batch_slot];
    int32_t index = batch.count++;
    orch->submit_pipeline_task_batch_count = batch.count;
    return index;
}

static PTO2SubmitCommitRecord *append_submit_task_batch_record_slot(PTO2OrchestratorState *orch) {
    int32_t index = append_submit_task_batch_record_index(orch);
    PTO2SubmitTaskBatchSlot &batch =
        orch->submit_pipeline_task_batch_slots[orch->submit_pipeline_current_task_batch_slot];
    return &batch.records[index];
}

static PTO2DeferredSubmitRecord *append_deferred_submit_task_batch_record_slot(PTO2OrchestratorState *orch) {
    int32_t index = append_submit_task_batch_record_index(orch);
    PTO2SubmitTaskBatchSlot &batch =
        orch->submit_pipeline_task_batch_slots[orch->submit_pipeline_current_task_batch_slot];
    batch.records[index].kind = PTO2SubmitPipelineRecordKind::TASK_DEFERRED;
#if PTO2_PROFILING
    orch->submit_pipeline_compact_deferred_count++;
#endif
    return &batch.deferred_records[index];
}

static void append_submit_task_batch_record(PTO2OrchestratorState *orch, const PTO2SubmitCommitRecord &record) {
    *append_submit_task_batch_record_slot(orch) = record;
}

static int32_t find_scope_tail_task_segment(
    PTO2SubmitTaskBatchSlot &batch, PTO2TaskSlotState **task_slot_states, int32_t count,
    bool compact_deferred_records
) {
    int32_t begin = batch.count - count;
    if (count <= 0 || count > PTO2_SUBMIT_PIPELINE_SMALL_SCOPE_BOUNDARY_TASK_CAP || begin < 0) {
        return -1;
    }
    for (int32_t i = 0; i < count; i++) {
        PTO2SubmitCommitRecord &record = batch.records[begin + i];
        PTO2TaskSlotState *slot_state =
            compact_deferred_records ? batch.deferred_records[begin + i].slot_state : record.slot_state;
        if (record.kind != PTO2SubmitPipelineRecordKind::TASK_DEFERRED || slot_state != task_slot_states[i]) {
            return -1;
        }
    }
    return begin;
}

static bool append_scope_end_to_task_batch_slot(
    PTO2OrchestratorState *orch, PTO2TaskSlotState **task_slot_states, int32_t count
) {
    if (count > PTO2_SUBMIT_PIPELINE_SCOPE_INLINE_CAP) {
        return false;
    }
    int32_t scope_task_batch_begin = -1;
    if (orch->submit_pipeline_current_task_batch_slot >= 0 &&
        orch->submit_pipeline_task_batch_count < PTO2_SUBMIT_PIPELINE_TASK_BATCH_CAP) {
        PTO2SubmitTaskBatchSlot &batch =
            orch->submit_pipeline_task_batch_slots[orch->submit_pipeline_current_task_batch_slot];
        scope_task_batch_begin =
            find_scope_tail_task_segment(batch, task_slot_states, count, orch->submit_pipeline_compact_deferred_records);
    }
    PTO2SubmitCommitRecord &record = *append_submit_task_batch_record_slot(orch);
    record.kind = PTO2SubmitPipelineRecordKind::SCOPE_END;
    record.scheduler = orch->scheduler;
    record.scope_task_count = count;
    record.scope_task_batch_begin = scope_task_batch_begin;
    if (scope_task_batch_begin < 0) {
        for (int32_t i = 0; i < count; i++) {
            record.scope_task_slot_states[i] = task_slot_states[i];
        }
    } else {
        PTO2SubmitTaskBatchSlot &batch =
            orch->submit_pipeline_task_batch_slots[orch->submit_pipeline_current_task_batch_slot];
        batch.small_scope_boundary_marker_count++;
    }
    PTO2SubmitTaskBatchSlot &batch =
        orch->submit_pipeline_task_batch_slots[orch->submit_pipeline_current_task_batch_slot];
    if (batch.small_scope_boundary_marker_count >= PTO2_SUBMIT_PIPELINE_SMALL_SCOPE_BOUNDARY_MARKER_FLUSH_CAP ||
        orch->submit_pipeline_task_batch_count >= PTO2_SUBMIT_PIPELINE_SCOPE_BATCH_RECORD_CAP) {
        orch->flush_submit_task_batch();
    }
    return true;
}

static void write_submit_descriptor(
    PTO2TaskDescriptor &task, PTO2TaskId task_id, void *packed_buffer_base, void *packed_buffer_end,
    int32_t aic_kernel_id, int32_t aiv0_kernel_id, int32_t aiv1_kernel_id
) {
    task.task_id = task_id;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIC)] = aic_kernel_id;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV0)] = aiv0_kernel_id;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV1)] = aiv1_kernel_id;
    task.packed_buffer_base = packed_buffer_base;
    task.packed_buffer_end = packed_buffer_end;
}

template <typename Emit>
static bool compute_payload_task_fanin(
    const PTO2TaskPayload &payload, PTO2TensorMap &tensor_map, bool in_manual_scope, Emit emit
) {
    if (in_manual_scope) {
        return true;
    }

    for (int32_t i = 0; i < payload.tensor_count; i++) {
        TensorArgType ptype = payload_arg_type(payload, i);
        if (ptype == TensorArgType::OUTPUT) {
            continue;
        }

        const Tensor *tensor = &payload.tensors[i];
        PTO2TaskId owner = tensor->owner_task_id;
        if (owner.is_valid() && !emit(owner)) {
            return false;
        }

        if (ptype != TensorArgType::INPUT && ptype != TensorArgType::INOUT) {
            continue;
        }
        if (tensor->manual_dep) {
            continue;
        }

        bool fatal = false;
        tensor_map.lookup(*tensor, [&](PTO2TensorMapEntry &entry, OverlapStatus overlap_status) -> bool {
            if (!emit(entry.producer_task_id)) {
                fatal = true;
                return false;
            }
            if (ptype == TensorArgType::INOUT && overlap_status == OverlapStatus::COVERED) {
                tensor_map.remove_entry(entry);
            }
            return true;
        });
        if (fatal) {
            return false;
        }
    }
    return true;
}

static void register_payload_task_outputs(
    const PTO2TaskPayload &payload, PTO2TaskId task_id, PTO2TensorMap &tensor_map, bool in_manual_scope
) {
    if (in_manual_scope) {
        return;
    }
    for (int32_t i = 0; i < payload.tensor_count; i++) {
        TensorArgType ptype = payload_arg_type(payload, i);
        if (ptype == TensorArgType::INOUT || ptype == TensorArgType::OUTPUT_EXISTING) {
            const Tensor *tensor = &payload.tensors[i];
            if (!tensor->manual_dep) {
                tensor_map.insert(*tensor, task_id);
            }
        }
    }
}

static void commit_submit_descriptor(PTO2SubmitCommitRecord &record) {
    write_submit_descriptor(
        *record.task, record.task_id, record.alloc_result.packed_base, record.alloc_result.packed_end,
        record.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIC)],
        record.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV0)],
        record.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV1)]
    );
}

template <typename SubmitRecord>
static void commit_submit_fanin(SubmitRecord &record) {
    PTO2TaskPayload &payload = *record.payload;
    if (record.fanin_actual_count <= 0) {
        payload.fanin_actual_count = 0;
        payload.fanin_spill_start = 0;
        payload.fanin_spill_pool = record.fanin_spill_pool;
        return;
    }

    for_each_fanin_storage(
        record.fanin_inline_slot_states, record.fanin_actual_count, record.fanin_spill_start,
        *record.fanin_spill_pool,
        [](PTO2TaskSlotState *producer) {
            producer->fanout_count++;
        }
    );

    int32_t inline_count = std::min(record.fanin_actual_count, PTO2_FANIN_INLINE_CAP);
    payload.fanin_actual_count = record.fanin_actual_count;
    payload.fanin_spill_start = record.fanin_spill_start;
    payload.fanin_spill_pool = record.fanin_spill_pool;
    for (int32_t i = 0; i < inline_count; i++) {
        payload.fanin_inline_slot_states[i] = record.fanin_inline_slot_states[i];
    }
}

template <typename SubmitRecord>
static uint64_t commit_submit_publish(SubmitRecord &record) {
    uint64_t spin_count = 0;
    while (!record.scheduler->wiring.queue.push(record.slot_state)) {
        spin_count++;
        SPIN_WAIT_HINT();
    }
    return spin_count;
}

template <typename SubmitRecord>
static void signal_scheduler_drain_if_needed(PTO2OrchestratorState *orch, SubmitRecord &record) {
    if (!orch->submit_pipeline_signal_scheduler_drain || record.scheduler == nullptr) {
        return;
    }
    record.scheduler->wiring.orch_drain_hint_seq.fetch_add(1, std::memory_order_release);
#if PTO2_PROFILING
    orch->submit_pipeline_scheduler_drain_hint_count++;
#endif
}

static void commit_submit_record(PTO2SubmitCommitRecord &record) {
    commit_submit_descriptor(record);
    commit_submit_fanin(record);
    commit_submit_publish(record);
}

static void commit_deferred_descriptor(PTO2SubmitCommitRecord &record) {
    if (record.task == nullptr) {
        return;
    }
    commit_submit_descriptor(record);
}

static void commit_deferred_descriptor(PTO2DeferredSubmitRecord &record) {
    write_submit_descriptor(
        *record.slot_state->task, record.task_id, record.packed_buffer_base, record.packed_buffer_end,
        record.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIC)],
        record.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV0)],
        record.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV1)]
    );
}

template <typename SubmitRecord>
static bool compute_deferred_submit_dependencies(
    PTO2OrchestratorState *orch, SubmitRecord &record, bool sync_tensormap
) {
    uint8_t ring_id = record.task_id.ring();
    if (sync_tensormap) {
        PTO2RingFlowControl &fc = orch->sm_header->rings[ring_id].fc;
        int32_t sm_last_task_alive = fc.last_task_alive.load(std::memory_order_acquire);
        orch->tensor_map.sync_tensormap(record.task_id, sm_last_task_alive);
    }

    PTO2FaninBuilder fanin_builder(orch, orch->rings[ring_id].fanin_pool, next_fanin_seen_epoch(orch));
    for (int32_t i = 0; i < record.explicit_dep_count; i++) {
        PTO2TaskId dep_task_id = record.explicit_deps[i];
        if (!dep_task_id.is_valid()) {
            orch->report_fatal(
                PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "Arg.set_dependencies(...) requires valid task ids"
            );
            return false;
        }
        uint8_t dep_ring_id = dep_task_id.ring();
        PTO2SharedMemoryRingHeader &dep_ring = orch->sm_header->rings[dep_ring_id];
        int32_t dep_local_task_id = static_cast<int32_t>(dep_task_id.local());
        int32_t dep_last_task_alive = dep_ring.fc.last_task_alive.load(std::memory_order_acquire);
        if (dep_local_task_id < dep_last_task_alive) {
            continue;
        }
        int32_t dep_slot = dep_ring.get_slot_by_task_id(dep_local_task_id);
        PTO2TaskSlotState *producer_slot_state = &dep_ring.get_slot_state_by_slot(dep_slot);
        if (!append_fanin_or_fail(orch, dep_ring_id, dep_slot, producer_slot_state, &fanin_builder, ring_id)) {
            return false;
        }
    }

    auto runtime_emit = [&](PTO2TaskId producer_task_id) -> bool {
        uint8_t prod_ring = producer_task_id.ring();
        PTO2SharedMemoryRingHeader &producer_ring = orch->sm_header->rings[prod_ring];
        int32_t prod_slot = producer_ring.get_slot_by_task_id(static_cast<int32_t>(producer_task_id.local()));
        PTO2TaskSlotState *prod_state = &producer_ring.get_slot_state_by_slot(prod_slot);
        return append_fanin_or_fail(orch, prod_ring, prod_slot, prod_state, &fanin_builder, ring_id);
    };

    if (!compute_payload_task_fanin(*record.payload, orch->tensor_map, record.in_manual_scope, runtime_emit)) {
        return false;
    }
    if (record.needs_tensormap_registration) {
        register_payload_task_outputs(*record.payload, record.task_id, orch->tensor_map, record.in_manual_scope);
    }

    record.fanin_actual_count = fanin_builder.count;
    record.fanin_spill_start = fanin_builder.spill_start;
    record.fanin_spill_pool = &fanin_builder.spill_pool;
    int32_t inline_count = std::min(fanin_builder.count, PTO2_FANIN_INLINE_CAP);
    for (int32_t i = 0; i < inline_count; i++) {
        record.fanin_inline_slot_states[i] = fanin_builder.inline_slots[i];
    }
    return true;
}

template <typename SubmitRecord>
static void commit_deferred_submit_record(
    PTO2OrchestratorState *orch, SubmitRecord &record, bool sync_tensormap = true
) {
    if (record.slot_state == nullptr || record.payload == nullptr || record.scheduler == nullptr) {
        return;
    }
    commit_deferred_descriptor(record);
#if PTO2_PROFILING
    uint64_t dep_start = get_sys_cnt_aicpu();
#endif
    if (!compute_deferred_submit_dependencies(orch, record, sync_tensormap)) {
        return;
    }
#if PTO2_PROFILING
    uint64_t dep_end = get_sys_cnt_aicpu();
    orch->submit_pipeline_deferred_dep_cycles += dep_end - dep_start;
    uint64_t fanin_start = dep_end;
#endif
    commit_submit_fanin(record);
#if PTO2_PROFILING
    uint64_t fanin_end = get_sys_cnt_aicpu();
    orch->submit_pipeline_deferred_fanin_cycles += fanin_end - fanin_start;
    uint64_t publish_start = fanin_end;
#endif
    uint64_t publish_spins = record.fanin_actual_count == 0
                                  ? record.scheduler->publish_ready_no_fanin(record.slot_state)
                                  : commit_submit_publish(record);
    if (record.fanin_actual_count != 0) {
        signal_scheduler_drain_if_needed(orch, record);
    }
#if PTO2_PROFILING
    uint64_t publish_end = get_sys_cnt_aicpu();
    orch->submit_pipeline_publish_cycles += publish_end - publish_start;
    orch->submit_pipeline_publish_spins += publish_spins;
    orch->submit_pipeline_deferred_commit_count++;
#else
    (void)publish_spins;
#endif
}

static void commit_scope_end_record(PTO2SubmitCommitRecord &record) {
    if (record.scheduler != nullptr && record.scope_task_count > 0) {
        record.scheduler->on_scope_end(record.scope_task_slot_states, record.scope_task_count);
    }
}

static void commit_scope_end_record_from_task_segment(
    PTO2OrchestratorState *orch, PTO2SubmitTaskBatchSlot &batch, PTO2SubmitCommitRecord &record
) {
    if (record.scheduler == nullptr || record.scope_task_count <= 0 || record.scope_task_batch_begin < 0) {
        commit_scope_end_record(record);
        return;
    }
    int32_t begin = record.scope_task_batch_begin;
    int32_t end = begin + record.scope_task_count;
    if (begin < 0 || end > batch.count) {
        commit_scope_end_record(record);
        return;
    }
    for (int32_t i = 0; i < record.scope_task_count; i++) {
        PTO2SubmitCommitRecord &task_record = batch.records[begin + i];
        PTO2TaskSlotState *slot_state = orch->submit_pipeline_compact_deferred_records
                                            ? batch.deferred_records[begin + i].slot_state
                                            : task_record.slot_state;
        if (task_record.kind != PTO2SubmitPipelineRecordKind::TASK_DEFERRED || slot_state == nullptr) {
            commit_scope_end_record(record);
            return;
        }
    }
    PTO2TaskSlotState *slot_states[PTO2_SUBMIT_PIPELINE_SMALL_SCOPE_BOUNDARY_TASK_CAP];
    for (int32_t i = begin; i < end; i++) {
        slot_states[i - begin] = orch->submit_pipeline_compact_deferred_records ? batch.deferred_records[i].slot_state
                                                                                : batch.records[i].slot_state;
    }
    record.scheduler->release_producers(slot_states, record.scope_task_count);
}

static void sync_deferred_submit_task_batch(
    PTO2OrchestratorState *orch, PTO2SubmitTaskBatchSlot &batch, int32_t begin, int32_t end
) {
    bool ring_seen[PTO2_MAX_RING_DEPTH] = {};
    for (int32_t i = begin; i < end; i++) {
        PTO2SubmitCommitRecord &record = batch.records[i];
        if (record.kind != PTO2SubmitPipelineRecordKind::TASK_DEFERRED) {
            continue;
        }
        PTO2TaskId task_id =
            orch->submit_pipeline_compact_deferred_records ? batch.deferred_records[i].task_id : record.task_id;
        ring_seen[task_id.ring()] = true;
    }

    for (int32_t ring_id = 0; ring_id < PTO2_MAX_RING_DEPTH; ring_id++) {
        if (!ring_seen[ring_id]) {
            continue;
        }
        PTO2RingFlowControl &fc = orch->sm_header->rings[ring_id].fc;
        int32_t sm_last_task_alive = fc.last_task_alive.load(std::memory_order_acquire);
        orch->tensor_map.sync_validity(ring_id, sm_last_task_alive);

        bool needs_cleanup =
            sm_last_task_alive - orch->tensor_map.last_cleanup[ring_id] >= PTO2_TENSORMAP_CLEANUP_INTERVAL;
        if (!needs_cleanup) {
            uint32_t cleanup_slot = orch->tensor_map.get_task_local_id_slot(
                static_cast<uint8_t>(ring_id), static_cast<uint32_t>(orch->tensor_map.last_cleanup[ring_id])
            );
            for (int32_t i = begin; i < end; i++) {
                PTO2SubmitCommitRecord &record = batch.records[i];
                if (record.kind != PTO2SubmitPipelineRecordKind::TASK_DEFERRED) {
                    continue;
                }
                PTO2TaskId task_id = orch->submit_pipeline_compact_deferred_records
                                          ? batch.deferred_records[i].task_id
                                          : record.task_id;
                if (task_id.ring() != ring_id) {
                    continue;
                }
                if (orch->tensor_map.get_task_local_id_slot(static_cast<uint8_t>(ring_id), task_id.local()) ==
                    cleanup_slot) {
                    needs_cleanup = true;
                    break;
                }
            }
        }
        if (needs_cleanup) {
            orch->tensor_map.cleanup_retired(ring_id, orch->tensor_map.last_cleanup[ring_id], sm_last_task_alive);
            orch->tensor_map.last_cleanup[ring_id] = sm_last_task_alive;
        }
    }
}

static void commit_submit_pipeline_task_record(
    PTO2OrchestratorState *orch, PTO2SubmitCommitRecord &record, bool sync_tensormap = true
) {
    if (record.kind == PTO2SubmitPipelineRecordKind::SCOPE_END) {
        commit_scope_end_record(record);
    } else if (record.kind == PTO2SubmitPipelineRecordKind::TASK_DEFERRED) {
        commit_deferred_submit_record(orch, record, sync_tensormap);
    } else {
        commit_submit_record(record);
    }
}

static int32_t count_submit_task_records(PTO2SubmitTaskBatchSlot &batch, int32_t count) {
    int32_t task_count = 0;
    for (int32_t i = 0; i < count; i++) {
        PTO2SubmitPipelineRecordKind kind = batch.records[i].kind;
        if (kind == PTO2SubmitPipelineRecordKind::TASK_COMMIT || kind == PTO2SubmitPipelineRecordKind::TASK_DEFERRED) {
            task_count++;
        }
    }
    return task_count;
}

static void commit_submit_task_batch_record(PTO2OrchestratorState *orch, PTO2SubmitCommitRecord &record) {
    int32_t slot = record.task_batch_slot;
    if (slot < 0 || slot >= PTO2_SUBMIT_PIPELINE_TASK_BATCH_SLOT_CAP) {
        return;
    }
    PTO2SubmitTaskBatchSlot &batch = orch->submit_pipeline_task_batch_slots[slot];
    int32_t count = batch.count;
    if (record.task_batch_count > 0 && record.task_batch_count < count) {
        count = record.task_batch_count;
    }
    int32_t begin = 0;
    while (begin < count) {
        if (batch.records[begin].kind == PTO2SubmitPipelineRecordKind::SCOPE_END) {
#if PTO2_PROFILING
            uint64_t scope_start = get_sys_cnt_aicpu();
#endif
            commit_scope_end_record_from_task_segment(orch, batch, batch.records[begin]);
#if PTO2_PROFILING
            uint64_t scope_cycles = get_sys_cnt_aicpu() - scope_start;
            orch->submit_pipeline_scope_release_cycles += scope_cycles;
            orch->submit_pipeline_scope_record_cycles += scope_cycles;
            orch->submit_pipeline_scope_record_count++;
#endif
            begin++;
            continue;
        }

        int32_t end = begin;
        while (end < count && batch.records[end].kind != PTO2SubmitPipelineRecordKind::SCOPE_END) {
            end++;
        }
#if PTO2_PROFILING
        uint64_t sync_start = get_sys_cnt_aicpu();
#endif
        sync_deferred_submit_task_batch(orch, batch, begin, end);
#if PTO2_PROFILING
        orch->submit_pipeline_batch_sync_cycles += get_sys_cnt_aicpu() - sync_start;
#endif
        for (int32_t i = begin; i < end; i++) {
            if (orch->submit_pipeline_compact_deferred_records &&
                batch.records[i].kind == PTO2SubmitPipelineRecordKind::TASK_DEFERRED) {
                commit_deferred_submit_record(orch, batch.deferred_records[i], false);
            } else {
                commit_submit_pipeline_task_record(orch, batch.records[i], false);
            }
        }
        begin = end;
    }
    if (batch.scope_scheduler != nullptr && batch.scope_task_count > 0) {
#if PTO2_PROFILING
        uint64_t scope_start = get_sys_cnt_aicpu();
#endif
        batch.scope_scheduler->on_scope_end(batch.scope_task_slot_states, batch.scope_task_count);
#if PTO2_PROFILING
        orch->submit_pipeline_scope_release_cycles += get_sys_cnt_aicpu() - scope_start;
#endif
    }
    release_submit_task_batch_slot(orch, slot);
}

static bool commit_single_deferred_task_batch_inline(PTO2OrchestratorState *orch, int32_t slot, int32_t count) {
    if (orch->submit_pipeline_commit_stages != 1 || !orch->submit_pipeline_defer_dependencies || count <= 0 ||
        count > 2) {
        return false;
    }

    PTO2SubmitTaskBatchSlot &batch = orch->submit_pipeline_task_batch_slots[slot];
    int32_t deferred_task_count = 0;
    for (int32_t i = 0; i < count; i++) {
        PTO2SubmitPipelineRecordKind kind = batch.records[i].kind;
        if (kind == PTO2SubmitPipelineRecordKind::TASK_DEFERRED) {
            deferred_task_count++;
        } else if (kind != PTO2SubmitPipelineRecordKind::SCOPE_END) {
            return false;
        }
    }
    if (deferred_task_count != 1) {
        return false;
    }

    PTO2SubmitCommitRecord record{};
    record.kind = PTO2SubmitPipelineRecordKind::TASK_BATCH;
    record.task_batch_slot = slot;
    record.task_batch_count = count;
    commit_submit_task_batch_record(orch, record);
    return true;
}

void PTO2OrchestratorState::enable_submit_pipeline(
    int32_t orch_thread_num, bool enqueue_submit_records, bool defer_dependencies, bool signal_scheduler_drain,
    bool compact_deferred_records
) {
    submit_pipeline_commit_stages = 0;
    if (orch_thread_num >= 4) {
        submit_pipeline_commit_stages = 3;
    } else if (orch_thread_num >= 2) {
        submit_pipeline_commit_stages = 1;
    }
    submit_pipeline_enabled = submit_pipeline_commit_stages > 0;
    submit_pipeline_enqueue_submit_records = submit_pipeline_enabled && enqueue_submit_records;
    submit_pipeline_defer_dependencies =
        submit_pipeline_enabled && submit_pipeline_commit_stages == 1 && defer_dependencies;
    submit_pipeline_signal_scheduler_drain =
        submit_pipeline_defer_dependencies && signal_scheduler_drain && scheduler != nullptr;
    submit_pipeline_compact_deferred_records = submit_pipeline_defer_dependencies && compact_deferred_records;
    submit_pipeline_current_task_batch_slot = -1;
    submit_pipeline_task_batch_count = 0;
    submit_pipeline_next_task_batch_slot = 0;
    submit_pipeline_stop.store(false, std::memory_order_release);
    submit_pipeline_work_available.store(false, std::memory_order_release);
    submit_pipeline_control.store(PTO2_SUBMIT_PIPELINE_CONTROL_IDLE, std::memory_order_release);
    submit_pipeline_completed.store(0, std::memory_order_release);
#if PTO2_PROFILING
    submit_pipeline_task_enqueue_cycles = 0;
    submit_pipeline_scope_enqueue_cycles = 0;
    submit_pipeline_flush_cycles = 0;
    submit_pipeline_batch_sync_cycles = 0;
    submit_pipeline_deferred_dep_cycles = 0;
    submit_pipeline_deferred_fanin_cycles = 0;
    submit_pipeline_publish_cycles = 0;
    submit_pipeline_scope_release_cycles = 0;
    submit_pipeline_scope_record_cycles = 0;
    submit_pipeline_publish_spins = 0;
    submit_pipeline_scheduler_drain_hint_count = 0;
    submit_pipeline_task_enqueue_count = 0;
    submit_pipeline_task_enqueue_batch_count = 0;
    submit_pipeline_scope_enqueue_count = 0;
    submit_pipeline_flush_count = 0;
    submit_pipeline_deferred_commit_count = 0;
    submit_pipeline_scope_record_count = 0;
    submit_pipeline_compact_deferred_count = 0;
#endif
    for (int32_t stage = 0; stage < PTO2_SUBMIT_PIPELINE_MAX_COMMIT_STAGES; stage++) {
        submit_pipeline_stage_done[stage].store(false, std::memory_order_release);
        submit_pipeline_queue_reset(submit_pipeline_queues[stage]);
    }
    for (int32_t i = 0; i < PTO2_SUBMIT_PIPELINE_TASK_BATCH_SLOT_CAP; i++) {
        submit_pipeline_task_batch_slot_state[i].store(0, std::memory_order_release);
        reset_submit_task_batch_slot(submit_pipeline_task_batch_slots[i]);
    }
}

uint64_t PTO2OrchestratorState::run_submit_pipeline_worker(int32_t stage_idx) {
    uint64_t committed = 0;
    if (!submit_pipeline_enabled || stage_idx < 1 || stage_idx > submit_pipeline_commit_stages) {
        while (!submit_pipeline_stop.load(std::memory_order_acquire)) {
            SPIN_WAIT_HINT();
        }
        return committed;
    }

    int32_t commit_stage = stage_idx - 1;
    PTO2SubmitPipelineQueue &input_queue = submit_pipeline_queues[commit_stage];
    if (commit_stage == 0) {
        while (submit_pipeline_control.load(std::memory_order_acquire) == PTO2_SUBMIT_PIPELINE_CONTROL_IDLE) {
            SPIN_WAIT_HINT();
        }
        if (submit_pipeline_control.load(std::memory_order_acquire) == PTO2_SUBMIT_PIPELINE_CONTROL_STOP &&
            submit_pipeline_queues[0].tail.load(std::memory_order_acquire) == 0) {
            submit_pipeline_stage_done[commit_stage].store(true, std::memory_order_release);
            return committed;
        }
        while (!submit_pipeline_work_available.load(std::memory_order_acquire)) {
            if (submit_pipeline_stop.load(std::memory_order_acquire) ||
                submit_pipeline_control.load(std::memory_order_acquire) == PTO2_SUBMIT_PIPELINE_CONTROL_STOP) {
                submit_pipeline_stage_done[commit_stage].store(true, std::memory_order_release);
                return committed;
            }
            SPIN_WAIT_HINT();
        }
    }

    while (true) {
        PTO2SubmitCommitRecord record{};
        if (submit_pipeline_queue_pop(input_queue, &record)) {
            if (submit_pipeline_commit_stages == 1) {
                if (record.kind == PTO2SubmitPipelineRecordKind::SCOPE_END) {
                    commit_scope_end_record(record);
                } else if (record.kind == PTO2SubmitPipelineRecordKind::TASK_BATCH) {
                    commit_submit_task_batch_record(this, record);
                } else if (record.kind == PTO2SubmitPipelineRecordKind::TASK_DEFERRED) {
                    commit_deferred_submit_record(this, record);
                } else {
                    commit_submit_record(record);
                }
            } else if (record.kind == PTO2SubmitPipelineRecordKind::SCOPE_END) {
                commit_scope_end_record(record);
            } else if (commit_stage == 0) {
                commit_submit_descriptor(record);
                submit_pipeline_queue_push(submit_pipeline_queues[1], record);
            } else if (commit_stage == 1) {
                commit_submit_fanin(record);
                submit_pipeline_queue_push(submit_pipeline_queues[2], record);
            } else {
                commit_submit_publish(record);
                signal_scheduler_drain_if_needed(this, record);
            }
            submit_pipeline_completed.fetch_add(1, std::memory_order_acq_rel);
            committed++;
            continue;
        }
        bool upstream_done = commit_stage == 0
                                 ? submit_pipeline_stop.load(std::memory_order_acquire)
                                 : submit_pipeline_stage_done[commit_stage - 1].load(std::memory_order_acquire);
        if (upstream_done && submit_pipeline_queue_empty(input_queue)) {
            submit_pipeline_stage_done[commit_stage].store(true, std::memory_order_release);
            return committed;
        }
        SPIN_WAIT_HINT();
    }
}

void PTO2OrchestratorState::flush_submit_task_batch() {
    if (!submit_pipeline_enabled || submit_pipeline_current_task_batch_slot < 0) {
        return;
    }
    int32_t slot = submit_pipeline_current_task_batch_slot;
    int32_t count = submit_pipeline_task_batch_count;
    bool has_scope_end = submit_pipeline_task_batch_slots[slot].scope_scheduler != nullptr &&
                         submit_pipeline_task_batch_slots[slot].scope_task_count > 0;
    if (count <= 0 && !has_scope_end) {
        release_submit_task_batch_slot(this, slot);
        submit_pipeline_current_task_batch_slot = -1;
        submit_pipeline_task_batch_count = 0;
        return;
    }
    submit_pipeline_task_batch_slots[slot].count = count;
    if (commit_single_deferred_task_batch_inline(this, slot, count)) {
        submit_pipeline_current_task_batch_slot = -1;
        submit_pipeline_task_batch_count = 0;
        return;
    }
#if PTO2_PROFILING
    uint64_t enqueue_start = get_sys_cnt_aicpu();
    int32_t task_count = count_submit_task_records(submit_pipeline_task_batch_slots[slot], count);
#endif
    mark_submit_pipeline_work_available(this);
    submit_pipeline_queue_push_task_batch(submit_pipeline_queues[0], slot, count);
#if PTO2_PROFILING
    submit_pipeline_task_enqueue_cycles += get_sys_cnt_aicpu() - enqueue_start;
    submit_pipeline_task_enqueue_count += static_cast<uint32_t>(task_count);
    submit_pipeline_task_enqueue_batch_count++;
#endif
    submit_pipeline_current_task_batch_slot = -1;
    submit_pipeline_task_batch_count = 0;
}

void PTO2OrchestratorState::flush_submit_pipeline() {
    if (!submit_pipeline_enabled) {
        return;
    }
    flush_submit_task_batch();
    uint64_t target = submit_pipeline_queues[0].tail.load(std::memory_order_acquire);
    if (submit_pipeline_completed.load(std::memory_order_acquire) >= target) {
        return;
    }
#if PTO2_PROFILING
    uint64_t flush_start = get_sys_cnt_aicpu();
#endif
    while (submit_pipeline_completed.load(std::memory_order_acquire) < target) {
        SPIN_WAIT_HINT();
    }
#if PTO2_PROFILING
    submit_pipeline_flush_cycles += get_sys_cnt_aicpu() - flush_start;
    submit_pipeline_flush_count++;
#endif
}

void PTO2OrchestratorState::log_submit_pipeline_diagnostics(int32_t thread_idx) const {
#if PTO2_PROFILING
    LOG_INFO_V9(
        "Thread %d: orch_pipeline_diag task_enq_count=%u task_enq_batches=%u task_enq_cost=%.3fus "
        "scope_enq_count=%u scope_enq_cost=%.3fus flush_count=%u flush_cost=%.3fus "
        "batch_sync_cost=%.3fus deferred_dep_cost=%.3fus deferred_fanin_cost=%.3fus publish_cost=%.3fus "
        "scope_release_cost=%.3fus scope_record_cost=%.3fus deferred_commit_count=%u scope_record_count=%u "
        "drain_hints=%u compact_deferred=%u publish_spins=%" PRIu64,
        thread_idx, submit_pipeline_task_enqueue_count, submit_pipeline_task_enqueue_batch_count,
        cycles_to_us(submit_pipeline_task_enqueue_cycles), submit_pipeline_scope_enqueue_count,
        cycles_to_us(submit_pipeline_scope_enqueue_cycles), submit_pipeline_flush_count,
        cycles_to_us(submit_pipeline_flush_cycles), cycles_to_us(submit_pipeline_batch_sync_cycles),
        cycles_to_us(submit_pipeline_deferred_dep_cycles), cycles_to_us(submit_pipeline_deferred_fanin_cycles),
        cycles_to_us(submit_pipeline_publish_cycles), cycles_to_us(submit_pipeline_scope_release_cycles),
        cycles_to_us(submit_pipeline_scope_record_cycles), submit_pipeline_deferred_commit_count,
        submit_pipeline_scope_record_count, submit_pipeline_scheduler_drain_hint_count,
        submit_pipeline_compact_deferred_count, static_cast<uint64_t>(submit_pipeline_publish_spins)
    );
#else
    (void)thread_idx;
#endif
}

void PTO2OrchestratorState::stop_submit_pipeline() {
    if (!submit_pipeline_enabled) {
        submit_pipeline_stop.store(true, std::memory_order_release);
        submit_pipeline_control.store(PTO2_SUBMIT_PIPELINE_CONTROL_STOP, std::memory_order_release);
        return;
    }
    flush_submit_pipeline();
    submit_pipeline_stop.store(true, std::memory_order_release);
    if (!submit_pipeline_work_available.load(std::memory_order_acquire)) {
        submit_pipeline_control.store(PTO2_SUBMIT_PIPELINE_CONTROL_STOP, std::memory_order_release);
    }
}

static bool enqueue_scope_end_record_if_supported(
    PTO2OrchestratorState *orch, PTO2TaskSlotState **task_slot_states, int32_t count
) {
    if (!orch->submit_pipeline_enabled || orch->submit_pipeline_commit_stages != 1 ||
        count > PTO2_SUBMIT_PIPELINE_SCOPE_INLINE_CAP) {
        return false;
    }

    if (append_scope_end_to_task_batch_slot(orch, task_slot_states, count)) {
        return true;
    }

    PTO2SubmitCommitRecord record{};
    record.kind = PTO2SubmitPipelineRecordKind::SCOPE_END;
    record.scheduler = orch->scheduler;
    record.scope_task_count = count;
    for (int32_t i = 0; i < count; i++) {
        record.scope_task_slot_states[i] = task_slot_states[i];
    }
    orch->flush_submit_task_batch();
#if PTO2_PROFILING
    uint64_t enqueue_start = get_sys_cnt_aicpu();
#endif
    mark_submit_pipeline_work_available(orch);
    submit_pipeline_queue_push(orch->submit_pipeline_queues[0], record);
#if PTO2_PROFILING
    orch->submit_pipeline_scope_enqueue_cycles += get_sys_cnt_aicpu() - enqueue_start;
    orch->submit_pipeline_scope_enqueue_count++;
#endif
    return true;
}

static void scope_tasks_push(PTO2OrchestratorState *orch, PTO2TaskSlotState *task_slot_state);

struct PTO2PreparedTask {
    PTO2TaskId task_id = PTO2TaskId::invalid();
    PTO2TaskAllocResult alloc_result = {-1, 0, nullptr, nullptr};
    PTO2TaskDescriptor *task = nullptr;
    PTO2TaskPayload *payload = nullptr;
    PTO2TaskSlotState *slot_state = nullptr;
};

static PTO2OutputLayout calculate_output_layout(const L0TaskArgs &args) {
    PTO2OutputLayout layout;
    for (int32_t i = 0; i < args.tensor_count(); i++) {
        TensorArgType tag = args.tag(i);
        if (tag == TensorArgType::INOUT || tag == TensorArgType::OUTPUT_EXISTING) {
            layout.needs_tensormap_registration = true;
        }
        if (tag != TensorArgType::OUTPUT) {
            continue;
        }
        layout.offsets[i] = layout.total_output_size;
        layout.buffer_sizes[i] =
            PTO2_ALIGN_UP(args.tensor(i).create_info().buffer_size_bytes(), PTO2_PACKED_OUTPUT_ALIGN);
        layout.total_output_size += layout.buffer_sizes[i];
    }
    return layout;
}

static bool check_scope_can_accept_task(PTO2OrchestratorState *orch, PTO2TaskAllocator &allocator, uint8_t ring_id) {
    always_assert(orch->scope_stack_top >= 0 && "Cannot submit task outside a scope");

    int32_t scope_task_count = orch->scope_tasks_size - orch->scope_begins[orch->scope_stack_top];
    if (scope_task_count < allocator.window_size() - 1) {
        return true;
    }

    int32_t active_count = allocator.active_count();

    LOG_ERROR("========================================");
    LOG_ERROR("FATAL: Scope Deadlock Detected! (ring %d)", ring_id);
    LOG_ERROR("========================================");
    LOG_ERROR("Tasks in current scope (%d) >= task_window_size (%d).", scope_task_count, allocator.window_size());
    LOG_ERROR("  scope_depth:        %d", orch->scope_stack_top + 1);
    LOG_ERROR("  ring_id:            %d", ring_id);
    LOG_ERROR("  scope_task_count:   %d", scope_task_count);
    LOG_ERROR("  active_tasks:       %d / %d", active_count, allocator.window_size());
    LOG_ERROR("Root Cause:");
    LOG_ERROR("  Tasks within a scope hold a fanout_count reference that is only");
    LOG_ERROR("  released at scope_end. When scope task count >= window_size,");
    LOG_ERROR("  no slots can be reclaimed -> deadlock.");
    LOG_ERROR("Solution:");
    LOG_ERROR("  1. Reduce tasks per scope (use batching/unroll)");
    LOG_ERROR("  2. Increase task window (current: %d)", allocator.window_size());
    LOG_ERROR("     Compile-time: PTO2_TASK_WINDOW_SIZE in pto_runtime2_types.h");
    LOG_ERROR("     Runtime env:  PTO2_RING_TASK_WINDOW=<power-of-2>");
    LOG_ERROR("  3. Split work across multiple scopes");
    LOG_ERROR("========================================");
    orch_mark_fatal(orch, PTO2_ERROR_SCOPE_DEADLOCK);
    return false;
}

static bool prepare_task(
    PTO2OrchestratorState *orch, const L0TaskArgs &args, int32_t total_output_size, ActiveMask active_mask,
    PTO2PreparedTask *out
) {
    uint8_t ring_id = orch->current_ring_id();
    auto &allocator = orch->rings[ring_id].task_allocator;

    if (!check_scope_can_accept_task(orch, allocator, ring_id)) {
        return false;
    }

    if (orch->submit_pipeline_defer_dependencies && total_output_size > 0 &&
        allocator.allocation_would_block_on_heap(total_output_size)) {
        orch->flush_submit_pipeline();
    }

    out->alloc_result = allocator.alloc(total_output_size);
    if (out->alloc_result.failed()) {
        orch_mark_fatal(orch, PTO2_ERROR_HEAP_RING_DEADLOCK);
        return false;
    }

    out->task_id = PTO2TaskId::make(ring_id, static_cast<uint32_t>(out->alloc_result.task_id));
    out->slot_state = &orch->sm_header->rings[ring_id].get_slot_state_by_slot(out->alloc_result.slot);
    out->task = &orch->sm_header->rings[ring_id].task_descriptors[out->alloc_result.slot];
    out->payload = &orch->sm_header->rings[ring_id].task_payloads[out->alloc_result.slot];

    out->payload->prefetch(args.tensor_count(), args.scalar_count());

    // Re-bind payload/task pointers each submit. Value is per-slot constant
    // (same as &task_payloads[slot] / &task_descriptors[slot]), but writing
    // here lets RingSchedState::init() skip the O(window_size) bind loop.
    // Both writes hit the same 64B slot_state cache line we're about to
    // dirty below, so the extra cost is two stores on an already-hot line.
    // Must precede the scheduler wiring.queue.push at the end of
    // submit_task_common — that push is the first read of slot_state->task /
    // slot_state->payload by another thread.
    out->slot_state->bind_buffers(out->payload, out->task);

    // prepare_task does NO payload writes: all payload content (tensors/scalars +
    // early-dispatch spec fields) is initialized in PTO2TaskPayload::init, the
    // single payload-init point, which runs before the scheduler wiring push.

    // Fields already reset by advance_ring_pointers (eager reset after CONSUMED):
    //   fanout_lock=0, fanout_count=1, fanout_head=nullptr,
    //   fanin_refcount=0, fanout_refcount=0, completed_subtasks=0, next_block_idx=0
    // Fields immutable after RingSchedState::init():
    //   ring_id
    // task_state left as CONSUMED by eager reset (safe for stale wait_for_tensor
    // observers); set to PENDING here when orchestrator actually reuses the slot.
    out->slot_state->task_state.store(PTO2_TASK_PENDING, std::memory_order_relaxed);
    int16_t block_num = args.launch_spec.block_num();
    out->slot_state->total_required_subtasks =
        static_cast<int16_t>(block_num * __builtin_popcount(active_mask.core_mask()));
    out->slot_state->logical_block_num = block_num;
    out->slot_state->active_mask = active_mask;
    // fanin_count is set by scheduler during wiring
    scope_tasks_push(orch, out->slot_state);

    return true;
}

// =============================================================================
// Scope Management
// =============================================================================

static void scope_tasks_push(PTO2OrchestratorState *orch, PTO2TaskSlotState *task_slot_state) {
    if (orch->scope_tasks_size >= orch->scope_tasks_capacity) {
        // scope_tasks lives in the per-Worker arena (single backing allocation),
        // so realloc is not legal. Capacity == PTO2_SCOPE_TASKS_CAP ==
        // PTO2_TASK_WINDOW_SIZE × PTO2_MAX_RING_DEPTH, the total in-flight slot
        // budget — hitting it means every ring is saturated, so no further push
        // could succeed regardless of buffer growth.
        orch->report_fatal(
            PTO2_ERROR_SCOPE_TASKS_OVERFLOW, __FUNCTION__,
            "scope_tasks buffer saturated at %d entries (all rings full)", orch->scope_tasks_capacity
        );
        return;
    }
    orch->scope_tasks[orch->scope_tasks_size++] = task_slot_state;
}

void PTO2OrchestratorState::begin_scope(PTO2ScopeMode mode) {
    auto *orch = this;
    if (orch->fatal) {
        return;
    }
    assert(orch->scope_stack_top < static_cast<int32_t>(orch->scope_stack_capacity - 1) && "Scope stack overflow");
    if (mode == PTO2ScopeMode::AUTO && orch->in_manual_scope()) {
        report_fatal(PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "auto scope nested inside manual scope is not supported");
        return;
    }

    bool already_in_manual_scope = orch->in_manual_scope();
    ++orch->scope_stack_top;
    orch->scope_begins[orch->scope_stack_top] = orch->scope_tasks_size;
    if (mode == PTO2ScopeMode::MANUAL && !already_in_manual_scope) {
        orch->manual_begin_depth = orch->scope_stack_top;
    }
#if PTO2_PROFILING
    // Gate via is_scope_stats_enabled() (weak-false in host builds) BEFORE the
    // collector call: when disabled we pay nothing. Sample the current ring's
    // task/heap start-end and tensormap usage at the scope boundary.
    if (is_scope_stats_enabled()) {
        uint8_t ring_id = orch->current_ring_id();
        auto &alloc = orch->rings[ring_id].task_allocator;
        int32_t dep_pool_tail = 0;
        int32_t dep_pool_top = 0;
        if (orch->scheduler) {
            orch->scheduler->ring_sched_states[ring_id].read_dep_pool_snapshot(dep_pool_tail, dep_pool_top);
        }
        scope_stats_begin(
            ring_id, alloc.task_tail(), alloc.task_head(), alloc.heap_tail(), alloc.heap_top(), dep_pool_tail,
            dep_pool_top, orch->tensor_map.current_used()
        );
    }
#endif
}

void PTO2OrchestratorState::end_scope() {
    auto *orch = this;
    if (orch->fatal) {
        return;
    }
    assert(orch->scope_stack_top >= 0 && "Scope stack underflow");

    // Snapshot the ring start/end BEFORE the orchestrator drains pending tasks
    // via scheduler->on_scope_end, so the end record reflects the scope's
    // occupancy at close, not the residual after teardown.
#if PTO2_PROFILING
    // Gate via is_scope_stats_enabled() (see begin_scope). One collector call
    // emits the end-boundary record and tears down bookkeeping.
    if (is_scope_stats_enabled()) {
        uint8_t ring_id = orch->current_ring_id();
        auto &alloc = orch->rings[ring_id].task_allocator;
        int32_t dep_pool_tail = 0;
        int32_t dep_pool_top = 0;
        if (orch->scheduler) {
            orch->scheduler->ring_sched_states[ring_id].read_dep_pool_snapshot(dep_pool_tail, dep_pool_top);
        }
        scope_stats_end(
            ring_id, alloc.task_tail(), alloc.task_head(), alloc.heap_tail(), alloc.heap_top(), dep_pool_tail,
            dep_pool_top, orch->tensor_map.current_used()
        );
    }
#endif

#if PTO2_ORCH_PROFILING
    uint64_t _se0 = get_sys_cnt_aicpu();
#endif

    bool ending_manual_scope = orch->scope_stack_top == orch->manual_begin_depth;
    int32_t begin = orch->scope_begins[orch->scope_stack_top--];
    int32_t count = orch->scope_tasks_size - begin;
    if (ending_manual_scope) {
        orch->manual_begin_depth = PTO2_MAX_SCOPE_DEPTH;
    }

    bool scope_end_queued = false;
    if (orch->scheduler && count > 0) {
        scope_end_queued = enqueue_scope_end_record_if_supported(orch, &orch->scope_tasks[begin], count);
    }
    if (!scope_end_queued) {
        orch->flush_submit_pipeline();
    }
    if (!scope_end_queued && orch->scheduler && count > 0) {
        orch->scheduler->on_scope_end(&orch->scope_tasks[begin], count);
    }

    // Rewind the task buffer — these entries are no longer needed
    orch->scope_tasks_size = begin;

#if PTO2_ORCH_PROFILING
    uint64_t _se1 = get_sys_cnt_aicpu();
    g_orch_scope_end_cycle += (_se1 - _se0);
#endif
}

// =============================================================================
// Task Submission
// =============================================================================

// Shared body for submit_task / submit_dummy_task. Caller has already validated
// args.has_error, decided active_mask (empty for dummy), and resolved the per-slot
// kernel_ids (all INVALID_KERNEL_ID for dummy). Performs tensormap sync, fanin
// computation (explicit_deps + auto), output registration, slot init, and pushes
// to the scheduler wiring queue.
static TaskOutputTensors submit_task_common(
    PTO2OrchestratorState *orch, const L0TaskArgs &args, ActiveMask active_mask, int32_t aic_kernel_id,
    int32_t aiv0_kernel_id, int32_t aiv1_kernel_id
) {
    CYCLE_COUNT_START();
    TaskOutputTensors result;
    PTO2OutputLayout layout = calculate_output_layout(args);
    bool heap_guard_sync = orch->submit_pipeline_defer_dependencies &&
                           layout.total_output_size >= PTO2_SUBMIT_PIPELINE_HEAP_GUARD_OUTPUT_BYTES;
    if (heap_guard_sync) {
        uint8_t guard_ring_id = orch->current_ring_id();
        auto &allocator = orch->rings[guard_ring_id].task_allocator;
        heap_guard_sync = allocator.allocation_would_block_on_heap(layout.total_output_size);
    }
    if (heap_guard_sync) {
        orch->flush_submit_pipeline();
        uint8_t guard_ring_id = orch->current_ring_id();
        auto &allocator = orch->rings[guard_ring_id].task_allocator;
        heap_guard_sync = allocator.allocation_would_block_on_heap(layout.total_output_size);
    }
    PTO2PreparedTask prepared;
    if (!prepare_task(orch, args, layout.total_output_size, active_mask, &prepared)) {
        return result;
    }
    uint8_t ring_id = prepared.task_id.ring();
    PTO2SchedulerState *sched = orch->scheduler;
    PTO2RingFlowControl &fc = orch->sm_header->rings[ring_id].fc;
    PTO2TaskId task_id = prepared.task_id;
    PTO2TaskSlotState &cur_slot_state = *prepared.slot_state;
    PTO2TaskDescriptor &task = *prepared.task;
    PTO2TaskPayload &payload = *prepared.payload;
    result.set_task_id(task_id);
    bool defer_dependencies = orch->submit_pipeline_defer_dependencies && !heap_guard_sync;
    if (defer_dependencies && args.explicit_dep_count() > PTO2_SUBMIT_PIPELINE_EXPLICIT_DEP_CAP) {
        orch->flush_submit_pipeline();
        defer_dependencies = false;
    }

    // dep_gen capture point: snapshot the orch submit_task inputs while the
    // tensormap is still in its pre-lookup state for this task. Replay reads
    // these records offline to reconstruct the complete dep graph — the sole
    // source of truth for fanout now that the swimlane hot path no longer
    // records it.
    if (is_dep_gen_enabled()) {
        const void *tensor_ptrs[MAX_TENSOR_ARGS];
        // TensorArgType is `enum class : int32_t` (4 bytes); the on-disk record
        // packs arg_types as uint8_t[16] (5-value enum fits in a byte). Narrow
        // each tag here rather than letting the AICPU writer reinterpret a
        // 4×-wider array as bytes — that path silently lost two of every three
        // tags on little-endian and synthesized phantom self-edges in replay.
        uint8_t arg_types_u8[MAX_TENSOR_ARGS];
        // Clamp to MAX_TENSOR_ARGS even though the Arg builder caps adds at
        // MAX_TENSOR_ARGS: defensive against any future builder bypass /
        // shared-memory bit-flip that could otherwise overrun the two
        // MAX_TENSOR_ARGS-sized stack buffers above.
        const int tc_raw = args.tensor_count();
        const int tc = tc_raw > MAX_TENSOR_ARGS ? MAX_TENSOR_ARGS : tc_raw;
        for (int i = 0; i < tc; i++) {
            // OUTPUT slots carry create_info (not yet a Tensor); skip them —
            // they have no producer to look up and replay's per-tensor loop
            // also skips OUTPUT.
            tensor_ptrs[i] = (args.tag(i) == TensorArgType::OUTPUT) ? nullptr : &args.tensor(i).ref();
            arg_types_u8[i] = static_cast<uint8_t>(args.tag(i));
        }
        const int32_t kernel_ids_capture[3] = {aic_kernel_id, aiv0_kernel_id, aiv1_kernel_id};
        dep_gen_aicpu_record_submit(
            task_id.raw, orch->in_manual_scope(), tc, tensor_ptrs, arg_types_u8,
            static_cast<int>(args.explicit_dep_count()), reinterpret_cast<const uint64_t *>(args.explicit_deps_data()),
            kernel_ids_capture
        );
    }

    CYCLE_COUNT_LAP(g_orch_alloc_cycle);

#if PTO2_PROFILING
    if (layout.total_output_size > 0) {
        orch->buffers_allocated++;
        orch->bytes_allocated += layout.total_output_size;
    }
#endif

    if (!defer_dependencies) {
        PTO2FaninBuilder fanin_builder(orch, orch->rings[ring_id].fanin_pool, next_fanin_seen_epoch(orch));

        // === STEP 2: Sync TensorMap validity and optional cleanup ===
        // Read current last_task_alive from shared memory for this ring
        int32_t sm_last_task_alive = fc.last_task_alive.load(std::memory_order_acquire);

        orch->tensor_map.sync_tensormap(task_id, sm_last_task_alive);

        CYCLE_COUNT_LAP(g_orch_sync_cycle);

        for (uint32_t i = 0; i < args.explicit_dep_count(); i++) {
            PTO2TaskId dep_task_id = args.explicit_dep(i);
            if (!dep_task_id.is_valid()) {
                orch->report_fatal(
                    PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "Arg.set_dependencies(...) requires valid task ids"
                );
                return result;
            }
            uint8_t dep_ring_id = dep_task_id.ring();
            PTO2SharedMemoryRingHeader &dep_ring = orch->sm_header->rings[dep_ring_id];
            int32_t dep_local_task_id = static_cast<int32_t>(dep_task_id.local());
            int32_t dep_last_task_alive = dep_ring.fc.last_task_alive.load(std::memory_order_acquire);
            if (dep_local_task_id < dep_last_task_alive) {
                continue;
            }
            int32_t dep_slot = dep_ring.get_slot_by_task_id(dep_local_task_id);
            PTO2TaskSlotState *producer_slot_state = &dep_ring.get_slot_state_by_slot(dep_slot);
            if (!append_fanin_or_fail(orch, dep_ring_id, dep_slot, producer_slot_state, &fanin_builder, ring_id)) {
                return result;
            }
        }

        // === STEP 3: Lookup inputs (creator retention + tensormap modifier lookup) ===
        DepInputs dep_inputs{
            args.tensor_count(), args.tensor_data(), args.tag_data(),
            static_cast<int32_t>(args.explicit_dep_count()), args.explicit_deps_data(),
        };

        auto runtime_emit = [&](PTO2TaskId producer_task_id) -> bool {
            uint8_t prod_ring = producer_task_id.ring();
            PTO2SharedMemoryRingHeader &producer_ring = orch->sm_header->rings[prod_ring];
            int32_t prod_slot = producer_ring.get_slot_by_task_id(static_cast<int32_t>(producer_task_id.local()));
            PTO2TaskSlotState *prod_state = &producer_ring.get_slot_state_by_slot(prod_slot);
            return append_fanin_or_fail(orch, prod_ring, prod_slot, prod_state, &fanin_builder, ring_id);
        };

        if (!compute_task_fanin(dep_inputs, orch->tensor_map, orch->in_manual_scope(), runtime_emit)) {
            return result;
        }

        CYCLE_COUNT_LAP(g_orch_lookup_cycle);

        // === STEP 4: Register outputs/inouts in TensorMap (must be separate from lookup) ===
        if (layout.needs_tensormap_registration) {
            register_task_outputs(dep_inputs, task_id, orch->tensor_map, orch->in_manual_scope());
        }

        CYCLE_COUNT_LAP(g_orch_insert_cycle);

        // === STEP 5: Batch-write to GM (single cache line burst) ===
        // Deferred from allocation phase to avoid scattered GM writes that get
        // evicted by TensorMap lookup/insert cache pressure.
        __builtin_prefetch(&task, 1, 1);
        write_submit_descriptor(
            task, task_id, prepared.alloc_result.packed_base, prepared.alloc_result.packed_end, aic_kernel_id,
            aiv0_kernel_id, aiv1_kernel_id
        );

        // Increment fanout_count on each producer (no lock — only orch writes this field).
        // Prevents premature CONSUMED: scope_end's release_producer checks fanout_refcount == fanout_count.
        for_each_fanin_storage(
            fanin_builder.inline_slots, fanin_builder.count, fanin_builder.spill_start, fanin_builder.spill_pool,
            [](PTO2TaskSlotState *producer) {
                producer->fanout_count++;
            }
        );

        int32_t inline_count = std::min(fanin_builder.count, PTO2_FANIN_INLINE_CAP);
        // Store fanin metadata in payload for scheduler to iterate.
        payload.fanin_actual_count = fanin_builder.count;
        payload.fanin_spill_start = fanin_builder.spill_start;
        payload.fanin_spill_pool = &fanin_builder.spill_pool;
        for (int i = 0; i < inline_count; i++) {
            payload.fanin_inline_slot_states[i] = fanin_builder.inline_slots[i];
        }
    }

    payload.init(args, result, prepared.alloc_result, layout);
#if PTO2_PROFILING
    if (is_dump_args_enabled()) {
        if (args.scalar_count() > 0) {
            set_dump_args_task_scalar_dtypes(
                task_id.raw, static_cast<uint32_t>(args.scalar_count()), args.scalar_dtypes()
            );
        }
        // Selective vs full dump is latched at dump_args_init from DumpDataHeader
        // (host-decided before any dispatch), so it is race-free regardless of
        // submission order. Here we only record each marked task's arg mask and
        // metadata flags, which selective collection consults.
        if (args.dump_arg_mask() != 0) {
            set_dump_args_task_mask(task_id.raw, args.dump_arg_mask(), args.dump_arg_index_ambiguous_mask());
        }
    }
#endif

    CYCLE_COUNT_LAP(g_orch_args_cycle);
#if PTO2_ORCH_PROFILING
    g_orch_args_atomic_count += 2;  // fanout_lock.store + fanout_count.store
#endif

    if (defer_dependencies) {
        PTO2DeferredSubmitRecord *record = nullptr;
        PTO2SubmitCommitRecord commit_record{};
        if (orch->submit_pipeline_compact_deferred_records) {
            record = append_deferred_submit_task_batch_record_slot(orch);
        } else {
            commit_record.kind = PTO2SubmitPipelineRecordKind::TASK_DEFERRED;
            commit_record.payload = &payload;
            commit_record.slot_state = &cur_slot_state;
            commit_record.scheduler = sched;
            commit_record.task = &task;
            commit_record.alloc_result = prepared.alloc_result;
            commit_record.task_id = task_id;
            commit_record.needs_tensormap_registration = layout.needs_tensormap_registration;
            record = nullptr;
        }
        PTO2DeferredSubmitRecord deferred_record{};
        PTO2DeferredSubmitRecord &target = record != nullptr ? *record : deferred_record;
        target.payload = &payload;
        target.slot_state = &cur_slot_state;
        target.scheduler = sched;
        target.packed_buffer_base = prepared.alloc_result.packed_base;
        target.packed_buffer_end = prepared.alloc_result.packed_end;
        target.task_id = task_id;
        target.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIC)] = aic_kernel_id;
        target.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV0)] = aiv0_kernel_id;
        target.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV1)] = aiv1_kernel_id;
        target.explicit_dep_count = static_cast<int32_t>(args.explicit_dep_count());
        target.in_manual_scope = orch->in_manual_scope();
        target.needs_tensormap_registration = layout.needs_tensormap_registration;
        for (int32_t i = 0; i < target.explicit_dep_count; i++) {
            target.explicit_deps[i] = args.explicit_dep(static_cast<uint32_t>(i));
        }
        if (!orch->submit_pipeline_compact_deferred_records) {
            commit_record.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIC)] = aic_kernel_id;
            commit_record.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV0)] = aiv0_kernel_id;
            commit_record.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV1)] = aiv1_kernel_id;
            commit_record.explicit_dep_count = target.explicit_dep_count;
            commit_record.in_manual_scope = target.in_manual_scope;
            for (int32_t i = 0; i < target.explicit_dep_count; i++) {
                commit_record.explicit_deps[i] = target.explicit_deps[i];
            }
            append_submit_task_batch_record(orch, commit_record);
        }
    } else {
        // === STEP 6: push to wiring queue ===
        // Deferred wiring: orchestrator only stores dependency metadata and increments
        // fanout_count. The actual fanout_head wiring (lock + dep_pool + early_finished)
        // is handled asynchronously by scheduler thread 0 via the wiring queue.
        // Push to global wiring queue — scheduler sets fanin_count, wires fanout, checks readiness.
        while (!sched->wiring.queue.push(&cur_slot_state)) {
            SPIN_WAIT_HINT();
        }
    }

    CYCLE_COUNT_LAP(g_orch_fanin_cycle);
    CYCLE_COUNT_ORCH_SUBMIT_RECORD(task_id.raw);

#if PTO2_PROFILING
    orch->tasks_submitted++;
#if PTO2_ORCH_PROFILING
    g_orch_submit_count++;
#endif
    g_orch_submit_idx++;
#endif
    return result;
}

TaskOutputTensors PTO2OrchestratorState::submit_task(const MixedKernels &mixed_kernels, const L0TaskArgs &args) {
    auto *orch = this;

    // Orchestration API should short-circuit after fatal, but keep this entry
    // robust as a no-op in case a caller reaches it directly.
    if (orch->fatal) {
        return TaskOutputTensors{};
    }

    // Validate Arg construction (errors recorded by add_input/add_output/etc.)
    if (args.has_error) {
        LOG_ERROR("========================================");
        LOG_ERROR("FATAL: Invalid Arg Detected!");
        LOG_ERROR("========================================");
        LOG_ERROR("Error: %s", args.error_msg ? args.error_msg : "(unknown)");
        LOG_ERROR("  tensor_count: %d, scalar_count: %d", args.tensor_count(), args.scalar_count());
        LOG_ERROR("This is a bug in the orchestration code.");
        LOG_ERROR("========================================");
        orch_mark_fatal(orch, PTO2_ERROR_INVALID_ARGS);
        return TaskOutputTensors{};
    }
    always_assert(orch->scheduler != nullptr);
    // === Validate submit inputs ===
    ActiveMask active_mask = mixed_kernels.to_active_mask();
    always_assert(static_cast<bool>(active_mask) && "MixedKernels must have at least one active slot");

    int16_t block_num = args.launch_spec.block_num();
    always_assert(block_num >= 1 && "block_num must be >= 1");

    // Normalize single-AIV tasks: if only aiv1 is set (no aic, no aiv0), move
    // it to the aiv0 slot.  This guarantees the dispatch path can always use
    // PTO2SubtaskSlot::AIV0 for single-AIV shapes without inspecting active_mask.
    // Mixed tasks (AIC+AIV) keep their original AIV identity so the correct
    // hardware channel (AIV0→AIC vs AIV1→AIC) is used at dispatch time.
    MixedKernels normalized = mixed_kernels;
    bool has_aic = active_mask.has_mask(PTO2_SUBTASK_MASK_AIC);
    bool has_aiv0 = active_mask.has_mask(PTO2_SUBTASK_MASK_AIV0);
    bool has_aiv1 = active_mask.has_mask(PTO2_SUBTASK_MASK_AIV1);
    if (!has_aic && has_aiv1 && !has_aiv0) {
        normalized.aiv0_kernel_id = normalized.aiv1_kernel_id;
        normalized.aiv1_kernel_id = INVALID_KERNEL_ID;
        active_mask = normalized.to_active_mask();
    }

    // Encode require_sync_start into active_mask bit 3 (only meaningful for tasks with block_num > 1)
    if (block_num > 1 && args.launch_spec.require_sync_start()) {
        // Deadlock check: block_num >= total available slots of the required type.
        // For MIX/AIC: limit is total_cluster_count (one AIC per cluster).
        // For AIV:     limit is total_aiv_count.
        PTO2ResourceShape shape = active_mask.to_shape();
        int32_t limit = (shape == PTO2ResourceShape::AIV) ? orch->total_aiv_count : orch->total_cluster_count;
        if (limit > 0 && block_num > limit) {
            report_fatal(
                PTO2_ERROR_REQUIRE_SYNC_START_INVALID, __FUNCTION__,
                "require_sync_start block_num=%d > limit=%d (deadlock guaranteed)", block_num, limit
            );
            return TaskOutputTensors{};
        }
        active_mask.set_sync_start();
    }

    return submit_task_common(
        orch, args, active_mask, normalized.aic_kernel_id, normalized.aiv0_kernel_id, normalized.aiv1_kernel_id
    );
}

// Submit a dependency-only task: full dependency graph participation
// (tensormap lookup/insert, explicit_deps, manual_dep, manual_scope) but no
// AICore dispatch. Empty active_mask routes the slot to the DUMMY ready
// bucket; dispatch loop short-circuits to completion. Accepts the same Arg
// shape as submit_task; scalars are permitted but never consumed.
TaskOutputTensors PTO2OrchestratorState::submit_dummy_task(const L0TaskArgs &args) {
    auto *orch = this;

    if (orch->fatal) {
        return TaskOutputTensors{};
    }

    if (args.has_error) {
        LOG_ERROR("========================================");
        LOG_ERROR("FATAL: Invalid Arg in submit_dummy_task!");
        LOG_ERROR("========================================");
        LOG_ERROR("Error: %s", args.error_msg ? args.error_msg : "(unknown)");
        LOG_ERROR("  tensor_count: %d, scalar_count: %d", args.tensor_count(), args.scalar_count());
        LOG_ERROR("========================================");
        orch_mark_fatal(orch, PTO2_ERROR_INVALID_ARGS);
        return TaskOutputTensors{};
    }
    always_assert(orch->scheduler != nullptr);

    return submit_task_common(orch, args, ActiveMask{}, INVALID_KERNEL_ID, INVALID_KERNEL_ID, INVALID_KERNEL_ID);
}

TaskOutputTensors PTO2OrchestratorState::alloc_tensors(const L0TaskArgs &args) {
    auto *orch = this;
    // Orchestration API should short-circuit after fatal, but keep this entry
    // robust as a no-op in case a caller reaches it directly.
    if (orch->fatal) {
        return TaskOutputTensors{};
    }

    if (args.tensor_count() <= 0) {
        report_fatal(PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "alloc_tensors requires at least one TensorCreateInfo");
        return TaskOutputTensors{};
    }
    if (args.scalar_count() != 0) {
        report_fatal(PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "alloc_tensors only accepts output TensorCreateInfo args");
        return TaskOutputTensors{};
    }
    for (int32_t i = 0; i < args.tensor_count(); i++) {
        if (args.tag(i) != TensorArgType::OUTPUT) {
            report_fatal(
                PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "alloc_tensors only accepts output TensorCreateInfo args"
            );
            return TaskOutputTensors{};
        }
    }

    CYCLE_COUNT_START();

    if (args.has_error) {
        report_fatal(
            PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "%s",
            args.error_msg ? args.error_msg : "alloc_tensors failed to construct output-only Arg"
        );
        return TaskOutputTensors{};
    }

    PTO2OutputLayout layout = calculate_output_layout(args);
    PTO2PreparedTask prepared;
    if (!prepare_task(orch, args, layout.total_output_size, ActiveMask{}, &prepared)) {
        return TaskOutputTensors{};
    }

    PTO2TaskDescriptor &task = *prepared.task;
    PTO2TaskPayload &payload = *prepared.payload;

    CYCLE_COUNT_LAP(g_orch_alloc_cycle);

#if PTO2_PROFILING
    if (layout.total_output_size > 0) {
        orch->buffers_allocated++;
        orch->bytes_allocated += layout.total_output_size;
    }
#endif

    task.task_id = prepared.task_id;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIC)] = INVALID_KERNEL_ID;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV0)] = INVALID_KERNEL_ID;
    task.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV1)] = INVALID_KERNEL_ID;
    task.packed_buffer_base = prepared.alloc_result.packed_base;
    task.packed_buffer_end = prepared.alloc_result.packed_end;

    TaskOutputTensors outputs;
    outputs.set_task_id(prepared.task_id);
    payload.init(args, outputs, prepared.alloc_result, layout);
    payload.fanin_actual_count = 0;
    payload.fanin_spill_start = 0;
    payload.fanin_spill_pool = &orch->rings[prepared.task_id.ring()].fanin_pool;
    CYCLE_COUNT_LAP(g_orch_args_cycle);

    if (prepared.slot_state != nullptr) {
        // Hidden alloc tasks complete inline in the orchestrator before any
        // consumer can exist, so they have no fanout to notify and no worker
        // subtasks to retire. Running the full on_task_complete path
        // would only pay unnecessary fanout_lock / traversal overhead here.
        // The generic slot initialization done in prepare_task() is still
        // required so scope_end can release the producer-side reference and
        // drive the slot to CONSUMED, but worker dispatch fields are never
        // observed for hidden alloc tasks.
        prepared.slot_state->task_state.store(PTO2_TASK_COMPLETED, std::memory_order_release);
    }
    orch->inline_completed_tasks++;

    CYCLE_COUNT_LAP(g_orch_fanin_cycle);
    CYCLE_COUNT_ORCH_SUBMIT_RECORD(prepared.task_id.raw);

#if PTO2_PROFILING
    orch->tasks_submitted++;
#if PTO2_ORCH_PROFILING
    g_orch_submit_count++;
#endif
    g_orch_submit_idx++;
#endif

    return outputs;
}

// =============================================================================
// Flow Control
// =============================================================================

void PTO2OrchestratorState::mark_done() {
    auto *orch = this;
    orch->flush_submit_pipeline();
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        int32_t total_tasks = orch->rings[r].task_allocator.active_count();
        if (total_tasks > 0) {
            LOG_INFO_V0("=== [Orchestrator] ring %d: total_tasks=%d ===", r, total_tasks);
        }
        auto &fanin_pool = orch->rings[r].fanin_pool;
        if (fanin_pool.top > 1) {
            LOG_INFO_V0(
                "=== [FaninPool %d] top=%d tail=%d used=%d high_water=%d capacity=%d ===", r, fanin_pool.top,
                fanin_pool.tail, fanin_pool.top - fanin_pool.tail, fanin_pool.high_water, fanin_pool.capacity
            );
        }
    }
    orch->sm_header->orchestrator_done.store(1, std::memory_order_release);
    orch->scope_tasks_size = 0;
    orch->scope_stack_top = -1;
    orch->manual_begin_depth = PTO2_MAX_SCOPE_DEPTH;
#if !PTO2_ORCH_PROFILING && PTO2_PROFILING
    g_orch_submit_idx = 0;
#endif
}

#if PTO2_ORCH_PROFILING
PTO2OrchProfilingData orchestrator_get_profiling() {
    PTO2OrchProfilingData d;
    d.sync_cycle = g_orch_sync_cycle;
    d.alloc_cycle = g_orch_alloc_cycle;
    d.args_cycle = g_orch_args_cycle;
    d.lookup_cycle = g_orch_lookup_cycle;
    d.insert_cycle = g_orch_insert_cycle;
    d.fanin_cycle = g_orch_fanin_cycle;
    d.scope_end_cycle = g_orch_scope_end_cycle;
    d.submit_count = g_orch_submit_count;
    d.alloc_wait_cycle = g_orch_alloc_wait_cycle;
    d.fanin_wait_cycle = g_orch_fanin_wait_cycle;
    d.alloc_atomic_count = g_orch_alloc_atomic_count;
    d.args_atomic_count = g_orch_args_atomic_count;
    d.scope_end_atomic_count = g_orch_scope_end_atomic_count;

    // Reset
    g_orch_sync_cycle = g_orch_alloc_cycle = g_orch_args_cycle = 0;
    g_orch_lookup_cycle = g_orch_insert_cycle = 0;
    g_orch_fanin_cycle = g_orch_scope_end_cycle = 0;
    g_orch_submit_count = 0;
    g_orch_submit_idx = 0;
    g_orch_alloc_wait_cycle = 0;
    g_orch_fanin_wait_cycle = 0;
    g_orch_alloc_atomic_count = 0;
    g_orch_args_atomic_count = 0;
    g_orch_scope_end_atomic_count = 0;
    return d;
}
#endif
