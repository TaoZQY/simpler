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

#include "aicpu/dep_gen_collector_aicpu.h"
#include "common/dep_gen.h"
#include "common/unified_log.h"
#include "pto_dep_compute.h"
#include "pto_runtime2_types.h"
#include "pto_shared_memory.h"
#include "pto_tensormap.h"
#include "pto_types.h"
#include "tensor.h"

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
// (same pattern as get_sys_cnt_aicpu / l2_perf_aicpu_record_orch_phase below).
extern "C" __attribute__((weak, visibility("hidden"))) bool is_dep_gen_enabled() { return false; }
__attribute__((weak, visibility("hidden"))) void
dep_gen_aicpu_record_submit(uint64_t, bool, int, const void *const *, const uint8_t *, int, const uint64_t *) {}

// =============================================================================
// Orchestrator Profiling (compile-time toggle)
// =============================================================================
#if PTO2_ORCH_PROFILING
#include "aicpu/device_time.h"
#include "aicpu/l2_perf_collector_aicpu.h"
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
// Weak fallback for builds that don't link l2_perf_collector_aicpu.cpp.
// The strong symbol from the AICPU build wins when profiling is available.
// Also hidden to prevent HOST .so from polluting the global symbol table.
__attribute__((weak, visibility("hidden"))) void
l2_perf_aicpu_record_orch_phase(AicpuPhaseId, uint64_t, uint64_t, uint32_t, uint64_t) {}
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
#define CYCLE_COUNT_START() uint64_t _t0 = get_sys_cnt_aicpu(), _t1
#define CYCLE_COUNT_LAP(acc)       \
    do {                           \
        _t1 = get_sys_cnt_aicpu(); \
        acc += (_t1 - _t0);        \
        _t0 = _t1;                 \
    } while (0)
#define CYCLE_COUNT_LAP_RECORD(acc, phase_id, tid)                                       \
    do {                                                                                 \
        _t1 = get_sys_cnt_aicpu();                                                       \
        acc += (_t1 - _t0);                                                              \
        l2_perf_aicpu_record_orch_phase((phase_id), _t0, _t1, g_orch_submit_idx, (tid)); \
        _t0 = _t1;                                                                       \
    } while (0)
#elif PTO2_PROFILING
#include "aicpu/device_time.h"
#include "aicpu/l2_perf_collector_aicpu.h"
__attribute__((weak, visibility("hidden"))) uint64_t get_sys_cnt_aicpu() { return 0; }
__attribute__((weak, visibility("hidden"))) void
l2_perf_aicpu_record_orch_phase(AicpuPhaseId, uint64_t, uint64_t, uint32_t, uint64_t) {}
// submit_idx needed for swimlane task_id tagging (no cycle accumulation at this level)
static uint32_t g_orch_submit_idx = 0;
#define CYCLE_COUNT_START()                                                \
    bool _prof_active = (orch->l2_perf_level >= L2PerfLevel::ORCH_PHASES); \
    uint64_t _t0 = _prof_active ? get_sys_cnt_aicpu() : 0, _t1 = 0
#define CYCLE_COUNT_LAP(acc) \
    do {                     \
    } while (0)
#define CYCLE_COUNT_LAP_RECORD(acc, phase_id, tid)                                           \
    do {                                                                                     \
        if (_prof_active) {                                                                  \
            _t1 = get_sys_cnt_aicpu();                                                       \
            l2_perf_aicpu_record_orch_phase((phase_id), _t0, _t1, g_orch_submit_idx, (tid)); \
            _t0 = _t1;                                                                       \
        }                                                                                    \
    } while (0)
#else
#define CYCLE_COUNT_START()
#define CYCLE_COUNT_LAP(acc)
#define CYCLE_COUNT_LAP_RECORD(acc, phase_id, tid)
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

struct PTO2FaninBuilder {
    PTO2FaninBuilder(PTO2FaninPool &spill_pool) :
        count(0),
        spill_start(0),
        spill_pool(spill_pool) {}
    int32_t count{0};
    int32_t spill_start{0};
    PTO2FaninPool &spill_pool;
    PTO2TaskSlotState *inline_slots[PTO2_FANIN_INLINE_CAP];

    template <typename Fn>
    PTO2FaninForEachReturn<Fn> for_each(Fn &&fn) const {
        return for_each_fanin_storage(inline_slots, count, spill_start, spill_pool, static_cast<Fn &&>(fn));
    }

    bool contains(PTO2TaskSlotState *prod_state) const {
        bool found = false;
        for_each([&](PTO2TaskSlotState *slot_state) {
            if (slot_state == prod_state) {
                found = true;
                return false;
            }
            return true;
        });
        if (found) {
            return true;
        }
        return false;
    }
};

static bool append_fanin_or_fail(
    PTO2OrchestratorState *orch, PTO2TaskSlotState *prod_state, PTO2FaninBuilder *fanin_builder, uint8_t ring_id
) {
    if (fanin_builder->contains(prod_state)) {
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

static void scope_tasks_push(PTO2OrchestratorState *orch, PTO2TaskSlotState *task_slot_state);

struct PTO2PreparedTask {
    PTO2TaskId task_id = PTO2TaskId::invalid();
    PTO2TaskAllocResult alloc_result = {-1, 0, nullptr, nullptr};
    PTO2TaskDescriptor *task = nullptr;
    PTO2TaskPayload *payload = nullptr;
    PTO2TaskSlotState *slot_state = nullptr;
};

static PTO2OutputLayout calculate_output_layout(const Arg &args) {
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
            PTO2_ALIGN_UP(args.tensor(i).create_info->buffer_size_bytes(), PTO2_PACKED_OUTPUT_ALIGN);
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

static void prefetch_payload(PTO2TaskPayload *payload, int32_t tensor_count, int32_t scalar_count) {
    for (int32_t i = 0; i < tensor_count; i++) {
        __builtin_prefetch(&payload->tensors[i], 1, 3);
        __builtin_prefetch(reinterpret_cast<char *>(&payload->tensors[i]) + 64, 1, 3);
    }
    for (int32_t i = 0; i < scalar_count; i += 8) {
        __builtin_prefetch(&payload->scalars[i], 1, 3);
    }
    __builtin_prefetch(payload, 1, 3);
    __builtin_prefetch(reinterpret_cast<char *>(payload) + 64, 1, 3);
    __builtin_prefetch(reinterpret_cast<char *>(payload) + 128, 1, 3);
}

static void flush_submit_pipeline_if_heap_pressure(
    PTO2OrchestratorState *orch, PTO2TaskAllocator &allocator, int32_t total_output_size
) {
    if (!orch->submit_pipeline_defer_dependencies || total_output_size <= 0) {
        return;
    }
    if (allocator.allocation_would_block_on_heap(total_output_size)) {
        orch->flush_submit_pipeline();
    }
}

static bool prepare_task(
    PTO2OrchestratorState *orch, const Arg &args, int32_t total_output_size, ActiveMask active_mask,
    PTO2PreparedTask *out
) {
#if PTO2_PROFILING
    uint64_t prepare_cursor = get_sys_cnt_aicpu();
#endif
    uint8_t ring_id = orch->current_ring_id();
    auto &allocator = orch->rings[ring_id].task_allocator;

    if (!check_scope_can_accept_task(orch, allocator, ring_id)) {
#if PTO2_PROFILING
        uint64_t prepare_after_check = get_sys_cnt_aicpu();
        orch->orch_prepare_scope_check_cycles += prepare_after_check - prepare_cursor;
#endif
        return false;
    }
#if PTO2_PROFILING
    uint64_t prepare_after_check = get_sys_cnt_aicpu();
    orch->orch_prepare_scope_check_cycles += prepare_after_check - prepare_cursor;
    prepare_cursor = prepare_after_check;
#endif

    flush_submit_pipeline_if_heap_pressure(orch, allocator, total_output_size);
#if PTO2_PROFILING
    PTO2TaskAllocProfile alloc_profile{};
    out->alloc_result = allocator.alloc(total_output_size, &alloc_profile);
    orch->orch_alloc_task_wait_cycles += alloc_profile.task_wait_cycles;
    orch->orch_alloc_heap_wait_cycles += alloc_profile.heap_wait_cycles;
    orch->orch_alloc_task_wait_spins += alloc_profile.task_wait_spins;
    orch->orch_alloc_heap_wait_spins += alloc_profile.heap_wait_spins;
    orch->orch_alloc_progress_count += alloc_profile.progress_count;
#else
    out->alloc_result = allocator.alloc(total_output_size);
#endif
    if (out->alloc_result.failed()) {
#if PTO2_PROFILING
        uint64_t prepare_after_alloc = get_sys_cnt_aicpu();
        orch->orch_prepare_alloc_cycles += prepare_after_alloc - prepare_cursor;
#endif
        orch_mark_fatal(orch, PTO2_ERROR_HEAP_RING_DEADLOCK);
        return false;
    }
#if PTO2_PROFILING
    uint64_t prepare_after_alloc = get_sys_cnt_aicpu();
    orch->orch_prepare_alloc_cycles += prepare_after_alloc - prepare_cursor;
    prepare_cursor = prepare_after_alloc;
#endif

    out->task_id = PTO2TaskId::make(ring_id, static_cast<uint32_t>(out->alloc_result.task_id));
    out->slot_state = &orch->sm_header->rings[ring_id].get_slot_state_by_slot(out->alloc_result.slot);
    out->task = &orch->sm_header->rings[ring_id].task_descriptors[out->alloc_result.slot];
    out->payload = &orch->sm_header->rings[ring_id].task_payloads[out->alloc_result.slot];
#if PTO2_PROFILING
    uint64_t prepare_after_ptr = get_sys_cnt_aicpu();
    orch->orch_prepare_ptr_cycles += prepare_after_ptr - prepare_cursor;
    prepare_cursor = prepare_after_ptr;
#endif

    prefetch_payload(out->payload, args.tensor_count(), args.scalar_count());
#if PTO2_PROFILING
    uint64_t prepare_after_prefetch = get_sys_cnt_aicpu();
    orch->orch_prepare_prefetch_cycles += prepare_after_prefetch - prepare_cursor;
    prepare_cursor = prepare_after_prefetch;
#endif

    // Fields already reset by advance_ring_pointers (eager reset after CONSUMED):
    //   fanout_lock=0, fanout_count=1, fanout_head=nullptr,
    //   fanin_refcount=0, fanout_refcount=0, completed_subtasks=0, next_block_idx=0
    // Fields immutable after RingSchedState::init():
    //   payload, task, ring_id
    // task_state left as CONSUMED by eager reset (safe for stale wait_for_tensor
    // observers); set to PENDING here when orchestrator actually reuses the slot.
    out->slot_state->task_state.store(PTO2_TASK_PENDING, std::memory_order_relaxed);
    int16_t block_num = args.launch_spec.block_num();
    out->slot_state->total_required_subtasks =
        static_cast<int16_t>(block_num * __builtin_popcount(active_mask.core_mask()));
    out->slot_state->logical_block_num = block_num;
    out->slot_state->active_mask = active_mask;
    // fanin_count is set by scheduler during wiring
#if PTO2_PROFILING
    uint64_t prepare_after_slot_state = get_sys_cnt_aicpu();
    orch->orch_prepare_slot_state_cycles += prepare_after_slot_state - prepare_cursor;
    prepare_cursor = prepare_after_slot_state;
#endif
    scope_tasks_push(orch, out->slot_state);
#if PTO2_PROFILING
    uint64_t prepare_after_scope_push = get_sys_cnt_aicpu();
    orch->orch_prepare_scope_push_cycles += prepare_after_scope_push - prepare_cursor;
#endif

    return true;
}

// =============================================================================
// Orchestrator Initialization
// =============================================================================

PTO2OrchestratorLayout PTO2OrchestratorState::reserve_layout(
    DeviceArena &arena, const int32_t task_window_sizes[PTO2_MAX_RING_DEPTH], int32_t dep_pool_capacity
) {
    PTO2OrchestratorLayout layout{};
    layout.dep_pool_capacity = dep_pool_capacity;
    layout.scope_tasks_cap = PTO2_SCOPE_TASKS_CAP;
    layout.scope_stack_capacity = PTO2_MAX_SCOPE_DEPTH;

    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        const size_t fanin_pool_bytes =
            PTO2_ALIGN_UP(static_cast<size_t>(dep_pool_capacity) * sizeof(PTO2FaninSpillEntry), PTO2_ALIGN_SIZE);
        layout.off_fanin_pool[r] = arena.reserve(fanin_pool_bytes, PTO2_ALIGN_SIZE);
    }
    layout.off_scope_tasks = arena.reserve(
        static_cast<size_t>(layout.scope_tasks_cap) * sizeof(PTO2TaskSlotState *), alignof(PTO2TaskSlotState *)
    );
    layout.off_scope_begins =
        arena.reserve(static_cast<size_t>(layout.scope_stack_capacity) * sizeof(int32_t), alignof(int32_t));
    layout.tensor_map = PTO2TensorMap::reserve_layout_default(arena, task_window_sizes);
    return layout;
}

bool PTO2OrchestratorState::init_from_layout(
    const PTO2OrchestratorLayout &layout, DeviceArena &arena, PTO2SharedMemoryHeader *sm_header_arg, void *gm_heap,
    uint64_t heap_size
) {
    uint64_t heap_sizes[PTO2_MAX_RING_DEPTH];
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        heap_sizes[r] = heap_size;
    }
    return init_from_layout_per_ring(layout, arena, sm_header_arg, gm_heap, heap_sizes);
}

bool PTO2OrchestratorState::init_from_layout_per_ring(
    const PTO2OrchestratorLayout &layout, DeviceArena &arena, PTO2SharedMemoryHeader *sm_header_arg, void *gm_heap,
    const uint64_t heap_sizes[PTO2_MAX_RING_DEPTH]
) {
    auto *orch = this;
    memset(orch, 0, sizeof(*orch));

    orch->sm_header = sm_header_arg;
    orch->gm_heap_base = gm_heap;
    orch->gm_heap_size = 0;
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        orch->gm_heap_size += heap_sizes[r];
    }
    orch->fatal = false;
    orch->submit_pipeline_enabled = false;
    orch->submit_pipeline_enqueue_submit_records = false;
    orch->submit_pipeline_defer_dependencies = false;
    orch->submit_pipeline_signal_scheduler_drain = false;
    orch->submit_pipeline_compact_deferred_records = false;
    orch->submit_pipeline_commit_stages = 0;
    orch->submit_pipeline_current_task_batch_slot = -1;
    orch->submit_pipeline_task_batch_count = 0;
    orch->submit_pipeline_next_task_batch_slot = 0;
    orch->submit_pipeline_stop.store(false, std::memory_order_relaxed);
    orch->submit_pipeline_completed.store(0, std::memory_order_relaxed);
    orch->submit_pipeline_control.store(PTO2_SUBMIT_PIPELINE_CONTROL_IDLE, std::memory_order_relaxed);
    for (int32_t stage = 0; stage < PTO2_SUBMIT_PIPELINE_MAX_COMMIT_STAGES; stage++) {
        orch->submit_pipeline_stage_done[stage].store(false, std::memory_order_relaxed);
        PTO2SubmitPipelineQueue &queue = orch->submit_pipeline_queues[stage];
        queue.tail.store(0, std::memory_order_relaxed);
        queue.head.store(0, std::memory_order_relaxed);
        for (int32_t i = 0; i < PTO2_SUBMIT_PIPELINE_QUEUE_CAP; i++) {
            queue.slot_state[i].store(0, std::memory_order_relaxed);
            queue.records[i] = PTO2SubmitCommitRecord{};
        }
    }
    for (int32_t i = 0; i < PTO2_SUBMIT_PIPELINE_TASK_BATCH_SLOT_CAP; i++) {
        orch->submit_pipeline_task_batch_slot_state[i].store(0, std::memory_order_relaxed);
        orch->submit_pipeline_task_batch_slots[i].count = 0;
    }

    uint64_t ring_heap_offset = 0;
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        void *ring_heap_base = reinterpret_cast<char *>(gm_heap) + ring_heap_offset;
        auto &ring = sm_header_arg->rings[r];
        uint64_t ring_heap_size = heap_sizes[r];
        ring_heap_offset += ring_heap_size;

        orch->rings[r].task_allocator.init(
            ring.task_descriptors, ring.task_window_size, &ring.fc.current_task_index, &ring.fc.last_task_alive,
            ring_heap_base, ring_heap_size, &sm_header_arg->orch_error_code
        );

        const size_t fanin_pool_bytes =
            PTO2_ALIGN_UP(static_cast<size_t>(layout.dep_pool_capacity) * sizeof(PTO2FaninSpillEntry), PTO2_ALIGN_SIZE);
        auto *fanin_entries = static_cast<PTO2FaninSpillEntry *>(arena.region_ptr(layout.off_fanin_pool[r]));
        // aligned_zalloc-equivalent: pool relies on zeroed entries.
        memset(fanin_entries, 0, fanin_pool_bytes);
        orch->rings[r].fanin_pool.init(fanin_entries, layout.dep_pool_capacity, &sm_header_arg->orch_error_code);
    }

    if (!orch->tensor_map.init_from_layout(layout.tensor_map, arena)) {
        return false;
    }
    orch->tensor_map.orch = orch;

    orch->scope_tasks = static_cast<PTO2TaskSlotState **>(arena.region_ptr(layout.off_scope_tasks));
    orch->scope_begins = static_cast<int32_t *>(arena.region_ptr(layout.off_scope_begins));
    orch->scope_tasks_size = 0;
    orch->scope_tasks_capacity = layout.scope_tasks_cap;
    orch->scope_stack_top = -1;
    orch->scope_stack_capacity = layout.scope_stack_capacity;
    orch->manual_begin_depth = PTO2_MAX_SCOPE_DEPTH;

    return true;
}

void PTO2OrchestratorState::destroy() {
    auto *orch = this;
    stop_submit_pipeline();
    orch->tensor_map.destroy();
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        orch->rings[r].fanin_pool.base = nullptr;
    }
    orch->scope_tasks = nullptr;
    orch->scope_begins = nullptr;
}

void PTO2OrchestratorState::set_scheduler(PTO2SchedulerState *scheduler) { this->scheduler = scheduler; }

// =============================================================================
// Submit Commit Pipeline
// =============================================================================

static void submit_pipeline_queue_reset(PTO2SubmitPipelineQueue &queue) {
    queue.tail.store(0, std::memory_order_release);
    queue.head.store(0, std::memory_order_release);
    for (int32_t i = 0; i < PTO2_SUBMIT_PIPELINE_QUEUE_CAP; i++) {
        queue.slot_state[i].store(0, std::memory_order_release);
        queue.records[i] = PTO2SubmitCommitRecord{};
    }
}

static void submit_pipeline_queue_push(PTO2SubmitPipelineQueue &queue, PTO2SubmitCommitRecord &record) {
    uint64_t seq = queue.tail.load(std::memory_order_acquire);
    int32_t slot = static_cast<int32_t>(seq % PTO2_SUBMIT_PIPELINE_QUEUE_CAP);
    while (queue.slot_state[slot].load(std::memory_order_acquire) != 0) {
        SPIN_WAIT_HINT();
    }
    queue.records[slot] = record;
    queue.slot_state[slot].store(1, std::memory_order_release);
    queue.tail.store(seq + 1, std::memory_order_release);
}

static void submit_pipeline_queue_push_task_batch(PTO2SubmitPipelineQueue &queue, int32_t task_batch_slot, int32_t count) {
    uint64_t seq = queue.tail.load(std::memory_order_acquire);
    int32_t slot = static_cast<int32_t>(seq % PTO2_SUBMIT_PIPELINE_QUEUE_CAP);
    while (queue.slot_state[slot].load(std::memory_order_acquire) != 0) {
        SPIN_WAIT_HINT();
    }
    PTO2SubmitCommitRecord &record = queue.records[slot];
    record.kind = PTO2SubmitPipelineRecordKind::TASK_BATCH;
    record.task_batch_slot = task_batch_slot;
    record.task_batch_count = count;
    queue.slot_state[slot].store(1, std::memory_order_release);
    queue.tail.store(seq + 1, std::memory_order_release);
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
    PTO2SubmitCommitRecord &slot_record = queue.records[slot];
    if (slot_record.kind == PTO2SubmitPipelineRecordKind::TASK_BATCH) {
        record->kind = PTO2SubmitPipelineRecordKind::TASK_BATCH;
        record->task_batch_slot = slot_record.task_batch_slot;
        record->task_batch_count = slot_record.task_batch_count;
    } else {
        *record = slot_record;
    }
    queue.slot_state[slot].store(0, std::memory_order_release);
    queue.head.store(seq + 1, std::memory_order_release);
    return true;
}

static bool submit_pipeline_queue_empty(PTO2SubmitPipelineQueue &queue) {
    return queue.head.load(std::memory_order_acquire) >= queue.tail.load(std::memory_order_acquire);
}

static void mark_submit_pipeline_work_available(PTO2OrchestratorState *orch) {
    orch->submit_pipeline_work_available.store(true, std::memory_order_release);
    orch->submit_pipeline_control.store(PTO2_SUBMIT_PIPELINE_CONTROL_WORK, std::memory_order_release);
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
            int32_t slot = (orch->submit_pipeline_next_task_batch_slot + probe) %
                           PTO2_SUBMIT_PIPELINE_TASK_BATCH_SLOT_CAP;
            if (orch->submit_pipeline_task_batch_slot_state[slot].load(std::memory_order_acquire) == 0) {
                orch->submit_pipeline_task_batch_slot_state[slot].store(1, std::memory_order_release);
                reset_submit_task_batch_slot(orch->submit_pipeline_task_batch_slots[slot]);
                orch->submit_pipeline_next_task_batch_slot =
                    (slot + 1) % PTO2_SUBMIT_PIPELINE_TASK_BATCH_SLOT_CAP;
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
    PTO2SubmitCommitRecord *record = &batch.records[index];
    return record;
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

static void append_submit_task_batch_record(PTO2OrchestratorState *orch, PTO2SubmitCommitRecord &record) {
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
        if (record.kind != PTO2SubmitPipelineRecordKind::TASK_DEFERRED ||
            slot_state != task_slot_states[i]) {
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
        scope_task_batch_begin = find_scope_tail_task_segment(
            batch, task_slot_states, count, orch->submit_pipeline_compact_deferred_records
        );
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

#if PTO2_PROFILING
static void record_orch_four_stage(
    PTO2OrchestratorState *orch, PTO2OrchFourStage stage, uint64_t start, uint64_t end
) {
    int32_t stage_idx = static_cast<int32_t>(stage);
    orch->orch_four_stage_cycles[stage_idx] += end - start;
    orch->orch_four_stage_count[stage_idx]++;
}
#endif

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

static void write_submit_descriptor(
    PTO2TaskDescriptor &task, PTO2TaskId task_id, const PTO2TaskAllocResult &alloc_result, int32_t aic_kernel_id,
    int32_t aiv0_kernel_id, int32_t aiv1_kernel_id
) {
    write_submit_descriptor(
        task, task_id, alloc_result.packed_base, alloc_result.packed_end, aic_kernel_id, aiv0_kernel_id,
        aiv1_kernel_id
    );
}

static void commit_submit_descriptor(PTO2SubmitCommitRecord &record) {
    write_submit_descriptor(
        *record.task, record.task_id, record.alloc_result,
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

static void commit_deferred_descriptor(PTO2SubmitCommitRecord &) {}

static void commit_deferred_descriptor(PTO2DeferredSubmitRecord &record) {
    write_submit_descriptor(
        *record.slot_state->task, record.task_id, record.packed_buffer_base, record.packed_buffer_end,
        record.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIC)],
        record.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV0)],
        record.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV1)]
    );
}

#if PTO2_PROFILING
struct PTO2DeferredDepProfile {
    uint64_t explicit_cycles{0};
    uint64_t owner_emit_cycles{0};
    uint64_t lookup_cycles{0};
    uint64_t lookup_emit_cycles{0};
    uint64_t remove_cycles{0};
    uint64_t register_cycles{0};
    uint64_t register_insert_cycles{0};
    uint32_t explicit_count{0};
    uint32_t owner_emit_count{0};
    uint32_t lookup_count{0};
    uint32_t lookup_skip_count{0};
    uint32_t lookup_range_skip_count{0};
    uint32_t lookup_match_count{0};
    uint32_t lookup_emit_count{0};
    uint32_t remove_count{0};
    uint32_t register_count{0};
    uint32_t fanin_actual_count{0};
};

static void accumulate_deferred_dep_profile(PTO2OrchestratorState *orch, const PTO2DeferredDepProfile &profile) {
    orch->submit_pipeline_dep_explicit_cycles += profile.explicit_cycles;
    orch->submit_pipeline_dep_owner_emit_cycles += profile.owner_emit_cycles;
    orch->submit_pipeline_dep_lookup_cycles += profile.lookup_cycles;
    orch->submit_pipeline_dep_lookup_emit_cycles += profile.lookup_emit_cycles;
    orch->submit_pipeline_dep_remove_cycles += profile.remove_cycles;
    orch->submit_pipeline_dep_register_cycles += profile.register_cycles;
    orch->submit_pipeline_dep_register_insert_cycles += profile.register_insert_cycles;
    orch->submit_pipeline_dep_explicit_count += profile.explicit_count;
    orch->submit_pipeline_dep_owner_emit_count += profile.owner_emit_count;
    orch->submit_pipeline_dep_lookup_count += profile.lookup_count;
    orch->submit_pipeline_dep_lookup_skip_count += profile.lookup_skip_count;
    orch->submit_pipeline_dep_lookup_range_skip_count += profile.lookup_range_skip_count;
    orch->submit_pipeline_dep_lookup_match_count += profile.lookup_match_count;
    orch->submit_pipeline_dep_lookup_emit_count += profile.lookup_emit_count;
    orch->submit_pipeline_dep_remove_count += profile.remove_count;
    orch->submit_pipeline_dep_register_count += profile.register_count;
    orch->submit_pipeline_dep_fanin_actual_count += profile.fanin_actual_count;
}

template <typename Emit>
static bool compute_payload_task_fanin_profiled(
    const PTO2TaskPayload &payload, PTO2TensorMap &tensor_map, bool in_manual_scope, Emit emit,
    PTO2DeferredDepProfile *profile
) {
    if (in_manual_scope) {
        return true;
    }

    for (int32_t i = 0; i < payload.tensor_count; i++) {
        TensorArgType ptype = static_cast<TensorArgType>(payload.arg_tags[i]);
        if (ptype == TensorArgType::OUTPUT) {
            continue;
        }

        const Tensor *tensor = &payload.tensors[i];
        PTO2TaskId owner = tensor->owner_task_id;
        if (owner.is_valid()) {
            profile->owner_emit_count++;
            uint64_t owner_start = get_sys_cnt_aicpu();
            if (!emit(owner)) {
                profile->owner_emit_cycles += get_sys_cnt_aicpu() - owner_start;
                return false;
            }
            profile->owner_emit_cycles += get_sys_cnt_aicpu() - owner_start;
        }

        if (ptype != TensorArgType::INPUT && ptype != TensorArgType::INOUT) {
            continue;
        }
        if (tensor->manual_dep) {
            continue;
        }
        PTO2TensorMapLookupSkipReason skip_reason = tensor_map.lookup_skip_reason(*tensor);
        if (skip_reason == PTO2TensorMapLookupSkipReason::EMPTY_ADDR) {
            profile->lookup_skip_count++;
            continue;
        }
        if (skip_reason == PTO2TensorMapLookupSkipReason::RANGE) {
            profile->lookup_range_skip_count++;
            continue;
        }

        bool fatal = false;
        profile->lookup_count++;
        uint64_t lookup_start = get_sys_cnt_aicpu();
        PTO2TensorMapEntry *cached_entry = nullptr;
        OverlapStatus cached_status = OverlapStatus::NO_OVERLAP;
        if (tensor_map.lookup_exact_cached(*tensor, cached_entry, cached_status)) {
            profile->lookup_match_count++;
            profile->lookup_emit_count++;
            uint64_t emit_start = get_sys_cnt_aicpu();
            if (!emit(cached_entry->producer_task_id)) {
                profile->lookup_emit_cycles += get_sys_cnt_aicpu() - emit_start;
                profile->lookup_cycles += get_sys_cnt_aicpu() - lookup_start;
                return false;
            }
            profile->lookup_emit_cycles += get_sys_cnt_aicpu() - emit_start;

            if (ptype == TensorArgType::INOUT && cached_status == OverlapStatus::COVERED) {
                profile->remove_count++;
                uint64_t remove_start = get_sys_cnt_aicpu();
                tensor_map.remove_entry(*cached_entry);
                profile->remove_cycles += get_sys_cnt_aicpu() - remove_start;
            }
            profile->lookup_cycles += get_sys_cnt_aicpu() - lookup_start;
            continue;
        }
        tensor_map.lookup(*tensor, [&](PTO2TensorMapEntry &entry, OverlapStatus overlap_status) -> bool {
            profile->lookup_match_count++;
            profile->lookup_emit_count++;
            uint64_t emit_start = get_sys_cnt_aicpu();
            if (!emit(entry.producer_task_id)) {
                profile->lookup_emit_cycles += get_sys_cnt_aicpu() - emit_start;
                fatal = true;
                return false;
            }
            profile->lookup_emit_cycles += get_sys_cnt_aicpu() - emit_start;

            if (ptype == TensorArgType::INOUT && overlap_status == OverlapStatus::COVERED) {
                profile->remove_count++;
                uint64_t remove_start = get_sys_cnt_aicpu();
                tensor_map.remove_entry(entry);
                profile->remove_cycles += get_sys_cnt_aicpu() - remove_start;
            }
            return true;
        });
        profile->lookup_cycles += get_sys_cnt_aicpu() - lookup_start;
        if (fatal) {
            return false;
        }
    }
    return true;
}

static void register_payload_task_outputs_profiled(
    const PTO2TaskPayload &payload, PTO2TaskId task_id, PTO2TensorMap &tensor_map, bool in_manual_scope,
    PTO2DeferredDepProfile *profile
) {
    uint64_t register_start = get_sys_cnt_aicpu();
    if (in_manual_scope) {
        profile->register_cycles += get_sys_cnt_aicpu() - register_start;
        return;
    }
    for (int32_t i = 0; i < payload.tensor_count; i++) {
        TensorArgType ptype = static_cast<TensorArgType>(payload.arg_tags[i]);
        if (ptype == TensorArgType::INOUT || ptype == TensorArgType::OUTPUT_EXISTING) {
            const Tensor *tensor = &payload.tensors[i];
            if (!tensor->manual_dep) {
                profile->register_count++;
                uint64_t insert_start = get_sys_cnt_aicpu();
                tensor_map.insert(*tensor, task_id);
                profile->register_insert_cycles += get_sys_cnt_aicpu() - insert_start;
            }
        }
    }
    profile->register_cycles += get_sys_cnt_aicpu() - register_start;
}
#endif

template <typename SubmitRecord>
static bool compute_deferred_submit_dependencies(PTO2OrchestratorState *orch, SubmitRecord &record, bool sync_tensormap) {
    uint8_t ring_id = record.task_id.ring();
#if PTO2_PROFILING
    PTO2DeferredDepProfile dep_profile{};
#endif
    if (sync_tensormap) {
        PTO2RingFlowControl &fc = orch->sm_header->rings[ring_id].fc;
        int32_t sm_last_task_alive = fc.last_task_alive.load(std::memory_order_acquire);
        orch->tensor_map.sync_tensormap(record.task_id, sm_last_task_alive);
    }

    PTO2FaninBuilder fanin_builder(orch->rings[ring_id].fanin_pool);

#if PTO2_PROFILING
    dep_profile.explicit_count += static_cast<uint32_t>(record.explicit_dep_count);
    uint64_t explicit_start = get_sys_cnt_aicpu();
#endif
    for (int32_t i = 0; i < record.explicit_dep_count; i++) {
        PTO2TaskId dep_task_id = record.explicit_deps[i];
        if (!dep_task_id.is_valid()) {
#if PTO2_PROFILING
            dep_profile.explicit_cycles += get_sys_cnt_aicpu() - explicit_start;
            accumulate_deferred_dep_profile(orch, dep_profile);
#endif
            orch->report_fatal(
                PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "Arg.set_dependencies(...) requires valid task ids"
            );
            return false;
        }
        PTO2SharedMemoryRingHeader &dep_ring = orch->sm_header->rings[dep_task_id.ring()];
        int32_t dep_local_task_id = static_cast<int32_t>(dep_task_id.local());
        int32_t dep_last_task_alive = dep_ring.fc.last_task_alive.load(std::memory_order_acquire);
        if (dep_local_task_id < dep_last_task_alive) {
            continue;
        }
        PTO2TaskSlotState *producer_slot_state = &dep_ring.get_slot_state_by_task_id(dep_local_task_id);
        if (!append_fanin_or_fail(orch, producer_slot_state, &fanin_builder, ring_id)) {
#if PTO2_PROFILING
            dep_profile.explicit_cycles += get_sys_cnt_aicpu() - explicit_start;
            accumulate_deferred_dep_profile(orch, dep_profile);
#endif
            return false;
        }
    }
#if PTO2_PROFILING
    dep_profile.explicit_cycles += get_sys_cnt_aicpu() - explicit_start;
#endif

    auto runtime_emit = [&](PTO2TaskId producer_task_id) -> bool {
        PTO2TaskSlotState *prod_state =
            &orch->sm_header->rings[producer_task_id.ring()].get_slot_state_by_task_id(producer_task_id.local());
        return append_fanin_or_fail(orch, prod_state, &fanin_builder, ring_id);
    };

#if PTO2_PROFILING
    if (!compute_payload_task_fanin_profiled(
            *record.payload, orch->tensor_map, record.in_manual_scope, runtime_emit, &dep_profile
        )) {
        dep_profile.fanin_actual_count += static_cast<uint32_t>(fanin_builder.count);
        accumulate_deferred_dep_profile(orch, dep_profile);
        return false;
    }
    if (record.needs_tensormap_registration) {
        register_payload_task_outputs_profiled(
            *record.payload, record.task_id, orch->tensor_map, record.in_manual_scope, &dep_profile
        );
    }
    dep_profile.fanin_actual_count += static_cast<uint32_t>(fanin_builder.count);
    accumulate_deferred_dep_profile(orch, dep_profile);
#else
    if (!compute_payload_task_fanin(*record.payload, orch->tensor_map, record.in_manual_scope, runtime_emit)) {
        return false;
    }
    if (record.needs_tensormap_registration) {
        register_payload_task_outputs(*record.payload, record.task_id, orch->tensor_map, record.in_manual_scope);
    }
#endif

    record.fanin_spill_pool = &fanin_builder.spill_pool;
    record.fanin_actual_count = fanin_builder.count;
    record.fanin_spill_start = fanin_builder.spill_start;
    int32_t inline_count = std::min(fanin_builder.count, PTO2_FANIN_INLINE_CAP);
    for (int32_t i = 0; i < inline_count; i++) {
        record.fanin_inline_slot_states[i] = fanin_builder.inline_slots[i];
    }
    return true;
}

template <typename SubmitRecord>
static void commit_deferred_submit_record(PTO2OrchestratorState *orch, SubmitRecord &record, bool sync_tensormap = true) {
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

#if PTO2_PROFILING
static void accumulate_scope_release_detail(
    PTO2OrchestratorState *orch, const PTO2ProducerReleaseDetail &detail
) {
    if (orch == nullptr) {
        return;
    }
    orch->submit_pipeline_scope_producer_release_cycles += detail.producer_release_cycles;
    orch->submit_pipeline_scope_ring_advance_cycles += detail.ring_advance_cycles;
    orch->submit_pipeline_scope_release_count += detail.release_count;
    orch->submit_pipeline_scope_consumed_count += detail.consumed_count;
    orch->submit_pipeline_scope_advance_attempt_count += detail.advance_attempt_count;
    orch->submit_pipeline_scope_advance_success_count += detail.advance_success_count;
}
#endif

static void commit_scope_end_record(PTO2SubmitCommitRecord &record, PTO2OrchestratorState *orch = nullptr) {
    if (record.scheduler != nullptr && record.scope_task_count > 0) {
#if PTO2_PROFILING
        PTO2ProducerReleaseDetail detail{};
        record.scheduler->on_scope_end(record.scope_task_slot_states, record.scope_task_count, &detail);
        accumulate_scope_release_detail(orch, detail);
#else
        (void)orch;
        record.scheduler->on_scope_end(record.scope_task_slot_states, record.scope_task_count);
#endif
    }
}

static void commit_scope_end_record_from_task_segment(
    PTO2OrchestratorState *orch, PTO2SubmitTaskBatchSlot &batch, PTO2SubmitCommitRecord &record
) {
    if (record.scheduler == nullptr || record.scope_task_count <= 0 || record.scope_task_batch_begin < 0) {
        commit_scope_end_record(record, orch);
        return;
    }
    int32_t begin = record.scope_task_batch_begin;
    int32_t end = begin + record.scope_task_count;
    if (begin < 0 || end > batch.count) {
        commit_scope_end_record(record, orch);
        return;
    }
    for (int32_t i = 0; i < record.scope_task_count; i++) {
        PTO2SubmitCommitRecord &task_record = batch.records[begin + i];
        PTO2TaskSlotState *slot_state = orch->submit_pipeline_compact_deferred_records
                                            ? batch.deferred_records[begin + i].slot_state
                                            : task_record.slot_state;
        if (task_record.kind != PTO2SubmitPipelineRecordKind::TASK_DEFERRED ||
            slot_state == nullptr) {
            commit_scope_end_record(record, orch);
            return;
        }
    }
#if PTO2_ORCH_PROFILING
    extern uint64_t g_orch_scope_end_atomic_count;
    PTO2DeferredSubmitRecord *deferred_records = batch.deferred_records;
    if (record.scope_task_count > 0) {
        PTO2TaskSlotState *first_slot_state = orch->submit_pipeline_compact_deferred_records
                                                  ? deferred_records[begin].slot_state
                                                  : batch.records[begin].slot_state;
        __builtin_prefetch(first_slot_state, 1, 0);
    }
    for (int32_t i = begin; i < end; i++) {
        if (i + 1 < end) {
            PTO2TaskSlotState *next_slot_state = orch->submit_pipeline_compact_deferred_records
                                                     ? deferred_records[i + 1].slot_state
                                                     : batch.records[i + 1].slot_state;
            __builtin_prefetch(next_slot_state, 1, 0);
        }
    }
    PTO2TaskSlotState *slot_states[PTO2_SUBMIT_PIPELINE_SMALL_SCOPE_BOUNDARY_TASK_CAP];
    for (int32_t i = begin; i < end; i++) {
        slot_states[i - begin] = orch->submit_pipeline_compact_deferred_records
                                     ? deferred_records[i].slot_state
                                     : batch.records[i].slot_state;
    }
    PTO2ProducerReleaseDetail detail{};
    record.scheduler->release_producers(slot_states, record.scope_task_count, g_orch_scope_end_atomic_count, &detail);
    accumulate_scope_release_detail(orch, detail);
#else
    (void)orch;
    PTO2DeferredSubmitRecord *deferred_records = batch.deferred_records;
    if (record.scope_task_count > 0) {
        PTO2TaskSlotState *first_slot_state = orch->submit_pipeline_compact_deferred_records
                                                  ? deferred_records[begin].slot_state
                                                  : batch.records[begin].slot_state;
        __builtin_prefetch(first_slot_state, 1, 0);
    }
    for (int32_t i = begin; i < end; i++) {
        if (i + 1 < end) {
            PTO2TaskSlotState *next_slot_state = orch->submit_pipeline_compact_deferred_records
                                                     ? deferred_records[i + 1].slot_state
                                                     : batch.records[i + 1].slot_state;
            __builtin_prefetch(next_slot_state, 1, 0);
        }
    }
    PTO2TaskSlotState *slot_states[PTO2_SUBMIT_PIPELINE_SMALL_SCOPE_BOUNDARY_TASK_CAP];
    for (int32_t i = begin; i < end; i++) {
        slot_states[i - begin] = orch->submit_pipeline_compact_deferred_records
                                     ? deferred_records[i].slot_state
                                     : batch.records[i].slot_state;
    }
    record.scheduler->release_producers(slot_states, record.scope_task_count);
#endif
}

static void commit_submit_pipeline_task_record(
    PTO2OrchestratorState *orch, PTO2SubmitCommitRecord &record, bool sync_tensormap = true
) {
    if (record.kind == PTO2SubmitPipelineRecordKind::SCOPE_END) {
        commit_scope_end_record(record, orch);
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
                if (orch->tensor_map.get_task_local_id_slot(ring_id, task_id.local()) == cleanup_slot) {
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
#if PTO2_PROFILING
        PTO2ProducerReleaseDetail detail{};
        batch.scope_scheduler->on_scope_end(batch.scope_task_slot_states, batch.scope_task_count, &detail);
        accumulate_scope_release_detail(orch, detail);
#else
        batch.scope_scheduler->on_scope_end(batch.scope_task_slot_states, batch.scope_task_count);
#endif
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
    int32_t orchestrator_threads, bool enqueue_submit_records, bool defer_submit_dependencies,
    bool signal_scheduler_drain, bool compact_deferred_records
) {
    submit_pipeline_commit_stages = 0;
    if (orchestrator_threads >= 4) {
        submit_pipeline_commit_stages = 3;
    } else if (orchestrator_threads >= 2) {
        submit_pipeline_commit_stages = 1;
    }
    submit_pipeline_enabled = submit_pipeline_commit_stages > 0;
    submit_pipeline_enqueue_submit_records = submit_pipeline_enabled && enqueue_submit_records;
    submit_pipeline_defer_dependencies =
        submit_pipeline_enabled && submit_pipeline_commit_stages == 1 && defer_submit_dependencies;
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
    submit_pipeline_dep_explicit_cycles = 0;
    submit_pipeline_dep_owner_emit_cycles = 0;
    submit_pipeline_dep_lookup_cycles = 0;
    submit_pipeline_dep_lookup_emit_cycles = 0;
    submit_pipeline_dep_remove_cycles = 0;
    submit_pipeline_dep_register_cycles = 0;
    submit_pipeline_dep_register_insert_cycles = 0;
    submit_pipeline_publish_cycles = 0;
    submit_pipeline_scope_release_cycles = 0;
    submit_pipeline_scope_record_cycles = 0;
    submit_pipeline_scope_producer_release_cycles = 0;
    submit_pipeline_scope_ring_advance_cycles = 0;
    submit_pipeline_publish_spins = 0;
    submit_pipeline_scheduler_drain_hint_count = 0;
    submit_pipeline_task_enqueue_count = 0;
    submit_pipeline_task_enqueue_batch_count = 0;
    submit_pipeline_scope_enqueue_count = 0;
    submit_pipeline_flush_count = 0;
    submit_pipeline_deferred_commit_count = 0;
    submit_pipeline_scope_record_count = 0;
    submit_pipeline_scope_release_count = 0;
    submit_pipeline_scope_consumed_count = 0;
    submit_pipeline_scope_advance_attempt_count = 0;
    submit_pipeline_scope_advance_success_count = 0;
    submit_pipeline_dep_explicit_count = 0;
    submit_pipeline_dep_owner_emit_count = 0;
    submit_pipeline_dep_lookup_count = 0;
    submit_pipeline_dep_lookup_skip_count = 0;
    submit_pipeline_dep_lookup_range_skip_count = 0;
    submit_pipeline_dep_lookup_match_count = 0;
    submit_pipeline_dep_lookup_emit_count = 0;
    submit_pipeline_dep_remove_count = 0;
    submit_pipeline_dep_register_count = 0;
    submit_pipeline_dep_fanin_actual_count = 0;
    submit_pipeline_compact_deferred_count = 0;
    for (int32_t i = 0; i < PTO2_ORCH_FOUR_STAGE_COUNT; i++) {
        orch_four_stage_cycles[i] = 0;
        orch_four_stage_count[i] = 0;
    }
    orch_submit_wrapper_cycles = 0;
    orch_alloc_tensors_cycles = 0;
    orch_scope_begin_cycles = 0;
    orch_scope_end_cycles = 0;
    orch_submit_layout_cycles = 0;
    orch_submit_prepare_cycles = 0;
    orch_submit_depgen_cycles = 0;
    orch_submit_sync_cycles = 0;
    orch_submit_explicit_dep_cycles = 0;
    orch_submit_lookup_cycles = 0;
    orch_submit_register_cycles = 0;
    orch_submit_payload_cycles = 0;
    orch_submit_descriptor_cycles = 0;
    orch_submit_deferred_meta_cycles = 0;
    orch_submit_enqueue_cycles = 0;
    orch_submit_return_cycles = 0;
    orch_prepare_scope_check_cycles = 0;
    orch_prepare_alloc_cycles = 0;
    orch_prepare_ptr_cycles = 0;
    orch_prepare_prefetch_cycles = 0;
    orch_prepare_slot_state_cycles = 0;
    orch_prepare_scope_push_cycles = 0;
    orch_alloc_task_wait_cycles = 0;
    orch_alloc_heap_wait_cycles = 0;
    orch_alloc_task_wait_spins = 0;
    orch_alloc_heap_wait_spins = 0;
    orch_alloc_progress_count = 0;
    orch_submit_wrapper_count = 0;
    orch_alloc_tensors_count = 0;
    orch_scope_begin_count = 0;
    orch_scope_end_count = 0;
    orch_submit_detail_count = 0;
    orch_submit_deferred_count = 0;
    orch_submit_sync_count = 0;
    orch_submit_heap_guard_sync_count = 0;
    orch_submit_tensor_count_total = 0;
    orch_submit_scalar_count_total = 0;
    orch_submit_explicit_dep_count_total = 0;
    orch_submit_output_bytes_total = 0;
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
    uint64_t active_cycles = 0;
    if (!submit_pipeline_enabled || stage_idx < 1 || stage_idx > submit_pipeline_commit_stages) {
        while (!submit_pipeline_stop.load(std::memory_order_acquire)) {
            SPIN_WAIT_HINT();
        }
        return active_cycles;
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
            return active_cycles;
        }
        while (!submit_pipeline_work_available.load(std::memory_order_acquire)) {
            if (submit_pipeline_stop.load(std::memory_order_acquire) ||
                submit_pipeline_control.load(std::memory_order_acquire) == PTO2_SUBMIT_PIPELINE_CONTROL_STOP) {
                submit_pipeline_stage_done[commit_stage].store(true, std::memory_order_release);
                return active_cycles;
            }
            SPIN_WAIT_HINT();
        }
    }
    while (true) {
        PTO2SubmitCommitRecord record{};
        if (!submit_pipeline_queue_pop(input_queue, &record)) {
            bool upstream_done = commit_stage == 0
                                     ? submit_pipeline_stop.load(std::memory_order_acquire)
                                     : submit_pipeline_stage_done[commit_stage - 1].load(std::memory_order_acquire);
            if (upstream_done && submit_pipeline_queue_empty(input_queue)) {
                submit_pipeline_stage_done[commit_stage].store(true, std::memory_order_release);
                return active_cycles;
            }
            SPIN_WAIT_HINT();
            continue;
        }

#if PTO2_PROFILING
        uint64_t record_start = get_sys_cnt_aicpu();
#endif
        if (submit_pipeline_commit_stages == 1) {
            if (record.kind == PTO2SubmitPipelineRecordKind::SCOPE_END) {
#if PTO2_PROFILING
                uint64_t scope_start = get_sys_cnt_aicpu();
#endif
                commit_scope_end_record(record, this);
#if PTO2_PROFILING
                uint64_t scope_cycles = get_sys_cnt_aicpu() - scope_start;
                submit_pipeline_scope_release_cycles += scope_cycles;
                submit_pipeline_scope_record_cycles += scope_cycles;
                submit_pipeline_scope_record_count++;
#endif
            } else if (record.kind == PTO2SubmitPipelineRecordKind::TASK_BATCH) {
                commit_submit_task_batch_record(this, record);
            } else if (record.kind == PTO2SubmitPipelineRecordKind::TASK_DEFERRED) {
                commit_deferred_submit_record(this, record);
            } else {
                commit_submit_record(record);
            }
            submit_pipeline_completed.fetch_add(1, std::memory_order_acq_rel);
#if PTO2_PROFILING
            active_cycles += get_sys_cnt_aicpu() - record_start;
#endif
            continue;
        }

        if (record.kind == PTO2SubmitPipelineRecordKind::SCOPE_END) {
#if PTO2_PROFILING
            uint64_t scope_start = get_sys_cnt_aicpu();
#endif
            commit_scope_end_record(record, this);
#if PTO2_PROFILING
            uint64_t scope_cycles = get_sys_cnt_aicpu() - scope_start;
            submit_pipeline_scope_release_cycles += scope_cycles;
            submit_pipeline_scope_record_cycles += scope_cycles;
            submit_pipeline_scope_record_count++;
#endif
            submit_pipeline_completed.fetch_add(1, std::memory_order_acq_rel);
#if PTO2_PROFILING
            active_cycles += get_sys_cnt_aicpu() - record_start;
#endif
            continue;
        }

        if (commit_stage == 0) {
            commit_submit_descriptor(record);
            submit_pipeline_queue_push(submit_pipeline_queues[1], record);
        } else if (commit_stage == 1) {
            commit_submit_fanin(record);
            submit_pipeline_queue_push(submit_pipeline_queues[2], record);
        } else {
            commit_submit_publish(record);
            signal_scheduler_drain_if_needed(this, record);
            submit_pipeline_completed.fetch_add(1, std::memory_order_acq_rel);
        }
#if PTO2_PROFILING
        active_cycles += get_sys_cnt_aicpu() - record_start;
#endif
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
        "Thread %d: orch_pipeline_diag task_enq_count=%u task_enq_batches=%u task_enq_cost=%.3fus scope_enq_count=%u "
        "scope_enq_cost=%.3fus flush_count=%u flush_cost=%.3fus batch_sync_cost=%.3fus deferred_dep_cost=%.3fus "
        "deferred_fanin_cost=%.3fus publish_cost=%.3fus scope_release_cost=%.3fus scope_record_cost=%.3fus "
        "scope_producer_release_cost=%.3fus scope_ring_advance_cost=%.3fus deferred_commit_count=%u "
        "scope_record_count=%u scope_release_count=%u scope_consumed_count=%u scope_advance_attempts=%u "
        "scope_advance_success=%u drain_hints=%u compact_deferred=%u publish_spins=%" PRIu64,
        thread_idx, submit_pipeline_task_enqueue_count, submit_pipeline_task_enqueue_batch_count,
        cycles_to_us(submit_pipeline_task_enqueue_cycles), submit_pipeline_scope_enqueue_count,
        cycles_to_us(submit_pipeline_scope_enqueue_cycles),
        submit_pipeline_flush_count, cycles_to_us(submit_pipeline_flush_cycles),
        cycles_to_us(submit_pipeline_batch_sync_cycles), cycles_to_us(submit_pipeline_deferred_dep_cycles),
        cycles_to_us(submit_pipeline_deferred_fanin_cycles), cycles_to_us(submit_pipeline_publish_cycles),
        cycles_to_us(submit_pipeline_scope_release_cycles), cycles_to_us(submit_pipeline_scope_record_cycles),
        cycles_to_us(submit_pipeline_scope_producer_release_cycles),
        cycles_to_us(submit_pipeline_scope_ring_advance_cycles), submit_pipeline_deferred_commit_count,
        submit_pipeline_scope_record_count, submit_pipeline_scope_release_count, submit_pipeline_scope_consumed_count,
        submit_pipeline_scope_advance_attempt_count, submit_pipeline_scope_advance_success_count,
        submit_pipeline_scheduler_drain_hint_count, submit_pipeline_compact_deferred_count,
        static_cast<uint64_t>(submit_pipeline_publish_spins)
    );
    LOG_INFO_V9(
        "Thread %d: orch_pipeline_dep_diag dep_explicit_cost=%.3fus dep_owner_emit_cost=%.3fus "
        "dep_lookup_cost=%.3fus dep_lookup_emit_cost=%.3fus dep_remove_cost=%.3fus "
        "dep_register_cost=%.3fus dep_register_insert_cost=%.3fus dep_explicit_count=%u "
        "dep_owner_emit_count=%u dep_lookup_count=%u dep_lookup_skip_count=%u dep_lookup_range_skip_count=%u "
        "dep_lookup_match_count=%u "
        "dep_lookup_emit_count=%u dep_remove_count=%u dep_register_count=%u dep_fanin_actual_count=%u",
        thread_idx, cycles_to_us(submit_pipeline_dep_explicit_cycles),
        cycles_to_us(submit_pipeline_dep_owner_emit_cycles), cycles_to_us(submit_pipeline_dep_lookup_cycles),
        cycles_to_us(submit_pipeline_dep_lookup_emit_cycles), cycles_to_us(submit_pipeline_dep_remove_cycles),
        cycles_to_us(submit_pipeline_dep_register_cycles),
        cycles_to_us(submit_pipeline_dep_register_insert_cycles), submit_pipeline_dep_explicit_count,
        submit_pipeline_dep_owner_emit_count, submit_pipeline_dep_lookup_count,
        submit_pipeline_dep_lookup_skip_count, submit_pipeline_dep_lookup_range_skip_count,
        submit_pipeline_dep_lookup_match_count, submit_pipeline_dep_lookup_emit_count, submit_pipeline_dep_remove_count,
        submit_pipeline_dep_register_count,
        submit_pipeline_dep_fanin_actual_count
    );
#else
    (void)thread_idx;
#endif
}

void PTO2OrchestratorState::log_four_stage_diagnostics(int32_t thread_idx) const {
#if PTO2_PROFILING
    static const char *stage_names[PTO2_ORCH_FOUR_STAGE_COUNT] = {
        "front",
        "submitter",
        "tensormap",
        "updater",
    };
    uint64_t cumulative_cycles = 0;
    for (int32_t i = 0; i < PTO2_ORCH_FOUR_STAGE_COUNT; i++) {
        cumulative_cycles += orch_four_stage_cycles[i];
        LOG_INFO_V9(
            "Thread %d: orch_4stage name=%s count=%u cost=%.3fus serial_end=%.3fus",
            thread_idx, stage_names[i], orch_four_stage_count[i], cycles_to_us(orch_four_stage_cycles[i]),
            cycles_to_us(cumulative_cycles)
        );
    }
#else
    (void)thread_idx;
#endif
}

void PTO2OrchestratorState::log_active_detail_diagnostics(
    int32_t thread_idx, uint64_t active_cycles, uint64_t bind_cycles, uint64_t p_bind_cycles,
    uint64_t outer_scope_begin_cycles, uint64_t p_func_cycles, uint64_t outer_scope_end_cycles
) const {
#if PTO2_PROFILING
    uint64_t o4_cycles = 0;
    for (int32_t i = 0; i < PTO2_ORCH_FOUR_STAGE_COUNT; i++) {
        o4_cycles += orch_four_stage_cycles[i];
    }
    uint64_t unclassified_cycles = active_cycles > o4_cycles ? active_cycles - o4_cycles : 0;
    LOG_INFO_V9(
        "Thread %d: orch_active_detail active=%.3fus o4_total=%.3fus unclassified=%.3fus "
        "submit_wrapper_count=%u submit_wrapper_cost=%.3fus alloc_count=%u alloc_cost=%.3fus "
        "scope_begin_count=%u scope_begin_cost=%.3fus scope_end_count=%u scope_end_cost=%.3fus "
        "bind_cost=%.3fus p_bind_cost=%.3fus outer_scope_begin_cost=%.3fus p_func_cost=%.3fus "
        "outer_scope_end_cost=%.3fus",
        thread_idx, cycles_to_us(active_cycles), cycles_to_us(o4_cycles), cycles_to_us(unclassified_cycles),
        orch_submit_wrapper_count, cycles_to_us(orch_submit_wrapper_cycles), orch_alloc_tensors_count,
        cycles_to_us(orch_alloc_tensors_cycles), orch_scope_begin_count, cycles_to_us(orch_scope_begin_cycles),
        orch_scope_end_count, cycles_to_us(orch_scope_end_cycles), cycles_to_us(bind_cycles), cycles_to_us(p_bind_cycles),
        cycles_to_us(outer_scope_begin_cycles), cycles_to_us(p_func_cycles), cycles_to_us(outer_scope_end_cycles)
    );
#else
    (void)thread_idx;
    (void)active_cycles;
    (void)bind_cycles;
    (void)p_bind_cycles;
    (void)outer_scope_begin_cycles;
    (void)p_func_cycles;
    (void)outer_scope_end_cycles;
#endif
}

void PTO2OrchestratorState::log_submit_detail_diagnostics(int32_t thread_idx) const {
#if PTO2_PROFILING
    uint32_t count = orch_submit_detail_count > 0 ? orch_submit_detail_count : 1;
    LOG_INFO_V9(
        "Thread %d: orch_submit_detail count=%u deferred=%u sync=%u heap_guard=%u tensors=%" PRIu64
        " scalars=%" PRIu64 " explicit_deps=%" PRIu64 " output_bytes=%" PRIu64
        " layout=%.3fus prepare=%.3fus depgen=%.3fus sync=%.3fus explicit=%.3fus lookup=%.3fus "
        "register=%.3fus payload=%.3fus descriptor=%.3fus deferred_meta=%.3fus enqueue=%.3fus "
        "return_tail=%.3fus prep_check=%.3fus prep_alloc=%.3fus prep_ptr=%.3fus prep_prefetch=%.3fus "
        "prep_slot=%.3fus prep_scope_push=%.3fus prep_alloc_task_wait=%.3fus prep_alloc_heap_wait=%.3fus "
        "prep_alloc_task_spins=%" PRIu64 " prep_alloc_heap_spins=%" PRIu64 " prep_alloc_progress=%" PRIu64
        " avg_layout=%.3fus avg_prepare=%.3fus avg_payload=%.3fus avg_deferred_meta=%.3fus avg_enqueue=%.3fus",
        thread_idx, orch_submit_detail_count, orch_submit_deferred_count, orch_submit_sync_count,
        orch_submit_heap_guard_sync_count, static_cast<uint64_t>(orch_submit_tensor_count_total),
        static_cast<uint64_t>(orch_submit_scalar_count_total),
        static_cast<uint64_t>(orch_submit_explicit_dep_count_total),
        static_cast<uint64_t>(orch_submit_output_bytes_total), cycles_to_us(orch_submit_layout_cycles),
        cycles_to_us(orch_submit_prepare_cycles), cycles_to_us(orch_submit_depgen_cycles),
        cycles_to_us(orch_submit_sync_cycles), cycles_to_us(orch_submit_explicit_dep_cycles),
        cycles_to_us(orch_submit_lookup_cycles), cycles_to_us(orch_submit_register_cycles),
        cycles_to_us(orch_submit_payload_cycles), cycles_to_us(orch_submit_descriptor_cycles),
        cycles_to_us(orch_submit_deferred_meta_cycles), cycles_to_us(orch_submit_enqueue_cycles),
        cycles_to_us(orch_submit_return_cycles), cycles_to_us(orch_prepare_scope_check_cycles),
        cycles_to_us(orch_prepare_alloc_cycles), cycles_to_us(orch_prepare_ptr_cycles),
        cycles_to_us(orch_prepare_prefetch_cycles), cycles_to_us(orch_prepare_slot_state_cycles),
        cycles_to_us(orch_prepare_scope_push_cycles), cycles_to_us(orch_alloc_task_wait_cycles),
        cycles_to_us(orch_alloc_heap_wait_cycles), static_cast<uint64_t>(orch_alloc_task_wait_spins),
        static_cast<uint64_t>(orch_alloc_heap_wait_spins), static_cast<uint64_t>(orch_alloc_progress_count),
        cycles_to_us(orch_submit_layout_cycles) / count,
        cycles_to_us(orch_submit_prepare_cycles) / count, cycles_to_us(orch_submit_payload_cycles) / count,
        cycles_to_us(orch_submit_deferred_meta_cycles) / count, cycles_to_us(orch_submit_enqueue_cycles) / count
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

static void enqueue_or_commit_submit_record(PTO2OrchestratorState *orch, PTO2SubmitCommitRecord &record) {
    if (record.kind == PTO2SubmitPipelineRecordKind::TASK_DEFERRED) {
        append_submit_task_batch_record(orch, record);
        return;
    }

    if (!orch->submit_pipeline_enabled ||
        (orch->submit_pipeline_commit_stages == 1 && !orch->submit_pipeline_enqueue_submit_records)) {
        commit_submit_record(record);
        return;
    }

    if (orch->submit_pipeline_commit_stages == 1 && orch->submit_pipeline_enqueue_submit_records) {
        append_submit_task_batch_record(orch, record);
        return;
    }

#if PTO2_PROFILING
    uint64_t enqueue_start = get_sys_cnt_aicpu();
#endif
    mark_submit_pipeline_work_available(orch);
    submit_pipeline_queue_push(orch->submit_pipeline_queues[0], record);
#if PTO2_PROFILING
    orch->submit_pipeline_task_enqueue_cycles += get_sys_cnt_aicpu() - enqueue_start;
    orch->submit_pipeline_task_enqueue_count++;
    orch->submit_pipeline_task_enqueue_batch_count++;
#endif
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
#if PTO2_PROFILING
    uint64_t scope_begin_start = get_sys_cnt_aicpu();
#endif
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
    uint64_t scope_begin_end = get_sys_cnt_aicpu();
    orch->orch_scope_begin_cycles += scope_begin_end - scope_begin_start;
    orch->orch_scope_begin_count++;
    record_orch_four_stage(orch, PTO2OrchFourStage::FRONT, scope_begin_start, scope_begin_end);
#endif
}

void PTO2OrchestratorState::end_scope() {
    auto *orch = this;
    if (orch->fatal) {
        return;
    }
    assert(orch->scope_stack_top >= 0 && "Scope stack underflow");

#if PTO2_PROFILING
    uint64_t scope_end_start = get_sys_cnt_aicpu();
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
        if (!scope_end_queued) {
            orch->flush_submit_pipeline();
            orch->scheduler->on_scope_end(&orch->scope_tasks[begin], count);
        }
    } else {
        orch->flush_submit_pipeline();
    }

    // Rewind the task buffer — these entries are no longer needed
    orch->scope_tasks_size = begin;

#if PTO2_ORCH_PROFILING
    uint64_t _se1 = get_sys_cnt_aicpu();
    g_orch_scope_end_cycle += (_se1 - _se0);
    // l2_perf_aicpu_record_orch_phase(AicpuPhaseId::ORCH_SCOPE_END, _se0, _se1, g_orch_submit_idx, -1);
#endif
#if PTO2_PROFILING
    uint64_t scope_end_end = get_sys_cnt_aicpu();
    orch->orch_scope_end_cycles += scope_end_end - scope_end_start;
    orch->orch_scope_end_count++;
    record_orch_four_stage(orch, PTO2OrchFourStage::UPDATER, scope_end_start, scope_end_end);
#endif
}

// =============================================================================
// Task Submission
// =============================================================================

template <typename SubmitRecord>
static bool fill_deferred_submit_metadata(
    PTO2OrchestratorState *orch, const Arg &args, const PTO2OutputLayout &layout, SubmitRecord &record
) {
    if (args.explicit_dep_count() > PTO2_SUBMIT_PIPELINE_EXPLICIT_DEP_CAP) {
        orch->report_fatal(
            PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "explicit dependency count %u exceeds deferred submit cap %d",
            args.explicit_dep_count(), PTO2_SUBMIT_PIPELINE_EXPLICIT_DEP_CAP
        );
        return false;
    }
    record.explicit_dep_count = static_cast<int32_t>(args.explicit_dep_count());
    record.in_manual_scope = orch->in_manual_scope();
    record.needs_tensormap_registration = layout.needs_tensormap_registration;
    for (int32_t i = 0; i < record.explicit_dep_count; i++) {
        record.explicit_deps[i] = args.explicit_dep(static_cast<uint32_t>(i));
    }
    return true;
}

// Shared body for submit_task / submit_dummy_task. Caller has already validated
// args.has_error, decided active_mask (empty for dummy), and resolved the per-slot
// kernel_ids (all INVALID_KERNEL_ID for dummy). Performs tensormap sync, fanin
// computation (explicit_deps + auto), output registration, slot init, and pushes
// to the scheduler wiring queue.
static TaskOutputTensors submit_task_common(
    PTO2OrchestratorState *orch, const Arg &args, ActiveMask active_mask, int32_t aic_kernel_id, int32_t aiv0_kernel_id,
    int32_t aiv1_kernel_id
) {
    CYCLE_COUNT_START();
#if PTO2_PROFILING
    uint64_t four_stage_start = get_sys_cnt_aicpu();
    uint64_t four_stage_cursor = four_stage_start;
    uint64_t detail_cursor = four_stage_start;
#endif
    TaskOutputTensors result;
    PTO2OutputLayout layout = calculate_output_layout(args);
#if PTO2_PROFILING
    uint64_t detail_after_layout = get_sys_cnt_aicpu();
    orch->orch_submit_layout_cycles += detail_after_layout - detail_cursor;
    detail_cursor = detail_after_layout;
#endif
    bool heap_guard_sync = orch->submit_pipeline_defer_dependencies &&
                           layout.total_output_size >= PTO2_SUBMIT_PIPELINE_HEAP_GUARD_OUTPUT_BYTES;
    if (heap_guard_sync) {
        uint8_t ring_id = orch->current_ring_id();
        auto &allocator = orch->rings[ring_id].task_allocator;
        heap_guard_sync = allocator.allocation_would_block_on_heap(layout.total_output_size);
    }
    if (heap_guard_sync) {
        orch->flush_submit_pipeline();
        uint8_t ring_id = orch->current_ring_id();
        auto &allocator = orch->rings[ring_id].task_allocator;
        heap_guard_sync = allocator.allocation_would_block_on_heap(layout.total_output_size);
    }
    PTO2PreparedTask prepared;
    if (!prepare_task(orch, args, layout.total_output_size, active_mask, &prepared)) {
        return result;
    }
#if PTO2_PROFILING
    uint64_t detail_after_prepare = get_sys_cnt_aicpu();
    orch->orch_submit_prepare_cycles += detail_after_prepare - detail_cursor;
    detail_cursor = detail_after_prepare;
    orch->orch_submit_detail_count++;
    if (orch->submit_pipeline_defer_dependencies && !heap_guard_sync) {
        orch->orch_submit_deferred_count++;
    } else {
        orch->orch_submit_sync_count++;
    }
    if (heap_guard_sync) {
        orch->orch_submit_heap_guard_sync_count++;
    }
    orch->orch_submit_tensor_count_total += static_cast<uint64_t>(args.tensor_count());
    orch->orch_submit_scalar_count_total += static_cast<uint64_t>(args.scalar_count());
    orch->orch_submit_explicit_dep_count_total += static_cast<uint64_t>(args.explicit_dep_count());
    orch->orch_submit_output_bytes_total += static_cast<uint64_t>(layout.total_output_size);
#endif
    uint8_t ring_id = prepared.task_id.ring();
    PTO2SchedulerState *sched = orch->scheduler;
    PTO2RingFlowControl &fc = orch->sm_header->rings[ring_id].fc;
    PTO2TaskId task_id = prepared.task_id;
    PTO2TaskSlotState &cur_slot_state = *prepared.slot_state;
    PTO2TaskDescriptor &task = *prepared.task;
    PTO2TaskPayload &payload = *prepared.payload;
    result.set_task_id(task_id);

    // dep_gen capture point: snapshot the orch submit_task inputs while the
    // tensormap is still in its pre-lookup state for this task. Replay reads
    // these records offline to reconstruct the complete dep graph, sidestepping
    // the race window in L2PerfRecord::fanout[] where an early-finishing
    // producer's record gets sealed before later-submitted consumers can
    // register themselves.
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
            tensor_ptrs[i] = (args.tag(i) == TensorArgType::OUTPUT) ? nullptr : args.tensor(i).ptr;
            arg_types_u8[i] = static_cast<uint8_t>(args.tag(i));
        }
        dep_gen_aicpu_record_submit(
            task_id.raw, orch->in_manual_scope(), tc, tensor_ptrs, arg_types_u8,
            static_cast<int>(args.explicit_dep_count()), reinterpret_cast<const uint64_t *>(args.explicit_deps_data())
        );
    }
#if PTO2_PROFILING
    uint64_t detail_after_depgen = get_sys_cnt_aicpu();
    orch->orch_submit_depgen_cycles += detail_after_depgen - detail_cursor;
    detail_cursor = detail_after_depgen;
#endif

    PTO2FaninBuilder fanin_builder(orch->rings[ring_id].fanin_pool);
#if PTO2_PROFILING
    uint64_t front_end = get_sys_cnt_aicpu();
    record_orch_four_stage(orch, PTO2OrchFourStage::FRONT, four_stage_cursor, front_end);
    four_stage_cursor = front_end;
#endif

    CYCLE_COUNT_LAP_RECORD(g_orch_alloc_cycle, AicpuPhaseId::ORCH_ALLOC, task_id.raw);

#if PTO2_PROFILING
    if (layout.total_output_size > 0) {
        orch->buffers_allocated++;
        orch->bytes_allocated += layout.total_output_size;
    }
#endif

    bool defer_dependencies = orch->submit_pipeline_defer_dependencies && !heap_guard_sync;
    if (defer_dependencies && args.explicit_dep_count() > PTO2_SUBMIT_PIPELINE_EXPLICIT_DEP_CAP) {
        orch->report_fatal(
            PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "explicit dependency count %u exceeds deferred submit cap %d",
            args.explicit_dep_count(), PTO2_SUBMIT_PIPELINE_EXPLICIT_DEP_CAP
        );
        return result;
    }
#if PTO2_PROFILING
    uint64_t sync_end = four_stage_cursor;
    uint64_t tensormap_end = four_stage_cursor;
#endif

    if (!defer_dependencies) {
        // === STEP 2: Sync TensorMap validity and optional cleanup ===
        // Read current last_task_alive from shared memory for this ring
        int32_t sm_last_task_alive = fc.last_task_alive.load(std::memory_order_acquire);

        orch->tensor_map.sync_tensormap(task_id, sm_last_task_alive);
#if PTO2_PROFILING
        sync_end = get_sys_cnt_aicpu();
        orch->orch_submit_sync_cycles += sync_end - detail_cursor;
        detail_cursor = sync_end;
#endif

        CYCLE_COUNT_LAP_RECORD(g_orch_sync_cycle, AicpuPhaseId::ORCH_SYNC, task_id.raw);

        uint64_t explicit_dep_start = 0;
#if PTO2_PROFILING
        explicit_dep_start = detail_cursor;
#endif
        for (uint32_t i = 0; i < args.explicit_dep_count(); i++) {
            PTO2TaskId dep_task_id = args.explicit_dep(i);
            if (!dep_task_id.is_valid()) {
                orch->report_fatal(
                    PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "Arg.set_dependencies(...) requires valid task ids"
                );
                return result;
            }
            PTO2SharedMemoryRingHeader &dep_ring = orch->sm_header->rings[dep_task_id.ring()];
            int32_t dep_local_task_id = static_cast<int32_t>(dep_task_id.local());
            int32_t dep_last_task_alive = dep_ring.fc.last_task_alive.load(std::memory_order_acquire);
            if (dep_local_task_id < dep_last_task_alive) {
                continue;
            }
            PTO2TaskSlotState *producer_slot_state = &dep_ring.get_slot_state_by_task_id(dep_local_task_id);
            if (!append_fanin_or_fail(orch, producer_slot_state, &fanin_builder, ring_id)) {
                return result;
            }
        }
#if PTO2_PROFILING
        uint64_t detail_after_explicit_dep = get_sys_cnt_aicpu();
        orch->orch_submit_explicit_dep_cycles += detail_after_explicit_dep - explicit_dep_start;
        detail_cursor = detail_after_explicit_dep;
#endif

        // === STEP 3: Lookup inputs (creator retention + tensormap modifier lookup) ===
        DepInputs dep_inputs{
            args.tensor_count(),       args.tensor_data(), args.tag_data(), static_cast<int32_t>(args.explicit_dep_count()),
            args.explicit_deps_data(),
        };

        auto runtime_emit = [&](PTO2TaskId producer_task_id) -> bool {
            PTO2TaskSlotState *prod_state =
                &orch->sm_header->rings[producer_task_id.ring()].get_slot_state_by_task_id(producer_task_id.local());
            return append_fanin_or_fail(orch, prod_state, &fanin_builder, ring_id);
        };

        if (!compute_task_fanin(dep_inputs, orch->tensor_map, orch->in_manual_scope(), runtime_emit)) {
            return result;
        }
#if PTO2_PROFILING
        uint64_t detail_after_lookup = get_sys_cnt_aicpu();
        orch->orch_submit_lookup_cycles += detail_after_lookup - detail_cursor;
        detail_cursor = detail_after_lookup;
#endif

        CYCLE_COUNT_LAP_RECORD(g_orch_lookup_cycle, AicpuPhaseId::ORCH_LOOKUP, task_id.raw);

        // === STEP 4: Register outputs/inouts in TensorMap (must be separate from lookup) ===
        if (layout.needs_tensormap_registration) {
            register_task_outputs(dep_inputs, task_id, orch->tensor_map, orch->in_manual_scope());
        }
#if PTO2_PROFILING
        tensormap_end = get_sys_cnt_aicpu();
        orch->orch_submit_register_cycles += tensormap_end - detail_cursor;
        detail_cursor = tensormap_end;
        record_orch_four_stage(orch, PTO2OrchFourStage::TENSORMAP, sync_end, tensormap_end);
#endif

        CYCLE_COUNT_LAP_RECORD(g_orch_insert_cycle, AicpuPhaseId::ORCH_INSERT, task_id.raw);
    }

    // === STEP 5: Materialize payload tensors/scalars synchronously ===
    // Returned TaskOutputTensors borrow from payload.tensors[], so this cannot
    // move to the commit stage without changing the orchestration API contract.
    payload.init(args, result, prepared.alloc_result, layout);
#if PTO2_PROFILING
    uint64_t submitter_end = get_sys_cnt_aicpu();
    orch->orch_submit_payload_cycles += submitter_end - detail_cursor;
    detail_cursor = submitter_end;
    record_orch_four_stage(orch, PTO2OrchFourStage::SUBMITTER, four_stage_cursor, sync_end);
    record_orch_four_stage(orch, PTO2OrchFourStage::SUBMITTER, tensormap_end, submitter_end);
#endif

    CYCLE_COUNT_LAP_RECORD(g_orch_args_cycle, AicpuPhaseId::ORCH_PARAMS, task_id.raw);
#if PTO2_ORCH_PROFILING
    g_orch_args_atomic_count += 2;  // fanout_lock.store + fanout_count.store
#endif

    if (defer_dependencies) {
        if (orch->submit_pipeline_compact_deferred_records) {
            PTO2DeferredSubmitRecord &deferred_record = *append_deferred_submit_task_batch_record_slot(orch);
            deferred_record.payload = &payload;
            deferred_record.slot_state = &cur_slot_state;
            deferred_record.scheduler = sched;
            deferred_record.packed_buffer_base = prepared.alloc_result.packed_base;
            deferred_record.packed_buffer_end = prepared.alloc_result.packed_end;
            deferred_record.task_id = task_id;
            deferred_record.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIC)] = aic_kernel_id;
            deferred_record.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV0)] = aiv0_kernel_id;
            deferred_record.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV1)] = aiv1_kernel_id;
            if (!fill_deferred_submit_metadata(orch, args, layout, deferred_record)) {
                return result;
            }
        } else {
            write_submit_descriptor(
                task, task_id, prepared.alloc_result, aic_kernel_id, aiv0_kernel_id, aiv1_kernel_id
            );
#if PTO2_PROFILING
            uint64_t detail_after_descriptor = get_sys_cnt_aicpu();
            orch->orch_submit_descriptor_cycles += detail_after_descriptor - detail_cursor;
            detail_cursor = detail_after_descriptor;
#endif
            PTO2SubmitCommitRecord &commit_record = *append_submit_task_batch_record_slot(orch);
            commit_record.kind = PTO2SubmitPipelineRecordKind::TASK_DEFERRED;
            commit_record.payload = &payload;
            commit_record.slot_state = &cur_slot_state;
            commit_record.scheduler = sched;
            commit_record.task_id = task_id;
            if (!fill_deferred_submit_metadata(orch, args, layout, commit_record)) {
                return result;
            }
        }
#if PTO2_PROFILING
        uint64_t detail_after_deferred_meta = get_sys_cnt_aicpu();
        orch->orch_submit_deferred_meta_cycles += detail_after_deferred_meta - detail_cursor;
        detail_cursor = detail_after_deferred_meta;
#endif
    } else {
        PTO2SubmitCommitRecord commit_record{};
        commit_record.task = &task;
        commit_record.payload = &payload;
        commit_record.slot_state = &cur_slot_state;
        commit_record.scheduler = sched;
        commit_record.alloc_result = prepared.alloc_result;
        commit_record.task_id = task_id;
        commit_record.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIC)] = aic_kernel_id;
        commit_record.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV0)] = aiv0_kernel_id;
        commit_record.kernel_id[static_cast<int>(PTO2SubtaskSlot::AIV1)] = aiv1_kernel_id;
        commit_record.fanin_spill_pool = &fanin_builder.spill_pool;
        commit_record.fanin_actual_count = fanin_builder.count;
        commit_record.fanin_spill_start = fanin_builder.spill_start;
        int32_t inline_count = std::min(fanin_builder.count, PTO2_FANIN_INLINE_CAP);
        for (int32_t i = 0; i < inline_count; i++) {
            commit_record.fanin_inline_slot_states[i] = fanin_builder.inline_slots[i];
        }
#if PTO2_PROFILING
        uint64_t detail_after_deferred_meta = get_sys_cnt_aicpu();
        orch->orch_submit_deferred_meta_cycles += detail_after_deferred_meta - detail_cursor;
        detail_cursor = detail_after_deferred_meta;
#endif
        enqueue_or_commit_submit_record(orch, commit_record);
    }
#if PTO2_PROFILING
    uint64_t updater_end = get_sys_cnt_aicpu();
    orch->orch_submit_enqueue_cycles += updater_end - detail_cursor;
    detail_cursor = updater_end;
    record_orch_four_stage(orch, PTO2OrchFourStage::UPDATER, submitter_end, updater_end);
#endif

    CYCLE_COUNT_LAP_RECORD(g_orch_fanin_cycle, AicpuPhaseId::ORCH_FANIN, task_id.raw);

#if PTO2_PROFILING
    uint64_t detail_after_return_tail = get_sys_cnt_aicpu();
    orch->orch_submit_return_cycles += detail_after_return_tail - detail_cursor;
    orch->tasks_submitted++;
#if PTO2_ORCH_PROFILING
    g_orch_submit_count++;
#endif
    g_orch_submit_idx++;
#endif
    return result;
}

TaskOutputTensors PTO2OrchestratorState::submit_task(const MixedKernels &mixed_kernels, const Arg &args) {
    auto *orch = this;
#if PTO2_PROFILING
    uint64_t wrapper_start = get_sys_cnt_aicpu();
#endif

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

#if PTO2_PROFILING
    uint64_t wrapper_end = get_sys_cnt_aicpu();
    orch->orch_submit_wrapper_cycles += wrapper_end - wrapper_start;
    orch->orch_submit_wrapper_count++;
    record_orch_four_stage(orch, PTO2OrchFourStage::FRONT, wrapper_start, wrapper_end);
#endif
    return submit_task_common(
        orch, args, active_mask, normalized.aic_kernel_id, normalized.aiv0_kernel_id, normalized.aiv1_kernel_id
    );
}

// Submit a dependency-only task: full dependency graph participation
// (tensormap lookup/insert, explicit_deps, manual_dep, manual_scope) but no
// AICore dispatch. Empty active_mask routes the slot to the DUMMY ready
// bucket; dispatch loop short-circuits to completion. Accepts the same Arg
// shape as submit_task; scalars are permitted but never consumed.
TaskOutputTensors PTO2OrchestratorState::submit_dummy_task(const Arg &args) {
    auto *orch = this;
#if PTO2_PROFILING
    uint64_t wrapper_start = get_sys_cnt_aicpu();
#endif

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

#if PTO2_PROFILING
    uint64_t wrapper_end = get_sys_cnt_aicpu();
    orch->orch_submit_wrapper_cycles += wrapper_end - wrapper_start;
    orch->orch_submit_wrapper_count++;
    record_orch_four_stage(orch, PTO2OrchFourStage::FRONT, wrapper_start, wrapper_end);
#endif
    return submit_task_common(orch, args, ActiveMask{}, INVALID_KERNEL_ID, INVALID_KERNEL_ID, INVALID_KERNEL_ID);
}

TaskOutputTensors PTO2OrchestratorState::alloc_tensors(const Arg &args) {
    auto *orch = this;
#if PTO2_PROFILING
    uint64_t alloc_start = get_sys_cnt_aicpu();
#endif
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
#if PTO2_PROFILING
    uint64_t alloc_front_end = get_sys_cnt_aicpu();
    record_orch_four_stage(orch, PTO2OrchFourStage::FRONT, alloc_start, alloc_front_end);
#endif

    PTO2TaskDescriptor &task = *prepared.task;
    PTO2TaskPayload &payload = *prepared.payload;

    CYCLE_COUNT_LAP_RECORD(g_orch_alloc_cycle, AicpuPhaseId::ORCH_ALLOC, prepared.task_id.raw);

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
#if PTO2_PROFILING
    uint64_t alloc_submitter_end = get_sys_cnt_aicpu();
    record_orch_four_stage(orch, PTO2OrchFourStage::SUBMITTER, alloc_front_end, alloc_submitter_end);
#endif
    payload.fanin_actual_count = 0;
    payload.fanin_spill_start = 0;
    payload.fanin_spill_pool = &orch->rings[prepared.task_id.ring()].fanin_pool;
    CYCLE_COUNT_LAP_RECORD(g_orch_args_cycle, AicpuPhaseId::ORCH_PARAMS, prepared.task_id.raw);

    if (prepared.slot_state != nullptr) {
        // Hidden alloc tasks complete inline in the orchestrator before any
        // consumer can exist, so they have no fanout to notify and no worker
        // subtasks to retire. Running the full on_mixed_task_complete path
        // would only pay unnecessary fanout_lock / traversal overhead here.
        // The generic slot initialization done in prepare_task() is still
        // required so scope_end can release the producer-side reference and
        // drive the slot to CONSUMED, but worker dispatch fields are never
        // observed for hidden alloc tasks.
        prepared.slot_state->task_state.store(PTO2_TASK_COMPLETED, std::memory_order_release);
    }
    orch->inline_completed_tasks++;
#if PTO2_PROFILING
    uint64_t alloc_end = get_sys_cnt_aicpu();
    orch->orch_alloc_tensors_cycles += alloc_end - alloc_start;
    orch->orch_alloc_tensors_count++;
    record_orch_four_stage(orch, PTO2OrchFourStage::UPDATER, alloc_submitter_end, alloc_end);
#endif

    CYCLE_COUNT_LAP_RECORD(g_orch_fanin_cycle, AicpuPhaseId::ORCH_FANIN, prepared.task_id.raw);

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
