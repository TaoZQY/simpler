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
#include <dlfcn.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#ifdef __linux__
#include <sys/mman.h>
#endif

#include "aicpu/device_time.h"
#include "aicpu/orch_so_file.h"
#include "callable_protocol.h"
#include "pto2_dispatch_payload.h"
#include "runtime.h"
#include "spin_hint.h"

// Runtime headers (full struct definition for create/destroy + PTO2_SCOPE)
#include "pto_runtime2.h"
#include "pto_runtime2_types.h"
#include "pto_shared_memory.h"

// Performance profiling headers
#include "aicpu/l2_perf_collector_aicpu.h"
#include "aicpu/tensor_dump_aicpu.h"
#include "aicpu/dep_gen_collector_aicpu.h"
#include "common/l2_perf_profiling.h"
#include "common/unified_log.h"

// Register-based communication
#include "aicpu/platform_regs.h"
#include "aicpu/platform_aicpu_affinity.h"
#include "common/platform_config.h"

// Core type definitions
#include "common/core_type.h"

// CoreCallable for resolved dispatch address
#include "callable.h"
#include "pipeline_strategy.h"

// Scheduler data structures (CoreExecState, CoreTracker, etc.)
#include "scheduler/scheduler_types.h"

// Scheduler context class
#include "scheduler/scheduler_context.h"

// Device orchestration function signature (loaded via dlopen).
// The executor binds the current thread's PTO2Runtime into orchestration TLS
// before calling the user entry.
typedef void (*DeviceOrchestrationFunc)(const ChipStorageTaskArgs &orch_args);
typedef void (*DeviceOrchestrationBindRuntimeFunc)(PTO2Runtime *rt);

// Config function exported by orchestration .so
typedef PTO2OrchestrationConfig (*DeviceOrchestrationConfigFunc)(const ChipStorageTaskArgs &orch_args);

// From orchestration/common.cpp linked into this DSO — updates g_current_runtime here (distinct from
// framework_bind_runtime in the dlopen'd libdevice_orch_*.so).
extern "C" void framework_bind_runtime(PTO2Runtime *rt);

constexpr const char *DEFAULT_ORCH_ENTRY_SYMBOL = "aicpu_orchestration_entry";
constexpr const char *DEFAULT_ORCH_CONFIG_SYMBOL = "aicpu_orchestration_config";

static int32_t read_pto2_runtime_status(Runtime *runtime) {
    if (runtime == nullptr) {
        return 0;
    }

    void *sm = runtime->get_gm_sm_ptr();
    if (sm == nullptr) {
        return 0;
    }

    auto *header = static_cast<PTO2SharedMemoryHeader *>(sm);
    int32_t orch_error_code = header->orch_error_code.load(std::memory_order_acquire);
    int32_t sched_error_code = header->sched_error_code.load(std::memory_order_acquire);
    return runtime_status_from_error_codes(orch_error_code, sched_error_code);
}

static PTO2Runtime *rt{nullptr};

#if PTO2_PROFILING
static uint32_t clamp_phase_counter_u32(uint64_t value) {
    constexpr uint64_t max_u32 = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
    return static_cast<uint32_t>(value > max_u32 ? max_u32 : value);
}
#endif

// Per-callable_id orchestration SO table. The executor dispatches
// `orch_so_table_[active_callable_id_]` (created on first sighting of
// that callable_id, kept warm across runs).
// MAX_REGISTERED_CALLABLE_IDS is the protocol hard cap on callable_id values
// (mailbox uint32 callable_id, register() returns small ints) and is shared
// with the host bounds check in DeviceRunner::register_prepared_callable —
// see src/common/task_interface/callable_protocol.h.

struct OrchSoEntry {
    bool in_use{false};
    void *handle{nullptr};
    char path[256]{};
    DeviceOrchestrationFunc func{nullptr};
    DeviceOrchestrationBindRuntimeFunc bind{nullptr};
    DeviceOrchestrationConfigFunc config_func{nullptr};
};

struct AicpuExecutor {
    int32_t sched_thread_num_;
    int32_t orch_thread_num_;
    int32_t primary_orch_thread_idx_;
    bool orch_to_sched_{false};
    PipelineLayout pipeline_layout_{};
    int32_t scheduler_index_by_thread_[MAX_AICPU_THREADS];
    int32_t orchestrator_stage_by_thread_[MAX_AICPU_THREADS];

    // ===== Thread management state =====
    std::atomic<int32_t> thread_idx_{0};
    std::atomic<bool> initialized_{false};
    std::atomic<bool> init_done_{false};
    std::atomic<bool> init_failed_{false};
    std::atomic<bool> finished_{false};

    int32_t thread_num_{0};

    // ===== Task queue state (managed by scheduler ready queues) =====

    std::atomic<int32_t> finished_count_{0};
    std::atomic<bool> runtime_init_ready_{false};

    // Per-Worker arena backing the PTO2Runtime + sm_handle + orch/sched/mailbox
    // sub-regions (created in runtime_create_from_sm, released in runtime_destroy).
    // Default-constructed: libc-backed backend, no ctx.
    DeviceArena runtime_arena_;

    // Cached orch args pointer set by the orchestration thread before scheduler
    // init; consumed by the (*p_func)(*orch_args_cached_) invocation below.
    const ChipStorageTaskArgs *orch_args_cached_{nullptr};

    // Per-callable_id table. Single orch thread today, so first-write/read
    // race is not possible; if multiple orch threads are ever introduced,
    // guard the in_use=false→true transition with a mutex.
    OrchSoEntry orch_so_table_[MAX_REGISTERED_CALLABLE_IDS];

    // ===== Scheduler context (owns all dispatch/completion/drain state) =====
    SchedulerContext sched_ctx_;

    // ===== Methods =====
    int32_t init(Runtime *runtime);
    int32_t run(Runtime *runtime);
    void deinit(Runtime *runtime);
    void resolve_thread_layout(Runtime *runtime);
    int32_t scheduler_idx_for_thread(int32_t thread_idx) const;
    int32_t orch_stage_idx_for_thread(int32_t thread_idx) const;
    int32_t scheduler_context_idx_for_thread(int32_t thread_idx) const;

    ~AicpuExecutor() {
        // Process-wide teardown (the single static instance dies here). Every
        // in-use callable_id slot is dlclose()'d here; each is otherwise kept
        // alive across runs for cache-hit reuse.
        for (auto &e : orch_so_table_) {
            if (!e.in_use) continue;
            if (e.handle != nullptr) dlclose(e.handle);
            if (e.path[0] != '\0') unlink(e.path);
            e = OrchSoEntry{};
        }
    }
};

static AicpuExecutor g_aicpu_executor;

// ===== AicpuExecutor Method Implementations =====

void AicpuExecutor::resolve_thread_layout(Runtime *runtime) {
    thread_num_ = runtime->sche_cpu_num;
    if (thread_num_ == 0) thread_num_ = 1;

    for (int32_t i = 0; i < MAX_AICPU_THREADS; ++i) {
        scheduler_index_by_thread_[i] = -1;
        orchestrator_stage_by_thread_[i] = -1;
    }

    const int32_t raw_strategy = runtime->pipeline_strategy;
    const int32_t baseline_strategy = -1;
    bool use_legacy_baseline = raw_strategy < 0;

    if (!use_legacy_baseline) {
        if (raw_strategy > static_cast<int32_t>(PipelineStrategy::S2_O2_PIPELINE_ORCH)) {
            LOG_WARN("Pipeline strategy %d is out of range; falling back to baseline layout", raw_strategy);
            use_legacy_baseline = true;
        } else {
            pipeline_layout_ = resolve_pipeline_layout(raw_strategy);
            int32_t requested_total = pipeline_layout_.scheduler_threads + pipeline_layout_.orchestrator_threads;
            if (thread_num_ < requested_total) {
                LOG_WARN(
                    "Pipeline strategy %s needs %d AICPU threads, got %d; falling back to baseline layout",
                    pipeline_layout_.name, requested_total, thread_num_
                );
                use_legacy_baseline = true;
            }
        }
    }

    if (use_legacy_baseline) {
        pipeline_layout_ = {
            PipelineStrategy::LEGACY_BASELINE, 0, 1, {-1, -1, -1, -1, -1, -1}, {-1, -1, -1, -1, -1, -1},
            "baseline", "baseline"
        };
        orch_thread_num_ = 1;
        sched_thread_num_ = thread_num_ - orch_thread_num_;
        for (int32_t i = 0; i < sched_thread_num_ && i < MAX_AICPU_THREADS; ++i) {
            scheduler_index_by_thread_[i] = i;
        }
        if (sched_thread_num_ >= 0 && sched_thread_num_ < MAX_AICPU_THREADS) {
            orchestrator_stage_by_thread_[sched_thread_num_] = 0;
        }
    } else {
        sched_thread_num_ = pipeline_layout_.scheduler_threads;
        orch_thread_num_ = pipeline_layout_.orchestrator_threads;
        for (int32_t i = 0; i < thread_num_ && i < PIPELINE_LAYOUT_MAX_THREADS; ++i) {
            scheduler_index_by_thread_[i] = pipeline_layout_.scheduler_index_by_thread[i];
            orchestrator_stage_by_thread_[i] = pipeline_layout_.orchestrator_stage_by_thread[i];
        }
    }

    runtime->pipeline_strategy = use_legacy_baseline ? baseline_strategy : raw_strategy;
    if (sched_thread_num_ < 0) sched_thread_num_ = 0;
    primary_orch_thread_idx_ = -1;
    for (int32_t i = 0; i < thread_num_ && i < MAX_AICPU_THREADS; ++i) {
        if (orchestrator_stage_by_thread_[i] == 0) {
            primary_orch_thread_idx_ = i;
            break;
        }
    }
    if (primary_orch_thread_idx_ < 0) {
        primary_orch_thread_idx_ = sched_thread_num_;
    }
}

int32_t AicpuExecutor::scheduler_idx_for_thread(int32_t thread_idx) const {
    if (thread_idx < 0 || thread_idx >= MAX_AICPU_THREADS) {
        return -1;
    }
    return scheduler_index_by_thread_[thread_idx];
}

int32_t AicpuExecutor::orch_stage_idx_for_thread(int32_t thread_idx) const {
    if (thread_idx < 0 || thread_idx >= MAX_AICPU_THREADS) {
        return -1;
    }
    return orchestrator_stage_by_thread_[thread_idx];
}

int32_t AicpuExecutor::scheduler_context_idx_for_thread(int32_t thread_idx) const {
    int32_t sched_idx = scheduler_idx_for_thread(thread_idx);
    if (sched_idx >= 0) {
        return sched_idx;
    }
    int32_t orch_stage_idx = orch_stage_idx_for_thread(thread_idx);
    if (orch_stage_idx >= 0) {
        return sched_thread_num_ + orch_stage_idx;
    }
    return thread_idx;
}

int32_t AicpuExecutor::init(Runtime *runtime) {
    bool expected = false;
    if (!initialized_.compare_exchange_strong(expected, true, std::memory_order_acq_rel, std::memory_order_acquire)) {
        return 0;
    }

    LOG_INFO_V0("AicpuExecutor: Initializing");

    if (runtime == nullptr) {
        LOG_ERROR("runtime is nullptr");
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    resolve_thread_layout(runtime);
    orch_to_sched_ = runtime->orch_to_sched;
    if (thread_num_ < 1 || thread_num_ > MAX_AICPU_THREADS) {
        LOG_ERROR("Invalid thread_num: %d", thread_num_);
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }
    LOG_INFO_V0(
        "AicpuExecutor layout: strategy=%s threads=%d sched=%d orch=%d primary_orch=%d clusters=%s",
        pipeline_layout_.name, thread_num_, sched_thread_num_, orch_thread_num_, primary_orch_thread_idx_,
        pipeline_layout_.cluster_layout
    );

    if (sched_ctx_.init(runtime, thread_num_, sched_thread_num_, orch_to_sched_, get_platform_regs()) != 0) {
        init_failed_.store(true, std::memory_order_release);
        return -1;
    }

    finished_count_.store(0, std::memory_order_release);

    init_done_.store(true, std::memory_order_release);
    LOG_INFO_V0("AicpuExecutor: Init complete");
    return 0;
}

/**
 * Shutdown AICore - Send exit signal via registers to all AICore kernels
 */
int32_t AicpuExecutor::run(Runtime *runtime) {
    int32_t assigned_thread_idx = platform_aicpu_affinity_logical_index();
    int32_t thread_idx =
        (assigned_thread_idx >= 0 && assigned_thread_idx < thread_num_) ? assigned_thread_idx : thread_idx_++;
    int32_t run_rc = 0;
    int32_t scheduler_idx = scheduler_idx_for_thread(thread_idx);
    int32_t scheduler_context_idx = scheduler_context_idx_for_thread(thread_idx);
    int32_t orch_stage_idx = orch_stage_idx_for_thread(thread_idx);
    LOG_INFO_V0("Thread %d: Start", thread_idx);

    // Orchestrator check
    if (thread_idx == primary_orch_thread_idx_) {
#if PTO2_PROFILING
        uint64_t orch_cycle_start = 0;
        uint64_t orch_end_ts = 0;
        uint64_t orch_active_end = 0;
        uint64_t orch_drain_end = 0;
        uint64_t bind_start = 0;
        uint64_t bind_end = 0;
        uint64_t p_bind_start = 0;
        uint64_t p_bind_end = 0;
        uint64_t outer_scope_begin_start = 0;
        uint64_t outer_scope_begin_end = 0;
        uint64_t p_func_start = 0;
        uint64_t p_func_end = 0;
        uint64_t outer_scope_end_start = 0;
        int32_t pto2_submitted_tasks = -1;
        uint64_t orch_active_cycles = 0;
        uint64_t orch_stop_cycles = 0;
        uint64_t orch_diag_cycles = 0;
        uint64_t orch_active_detail_log_cycles = 0;
        uint64_t orch_depgen_cycles = 0;
        uint64_t orch_task_count_cycles = 0;
        uint64_t orch_done_signal_cycles = 0;
        uint64_t orch_sched_handoff_cycles = 0;
        int32_t pto2_completed_tasks_at_handoff = 0;
#endif
        // Orchestrator thread: load + run the device orchestration SO. The braces
        // scope the per-callable dlopen / SO-table locals to this block.
        {
            // Per-callable_id dispatch: the orch SO state lives in
            // `orch_so_table_[callable_id]` keyed by registration order;
            // reload is governed by `register_new_callable_id_`.
            const int32_t callable_id = runtime->get_active_callable_id();
            if (callable_id < 0 || callable_id >= MAX_REGISTERED_CALLABLE_IDS) {
                LOG_ERROR(
                    "Thread %d: invalid callable_id %d (limit=%d)", thread_idx, callable_id, MAX_REGISTERED_CALLABLE_IDS
                );
                runtime_init_ready_.store(true, std::memory_order_release);
                return -1;
            }
            void **p_handle = &orch_so_table_[callable_id].handle;
            char *p_path = orch_so_table_[callable_id].path;
            DeviceOrchestrationFunc *p_func = &orch_so_table_[callable_id].func;
            DeviceOrchestrationBindRuntimeFunc *p_bind = &orch_so_table_[callable_id].bind;
            DeviceOrchestrationConfigFunc *p_config_func = &orch_so_table_[callable_id].config_func;
            const bool reload_so = runtime->register_new_callable_id();

            if (reload_so) {
                LOG_INFO_V0("Thread %d: New orch SO detected (callable_id=%d), (re)loading", thread_idx, callable_id);
                if (*p_handle != nullptr) {
                    dlclose(*p_handle);
                    *p_handle = nullptr;
                    *p_func = nullptr;
                    *p_bind = nullptr;
                    if (p_path[0] != '\0') {
                        // Unlink the old file so the new open() lands on a
                        // fresh inode — protects against SIGBUS / ETXTBSY when
                        // the kernel still has the old mapping pinned.
                        unlink(p_path);
                        p_path[0] = '\0';
                    }
                }

                const void *so_data = reinterpret_cast<const void *>(runtime->get_dev_orch_so_addr());
                size_t so_size = runtime->get_dev_orch_so_size();

                if (so_data == nullptr || so_size == 0) {
                    LOG_ERROR("Thread %d: Device orchestration SO not set", thread_idx);
                    // Unblock scheduler threads before returning so they don't spin forever.
                    runtime_init_ready_.store(true, std::memory_order_release);
                    return -1;
                }

                // Try multiple paths that may allow execution on AICPU
                char so_path[256];
                bool file_created = false;
                const char *candidate_dirs[] = {
                    "/usr/lib64/aicpu_kernels/0/aicpu_kernels_device", "/usr/lib64", "/lib64", "/var/tmp", "/tmp"
                };
                const int32_t num_candidates = sizeof(candidate_dirs) / sizeof(candidate_dirs[0]);

                for (int32_t i = 0; i < num_candidates && !file_created; i++) {
                    int32_t fd = create_orch_so_file(candidate_dirs[i], callable_id, so_path, sizeof(so_path));
                    if (fd < 0) {
                        LOG_INFO_V0(
                            "Thread %d: Cannot create SO at %s (errno=%d), trying next path", thread_idx, so_path, errno
                        );
                        continue;
                    }
                    ssize_t written = write(fd, so_data, so_size);
                    close(fd);
                    if (written != static_cast<ssize_t>(so_size)) {
                        LOG_INFO_V0(
                            "Thread %d: Cannot write SO to %s (errno=%d), trying next path", thread_idx, so_path, errno
                        );
                        unlink(so_path);
                        continue;
                    }
                    file_created = true;
                    LOG_INFO_V0("Thread %d: Created SO file at %s (%zu bytes)", thread_idx, so_path, so_size);
                }

                if (!file_created) {
                    LOG_ERROR("Thread %d: Failed to create SO file in any candidate path", thread_idx);
                    // Unblock scheduler threads before returning so they don't spin forever.
                    runtime_init_ready_.store(true, std::memory_order_release);
                    return -1;
                }

                dlerror();
                void *handle = dlopen(so_path, RTLD_LAZY | RTLD_LOCAL);
                const char *dlopen_err = dlerror();
                if (handle == nullptr) {
                    LOG_ERROR("Thread %d: dlopen failed: %s", thread_idx, dlopen_err ? dlopen_err : "unknown");
                    unlink(so_path);
                    // Unblock scheduler threads before returning so they don't spin forever.
                    runtime_init_ready_.store(true, std::memory_order_release);
                    return -1;
                }
                LOG_INFO_V0("Thread %d: dlopen succeeded, handle=%p", thread_idx, handle);

                // Unlink the on-disk SO immediately: dlopen has already mmap'd
                // the image, so the kernel keeps the inode alive until the
                // matching dlclose / process exit. This prevents stale
                // libdevice_orch_<pid>_<cid>.so files from accumulating in
                // /tmp when child processes exit via os._exit(0), which skips
                // ~AicpuExecutor (worker.py: _sub/_chip/_child loops).
                unlink(so_path);

                const char *entry_symbol = runtime->get_device_orch_func_name();
                if (entry_symbol == nullptr || entry_symbol[0] == '\0') {
                    entry_symbol = DEFAULT_ORCH_ENTRY_SYMBOL;
                }
                const char *config_symbol = runtime->get_device_orch_config_name();
                if (config_symbol == nullptr || config_symbol[0] == '\0') {
                    config_symbol = DEFAULT_ORCH_CONFIG_SYMBOL;
                }

                dlerror();
                DeviceOrchestrationFunc orch_func =
                    reinterpret_cast<DeviceOrchestrationFunc>(dlsym(handle, entry_symbol));
                const char *entry_dlsym_error = dlerror();
                if (entry_dlsym_error != nullptr) {
                    LOG_ERROR(
                        "Thread %d: dlsym failed for entry symbol '%s': %s", thread_idx, entry_symbol, entry_dlsym_error
                    );
                    dlclose(handle);
                    unlink(so_path);
                    // Unblock scheduler threads before returning so they don't spin forever.
                    runtime_init_ready_.store(true, std::memory_order_release);
                    return -1;
                }
                if (orch_func == nullptr) {
                    LOG_ERROR("Thread %d: dlsym returned NULL for entry symbol '%s'", thread_idx, entry_symbol);
                    dlclose(handle);
                    unlink(so_path);
                    // Unblock scheduler threads before returning so they don't spin forever.
                    runtime_init_ready_.store(true, std::memory_order_release);
                    return -1;
                }

                dlerror();
                auto config_func = reinterpret_cast<DeviceOrchestrationConfigFunc>(dlsym(handle, config_symbol));
                const char *config_dlsym_error = dlerror();
                if (config_dlsym_error != nullptr || config_func == nullptr) {
                    LOG_ERROR(
                        "Thread %d: dlsym failed for config symbol '%s': %s", thread_idx, config_symbol,
                        config_dlsym_error ? config_dlsym_error : "NULL function pointer"
                    );
                    config_func = nullptr;
                }

                dlerror();
                auto bind_runtime_func =
                    reinterpret_cast<DeviceOrchestrationBindRuntimeFunc>(dlsym(handle, "framework_bind_runtime"));
                const char *bind_runtime_error = dlerror();
                if (bind_runtime_error != nullptr) {
                    LOG_ERROR("Thread %d: dlsym failed for framework_bind_runtime: %s", thread_idx, bind_runtime_error);
                    bind_runtime_func = nullptr;
                }

                *p_handle = handle;
                *p_func = orch_func;
                *p_bind = bind_runtime_func;
                *p_config_func = config_func;
                snprintf(p_path, 256, "%s", so_path);
                orch_so_table_[callable_id].in_use = true;
            } else {
                LOG_INFO_V0(
                    "Thread %d: Reusing cached orch SO handle=%p (callable_id=%d)", thread_idx, *p_handle, callable_id
                );
                if (*p_handle == nullptr || *p_func == nullptr) {
                    LOG_ERROR(
                        "Thread %d: reload=false but no cached SO handle/func for callable_id=%d", thread_idx,
                        callable_id
                    );
                    // Unblock scheduler threads before returning so they don't spin forever.
                    runtime_init_ready_.store(true, std::memory_order_release);
                    return -1;
                }
            }

            // Validate arg count on every run (reload or cache hit).
            if (*p_config_func != nullptr) {
                PTO2OrchestrationConfig cfg = (*p_config_func)(runtime->get_orch_args());
                LOG_INFO_V0("Thread %d: Config: expected_args=%d", thread_idx, cfg.expected_arg_count);
                if (cfg.expected_arg_count > 0) {
                    const ChipStorageTaskArgs &args_validate = runtime->get_orch_args();
                    int32_t actual_arg_count = args_validate.tensor_count() + args_validate.scalar_count();
                    if (actual_arg_count < cfg.expected_arg_count) {
                        LOG_ERROR(
                            "Thread %d: arg_count %d < expected %d", thread_idx, actual_arg_count,
                            cfg.expected_arg_count
                        );
                        // Clean up cached state so a subsequent run does a full reload.
                        if (*p_handle != nullptr) {
                            dlclose(*p_handle);
                            *p_handle = nullptr;
                        }
                        if (p_path[0] != '\0') {
                            unlink(p_path);
                            p_path[0] = '\0';
                        }
                        *p_func = nullptr;
                        *p_bind = nullptr;
                        *p_config_func = nullptr;
                        orch_so_table_[callable_id].in_use = false;
                        // Unblock scheduler threads before returning so they don't spin forever.
                        runtime_init_ready_.store(true, std::memory_order_release);
                        return -1;
                    }
                }
            } else {
                LOG_INFO_V0("Thread %d: No config function, using defaults", thread_idx);
            }

            // sm_handle / rt are bound to *this* run's memory and must be
            // (re)created every run, regardless of whether the SO itself was
            // reused above.
            const ChipStorageTaskArgs &args = runtime->get_orch_args();
            int32_t arg_count = args.tensor_count() + args.scalar_count();
            LOG_INFO_V0("Thread %d: sm_ptr=%p, arg_count=%d", thread_idx, runtime->get_gm_sm_ptr(), arg_count);
            for (int32_t i = 0; i < args.tensor_count() && i < 20; i++) {
                const ContinuousTensor &t = args.tensor(i);
                LOG_INFO_V0(
                    "Thread %d: orch_args[%d] = TENSOR(data=0x%lx, ndims=%u, dtype=%u)", thread_idx, i,
                    static_cast<uint64_t>(t.data), t.ndims, static_cast<unsigned>(t.dtype)
                );
            }
            for (int32_t i = 0; i < args.scalar_count() && (args.tensor_count() + i) < 20; i++) {
                LOG_INFO_V0(
                    "Thread %d: orch_args[%d] = SCALAR(0x%lx)", thread_idx, args.tensor_count() + i,
                    static_cast<uint64_t>(args.scalar(i))
                );
            }

            uint64_t task_window_size = PTO2_TASK_WINDOW_SIZE;
            uint64_t heap_size = PTO2_HEAP_SIZE;

            if (runtime->task_window_size > 0) {
                task_window_size = runtime->task_window_size;
            }
            if (runtime->heap_size > 0) {
                heap_size = runtime->heap_size;
            }
            uint64_t heap_sizes[PTO2_MAX_RING_DEPTH];
            for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
                heap_sizes[r] = runtime->ring_heap_sizes[r] ? runtime->ring_heap_sizes[r] : heap_size;
            }
            int32_t dep_pool_capacity = PTO2_DEP_LIST_POOL_SIZE;
            if (runtime->dep_pool_size > 0) {
                dep_pool_capacity = static_cast<int32_t>(runtime->dep_pool_size);
            }
            LOG_INFO_V0(
                "Thread %d: Ring sizes: task_window=%lu, heap=[%lu,%lu,%lu,%lu], dep_pool=%d", thread_idx,
                static_cast<uint64_t>(task_window_size), heap_sizes[0], heap_sizes[1], heap_sizes[2], heap_sizes[3],
                dep_pool_capacity
            );

            void *sm_ptr = runtime->get_gm_sm_ptr();
            void *gm_heap = runtime->get_gm_heap_ptr();

            uint64_t sm_size = PTO2SharedMemoryHandle::calculate_size(task_window_size);
            rt = runtime_create_from_sm_per_ring(
                PTO2_MODE_EXECUTE, sm_ptr, sm_size, task_window_size, gm_heap, heap_sizes, runtime_arena_,
                dep_pool_capacity
            );
            if (!rt) {
                LOG_ERROR("Thread %d: Failed to create PTO2Runtime", thread_idx);
                // Unblock scheduler threads before returning so they don't spin forever.
                runtime_init_ready_.store(true, std::memory_order_release);
                return -1;
            }

#if PTO2_PROFILING
            rt->orchestrator.l2_perf_level = get_l2_perf_level();
#endif

            // Total core counts = aic_count_ / aiv_count_ (set once at runtime init).
            rt->orchestrator.total_cluster_count = sched_ctx_.aic_count();
            rt->orchestrator.total_aiv_count = sched_ctx_.aiv_count();
            bool use_pipeline_orch = pipeline_layout_.strategy == PipelineStrategy::S2_O2_PIPELINE_ORCH;
            bool use_strategy0_submit_ctrl =
                pipeline_layout_.strategy == PipelineStrategy::S2_O2_CROSS_CLUSTER_DEBUG_STRATEGY0;
            bool use_strategy1_submit_ctrl =
                pipeline_layout_.strategy == PipelineStrategy::S2_O2_SPLIT_CTRL_STRATEGY1;
            bool use_strategy2_submit_ctrl =
                pipeline_layout_.strategy == PipelineStrategy::S2_O2_CROSS_CLUSTER_STRATEGY2;
            bool use_deferred_submit =
                (use_strategy0_submit_ctrl || use_strategy1_submit_ctrl || use_strategy2_submit_ctrl ||
                 use_pipeline_orch) &&
                sched_thread_num_ == 2 && orch_thread_num_ == 2;
            if (runtime->pipeline_defer_submit_disabled) {
                use_deferred_submit = false;
            }
            bool strategy0_signal_scheduler_drain = use_strategy0_submit_ctrl && use_deferred_submit;
            bool strategy1_signal_scheduler_drain = use_strategy1_submit_ctrl && use_deferred_submit;
            bool strategy2_signal_scheduler_drain = use_strategy2_submit_ctrl && use_deferred_submit;
            bool strategy0_compact_deferred_records = use_strategy0_submit_ctrl && use_deferred_submit;
            bool strategy1_compact_deferred_records = use_strategy1_submit_ctrl && use_deferred_submit;
            bool strategy2_compact_deferred_records = use_strategy2_submit_ctrl && use_deferred_submit;
            bool signal_scheduler_drain =
                strategy0_signal_scheduler_drain || strategy1_signal_scheduler_drain || strategy2_signal_scheduler_drain;
            bool compact_deferred_records = strategy0_compact_deferred_records || strategy1_compact_deferred_records ||
                                            strategy2_compact_deferred_records;
            rt->orchestrator.enable_submit_pipeline(
                orch_thread_num_, use_pipeline_orch, use_deferred_submit, signal_scheduler_drain,
                compact_deferred_records
            );

            // With multi-ring, slot_states are per-ring inside the scheduler.
            runtime->set_slot_states_ptr(nullptr);

            orch_args_cached_ = &args;

            // Wire scheduler context to the newly created PTO2Runtime before
            // releasing scheduler threads from runtime_init_ready_.
            sched_ctx_.bind_runtime(rt);

            runtime_init_ready_.store(true, std::memory_order_release);

            // Wait for scheduler's one-time init to complete
            sched_ctx_.wait_pto2_init_complete();

#if PTO2_PROFILING
            if (get_l2_perf_level() >= L2PerfLevel::ORCH_PHASES) {
                l2_perf_aicpu_set_orch_thread_idx(scheduler_context_idx);
            }
#endif

            // dep_gen plugs into the orchestrator thread (single-instance subsystem):
            // set the per-thread queue index and pop the initial buffer before any
            // submit_task can fire inside orch_func_.
            if (is_dep_gen_enabled()) {
                dep_gen_aicpu_set_orch_thread_idx(scheduler_context_idx);
                dep_gen_aicpu_init();
            }

#if PTO2_PROFILING
            orch_cycle_start = get_sys_cnt_aicpu();
            bind_start = orch_cycle_start;
#endif
            framework_bind_runtime(rt);
#if PTO2_PROFILING
            bind_end = get_sys_cnt_aicpu();
            p_bind_start = bind_end;
#endif
            if (*p_bind != nullptr) {
                (*p_bind)(rt);
            }
#if PTO2_PROFILING
            p_bind_end = get_sys_cnt_aicpu();
            outer_scope_begin_start = p_bind_end;
#endif
            rt_scope_begin(rt);
#if PTO2_PROFILING
            outer_scope_begin_end = get_sys_cnt_aicpu();
            p_func_start = outer_scope_begin_end;
#endif
            (*p_func)(*orch_args_cached_);
#if PTO2_PROFILING
            p_func_end = get_sys_cnt_aicpu();
            outer_scope_end_start = p_func_end;
#endif
            rt_scope_end(rt);
#if PTO2_PROFILING
            orch_active_end = get_sys_cnt_aicpu();
#endif
            rt->orchestrator.stop_submit_pipeline();
#if PTO2_PROFILING
            orch_drain_end = get_sys_cnt_aicpu();
            orch_active_cycles = orch_active_end - orch_cycle_start;
            orch_stop_cycles = orch_drain_end - orch_active_end;
            if (get_l2_perf_level() >= L2PerfLevel::ORCH_PHASES) {
                l2_perf_aicpu_record_orch_counter_phase(
                    AicpuPhaseId::ORCH_PIPELINE_COUNTERS, orch_drain_end,
                    static_cast<uint32_t>(rt->orchestrator.tasks_submitted), 0,
                    rt->orchestrator.submit_pipeline_compact_deferred_count,
                    rt->orchestrator.submit_pipeline_deferred_commit_count
                );
                l2_perf_aicpu_record_orch_counter_phase(
                    AicpuPhaseId::ORCH_O1_FOUR_STAGE_COUNTERS, orch_drain_end, 0, 0,
                    clamp_phase_counter_u32(rt->orchestrator.orch_four_stage_cycles[0]),
                    clamp_phase_counter_u32(rt->orchestrator.orch_four_stage_cycles[1])
                );
                l2_perf_aicpu_record_orch_counter_phase(
                    AicpuPhaseId::ORCH_O1_FOUR_STAGE_COUNTERS, orch_drain_end, 1, 1,
                    clamp_phase_counter_u32(rt->orchestrator.orch_four_stage_cycles[2]),
                    clamp_phase_counter_u32(rt->orchestrator.orch_four_stage_cycles[3])
                );
                l2_perf_aicpu_record_orch_counter_phase(
                    AicpuPhaseId::ORCH_O1_SUBMIT_COUNTERS_A, orch_drain_end, 0, 0,
                    clamp_phase_counter_u32(rt->orchestrator.orch_submit_layout_cycles),
                    clamp_phase_counter_u32(rt->orchestrator.orch_submit_prepare_cycles)
                );
                l2_perf_aicpu_record_orch_counter_phase(
                    AicpuPhaseId::ORCH_O1_SUBMIT_COUNTERS_A, orch_drain_end, 1, 1,
                    clamp_phase_counter_u32(rt->orchestrator.orch_submit_descriptor_cycles),
                    clamp_phase_counter_u32(rt->orchestrator.orch_submit_deferred_meta_cycles)
                );
                l2_perf_aicpu_record_orch_counter_phase(
                    AicpuPhaseId::ORCH_O1_SUBMIT_COUNTERS_B, orch_drain_end, 0, 0,
                    clamp_phase_counter_u32(rt->orchestrator.orch_submit_enqueue_cycles),
                    clamp_phase_counter_u32(rt->orchestrator.orch_submit_return_cycles)
                );
                l2_perf_aicpu_record_orch_counter_phase(
                    AicpuPhaseId::ORCH_O1_SUBMIT_COUNTERS_B, orch_drain_end, 1, 1,
                    clamp_phase_counter_u32(rt->orchestrator.orch_submit_payload_cycles),
                    clamp_phase_counter_u32(rt->orchestrator.orch_submit_wrapper_cycles)
                );
                l2_perf_aicpu_record_orch_counter_phase(
                    AicpuPhaseId::ORCH_O1_PREPARE_COUNTERS, orch_drain_end, 0, 0,
                    clamp_phase_counter_u32(rt->orchestrator.orch_prepare_alloc_cycles),
                    clamp_phase_counter_u32(rt->orchestrator.orch_prepare_slot_state_cycles)
                );
                l2_perf_aicpu_record_orch_counter_phase(
                    AicpuPhaseId::ORCH_O1_PREPARE_COUNTERS, orch_drain_end, 1, 1,
                    clamp_phase_counter_u32(rt->orchestrator.orch_prepare_ptr_cycles),
                    clamp_phase_counter_u32(rt->orchestrator.orch_prepare_prefetch_cycles)
                );
                l2_perf_aicpu_record_orch_counter_phase(
                    AicpuPhaseId::ORCH_O1_ACTIVITY_COUNTERS, orch_drain_end, 0, 0,
                    clamp_phase_counter_u32(rt->orchestrator.orch_submit_detail_count),
                    clamp_phase_counter_u32(rt->orchestrator.orch_submit_deferred_count)
                );
                l2_perf_aicpu_record_orch_counter_phase(
                    AicpuPhaseId::ORCH_O1_ACTIVITY_COUNTERS, orch_drain_end, 1, 1,
                    clamp_phase_counter_u32(rt->orchestrator.orch_submit_output_bytes_total / 1024),
                    clamp_phase_counter_u32(rt->orchestrator.orch_submit_heap_guard_sync_count)
                );
            }
#endif

            // Flush the (potentially partially-filled) DepGenBuffer so the host
            // collector can pick it up before this orchestrator thread joins.
            if (is_dep_gen_enabled()) {
                dep_gen_aicpu_flush();
            }
#if PTO2_PROFILING
            uint64_t orch_depgen_end = get_sys_cnt_aicpu();
            orch_depgen_cycles = orch_depgen_end - orch_drain_end;
            orch_end_ts = orch_depgen_end;
#endif

            // Latch task count from PTO2 shared memory to hand off to the
            // scheduler. The orchestrator's run window (start_time / end_time /
            // submit_count) is no longer published to shared memory — the
            // device LOG_INFO_V9 "orch_start=… orch_end=… orch_cost=…" line
            // below carries the same envelope info for debugging, and
            // host-side swimlane derives per-phase timing from the per-event
            // AicpuPhaseRecord[] stream that already covers everything inside
            // submit_task().
            int32_t total_tasks = 0;
            if (rt->orchestrator.sm_header) {
                for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
                    total_tasks +=
                        rt->orchestrator.sm_header->rings[r].fc.current_task_index.load(std::memory_order_acquire);
                }
            }

#if PTO2_PROFILING
            pto2_submitted_tasks = total_tasks;
            uint64_t orch_task_count_end = get_sys_cnt_aicpu();
            orch_task_count_cycles = orch_task_count_end - orch_depgen_end;
#endif

            // Signal completion to the orchestrator state machine
            rt_orchestration_done(rt);
#if PTO2_PROFILING
            uint64_t orch_done_signal_end = get_sys_cnt_aicpu();
            orch_done_signal_cycles = orch_done_signal_end - orch_task_count_end;
#endif

            sched_ctx_.on_orchestration_done(runtime, rt, scheduler_context_idx, total_tasks);
#if PTO2_PROFILING
            uint64_t orch_sched_handoff_end = get_sys_cnt_aicpu();
            orch_sched_handoff_cycles = orch_sched_handoff_end - orch_done_signal_end;
            orch_end_ts = orch_sched_handoff_end;
            pto2_completed_tasks_at_handoff = sched_ctx_.completed_tasks_count();
            LOG_INFO_V9(
                "Thread %d: orch_start=%" PRIu64 " orch_end=%" PRIu64 " orch_cost=%.3fus", thread_idx,
                static_cast<uint64_t>(orch_cycle_start), static_cast<uint64_t>(orch_end_ts),
                cycles_to_us(orch_end_ts - orch_cycle_start)
            );
            uint64_t orch_post_active_cycles = orch_end_ts > orch_cycle_start + orch_active_cycles
                                                   ? orch_end_ts - (orch_cycle_start + orch_active_cycles)
                                                   : 0;

            LOG_INFO_V9(
                "Thread %d: orch_active_start=%" PRIu64 " orch_active_end=%" PRIu64
                " orch_active_cost=%.3fus orch_drain_start=%" PRIu64 " orch_drain_end=%" PRIu64
                " orch_drain_cost=%.3fus",
                thread_idx, static_cast<uint64_t>(orch_cycle_start), static_cast<uint64_t>(orch_active_end),
                cycles_to_us(orch_active_cycles), static_cast<uint64_t>(orch_active_end),
                static_cast<uint64_t>(orch_drain_end), cycles_to_us(orch_stop_cycles)
            );
            if (!runtime->suppress_orch_diag_logs) {
                uint64_t orch_diag_start = get_sys_cnt_aicpu();
                rt->orchestrator.log_submit_pipeline_diagnostics(thread_idx);
                rt->orchestrator.log_four_stage_diagnostics(thread_idx);
                rt->orchestrator.log_submit_detail_diagnostics(thread_idx);
                uint64_t orch_diag_end = get_sys_cnt_aicpu();
                orch_diag_cycles = orch_diag_end - orch_diag_start;
                rt->orchestrator.log_active_detail_diagnostics(
                    thread_idx, orch_active_cycles, bind_end - bind_start, p_bind_end - p_bind_start,
                    outer_scope_begin_end - outer_scope_begin_start, p_func_end - p_func_start,
                    orch_active_end - outer_scope_end_start
                );
                uint64_t orch_active_detail_log_end = get_sys_cnt_aicpu();
                orch_active_detail_log_cycles = orch_active_detail_log_end - orch_diag_end;
                LOG_INFO_V9(
                    "Thread %d: orch_fixed_detail active=%.3fus post_active=%.3fus stop=%.3fus diag=%.3fus "
                    "active_detail_log=%.3fus depgen_flush=%.3fus task_count=%.3fus done_signal=%.3fus "
                    "sched_handoff=%.3fus",
                    thread_idx, cycles_to_us(orch_active_cycles), cycles_to_us(orch_post_active_cycles),
                    cycles_to_us(orch_stop_cycles), cycles_to_us(orch_diag_cycles),
                    cycles_to_us(orch_active_detail_log_cycles), cycles_to_us(orch_depgen_cycles),
                    cycles_to_us(orch_task_count_cycles), cycles_to_us(orch_done_signal_cycles),
                    cycles_to_us(orch_sched_handoff_cycles)
                );
                if (pto2_submitted_tasks >= 0) {
                    LOG_INFO_V9(
                        "PTO2 total submitted tasks = %d, already executed %d tasks", pto2_submitted_tasks,
                        pto2_completed_tasks_at_handoff
                    );
                }
            }
#endif

#if PTO2_ORCH_PROFILING
            if (!runtime->suppress_orch_diag_logs) {
                PTO2OrchProfilingData p = orchestrator_get_profiling();
                uint64_t total =
                    p.sync_cycle + p.alloc_cycle + p.args_cycle + p.lookup_cycle + p.insert_cycle + p.fanin_cycle;
                if (total == 0) total = 1;  // avoid div-by-zero
                LOG_INFO_V9(
                    "Thread %d: === Orchestrator Profiling: %" PRId64 " tasks, total=%.3fus ===", thread_idx,
                    static_cast<int64_t>(p.submit_count), cycles_to_us(total)
                );
                LOG_INFO_V9(
                    "Thread %d:   task+heap_alloc: %.3fus (%.1f%%)  work=%.3fus wait=%.3fus  atomics=%" PRIu64 "",
                    thread_idx, cycles_to_us(p.alloc_cycle), p.alloc_cycle * 100.0 / total,
                    cycles_to_us(p.alloc_cycle - p.alloc_wait_cycle), cycles_to_us(p.alloc_wait_cycle),
                    static_cast<uint64_t>(p.alloc_atomic_count)
                );
                LOG_INFO_V9(
                    "Thread %d:   sync_tensormap : %.3fus (%.1f%%)", thread_idx, cycles_to_us(p.sync_cycle),
                    p.sync_cycle * 100.0 / total
                );
                LOG_INFO_V9(
                    "Thread %d:   lookup+dep     : %.3fus (%.1f%%)", thread_idx, cycles_to_us(p.lookup_cycle),
                    p.lookup_cycle * 100.0 / total
                );
                LOG_INFO_V9(
                    "Thread %d:   tensormap_ins  : %.3fus (%.1f%%)", thread_idx, cycles_to_us(p.insert_cycle),
                    p.insert_cycle * 100.0 / total
                );
                LOG_INFO_V9(
                    "Thread %d:   param_copy     : %.3fus (%.1f%%)  atomics=%" PRIu64 "", thread_idx,
                    cycles_to_us(p.args_cycle), p.args_cycle * 100.0 / total,
                    static_cast<uint64_t>(p.args_atomic_count)
                );
                LOG_INFO_V9(
                    "Thread %d:   fanin+ready    : %.3fus (%.1f%%)  work=%.3fus wait=%.3fus", thread_idx,
                    cycles_to_us(p.fanin_cycle), p.fanin_cycle * 100.0 / total,
                    cycles_to_us(p.fanin_cycle - p.fanin_wait_cycle), cycles_to_us(p.fanin_wait_cycle)
                );
                LOG_INFO_V9(
                    "Thread %d:   avg/task       : %.3fus", thread_idx,
                    p.submit_count > 0 ? cycles_to_us(total) / p.submit_count : 0.0
                );

#if PTO2_TENSORMAP_PROFILING
                PTO2TensorMapProfilingData tp = pto2_tensormap_get_profiling();
                LOG_INFO_V9("Thread %d: === TensorMap Lookup Stats ===", thread_idx);
                LOG_INFO_V9(
                    "Thread %d:   lookups        : %" PRIu64 ", inserts: %" PRIu64 "", thread_idx,
                    static_cast<uint64_t>(tp.lookup_count), static_cast<uint64_t>(tp.insert_count)
                );
                LOG_INFO_V9(
                    "Thread %d:   chain walked   : total=%" PRIu64 ", avg=%.1f, max=%d", thread_idx,
                    static_cast<uint64_t>(tp.lookup_chain_total),
                    tp.lookup_count > 0 ? static_cast<double>(tp.lookup_chain_total) / tp.lookup_count : 0.0,
                    tp.lookup_chain_max
                );
                LOG_INFO_V9(
                    "Thread %d:   overlap checks : %" PRIu64 ", hits=%" PRIu64 " (%.1f%%)", thread_idx,
                    static_cast<uint64_t>(tp.overlap_checks), static_cast<uint64_t>(tp.overlap_hits),
                    tp.overlap_checks > 0 ? tp.overlap_hits * 100.0 / tp.overlap_checks : 0.0
                );
#endif
            }
#endif  // PTO2_ORCH_PROFILING
        }
        LOG_INFO_V0("Thread %d: Orchestrator completed", thread_idx);
    } else if (orch_stage_idx > 0) {
        LOG_INFO_V0(
            "Thread %d: Orchestrator pipeline stage %d waiting for primary orchestrator", thread_idx, orch_stage_idx
        );
        while (!runtime_init_ready_.load(std::memory_order_acquire)) {
            SPIN_WAIT_HINT();
        }
        if (rt != nullptr && orch_stage_idx > 0) {
            LOG_INFO_V0(
                "Thread %d: Orchestrator pipeline stage %d entering submit commit worker", thread_idx, orch_stage_idx
            );
#if PTO2_PROFILING
            uint64_t orch_stage_cycle_start = get_sys_cnt_aicpu();
            uint64_t orch_stage_active_cycles = rt->orchestrator.run_submit_pipeline_worker(orch_stage_idx);
#else
            (void)rt->orchestrator.run_submit_pipeline_worker(orch_stage_idx);
#endif
#if PTO2_PROFILING
            uint64_t orch_stage_cycle_end = get_sys_cnt_aicpu();
            if (rt->orchestrator.submit_pipeline_work_available.load(std::memory_order_acquire) ||
                orch_stage_active_cycles > 0) {
                LOG_INFO_V9(
                    "Thread %d: orch_stage_start=%" PRIu64 " orch_stage_end=%" PRIu64
                    " orch_stage=%d orch_stage_cost=%.3fus orch_stage_active_cost=%.3fus",
                    thread_idx, static_cast<uint64_t>(orch_stage_cycle_start),
                    static_cast<uint64_t>(orch_stage_cycle_end), orch_stage_idx,
                    cycles_to_us(orch_stage_cycle_end - orch_stage_cycle_start), cycles_to_us(orch_stage_active_cycles)
                );
            }
#endif
        } else if (orch_to_sched_) {
            LOG_INFO_V0("Thread %d: Reserved orchestrator pipeline stage entering scheduler path", thread_idx);
        } else {
            while (rt != nullptr && !sched_ctx_.is_completed() && read_pto2_runtime_status(runtime) == 0) {
                SPIN_WAIT_HINT();
            }
        }
        LOG_INFO_V0("Thread %d: Reserved orchestrator pipeline stage completed", thread_idx);
    }

    // Scheduler thread (orchestrator threads skip dispatch when orch_to_sched_ is false)
    if (!sched_ctx_.is_completed() && (scheduler_idx >= 0 || (orch_to_sched_ && orch_stage_idx >= 0))) {
        // Device orchestration: wait for the primary orchestrator to initialize the SM header
        while (!runtime_init_ready_.load(std::memory_order_acquire)) {
            SPIN_WAIT_HINT();
        }
        if (rt == nullptr) {
            LOG_ERROR("Thread %d: rt is null after orchestrator error, skipping dispatch", thread_idx);
        } else {
            sched_ctx_.bind_runtime(rt);
            int32_t dispatch_idx = (scheduler_idx >= 0) ? scheduler_idx : scheduler_context_idx;
            int32_t completed = sched_ctx_.resolve_and_dispatch(runtime, dispatch_idx);
            if (completed < 0) {
                LOG_ERROR("Thread %d: Scheduler failed with rc=%d", thread_idx, completed);
                run_rc = completed;
            } else {
                LOG_INFO_V0("Thread %d: Executed %d tasks from runtime", thread_idx, completed);
            }
        }
    }

    // Always shutdown AICore — even if sched_ctx_.completed_ was already true.
    // platform_deinit_aicore_regs is idempotent; orchestrator threads have
    // core_trackers_[thread_idx].core_num() == 0 so they skip the loop harmlessly.
    int32_t shutdown_rc = sched_ctx_.shutdown(scheduler_context_idx);
    if (shutdown_rc != 0 && run_rc == 0) {
        run_rc = shutdown_rc;
    }

    LOG_INFO_V0("Thread %d: Completed", thread_idx);

    // Check if this is the last thread to finish
    int32_t prev_finished = finished_count_.fetch_add(1, std::memory_order_acq_rel);
    if (prev_finished + 1 == thread_num_) {
        finished_.store(true, std::memory_order_release);
        // Destroy PTO2 runtime. sm_handle / rt are recreated every run so we
        // always tear them down here, but we keep the per-cid orch SO entries
        // alive for the next run's cache-hit reuse (see run() reload_so branch).
        if (rt != nullptr) {
            // Clear g_current_runtime in this DSO and in the orchestration SO before destroying rt.
            const int32_t callable_id = runtime->get_active_callable_id();
            framework_bind_runtime(nullptr);
            if (callable_id >= 0 && callable_id < MAX_REGISTERED_CALLABLE_IDS) {
                DeviceOrchestrationBindRuntimeFunc bind = orch_so_table_[callable_id].bind;
                if (bind != nullptr) {
                    bind(nullptr);
                }
            }
            runtime_destroy(rt, runtime_arena_);
            rt = nullptr;
        }
    }

    return run_rc;
}

void AicpuExecutor::deinit(Runtime *runtime) {
    // 1. Invalidate AICPU cache for Runtime address range.
    //    Next round's Host DMA (rtMemcpy) writes fresh Runtime to HBM but
    //    bypasses this cache. Invalidating now ensures next round reads from HBM.
    cache_invalidate_range(runtime, sizeof(Runtime));

    // Reset all SchedulerContext-owned state in one place.
    sched_ctx_.deinit();

    finished_count_.store(0, std::memory_order_release);
    runtime_init_ready_.store(false, std::memory_order_release);

    thread_num_ = 0;
    sched_thread_num_ = 0;
    orch_thread_num_ = 0;
    primary_orch_thread_idx_ = 0;
    orch_to_sched_ = false;
    pipeline_layout_ = {};
    for (int32_t i = 0; i < MAX_AICPU_THREADS; ++i) {
        scheduler_index_by_thread_[i] = -1;
        orchestrator_stage_by_thread_[i] = -1;
    }

    orch_args_cached_ = nullptr;
    // orch_so_table_ entries are intentionally preserved across deinit: the
    // next run reuses cached handles when register_new_callable_id() returns
    // false. The destructor releases them at process teardown.

    // Clear file-scope PTO2Runtime pointer (freed by orchestrator thread before deinit)
    rt = nullptr;

    // Clear dep_gen file-local bookkeeping. No-op when dep_gen is disabled.
    dep_gen_aicpu_finalize();

    LOG_INFO_V0("DeInit: Runtime execution state reset");

    initialized_.store(false, std::memory_order_release);
    init_done_.store(false, std::memory_order_release);
    init_failed_.store(false, std::memory_order_release);
    thread_idx_.store(0, std::memory_order_release);
    finished_.store(false, std::memory_order_release);

    LOG_INFO_V0("DeInit: AicpuExecutor reset complete");
}

// ===== Public Entry Point =====

/**
 * aicpu_execute - Main AICPU kernel execution entry point
 *
 * This is called by DynTileFwkBackendKernelServer in kernel.cpp.
 * Orchestrates the complete task runtime execution:
 * 1. Initialize executor (thread-safe, first thread only)
 * 2. Wait for initialization to complete
 * 3. Execute tasks on managed cores
 * 4. Cleanup when last thread finishes
 *
 * @param runtime Pointer to Runtime structure
 * @return 0 on success, non-zero on error
 */
extern "C" int32_t aicpu_execute(Runtime *runtime) {
    if (runtime == nullptr) {
        LOG_ERROR("%s", "Invalid argument: null Runtime pointer");
        return -1;
    }

    LOG_INFO_V0("%s", "aicpu_execute: Starting AICPU kernel execution");

    g_aicpu_executor.init(runtime);

    while (!g_aicpu_executor.init_done_.load(std::memory_order_acquire)) {
        if (g_aicpu_executor.init_failed_.load(std::memory_order_acquire)) {
            LOG_ERROR("%s", "aicpu_execute: Initialization failed, aborting execution");
            return -1;
        }
    }

    int32_t rc = g_aicpu_executor.run(runtime);
    if (rc != 0) {
        LOG_ERROR("aicpu_execute: Thread execution failed with rc=%d", rc);
    }

    int32_t runtime_rc = read_pto2_runtime_status(runtime);

    // Last thread cleans up
    if (g_aicpu_executor.finished_.load(std::memory_order_acquire)) {
        LOG_INFO_V0("aicpu_execute: Last thread finished, cleaning up");
        g_aicpu_executor.deinit(runtime);
    }

    if (runtime_rc != 0) {
        LOG_ERROR("aicpu_execute: PTO2 runtime failed with rc=%d", runtime_rc);
        return runtime_rc;
    }

    if (rc != 0) {
        return rc;
    }

    LOG_INFO_V0("%s", "aicpu_execute: Kernel execution completed successfully");
    return 0;
}
