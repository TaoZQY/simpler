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
#include "aicpu/platform_aicpu_affinity.h"

#include <atomic>
#include <cstdint>
#ifdef __linux__
#include <sched.h>
#endif

#include "common/unified_log.h"

static constexpr int32_t AICPU_CORES_PER_CHIP = 8;
static constexpr int32_t MAX_CLUSTERS = 2;
static constexpr int32_t CPUS_PER_CLUSTER = 4;
static constexpr int32_t MAX_GATE_THREADS = 8;
static constexpr int32_t STRATEGY0_TWO_ORCH_TWO_SCHED_CROSS_CLUSTER = 0;
static constexpr int32_t STRATEGY2_TWO_ORCH_TWO_SCHED_CROSS_CLUSTER = 2;

static std::atomic<uint64_t> s_cpumask{0};
static std::atomic<int32_t> s_reported{0};
static std::atomic<int32_t> s_gate_init{0};
static std::atomic<int32_t> s_gate_ready{0};

static int32_t s_thread_cpu[MAX_GATE_THREADS];
static bool s_thread_survive[MAX_GATE_THREADS];
static int32_t s_thread_logical_index[MAX_GATE_THREADS];
static thread_local int32_t s_current_logical_index = -1;

struct ClusterInfo {
    int32_t count{0};
    int32_t tids[MAX_GATE_THREADS];
};

static inline int32_t popcount64(uint64_t v) { return __builtin_popcountll(static_cast<unsigned long long>(v)); }

static void sort_cluster_by_cpu(ClusterInfo &info) {
    for (int32_t i = 1; i < info.count; ++i) {
        int32_t tid = info.tids[i];
        int32_t cpu = s_thread_cpu[tid];
        int32_t j = i - 1;
        while (j >= 0 && s_thread_cpu[info.tids[j]] > cpu) {
            info.tids[j + 1] = info.tids[j];
            --j;
        }
        info.tids[j + 1] = tid;
    }
}

int32_t platform_aicpu_affinity_logical_index() {
    return s_current_logical_index;
}

bool platform_aicpu_affinity_gate(int32_t logical_count, int32_t total_launched, int32_t pipeline_strategy) {
    s_current_logical_index = -1;
    bool use_strategy0_rank = pipeline_strategy == STRATEGY0_TWO_ORCH_TWO_SCHED_CROSS_CLUSTER &&
                              logical_count == 4 && total_launched == 6;
    bool use_strategy2_rank = pipeline_strategy == STRATEGY2_TWO_ORCH_TWO_SCHED_CROSS_CLUSTER &&
                              logical_count == 4 && total_launched == 6;
    bool use_cross_cluster_rank = use_strategy0_rank || use_strategy2_rank;
    if (logical_count >= total_launched && !use_cross_cluster_rank) {
        return true;
    }

    // Assign thread index
    int32_t idx = s_reported.fetch_add(1, std::memory_order_acq_rel);

    // Report CPU
#if defined(__aarch64__)
    int32_t cpu = sched_getcpu();
#elif defined(__x86_64__)
    int32_t cpu = sched_getcpu();
#else
    int32_t cpu = -1;
#endif

    int32_t normalized_cpu = -1;
    if (cpu >= 0) {
        if (cpu < 63) {
            s_cpumask.fetch_or(1ULL << cpu, std::memory_order_release);
        }
        normalized_cpu = cpu % AICPU_CORES_PER_CHIP;
    }
    if (idx < MAX_GATE_THREADS) {
        s_thread_cpu[idx] = normalized_cpu;
    }

    // Barrier: wait until all total_launched threads have reported
    while (popcount64(s_cpumask.load(std::memory_order_acquire)) < total_launched &&
           s_reported.load(std::memory_order_acquire) < total_launched) {}

    // CAS winner does cluster classification
    int32_t expected = 0;
    if (s_gate_init.compare_exchange_strong(expected, 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
        // Initialize survive flags
        for (int32_t i = 0; i < total_launched; ++i) {
            s_thread_survive[i] = false;
            s_thread_logical_index[i] = -1;
        }

        ClusterInfo clusters[MAX_CLUSTERS];

        for (int32_t tid = 0; tid < total_launched; ++tid) {
            int32_t c = s_thread_cpu[tid];
            if (c < 0) continue;
            int32_t cluster_id = c / CPUS_PER_CLUSTER;
            if (cluster_id < 0 || cluster_id >= MAX_CLUSTERS) continue;
            ClusterInfo &info = clusters[cluster_id];
            if (info.count < MAX_GATE_THREADS) info.tids[info.count++] = tid;
        }

        int32_t major_id = (clusters[0].count >= clusters[1].count) ? 0 : 1;
        int32_t minor_id = 1 - major_id;
        int32_t major_cnt = clusters[major_id].count;
        int32_t minor_cnt = clusters[minor_id].count;
        sort_cluster_by_cpu(clusters[major_id]);
        sort_cluster_by_cpu(clusters[minor_id]);

        LOG_INFO_V0(
            "AICPU affinity gate: major=%d(cnt=%d) minor=%d(cnt=%d) logical=%d", major_id, major_cnt, minor_id,
            minor_cnt, logical_count
        );

        if (use_cross_cluster_rank && minor_cnt == 2 && major_cnt >= 2) {
            for (int32_t i = 0; i < clusters[minor_id].count; ++i) {
                int32_t tid = clusters[minor_id].tids[i];
                s_thread_survive[tid] = true;
                s_thread_logical_index[tid] = i;
            }
            for (int32_t i = 0; i < 2; ++i) {
                int32_t tid = clusters[major_id].tids[i];
                s_thread_survive[tid] = true;
                s_thread_logical_index[tid] = i + clusters[minor_id].count;
            }
            LOG_INFO_V0(
                "AICPU affinity gate: strategy%d cross-cluster placement O cluster=%d S cluster=%d",
                pipeline_strategy, minor_id, major_id
            );
        } else if (use_cross_cluster_rank) {
            LOG_WARN(
                "AICPU affinity gate: strategy%d expected O cluster with 2 threads and S cluster with >=2 threads, "
                "got major=%d minor=%d; "
                "falling back to index rank",
                pipeline_strategy, major_cnt, minor_cnt
            );
            for (int32_t i = 0; i < logical_count && i < total_launched; ++i) {
                s_thread_survive[i] = true;
                s_thread_logical_index[i] = i;
            }
        } else if (major_cnt == logical_count && minor_cnt == (total_launched - logical_count)) {
            // Expected topology: major cluster threads survive
            for (int32_t i = 0; i < clusters[major_id].count; ++i) {
                s_thread_survive[clusters[major_id].tids[i]] = true;
            }
        } else {
            // Unexpected topology: fall back to first logical_count threads
            LOG_WARN(
                "AICPU affinity gate: unexpected topology (major=%d minor=%d), "
                "falling back to index-based cutoff",
                major_cnt, minor_cnt
            );
            for (int32_t i = 0; i < logical_count && i < total_launched; ++i) {
                s_thread_survive[i] = true;
            }
        }

        s_gate_ready.store(1, std::memory_order_release);
    }

    // Wait for classification to complete
    while (s_gate_ready.load(std::memory_order_acquire) == 0) {}

    bool survive = (idx < total_launched) ? s_thread_survive[idx] : false;
    if (survive && idx < total_launched) {
        s_current_logical_index = s_thread_logical_index[idx];
    }

    // Last thread resets state for next invocation
    int32_t finished = s_reported.load(std::memory_order_acquire);
    (void)finished;
    // Reset is deferred: the statics persist but are re-initialized by the CAS winner
    // on next call. We reset the atomics after all threads have read their result.
    // Use a second atomic counter for cleanup.
    static std::atomic<int32_t> s_cleanup{0};
    int32_t cleanup_idx = s_cleanup.fetch_add(1, std::memory_order_acq_rel);
    if (cleanup_idx + 1 == total_launched) {
        s_cpumask.store(0, std::memory_order_release);
        s_reported.store(0, std::memory_order_release);
        s_gate_init.store(0, std::memory_order_release);
        s_gate_ready.store(0, std::memory_order_release);
        s_cleanup.store(0, std::memory_order_release);
    }

    if (!survive) {
        LOG_INFO_V0("AICPU affinity gate: thread idx=%d cpu=%d DROPPED", idx, normalized_cpu);
    } else {
        LOG_INFO_V0(
            "AICPU affinity gate: thread idx=%d cpu=%d logical=%d ACTIVE", idx, normalized_cpu,
            s_current_logical_index
        );
    }

    return survive;
}
