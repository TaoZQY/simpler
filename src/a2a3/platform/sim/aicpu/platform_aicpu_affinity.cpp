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

#include "common/unified_log.h"

static constexpr int32_t STRATEGY0_TWO_ORCH_TWO_SCHED_CROSS_CLUSTER = 0;
static constexpr int32_t STRATEGY2_TWO_ORCH_TWO_SCHED_CROSS_CLUSTER = 2;

static std::atomic<int32_t> s_thread_counter{0};
static std::atomic<int32_t> s_cleanup_counter{0};
static thread_local int32_t s_current_logical_index = -1;

int32_t platform_aicpu_affinity_logical_index() {
    return s_current_logical_index;
}

bool platform_aicpu_affinity_gate(int32_t logical_count, int32_t total_launched, int32_t pipeline_strategy) {
    s_current_logical_index = -1;
    int32_t idx = s_thread_counter.fetch_add(1, std::memory_order_acq_rel);
    bool survive = idx < logical_count || logical_count >= total_launched;
    bool use_cross_cluster_rank = (pipeline_strategy == STRATEGY0_TWO_ORCH_TWO_SCHED_CROSS_CLUSTER ||
                                   pipeline_strategy == STRATEGY2_TWO_ORCH_TWO_SCHED_CROSS_CLUSTER) &&
                                  logical_count == 4;
    if (survive && use_cross_cluster_rank) {
        s_current_logical_index = idx;
    }

    if (!survive) {
        LOG_INFO_V0(
            "AICPU affinity gate (sim): thread idx=%d DROPPED (logical=%d, launched=%d)", idx, logical_count,
            total_launched
        );
    }

    // Last thread resets state for next invocation
    int32_t cleanup_idx = s_cleanup_counter.fetch_add(1, std::memory_order_acq_rel);
    if (cleanup_idx + 1 == total_launched) {
        s_thread_counter.store(0, std::memory_order_release);
        s_cleanup_counter.store(0, std::memory_order_release);
    }

    return survive;
}
