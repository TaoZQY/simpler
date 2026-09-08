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
#pragma once

#include <cstdint>

#include "host_build_graph/graph_execution.h"
#include "host_build_graph/graph_host_state.h"
#include "worker/pipeline_contract.h"

namespace hbg {

// Host-only value snapshot, independent of the build's borrowed buffers. Sizes
// exclude caller tensor storage, code residency and CANN-owned capture packets.
struct KernelResourceRequirements {
    uint64_t gm_heap_bytes{0};
    uint64_t runtime_arena_bytes{0};  // Includes the compact SM image exactly once.
    uint64_t graph_definition_bytes{0};
    uint64_t scheduler_state_bytes{0};  // A5 upper bound; zero for the Graph fallback.

    PipelineContract pipeline_contract() const {
        return {
            PTO_PIPELINE_CONTRACT_ABI_VERSION,
            4,
            1,
            {
                {PTO_PIPELINE_GM_HEAP, PTO_PIPELINE_HOST_PER_RUN, gm_heap_bytes},
                {PTO_PIPELINE_RUNTIME_IMAGE, PTO_PIPELINE_HOST_PER_RUN, runtime_arena_bytes},
                {PTO_PIPELINE_AICPU_STREAM, PTO_PIPELINE_EXEC_HANDLE, 0},
                {PTO_PIPELINE_AICORE_STREAM, PTO_PIPELINE_EXEC_HANDLE, 0},
            },
        };
    }

    // Accounts for the separately allocated Definition and scheduler blocks too.
    bool runtime_owned_bytes(uint64_t &bytes) const {
        const PipelineContract contract = pipeline_contract();
        if (!is_valid_hbg_kernel_pipeline_contract(&contract)) return false;
        uint64_t total = gm_heap_bytes + runtime_arena_bytes;
        for (uint64_t extra : {graph_definition_bytes, scheduler_state_bytes}) {
            if (extra > UINT64_MAX - total) return false;
            total += extra;
        }
        bytes = total;
        return true;
    }
};

// Same packing as bind_graph_definitions: retained prefix plus aligned spill
// objects, each distinct Definition once regardless of its submission count.
inline bool
graph_definition_block_bytes(const GraphHostDefinitionList &definitions, uint64_t arena_used, uint64_t &bytes) {
    if (arena_used % GRAPH_DEFINITION_OBJECT_ALIGN != 0) return false;
    uint64_t total = arena_used;
    for (const GraphHostDefinition &entry : definitions.entries) {
        if (entry.bytes < sizeof(GraphDefinition) ||
            entry.bytes > UINT64_MAX - sizeof(GraphDefinitionHeader) - (GRAPH_DEFINITION_OBJECT_ALIGN - 1))
            return false;
        const uint64_t object_bytes =
            (sizeof(GraphDefinitionHeader) + entry.bytes + GRAPH_DEFINITION_OBJECT_ALIGN - 1) &
            ~(static_cast<uint64_t>(GRAPH_DEFINITION_OBJECT_ALIGN) - 1);
        if (entry.spill == nullptr) {
            if (entry.object_offset % GRAPH_DEFINITION_OBJECT_ALIGN != 0 || entry.object_offset > arena_used ||
                object_bytes > arena_used - entry.object_offset)
                return false;
        } else {
            if (entry.object_offset != GRAPH_NO_OBJECT_OFFSET || object_bytes > UINT64_MAX - total) return false;
            total += object_bytes;
        }
    }
    bytes = total;
    return true;
}

}  // namespace hbg
