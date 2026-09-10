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

#include "host/kernel_execution_state.h"
#include "task_interface/kernel_invocation_validation.h"

struct KernelInvocationPlaceholder {
    uint32_t address_offset;
    uint32_t data_offset;
};

// Borrowed under the callable owner's submission lease. The packet and resolved
// handles outlive enqueue; device/captured ownership survives Host return.
struct KernelInvocationBinding {
    int device_id{-1};
    uint64_t context_generation{0};
    uint64_t resource_schema{0};
    const uint64_t *required_resources{nullptr};
    size_t resource_count{0};
    simpler::kernel::ByteSpan packet;
    simpler::kernel::PreparedInvocationView callable{};
    const KernelInvocationPlaceholder *placeholders{nullptr};
    size_t placeholder_count{0};
};

// Read-only preflight operations, separate from the six enqueue operations.
// Validation covers runtime-specific packet binding and prepared launch handles.
// The callable owner holds its registration lease throughout both operations.
struct KernelLaunchGateOps {
    void *context{nullptr};
    int (*validate_binding)(void *, const KernelInvocationBinding &, const KernelResourceBinding &) noexcept {nullptr};
    int (*query_tail)(void *, void *event, bool *complete) noexcept {nullptr};
};

enum class KernelLaunchStep : uint8_t {
    Validate,
    QueryTail,
    PrepareWait,
    Clear,
    Start,
    AicoreWait,
    AicoreLaunch,
    AicoreDone,
    AicpuWait,
    AicpuLaunch,
    AicpuDone,
    JoinAicpu,
    JoinAicore,
    SerialTail
};

struct KernelLaunchResult {
    int status{0};
    int cleanup_status{0};
    KernelLaunchStep failed_step{KernelLaunchStep::Validate};
    bool enqueue_started{false};
    bool tail_recorded{false};
};

// Exactly one enqueue composition after read-only preflight and stream-switch
// admission. Concurrent Host submissions fail rather than racing mutable args.
KernelLaunchResult launch_bound_kernel(
    KernelExecutionState &state, const KernelInvocationBinding &binding, void *caller_stream,
    const KernelLaunchGateOps &gate, const KernelLaunchOps &ops
);
