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
#include "host/kernel_launch_binder.h"

#include "kernel_launch_sequence.h"

KernelLaunchResult launch_bound_kernel(
    KernelExecutionState &state, const KernelInvocationBinding &binding, void *caller_stream,
    const KernelLaunchGateOps &gate, const KernelLaunchOps &ops
) {
    KernelLaunchResult rejected;
    rejected.status = PTO_RUNTIME_ERR_INVALID_STATE;
    std::unique_lock lock(state.mutex_, std::try_to_lock);
    if (!lock.owns_lock() || state.phase_ != KernelContextPhase::ReadyEnqueued ||
        binding.device_id != state.device_id_ || binding.context_generation != state.context_generation_)
        return rejected;
    if (!ops.valid() || !gate.validate_binding || !gate.query_tail) {
        rejected.status = PTO_RUNTIME_ERR_INTERNAL;
        return rejected;
    }
    KernelLaunchHandles handles{
        caller_stream,
        state.streams_[static_cast<size_t>(KernelStreamKind::Aicpu)],
        state.streams_[static_cast<size_t>(KernelStreamKind::Aicore)],
        state.events_[static_cast<size_t>(KernelEventKind::PrepareTail)],
        state.events_[static_cast<size_t>(KernelEventKind::Start)],
        state.events_[static_cast<size_t>(KernelEventKind::AicoreDone)],
        state.events_[static_cast<size_t>(KernelEventKind::AicpuDone)],
        state.events_[static_cast<size_t>(KernelEventKind::SerialTail)],
        state.prepare_tail_pending_
    };
    if (!handles.valid()) return rejected;
    int current_device = -1;
    rejected.status = state.ops_.get_current_device(state.ops_.context, &current_device);
    if (rejected.status != 0) return rejected;
    if (current_device != binding.device_id) {
        rejected.status = PTO_RUNTIME_ERR_INVALID_STATE;
        return rejected;
    }
    KernelResourceBinding resources;
    rejected.status =
        state.resources_.bind(binding.resource_schema, binding.required_resources, binding.resource_count, resources);
    if (rejected.status != 0) return rejected;
    SimplerKernelInvocationHeader invocation{};
    if (simpler::kernel::validate_invocation_header(binding.packet, binding.callable, &invocation) !=
        simpler::kernel::InvocationStatus::Ok) {
        rejected.status = PTO_RUNTIME_ERR_INTERNAL;
        return rejected;
    }
    rejected.status = gate.validate_binding(gate.context, binding, resources);
    if (rejected.status != 0) return rejected;
    if (state.last_caller_identity_ != 0 && state.last_caller_identity_ != reinterpret_cast<uintptr_t>(caller_stream)) {
        bool complete = false;
        rejected.failed_step = KernelLaunchStep::QueryTail;
        rejected.status = gate.query_tail(gate.context, handles.serial_tail, &complete);
        if (rejected.status != 0) return rejected;
        if (!complete) {
            rejected.status = PTO_RUNTIME_ERR_INVALID_STATE;
            return rejected;
        }
    }
    const auto result = enqueue_kernel_launch_sequence(ops, handles);
    if (result.status != 0) {
        if (result.enqueue_started) {
            state.phase_ = KernelContextPhase::Poisoned;
            if (state.last_runtime_error_ == 0) state.last_runtime_error_ = result.status;
            if (state.unexpected_teardown_error_ == 0) state.unexpected_teardown_error_ = result.cleanup_status;
        }
        return result;
    }
    state.last_caller_identity_ = reinterpret_cast<uintptr_t>(caller_stream);
    state.prepare_tail_pending_ = false;
    return result;
}
