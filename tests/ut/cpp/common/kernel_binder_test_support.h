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

#include <gtest/gtest.h>

#include <cstdlib>
#include <future>
#include <vector>

#include "host/kernel_launch_binder.h"
#include "kernel_launch_sequence.h"

namespace kernel_binder_test {
using Step = KernelLaunchStep;
constexpr Step Cancel = static_cast<Step>(100);
inline void *ptr(uintptr_t n) { return reinterpret_cast<void *>(n); }

struct Fake {
    int device{0};
    uintptr_t handle{1};
    int allocations{0}, frees{0}, creates{0}, destroys{0}, queries{0}, validations{0};
    int calls{0}, fail_at{0}, second_fail_at{0}, query_error{0}, validation_error{0};
    bool complete{true};
    std::vector<Step> trace;
    std::promise<void> *entered{nullptr};
    std::shared_future<void> release;
    Fake() { trace.reserve(128); }
    int append(Step step) noexcept {
        trace.push_back(step);
        ++calls;
        return calls == fail_at || calls == second_fail_at ? -1700 - calls : 0;
    }
    KernelContextOps context_ops() {
        return {
            this,
            [](void *p, int *id) {
                *id = static_cast<Fake *>(p)->device;
                return 0;
            },
            [](void *p, KernelStreamKind, void **s) {
                auto &f = *static_cast<Fake *>(p);
                ++f.creates;
                *s = ptr(f.handle++);
                return 0;
            },
            [](void *p, KernelStreamKind, void *) {
                ++static_cast<Fake *>(p)->destroys;
                return 0;
            },
            [](void *p, void **e) {
                auto &f = *static_cast<Fake *>(p);
                ++f.creates;
                *e = ptr(f.handle++);
                return 0;
            },
            [](void *p, void *) {
                ++static_cast<Fake *>(p)->destroys;
                return 0;
            }
        };
    }
    KernelResourceOps resource_ops() {
        return {
            this,
            [](void *p, size_t bytes) {
                ++static_cast<Fake *>(p)->allocations;
                return std::malloc(bytes);
            },
            [](void *p, void *address) {
                ++static_cast<Fake *>(p)->frees;
                std::free(address);
                return 0;
            }
        };
    }
    KernelLaunchGateOps gate() {
        return {
            this,
            [](void *p, const KernelInvocationBinding &, const KernelResourceBinding &) noexcept {
                auto &f = *static_cast<Fake *>(p);
                ++f.validations;
                if (f.entered) {
                    f.entered->set_value();
                    f.release.wait();
                }
                return f.validation_error;
            },
            [](void *p, void *, bool *complete) noexcept {
                auto &f = *static_cast<Fake *>(p);
                ++f.queries;
                *complete = f.complete;
                return f.query_error;
            }
        };
    }
    KernelLaunchOps ops() {
        return {
            this,
            [](void *p, void *stream, void *event) noexcept {
                auto &f = *static_cast<Fake *>(p);
                const auto step = event == ptr(3) ? Step::PrepareWait :
                                  event == ptr(4) ? (stream == ptr(2) ? Step::AicoreWait : Step::AicpuWait) :
                                  event == ptr(6) ? Step::JoinAicpu :
                                                    Step::JoinAicore;
                return f.append(step);
            },
            [](void *p, void *) noexcept {
                return static_cast<Fake *>(p)->append(Step::Clear);
            },
            [](void *p, void *event, void *) noexcept {
                return static_cast<Fake *>(p)->append(
                    event == ptr(4) ? Step::Start :
                    event == ptr(5) ? Step::AicoreDone :
                    event == ptr(6) ? Step::AicpuDone :
                                      Step::SerialTail
                );
            },
            [](void *p, void *) noexcept {
                return static_cast<Fake *>(p)->append(Step::AicpuLaunch);
            },
            [](void *p, void *) noexcept {
                return static_cast<Fake *>(p)->append(Step::AicoreLaunch);
            },
            [](void *p, void *) noexcept {
                return static_cast<Fake *>(p)->append(Cancel);
            }
        };
    }
    void clear_trace() {
        trace.clear();
        calls = 0;
    }
};

const std::vector<Step> success{Step::PrepareWait,  Step::Clear,      Step::Start,      Step::AicoreWait,
                                Step::AicoreLaunch, Step::AicoreDone, Step::AicpuWait,  Step::AicpuLaunch,
                                Step::AicpuDone,    Step::JoinAicpu,  Step::JoinAicore, Step::SerialTail};

struct Fixture {
    Fake fake;
    KernelExecutionState state;
    SimplerKernelInvocationHeader packet{};
    uint64_t required{64};
    KernelInvocationBinding binding;
    void initialize(bool frozen = true, bool ready = true) {
        ASSERT_EQ(state.initialize(0, fake.context_ops(), 23), 0);
        KernelResourceLayout layout{
            91,
            {PTO_PIPELINE_CONTRACT_ABI_VERSION,
             3,
             1,
             {{PTO_PIPELINE_GM_HEAP, PTO_PIPELINE_DEVICE_SCRATCH, 128},
              {PTO_PIPELINE_AICPU_STREAM, PTO_PIPELINE_EXEC_HANDLE, 0},
              {PTO_PIPELINE_AICORE_STREAM, PTO_PIPELINE_EXEC_HANDLE, 0}}},
            {{PTO_PIPELINE_GM_HEAP, 0, 128}}
        };
        ASSERT_EQ(state.prepare_resources(layout, fake.resource_ops()), 0);
        if (frozen) ASSERT_EQ(state.freeze_resources(), 0);
        if (ready) ASSERT_EQ(state.mark_ready_enqueued(), 0);
        packet.abi_version = SIMPLER_KERNEL_INVOCATION_ABI_VERSION;
        packet.header_bytes = sizeof(packet);
        packet.mode = SIMPLER_MODE_KERNEL;
        packet.callable_id = 2;
        packet.generation = 7;
        binding = {0, 23, 91, &required, 1, {reinterpret_cast<const uint8_t *>(&packet), sizeof(packet)}, {2, 0, 0, 7}};
    }
    KernelLaunchResult launch(void *caller = ptr(100)) {
        return launch_bound_kernel(state, binding, caller, fake.gate(), fake.ops());
    }
    ~Fixture() { state.close(); }
};

}  // namespace kernel_binder_test
