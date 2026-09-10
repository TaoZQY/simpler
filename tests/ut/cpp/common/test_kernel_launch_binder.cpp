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

#include "kernel_binder_test_support.h"

namespace kernel_binder_test {
TEST(KernelBinder, ThreeStreamSuccessAndSteadyStateHaveExactTrace) {
    Fixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize());
    const auto creates = f.fake.creates;
    const auto allocations = f.fake.allocations;
    ASSERT_EQ(f.launch().status, 0);
    EXPECT_EQ(f.fake.trace, success);
    for (int i = 0; i < 10; ++i) {
        f.fake.clear_trace();
        const auto result = f.launch();
        EXPECT_EQ(result.status, 0);
        EXPECT_TRUE(result.tail_recorded);
        EXPECT_EQ(f.fake.trace, (std::vector<Step>(success.begin() + 1, success.end())));
    }
    EXPECT_EQ(f.fake.queries, 0);
    EXPECT_EQ(f.fake.creates, creates);
    EXPECT_EQ(f.fake.allocations, allocations);
    EXPECT_EQ(f.fake.frees, 0);
    EXPECT_EQ(f.fake.destroys, 0);
}

TEST(KernelBinder, StreamSwitchQueriesButNeverWaitsOldTail) {
    Fixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize());
    ASSERT_EQ(f.launch().status, 0);
    f.fake.clear_trace();
    f.fake.complete = false;
    EXPECT_NE(f.launch(ptr(101)).status, 0);
    EXPECT_TRUE(f.fake.trace.empty());
    EXPECT_EQ(f.state.phase(), KernelContextPhase::ReadyEnqueued);
    f.fake.complete = true;
    f.fake.query_error = -22;
    EXPECT_EQ(f.launch(ptr(101)).status, -22);
    EXPECT_TRUE(f.fake.trace.empty());
    f.fake.query_error = 0;
    EXPECT_EQ(f.launch(ptr(101)).status, 0);
    EXPECT_EQ(f.fake.queries, 3);
    EXPECT_EQ(f.fake.trace, (std::vector<Step>(success.begin() + 1, success.end())));
    f.fake.clear_trace();
    EXPECT_EQ(f.launch(ptr(101)).status, 0);
    EXPECT_EQ(f.fake.queries, 3);
}

TEST(KernelBinder, ReadOnlyRejectionsPreserveFirstPrepareDependency) {
    for (int fault = 0; fault < 12; ++fault) {
        SCOPED_TRACE(fault);
        Fixture f;
        ASSERT_NO_FATAL_FAILURE(f.initialize());
        auto b = f.binding;
        auto packet = f.packet;
        auto gate = f.fake.gate();
        auto ops = f.fake.ops();
        void *caller = ptr(100);
        switch (fault) {
        case 0:
            b.device_id++;
            break;
        case 1:
            b.context_generation++;
            break;
        case 2:
            b.resource_schema++;
            break;
        case 3:
            f.required = 129;
            break;
        case 4:
            f.packet.generation++;
            break;
        case 5:
            f.packet.reserved0 = 1;
            break;
        case 6:
            f.fake.device++;
            break;
        case 7:
            f.fake.validation_error = -1;
            break;
        case 8:
            caller = nullptr;
            break;
        case 9:
            caller = ptr(1);
            break;
        case 10:
            gate.validate_binding = nullptr;
            break;
        case 11:
            ops.launch_aicore = nullptr;
            break;
        }
        EXPECT_NE(launch_bound_kernel(f.state, b, caller, gate, ops).status, 0);
        EXPECT_TRUE(f.fake.trace.empty());
        EXPECT_EQ(f.state.phase(), KernelContextPhase::ReadyEnqueued);
        f.required = 64;
        f.packet = packet;
        f.fake.device = 0;
        f.fake.validation_error = 0;
        EXPECT_EQ(f.launch().status, 0);
        EXPECT_EQ(f.fake.trace, success);
    }
}

TEST(KernelBinder, EveryEnqueueFailurePoisonsWithExactCompensationTrace) {
    for (int fail = 1; fail <= static_cast<int>(success.size()); ++fail) {
        SCOPED_TRACE(fail);
        Fixture f;
        ASSERT_NO_FATAL_FAILURE(f.initialize());
        f.fake.fail_at = fail;
        const auto result = f.launch();
        EXPECT_EQ(result.status, -1700 - fail);
        EXPECT_EQ(result.failed_step, success[fail - 1]);
        auto expected = std::vector<Step>(success.begin(), success.begin() + fail);
        if (fail == 6) expected.insert(expected.end(), {Cancel, Step::AicoreDone, Step::JoinAicore, Step::SerialTail});
        if (fail == 7) expected.insert(expected.end(), {Cancel, Step::JoinAicore, Step::SerialTail});
        if (fail == 8)
            expected.insert(
                expected.end(), {Cancel, Step::AicpuDone, Step::JoinAicpu, Step::JoinAicore, Step::SerialTail}
            );
        EXPECT_EQ(f.fake.trace, expected);
        EXPECT_EQ(f.state.phase(), KernelContextPhase::Poisoned);
        EXPECT_EQ(f.state.last_runtime_error(), result.status);
        EXPECT_EQ(f.state.mark_ready_enqueued(), PTO_RUNTIME_ERR_INVALID_STATE);
        f.fake.clear_trace();
        EXPECT_NE(f.launch().status, 0);
        EXPECT_TRUE(f.fake.trace.empty());
        EXPECT_EQ(f.state.close(), 0);
    }
}

TEST(KernelBinder, CompensationFailuresKeepBothErrorsAndStopAtFailure) {
    for (int primary : {6, 7, 8}) {
        const int cleanup_steps = primary == 6 ? 4 : primary == 7 ? 3 : 5;
        for (int step = 1; step <= cleanup_steps; ++step) {
            SCOPED_TRACE(primary * 100 + step);
            Fixture f;
            ASSERT_NO_FATAL_FAILURE(f.initialize());
            f.fake.fail_at = primary;
            f.fake.second_fail_at = primary + step;
            const auto result = f.launch();
            EXPECT_EQ(result.status, -1700 - primary);
            EXPECT_EQ(result.cleanup_status, -1700 - primary - step);
            EXPECT_FALSE(result.tail_recorded);
            EXPECT_EQ(f.fake.calls, primary + step);
            EXPECT_EQ(f.state.last_runtime_error(), result.status);
            EXPECT_EQ(f.state.unexpected_teardown_error(), result.cleanup_status);
        }
    }
}

TEST(KernelBinder, MissingFreezeOrPrepareRejectsWithoutEnqueue) {
    for (bool frozen : {false, true}) {
        Fixture f;
        ASSERT_NO_FATAL_FAILURE(f.initialize(frozen, !frozen));
        EXPECT_NE(f.launch().status, 0);
        EXPECT_TRUE(f.fake.trace.empty());
    }
}

TEST(KernelBinder, ConcurrentHostLaunchIsRejectedWithoutTouchingArgs) {
    Fixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize());
    std::promise<void> entered, release;
    f.fake.entered = &entered;
    f.fake.release = release.get_future().share();
    auto first = std::async(std::launch::async, [&] {
        return f.launch();
    });
    entered.get_future().wait();
    auto second = f.launch();
    EXPECT_NE(second.status, 0);
    EXPECT_FALSE(second.enqueue_started);
    release.set_value();
    EXPECT_EQ(first.get().status, 0);
    EXPECT_EQ(f.fake.validations, 1);
}

TEST(KernelBinder, NewPrepareTailIsConsumedExactlyOnce) {
    Fixture f;
    ASSERT_NO_FATAL_FAILURE(f.initialize());
    ASSERT_EQ(f.launch().status, 0);
    ASSERT_EQ(f.state.mark_ready_enqueued(), 0);
    f.fake.clear_trace();
    EXPECT_EQ(f.launch().status, 0);
    EXPECT_EQ(f.fake.trace, success);
}
}  // namespace kernel_binder_test
