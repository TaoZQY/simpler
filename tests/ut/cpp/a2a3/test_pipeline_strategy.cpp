/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. Please read the LICENSE file for details.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#include "pipeline_strategy.h"

namespace {

class ScopedEnvVar {
public:
    explicit ScopedEnvVar(const char *key) : key_(key) {
        const char *value = std::getenv(key);
        if (value != nullptr) {
            original_had_value_ = true;
            original_value_ = value;
        }
    }

    ~ScopedEnvVar() { restore(); }

    void set(const char *value) {
        ::setenv(key_.c_str(), value, 1);
    }

    void unset() {
        ::unsetenv(key_.c_str());
    }

private:
    void restore() {
        if (restored_) return;
        restored_ = true;
        if (original_had_value_) {
            ::setenv(key_.c_str(), original_value_.c_str(), 1);
        } else {
            ::unsetenv(key_.c_str());
        }
    }

    std::string key_;
    std::string original_value_;
    bool original_had_value_{false};
    bool restored_{false};
};

}  // namespace

TEST(PipelineStrategy, StrategyZeroUsesIndependentCrossClusterDebugLayout) {
    const PipelineLayout layout = resolve_pipeline_layout(0);
    EXPECT_EQ(layout.strategy, PipelineStrategy::S2_O2_CROSS_CLUSTER_DEBUG_STRATEGY0);
    EXPECT_EQ(layout.scheduler_threads, 2);
    EXPECT_EQ(layout.orchestrator_threads, 2);
    EXPECT_STREQ(layout.name, "2O2S_cross_cluster_debug_strategy0");
    EXPECT_STREQ(layout.cluster_layout, "cluster0=O0,O1;cluster1=S0,S1");
    EXPECT_EQ(layout.orchestrator_stage_by_thread[0], 0);
    EXPECT_EQ(layout.orchestrator_stage_by_thread[1], 1);
    EXPECT_EQ(layout.scheduler_index_by_thread[2], 0);
    EXPECT_EQ(layout.scheduler_index_by_thread[3], 1);
    EXPECT_EQ(layout.scheduler_index_by_thread[4], -1);
    EXPECT_EQ(layout.scheduler_index_by_thread[5], -1);
}

TEST(PipelineStrategy, StrategyOneUsesIndependentTwoSchedulerTwoOrchestratorLayout) {
    const PipelineLayout layout = resolve_pipeline_layout(1);
    EXPECT_EQ(layout.strategy, PipelineStrategy::S2_O2_SPLIT_CTRL_STRATEGY1);
    EXPECT_EQ(layout.scheduler_threads, 2);
    EXPECT_EQ(layout.orchestrator_threads, 2);
    EXPECT_STREQ(layout.name, "2S2O_split_ctrl_strategy1");
    EXPECT_STREQ(layout.cluster_layout, "cluster0=S0,S1;cluster1=O0,O1");
    EXPECT_EQ(layout.scheduler_index_by_thread[0], 0);
    EXPECT_EQ(layout.scheduler_index_by_thread[1], 1);
    EXPECT_EQ(layout.orchestrator_stage_by_thread[2], 0);
    EXPECT_EQ(layout.orchestrator_stage_by_thread[3], 1);
}

TEST(PipelineStrategy, StrategyTwoUsesCrossClusterTwoOrchestratorsAndTwoSchedulers) {
    const PipelineLayout layout = resolve_pipeline_layout(2);
    EXPECT_EQ(layout.strategy, PipelineStrategy::S2_O2_CROSS_CLUSTER_STRATEGY2);
    EXPECT_EQ(layout.scheduler_threads, 2);
    EXPECT_EQ(layout.orchestrator_threads, 2);
    EXPECT_STREQ(layout.name, "2O2S_cross_cluster_strategy2");
    EXPECT_STREQ(layout.cluster_layout, "cluster0=O0,O1;cluster1=S0,S1");
    EXPECT_EQ(layout.orchestrator_stage_by_thread[0], 0);
    EXPECT_EQ(layout.orchestrator_stage_by_thread[1], 1);
    EXPECT_EQ(layout.scheduler_index_by_thread[2], 0);
    EXPECT_EQ(layout.scheduler_index_by_thread[3], 1);
    EXPECT_EQ(layout.scheduler_index_by_thread[4], -1);
    EXPECT_EQ(layout.scheduler_index_by_thread[5], -1);
}

TEST(PipelineStrategy, StrategyFiveKeepsTwoSchedulersAndPipelinesTwoOrchestrators) {
    const PipelineLayout layout = resolve_pipeline_layout(5);
    EXPECT_EQ(layout.scheduler_threads, 2);
    EXPECT_EQ(layout.orchestrator_threads, 2);
    EXPECT_STREQ(layout.name, "2S2O_pipeline_orch");
    EXPECT_STREQ(layout.cluster_layout, "cluster0=S0,S1;cluster1=O0,O1");
    EXPECT_EQ(layout.scheduler_index_by_thread[0], 0);
    EXPECT_EQ(layout.scheduler_index_by_thread[1], 1);
    EXPECT_EQ(layout.orchestrator_stage_by_thread[2], 0);
    EXPECT_EQ(layout.orchestrator_stage_by_thread[3], 1);
}

TEST(PipelineStrategy, UnsetEnvKeepsBaselineSentinel) {
    ScopedEnvVar guard("SIMPLER_PIPELINE_STRATEGY");
    guard.unset();
    EXPECT_EQ(resolve_pipeline_strategy_with_env(-1), -1);
}

TEST(PipelineStrategy, ExplicitZeroEnvOverridesToControlLayout) {
    ScopedEnvVar guard("SIMPLER_PIPELINE_STRATEGY");
    guard.set("0");
    EXPECT_EQ(resolve_pipeline_strategy_with_env(-1), 0);
}

TEST(PipelineStrategy, EnvOverridesNegativeConfig) {
    ScopedEnvVar guard("SIMPLER_PIPELINE_STRATEGY");
    guard.set("2");
    EXPECT_EQ(resolve_pipeline_strategy_with_env(-2), 2);
}
