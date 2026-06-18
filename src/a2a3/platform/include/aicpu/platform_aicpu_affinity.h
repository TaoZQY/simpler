#pragma once
#include <cstdint>

// Returns true if this thread should call aicpu_execute().
// Returns false if this thread should exit (dropped).
// logical_count: desired active threads (from runtime.sche_cpu_num)
// total_launched: actual threads launched (PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH)
// pipeline_strategy: resolved runtime strategy, used only by strategy-specific placement.
bool platform_aicpu_affinity_gate(int32_t logical_count, int32_t total_launched, int32_t pipeline_strategy);

// Returns a preassigned logical AICPU thread index, or -1 when the executor
// should keep its legacy atomic entry-order numbering.
int32_t platform_aicpu_affinity_logical_index();
