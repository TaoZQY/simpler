# PTO Runtime - 任务运行时执行框架

本目录是 `simpler/` 的实验副本，用于构建和执行 Ascend 设备上的任务依赖图，
并验证 A3 六核条件下不同 Orchestrator / Scheduler 切分策略的收益。

框架由 Host `.so`、AICPU `.so`、AICore `.o` 三类独立编译产物组成，
通过统一接口协同执行。

## 快速开始

```bash
# 进入实验目录
cd simpler-main

# 安装依赖。建议使用本地 venv，详见 .claude/rules/venv-isolation.md
pip install --no-build-isolation -e '.[test]'

# 运行无需硬件的仿真 vector 示例
python examples/a2a3/tensormap_and_ringbuffer/vector_example/test_vector_example.py -p a2a3sim
```

首次运行会自动拉取 PTO ISA 头文件。手动配置和排障请参考
[Getting Started](docs/getting-started.md)。

## 平台

| 平台 | 说明 | 依赖 |
| ---- | ---- | ---- |
| `a2a3` | 真实 Ascend A2/A3 硬件 | CANN toolkit、ccec、aarch64 交叉编译器 |
| `a2a3sim` | A2/A3 线程仿真 | 仅需 gcc/g++，不需要 Ascend SDK |
| `a5` | 真实 Ascend A5 硬件 | CANN toolkit、ccec、aarch64 交叉编译器 |
| `a5sim` | A5 线程仿真 | 仅需 gcc/g++，不需要 Ascend SDK |

## Runtime 类型

`src/{arch}/runtime/` 下有两类 runtime：

| Runtime | 构图位置 | 适用场景 |
| ------- | -------- | -------- |
| `host_build_graph` | Host CPU | 开发、调试 |
| `tensormap_and_ringbuffer` | AICPU 侧 | 生产和性能测试 |

架构说明见 [a2a3](src/a2a3/docs/runtimes.md) 和
[a5](src/a5/docs/runtimes.md) runtime 文档。

## 常规测试

```bash
# 仿真场景测试，不需要硬件
pytest examples tests/st --platform a2a3sim

# 硬件场景测试，需要 Ascend 设备
pytest examples tests/st --platform a2a3 --device 4-7

# Python 单元测试
pytest tests/ut -m "not requires_hardware" -v

# C++ 单元测试
cmake -B tests/ut/cpp/build -S tests/ut/cpp
cmake --build tests/ut/cpp/build
ctest --test-dir tests/ut/cpp/build --output-on-failure
```

完整测试说明见 [Testing Guide](docs/testing.md)。

## Pipeline 策略测试

`simpler-main` 用来对比 `../DOC/代码修改.md` 和 `../DOC/设计方案.md`
中定义的 A3 六核 Orchestrator / Scheduler 布局策略。

当前收敛后的策略编号：

| ID | 名称 | 布局含义 |
| -- | ---- | -------- |
| unset | base/current baseline | 当前 runtime，3S1O 基线，不设置策略环境变量 |
| 0 | 2O2S_cross_cluster_debug_strategy0 | 调测模块；当前复制 strategy2 的 2O+2S 跨 cluster 行为，但保持独立分支 |
| 1 | 2S2O_split_ctrl_strategy1 | 2 Scheduler + 2 Orchestrator，独立调优分支 |
| 2 | 2O2S_cross_cluster_strategy2 | 复用 strategy1 submit 方式，2O 放 2 核 cluster，2S 从 4 核 cluster 选取 |
| 3 | 2S4O_split_orch | 2 Scheduler + 4 Orchestrator，4O 与 2S 分 cluster |
| 4 | 2S4O_split_mixed | 2 Scheduler + 4 Orchestrator，2O 与 2S+2O 分 cluster |

当前代码已将策略编号下发到 AICPU runtime。`unset` 继续表示 3S1O
基线，策略 0-4 启用对应的 Scheduler / Orchestrator-role 线程布局：

- 策略 0：cluster0=`O0 O1`，cluster1=`S0 S1`
- 策略 1：cluster0=`S0 S1`，cluster1=`O0 O1`
- 策略 2：cluster0=`O0 O1`，cluster1=`S0 S1`
- 策略 3：cluster0=`O0 O1 O2 O3`，cluster1=`S0 S1`
- 策略 4：cluster0=`S0 S1 O2 O3`，cluster1=`O0 O1`

策略 0-4 都启用 Orchestrator submit pipeline。策略 0 是调测模块，当前从
strategy2 复制 2O+2S 跨 cluster 布局和 compact deferred submit 行为，但保留
独立的 strategy enum、布局名称和判断分支；后续修改 strategy0 或 strategy2
不会自动影响另一个策略。策略 1 使用独立的 strategy enum 和布局名称作为
2S2O 调优分支；当前策略 1 使用 compact deferred
metadata 降低 O0->O1 task batch 边界负担，让 O1 完成 deferred dependency /
fanin / wiring publication，并在 publish 后发出 S0 drain hint，使小 batch
更接近 base 的 S entry 可见性。
策略 2 保留 strategy1 的 compact deferred submit / O1 fanin /
wiring publication 逻辑，但把 role layout 改为跨 cluster 的 2O+2S：
O0/O1 落在 2 核 cluster，S0/S1 从 4 核 cluster 选取。策略 3/4 暂时保持原 4O
commit pipeline：O1/O2/O3 分别处理 descriptor 写回、fanin metadata
写回、wiring queue 发布，`scope_end` 仍走保守 flush 路径。

### 测试当前基线

不设置 `SIMPLER_PIPELINE_STRATEGY`，用于得到当前代码的 base 表现：

```bash
unset SIMPLER_PIPELINE_STRATEGY
./tools/benchmark_rounds.sh -p a2a3 -r tensormap_and_ringbuffer -n 100
```

### 测试单个策略

通过 `SIMPLER_PIPELINE_STRATEGY` 指定策略编号，例如测试策略 1：

```bash
SIMPLER_PIPELINE_STRATEGY=1 \
  ./tools/benchmark_rounds.sh -p a2a3 -r tensormap_and_ringbuffer -n 100
```

也可以在 Python 配置中设置：

```python
config = CallConfig()
config.block_dim = 24
config.aicpu_thread_num = 4
config.pipeline_strategy = 1
```

环境变量默认优先级高于 `CallConfig.pipeline_strategy`。所有 case 都应通过
`SIMPLER_PIPELINE_STRATEGY` 统一选择策略；如果某个 workload 需要特殊处理，
应在所选策略内部的调度路径中动态处理，而不是在 case 配置中屏蔽策略环境变量。

### 一次跑完基线和所有策略

默认入口会依次运行：

1. `base/current baseline`，不设置 `SIMPLER_PIPELINE_STRATEGY`
2. `strategy_0`

执行：

```bash
./tools/run_pipeline_all.sh

# 910B 常用：覆盖所有 case 的 block_dim 为 20。
./tools/run_pipeline_all.sh --block-dim 20
```

默认每个 base / strategy 只 run 1 次，每次 run 内部执行 100 个 round，
并默认运行全量 benchmark case，便于验证调测策略 0 相对 base 的整体效果。

默认情况下，脚本会把每轮子命令的 stdout / stderr 保存到日志文件，
终端只打印当前运行到哪个 base / strategy，以及该轮结束状态。这样做是为了
避免多个策略输出混在一起。如果希望像单独运行一样实时打屏，可以使用：

```bash
STREAM_OUTPUT=1 ./tools/run_pipeline_all.sh
```

结果输出到：

```text
results/pipeline/<run-id>/
```

目录内包含：

- `summary.csv`
- `summary.md`
- `best_strategy.json`
- `run_config.json`
- 每个 base / strategy 的 `stdout.log`、`stderr.log`、`metrics.json`

### 覆盖默认测试命令

默认命令是：

```bash
./tools/benchmark_rounds.sh -p a2a3 -r tensormap_and_ringbuffer -n 100
```

`run_pipeline_all.sh` 会默认设置 `BENCH_CASE_SET=all`，因此上述命令在
pipeline 对比入口中会运行全量 benchmark case。若单独运行
`benchmark_rounds.sh`，该命令默认只跑 `paged_attention_unroll:Case1`；
如果要单独恢复全量 case 验证，使用：

```bash
BENCH_CASE_SET=all ./tools/benchmark_rounds.sh -p a2a3 -r tensormap_and_ringbuffer -n 100
```

策略验证时如果设置了 `SIMPLER_PIPELINE_STRATEGY`，脚本会在未显式设置
`SIMPLER_AICPU_THREAD_NUM` 时自动导出匹配线程数：策略 0、1、2 和 5
使用 4 线程 debug strategy0 / strategy1 / cross-cluster strategy2 /
2O pipeline layout；策略 3 和 4 使用 6 线程 pipeline layout。

`run_pipeline_all.sh` 默认固定全量 base vs strategy0。为了避免 shell 中残留的
`STRATEGIES` 或 `BENCH_CASE_SET` 误导测试，只有设置
`ALLOW_RUN_ALL_OVERRIDES=1` 时才允许覆盖策略列表和 case set。可以用环境变量覆盖
重复次数、策略列表和实际测试命令：

`--block-dim N` 会导出 `SIMPLER_BLOCK_DIM=N`，并通过 `SceneTestCase`
覆盖每个 case 的 `CallConfig.block_dim`；结果目录的 `run_config.json`
会记录实际环境值，便于确认 910B 是否按预期的 block_dim 运行。

```bash
ALLOW_RUN_ALL_OVERRIDES=1 REPEAT=1 STRATEGIES=0,1,2 \
CMD="python3 -c 'print(\"elapsed 10 sched 8 orch 5\")'" \
./tools/run_pipeline_all.sh
```

这些环境变量可以和 `STREAM_OUTPUT=1` 一起使用：

```bash
ALLOW_RUN_ALL_OVERRIDES=1 STREAM_OUTPUT=1 REPEAT=1 STRATEGIES=0,1,2 \
  ./tools/run_pipeline_all.sh
```

更多工具参数见 [tools/README.md](tools/README.md)。

## 环境配置

```bash
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
```

## 主要文档

| 文档 | 内容 |
| ---- | ---- |
| [Chip-Level Architecture](docs/chip-level-arch.md) | L2 单芯片三程序模型、API 层次、握手协议 |
| [Hierarchical Level Runtime](docs/hierarchical_level_runtime.md) | L0-L6 层级模型和组件组合 |
| [Task Flow](docs/task-flow.md) | Callable / TaskArgs / CallConfig 到 IWorker 的端到端流程 |
| [Orchestrator](docs/orchestrator.md) | DAG 提交流程、TensorMap、Scope、Ring、任务状态机 |
| [Scheduler](docs/scheduler.md) | wiring / ready / completion 队列和调度循环 |
| [Worker Manager](docs/worker-manager.md) | Worker 池、WorkerThread、THREAD / PROCESS 模式 |
| [Getting Started](docs/getting-started.md) | 安装、前置依赖、构建和排障 |
| [Developer Guide](docs/developer-guide.md) | 目录结构、所有权和开发约定 |
| [Testing Guide](docs/testing.md) | CI 流水线、测试类型和新增测试方法 |

### 分架构文档

| 文档 | a2a3 | a5 |
| ---- | ---- | -- |
| Runtimes | [a2a3/docs/runtimes.md](src/a2a3/docs/runtimes.md) | [a5/docs/runtimes.md](src/a5/docs/runtimes.md) |
| Platform | [a2a3/docs/platform.md](src/a2a3/docs/platform.md) | [a5/docs/platform.md](src/a5/docs/platform.md) |

## License

本项目使用 **CANN Open Software License Agreement Version 2.0**。
完整协议见 [LICENSE](LICENSE)。

## 代码位置

- [src/a2a3/platform/](src/a2a3/platform/)：a2a3 platform 实现
- [src/a2a3/runtime/](src/a2a3/runtime/)：a2a3 runtime 实现
- [examples/a2a3/](examples/a2a3/)：a2a3 示例
- [simpler_setup/](simpler_setup/)：SceneTestCase、runtime builder、kernel compiler
- [python/](python/)：Python 绑定和用户侧 API
