# Developer Scripts

Repo-local scripts that are **not** shipped in the wheel. They assume a full
source checkout and known repo layout.

End-user profiling / debug CLIs live in
[`simpler_setup/tools/`](../simpler_setup/tools/) and ship with the wheel —
invoke them via `python -m simpler_setup.tools.<name>`.

## benchmark_rounds.sh

Run the selected ST benchmark cases on hardware, parse `orch_start` /
`orch_end` / `sched_end` timestamps from the device log, and report per-round
elapsed time.

```bash
# Use defaults: one tuning case, device 0, 100 rounds
./tools/benchmark_rounds.sh

# Specify device / rounds / runtime
./tools/benchmark_rounds.sh -p a2a3 -d 4 -n 20 -r tensormap_and_ringbuffer

# Restore the full benchmark set after tuning
BENCH_CASE_SET=all ./tools/benchmark_rounds.sh -p a2a3 -r tensormap_and_ringbuffer -n 100
```

Requires `PTO2_PROFILING=1` in the runtime; device log must include the
`orch_*` / `sched_*` lines. By default this low-level script runs one selected
tuning case. Use `BENCH_CASE=<example>:<case>` for a different single case or
`BENCH_CASE_SET=all` for the full set.

When `SIMPLER_PIPELINE_STRATEGY` is set and `SIMPLER_AICPU_THREAD_NUM` is not
set, the script exports the matching thread count: strategies 0, 1, 2, and 5
use 4 threads for the 2S2O control layouts, the cross-cluster strategy2
layout, and the 2O-pipeline layout; strategies 3 and 4 use 6 threads for the
larger pipeline layouts.

For workload-gate experiments, `BENCH_BASELINE_CASES` can force selected
strategy-run cases back to the unset baseline layout. Keep this as an
experiment hook only: production strategy selection should stay under
`SIMPLER_PIPELINE_STRATEGY`, with workload-specific behavior handled inside the
selected scheduler strategy.

```bash
BENCH_BASELINE_CASES=spmd_paged_attention:Case1 \
BENCH_CASE_SET=all ./tools/benchmark_rounds.sh -p a2a3 -r tensormap_and_ringbuffer -n 100
```

The value is a comma-separated list of `example:Case` entries. Use
`example:*` or `example` to match every case in an example.

## setup_910_env_and_build.sh

Set up the 910B/910C environment and rebuild the editable package. For 910B
it sources CANN, exports `PTO_ISA_ROOT`, activates `.venv`, exports
`SIMPLER_BLOCK_DIM`, and runs `pip install --no-build-isolation -e '.[test]'`.

```bash
source ./tools/setup_910_env_and_build.sh --chip 910b --block-dim 20
```

Use `--no-build` when you only want to refresh the environment:

```bash
source ./tools/setup_910_env_and_build.sh --chip 910b --block-dim 20 --no-build
```

## run_strategy2_gate_block20_910b.sh

Run the current strategy1 and cross-cluster 2O+2S strategy2 checkpoints on
910B with the default block20 full sweep. The script sources
`setup_910_env_and_build.sh`, clears stale experiment knobs, and then runs
`run_pipeline_all.sh --block-dim 20`. It forces `STRATEGIES=1,2` and clears
`SIMPLER_AICPU_THREAD_NUM`, so `benchmark_rounds.sh` assigns 4 AICPU threads
to both strategies. Strategy2 keeps the strategy1 submit path but places O0/O1
on the 2-core AICPU cluster and S0/S1 on the 4-core cluster.

```bash
DEVICE=3 ./tools/run_strategy2_gate_block20_910b.sh

# Reuse an already-prepared shell and avoid rebuilding.
SKIP_SETUP=1 DEVICE=3 ./tools/run_strategy2_gate_block20_910b.sh
```

Useful overrides: `BLOCK_DIM=20`, `ROUNDS=100`, `PLATFORM=a2a3`, and
`RUNTIME=tensormap_and_ringbuffer`.

## run_pipeline_all.sh

One-command entry point for the default base + strategy 0 comparison across
the full benchmark set. Strategy 0 is the debug module, so this entry validates
the current debug branch against base before moving changes into other
strategies.

```bash
./tools/run_pipeline_all.sh

# Override all case configs to block_dim=20, useful for 910B validation.
./tools/run_pipeline_all.sh --block-dim 20
```

Defaults:

- base tree: current `simpler-main`, with `SIMPLER_PIPELINE_STRATEGY` unset
- pipeline tree: current `simpler-main`, with strategy values set per run
- strategies: `0`
- block_dim: case default unless `--block-dim`, `SIMPLER_BLOCK_DIM`, or `BLOCK_DIM` is set
- repeat: `1`
- command: `./tools/benchmark_rounds.sh -p a2a3 -r tensormap_and_ringbuffer -n 100`
- benchmark case set: `all`
- stale `SIMPLER_PIPELINE_DEFER_SUBMIT=0` is cleared unless
  `PRESERVE_PIPELINE_DEFER_SUBMIT=1`

`--block-dim N` exports `SIMPLER_BLOCK_DIM=N` before the base and strategy
runs. `SceneTestCase` reads that environment variable and overrides every
case's `CallConfig.block_dim`, so the final runtime launch uses the requested
block dimension.

The wrapper intentionally forces `STRATEGIES=0` and `BENCH_CASE_SET=all` by
default so stale shell variables cannot accidentally run another strategy or a
single case. To override them for an explicit experiment, set
`ALLOW_RUN_ALL_OVERRIDES=1`:

```bash
DEVICE=3 \
CMD="./tools/benchmark_rounds.sh -p a2a3 -d 3 -r tensormap_and_ringbuffer -n 100 -v" \
./tools/run_pipeline_all.sh --block-dim 20
```

For an explicit multi-strategy experiment:

```bash
ALLOW_RUN_ALL_OVERRIDES=1 REPEAT=1 STRATEGIES=0,5 \
CMD="python3 -c 'print(\"elapsed 10 sched 8 orch 5\")'" ./tools/run_pipeline_all.sh
```

By default the child command output is saved to each run directory and only
start/done progress is printed. To stream stdout/stderr to the console while
still saving logs:

```bash
STREAM_OUTPUT=1 ./tools/run_pipeline_all.sh
```

The final `Unified Case Comparison` follows the review order used for Tech1
pipeline tuning:

1. Full sweep: check all non-target cases for Total regressions.
2. Target case: compare `BaseO`, `StratO`, and `dO`, where strategy O
   completion prefers O2 Wall and base O completion uses base O1 Active.
3. Then inspect `dTot` to decide whether scheduler/device tail is hiding an
   O-side improvement.

## run_pipeline_benchmark.py

Runs a base command plus this pipeline tree across a list of
`SIMPLER_PIPELINE_STRATEGY` values, then writes `summary.csv`, `summary.md`,
and `best_strategy.json`.

```bash
python tools/run_pipeline_benchmark.py \
  --base-dir . \
  --pipeline-dir . \
  --strategies 0,5 \
  --repeat 1 \
  --stream-output \
  --cmd "./tools/benchmark_rounds.sh -p a2a3 -r tensormap_and_ringbuffer -n 100"
```

Base runs in the selected base directory with `SIMPLER_PIPELINE_STRATEGY`
unset as the 4T `O+3S` baseline. Strategy `0` runs the 4T `2O2S`
cross-cluster debug module, currently copied from strategy2 but kept on an
independent enum/name/predicate branch. Strategy `1` is the fixed 4T `2S2O`
tuning strategy, strategy `2` runs the 4T `2O2S` cross-cluster strategy with
O0/O1 on the 2-core AICPU cluster and S0/S1 on the 4-core cluster, and
strategy `5` runs the 4T `2S2O` submit-pipeline layout.

## run_pipeline_strategies.py

Runs only this tree across strategy values and writes per-run `metrics.json`
files under `results/pipeline/`.

```bash
python tools/run_pipeline_strategies.py \
  --strategies 0,5 \
  --repeat 3 \
  --cmd "python3 tests/st/a2a3/tensormap_and_ringbuffer/batch_paged_attention/test_batch_paged_attention.py -p a2a3"
```

## verify_packaging.sh

Exercises all 5 install paths × 2 entry points from a fully clean state.
CI calls this directly; see [docs/python-packaging.md](../docs/python-packaging.md).
Must run from the repo root inside an activated venv.

```bash
source .venv/bin/activate
bash tools/verify_packaging.sh
```
