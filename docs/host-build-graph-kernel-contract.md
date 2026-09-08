# HBG kernel resource declarations

Reference: [K1 interface freeze](https://icc.gt.tc/vllm-pto#k1-freeze).
The declarations below are internal host C++ interfaces. They do not change
K1's C entry points, context-control POD or invocation wire layout. Shared
validation and hook signatures align with
[TMR resource declarations at 317f5597](https://github.com/Leaf-Salix/simpler/commit/317f55971ce092244c2f370ba17899260092adc6).
That branch carries a different K1 wire revision; only contract changes are
adopted here. Execution modes live in the shared neutral `execution_mode.h`.

## Query and ownership

After `hbg::build_graph` succeeds, call
`hbg::get_kernel_resource_requirements(build, layout, requirements)` before
upload. Supply the same runtime layout used to initialize the build context.
The query allocates no device memory, binds no addresses and performs no H2D.
It can enumerate host Definition records, so call it outside capture.

`KernelResourceRequirements` is a value snapshot: it remains valid after the
build buffers are reused. The query builds a call-local candidate and
publishes it only on success; errors leave the caller's output unchanged. An unready/failed build returns `INVALID_STATE`;
invalid layout/zero required sizes return `INTERNAL`; size overflow or an
unrepresentable Definition block returns `CAPACITY_EXCEEDED`.

`requirements.pipeline_contract()` returns the sized kernel declaration by
value. The existing no-argument C `get_pipeline_contract()` remains the static
program declaration. There is no mutable global size table or new public C
query symbol. The context owner must explicitly consume the new snapshot;
`ChipWorker`'s program loader consumes the original declaration and checks
both arena and stream topology before creating a device context.

The common internal `build_kernel_pipeline_contract_impl(config, out)` hook is
also declared with the TMR signature and output-preservation contract. HBG
returns `UNSUPPORTED` there: a CallConfig cannot determine its graph-dependent
heap and image sizes. Exact HBG requirements come from the post-build query;
it does not fabricate configuration-only sizes to enable kernel init. The
common platform init wiring and TMR sizing implementation belong to the TMR
change and are not imported into this HBG branch.

## Resource accounting

| Resource | Declaration | Bytes per copy |
| -------- | ----------- | -------------- |
| GM heap | `GM_HEAP`, `HOST_PER_RUN` | Build-measured heap, including Graph execution storage, rounded to the existing arena alignment |
| Runtime arena | `RUNTIME_IMAGE`, `HOST_PER_RUN` | `layout.off_copied_end + build.image_bytes`, including the device prefix and compact SM tail |
| AICPU stream | `AICPU_STREAM`, `EXEC_HANDLE` | 0; dedicated, non-hidden, distinct from caller stream |
| AICore stream | `AICORE_STREAM`, `EXEC_HANDLE` | 0; hidden, distinct from caller stream |

Kernel depth is one. This is one serialized execution slot; it does not describe
CANN's separate argument snapshots for different captured nodes. No caller
stream is owned or declared. HBG omits `GM_SM` because it is part of the arena.
In the shared contract, every declared arena must have nonzero usable bytes.
An unused `GM_SM` is omitted, not declared with zero bytes. This is distinct
from passing `gm_sm_size=0` to the existing allocator (still valid for HBG).
`TASK_ARGS`, when declared by TMR, has zero bytes: this reserved field does
not encode task-argument size limits. HBG does not declare it.

Two allocations outside those pooled regions are reported separately:

- `graph_definition_bytes`: retained used prefix plus aligned spill objects,
  including each object's `GraphDefinitionHeader`. Shared Definitions are
  counted once per distinct Definition, not once per Graph submission.
- `scheduler_state_bytes`: zero on A2/A3 and A5 Graph fallback; otherwise an A5
  resident-scheduler allocation upper bound from its existing layout planner,
  including base-alignment slack. A shape choosing the non-Graph legacy fallback
  can use less than this bound.

`runtime_owned_bytes()` sums both pooled regions and both separate blocks with
checked arithmetic. Do not account for only `pipeline_contract()` when sizing
all runtime-owned storage. This total is a requirement/bound, not a measurement
of retained high-water allocations or total process HBM. Caller tensors, code
residency and CANN capture packets remain separate accounting categories owned
by the framework/platform and cannot be measured from a Host graph alone.

The context owner can retain snapshots when collecting callable requirements,
aggregate the reused execution slot's per-region high-water marks, and compare
new requirements against frozen capacities. Allocation, aggregation and FREEZE
enforcement are owner responsibilities; this interface performs none of them.
In particular, K1's zero CONFIGURE capacity means a default intent, not the
zero-size meaning of a measured kernel declaration. K1 does not yet provide
capacity fields for the separate Definition or scheduler blocks.

## Admission rules

Validation uses the same layers as TMR:

1. `is_valid_pipeline_contract(contract, mode)` checks ABI, known kinds/classes,
   supported depth and byte rules. Program sizes remain zero; in kernel mode,
   declared arena sizes are positive and stream/TASK_ARGS sizes remain zero.
   This structural check does not enforce a runtime's resource set.
2. `has_serviceable_arena_topology` checks duplicate arenas and consistent copy
   counts. `has_serviceable_stream_topology` requires exactly one AICPU and one
   AICore declaration, both `EXEC_HANDLE`.
3. `is_valid_hbg_kernel_pipeline_contract` combines those checks with HBG's
   exact four-resource set, depth one, `HOST_PER_RUN` memory classification and
   pooled-byte overflow rejection. The peer `is_valid_tmr_kernel_pipeline_contract`
   keeps TMR's six-resource set and device-scratch classifications.

`runtime_owned_bytes()` additionally checks the sum with Definition and A5
scheduler storage. All size helpers publish output only after successful
validation. Concurrent queries use distinct caller-owned output objects and
read a stable completed build whose buffers remain leased.

A resource declaration is not a capability claim: kernel support stays disabled
in K1's stubs. These helpers neither create streams nor enable capture/replay.

## Host tensor-data requirement semantics

The independent orchestration requirements metadata describes generated Host
behavior, not whether a tensor argument exists or carries a device address.
The Host tensor-data capability bit has these semantics for the gate producer
and consumer:

| Host orchestration operation | Requires Host tensor-data capability |
| ---------------------------- | ------------------------------------ |
| Inspect shape, dtype, stride or scalar arguments | No |
| Carry device addresses or construct tensor views without dereferencing storage | No |
| Emit device predicate metadata (address, comparison, element size) | No; device evaluates the value |
| Execute `get_tensor_data` / read tensor element values on Host | Yes, including reads through a staged Host mirror |
| Execute `set_tensor_data` / write tensor values on Host | Yes, including mirror writes followed by H2D |

The gate must reject that capability for kernel mode before build or execution
resource mutation. Missing metadata and unknown bits also fail closed in kernel
mode. Program mode retains its existing Host accessor behavior and permits old
orchestration libraries without metadata. An explicit future Host-copy argument
ABI must be treated separately; it does not make an arbitrary device tensor
Host-readable. K1's `host_copy_tensor_count` remains zero.

This declaration change does not load or gate requirements symbols and does not
assign a new competing bit number. Producer bit assignments and optional symbol
loading belong to the capability-gate integration with PyPTO.
