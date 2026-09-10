# Kernel launch binder

The binder implements the three-stream enqueue protocol from the
[v9 kernel-mode design](https://icc.gt.tc/vllm-pto?i=1#v9-design).
The implementation follows its September 8 final-decision table: caller,
dedicated **non-hidden AICPU**, and **hidden AICore** are distinct streams.
The AICPU stream must not be replaced by caller or created as hidden.

## Prepared inputs and ownership

`launch_bound_kernel_native` takes a `KernelExecutionState`, a
`KernelInvocationBinding`, prepared `KernelNativeInvocation` arguments, and a
borrowed caller stream. `launch_bound_kernel` provides the same binder with
injected read-only gates and the six `KernelLaunchOps` enqueue operations.

The owner must complete the following outside launch:

- Create the two internal streams and five distinct events. The native context
  adapter must create events with `aclrtCreateEventExWithFlag(..., ACL_EVENT_SYNC)`.
- Collect resource requirements, allocate context resources, and freeze capacity.
- Register and retain the AICore/AICPU function handles and their loaded binaries.
- Resolve the callable under a registration lease. The trusted callable view,
  runtime binding, packet, and handles must describe that same registration.
- Enqueue device preparation/slot registration, record `PrepareTail` on its
  preparation stream, then call `mark_ready_enqueued`. Ready means submitted;
  an externally required warmup completion check belongs outside launch.
- Build an immutable HBG template and a separate writable `GraphHostArgs` buffer;
  prepare core arguments, clear regions, cancel words, and placeholder offsets.
  The clear regions must include all per-launch handshake/report state. The
  cancel region consists only of the waiting AICore's cancel words.

HostArgs staging is exclusive through submission. Native HostArgs copies the
packet into task-owned storage before returning; the adapter restores zero Host
placeholders for reuse. Function handles, device memory, streams, events and
context ownership survive all executions and captured graphs until external
quiescence. Host return is not permission to close or recycle those resources.
The binder stores caller identity only, and never owns or destroys caller.

The binder preflight checks Host submission exclusion, ready/device/generation
identity, operation tables, distinct handles, current device, frozen capacity and
schema, common invocation ABI/callable identity, runtime binding and prepared
native arguments. HBG's `GraphKernelBinding::validate` additionally checks the
Host-template checksum/framing, exact placeholder, independently sealed slot,
and all five context resource bases/capacities. It performs no device IO.
A rejected preflight leaves context state and streams untouched.

## Input to output sequence

When caller changes, query the previous `SerialTail` **before any enqueue**.
Not-ready or query error rejects cleanly. Completion permits submission without
waiting on the old tail. Repeated submission on the same caller performs no
query or old-tail wait; caller FIFO and both branch joins provide ordering.
External replay overlap remains the caller's responsibility under the v1 contract.

| Host order | Stream | Operation |
| ---------- | ------ | --------- |
| 1 | caller | Wait on pending `PrepareTail`, once per ready publication |
| 2 | caller | Clear the prepared handshake/report regions asynchronously |
| 3 | caller | Record `Start` |
| 4 | hidden AICore | Wait `Start`, launch AICore, record `AicoreDone` |
| 5 | non-hidden AICPU | Wait `Start`, launch with HostArgs, record `AicpuDone` |
| 6 | caller | Wait `AicpuDone`, wait `AicoreDone`, record `SerialTail` |

AICore is submitted first. Its early waiter must accept the pre-window cancel
sentinel, and AICPU must publish the normal handshake only after successful task
admission. This also avoids placing a waiting AICPU task ahead of the core SQE
that would release it. Device branches still share the caller's `Start` gate.

For HBG, the task-owned packet is subsequently validated against the resident
slot and restored by the AICPU leader. Followers and AICore remain gated until
restore publication. Normal/controlled-failure completion clears device aliases
and execution state before retiring the slot; fatal completion poisons it.
Those device entry actions consume the existing HBG restore/retirement APIs.
Binder success reports **submission**, not numerical results or device completion.
Subsequent Ascend C/Triton work on caller follows the two joins in eager/capture;
onboard verification is required for capture/replay behavior of these event types.

## Failure contract

Every failed enqueue poisons the context, including failure of the first enqueue.
`KernelLaunchResult` retains the first error and failing step independently from
`cleanup_status`; `tail_recorded` states whether a caller tail was established.
A successful compensation still leaves the context poisoned.

| Failure location | Compensation |
| ---------------- | ------------ |
| Before successful core launch | Stop enqueue; no waiting core requires cancellation |
| First `AicoreDone` record | Cancel on caller, retry done record once, join core, record tail |
| AICPU start wait | Cancel on caller, join core, record tail |
| AICPU launch | Cancel on caller, record/join the already forked AICPU wait branch, join core, record tail |
| After successful AICPU launch | Stop; Host must not overwrite live handshake with cancel |

Cancel is one `aclrtMemsetAsync(..., 0xff, ...)` over prepared 32-bit words,
representing `UINT32_MAX`; it needs no pinned Host scalar. If any compensation
operation fails, stop compensation and retain that error. There is then no
provable caller join; even external stream synchronization may time out for an
uncancelled waiting core. This is an accepted terminal runtime failure, not a
recoverable slot. Teardown must follow external quiescence/reset ownership.

## Integration boundary and validation

The common sequence, binder gates, HBG binding validator and real CANN enqueue
adapter are implemented. Existing program-mode launch remains separate. The
public K1 launch entry stays unsupported until persistent function/handshake
preparation, callable residency resolution, and the native HBG device entry are
wired through the owners. Passing an arbitrary caller-supplied callable view is
not a substitute for that resolver; the internal binder requires its lease.
The local common invocation ABI remains 64 bytes.

`test_kernel_launch_binder` injects every enqueue and compensation failure,
checks clean stream-switch rejection, same-stream FIFO, and Host concurrency.
`test_kernel_launch_native` compiles against CANN declarations with fake ACL
symbols, checking actual adapter routing, HostArgs copy lifetime, cancel fills,
and rejection before enqueue. Build it with the existing
`SIMPLER_ENABLE_HARDWARE_TESTS=ON` option; its execution needs no hardware.
HBG tests check packet/slot/resource mismatches without device mutation.
`test_kernel_launch_source_guard.py` rejects allocation, synchronization, capture
queries and unapproved CANN calls in the submission sources.
These tests do not establish native capture/replay correctness, precision or
performance. That requires the integrated device entry and mixed-operator
onboard tests, including event Probe B and changed packet payloads on replay.
