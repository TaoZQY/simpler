# HBG host graph build

Reference: the supplied `kernel-mode-design.md`, [v9 final decisions](https://icc.gt.tc/vllm-pto#v9-design).
These are internal Host C++ interfaces; public K1 init/prepare/launch remain unsupported stubs.

## Host build and program upload

`hbg::build_graph` performs Host orchestration against already-staged arguments.
It neither commits device execution regions nor uploads the graph.
`GraphBuild` owns orchestration and Definition records but borrows the SM mirror,
Definition staging and matching runtime workspace. These buffers must remain
leased until the result is no longer used; a second build cannot reuse them
while the first result is still being consumed. Build and upload require
exclusive workspace access.

`hbg::upload_program_graph` is the explicit program-only allocation and
synchronous H2D boundary. It creates a compact image from the virtual-address
source on each upload. If Definition staging grows and moves, the upload owner
preserves its contents and rebinds the build's staging reference before further
processing, including before a potentially failing copy. Repeated upload and
retry therefore keep a valid source. Program execution still calls build and
upload in sequence and retains its existing resource management.
