# Compact light-tree storage

Implemented after checkpoint `112f7ad` on 2026-09-14.

The **Compact light tree** checkbox under **Light sampling** defaults to off.
Off uses the original 64-byte full-precision mesh and top-level records. On uses
32-byte mesh nodes and 48-byte top-level nodes. Only the selected representation
is allocated. Switching drains prior rendering, rebuilds regular light buffers,
discards pending refits, resets learned lighting and reconstruction history, and
rebinds streamed light storage. Existing streamed chunks remesh their lights into
the new allocation generation, so switching can cause a brief pause and lighting
recovery. It requires no shader reload or restart.

With compaction enabled, top-level nodes retain
their world-space FP32 bounds. Emitted power remains FP32 in both layouts.
CPU build/refit nodes retain their original precision; shaders decode packed
records into registers.

## Layout and publication

`shaders/LightTreePacked.h` defines both GPU layouts for C++ and HLSL.
`rdn/Lighting/LightTreePacking.h` encodes them; `shaders/LightTreeDecode.hlsli`
provides the decoding used by traversal, emitter PDFs, and learned cuts.

In compact mode, each BLAS starts with a 32-byte header containing the mesh root's FP32 minimum
and maximum bounds. `LightSlotGpu::nodeOffset` and `BlasRangeGpu::nodeOffset`
address this header. Node IDs remain local, with root ID zero; the physical
record is at `nodeOffset + 1 + nodeID`. A range's `nodeCount` excludes the header,
while streamed allocation counts include it. Leaf indices remain global in the
shared leaf-index buffer.

In full-precision mode, `nodeOffset` addresses node zero directly and there is no
header or quantization. Both layouts use 16-byte `uint4` structured views, decoded
according to the frame-uniform `RS_FLAG_COMPACT_LIGHT_TREE` flag. Node offsets
remain in units of the selected node size. A streaming job freezes its layout
when allocating nodes; stale jobs are rejected after a buffer-generation change.

The header allows direct access to learned subtrees without decoding parent
chains. Traversal loads it once on entering a mesh. Each bounds component uses
16 bits relative to the header, with outward quantization and a guard for FP32
interpolation error. Degenerate axes and exact endpoints are handled explicitly.
Directions use two 16-bit octahedral coordinates. Cones widen by 0.001 radians
to cover direction encoding and float decoding error. Mesh nodes reconstruct
the sine; top-level nodes retain it. Both sampling and PDF evaluation consume
the same decoded representation.

In both modes, the initial mesh/top-level builders and top-level rebuilder construct child
roots directly in their reserved sibling slots. They no longer leave swapped-out
placeholders in the uploaded vectors. Sibling ordering and triangle/slot trails
are preserved. Incremental updates retain stable IDs and tombstones until their
existing rebuild policy applies, preserving the existing learning revalidation
and reset behavior.

Static meshes are encoded on upload; streamed meshes are encoded by mesh workers.
Top-level updates are encoded by the refit worker and published with their full
CPU topology, trails, and slot records. With compaction off, refits skip encoding
and upload their original records. Transform
updates do not re-encode unchanged mesh trees. All streamed node allocations,
copies, and descriptors use the selected stride and add a header only in compact mode.

## Measurements

The following measurements preceded the runtime toggle and compare the original
checkpoint against the packed-only implementation. They do not measure the
current off/on shader branches. The current off mode also removes unreachable
placeholders, so its memory difference from on is smaller than the figures below.

Matched baseline and compact builds on an RTX 5090, using 262,144 triangle lights
in one BLAS, 512 receivers, 2,073,600 queries, and mature cuts averaging 31 entries.
Dedicated sample and sample-plus-PDF shaders avoid unrelated test branches.
Each entry below is a median of seven GPU dispatches from the second paired run.
The first paired run showed the same direction of change. These are isolated
GPU timings, not whole-frame performance or hardware bandwidth-counter readings.

| Query | Baseline | Compact |
| --- | ---: | ---: |
| Coherent ordinary sample | 0.514 ms | 0.523 ms |
| Scattered ordinary sample | 0.527 ms | 0.528 ms |
| Coherent learned sample | 0.498 ms | 0.462 ms |
| Scattered learned sample | 0.626 ms | 0.557 ms |
| Coherent ordinary sample + PDF | 0.935 ms | 0.988 ms |
| Scattered ordinary sample + PDF | 0.968 ms | 0.997 ms |
| Coherent learned sample + PDF | 0.979 ms | 0.881 ms |
| Scattered learned sample + PDF | 1.307 ms | 1.039 ms |

Node storage fell from **44,739,136 bytes (42.67 MiB)** to **11,184,832 bytes
(10.67 MiB)**: removal of 349,524 unreachable placeholders, then encoding of
349,525 reachable nodes plus one header. The configured voxel node reserve falls
from 384 MiB to 192 MiB. Triangle records, trails, and the 297.03 MiB learning
allocation are separate and retain their existing layouts.

Single-thread CPU encoding, including output allocation, measured **6.22–6.32 ms
for 349,525 mesh nodes** and **0.13–0.22 ms for 10,000 top-level nodes** over the
two runs. These exclude tree construction, GPU upload, and synchronization. The
large mesh encoding is build-time work; typical per-frame top-level encoding
can run inside the existing asynchronous refit. Re-encoding hundreds of thousands
of changing mesh nodes every frame would need a separate budget assessment.

Learned sampling improved in this fixture. Ordinary traversal was approximately
unchanged for sampling alone and 3–6% slower when also evaluating the PDF, due to
decoding overhead. Compression is not a universal traversal speedup. Large meshes
whose bounds dwarf individual emitters can also acquire looser quantized bounds;
conservative support and matching PDFs preserve correctness, but variance can
change. Scene image quality and whole-frame gains have not been measured.

## Verification

- Baseline and compact versions pass `tests/run_light_tree_tests.ps1`.
- Added CPU/GPU containment, cone support, exact-power and topology checks for
  8,193 nodes across six coordinate domains, including flat bounds, tiny meshes,
  large offsets, and very large/small extents.
- Nonzero mesh-header offsets and streamed leaf rebasing, reachable-only regular
  and streamed trees, stable refit trails, tombstones, and asynchronous packed
  publication are checked.
- Existing deep-tree, learning, motion, LOD, sampling/PDF, and integral checks pass.
- Repeated off/on/off/on rebuilds reuse a builder and descriptor heap; both layouts
  pass deep-tree traversal, refit, and sample/PDF checks. Full-precision GPU
  decoding is checked bit-for-bit for bounds, axes, cone values, and power.
- Learned sampling and integral tests run in both modes; disabled refits produce
  no packed output. Pipeline tests verify default-off and history invalidation.
- Render-pipeline tests pass. The RelWithDebInfo application builds and links.
- Production PT, SHARC, primary NEE, learning, and fast PT/SHARC shaders compile.

Local logs and the original benchmark harness are under ignored
`out/light-tree-compaction/`. The saved harness uses the earlier packed-only
bindings and needs adaptation before benchmarking the runtime toggle.
