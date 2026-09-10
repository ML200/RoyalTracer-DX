Run `./tests/run_light_tree_tests.ps1` from a Visual Studio Developer PowerShell.
The standalone runner requires D3D12 and compiles the production light-tree HLSL.

Coverage:

- TLAS and BLAS trails at depths 0, 16, 17, 31, and 32, including all four child
  indices and paths crossing the low/high 32-bit word boundary.
- Malformed depth-33 trees: sampling and PDF evaluation both return zero.
- Skewed scenes through the initial BLAS/TLAS builders and the TLAS refitter.
  Every uploaded trail is compared with a separate topology walk, and every
  triangle's GPU PDF is compared with the reference path probability.
- Coincident triangles and instances, single-triangle leaves, the full 32-bit
  item-count depth budget, and rejection of attempts to encode past capacity.

The shared trail layout is `shaders/LightTreeTrail.h`: 64 bits per trail,
uploaded as `R32G32_UINT` and read as `uint2`. Builders reserve enough remaining
levels for binary median splits, so skewed SAOH choices cannot exceed 32 levels.
Sampling and PDF traversal both allow a leaf at exactly depth 32.

SG and learned-cut regressions are documented in docs/LIGHT_TREE_LEARNING.md. The test builds both the production sampler and frame-update shader. Learning tests bind a dedicated 34 MiB buffer at the production UAV slot and exercise actual GPU feedback, updates and refinement.
