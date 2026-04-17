#pragma once
// ═══════════════════════════════════════════════════════════════════
// ReuseTextureGen.h — CPU-side generator for self-inverting reuse
//   textures used by paired spatial reuse
//   (Lin, Kettunen, Wyman 2026).
//
// Each texel stores a 2D int16 delta (dx, dy). Applying the delta
// from any texel lands on its paired partner; applying the partner's
// delta lands back on the original. The texture is tileable under
// per-axis wrap at `size`.
// ═══════════════════════════════════════════════════════════════════

#include <cstdint>
#include <vector>

// Generate a square reuse texture of dimension `size` × `size`.
// `size` must be even (the pairing requires an even pixel count).
//
// `sigma` controls the approximate standard deviation of the (dx, dy)
// distribution in pixels. Must be ≥ 0.8. Paper example uses sizes
// 254 / 230 / 210 for a 3-slot setup.
//
// `seed` seeds the internal RNG so generation is deterministic.
//
// `outRG` is resized to 2 × size × size int16 values, interleaved as
// [dx0, dy0, dx1, dy1, …] in row-major order, suitable for upload as
// DXGI_FORMAT_R16G16_SINT.
void GenerateReuseTexture(int size,
                          float sigma,
                          uint32_t seed,
                          std::vector<int16_t>& outRG);

// Verify the self-inverting property: for every texel (x, y) with
// delta (dx, dy), the texel at ((x+dx) mod size, (y+dy) mod size) has
// a delta that, when applied, lands back at (x, y). Returns true on
// success; on failure `outFirstBadTexel` (if non-null) is set to the
// linear index of the first offending texel.
bool ValidateReuseTexture(int size,
                          const std::vector<int16_t>& rg,
                          int* outFirstBadTexel = nullptr);
