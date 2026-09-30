#pragma once
#include "PathStateLayout.h"

uint ps_numPx() { return ((IMG_W + 7u) / 8u) * ((IMG_H + 3u) / 4u) * 32u; }
uint ps_plane(uint plane, uint bytes, uint px) { return ps_numPx() * plane + px * bytes; }

// Primary-vertex extras, written by the camera pass.
void store_rg_primaryExtra(uint pixelIdx, float2 iors, uint mediumMatID, float3 absorptionTint)
{
    const uint base = ps_plane(PS_PRIMARY_PLANE, PS_PRIMARY_BYTES, pixelIdx);
    g_pathStateBuffer.Store4(base, uint4(f32tof16(iors.x) | (f32tof16(iors.y) << 16), mediumMatID,
        f32tof16(absorptionTint.x) | (f32tof16(absorptionTint.y) << 16), f32tof16(absorptionTint.z)));
}

void load_rg_primaryExtra(uint pixelIdx, out float2 iors, out uint mediumMatID, out float3 absorptionTint)
{
    const uint4 w = g_pathStateBuffer.Load4(ps_plane(PS_PRIMARY_PLANE, PS_PRIMARY_BYTES, pixelIdx));
    iors           = float2(f16tof32(w.x & 0xFFFFu), f16tof32(w.x >> 16));
    mediumMatID    = w.y;
    absorptionTint = float3(f16tof32(w.z & 0xFFFFu), f16tof32(w.z >> 16), f16tof32(w.w & 0xFFFFu));
}

// Addressed by training lane, not pixel.
uint ps_addr_guideRoot(uint lane, uint r) { return ps_numPx() * PS_GUIDE_ROOTS_PLANE + (lane * 2u + r) * 32u; }
bool ps_guideRootBacked(uint lane) { return (lane + 1u) * 64u <= ps_numPx() * PS_GUIDE_ROOTS_BYTES; }
uint ps_addr_trainSpill(uint lane) { return ps_numPx() * PS_TRAIN_SPILL_PLANE + lane * 48u; }
