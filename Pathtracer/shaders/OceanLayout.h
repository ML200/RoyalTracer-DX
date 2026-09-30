#pragma once
// Shared by C++ and HLSL: only #define and plain structs.
// FFT ocean (Tessendorf 2001), JONSWAP (Hasselmann et al. 1973), spreading (Donelan et al. 1985, Banner 1990).

#define OCEAN_CASCADES 4

// Material block: slot = foam step + OCEAN_FOAM_STEPS * bubble step, then the debug slot.
#define OCEAN_FOAM_STEPS 16
#define OCEAN_BUBBLE_STEPS 4
#define OCEAN_MATERIAL_LEVELS (OCEAN_FOAM_STEPS * OCEAN_BUBBLE_STEPS)
#define OCEAN_MATERIAL_DEBUG OCEAN_MATERIAL_LEVELS
#define OCEAN_MATERIAL_COUNT (OCEAN_MATERIAL_LEVELS + 1)

// residualSlope entry i: footprint width 2^(i + OFFSET) m.
#define OCEAN_ROUGHNESS_ENTRIES 24
#define OCEAN_ROUGHNESS_OFFSET (-10)

#ifndef OCEAN_FFT_SIZE
#define OCEAN_FFT_SIZE 512
#endif
#ifndef OCEAN_FFT_LOG2
#define OCEAN_FFT_LOG2 9
#endif

#define OCEAN_MIP_LEVELS (OCEAN_FFT_LOG2 + 1) // displacement and derivative arrays

#define OCEAN_FFT_THREADS (OCEAN_FFT_SIZE / 2) // one radix-2 butterfly each

#define OCEAN_TILE_GRID 32 // quads per tile edge
#define OCEAN_TILE_EDGE_VERTS (OCEAN_TILE_GRID + 1)
#define OCEAN_TILE_VERTS (OCEAN_TILE_EDGE_VERTS * OCEAN_TILE_EDGE_VERTS)
#define OCEAN_TILE_TRIS (OCEAN_TILE_GRID * OCEAN_TILE_GRID * 2)
#define OCEAN_TILE_INDICES (OCEAN_TILE_TRIS * 3)

#define OCEAN_MAX_TILES 512
#define OCEAN_PREVIOUS_POSITION_SLOT 13
// Albedo-guide scratch slice: (foam cover, bubble density, unused, unused).
#define OCEAN_GUIDE_SLOT 6

// OceanTileGPU.stitch: 3 bits per edge (-X, +X, -Z, +Z), levels to the coarser neighbour.
#define OCEAN_STITCH_BITS 3u
#define OCEAN_STITCH_MASK 7u
#define OCEAN_EDGE_NEG_X 0u
#define OCEAN_EDGE_POS_X 1u
#define OCEAN_EDGE_NEG_Z 2u
#define OCEAN_EDGE_POS_Z 3u

// Descriptor heap slots (ResourceDescriptorHeap).
#define OCEAN_HEAP_BASE 90u

// Texture2DArray<float4> (Dx, Dy, Dz, dDx/dz), half, mipped; dispParity ping-pong.
#define OCEAN_SRV_DISP0 (OCEAN_HEAP_BASE + 0u)
#define OCEAN_SRV_DISP1 (OCEAN_HEAP_BASE + 1u)
#define OCEAN_SRV_DERIV (OCEAN_HEAP_BASE + 2u)  // Texture2DArray<float4> (dy/dx, dy/dz, dDx/dx, dDz/dz), mipped
#define OCEAN_SRV_PARAMS (OCEAN_HEAP_BASE + 3u) // StructuredBuffer<OceanParamsGPU>, one element
#define OCEAN_SRV_H0 (OCEAN_HEAP_BASE + 4u)     // Texture2DArray<float4>, h0(k) and conj(h0(-k))
#define OCEAN_SRV_TILES (OCEAN_HEAP_BASE + 5u)  // StructuredBuffer<OceanTileGPU>

// Sea-state gain fBm: Texture2D<float4> (gain, d(gain)/dx, d(gain)/dz, unused), wrapping.
#define OCEAN_SRV_TURBULENCE (OCEAN_HEAP_BASE + 6u)
#define OCEAN_TURBULENCE_SIZE 512

#define OCEAN_UAV_FFT (OCEAN_HEAP_BASE + 7u)   // RWTexture2DArray<float4>, 2 complex pairs per slice
#define OCEAN_UAV_VERTS (OCEAN_HEAP_BASE + 8u)

// Whitecap cover clipmap, after Crest: Texture2DArray<float>, half, mipped; foamParity ping-pong.
#define OCEAN_SRV_FOAM0 (OCEAN_HEAP_BASE + 9u)
#define OCEAN_SRV_FOAM1 (OCEAN_HEAP_BASE + 11u)
#define OCEAN_FOAM_SIZE 2048
#define OCEAN_FOAM_MIPS 12
#define OCEAN_FOAM_LEVELS 5
#define OCEAN_FOAM_TEXEL0 0.0625f // m; doubles per level
#define OCEAN_FOAM_REFERENCE_LEVEL 2u // cover is measured and steered here
#define OCEAN_FOAM_BREAK_WIDTH 1.0f // m, surface filter for the breaking test
#define OCEAN_FOAM_FRESH_DECAY 5.0f // fresh foam fades this much faster than lace

// UAV mip chains: mip L at base + L.
#define OCEAN_UAV_DISP0_MIPS (OCEAN_HEAP_BASE + 12u)
#define OCEAN_UAV_DISP1_MIPS (OCEAN_UAV_DISP0_MIPS + OCEAN_MIP_LEVELS)
#define OCEAN_UAV_DERIV_MIPS (OCEAN_UAV_DISP1_MIPS + OCEAN_MIP_LEVELS)
#define OCEAN_UAV_FOAM0_MIPS (OCEAN_UAV_DERIV_MIPS + OCEAN_MIP_LEVELS)
#define OCEAN_UAV_FOAM1_MIPS (OCEAN_UAV_FOAM0_MIPS + OCEAN_FOAM_MIPS)

// RWByteAddressBuffer: a Jacobian histogram per foam level, then OceanFoamState.
#define OCEAN_UAV_FOAM_STATS (OCEAN_UAV_FOAM1_MIPS + OCEAN_FOAM_MIPS)
#define OCEAN_FOAM_BINS 256
#define OCEAN_FOAM_HIST_MIN (-1.0f)
#define OCEAN_FOAM_HIST_STEP (2.0f / OCEAN_FOAM_BINS)
#define OCEAN_FOAM_HIST_STRIDE 4
#define OCEAN_FOAM_STATE_OFFSET (OCEAN_FOAM_LEVELS * OCEAN_FOAM_BINS * 4)
#define OCEAN_FOAM_STATE_BYTES 128
// Then the reference level's white-foam sum, in 1/OCEAN_FOAM_WHITE_SCALE.
#define OCEAN_FOAM_WHITE_OFFSET (OCEAN_FOAM_STATE_OFFSET + OCEAN_FOAM_STATE_BYTES)
#define OCEAN_FOAM_WHITE_SCALE 1024.0f
#define OCEAN_FOAM_STATS_BYTES (OCEAN_FOAM_WHITE_OFFSET + 16)

#define OCEAN_HEAP_COUNT ((OCEAN_UAV_FOAM_STATS + 1u) - OCEAN_HEAP_BASE)

// Diagnostic views, OceanParamsGPU.debugMode.
#define OCEAN_DEBUG_BEAUTY 0u
#define OCEAN_DEBUG_NORMALS 1u
#define OCEAN_DEBUG_COMPRESSION 2u
#define OCEAN_DEBUG_GEOMETRY 3u // tile level and quad grid
#define OCEAN_DEBUG_MIP 4u
#define OCEAN_DEBUG_FOAM 5u      // whitecap coverage over the normal map
#define OCEAN_DEBUG_ROUGHNESS 6u // footprint roughness the hit hands the BRDF
#define OCEAN_DEBUG_COUNT 7u

#ifdef __cplusplus
#define OCEAN_FLOAT2 DirectX::XMFLOAT2
#define OCEAN_FLOAT3 DirectX::XMFLOAT3
#define OCEAN_FLOAT4 DirectX::XMFLOAT4
#define OCEAN_INT4 DirectX::XMINT4
#define OCEAN_UINT uint32_t
#else
#define OCEAN_FLOAT2 float2
#define OCEAN_FLOAT3 float3
#define OCEAN_FLOAT4 float4
#define OCEAN_INT4 int4
#define OCEAN_UINT uint
#endif

// Vertices are relative to anchor, for fp32 precision.
struct OceanTileGPU {
    OCEAN_FLOAT3 anchor;   // tile origin relative to the scene origin
    float size;            // tile edge length in metres
    OCEAN_UINT vertexBase; // first vertex of this tile in the global vertex buffer
    OCEAN_UINT stitch;     // OCEAN_EDGE_* bits
    OCEAN_UINT level;      // quadtree level, 0 = root
    OCEAN_UINT pad;
    // Earth curvature: drop(p) = curveBase + dot(curveGrad, p) + |p|^2 / 2R, p tile-local.
    float curveBase;
    OCEAN_FLOAT2 curveGrad;
    float invCurveRadius; // 1 / R, or zero when curvature is switched off
};

struct OceanParamsGPU {
    OCEAN_FLOAT4 cascadeLength; // L per cascade, metres

    // Per-cascade crest sharpening: skew coefficient and elevation variance (OceanSkewHeight).
    OCEAN_FLOAT4 crestSkew;
    OCEAN_FLOAT4 cascadeVariance;

    // Scene origin wrapped to each cascade's period.
    OCEAN_FLOAT4 originWrapX;
    OCEAN_FLOAT4 originWrapZ;

    float turbulencePeriod; // m
    float turbulenceStrength; // 0 disables
    float choppiness; // horizontal displacement gain
    float surfaceY; // mean sea level in scene-relative coordinates

    OCEAN_FLOAT2 curveOrigin; // absolute scene origin, for Earth curvature
    float invRadius;
    float halfExtent; // finite ocean boundary in absolute XZ

    OCEAN_FLOAT3 originDelta; // current minus previous scene origin
    float historyValid;

    // Conservative displaced-height bounds about surfaceY.
    float crestHeight;
    float troughDepth;
    float filterScale; // diagnostics only: ray footprint multiplier
    float sunLobeRoughness; // roughness floor for sun sampling and NEE only

    OCEAN_UINT materialBase; // first slot of the OCEAN_MATERIAL_COUNT block
    OCEAN_UINT debugMode;    // OCEAN_DEBUG_*
    OCEAN_UINT dispParity;   // which displacement array holds this frame; the other one is history
    OCEAN_UINT pad;

    // Tile selection's ocean::LodRule, for OceanGeometryWidth; lodCamera is scene-relative.
    OCEAN_FLOAT3 lodCamera;
    float lodRatio; // filter width / distance in view
    OCEAN_FLOAT3 lodForward;
    float lodTanH;
    OCEAN_FLOAT3 lodRight;
    float lodTanV;
    OCEAN_FLOAT3 lodUp;
    float lodOffscreen; // ratio multiplier outside the view
    float lodNearKeep;
    float lodSilhouette; // distance past which the ratio grows with distance; 0 disables
    float lodMinWidth;
    float lodPad;

    // Per foam level: xy scene-relative corner of texel (0, 0), zw texels moved since last frame.
    OCEAN_FLOAT4 foamLevel[OCEAN_FOAM_LEVELS];
    // Per foam level: xy absolute index of texel (0, 0), zw unused.
    OCEAN_INT4 foamCell[OCEAN_FOAM_LEVELS];
    float foamCover; // target whitecap cover
    float foamSteer; // per second, in log share per log cover error
    float foamRate; // cover per second of full breaking
    float foamDecayRate;  // natural-log decay per second
    float foamStrength;   // 0 switches foam off
    OCEAN_UINT frameIndex; // per-frame foam dither seed
    OCEAN_UINT foamParity; // which foam texture holds this frame
    OCEAN_UINT foamHistory; // 0 when last frame's foam cannot be carried over
    OCEAN_FLOAT4 chopBand; // OceanChopGain band
    float foamBreakMax; // highest breaking point (Jacobian)
    OCEAN_FLOAT2 foamWind; // unit heading the wind blows towards, world xz
    float foamPad0;

    // Unresolved slope variance per footprint width (OCEAN_ROUGHNESS_ENTRIES).
    OCEAN_FLOAT4 residualSlope[OCEAN_ROUGHNESS_ENTRIES / 4];
};

// At OCEAN_FOAM_STATE_OFFSET of OCEAN_UAV_FOAM_STATS; the host reads it back.
struct OceanFoamState {
    float threshold[OCEAN_FOAM_LEVELS]; // Jacobian each level breaks below
    float softness[OCEAN_FOAM_LEVELS];  // how much further below that its crests break in full
    float match[OCEAN_FOAM_LEVELS]; // breaking share vs. the reference level's, for equal cover
    float coverage[OCEAN_FOAM_LEVELS]; // each level's mean foam cover over its own ground
    float white;     // share of the reference level's ground plainly white (OceanFoamWhite)
    float share;     // share of the reference level the steering asks to break
    float breaking;  // share of it that does, which foamBreakMax can hold under that
    float target;    // the cover the steering last aimed at
    float decayRate; // and the foam's decay rate then
    float settle;    // seconds until the steering resumes after the foam started over
    OCEAN_UINT valid; // 0 until the first breaking points are set
};
