#pragma once
// Layout shared by the ocean host code and every ocean shader. Included from C++ (through
// rdn/ocean/OceanCommon.h) and from HLSL, so it must stay free of anything either side cannot
// parse: no namespaces, no templates, only #define and plain structs.
//
// The ocean is a spectral (Tessendorf) surface: a JONSWAP frequency spectrum with Donelan-Banner
// directional spreading is sampled into OCEAN_CASCADES tiling wavenumber bands, inverse-FFT'd
// every frame. Mesh displacement is filtered to geometry spacing; shading filters the normal to
// the ray footprint and turns the ripples it averages away into roughness. Breaking crests leave
// whitecap foam that decays in place.

// ---------------------------------------------------------------------------------------------
// Simulation
// ---------------------------------------------------------------------------------------------

// Cascades are overlapping rather than banded: each one synthesises every wavelength its grid can
// represent, and a partition of unity over wavenumber splits the energy between them. That keeps
// every grid densely populated, and a given wavelength is carried by several tiles of different,
// mutually incommensurate periods, which is what breaks up the repetition an FFT patch would
// otherwise show across a wide field of view.
#define OCEAN_CASCADES 4

// The ocean's material block: OCEAN_FOAM_STEPS steps of whitecap coverage from clear water (slot 0)
// to solid foam, repeated for each of OCEAN_BUBBLE_STEPS densities of the bubble cloud a breaker
// leaves under the surface (slot = foam step + OCEAN_FOAM_STEPS * bubble step), then an opaque slot
// the diagnostic views shade with. A hit picks its steps from the foam it samples, dithered between
// neighbouring steps so neither ramp shows bands.
#define OCEAN_FOAM_STEPS 16
#define OCEAN_BUBBLE_STEPS 4
#define OCEAN_MATERIAL_LEVELS (OCEAN_FOAM_STEPS * OCEAN_BUBBLE_STEPS)
#define OCEAN_MATERIAL_DEBUG OCEAN_MATERIAL_LEVELS
#define OCEAN_MATERIAL_COUNT (OCEAN_MATERIAL_LEVELS + 1)

// Residual slope variance against footprint width: entry i is for a width of 2^(i + OFFSET) m.
#define OCEAN_ROUGHNESS_ENTRIES 24
#define OCEAN_ROUGHNESS_OFFSET (-10)

// 512 samples per edge. The finest cascade's 17 m patch then resolves 6.6 cm waves, below what a
// pixel covers at any distance the sea is normally seen from; the four 1024^2 fields this replaces
// cost four times the transform and sixteen times the memory for ripples that only resolved as
// sparkle noise.
#ifndef OCEAN_FFT_SIZE
#define OCEAN_FFT_SIZE 512
#endif
#ifndef OCEAN_FFT_LOG2
#define OCEAN_FFT_LOG2 9
#endif

// Mip levels on the displacement and derivative arrays: log2(N) + 1.
#define OCEAN_MIP_LEVELS (OCEAN_FFT_LOG2 + 1)

// Threads per line in the FFT passes; each one owns a single radix-2 butterfly per field pair.
#define OCEAN_FFT_THREADS (OCEAN_FFT_SIZE / 2)

// ---------------------------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------------------------

// Quads per tile edge. Every resident tile is refitted every frame because its vertices move, so
// tiles times this squared is the triangle count the ocean pays for before a ray is traced.
// Geometry only has to carry the waves that change a silhouette or a hit position; everything
// finer is in the full-resolution normal the hit evaluator samples, so the grid can be coarse.
// Selection keeps quads near 1.6% of their distance inside the view and coarsens everything else
// (see ocean::LodRule). The default sea seen from deck height is then ~300 tiles and 0.61M
// triangles, where the 64^2 grid it replaces built 3.1M for the same view.
#define OCEAN_TILE_GRID 32
#define OCEAN_TILE_EDGE_VERTS (OCEAN_TILE_GRID + 1)
#define OCEAN_TILE_VERTS (OCEAN_TILE_EDGE_VERTS * OCEAN_TILE_EDGE_VERTS)
#define OCEAN_TILE_TRIS (OCEAN_TILE_GRID * OCEAN_TILE_GRID * 2)
#define OCEAN_TILE_INDICES (OCEAN_TILE_TRIS * 3)

#define OCEAN_MAX_TILES 512
#define OCEAN_PREVIOUS_POSITION_SLOT 13
// Scratch slice the camera pass leaves an ocean pixel's foam in for the reconstruction's albedo
// guide: (foam cover, bubble-cloud density, unused, unused), as OceanSurface carries them.
#define OCEAN_GUIDE_SLOT 6

// OceanTileGPU.stitch holds three bits per edge, in the order -X, +X, -Z, +Z: how many levels
// coarser the neighbour across that edge is. The neighbour's vertices fall on every 2^k-th of
// ours along it, and the ones in between are interpolated onto its edge to close the crack.
#define OCEAN_STITCH_BITS 3u
#define OCEAN_STITCH_MASK 7u
#define OCEAN_EDGE_NEG_X 0u
#define OCEAN_EDGE_POS_X 1u
#define OCEAN_EDGE_NEG_Z 2u
#define OCEAN_EDGE_POS_Z 3u

// ---------------------------------------------------------------------------------------------
// Descriptor heap slots (the root signature sets CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED, so every
// ocean resource is reached through ResourceDescriptorHeap rather than a root parameter).
// ---------------------------------------------------------------------------------------------

#define OCEAN_HEAP_BASE 90u

// Displacement ping-pongs between two arrays: one frame's field is the next frame's history, which
// is what motion vectors difference against, so no copy is needed to keep it.
// Texture2DArray<float4> (Dx, Dy, Dz, dDx/dz), half precision, mipped. OceanParamsGPU.dispParity
// names the current one.
#define OCEAN_SRV_DISP0 (OCEAN_HEAP_BASE + 0u)
#define OCEAN_SRV_DISP1 (OCEAN_HEAP_BASE + 1u)
#define OCEAN_SRV_DERIV (OCEAN_HEAP_BASE + 2u)  // Texture2DArray<float4> (dy/dx, dy/dz, dDx/dx, dDz/dz), mipped
#define OCEAN_SRV_PARAMS (OCEAN_HEAP_BASE + 3u) // StructuredBuffer<OceanParamsGPU>, one element
#define OCEAN_SRV_H0 (OCEAN_HEAP_BASE + 4u)     // Texture2DArray<float4>, h0(k) and conj(h0(-k))
#define OCEAN_SRV_TILES (OCEAN_HEAP_BASE + 5u)  // StructuredBuffer<OceanTileGPU>

// Kilometre-scale sea-state field: a tiling fBm of the gain the wave field is multiplied by,
// with its own analytic gradient alongside it so the surface derivatives stay exact.
// (gain, d(gain)/dx, d(gain)/dz, unused), Texture2D<float4>, bilinear, wrapping.
#define OCEAN_SRV_TURBULENCE (OCEAN_HEAP_BASE + 6u)
#define OCEAN_TURBULENCE_SIZE 512

#define OCEAN_UAV_FFT (OCEAN_HEAP_BASE + 7u)   // RWTexture2DArray<float4>, 2 complex pairs per slice
#define OCEAN_UAV_VERTS (OCEAN_HEAP_BASE + 8u)

// Whitecap foam, after Crest: world-space textures centred on the camera, not the tiling wave
// cascades - a crest breaks where the whole surface is squeezed, long waves and short together,
// and that is not periodic in any one cascade's tile. A clipmap of OCEAN_FOAM_LEVELS levels, one
// array slice each, of OCEAN_FOAM_SIZE texels: level L has texels of OCEAN_FOAM_TEXEL0 * 2^L
// metres, so the finest reaches 64 m from the camera and each coarser one twice as far, and a
// texel is about the width a pixel covers at the edge of its level. Each frame the breaking crests
// add foam and all of it fades; the levels scroll with the camera in whole texels, carrying the
// foam over. Two textures ping-pong (OceanParamsGPU.foamParity names this frame's), so last
// frame's is read while this one is written. Texture2DArray<float>, half precision, mipped.
//
// A texel holds the share of its area foam covers, which is the amount the simulation runs on as
// well: it averages, so every mip and the blend between the levels keep the white a footprint
// actually holds.
//
// Nothing in the foam tiles. What a raft looks like inside - its strands and holes as it dissolves
// - is drawn where the foam is shown, from noise at the absolute world position (OceanFoamWeb).
#define OCEAN_SRV_FOAM0 (OCEAN_HEAP_BASE + 9u)
#define OCEAN_SRV_FOAM1 (OCEAN_HEAP_BASE + 11u)
#define OCEAN_FOAM_SIZE 2048
#define OCEAN_FOAM_MIPS 12
#define OCEAN_FOAM_LEVELS 5
#define OCEAN_FOAM_TEXEL0 0.0625f
// The level the whitecap cover is measured and steered on (OceanFoamStats): a quarter-metre grid
// half a kilometre wide, enough breakers that its mean holds still.
#define OCEAN_FOAM_REFERENCE_LEVEL 2u
// Width the composite surface is filtered to when testing for breaking, metres. Air-entraining
// breakers - the ones that leave a whitecap - are waves of a few metres and up, travelling at a
// tenth to two thirds of the dominant waves' speed; shorter ones spill without trapping air. At a
// quarter of a metre every choppy ripple broke, and the sea was sprinkled with small white flecks
// that belonged to no crest. Filtered to a metre, what breaks is the crests of real waves, and a
// cap forms along a crest and is left behind it as the wave runs on. A level coarser than this
// tests at its own texel instead.
#define OCEAN_FOAM_BREAK_WIDTH 1.0f
// How many times faster fresh foam fades than the lace it leaves (OceanFoam).
#define OCEAN_FOAM_FRESH_DECAY 5.0f

// Mip chains: level L of each array is at its base + L. Level 0 is what the vertical FFT (or the
// foam update) writes.
#define OCEAN_UAV_DISP0_MIPS (OCEAN_HEAP_BASE + 12u)
#define OCEAN_UAV_DISP1_MIPS (OCEAN_UAV_DISP0_MIPS + OCEAN_MIP_LEVELS)
#define OCEAN_UAV_DERIV_MIPS (OCEAN_UAV_DISP1_MIPS + OCEAN_MIP_LEVELS)
#define OCEAN_UAV_FOAM0_MIPS (OCEAN_UAV_DERIV_MIPS + OCEAN_MIP_LEVELS)
#define OCEAN_UAV_FOAM1_MIPS (OCEAN_UAV_FOAM0_MIPS + OCEAN_FOAM_MIPS)

// Where each foam level breaks, measured rather than assumed. The foam update histograms the
// Jacobian of the unit-gain sea it tests (every OCEAN_FOAM_HIST_STRIDE-th texel each way,
// OCEAN_FOAM_BINS bins from OCEAN_FOAM_HIST_MIN in steps of OCEAN_FOAM_HIST_STEP; nothing
// stretched past its rest area is counted, as no breaking point lies there); after it, each
// level's breaking point is set to the quantile that breaks the share OceanFoamState steers.
// RWByteAddressBuffer: the histograms, then OceanFoamState.
#define OCEAN_UAV_FOAM_STATS (OCEAN_UAV_FOAM1_MIPS + OCEAN_FOAM_MIPS)
#define OCEAN_FOAM_BINS 256
#define OCEAN_FOAM_HIST_MIN (-1.0f)
#define OCEAN_FOAM_HIST_STEP (2.0f / OCEAN_FOAM_BINS)
#define OCEAN_FOAM_HIST_STRIDE 4
#define OCEAN_FOAM_STATE_OFFSET (OCEAN_FOAM_LEVELS * OCEAN_FOAM_BINS * 4)
#define OCEAN_FOAM_STATE_BYTES 128
// After the state: the reference level's white foam summed over the same texels as the histogram,
// in 1/OCEAN_FOAM_WHITE_SCALE (OceanFoamWhite).
#define OCEAN_FOAM_WHITE_OFFSET (OCEAN_FOAM_STATE_OFFSET + OCEAN_FOAM_STATE_BYTES)
#define OCEAN_FOAM_WHITE_SCALE 1024.0f
#define OCEAN_FOAM_STATS_BYTES (OCEAN_FOAM_WHITE_OFFSET + 16)

#define OCEAN_HEAP_COUNT ((OCEAN_UAV_FOAM_STATS + 1u) - OCEAN_HEAP_BASE)

// Diagnostic views, OceanParamsGPU.debugMode.
#define OCEAN_DEBUG_BEAUTY 0u
#define OCEAN_DEBUG_NORMALS 1u
#define OCEAN_DEBUG_COMPRESSION 2u
#define OCEAN_DEBUG_GEOMETRY 3u // tile level and quad grid: what the acceleration structures hold
#define OCEAN_DEBUG_MIP 4u
#define OCEAN_DEBUG_FOAM 5u      // whitecap coverage over the normal map
#define OCEAN_DEBUG_ROUGHNESS 6u // footprint roughness the hit hands the BRDF
#define OCEAN_DEBUG_COUNT 7u

// ---------------------------------------------------------------------------------------------
// Shared records
// ---------------------------------------------------------------------------------------------

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

#define OCEAN_PI_CONST 3.14159265358979f

// One ocean tile: a square of the world XZ plane meshed at OCEAN_TILE_GRID quads per edge.
// Positions are written relative to `anchor` so a 100 km ocean stays in fp32 range.
struct OceanTileGPU {
    OCEAN_FLOAT3 anchor;   // tile origin relative to the scene origin
    float size;            // tile edge length in metres
    OCEAN_UINT vertexBase; // first vertex of this tile in the global vertex buffer
    OCEAN_UINT stitch;     // OCEAN_EDGE_* bits
    OCEAN_UINT level;      // quadtree level, 0 = root
    OCEAN_UINT pad;
    // Earth curvature, expanded about the tile anchor so the quadratic stays in fp32 range:
    // drop(p) = curveBase + dot(curveGrad, p) + |p|^2 / 2R, with p local to the tile. Without it
    // the horizon sits at infinity and distant ships never go hull-down.
    float curveBase;
    OCEAN_FLOAT2 curveGrad;
    float invCurveRadius; // 1 / R, or zero when curvature is switched off
};

// Everything the shading path needs to reconstruct the surface at a world position. Uploaded once
// per frame; the per-cascade arrays are indexed by cascade.
struct OceanParamsGPU {
    OCEAN_FLOAT4 cascadeLength; // L per cascade, metres

    // Second-order Stokes crest sharpening, per cascade: the coefficient of the bound harmonic
    // that rides in phase with the crest, and the band's elevation variance that keeps the warp's
    // mean where it was. Solved on the host so the trough side never turns back up.
    OCEAN_FLOAT4 crestSkew;
    OCEAN_FLOAT4 cascadeVariance;

    // The floating origin moves in kilometre steps that are not multiples of the cascade periods,
    // so each cascade carries its own pre-wrapped origin. Folding the shift in on the host keeps
    // the shader's texture coordinates small and the waves pinned to absolute world space.
    OCEAN_FLOAT4 originWrapX;
    OCEAN_FLOAT4 originWrapZ;

    // Kilometre-scale sea-state field. The texture tiles over turbulencePeriod metres and its
    // gain is centred on one; turbulenceStrength fades the whole thing out at zero.
    float turbulencePeriod;
    float turbulenceStrength;
    // Horizontal displacement gain, already limited on the host so the composite map stays
    // clear of folding across the whole sea (see OceanSystem::ConditionedChoppiness).
    float choppiness;
    float surfaceY; // mean sea level in scene-relative coordinates

    OCEAN_FLOAT2 curveOrigin; // absolute scene origin, for Earth curvature
    float invRadius;
    float halfExtent; // finite ocean boundary in absolute XZ

    OCEAN_FLOAT3 originDelta; // current minus previous scene origin
    float historyValid;

    // Conservative bounds on the displaced height about surfaceY. A point above the crest bound is
    // in air and a point below the trough bound is in water without solving for the surface,
    // which is what lets shadow rays and the camera test skip the height field almost everywhere.
    float crestHeight;
    float troughDepth;
    // Diagnostics only: the ray footprint is multiplied by this before it picks a filter width.
    float filterScale;
    // Roughness floor the sun sampler and NEE widen the water surface to, and nothing else.
    // Trades highlight sharpness against the noise of the glitter track; see OceanOptics.hlsli.
    float sunLobeRoughness;

    OCEAN_UINT materialBase; // first slot of the OCEAN_MATERIAL_COUNT block
    OCEAN_UINT debugMode;    // OCEAN_DEBUG_*
    OCEAN_UINT dispParity;   // which displacement array holds this frame; the other one is history
    OCEAN_UINT pad;

    // The detail rule tile selection used, restated per point so the geometry's filter width is a
    // property of the surface position alone (see OceanGeometryWidth). Every tile that shares a
    // vertex then samples it identically, whatever its own level, which is what keeps stitched
    // edges watertight. Camera-relative basis of the widened view cone, scene-relative origin.
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

    // Whitecaps (OCEAN_SRV_FOAM0). The composite surface breaks where its Jacobian falls below the
    // level's breaking point (OceanFoamState), in full a little further below, and while it breaks
    // foam gathers at `foamRate` per second; everywhere it fades at `foamDecayRate`. The breaking
    // point is steered until the foam covers `foamCover` of the sea, and never lies above
    // `foamBreakMax`: a crest has to be squeezed at least that far to break at all.
    // Per foam level: xy the scene-relative xz of texel (0, 0)'s corner, zw the whole texels the
    // level moved since last frame.
    OCEAN_FLOAT4 foamLevel[OCEAN_FOAM_LEVELS];
    // Per foam level: xy the absolute world index of texel (0, 0), which the foam marbling is drawn
    // from (zw unused).
    OCEAN_INT4 foamCell[OCEAN_FOAM_LEVELS];
    float foamCover;
    float foamSteer; // per second, in log share per log cover error
    float foamRate;
    float foamDecayRate;  // natural-log decay per second
    float foamStrength;   // 0 switches foam off
    OCEAN_UINT frameIndex; // decorrelates the foam dither and sampling jitter from frame to frame
    OCEAN_UINT foamParity; // which foam texture holds this frame
    OCEAN_UINT foamHistory; // 0 when last frame's foam cannot be carried over
    OCEAN_FLOAT4 chopBand; // OceanChopGain: the short waves' extra horizontal displacement
    float foamBreakMax;
    OCEAN_FLOAT2 foamWind; // unit heading the wind blows towards, world xz
    float foamPad0;

    // Slope variance of everything a footprint of each width averages away - the ripples the mip
    // filtered out plus the capillary tail no cascade carries - which is what the hit turns into
    // roughness. See OCEAN_ROUGHNESS_ENTRIES.
    OCEAN_FLOAT4 residualSlope[OCEAN_ROUGHNESS_ENTRIES / 4];
};

// What the foam statistics pass keeps from frame to frame, at OCEAN_FOAM_STATE_OFFSET of
// OCEAN_UAV_FOAM_STATS; the host reads it back for the readouts.
struct OceanFoamState {
    float threshold[OCEAN_FOAM_LEVELS]; // Jacobian each level breaks below
    float softness[OCEAN_FOAM_LEVELS];  // how much further below that its crests break in full
    // Each level's breaking share relative to the reference level's. Their surfaces are filtered
    // to different widths and their foam drawn on different grids, so a breaking share paints foam
    // at a different rate on each; this is held at whatever makes a level lay the same cover as
    // the reference level on the ground they share. 1 on the reference level itself.
    float match[OCEAN_FOAM_LEVELS];
    float coverage[OCEAN_FOAM_LEVELS]; // each level's mean foam cover over its own ground
    float white;     // share of the reference level's ground plainly white (OceanFoamWhite)
    float share;     // share of the reference level the steering asks to break
    float breaking;  // share of it that does, which foamBreakMax can hold under that
    float target;    // the cover the steering last aimed at
    float decayRate; // and the foam's decay rate then
    float settle;    // seconds until the steering resumes after the foam started over
    OCEAN_UINT valid; // 0 until the first breaking points are set
};
