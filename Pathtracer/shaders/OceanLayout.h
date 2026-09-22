#pragma once
// Layout shared by the ocean host code and every ocean shader. Included from C++ (through
// rdn/ocean/OceanCommon.h) and from HLSL, so it must stay free of anything either side cannot
// parse: no namespaces, no templates, only #define and plain structs.
//
// The ocean is a spectral (Tessendorf) surface: a JONSWAP frequency spectrum with Donelan-Banner
// directional spreading is sampled into OCEAN_CASCADES tiling wavenumber bands, inverse-FFT'd
// every frame. Mesh displacement is filtered to geometry spacing. Shading uses full-resolution
// wave normals with zero default roughness. Only direct-light NEE widens its highlight lobe.
// Raw moment mips remain for conditioning/diagnostics.

// ---------------------------------------------------------------------------------------------
// Simulation
// ---------------------------------------------------------------------------------------------

// Cascades are overlapping rather than banded: each one synthesises every wavelength its grid can
// represent, and a partition of unity over wavenumber splits the energy between them. That keeps
// every grid densely populated - a hard band split leaves each cascade with only as many radial
// modes as the ratio between neighbouring tile lengths, which at four cascades is a handful of
// sinusoids and reads as obviously artificial. It also means a given wavelength is carried by
// several tiles of different, mutually incommensurate periods, which is what breaks up the
// repetition an FFT patch would otherwise show across a wide field of view.
#define OCEAN_CASCADES 4
#define OCEAN_MATERIAL_LEVELS 64
#define OCEAN_ANISO_LEVELS 4
#define OCEAN_DIRECTION_LEVELS 16
#define OCEAN_MATERIAL_COUNT (OCEAN_MATERIAL_LEVELS * OCEAN_ANISO_LEVELS * OCEAN_DIRECTION_LEVELS)

// 1024 samples per edge: 16 KiB shared memory and 512 butterfly threads per FFT line.
// Keep intermediate wave patches large enough to contain many distinct short-wave modes.
#ifndef OCEAN_FFT_SIZE
#define OCEAN_FFT_SIZE 1024
#endif
#ifndef OCEAN_FFT_LOG2
#define OCEAN_FFT_LOG2 10
#endif

// Mip levels on the displacement and derivative arrays: log2(N) + 1.
#define OCEAN_MIP_LEVELS (OCEAN_FFT_LOG2 + 1)

// Threads per line in the FFT passes; each one owns a single radix-2 butterfly.
#define OCEAN_FFT_THREADS (OCEAN_FFT_SIZE / 2)

// ---------------------------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------------------------

// Quads per tile edge, and the single number the ocean's per-frame cost turns on. It does not
// add acceleration-structure *builds* - the tile count does that - but it squares the triangles
// inside each one, and every resident tile is rebuilt or refitted every frame because its
// vertices move. At 128 a full budget of tiles is 16.8 million triangles of structure work per
// frame, which is most of what the ocean costs before a ray is traced; at 64 it is 4.2 million.
//
// What that buys is quad size: tile edge / this. At the default eight-metre finest tile, 64 puts
// quads at 12.5 cm. Waves finer than a couple of quads are carried by the slope variance and the
// BRDF rather than by geometry, which is what the filter width exists to arrange, so the detail
// is not lost so much as moved to where it is cheaper.
#define OCEAN_TILE_GRID 64
#define OCEAN_TILE_EDGE_VERTS (OCEAN_TILE_GRID + 1)
#define OCEAN_TILE_VERTS (OCEAN_TILE_EDGE_VERTS * OCEAN_TILE_EDGE_VERTS)
#define OCEAN_TILE_TRIS (OCEAN_TILE_GRID * OCEAN_TILE_GRID * 2)
#define OCEAN_TILE_INDICES (OCEAN_TILE_TRIS * 3)

#define OCEAN_MAX_TILES 512
#define OCEAN_FOAM_CASCADE 2
#define OCEAN_PREVIOUS_POSITION_SLOT 13

// Edge bits in OceanTileGPU.stitch: the neighbour across that edge is one level coarser, so the
// odd vertices along it collapse onto their even neighbours to close the crack.
#define OCEAN_EDGE_NEG_X 1u
#define OCEAN_EDGE_POS_X 2u
#define OCEAN_EDGE_NEG_Z 4u
#define OCEAN_EDGE_POS_Z 8u

// ---------------------------------------------------------------------------------------------
// Descriptor heap slots (the root signature sets CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED, so every
// ocean resource is reached through ResourceDescriptorHeap rather than a root parameter).
// ---------------------------------------------------------------------------------------------

#define OCEAN_HEAP_BASE 90u

#define OCEAN_SRV_DISP (OCEAN_HEAP_BASE + 0u)   // Texture2DArray<float4> (Dx, Dy, Dz, dDx/dz), mipped
#define OCEAN_SRV_DERIV (OCEAN_HEAP_BASE + 1u)  // Texture2DArray<float4> (dy/dx, dy/dz, dDx/dx, dDz/dz), mipped
#define OCEAN_SRV_VERTS (OCEAN_HEAP_BASE + 2u)  // StructuredBuffer<BTriVertex>
#define OCEAN_SRV_INDICES (OCEAN_HEAP_BASE + 3u)
#define OCEAN_SRV_PARAMS (OCEAN_HEAP_BASE + 4u) // StructuredBuffer<OceanParamsGPU>, one element
#define OCEAN_SRV_H0 (OCEAN_HEAP_BASE + 5u)
#define OCEAN_SRV_WAVE (OCEAN_HEAP_BASE + 6u)
#define OCEAN_SRV_TILES (OCEAN_HEAP_BASE + 7u) // StructuredBuffer<OceanTileGPU>

#define OCEAN_UAV_FFT (OCEAN_HEAP_BASE + 8u) // RWTexture2DArray<float4>, 2 complex pairs per slice
#define OCEAN_UAV_VERTS (OCEAN_HEAP_BASE + 9u)

// Whitecap coverage ping-pongs between two arrays. Both stay in the unordered-access state for
// the whole frame and the advection fetch is filtered by hand, which keeps the pair free of any
// per-frame resource transitions.
#define OCEAN_UAV_FOAM_A (OCEAN_HEAP_BASE + 10u)
#define OCEAN_UAV_FOAM_B (OCEAN_HEAP_BASE + 11u)

// Mip chains for the downsample pass: level L of the displacement array is at
// OCEAN_UAV_DISP_MIPS + L. Level 0 doubles as the assemble pass's output.
#define OCEAN_UAV_DISP_MIPS (OCEAN_HEAP_BASE + 12u)
#define OCEAN_UAV_DERIV_MIPS (OCEAN_UAV_DISP_MIPS + OCEAN_MIP_LEVELS)

// One float4 per cascade, written by the last level of the surface pyramid. The .z channel is
// the mean whitecap coverage over the whole tile, which the host reads back to steer the folding
// threshold toward the coverage the wind speed implies.
#define OCEAN_UAV_STATS (OCEAN_UAV_DERIV_MIPS + OCEAN_MIP_LEVELS)
#define OCEAN_SRV_MOMENTS (OCEAN_UAV_STATS + 1u)
#define OCEAN_SRV_SURFACE (OCEAN_UAV_STATS + 2u)
#define OCEAN_SRV_PREV_DISP (OCEAN_UAV_STATS + 3u)
#define OCEAN_UAV_MOMENT_MIPS (OCEAN_UAV_STATS + 4u)
#define OCEAN_UAV_SURFACE_MIPS (OCEAN_UAV_MOMENT_MIPS + OCEAN_MIP_LEVELS)

// Kilometre-scale sea-state field: a tiling fBm of the gain the wave field is multiplied by,
// with its own analytic gradient alongside it so the surface derivatives stay exact.
// (gain, d(gain)/dx, d(gain)/dz, unused), Texture2D<float4>, bilinear, wrapping.
#define OCEAN_SRV_TURBULENCE (OCEAN_UAV_SURFACE_MIPS + OCEAN_MIP_LEVELS)
#define OCEAN_TURBULENCE_SIZE 512

#define OCEAN_HEAP_COUNT ((OCEAN_SRV_TURBULENCE + 1u) - OCEAN_HEAP_BASE)

// ---------------------------------------------------------------------------------------------
// Shared records
// ---------------------------------------------------------------------------------------------

#ifdef __cplusplus
#define OCEAN_FLOAT2 DirectX::XMFLOAT2
#define OCEAN_FLOAT3 DirectX::XMFLOAT3
#define OCEAN_FLOAT4 DirectX::XMFLOAT4
#define OCEAN_UINT uint32_t
#else
#define OCEAN_FLOAT2 float2
#define OCEAN_FLOAT3 float3
#define OCEAN_FLOAT4 float4
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
    OCEAN_FLOAT4 cascadeKMin;   // band start, rad/m
    OCEAN_FLOAT4 cascadeKMax;   // band end, rad/m
    OCEAN_FLOAT4 slopeVarAlong; // along-wind slope variance carried by each cascade
    OCEAN_FLOAT4 slopeVarCross; // cross-wind slope variance carried by each cascade

    // Second-order Stokes crest sharpening, per cascade: the coefficient of the bound harmonic
    // that rides in phase with the crest, and the band's elevation variance that keeps the warp's
    // mean where it was. Solved on the host so the trough side never turns back up.
    OCEAN_FLOAT4 crestSkew;
    OCEAN_FLOAT4 cascadeVariance;

    // Kilometre-scale sea-state field. The texture tiles over turbulencePeriod metres and its
    // gain is centred on one; turbulenceStrength fades the whole thing out at zero.
    float turbulencePeriod;
    float turbulenceStrength;
    // Largest gain the field can reach. The composite deformation bound is divided by it, so a
    // patch that gets the full boost still cannot fold the surface into itself.
    float turbulenceMaxGain;
    // Fraction of the no-fold threshold the conditioning pass may spend, after that division.
    float deformationBudget;

    // The floating origin moves in kilometre steps that are not multiples of the cascade periods,
    // so each cascade carries its own pre-wrapped origin. Folding the shift in on the host keeps
    // the shader's texture coordinates small and the waves pinned to absolute world space.
    OCEAN_FLOAT4 originWrapX;
    OCEAN_FLOAT4 originWrapZ;

    OCEAN_FLOAT2 windDir;    // unit vector in XZ
    float slopeVarTailAlong; // variance above the finest cascade's Nyquist
    float slopeVarTailCross;

    OCEAN_FLOAT3 upwelling; // diffuse reflectance of the water body, from its absorption and
                            // backscattering coefficients
    float surfaceY;         // mean sea level in scene-relative coordinates

    float bodyStrength;

    // Jacobian below which a cascade counts as folded, and how sharply coverage ramps up past it.
    // Both are solved per cascade at bake time so that the combined whitecap coverage matches the
    // measured fraction for the current wind speed.
    OCEAN_FLOAT4 foamThreshold;
    OCEAN_FLOAT4 foamSharpness;

    OCEAN_FLOAT3 foamAlbedo;
    float foamCoverageScale; // multiplies the prefiltered coverage before shading

    float choppiness;
    float displacementScale;

    // Filter width: the ray footprint is multiplied by this before it picks the cut wavenumber.
    // Above 1 the surface filters harder (smoother, safer); below 1 it keeps more geometry detail.
    float filterScale;
    float foamRoughness;
    // Roughness floor the sun sampler and NEE widen the water surface to, and nothing else.
    // Trades highlight sharpness against the noise of the glitter track; see OceanOptics.hlsli.
    float sunLobeRoughness;
    float waveHeightScale;

    float invRadius;
    OCEAN_FLOAT2 curveOrigin; // absolute scene origin, for Earth curvature
    OCEAN_UINT materialBase; // contiguous water/whitecap coverage materials
    OCEAN_UINT debugMode;
    OCEAN_FLOAT3 originDelta; // current minus previous scene origin
    float historyValid;
    float bodyWeight; // optional authored diffuse surface opacity
    float halfExtent; // finite ocean boundary in absolute XZ
};




