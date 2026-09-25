#pragma once
#include "OceanLayout.h"
#include "OceanMath.hlsli"

// Shared material-coordinate surface evaluation. Raw derivatives are summed before inversion.
// Geometry is filtered to its mesh spacing and shading to the ray footprint, whose unresolved
// ripples become roughness; direct-light evaluation additionally widens its highlight lobe.
OceanParamsGPU OceanParams() {
    StructuredBuffer<OceanParamsGPU> b = ResourceDescriptorHeap[OCEAN_SRV_PARAMS];
    return b[0];
}
uint OceanDispSlot(OceanParamsGPU P) {
    return OCEAN_SRV_DISP0 + (P.dispParity & 1u);
}
uint OceanPrevDispSlot(OceanParamsGPU P) {
    return OCEAN_SRV_DISP0 + ((P.dispParity & 1u) ^ 1u);
}
float2 OceanCascadeUV(float2 q, OceanParamsGPU P, uint c) {
    // FFT node i represents i*L/N; texture texel i is centred at (i+0.5)/N.
    return (q + float2(P.originWrapX[c], P.originWrapZ[c])) / P.cascadeLength[c] + 0.5f / OCEAN_FFT_SIZE;
}
float OceanCascadeMip(float widthM, float L) {
    return clamp(log2(max(widthM * OCEAN_FFT_SIZE / L, 1.0f)), 0.0f, OCEAN_MIP_LEVELS - 1.0f);
}

// Filter width the geometry is displaced with at scene-relative sea position q: the selection's
// detail rule (ocean::LodRule) restated per point. It depends on the position alone, never on the
// tile doing the sampling, so every tile that shares a vertex - across a level change, at a
// corner where four meet - computes the same displacement for it and no crack can open between
// them. It is also continuous, so the geometry smooths gradually with distance rather than
// stepping at tile boundaries.
float OceanGeometryWidth(float2 q, OceanParamsGPU P) {
    const float3 v = float3(q.x, P.surfaceY, q.y) - P.lodCamera;
    const float d = length(v);
    const float z = dot(v, P.lodForward);
    const float outside = z > 1e-3f ? max(abs(dot(v, P.lodRight)) / (z * P.lodTanH),
                                          abs(dot(v, P.lodUp)) / (z * P.lodTanV)) : 1e6f;
    // Both blends finish where selection's own tests switch to coarse tiles (the widened cone
    // is `outside` = 1), so no coarse tile is ever displaced with the fine width - that would
    // point-sample detail its vertices cannot hold. The price is some extra smoothing on the fine
    // tiles in the margin, which the camera does not see.
    const float offscreen = smoothstep(0.8f, 1.0f, outside) *
                            saturate((d - 0.67f * P.lodNearKeep) / (0.33f * P.lodNearKeep + 1e-3f));
    float ratio = P.lodRatio * lerp(1.0f, P.lodOffscreen, offscreen);
    if (P.lodSilhouette > 0.0f)
        ratio *= max(1.0f, d / P.lodSilhouette);
    return max(ratio * d, P.lodMinWidth);
}

// Levels to the coarser neighbour across one edge (OCEAN_EDGE_*).
uint OceanStitchStep(uint stitch, uint edge) {
    return (stitch >> (OCEAN_STITCH_BITS * edge)) & OCEAN_STITCH_MASK;
}

// The grid points a tile vertex is built from. Along an edge whose neighbour is k levels coarser,
// that neighbour's vertices fall on every 2^k-th of ours; the ones in between are placed on its
// edge by interpolating the two it lies between, which closes the crack exactly. Everything else
// is its own grid point (lo == hi, w == 0).
struct OceanVertexSource {
    uint2 lo;
    uint2 hi;
    float w;
};
OceanVertexSource OceanStitchSource(uint i, uint j, uint stitch) {
    const uint G = OCEAN_TILE_GRID;
    OceanVertexSource s;
    s.lo = uint2(i, j);
    s.hi = s.lo;
    s.w = 0.0f;
    uint k = 0u;
    bool alongZ = false;
    if (i == 0u || i == G) {
        k = OceanStitchStep(stitch, i == 0u ? OCEAN_EDGE_NEG_X : OCEAN_EDGE_POS_X);
        alongZ = true;
    }
    if (k == 0u && (j == 0u || j == G)) {
        k = OceanStitchStep(stitch, j == 0u ? OCEAN_EDGE_NEG_Z : OCEAN_EDGE_POS_Z);
        alongZ = false;
    }
    if (k == 0u)
        return s;
    const uint along = alongZ ? j : i;
    const uint span = 1u << k;
    const uint t0 = along & ~(span - 1u);
    if (t0 == along)
        return s;
    const uint t1 = min(t0 + span, G);
    if (alongZ) { s.lo.y = t0; s.hi.y = t1; }
    else        { s.lo.x = t0; s.hi.x = t1; }
    s.w = float(along - t0) / float(span);
    return s;
}
// Kilometre-scale sea state at this position: the gain the whole wave field is multiplied by,
// with its world-space gradient in yz. A real ocean has patches of steeper, more broken water
// drifting between calmer lanes; this is that field, baked once and tiled.
float3 OceanTurbulence(float2 q, OceanParamsGPU P) {
    if (P.turbulenceStrength <= 0.0f)
        return float3(1.0f, 0.0f, 0.0f);
    Texture2D<float4> field = ResourceDescriptorHeap[OCEAN_SRV_TURBULENCE];
    const float2 absoluteQ = q + P.curveOrigin;
    return field.SampleLevel(g_sampler, absoluteQ / P.turbulencePeriod, 0.0f).xyz;
}

// Everything one surface query needs, fetched in a single walk over the cascades: the displaced
// position and the height gradient and horizontal stretch the normal is built from. The pieces
// used to be separate walks over the same texels.
struct OceanSample {
    float3 displacement;
    float2 gradient;
    float3 stretch; // (1 + dDx/dx, 1 + dDz/dz, dDx/dz)
    float gain;     // kilometre-scale sea-state gain at the point
};

OceanSample OceanSampleSurfaceFrom(float2 q, float widthM, uint dispSlot, bool derivatives) {
    const OceanParamsGPU P = OceanParams();
    Texture2DArray<float4> disp = ResourceDescriptorHeap[dispSlot];
    Texture2DArray<float4> deriv = ResourceDescriptorHeap[OCEAN_SRV_DERIV];
    // The sea-state gain varies across the surface, so the product rule contributes wherever it
    // does. Leaving those terms out would tilt the shading normal away from the geometry along
    // every patch boundary.
    const float3 turbulence = OceanTurbulence(q, P);
    const float gain = turbulence.x;
    const float2 dGain = turbulence.yz;

    OceanSample s;
    s.displacement = 0.0f;
    s.gradient = 0.0f;
    s.stretch = float3(1.0f, 1.0f, 0.0f);
    s.gain = gain;
    [loop] for (uint c = 0; c < OCEAN_CASCADES; ++c) {
        const float3 uv = float3(OceanCascadeUV(q, P, c), c);
        const float mip = OceanCascadeMip(widthM, P.cascadeLength[c]);
        const float4 h = disp.SampleLevel(g_sampler, uv, mip);
        // Each band is sharpened on its own elevation, so the small chop peaks like the swell does
        // rather than riding along as a symmetric ripple. The sea-state gain multiplies the band
        // first, so a rough patch is both taller and - the warp being quadratic - more peaked.
        const float skew = P.crestSkew[c];
        const float variance = P.cascadeVariance[c] * gain * gain;
        s.displacement.xz += h.xz * gain;
        s.displacement.y += OceanSkewHeight(h.y * gain, skew, variance);
        if (derivatives) {
            const float4 d = deriv.SampleLevel(g_sampler, uv, mip);
            const float slope = OceanSkewSlope(h.y * gain, skew);
            s.gradient += slope * gain * d.xy + (slope * h.y - 2.0f * skew * gain * P.cascadeVariance[c]) * dGain;
            // Horizontal displacement is scaled by the same field, so its strain picks up the
            // gain's own gradient against the displacement it is scaling.
            s.stretch += float3(gain * d.z + h.x * dGain.x, gain * d.w + h.z * dGain.y, gain * h.w + h.x * dGain.y);
        }
    }
    return s;
}
OceanSample OceanSampleSurface(float2 q, float widthM) {
    return OceanSampleSurfaceFrom(q, widthM, OceanDispSlot(OceanParams()), true);
}

float3 OceanDisplacementFrom(float2 q, float widthM, uint dispSlot) {
    return OceanSampleSurfaceFrom(q, widthM, dispSlot, false).displacement;
}
float3 OceanDisplacement(float2 q, float widthM) {
    return OceanDisplacementFrom(q, widthM, OceanDispSlot(OceanParams()));
}
float3 OceanNormalFromSample(OceanSample s, float2 curveGradient) {
    const float2 slope = OceanWarpedSlope(s.gradient - curveGradient, s.stretch);
    return normalize(float3(-slope.x, 1, -slope.y));
}

struct OceanSurface {
    float3 normal;
    float3 albedo;
    float roughness;
    uint materialOffset;
    // What the material step is dithered from, which the step only matches on average: the foam
    // cover and bubble-cloud density. The reconstruction's albedo guide is written from these
    // (OceanGuideAlbedo), so it holds still from frame to frame.
    float foam;
    float bubbles;
};

float3 OceanHue(float t) {
    return saturate(abs(frac(t + float3(0.0f, 2.0f / 3.0f, 1.0f / 3.0f)) * 6.0f - 3.0f) - 1.0f);
}

// Slope variance a footprint of this width averages away (OceanParamsGPU.residualSlope).
float OceanResidualSlope(float widthM, OceanParamsGPU P) {
    const float t = clamp(log2(max(widthM, 1e-6f)) - (float)OCEAN_ROUGHNESS_OFFSET, 0.0f,
                          (float)(OCEAN_ROUGHNESS_ENTRIES - 1));
    const uint i0 = (uint)t;
    const uint i1 = min(i0 + 1u, (uint)OCEAN_ROUGHNESS_ENTRIES - 1u);
    return lerp(P.residualSlope[i0 >> 2u][i0 & 3u], P.residualSlope[i1 >> 2u][i1 & 3u], t - (float)i0);
}

uint OceanHash(uint x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}

// Gradient noise over the unit lattice, about -0.7 to 0.7.
float OceanFoamGrad(float2 p, uint seed) {
    const float2 i = floor(p), f = p - i;
    const float2 u = f * f * f * (f * (f * 6.0f - 15.0f) + 10.0f);
    float n[4];
    [unroll] for (uint k = 0u; k < 4u; ++k) {
        const float2 c = float2(k & 1u, k >> 1u);
        const uint h = OceanHash((uint)(int)(i.x + c.x) * 0x8DA6B343u ^ OceanHash((uint)(int)(i.y + c.y) * 0xD8163841u + seed));
        const float2 g = float2(h & 0xFFFFu, h >> 16) / 32768.0f - 1.0f;
        n[k] = dot(g, f - c);
    }
    return lerp(lerp(n[0], n[1], u.x), lerp(n[2], n[3], u.x), u.y);
}

// World xz into the frame the foam's web is laid out in: x along the wind, z across it.
float2 OceanFoamFrame(float2 world, float2 wind) {
    return float2(dot(world, wind), dot(world, float2(-wind.y, wind.x)));
}

// Whitecaps over a footprint, after Crest (wave-harmonic/crest, MIT). The simulation says how much
// foam there is (OceanFoam): a cap builds while its crest breaks, and the cover it leaves fades
// over seconds. How that cover looks is drawn here from the cover alone - a fresh cap solid white,
// a fading raft a sheet that holes open in, then a lace of strands (OceanFoamRaft over
// OceanFoamWeb). No texture is laid over the water, and nothing on it repeats.
//
// Under a young cap is the bubble cloud the break drove down, which lights the water turquoise
// around and behind the white. It is a density here, for the material block (WriteMaterialSlot),
// which returns it from under the water's own clear surface. Crest paints it on the surface
// instead; over real volume scattering that read as foam floating inside the water.
struct OceanFoam {
    float coverage; // step of the foam ramp, 0 clear water to 1 solid foam
    float bubbles;  // density of the bubble cloud under the surface, 0 to 1 fresh
    float2 slope;   // surface slope of the raft's lumps, world xz
};
static const float OCEAN_FOAM_RELIEF = 0.03f;   // metres of relief from bare water to solid foam
static const float OCEAN_FOAM_ROUGHNESS = 1.0f; // perceptual roughness of solid foam's surface
static const float OCEAN_FOAM_CLOUD = 0.4f;     // metres the bubble cloud spreads round its cap

float OceanFoamTexel(uint level) {
    return OCEAN_FOAM_TEXEL0 * exp2((float)level);
}

// A raft of foam at a point where its web (OceanFoamWeb) reads u and its alpha map (the bubbles and
// clumps it is made of, OceanFoamDetail) reads `detail`, for an amount from 0 to 1. The web sets how
// much of the plane is foam - a share that rises from nothing to OCEAN_FOAM_OPEN over `feather` of
// u above a point that falls as the amount rises - and the alpha map where: foam wherever it reads
// above that share's complement, so as the share falls towards a raft's sides and as it dissolves,
// the foam breaks up into ever sparser clumps and specks of bubbles instead of ending at an edge.
// Even the densest raft keeps its biggest bubbles open. `detailed` is how much of the alpha map the
// footprint resolves; where it resolves none the share is drawn as it is. How white the foam is
// goes with the amount to the power OCEAN_FOAM_DIM, so a thin raft is a faint one too, and with the
// web: foam is piled thickest along the strands that hold on longest and thinner between them
// (OCEAN_FOAM_GRAIN). Under it all is a faint milky film, densest against the strands, fading with
// the amount's square. A fresh raft is dense white, a fading one a sheet with holes in it, then a
// lace of ever sparser, fainter specks. Returns the share of the point foam covers; a raft never
// covers more than OCEAN_FOAM_DENSEST.
static const float OCEAN_FOAM_FILM = 0.15f;
static const float OCEAN_FOAM_GRAIN = 0.3f;
static const float OCEAN_FOAM_DIM = 0.35f;
static const float OCEAN_FOAM_OPEN = 0.9f; // share of the plane even the densest raft covers
static const float OCEAN_FOAM_EDGE = 0.3f; // how softly the alpha map is cut, as a share of its range
static const float OCEAN_FOAM_DENSEST = OCEAN_FOAM_OPEN * (1.0f - 0.5f * OCEAN_FOAM_GRAIN) +
                                        OCEAN_FOAM_FILM * (1.0f - OCEAN_FOAM_OPEN + OCEAN_FOAM_OPEN * OCEAN_FOAM_GRAIN / 3.0f);
float OceanFoamRaft(float u, float amount, float feather, float detail, float detailed) {
    const float a = saturate(amount);
    const float share = saturate((u - 1.0f + a * (1.0f + feather)) / feather) * OCEAN_FOAM_OPEN;
    // A linear cut centred on the threshold and narrowed towards either end of the range, so that
    // over an alpha map uniform on [0, 1] exactly `share` of it is foam.
    const float edge = max(2.0f * OCEAN_FOAM_EDGE * min(share, 1.0f - share), 1e-4f);
    const float alpha = lerp(share, saturate((detail - 1.0f + share) / edge + 0.5f), detailed);
    const float web = alpha * pow(a, OCEAN_FOAM_DIM) * (1.0f - OCEAN_FOAM_GRAIN * (1.0f - u));
    return web + (1.0f - web) * OCEAN_FOAM_FILM * a * a * 2.0f * u;
}
// Its mean over u and the alpha map, both uniform on [0, 1] and independent: the cover a raft of
// this amount lays down. With r the web's ramp, E[r], E[r u] and E[r u^2] in closed form.
float OceanFoamMean(float amount, float feather) {
    const float a = saturate(amount), b = 1.0f - a * (1.0f + feather);
    const float lo = max(b, 0.0f), hi = min(b + feather, 1.0f);
    const float lo2 = lo * lo, hi2 = hi * hi, lo3 = lo2 * lo, hi3 = hi2 * hi;
    const float r0 = ((hi - b) * (hi - b) - (lo - b) * (lo - b)) / (2.0f * feather) + (1.0f - hi);
    const float r1 = ((hi3 - lo3) / 3.0f - 0.5f * b * (hi2 - lo2)) / feather + 0.5f * (1.0f - hi2);
    const float r2 = ((hi3 * hi - lo3 * lo) / 4.0f - b * (hi3 - lo3) / 3.0f) / feather + (1.0f - hi3) / 3.0f;
    const float s = OCEAN_FOAM_OPEN * pow(a, OCEAN_FOAM_DIM), film = OCEAN_FOAM_FILM * a * a, g = OCEAN_FOAM_GRAIN;
    const float web = s * ((1.0f - g) * r0 + g * r1);
    const float webU = s * ((1.0f - g) * r1 + g * r2);
    return web + film - 2.0f * film * webU;
}
// The amount whose raft lays down this cover. The mean rises with the amount throughout, so
// halving the interval finds it.
float OceanFoamAmount(float cover, float feather) {
    float lo = 0.0f, hi = 1.0f;
    [unroll] for (uint i = 0u; i < 12u; ++i) {
        const float mid = 0.5f * (lo + hi);
        if (OceanFoamMean(mid, feather) < cover) lo = mid;
        else hi = mid;
    }
    return 0.5f * (lo + hi);
}

// Distance from p to the nearest wall between the holes bursting from the sites of a jittered unit
// lattice - F2 - F1 of the distances to its two nearest sites: 0 on a wall, up to about 0.7 at a
// site. The holes do not all start at once: each site's distance carries a random head start of up
// to OCEAN_FOAM_HEADSTART cells (an additively weighted Voronoi diagram), so the cells come in
// every size and their walls curve, where equal cells drew a net of straight-edged meshes.
static const float OCEAN_FOAM_HEADSTART = 0.8f;
float OceanFoamWalls(float2 p, uint seed) {
    const float2 i = floor(p);
    float f1 = 9.0f, f2 = 9.0f;
    [unroll] for (uint k = 0u; k < 9u; ++k) {
        const float2 c = i + float2(float(k % 3u) - 1.0f, float(k / 3u) - 1.0f);
        const uint h = OceanHash((uint)(int)c.x * 0x8DA6B343u ^ OceanHash((uint)(int)c.y * 0xD8163841u + seed));
        const float d = length(c + float2(h & 0xFFFFu, h >> 16) / 65536.0f - p) +
                        OCEAN_FOAM_HEADSTART * (OceanHash(h) & 0xFFFFu) / 65536.0f;
        f2 = min(f2, max(f1, d));
        f1 = min(f1, d);
    }
    return f2 - f1;
}
// Measured distributions' quantiles in steps of 1/16 (from millions of samples), and a value's
// place in one of them, 0 to 1: the web's walls alone and perforated (OceanFoamWeb), and the alpha
// map standardised (OceanFoamDetail).
static const float kFoamWebCoarse[17] = {-1.7611f, -0.7273f, -0.5949f, -0.5062f, -0.4366f, -0.3785f, -0.3283f, -0.2835f, -0.2429f,
                                         -0.2055f, -0.1709f, -0.1386f, -0.1080f, -0.0791f, -0.0516f, -0.0247f, 0.0635f};
static const float kFoamWebFine[17] = {-1.9822f, -0.8286f, -0.6938f, -0.6035f, -0.5334f, -0.4748f, -0.4240f, -0.3787f, -0.3377f,
                                       -0.2998f, -0.2641f, -0.2299f, -0.1962f, -0.1621f, -0.1256f, -0.0832f, 0.0595f};
static const float kFoamDetailZ[17] = {-3.0890f, -1.5063f, -1.1640f, -0.9176f, -0.7104f, -0.5274f, -0.3553f, -0.1904f, -0.0277f,
                                       0.1364f, 0.3047f, 0.4836f, 0.6801f, 0.9060f, 1.1827f, 1.5851f, 4.2297f};
float OceanFoamQuantile(uint table, uint i) {
    return table == 0u ? kFoamWebCoarse[i] : table == 1u ? kFoamWebFine[i] : kFoamDetailZ[i];
}
float OceanFoamRank(float v, uint table) {
    uint i = 0u;
    [unroll] for (uint step = 8u; step > 0u; step >>= 1u) {
        if (i + step <= 16u && OceanFoamQuantile(table, i + step) <= v)
            i += step;
    }
    i = min(i, 15u);
    const float q0 = OceanFoamQuantile(table, i), q1 = OceanFoamQuantile(table, i + 1u);
    return saturate((float(i) + saturate((v - q0) / (q1 - q0))) / 16.0f);
}

// A lattice of domes: each unit cell holds one at a random place, a fifth to three quarters of a
// cell across. Returns how deep into a dome cell + f lies - 1 at its middle, 0 at its rim, and
// falling towards -1 between domes, so the value has no flat stretch to tie on - and in `dome` the
// offset from there to that dome's middle over its radius, the slope of the dome up to a scale, or
// nothing between domes.
static const float OCEAN_DOME_MEAN = 0.1788f; // the value's mean over the plane (two million samples)
static const float OCEAN_DOME_STD = 0.5578f;  // and its standard deviation
float OceanFoamHollow(int2 cell, float2 f, uint seed, out float2 dome) {
    float hollow = -1e30f;
    float2 best = 0.0f;
    [unroll] for (uint k = 0u; k < 9u; ++k) {
        const int2 o = int2(int(k % 3u) - 1, int(k / 3u) - 1);
        const int2 c = cell + o;
        const uint h = OceanHash((uint)c.x * 0x8DA6B343u ^ OceanHash((uint)c.y * 0xD8163841u + seed));
        const float2 d = float2(o) + float2(h & 0xFFFFu, h >> 16) / 65536.0f - f;
        const float r = 0.2f + 0.55f * (OceanHash(h) & 0xFFFFu) / 65536.0f;
        const float v = 1.0f - dot(d, d) / (r * r);
        if (v > hollow) {
            hollow = v;
            best = d / r;
        }
    }
    dome = hollow > 0.0f ? best : 0.0f;
    return hollow >= 0.0f ? hollow : exp(hollow) - 1.0f;
}
// The alpha map of a raft: where in it foam holds on longest. Foam is a pile of bubbles from a
// millimetre to a few centimetres across, whipped by the breaking crest into clumps that are left
// behind drawn out into streaks; as it thins, the middles of the biggest bubbles open first, then
// whole bubbles and the gaps between the clumps, until only specks are left. Five lattices of domes
// (OceanFoamHollow): bubbles in cells of 6, 3 and 1.3 cm, and clumps in cells of 12 and 30 cm three
// times as long downwind, each drawn where the footprint resolves it, summed and standardised by the
// spread of what was drawn, then mapped through the full sum's measured distribution so the alpha
// map comes out uniform on [0, 1] (`detail`, high on the walls between bubbles, which hold on
// longest). `detailed` is how much of it the footprint resolves. The slope of the domes is added to
// `slope` as a height gradient in world xz. The bubbles are indexed from level 0's absolute texel,
// so they are exact however far out the sea runs; `w` is in the frame of OceanFoamFrame, `wind` its
// heading.
static const uint kBubblesPerTexel[3] = {1u, 2u, 5u};
static const float kBubbleWeight[3] = {0.9f, 0.9f, 0.8f};
static const float kClumpCell[2] = {0.12f, 0.3f};
static const float kClumpWeight[2] = {0.65f, 0.55f};
static const float OCEAN_CLUMP_STRETCH = 3.0f;
static const float OCEAN_BUBBLE_DOME = 0.45f; // height gradient at a bubble's rim
static const float OCEAN_CLUMP_DOME = 0.35f;  // and at a clump's
float OceanFoamDetail(float2 q, float2 w, float2 wind, float widthM, OceanParamsGPU P, inout float2 slope, out float detailed) {
    float sum = 0.0f, drawn = 0.0f;
    [unroll] for (uint l = 0u; l < 3u; ++l) {
        const float keep = saturate(2.0f - 2.0f * widthM * kBubblesPerTexel[l] / OCEAN_FOAM_TEXEL0);
        if (keep > 0.0f) {
            const float2 t = (q - P.foamLevel[0].xy) * (float(kBubblesPerTexel[l]) / OCEAN_FOAM_TEXEL0);
            const float2 i = floor(t);
            float2 dome;
            const float hollow = OceanFoamHollow(P.foamCell[0].xy * int(kBubblesPerTexel[l]) + int2(i), t - i,
                                                 71u + l, dome);
            sum += keep * kBubbleWeight[l] * (hollow - OCEAN_DOME_MEAN);
            drawn += keep * keep * kBubbleWeight[l] * kBubbleWeight[l];
            slope += keep * OCEAN_BUBBLE_DOME * dome;
        }
    }
    [unroll] for (uint c = 0u; c < 2u; ++c) {
        const float keep = saturate(2.0f - 2.0f * widthM / kClumpCell[c]);
        if (keep > 0.0f) {
            const float2 p = float2(w.x / OCEAN_CLUMP_STRETCH, w.y) / kClumpCell[c];
            const float2 i = floor(p);
            float2 dome;
            const float hollow = OceanFoamHollow(int2(i), p - i, 81u + c, dome);
            sum += keep * kClumpWeight[c] * (hollow - OCEAN_DOME_MEAN);
            drawn += keep * keep * kClumpWeight[c] * kClumpWeight[c];
            const float2 g = keep * OCEAN_CLUMP_DOME * float2(dome.x / OCEAN_CLUMP_STRETCH, dome.y);
            slope += g.x * wind + g.y * float2(-wind.y, wind.x);
        }
    }
    const float spread = sqrt(drawn);
    // Resolved in full while the coarsest clumps are, fading out with them.
    detailed = saturate(spread / (0.8f * kClumpWeight[1]));
    return spread > 1e-4f ? OceanFoamRank(-sum / (OCEAN_DOME_STD * spread), 2u) : 0.5f;
}
// How thick a trail's foam lies from place to place, one on average: the patches and streaks it
// breaks up into as it is drawn out behind its crest, the same domes (OceanFoamHollow) in cells of
// 0.8 and 2.4 m, three times as long downwind. Laid on the cover itself, so they carry to any
// distance a footprint resolves them at. `w` is in the frame of OceanFoamFrame.
static const float kPatchCell[2] = {0.8f, 2.4f};
static const float kPatchStrength[2] = {0.45f, 0.4f};
float OceanFoamPatches(float2 w, float widthM) {
    float patches = 1.0f;
    [unroll] for (uint l = 0u; l < 2u; ++l) {
        const float keep = saturate(2.0f - 2.0f * widthM / kPatchCell[l]);
        if (keep > 0.0f) {
            const float2 p = float2(w.x / OCEAN_CLUMP_STRETCH, w.y) / kPatchCell[l];
            const float2 i = floor(p);
            float2 dome;
            patches *= 1.0f + keep * kPatchStrength[l] * (OCEAN_DOME_MEAN - OceanFoamHollow(int2(i), p - i, 83u + l, dome));
        }
    }
    return patches;
}

// Where a dissolving raft holds on longest, as u about uniform on [0, 1], at `w` in the frame of
// OceanFoamFrame. A raft drains and bursts in holes that grow until they meet, so what is left is a
// web along the walls between them: cells of about a metre, drawn out downwind and warped so no
// wall runs straight, their strands perforated by holes a fifth that size, which leaves their edges
// ragged and breaks them up at the end. The walls are mapped through their measured distribution
// (four million samples), so a raft drawn over u averages the cover it was asked for.
//
// Only what the footprint resolves is drawn. The perforation falls away first, where a footprint
// outgrows its strands; the raft is feathered over OCEAN_FOAM_FEATHER of u, or what the footprint
// spans if that is more (`feather`); and `resolved`, how much of the web the footprint still holds,
// fades the whole of it out once it spans a good part of a cell. A feather of a sixth of u cut
// every raft off at a hard edge, however thin it had got.
static const float OCEAN_FOAM_CELL = 0.6f;      // metres across a cell of the web
static const float OCEAN_FOAM_PERFORATION = 0.15f; // and across a hole in its strands
static const float OCEAN_FOAM_STRETCH = 2.0f;   // how much longer the cells run downwind
static const float OCEAN_FOAM_FEATHER = 0.55f;  // share of u the raft's foam rises over
float OceanFoamWeb(float2 w, float widthM, out float resolved, out float feather) {
    float2 p = float2(w.x / OCEAN_FOAM_STRETCH, w.y);
    p += (0.5f / 1.5f) * float2(OceanFoamGrad(p * 0.4f, 11u) + 0.5f * OceanFoamGrad(p * 0.8f, 12u),
                                OceanFoamGrad(p * 0.4f, 17u) + 0.5f * OceanFoamGrad(p * 0.8f, 18u));
    float2 q = p / OCEAN_FOAM_CELL;
    q += 0.45f * float2(OceanFoamGrad(q * 1.5f, 21u), OceanFoamGrad(q * 1.5f, 27u));
    const float walls = OceanFoamWalls(q, 41u) + 0.1f * OceanFoamGrad(q * 4.0f, 51u);
    float u = OceanFoamRank(-walls, 0u);
    const float perforated = saturate(2.0f - 4.0f * widthM / OCEAN_FOAM_PERFORATION);
    if (perforated > 0.0f) {
        float2 r = p / OCEAN_FOAM_PERFORATION;
        r += 0.45f * float2(OceanFoamGrad(r * 1.5f, 31u), OceanFoamGrad(r * 1.5f, 37u));
        const float holes = OceanFoamWalls(r, 43u) + 0.1f * OceanFoamGrad(r * 4.0f, 61u);
        u = lerp(u, OceanFoamRank(-(walls + 0.3f * holes), 1u), perforated);
    }
    // u runs across its range over about half a cell, so a footprint spans about 2 / cell of it.
    feather = clamp(2.0f * widthM / OCEAN_FOAM_CELL, OCEAN_FOAM_FEATHER, 1.0f);
    resolved = saturate(2.0f - widthM / (0.15f * OCEAN_FOAM_CELL));
    return u;
}

// Cubic B-spline through four bilinear taps, for a level seen closer than its texels: a strand a
// couple of texels wide then reads as a smooth line rather than the chain of diamonds bilinear
// filtering draws it as.
float OceanFoamSmooth(Texture2DArray<float> foam, float2 uv, uint level) {
    const float2 t = uv * OCEAN_FOAM_SIZE - 0.5f;
    const float2 base = floor(t), f = t - base;
    const float2 f2 = f * f, f3 = f2 * f;
    const float2 w0 = (1.0f - 3.0f * f + 3.0f * f2 - f3) / 6.0f;
    const float2 w1 = (4.0f - 6.0f * f2 + 3.0f * f3) / 6.0f;
    const float2 w2 = (1.0f + 3.0f * f + 3.0f * f2 - 3.0f * f3) / 6.0f;
    const float2 w3 = f3 / 6.0f;
    const float2 g0 = w0 + w1, g1 = w2 + w3;
    const float2 c0 = (base - 0.5f + w1 / g0) / OCEAN_FOAM_SIZE;
    const float2 c1 = (base + 1.5f + w3 / g1) / OCEAN_FOAM_SIZE;
    return g0.y * (g0.x * foam.SampleLevel(g_sampler, float3(c0.x, c0.y, level), 0.0f) +
                   g1.x * foam.SampleLevel(g_sampler, float3(c1.x, c0.y, level), 0.0f)) +
           g1.y * (g0.x * foam.SampleLevel(g_sampler, float3(c0.x, c1.y, level), 0.0f) +
                   g1.x * foam.SampleLevel(g_sampler, float3(c1.x, c1.y, level), 0.0f));
}

// `gain` is the kilometre-scale sea-state gain at q (OceanSample.gain).
OceanFoam OceanFoamAt(float2 q, float widthM, float gain, OceanParamsGPU P) {
    OceanFoam f;
    f.coverage = 0.0f;
    f.bubbles = 0.0f;
    f.slope = 0.0f;
    if (P.foamStrength <= 0.0f)
        return f;
    // The foam cover over this footprint, from the finest level that holds this point, blended
    // into the next coarser one across its outer part so no edge shows. Cover averages, so the mip
    // and the blend both keep the white the footprint holds. Past the coarsest level the foam is
    // too small to resolve anyway, and its mean stands in for it. The bubble cloud is the same
    // cover spread over some forty centimetres round it.
    Texture2DArray<float> foam = ResourceDescriptorHeap[(P.foamParity & 1u) == 0u ? OCEAN_SRV_FOAM0 : OCEAN_SRV_FOAM1];
    float cover = 0.0f, cloud = 0.0f, rest = 1.0f;
    uint finest = OCEAN_FOAM_LEVELS;
    [loop] for (uint level = 0u; level < OCEAN_FOAM_LEVELS && rest > 1e-3f; ++level) {
        const float texel = OceanFoamTexel(level);
        const float2 uv = (q - P.foamLevel[level].xy) / (texel * OCEAN_FOAM_SIZE);
        const float w = 1.0f - smoothstep(0.35f, 0.48f, max(abs(uv.x - 0.5f), abs(uv.y - 0.5f)));
        if (w <= 0.0f)
            continue;
        finest = min(finest, level);
        const float mip = clamp(log2(widthM / texel), 0.0f, OCEAN_FOAM_MIPS - 1.0f);
        float c = foam.SampleLevel(g_sampler, float3(uv, level), mip);
        const float magnified = saturate(2.0f - 2.0f * widthM / texel);
        if (magnified > 0.0f)
            c = lerp(c, OceanFoamSmooth(foam, uv, level), magnified);
        const float cloudMip = clamp(log2(max(widthM, OCEAN_FOAM_CLOUD) / texel), 0.0f, OCEAN_FOAM_MIPS - 1.0f);
        cover += rest * w * c;
        cloud += rest * w * foam.SampleLevel(g_sampler, float3(uv, level), cloudMip);
        rest *= 1.0f - w;
    }
    if (rest > 1e-3f) {
        const float mean = foam.SampleLevel(g_sampler, float3(0.5f, 0.5f, OCEAN_FOAM_LEVELS - 1u), OCEAN_FOAM_MIPS - 1.0f);
        cover += rest * mean;
        cloud += rest * mean;
    }
    // The simulation runs on the unit-gain sea. A rougher patch is one a stronger wind raised, and
    // whitecap cover goes with the wind as U^3.41 where wave height goes as U^2, so its cover
    // follows the patch's gain to the power 1.7: half again as much foam for a third more wave.
    // No cap is one flat, blinding white: the cover is held to what the densest raft lays down, its
    // grain and all, near and far alike.
    const float rough = pow(max(gain, 0.0f), 1.7f);
    f.coverage = min(saturate(cover) * rough, OCEAN_FOAM_DENSEST);
    // The cloud is dense only under a cap still breaking and clears as its bubbles rise, within a
    // second or so of the crest running on - a tint round the fresh white, not a glow of its own.
    f.bubbles = smoothstep(0.35f, 0.9f, min(saturate(cloud) * rough, 1.0f));

    // Up close, the cover is drawn as a raft that lays down exactly that much foam (OceanFoamRaft):
    // the web of cells and strands it dissolves along (OceanFoamWeb) says how much of each place
    // is foam, the bubbles and clumps it is made of (OceanFoamDetail) are the alpha map that says
    // where, and the whole fades into the plain cover where the footprint outgrows the web. The
    // position is rebuilt from the finest level's absolute texel index, as the simulation's is.
    float2 domes = 0.0f;
    if (f.coverage > 0.0f) {
        const float2 world = q - P.foamLevel[0].xy + float2(P.foamCell[0].xy) * OCEAN_FOAM_TEXEL0;
        const float2 w = OceanFoamFrame(world, P.foamWind);
        f.coverage = min(f.coverage * OceanFoamPatches(w, widthM), OCEAN_FOAM_DENSEST);
        float resolved, feather;
        const float u = OceanFoamWeb(w, widthM, resolved, feather);
        if (resolved > 0.0f) {
            float detailed;
            const float detail = OceanFoamDetail(q, w, P.foamWind, widthM, P, domes, detailed);
            f.coverage = lerp(f.coverage,
                              OceanFoamRaft(u, OceanFoamAmount(f.coverage, feather), feather, detail, detailed), resolved);
            domes *= resolved;
        }
    }

    // A raft is a lumpy pile of bubbles rather than a painted film: where the finest level resolves
    // it, its relief follows how much foam is piled up, and fades out as the footprint outgrows it.
    if (finest < OCEAN_FOAM_LEVELS && f.coverage > 0.0f) {
        const float texel = OceanFoamTexel(finest);
        const float resolved = saturate(2.0f - widthM / texel);
        if (resolved > 0.0f) {
            const float2 uv = (q - P.foamLevel[finest].xy) / (texel * OCEAN_FOAM_SIZE);
            const float du = 1.0f / OCEAN_FOAM_SIZE;
            const float c0 = foam.SampleLevel(g_sampler, float3(uv, finest), 0.0f);
            const float cx = foam.SampleLevel(g_sampler, float3(uv + float2(du, 0.0f), finest), 0.0f);
            const float cz = foam.SampleLevel(g_sampler, float3(uv + float2(0.0f, du), finest), 0.0f);
            f.slope = OCEAN_FOAM_RELIEF * resolved * float2(cx - c0, cz - c0) / texel;
        }
    }
    // Its bubbles and clumps hump it further, down to the smallest the footprint resolves.
    f.slope += domes;
    return f;
}

// `widthM` is the ray's footprint at the hit. `tileUV` and `tileSize` locate the hit inside the
// tile it belongs to; only the diagnostic views read them.
OceanSurface OceanEvalSurface(float2 q, float widthM, float2 tileUV, float tileSize) {
    const OceanParamsGPU P = OceanParams();
    // By default (filterScale 0) the normal is the full-resolution wave field at every distance
    // and the surface a mirror: the distant sheen is built by the denoiser from the sharp facets
    // themselves. Filtering to the footprint instead takes the mean normal over it and turns
    // everything averaged away into roughness. Foam always filters to the true footprint - it is
    // a coverage, which averages without changing what it looks like.
    const float filterW = widthM * P.filterScale;
    const OceanSample sample = OceanSampleSurface(q, filterW);
    const float3 A = sample.stretch;
    OceanSurface s;
    s.normal = OceanNormalFromSample(sample, (q + P.curveOrigin) * P.invRadius);
    const float alpha = P.filterScale > 0.0f ? min(sqrt(OceanResidualSlope(filterW, P)) * sample.gain, 0.35f) : 0.0f;
    s.roughness = sqrt(alpha); // perceptual: the BRDF squares it
    s.albedo = 1.0f.xxx;

    // Whitecaps pick their step of the foam ramp and of the bubble cloud's, each dithered afresh
    // every frame so the steps do not show. A step that is mostly foam takes the raft's relief; a
    // mostly clear one keeps the water's normal.
    const OceanFoam foam = OceanFoamAt(q, widthM, sample.gain, P);
    const float coverage = foam.coverage;
    const uint dither = OceanHash(asuint(q.x) * 0x9E3779B9u ^ asuint(q.y) ^ P.frameIndex * 0x85EBCA6Bu);
    const uint foamStep = min((uint)(coverage * (OCEAN_FOAM_STEPS - 1) + (dither & 0xFFFFu) / 65536.0f),
                              (uint)OCEAN_FOAM_STEPS - 1u);
    const uint bubbleStep = min((uint)(foam.bubbles * (OCEAN_BUBBLE_STEPS - 1) + (dither >> 16) / 65536.0f),
                                (uint)OCEAN_BUBBLE_STEPS - 1u);
    s.materialOffset = foamStep + OCEAN_FOAM_STEPS * bubbleStep;
    s.foam = coverage;
    s.bubbles = foam.bubbles;
    if (foamStep * 2u >= (uint)OCEAN_FOAM_STEPS)
        s.normal = normalize(s.normal - float3(foam.slope.x, 0.0f, foam.slope.y));
    // A bubble raft is rough at every scale, which is what keeps it white from any angle. Left with
    // the water's mirror coat, distant foam - seen at a grazing angle, where that coat reflects
    // most of the sky - read as water and thinned out towards the horizon. The cloud under the
    // surface leaves the coat as it is.
    s.roughness = max(s.roughness, OCEAN_FOAM_ROUGHNESS * float(foamStep) / (OCEAN_FOAM_STEPS - 1));

    // Opt-in diagnostics shade an opaque material; they never feed beauty shading.
    if (P.debugMode != OCEAN_DEBUG_BEAUTY) {
        const float det = A.x * A.y - A.z * A.z;
        const float minStretch = 0.5f * (A.x + A.y - length(float2(A.x - A.y, 2 * A.z)));
        if (P.debugMode == OCEAN_DEBUG_NORMALS) s.albedo = s.normal * 0.5f + 0.5f;
        if (P.debugMode == OCEAN_DEBUG_FOAM)
            s.albedo = lerp(lerp(s.normal * 0.25f + 0.25f, float3(0.1f, 0.6f, 0.7f), foam.bubbles), 1.0f.xxx, coverage);
        if (P.debugMode == OCEAN_DEBUG_ROUGHNESS) s.albedo = float3(s.roughness, s.roughness, s.roughness);
        if (P.debugMode == OCEAN_DEBUG_COMPRESSION) s.albedo = saturate(float3(1.0f - det, minStretch, det));
        if (P.debugMode == OCEAN_DEBUG_GEOMETRY) {
            // One hue per tile size, quad edges dark, tile edges white: the density the
            // acceleration structures actually hold, and where the stitching falls.
            const float2 cell = frac(tileUV * OCEAN_TILE_GRID);
            const float2 quadEdge = min(cell, 1.0f - cell);
            const float2 tileEdge = min(tileUV, 1.0f - tileUV) * OCEAN_TILE_GRID;
            s.albedo = OceanHue(log2(max(tileSize, 1.0f)) * 0.13f) * 0.8f + 0.1f;
            if (min(quadEdge.x, quadEdge.y) < 0.06f) s.albedo *= 0.35f;
            if (min(tileEdge.x, tileEdge.y) < 0.12f) s.albedo = 1.0f.xxx;
        }
        if (P.debugMode == OCEAN_DEBUG_MIP) s.albedo = saturate(float3(OceanCascadeMip(filterW, P.cascadeLength.x),
            OceanCascadeMip(filterW, P.cascadeLength.y), OceanCascadeMip(filterW, P.cascadeLength.z)) / (OCEAN_MIP_LEVELS - 1));
        s.materialOffset = OCEAN_MATERIAL_DEBUG;
        s.roughness = 1.0f;
    }
    return s;
}
