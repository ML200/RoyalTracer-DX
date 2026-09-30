#pragma once
#include "OceanLayout.h"
#include "OceanMath.hlsli"

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
    // FFT node i is at i*L/N; texel i is centred at (i+0.5)/N.
    return (q + float2(P.originWrapX[c], P.originWrapZ[c])) / P.cascadeLength[c] + 0.5f / OCEAN_FFT_SIZE;
}
float OceanCascadeMip(float widthM, float L) {
    return clamp(log2(max(widthM * OCEAN_FFT_SIZE / L, 1.0f)), 0.0f, OCEAN_MIP_LEVELS - 1.0f);
}

// Must match ocean::LodRule; position-only, so shared vertices never crack.
float OceanGeometryWidth(float2 q, OceanParamsGPU P) {
    const float3 v = float3(q.x, P.surfaceY, q.y) - P.lodCamera;
    const float d = length(v);
    const float z = dot(v, P.lodForward);
    const float outside = z > 1e-3f ? max(abs(dot(v, P.lodRight)) / (z * P.lodTanH),
                                          abs(dot(v, P.lodUp)) / (z * P.lodTanV)) : 1e6f;
    // Blends end where selection goes coarse (outside = 1).
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

// Edge vertices lerp onto a coarser neighbour's edge to close cracks.
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
// Kilometre-scale sea-state gain (x) and its world gradient (yz).
float3 OceanTurbulence(float2 q, OceanParamsGPU P) {
    if (P.turbulenceStrength <= 0.0f)
        return float3(1.0f, 0.0f, 0.0f);
    Texture2D<float4> field = ResourceDescriptorHeap[OCEAN_SRV_TURBULENCE];
    const float2 absoluteQ = q + P.curveOrigin;
    return field.SampleLevel(g_sampler, absoluteQ / P.turbulencePeriod, 0.0f).xyz;
}

struct OceanSample {
    float3 displacement;
    float2 gradient;
    float3 stretch; // (1 + dDx/dx, 1 + dDz/dz, dDx/dz)
    float gain;     // kilometre-scale sea-state gain
};

// Raw derivatives are summed over cascades before inversion.
OceanSample OceanSampleSurfaceFrom(float2 q, float widthM, uint dispSlot, bool derivatives) {
    const OceanParamsGPU P = OceanParams();
    Texture2DArray<float4> disp = ResourceDescriptorHeap[dispSlot];
    Texture2DArray<float4> deriv = ResourceDescriptorHeap[OCEAN_SRV_DERIV];
    // Product rule: the gain varies across the surface.
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
        // Each band sharpened on its own elevation, after the gain.
        const float skew = P.crestSkew[c];
        const float variance = P.cascadeVariance[c] * gain * gain;
        s.displacement.xz += h.xz * gain;
        s.displacement.y += OceanSkewHeight(h.y * gain, skew, variance);
        if (derivatives) {
            const float4 d = deriv.SampleLevel(g_sampler, uv, mip);
            const float slope = OceanSkewSlope(h.y * gain, skew);
            s.gradient += slope * gain * d.xy + (slope * h.y - 2.0f * skew * gain * P.cascadeVariance[c]) * dGain;
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
float3 OceanNormalFromSample(OceanSample s, float2 curveGradient) {
    const float2 slope = OceanWarpedSlope(s.gradient - curveGradient, s.stretch);
    return normalize(float3(-slope.x, 1, -slope.y));
}

struct OceanSurface {
    float3 normal;
    float3 albedo;
    float roughness;
    uint materialOffset;
    // Undithered, for the albedo guide (OceanGuideAlbedo).
    float foam;
    float bubbles;
};

float3 OceanHue(float t) {
    return saturate(abs(frac(t + float3(0.0f, 2.0f / 3.0f, 1.0f / 3.0f)) * 6.0f - 3.0f) - 1.0f);
}

// Slope variance averaged away at this footprint width.
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

// Gradient noise, about -0.7..0.7.
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

// x along the wind, z across it.
float2 OceanFoamFrame(float2 world, float2 wind) {
    return float2(dot(world, wind), dot(world, float2(-wind.y, wind.x)));
}

// Whitecaps over a footprint, after Crest (wave-harmonic/crest, MIT).
struct OceanFoam {
    float coverage; // 0 clear water .. 1 solid foam
    float bubbles;  // bubble-cloud density, 0..1
    float2 slope;   // raft relief slope, world xz
};
static const float OCEAN_FOAM_RELIEF = 0.03f;   // m, bare water to solid foam
static const float OCEAN_FOAM_ROUGHNESS = 1.0f; // perceptual, solid foam
static const float OCEAN_FOAM_CLOUD = 0.4f;     // m, bubble-cloud spread round a cap

float OceanFoamTexel(uint level) {
    return OCEAN_FOAM_TEXEL0 * exp2((float)level);
}

// Foam raft cover from web u (OceanFoamWeb) and alpha map `detail` (OceanFoamDetail).
static const float OCEAN_FOAM_FILM = 0.15f;
static const float OCEAN_FOAM_GRAIN = 0.3f;
static const float OCEAN_FOAM_DIM = 0.35f;
static const float OCEAN_FOAM_OPEN = 0.9f; // max share of the plane covered
static const float OCEAN_FOAM_EDGE = 0.3f; // alpha-map cut softness, share of its range
static const float OCEAN_FOAM_DENSEST = OCEAN_FOAM_OPEN * (1.0f - 0.5f * OCEAN_FOAM_GRAIN) +
                                        OCEAN_FOAM_FILM * (1.0f - OCEAN_FOAM_OPEN + OCEAN_FOAM_OPEN * OCEAN_FOAM_GRAIN / 3.0f);
// Cut keeping exactly `share` of a uniform alpha map.
float OceanFoamCut(float detail, float share) {
    const float edge = max(2.0f * OCEAN_FOAM_EDGE * min(share, 1.0f - share), 1e-4f);
    return saturate((detail - 1.0f + share) / edge + 0.5f);
}
float OceanFoamRaft(float u, float amount, float feather, float detail, float detailed) {
    const float a = saturate(amount);
    const float share = saturate((u - 1.0f + a * (1.0f + feather)) / feather) * OCEAN_FOAM_OPEN;
    const float alpha = lerp(share, OceanFoamCut(detail, share), detailed);
    const float web = alpha * pow(a, OCEAN_FOAM_DIM) * (1.0f - OCEAN_FOAM_GRAIN * (1.0f - u));
    return web + (1.0f - web) * OCEAN_FOAM_FILM * a * a * 2.0f * u;
}
// Mean raft cover over uniform u and alpha map, closed form.
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
// Inverts OceanFoamMean (monotonic) by bisection.
float OceanFoamAmount(float cover, float feather) {
    float lo = 0.0f, hi = 1.0f;
    [unroll] for (uint i = 0u; i < 12u; ++i) {
        const float mid = 0.5f * (lo + hi);
        if (OceanFoamMean(mid, feather) < cover) lo = mid;
        else hi = mid;
    }
    return 0.5f * (lo + hi);
}

// Additively weighted Voronoi F2 - F1: 0 on a wall, ~0.7 at a site.
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
// Measured quantiles: web walls (1/16 steps), alpha map (1/64 steps).
static const float kFoamWebCoarse[17] = {-1.7611f, -0.7273f, -0.5949f, -0.5062f, -0.4366f, -0.3785f, -0.3283f, -0.2835f, -0.2429f,
                                         -0.2055f, -0.1709f, -0.1386f, -0.1080f, -0.0791f, -0.0516f, -0.0247f, 0.0635f};
static const float kFoamWebFine[17] = {-1.9822f, -0.8286f, -0.6938f, -0.6035f, -0.5334f, -0.4748f, -0.4240f, -0.3787f, -0.3377f,
                                       -0.2998f, -0.2641f, -0.2299f, -0.1962f, -0.1621f, -0.1256f, -0.0832f, 0.0595f};
static const float kFoamDetailZ[65] = {
    -3.6436f, -2.0306f, -1.7933f, -1.6331f, -1.5086f, -1.4035f, -1.3126f, -1.2311f, -1.1573f, -1.0884f, -1.0240f, -0.9641f, -0.9069f,
    -0.8524f, -0.8001f, -0.7495f, -0.7008f, -0.6532f, -0.6068f, -0.5615f, -0.5169f, -0.4732f, -0.4302f, -0.3879f, -0.3461f, -0.3051f,
    -0.2644f, -0.2238f, -0.1836f, -0.1433f, -0.1028f, -0.0628f, -0.0225f, 0.0182f, 0.0586f, 0.0991f, 0.1398f, 0.1811f, 0.2226f,
    0.2648f, 0.3072f, 0.3503f, 0.3940f, 0.4384f, 0.4836f, 0.5300f, 0.5775f, 0.6266f, 0.6769f, 0.7294f, 0.7834f, 0.8399f, 0.8994f,
    0.9617f, 1.0282f, 1.0983f, 1.1744f, 1.2575f, 1.3490f, 1.4523f, 1.5731f, 1.7203f, 1.9114f, 2.2040f, 4.5966f};
float OceanFoamRank(float v, bool fine) {
    uint i = 0u;
    [unroll] for (uint step = 8u; step > 0u; step >>= 1u) {
        if (i + step <= 16u && (fine ? kFoamWebFine[i + step] : kFoamWebCoarse[i + step]) <= v)
            i += step;
    }
    i = min(i, 15u);
    const float q0 = fine ? kFoamWebFine[i] : kFoamWebCoarse[i];
    const float q1 = fine ? kFoamWebFine[i + 1u] : kFoamWebCoarse[i + 1u];
    return saturate((float(i) + saturate((v - q0) / (q1 - q0))) / 16.0f);
}
float OceanFoamDetailRank(float v) {
    uint i = 0u;
    [unroll] for (uint step = 32u; step > 0u; step >>= 1u) {
        if (i + step <= 64u && kFoamDetailZ[i + step] <= v)
            i += step;
    }
    i = min(i, 63u);
    return saturate((float(i) + saturate((v - kFoamDetailZ[i]) / (kFoamDetailZ[i + 1u] - kFoamDetailZ[i]))) / 64.0f);
}

// Random dome per cell: 1 at its middle, 0 at the rim, toward -1 between.
static const float OCEAN_DOME_MEAN = 0.1788f; // measured mean over the plane
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
// Foam alpha map, ~uniform on [0, 1], from the domes the footprint resolves.
static const uint kBubblesPerTexel[3] = {1u, 2u, 5u};
static const float kBubbleWeight[3] = {0.9f, 0.9f, 0.8f};
static const float kClumpCell[5] = {0.12f, 0.3f, 0.8f, 2.4f, 7.2f}; // m
static const float kClumpWeight[5] = {0.65f, 0.55f, 0.5f, 0.45f, 0.4f};
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
    [unroll] for (uint c = 0u; c < 5u; ++c) {
        const float keep = saturate(2.0f - 2.0f * widthM / kClumpCell[c]);
        if (keep > 0.0f) {
            const float2 p = float2(w.x / OCEAN_CLUMP_STRETCH, w.y) / kClumpCell[c];
            const float2 i = floor(p);
            float2 dome;
            const float hollow = OceanFoamHollow(int2(i), p - i, 81u + c, dome);
            sum += keep * kClumpWeight[c] * (hollow - OCEAN_DOME_MEAN);
            drawn += keep * keep * kClumpWeight[c] * kClumpWeight[c];
            if (c < 2u) {
                const float2 g = keep * OCEAN_CLUMP_DOME * float2(dome.x / OCEAN_CLUMP_STRETCH, dome.y);
                slope += g.x * wind + g.y * float2(-wind.y, wind.x);
            }
        }
    }
    const float spread = sqrt(drawn);
    // Fades out with the coarsest patches.
    detailed = saturate(spread / (0.8f * kClumpWeight[4]));
    return spread > 1e-4f ? OceanFoamDetailRank(-sum / (OCEAN_DOME_STD * spread)) : 0.5f;
}
// Foam web: u ~ uniform on [0, 1], highest where a raft holds longest.
static const float OCEAN_FOAM_CELL = 0.6f;      // m, web cell
static const float OCEAN_FOAM_PERFORATION = 0.15f; // m, hole in a strand
static const float OCEAN_FOAM_STRETCH = 2.0f;   // downwind elongation
static const float OCEAN_FOAM_FEATHER = 0.55f;  // min share of u the raft rises over
float OceanFoamWeb(float2 w, float widthM, out float resolved, out float feather) {
    float2 p = float2(w.x / OCEAN_FOAM_STRETCH, w.y);
    p += (0.5f / 1.5f) * float2(OceanFoamGrad(p * 0.4f, 11u) + 0.5f * OceanFoamGrad(p * 0.8f, 12u),
                                OceanFoamGrad(p * 0.4f, 17u) + 0.5f * OceanFoamGrad(p * 0.8f, 18u));
    float2 q = p / OCEAN_FOAM_CELL;
    q += 0.45f * float2(OceanFoamGrad(q * 1.5f, 21u), OceanFoamGrad(q * 1.5f, 27u));
    const float walls = OceanFoamWalls(q, 41u) + 0.1f * OceanFoamGrad(q * 4.0f, 51u);
    float u = OceanFoamRank(-walls, false);
    const float perforated = saturate(2.0f - 4.0f * widthM / OCEAN_FOAM_PERFORATION);
    if (perforated > 0.0f) {
        float2 r = p / OCEAN_FOAM_PERFORATION;
        r += 0.45f * float2(OceanFoamGrad(r * 1.5f, 31u), OceanFoamGrad(r * 1.5f, 37u));
        const float holes = OceanFoamWalls(r, 43u) + 0.1f * OceanFoamGrad(r * 4.0f, 61u);
        u = lerp(u, OceanFoamRank(-(walls + 0.3f * holes), true), perforated);
    }
    // A footprint spans ~2 widthM / cell of u.
    feather = clamp(2.0f * widthM / OCEAN_FOAM_CELL, OCEAN_FOAM_FEATHER, 1.0f);
    resolved = saturate(2.0f - widthM / (0.15f * OCEAN_FOAM_CELL));
    return u;
}

// Cubic B-spline from four bilinear taps (Sigg & Hadwiger 2005).
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

// gain: OceanSample.gain at q.
OceanFoam OceanFoamAt(float2 q, float widthM, float gain, OceanParamsGPU P) {
    OceanFoam f;
    f.coverage = 0.0f;
    f.bubbles = 0.0f;
    f.slope = 0.0f;
    if (P.foamStrength <= 0.0f)
        return f;
    // Finest level holding q, blended into coarser; mean past the last.
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
    // Cover ~ U^3.41, height ~ U^2: gain^1.7 (Monahan & O'Muircheartaigh 1980).
    const float rough = pow(max(gain, 0.0f), 1.7f);
    f.coverage = min(saturate(cover) * rough, OCEAN_FOAM_DENSEST);
    // Dense only under a still-breaking cap.
    f.bubbles = smoothstep(0.35f, 0.9f, min(saturate(cloud) * rough, 1.0f));

    // Near: raft on the web; far: cover cut from the alpha map.
    float2 domes = 0.0f;
    if (f.coverage > 0.0f) {
        const float2 world = q - P.foamLevel[0].xy + float2(P.foamCell[0].xy) * OCEAN_FOAM_TEXEL0; // as the simulation
        const float2 w = OceanFoamFrame(world, P.foamWind);
        float resolved, feather, detailed;
        const float u = OceanFoamWeb(w, widthM, resolved, feather);
        const float detail = OceanFoamDetail(q, w, P.foamWind, widthM, P, domes, detailed);
        const float amount = OceanFoamAmount(f.coverage, feather);
        const float piece = max(pow(amount, OCEAN_FOAM_DIM) * (1.0f - 0.5f * OCEAN_FOAM_GRAIN), f.coverage);
        const float share = f.coverage / max(piece, 1e-4f);
        const float far = lerp(share, OceanFoamCut(detail, share), detailed) * piece;
        f.coverage = resolved > 0.0f ? lerp(far, OceanFoamRaft(u, amount, feather, detail, detailed), resolved) : far;
        domes *= resolved;
    }

    // Raft relief from the cover gradient, where resolved.
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
    // Plus the bubble and clump domes.
    f.slope += domes;
    return f;
}

// widthM: ray footprint; tileUV, tileSize: debug views only.
OceanSurface OceanEvalSurface(float2 q, float widthM, float2 tileUV, float tileSize) {
    const OceanParamsGPU P = OceanParams();
    // filterScale 0: full-res normal, mirror. Foam always uses the true footprint.
    const float filterW = widthM * P.filterScale;
    const OceanSample sample = OceanSampleSurface(q, filterW);
    const float3 A = sample.stretch;
    OceanSurface s;
    s.normal = OceanNormalFromSample(sample, (q + P.curveOrigin) * P.invRadius);
    const float alpha = P.filterScale > 0.0f ? min(sqrt(OceanResidualSlope(filterW, P)) * sample.gain, 0.35f) : 0.0f;
    s.roughness = sqrt(alpha); // perceptual: the BRDF squares it
    s.albedo = 1.0f.xxx;

    // Foam and bubble steps, dithered per frame.
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
    // Rough foam stays white at grazing angles.
    s.roughness = max(s.roughness, OCEAN_FOAM_ROUGHNESS * float(foamStep) / (OCEAN_FOAM_STEPS - 1));

    // Debug views: opaque material, never beauty.
    if (P.debugMode != OCEAN_DEBUG_BEAUTY) {
        const float det = A.x * A.y - A.z * A.z;
        const float minStretch = 0.5f * (A.x + A.y - length(float2(A.x - A.y, 2 * A.z)));
        if (P.debugMode == OCEAN_DEBUG_NORMALS) s.albedo = s.normal * 0.5f + 0.5f;
        if (P.debugMode == OCEAN_DEBUG_FOAM)
            s.albedo = lerp(lerp(s.normal * 0.25f + 0.25f, float3(0.1f, 0.6f, 0.7f), foam.bubbles), 1.0f.xxx, coverage);
        if (P.debugMode == OCEAN_DEBUG_ROUGHNESS) s.albedo = float3(s.roughness, s.roughness, s.roughness);
        if (P.debugMode == OCEAN_DEBUG_COMPRESSION) s.albedo = saturate(float3(1.0f - det, minStretch, det));
        if (P.debugMode == OCEAN_DEBUG_GEOMETRY) {
            // Hue per tile size; quad edges dark, tile edges white.
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
