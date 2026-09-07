// Runs the production hash, interpolation, accumulation, lifecycle and suffix
// propagation on D3D12. Only camera/material bindings are replaced by fixtures.
cbuffer TestConstants : register(b0)
{
    uint sharc_reset, sharc_frame, sharc_minSamples, sharc_historyFrames;
    uint sharc_maxAge, testMode;
    float sharc_cellSize, sharc_lodScale;
    float sharc_queryFootprint;
    float3 sceneOriginWorld;
    float3 testCamera;
    uint testIndex;
    uint guide_params; // SharcLayout.h GUIDE_PARAM_* packing, as in production
    uint3 testPad;
};
float3 InitOrigin() { return testCamera; }
float Luma(float3 c) { return dot(c, float3(0.2126f, 0.7152f, 0.0722f)); }
#include "Compression_v8.hlsli"
#include "Random_v8.hlsli"
#define SHARC_TEST 1
#define SHARC_UPDATE_PASS 1
#include "SharcPath_v8.hlsli"
#include "SharcGuide_v8.hlsli"
#include "SharcDebug_v8.hlsli"
// ReSTIR lite: the pure resampling math (packing, links, targets, MIS).
#ifndef PI
#define PI 3.1415926535
#endif
#define RAY_TMAX_PLANET 1e9f
#define ucw_clampMax 0.0f
#include "RestirLite_v8.hlsli"
#define main prepare
#include "Pass_sharc_prepare_v8.hlsl"
#undef main
#define main resolve
#include "Pass_sharc_resolve_v8.hlsl"
#undef main
RWByteAddressBuffer results : register(u0);
#include "MaterialGpuTests.hlsli"

SharcSurface Surface(uint mode)
{
    SharcSurface s = (SharcSurface)0;
    s.position = float3(0.04f, 0.0f, 0.04f);
    s.geometricNormal = s.normal = float3(0, 1, 0);
    s.demodulator = 1.0f;
    s.instance = s.material = 1u;
    s.roughness = 0.8f;
    if (mode == 1u) // underside of the same 1 mm sheet
    {
        s.position.y = -0.001f;
        s.geometricNormal = s.normal = float3(0, -1, 0);
        s.variant = 1u;
    }
    if (mode == 2u) s.position.y = 0.02f; // parallel layer within the SAME cell
    if (mode == 16u || mode == 17u) s.demodulator = float3(0.25f, 0.5f, 0.75f);
    if (mode == 17u) s.position.x += 0.5f;
    if (mode == 8u) s.geometricNormal = s.normal = normalize(float3(1, 1, 0));
    return s;
}

[numthreads(64, 1, 1)]
void fill(uint3 tid : SV_DispatchThreadID)
{
    // Maintenance fixture: fully occupied table, with a controllable fraction
    // receiving observations. This separates table occupancy from update rate.
    if (testMode == 30u || testMode == 31u)
    {
        if (tid.x >= SHARC_CAPACITY) return;
        uint e = SharcEntryAddress(tid.x);
        if (testMode == 30u)
        {
            SharcInitializeEntry(e, int3(0, 0, 0), 0u, Surface(0u), 0.0f);
            g_sharc.Store(SharcStateAddress(tid.x), 1u);
        }
        else if (tid.x % max(testIndex, 1u) == 0u)
            SharcAccumulate(e, float3(2, 3, 4), 1.0f);
        return;
    }
    uint seed = Hash32(tid.x ^ Hash32(sharc_frame) ^ Hash32(testMode));
    SharcSurface s = Surface(testMode);
    if (testMode == 16u)
    {
        SharcTrainingState state;
        SharcTrainingInit(state);
        SharcTrainingVertex(state, s, 1.0f, Hash32(seed));
        SharcTrainingScatter(state, float3(0.5f, 0.75f, 0.25f));
        SharcTrainingRadiance(state, float3(2, 3, 4));
        s.position.x += 0.5f;
        SharcTrainingVertex(state, s, 1.0f, Hash32(seed ^ 123u));
        SharcTrainingScatter(state, float3(0.2f, 0.3f, 0.4f));
        SharcTrainingRadiance(state, float3(5, 6, 7));
        SharcTrainingCommit(state);
        return;
    }
    if (testMode == 14u)
    {
        s.geometricNormal = s.normal = normalize(float3(1, 1, 0));
        uint c = tid.x & 7u, axis = (tid.x >> 3u) & 1u, level = (tid.x >> 4u) & 1u;
        int3 base; float3 f;
        SharcGrid(s.position, level, base, f);
        int3 corner = int3(c & 1u, (c >> 1u) & 1u, c >> 2u);
        float3 relative = (f - float3(corner)) * SharcCellSize(level);
        float w;
        uint a = SharcFindOrInsert(base + corner, level, axis, s, relative, w);
        float value = ((c & 1u) != 0u ? 8.0f : 1.0f) + axis * 4.0f + level * 12.0f;
        SharcAccumulate(a, float3(value, value * 2.0f, 1.0f), 1.0f);
        return;
    }
    if (testMode == 11u && tid.x != 0u) return; // one sparse observation
    if (testMode == 13u) s.position.x = asfloat(0x7fc00000u);
    float3 value = testMode == 0u ? 100.0f : 0.001f;
    if (testMode == 3u) value = 0.0f;
    if (testMode == 11u) value = 2.0f;
    if (testMode == 4u)
    {
        s.position.xz = float2(RandomFloatSingle(seed), RandomFloatSingle(seed)) - 0.5f;
        value = float3(2, 3, 4);
    }
    if (testMode == 8u)
    {
        float delta = (RandomFloatSingle(seed) - 0.5f) * 0.04f;
        s.geometricNormal = s.normal = normalize(float3(1 + delta, 1 - delta, 0));
        value = float3(2, 3, 4);
    }
    if (testMode == 5u)
    {
        SharcTrainingState state;
        SharcTrainingInit(state);
        SharcTrainingVertex(state, s, 1.0f, Hash32(seed ^ 0x53504c54u));
        bool survives = true;
        // Analytic reference: 10 * 0.8^20. Survival compensation must reach
        // the suffix even after 20 bounces; all failed paths count as zeros.
        [loop] for (uint b = 0u; b < 20u; ++b)
        {
            if (RandomFloatSingle(seed) >= 0.9f) { survives = false; break; }
            SharcTrainingScatter(state, 0.8f / 0.9f);
        }
        if (survives) SharcTrainingRadiance(state, 10.0f);
        SharcTrainingCommit(state);
    }
    else SharcSplat(s, value, seed);
}

[numthreads(64, 1, 1)]
void query(uint3 tid : SV_DispatchThreadID)
{
    if (testMode == 31u)
    {
        uint e = SharcEntryAddress(tid.x + testPad.x);
        results.Store4(tid.x * 32u, uint4(g_sharc.Load3(e + SHARC_MEAN),
            asuint((float)g_sharc.Load(e + SHARC_FRAMES))));
        return;
    }
    if (tid.x >= 64u) return;
    if (testMode == 31u)
    {
        SharcDebugSample cell = SharcDebugLookup(Surface(0u), 0u);
        uint a = cell.address;
        if (a == SHARC_INVALID) { results.Store4(tid.x * 32u, asuint(float4(-1, -1, -1, -1))); return; }
        results.Store4(tid.x * 32u, asuint(float4(asfloat(g_sharc.Load(a + SHARC_HISTORY_W)),
            asfloat(g_sharc.Load(a + SHARC_HISTORY_W2)), (float)g_sharc.Load(a + SHARC_FRAMES),
            SharcConfidence(SharcLoadHistory(a)))));
        results.Store4(tid.x * 32u + 16u, asuint(float4(asfloat(g_sharc.Load3(a + SHARC_MEAN)),
            (float)(sharc_frame - g_sharc.Load(a + SHARC_LAST_UPDATE)))));
        return;
    }
    if (testMode == 30u)
    {
        SharcSurface s = Surface(8u);
        float lod = SharcLevel(s.position);
        float3 a, b; float wa, wb;
        SharcQueryLevel(s, (uint)lod, a, wa);
        SharcQueryLevel(s, (uint)lod + 1u, b, wb);
        float4 reference = lerp(float4(a, wa), float4(b, wb), frac(lod)) * (31.0f / 32.0f);
        float4 estimate = 0.0f;
        uint seed = Hash32(tid.x);
        [loop] for (uint sample = 0u; sample < 2048u; ++sample)
        {
            float3 radiance;
            if (SharcQuery(s, 100.0f, seed, radiance)) estimate += float4(radiance, 1.0f);
        }
        results.Store4(tid.x * 32u, asuint(reference));
        results.Store4(tid.x * 32u + 16u, asuint(estimate / 2048.0f));
        return;
    }
    if (testMode >= 20u && testMode <= 24u)
    {
        // Mode 24 used to flip the camera direction; records no longer carry one.
        SharcSurface debugSurface = Surface(testMode == 24u ? 0u : testMode - 20u);
        results.Store4(tid.x * 32u, asuint(SharcDebugColor(debugSurface, testIndex, SHARC_DEBUG_LIGHTING)));
        results.Store4(tid.x * 32u + 16u, asuint(SharcDebugColor(debugSurface, testIndex, SHARC_DEBUG_CELLS)));
        return;
    }
    SharcSurface s = Surface(testMode);
    if (testMode == 4u) s.position.x = -0.2f + 0.4f * (float)tid.x / 63.0f;
    if (testMode == 8u)
    {
        float delta = 0.04f * ((float)tid.x / 63.0f - 0.5f);
        s.geometricNormal = s.normal = normalize(float3(1 + delta, 1 - delta, 0));
    }
    if (testMode == 6u)
    {
        // Two different km origins represent the same centimetre-scale point
        // eight million metres from the original scene origin.
        float3 p = float3(8000000, -8000000, 8000000) - sceneOriginWorld +
            float3(0.03125f, 0.0625f, -0.03125f);
        int3 node; float3 f;
        SharcGrid(p, testIndex, node, f);
        results.Store3(tid.x * 32u, asuint(node));
        results.Store3(tid.x * 32u + 16u, asuint(f));
        return;
    }
    float3 sum; float support;
    SharcQueryLevel(s, testIndex, sum, support);
    results.Store4(tid.x * 32u, asuint(float4(sum / max(support, 1e-20f), support)));
    uint seed = Hash32(tid.x);
    float3 radiance;
    bool hit = SharcQuery(s, testMode == 7u ? 0.0001f : 100.0f, seed, radiance);
    results.Store4(tid.x * 32u + 16u, asuint(float4(radiance, hit ? 1.0f : 0.0f)));
}

[numthreads(256, 1, 1)]
void eraseTop(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= SHARC_CAPACITY) return;
    uint stateAddress = SharcStateAddress(tid.x);
    uint a = SharcEntryAddress(tid.x);
    if (testMode == 10u) { g_sharc.Store(stateAddress, 1u); return; } // force a saturated table
    if (g_sharc.Load(stateAddress) == 0u) return;
    if (testMode == 15u)
    {
        if ((g_sharc.Load(a + SHARC_NODE) & 1u) != 0u)
        {
            g_sharc.Store(stateAddress, 0u);
            [unroll] for (uint b = 0u; b < SHARC_ENTRY_BYTES; b += 16u) g_sharc.Store4(a + b, 0u);
        }
        return;
    }
    if (testMode == 12u) // simulate corrupt persistent history, not just a bad input
    {
        g_sharc.Store(a + SHARC_MEAN, 0x7fc00000u);
        g_sharc.Store(a + SHARC_HISTORY_W, 0x7fc00000u);
        g_sharc.Store(a + SHARC_CONFIDENCE, asuint(1.0f));
        return;
    }
    if (testMode == 9u) // force equal fingerprints with DIFFERENT exact coordinates
    {
        g_sharc.Store(a + SHARC_NODE, g_sharc.Load(a + SHARC_NODE) ^ 0x40000000u);
        return;
    }
    float3 normal = UnpackNormal(g_sharc.Load(a + SHARC_GEOMETRIC_NORMAL));
    float3 relative = asfloat(g_sharc.Load3(a + SHARC_REPRESENTATIVE));
    // The second same-normal layer lies after the first in its probe bucket.
    // Remove the first to exercise searching past eviction holes.
    if (normal.y > 0.9f && abs(relative.y) < 1e-5f)
    {
        g_sharc.Store(stateAddress, 0u);
        [unroll] for (uint b = 0u; b < SHARC_ENTRY_BYTES; b += 16u) g_sharc.Store4(a + b, 0u);
    }
}

// GPU timestamp benchmark; every query writes its own result to prevent DCE
// without serializing the measurement on a shared atomic counter.
[numthreads(64, 1, 1)]
void benchmark(uint3 tid : SV_DispatchThreadID)
{
    uint seed = Hash32(tid.x);
    SharcSurface s = Surface(0u);
    if (testMode == 4u)
        s.position = float3((RandomFloatSingle(seed) - 0.5f) * 0.4f, 0,
            (RandomFloatSingle(seed) - 0.5f) * 0.4f);
    else
    {
        s.position = float3(RandomFloatSingle(seed), RandomFloatSingle(seed), RandomFloatSingle(seed)) * 32.0f;
        s.geometricNormal = s.normal = normalize(float3(0.3f, 0.8f, 0.5f));
    }
    float3 radiance;
    bool hit = SharcQuery(s, 100.0f, seed, radiance);
    results.Store(tid.x * 4u, asuint(radiance.x) ^ (hit ? 1u : 0u));
}

//====================================
// PATH GUIDING (SharcGuide_v8.hlsli)
//====================================
// Fixture: a receiver on the floor at the origin cell, a bright patch three
// metres above it facing down and a dim patch off to the side. Rebase modes
// express the same absolute points in a local frame eight thousand km away.
float3 GuideLocal(float3 p, bool rebase)
{
    return rebase ? float3(8000000, -8000000, 8000000) - sceneOriginWorld + p : p;
}

float3 GuideCosineDirection(float3 n, inout uint seed)
{
    float u1 = RandomFloatSingle(seed), u2 = RandomFloatSingle(seed);
    float r = sqrt(u1), phi = 2.0f * GUIDE_PI * u2;
    float3 t, b;
    GuideBasis(n, t, b);
    return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(0.0f, 1.0f - u1)));
}

// Integrand with a bright blob around the bright patch: cosine sampling
// resolves it poorly, the guided mixture must resolve it without bias.
float GuideTestIntegrand(float3 w, float3 n, float3 blob)
{
    return max(dot(n, w), 0.0f) * (1.0f + 200.0f * smoothstep(cos(12.0f * GUIDE_PI / 180.0f),
        cos(8.0f * GUIDE_PI / 180.0f), dot(w, blob)));
}

[numthreads(64, 1, 1)]
void guideFill(uint3 tid : SV_DispatchThreadID)
{
    const bool rebase = testMode == 3u;
    // Single-frame modes insert from one lane: a contended first insertion
    // deliberately misses (retry next visit), which would leave the fixture empty.
    if (testMode != 0u && tid.x != 0u) return;
    float3 receiver = GuideLocal(float3(0.04f, 0.0f, 0.04f), rebase);
    uint e = GuideFindOrInsert(GuideKeyCentre(receiver, float3(0, 1, 0)));
    if (testMode == 4u)
    {
        // Visibility feedback: the bright patch is stored, then guided rays
        // toward slot 0 miss it (testIndex times), then reach it (testIndex times).
        GuideDiscover(e, float3(0.5f, 3.2f, 0.5f), float3(0, -1, 0), 100.0f, false);
        [loop] for (uint i = 0u; i < testIndex; ++i) GuideReport(e, 0u, true, float3(0.5f, 1.5f, 0.5f));
        [loop] for (uint j = 0u; j < testIndex; ++j) GuideReport(e, 0u, true, float3(0.6f, 3.4f, 0.4f));
        return;
    }
    if (testMode == 0u || testMode == 3u)
    {
        // Mode 0: observations first (frame 1), discoveries once the mean is
        // known, so the dim patch below the mean radiance is rejected
        // deterministically. Mode 3 registers both patches against a cold
        // mean for the rebase comparison.
        if (tid.x == 0u && (testMode == 3u || sharc_frame > 1u))
        {
            GuideDiscover(e, GuideLocal(float3(0.5f, 3.2f, 0.5f), rebase), float3(0, -1, 0), 100.0f, false);
            GuideDiscover(e, GuideLocal(float3(4.2f, 1.1f, 0.3f), rebase), float3(-1, 0, 0), 1.0f, false);
        }
        if (testMode == 0u) GuideObserve(e, 20.0f); // 64 concurrent observations per frame
    }
    else if (testMode == 1u && tid.x == 0u)
    {
        // Twenty patches straight above the receiver with strictly increasing
        // contribution (luminance grows faster than distance squared).
        [loop] for (uint i = 0u; i < 20u; ++i)
        {
            float d = 3.0f + (float)i;
            GuideDiscover(e, float3(0.5f, 3.2f + (float)i, 0.5f), float3(0, -1, 0), (float)(i + 1u) * d * d, false);
        }
    }
    else if (testMode == 2u && tid.x == 0u)
    {
        // Three hits inside the same coarse patch merge into one slot.
        GuideDiscover(e, float3(0.5f, 3.2f, 0.5f), float3(0, -1, 0), 100.0f, false);
        GuideDiscover(e, float3(0.5f, 3.3f, 0.4f), float3(0, -1, 0), 50.0f, false);
        GuideDiscover(e, float3(0.6f, 3.1f, 0.5f), float3(0, -1, 0), 50.0f, false);
    }
}

[numthreads(64, 1, 1)]
void guideQuery(uint3 tid : SV_DispatchThreadID)
{
    const bool rebase = testIndex == 1u;
    const float3 n = float3(0, 1, 0);
    float3 receiver = GuideLocal(float3(0.04f, 0.0f, 0.04f), rebase);
    GuideKey key = GuideKeyCentre(receiver, n);
    uint e = GuideFind(key);
    if (testMode == 2u)
    {
        // Stochastic keys: over many draws every neighbour of the receiver cell
        // must be chosen with a tent probability that sums to one, and the
        // receiver 0.04 m from the corner must land in its own cell most often.
        if (tid.x != 0u) return;
        uint seed = 12345u;
        uint own = 0u, other = 0u;
        [loop] for (uint i = 0u; i < 4096u; ++i)
        {
            GuideKey k = GuideKeyOf(receiver, n, seed);
            if (GuideKeyMatches(k, key)) ++own; else ++other;
        }
        results.Store4(0u, asuint(float4((float)own, (float)other, 0.0f, 0.0f)));
        return;
    }
    if (testMode == 0u)
    {
        if (tid.x != 0u) return;
        if (e == GUIDE_INVALID) { results.Store4(0u, uint4(0u, 0u, 0u, 0u)); return; }
        uint4 header = g_sharc.Load4(e + GUIDE_IRRADIANCE);
        GuideSet g = GuideBuild(e, receiver, n);
        uint valid = 0u;
        [unroll] for (uint v = 0u; v < GUIDE_SLOTS; ++v)
            if ((g_sharc.Load(GuideSlotAddress(e, v) + 4u) & GUIDE_SLOT_VALID) != 0u) ++valid;
        float mean = SharcFloat(header.xy, rcp(SHARC_RADIANCE_SCALE)) / max((float)header.z, 1.0f);
        results.Store4(0u, asuint(float4(1.0f, (float)header.z, mean, g.q)));
        results.Store4(16u, asuint(float4(g.weightSum, (float)valid, 0.0f, 0.0f)));
        results.Store4(32u, uint4(asuint(key.cell), key.meta));
        [unroll] for (uint c = 0u; c < GUIDE_SLOTS; ++c)
        {
            uint4 w = g_sharc.Load4(GuideSlotAddress(e, c));
            GuideSlot s = GuideDecodeSlot(w);
            results.Store4(64u + c * 32u, asuint(float4(s.offset, (float)s.level)));
            results.Store4(80u + c * 32u, asuint(float4(s.luminance, (float)s.hits,
                (float)s.visibility, s.valid ? 1.0f : 0.0f)));
            results.Store4(320u + c * 16u, w);
        }
        return;
    }
    if (testMode == 1u)
    {
        // Monte Carlo per lane: the guide pdf integrates to one over the sphere,
        // the production mixture estimates the blob integrand without bias and
        // with far less variance than cosine sampling.
        GuideSet g = GuideBuild(e, receiver, n);
        float3 blob = normalize(GuideLocal(float3(0.5f, 3.5f, 0.5f), rebase) - receiver);
        uint seed = Hash32(tid.x * 7919u + 17u);
        const uint N = 16384u;
        float pdfIntegral = 0.0f, mix = 0.0f, mix2 = 0.0f, cosine = 0.0f, cosine2 = 0.0f;
        [loop] for (uint i = 0u; i < N; ++i)
        {
            float z = 1.0f - 2.0f * RandomFloatSingle(seed);
            float phi = 2.0f * GUIDE_PI * RandomFloatSingle(seed);
            float s = sqrt(max(0.0f, 1.0f - z * z));
            pdfIntegral += GuidePdf(g, float3(cos(phi) * s, sin(phi) * s, z)) * 4.0f * GUIDE_PI;

            uint pick;
            float3 dm = RandomFloatSingle(seed) < g.q ? GuideSample(g, seed, pick) : GuideCosineDirection(n, seed);
            // A Lambertian receiver: the guided share is the whole cosine lobe.
            float pm = GuideMixPdf(g, 1.0f, GuideLambertPdf(n, dm), dm, GuideLambertPdf(n, dm));
            float fm = pm > 0.0f ? GuideTestIntegrand(dm, n, blob) / pm : 0.0f;
            mix += fm; mix2 += fm * fm;

            float3 dc = GuideCosineDirection(n, seed);
            float pc = GuideLambertPdf(n, dc);
            float fc = pc > 0.0f ? GuideTestIntegrand(dc, n, blob) / pc : 0.0f;
            cosine += fc; cosine2 += fc * fc;
        }
        results.Store4(tid.x * 32u, asuint(float4(pdfIntegral, mix, mix2, cosine) / (float)N));
        results.Store4(tid.x * 32u + 16u, asuint(float4(cosine2 / (float)N, g.q, g.weightSum, e == GUIDE_INVALID ? 0.0f : 1.0f)));
    }
}

//====================================
//RESTIR LITE (RestirLite_v8.hlsli): PACKING AND RESAMPLING MIS
//====================================
// Mode 0: reservoir pack/unpack round trip.
// Mode 1: pairwise (spatial) and balance (temporal) MIS weights sum to one
//         for random targets and confidences, including partners that
//         cannot produce the sample.
// Mode 2: end-to-end resampling on a discrete domain, three pixels with
//         different generation densities, targets and supports through the
//         paired merge; mode 3 the same through the two-strategy temporal
//         merge. The mean of f(y) * W over many trials must equal the
//         exact sum: the merges are unbiased for any target proxy as long
//         as every strategy's target vanishes exactly where it cannot
//         produce a sample.
static const uint LITE_TEST_ITEMS = 8u;
static const uint LITE_TEST_TRIALS = 4096u;

float LiteTestValue(uint pixel, uint k)
{
    const float base = (float)(k + 1u);
    if (pixel == 0u) return base;                        // canonical target = integrand
    if (pixel == 1u) return base * ((k & 1u) ? 0.5f : 2.0f);
    return k >= 6u ? 0.0f : base * 1.5f;                 // pixel 2 cannot produce items 6, 7
}

float LiteTestDensity(uint pixel, uint k)
{
    if (pixel == 0u) return (float)(9u - k);
    if (pixel == 1u) return 1.0f;
    return k >= 6u ? 0.0f : (float)(k + 2u);
}

uint LiteTestDraw(uint pixel, inout uint seed, out float pdf)
{
    float total = 0.0f;
    for (uint i = 0u; i < LITE_TEST_ITEMS; ++i) total += LiteTestDensity(pixel, i);
    float u = RandomFloatPCG(seed) * total;
    uint last = 0u;
    for (uint k = 0u; k < LITE_TEST_ITEMS; ++k)
    {
        const float d = LiteTestDensity(pixel, k);
        if (d <= 0.0f) continue;
        if (u < d) { pdf = d / total; return k; }
        u -= d;
        last = k;
    }
    pdf = LiteTestDensity(pixel, last) / total;
    return last;
}

[numthreads(64, 1, 1)]
void liteCheck(uint3 tid : SV_DispatchThreadID)
{
    uint seed = Hash32(tid.x * 0x9E3779B9u + testMode * 0x85EBCA6Bu + 7u);
    if (testMode == 0u)
    {
        if (tid.x != 0u) return;
        LiteReservoir r;
        r.s.position = float3(1.5f, -2.25f, 3.0f);
        r.s.instance = 7u;
        r.s.radiance = float3(0.5f, 123.0f, 4000.0f);
        r.s.normal = normalize(float3(0.3f, 0.8f, -0.5f));
        r.s.kind = LITE_KIND_SURFACE;
        r.W = 3.75f;
        r.M = 200u;
        r.tint = float3(1.0f, 0.5f, 0.25f);
        LiteStore(results, 8u, r);
        const LiteReservoir q = LiteLoad(results, 8u);
        const float3 relL = abs(q.s.radiance - r.s.radiance) / max(r.s.radiance.x, max(r.s.radiance.y, r.s.radiance.z));
        results.Store4(0u, asuint(float4(length(q.s.position - r.s.position), (float)q.s.instance,
            max(relL.x, max(relL.y, relL.z)), dot(q.s.normal, r.s.normal))));
        results.Store4(16u, asuint(float4(q.W, (float)q.M, (float)q.s.kind, length(q.tint - r.tint))));
        return;
    }
    if (testMode == 1u)
    {
        float worst = 0.0f;
        for (uint trial = 0u; trial < 256u; ++trial)
        {
            const uint n = 1u + (trial % 3u);
            const float Mc = 1.0f + floor(RandomFloatPCG(seed) * 8.0f);
            const float pcc = 0.05f + RandomFloatPCG(seed); // the canonical can produce the sample
            float Mi[3], pii[3];
            float Msum = Mc;
            for (uint i = 0u; i < 3u; ++i)
            {
                Mi[i] = i < n ? 1.0f + floor(RandomFloatPCG(seed) * 16.0f) : 0.0f;
                pii[i] = (RandomFloatPCG(seed) < 0.25f) ? 0.0f : RandomFloatPCG(seed) * 3.0f;
                Msum += Mi[i];
            }
            const float O = Msum - Mc;
            float sum = Mc / Msum;
            for (uint j = 0u; j < 3u; ++j)
            {
                if (j >= n) continue;
                sum += LiteMisCanonicalTerm(Mi[j], pcc, Mc, pii[j], O, Msum);
                sum += LiteMisPartner(Mi[j], pii[j], Mc, pcc, O, Msum);
            }
            worst = max(worst, abs(sum - 1.0f));
        }
        results.Store(tid.x * 4u, asuint(worst));
        return;
    }
    // Mode 2: Monte Carlo estimate of sum_k f_c(k) through the paired spatial merge.
    float exact = 0.0f;
    for (uint k = 0u; k < LITE_TEST_ITEMS; ++k) exact += LiteTestValue(0u, k);
    const float Mc = 1.0f, M1 = 3.0f, M2 = 2.0f;
    float estimate = 0.0f;
    for (uint trial = 0u; trial < LITE_TEST_TRIALS; ++trial)
    {
        float p0, p1, p2;
        const uint k0 = LiteTestDraw(0u, seed, p0);
        const uint k1 = LiteTestDraw(1u, seed, p1);
        const uint k2 = LiteTestDraw(2u, seed, p2);
        // one-candidate RIS per pixel: W = 1 / p
        const float W0 = 1.0f / p0, W1 = 1.0f / p1, W2 = 1.0f / p2;
        const float Msum = Mc + M1 + M2, O = M1 + M2;
        const float pcc = LiteTestValue(0u, k0);
        const float pc1 = LiteTestValue(0u, k1), p1c = LiteTestValue(1u, k0), p11 = LiteTestValue(1u, k1);
        const float pc2 = LiteTestValue(0u, k2), p2c = LiteTestValue(2u, k0), p22 = LiteTestValue(2u, k2);
        const float mc = Mc / Msum + LiteMisCanonicalTerm(M1, pcc, Mc, p1c, O, Msum)
            + LiteMisCanonicalTerm(M2, pcc, Mc, p2c, O, Msum);
        const float m1 = LiteMisPartner(M1, p11, Mc, pc1, O, Msum);
        const float m2 = LiteMisPartner(M2, p22, Mc, pc2, O, Msum);
        const float w0 = mc * pcc * W0, w1 = m1 * pc1 * W1, w2 = m2 * pc2 * W2;
        float wsum = w0, phatSel = pcc;
        uint sel = k0;
        wsum += w1; if (w1 > 0.0f && RandomFloatPCG(seed) * wsum < w1) { sel = k1; phatSel = pc1; }
        wsum += w2; if (w2 > 0.0f && RandomFloatPCG(seed) * wsum < w2) { sel = k2; phatSel = pc2; }
        if (wsum > 0.0f && phatSel > 0.0f) estimate += LiteTestValue(0u, sel) * wsum / phatSel;
    }
    results.Store(tid.x * 4u, asuint(estimate / (float)LITE_TEST_TRIALS));
    if (tid.x == 0u) results.Store(64u * 4u, asuint(exact));
}
