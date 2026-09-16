cbuffer TestConstants : register(b0)
{
    uint sharc_reset, sharc_frame, sharc_minSamples, sharc_historyFrames;
    uint sharc_maxAge, testMode;
    float sharc_cellSize, sharc_lodScale;
    float sharc_queryFootprint;
    float3 sceneOriginWorld;
    float3 testCamera;
    uint testIndex;
    uint guide_params;
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
#ifndef PI
#define PI 3.1415926535
#endif
#define RAY_TMAX_PLANET 1e9f
#define ucw_clampMax 10000.0f
#include "RestirLite_v8.hlsli"
#include "Temporal_ReuseMath_v8.hlsli"
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
    if (mode == 1u)
    {
        s.position.y = -0.001f;
        s.geometricNormal = s.normal = float3(0, -1, 0);
        s.variant = 1u;
    }
    if (mode == 2u) s.position.y = 0.02f;
    if (mode == 16u || mode == 17u) s.demodulator = float3(0.25f, 0.5f, 0.75f);
    if (mode == 17u) s.position.x += 0.5f;
    if (mode == 8u) s.geometricNormal = s.normal = normalize(float3(1, 1, 0));
    return s;
}

[numthreads(64, 1, 1)]
void fill(uint3 tid : SV_DispatchThreadID)
{
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
    if (testMode == 11u && tid.x != 0u) return;
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
    if (testMode == 10u) { g_sharc.Store(stateAddress, 1u); return; }
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
    if (testMode == 12u)
    {
        g_sharc.Store(a + SHARC_MEAN, 0x7fc00000u);
        g_sharc.Store(a + SHARC_HISTORY_W, 0x7fc00000u);
        g_sharc.Store(a + SHARC_CONFIDENCE, asuint(1.0f));
        g_sharc.Store2(a + SHARC_QUERY + 20u, uint2(0x7e007e00u, 0xff007e00u));
        return;
    }
    if (testMode == 9u)
    {
        g_sharc.Store(a + SHARC_NODE, g_sharc.Load(a + SHARC_NODE) ^ 0x40000000u);
        uint4 key; uint2 idm;
        SharcLoadKey(a, key, idm);
        g_sharc.Store(a + SHARC_QUERY, SharcCheckHash(asint(key.xyz), key.w, idm.x, idm.y));
        return;
    }
    float3 normal = UnpackNormal(g_sharc.Load(a + SHARC_GEOMETRIC_NORMAL));
    float3 relative = SharcLoadRepresentative(a);
    if (normal.y > 0.9f && abs(relative.y) < 1e-5f)
    {
        g_sharc.Store(stateAddress, 0u);
        [unroll] for (uint b = 0u; b < SHARC_ENTRY_BYTES; b += 16u) g_sharc.Store4(a + b, 0u);
    }
}

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

float GuideBlob(float3 w, float3 blob)
{
    return smoothstep(cos(12.0f * GUIDE_PI / 180.0f), cos(8.0f * GUIDE_PI / 180.0f), dot(w, blob));
}

float GuideTestRadiance(float3 w, uint mode, float3 blob, float3 blob2)
{
    if (mode == 6u) return 1.0f;
    float L = 1.0f + 200.0f * GuideBlob(w, blob);
    if (mode == 5u)
        L += 20.0f * smoothstep(cos(20.0f * GUIDE_PI / 180.0f), cos(15.0f * GUIDE_PI / 180.0f), dot(w, blob2));
    return L;
}

float GuideTestIntegrand(float3 w, float3 n, float3 blob)
{
    return max(dot(n, w), 0.0f) * (1.0f + 200.0f * GuideBlob(w, blob));
}

float3 GuideTestBlob2() { return normalize(float3(0.866f, 0.5f, 0.0f)); }

[numthreads(64, 1, 1)]
void guideFill(uint3 tid : SV_DispatchThreadID)
{
    const bool rebase = testMode == 3u;
    const uint mode = (rebase || testMode == 8u) ? 0u : testMode;
    const float3 n = float3(0, 1, 0);
    const float3 receiver = GuideLocal(float3(0.04f, 0.0f, 0.04f), rebase);
    GuideKey key = GuideKeyCentre(receiver, n);
    if (testMode == 8u) key = GuideParentKey(key);
    const uint e = GuideFindOrInsert(key);
    if (e == GUIDE_INVALID) return;
    const float3 blob = normalize(GuideLocal(float3(0.5f, 3.5f, 0.5f), rebase) - receiver);
    const float3 blob2 = GuideTestBlob2();
    const float3 center = GuideCellCenter(GuideLoadKey(e));
    uint seed = Hash32(tid.x * 7919u + sharc_frame * 104729u + 17u);
    GuideSet g = GuideEmpty();
    if ((testIndex & 256u) != 0u) g = GuideBuild(e, receiver, n, true);
    const uint pilots = max(testIndex & 255u, 1u);
    [loop] for (uint i = 0u; i < pilots; ++i)
    {
        uint pick;
        const float3 w = RandomFloatSingle(seed) < g.q ? GuideSample(g, seed, pick) : GuideCosineDirection(n, seed);
        const float pc = GuideLambertPdf(n, w);
        const float pdf = GuideMixPdf(g, 1.0f, pc, w, pc);
        const float ratio = pdf > 0.0f ? pc / pdf : 0.0f;
        const float ret = ratio * 0.5f * GuideTestRadiance(w, mode, blob, blob2);
        float3 wRef = w;
        float invDist = 0.0f;
        if (dot(w, blob) > cos(12.0f * GUIDE_PI / 180.0f))
        {
            const float3 v = receiver + 3.5f * w - center;
            invDist = rsqrt(dot(v, v));
            wRef = v * invDist;
        }
        GuideObserve(e, wRef, invDist, ret, ratio, seed);
        seed = Hash32(seed);
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
        results.Store4(448u, asuint(float4(GuideBuildAt(key, receiver, n, false).q, 0.0f, 0.0f, 0.0f)));
        if (e == GUIDE_INVALID) { results.Store4(0u, uint4(0u, 0u, 0u, 0u)); return; }
        const uint4 header = g_sharc.Load4(e + GUIDE_EVIDENCE);
        const uint2 counts = g_sharc.Load2(e + GUIDE_OPPORTUNITIES);
        const GuideSet render = GuideBuild(e, receiver, n, false);
        const GuideSet training = GuideBuild(e, receiver, n, true);
        float measuredBranch = 0.0f;
        [unroll] for (uint c = 0u; c < GUIDE_SLOTS; ++c)
        {
            const uint4 w = g_sharc.Load4(GuideLobeAddress(e, c));
            const uint m = GuideMomentAddress(e, c);
            float support;
            const float mass = GuideMassOf(GuideWord(g_sharc.Load2(m)), support);
            const float phi = (float)g_sharc.Load(m + GUIDE_MOMENT_PHI) / (float)GUIDE_OPPORTUNITY_SCALE;
            const bool valid = (w.w & GUIDE_LOBE_VALID) != 0u, measured = (w.w & GUIDE_LOBE_MEASURED) != 0u;
            if (valid && measured) measuredBranch += f16tof32(w.y >> 16u);
            results.Store4(64u + c * 48u, asuint(float4(UnpackNormal(w.x), f16tof32(w.y & 0xffffu))));
            results.Store4(80u + c * 48u, asuint(float4(f16tof32(w.y >> 16u), asfloat(w.z), mass, support)));
            results.Store4(96u + c * 48u, asuint(float4(phi, measured ? 1.0f : 0.0f, (float)(w.w & 255u), valid ? 1.0f : 0.0f)));
        }
        results.Store4(0u, asuint(float4(1.0f, (float)counts.x / (float)GUIDE_OPPORTUNITY_SCALE, asfloat(header.y), render.q)));
        results.Store4(16u, asuint(float4(render.weightSum, (float)((header.z >> 4u) & 15u), (float)(header.z & 15u),
            (float)counts.y / (float)GUIDE_OPPORTUNITY_SCALE)));
        results.Store4(32u, uint4(asuint(key.cell), key.meta));
        results.Store4(48u, asuint(float4(GuideUnsignedOf(GuideWord(g_sharc.Load2(e + GUIDE_UNEXPLAINED))), training.q,
            measuredBranch, (g_sharc.Load(e + GUIDE_META) & GUIDE_META_WORTH) != 0u ? 1.0f : 0.0f)));
        return;
    }
    if (testMode == 1u)
    {
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
    if (testMode == 4u)
    {
        const float apertures[4] = { 0.001f, 0.1f, 1.0f, 2.0f };
        const float a = apertures[tid.x & 3u];
        const float3 axis = normalize(float3(0.3f, 0.8f, -0.5f));
        float3 t, b;
        GuideBasis(axis, t, b);
        uint seed = Hash32(tid.x * 6271u + 3u);
        const uint N = 65536u;
        float integral = 0.0f, meanCos = 0.0f;
        [loop] for (uint i = 0u; i < N; ++i)
        {
            const float z = 1.0f - a * ((float)i + 0.5f) / (float)N;
            const float3 d = axis * z + t * sqrt(max(1.0f - z * z, 0.0f));
            integral += GuideCapPdf(axis, a, d) * 2.0f * GUIDE_PI * a;
            const float3 s = GuideCapSample(axis, a, float2(RandomFloatSingle(seed), RandomFloatSingle(seed)));
            meanCos += dot(s, axis);
        }
        results.Store4(tid.x * 16u, asuint(float4(integral / (float)N, meanCos / (float)N, 1.0f - a / 3.0f, a)));
    }
}

static const uint LITE_TEST_ITEMS = 8u;
static const uint LITE_TEST_TRIALS = 4096u;

float LiteTestValue(uint pixel, uint k)
{
    const float base = (float)(k + 1u);
    if (pixel == 0u) return base;
    if (pixel == 1u) return base * ((k & 1u) ? 0.5f : 2.0f);
    return k >= 6u ? 0.0f : base * 1.5f;
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
    if (testMode == 4u)
    {
        if (tid.x != 0u) return;
        results.Store4(0u, asuint(float4(
            TemporalConfidenceCap(8u, 0.0f, 0.025f, false),
            TemporalConfidenceCap(8u, 1.0f, 0.025f, false),
            TemporalConfidenceCap(8u, 1.0f, 0.025f, true),
            TemporalConfidenceCap(0u, 0.0f, 0.025f, false))));
        float error = 0.0f;
        uint lastCap = 8u;
        for (uint i = 0u; i <= 288u; ++i)
        {
            const uint cap = TemporalConfidenceCap(8u, (float)i / 288.0f, 0.025f, false);
            if (cap > lastCap || cap < 1u) error += 1.0f;
            lastCap = cap;
        }
        results.Store(32u, asuint(error));
        for (uint permutation = 0u; permutation < 16u; ++permutation)
        {
            for (int x = -4; x < 20; ++x)
            {
                const int2 original = int2(x, 17 - x);
                int2 mapped = original;
                ApplyPermutationSampling(mapped, permutation);
                ApplyPermutationSampling(mapped, permutation);
                if (any(mapped != original)) error += 1.0f;
            }
        }
        results.Store(36u, asuint(error));
        return;
    }
    if (testMode == 1u)
    {
        float worst = 0.0f;
        for (uint trial = 0u; trial < 256u; ++trial)
        {
            const uint n = 1u + (trial % 3u);
            const float Mc = 1.0f + floor(RandomFloatPCG(seed) * 8.0f);
            const float pcc = 0.05f + RandomFloatPCG(seed);
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
    if (testMode == 3u)
    {
        const float dist = exp2((float)(tid.x % 16u));
        const float cosX = 0.2f + 0.1f * (float)(tid.x % 8u);
        const float cosY = 0.2f + 0.1f * (float)(tid.x / 8u);
        const float pdf = cosX * LITE_INV_PI;
        const LiteLink link = LiteLinkFrom(dist, cosX, cosY, false);
        LiteSample sample = LiteEmpty(1u).s;
        sample.instance = 0u;
        sample.kind = LITE_KIND_SURFACE;
        sample.radiance = float3(2.0f, 3.0f, 4.0f);
        const float3 albedo = float3(0.25f, 0.5f, 0.75f);
        const float phat = LiteTarget(albedo, sample, link, 0.0f, 1.0f);
        const float expected = Luma(albedo * sample.radiance);
        const float wsum = expected * LITE_INV_PI * cosX / pdf;
        const float W = LiteSanitizeWeight(wsum / phat);
        results.Store(tid.x * 4u, asuint(phat * W / expected));
        return;
    }
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

#define IMG_W 53u
#define IMG_H 45u
#define DUP_KEY uint
#include "Duplication_Map_v8.hlsli"
[numthreads(16, 16, 1)]
void legacyDupCheck(uint3 local : SV_GroupThreadID)
{
    const uint2 group = testMode == 1u ? uint2(0u, 0u)
        : (testMode == 2u ? uint2(3u, 2u) : uint2(1u, 1u));
    const uint2 pixel = group * 16u + local.xy;
    const uint tlin = local.y * TILE_W + local.x;
    for (uint i = 0u; i < LOADS_PER_THREAD; ++i)
    {
        const uint index = tlin * LOADS_PER_THREAD + i;
        const int2 p = int2(group * 16u) - int2(WIN_R, WIN_R) + int2(index % CACHE_W, index / CACHE_W);
        uint key = 7u;
        if (testMode == 3u) key = (uint)(p.y * (int)IMG_W + p.x);
        if (testMode == 4u) key = (uint)p.x & 1u;
        if (testMode == 5u) key = 0u;
        if (any(p < 0) || any(p >= int2(IMG_W, IMG_H))) key = 0u;
        s_V2[index / CACHE_W][index % CACHE_W] = key;
    }
    GroupMemoryBarrierWithGroupSync();
    const float D = any(pixel >= uint2(IMG_W, IMG_H)) ? 0.0f
        : DuplicationFraction(uint3(pixel, 0u), uint3(group, 0u), local);
    results.Store(tlin * 4u, asuint(D));
}
