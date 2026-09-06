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
};
float3 InitOrigin() { return testCamera; }
float Luma(float3 c) { return dot(c, float3(0.2126f, 0.7152f, 0.0722f)); }
#include "../shaders/Compression_v8.hlsli"
#include "../shaders/Random_v8.hlsli"
#define SHARC_TEST 1
#define SHARC_UPDATE_PASS 1
#include "../shaders/SharcPath_v8.hlsli"
#include "../shaders/SharcDebug_v8.hlsli"
#define main prepare
#include "../shaders/Pass_sharc_prepare_v8.hlsl"
#undef main
#define main resolve
#include "../shaders/Pass_sharc_resolve_v8.hlsl"
#undef main
RWByteAddressBuffer results : register(u0);

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
