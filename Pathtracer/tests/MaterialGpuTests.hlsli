// Production layered-material math with deterministic material/LUT fixtures.
// No renderer or textures required. Covers diffuse, dielectric, metal, coat,
// sheen, solid/thin glass, anisotropy, grazing angles and removed diffuse lobes.
#include "Constants_v8.hlsli"
float Avg3(float3 v) { return (v.x + v.y + v.z) / 3.0f; }
float LoadKd_w(uint m) { return m == 5u || m == 6u ? 0.0f : 1.0f; }
float LoadNi(uint m) { return m == 0u ? 1.0f : 1.5f; }
float LoadAniso(uint m) { return m == 7u ? 0.8f : 0.0f; }
float LoadAnisoRot(uint m) { return m == 7u ? 0.3f : 0.0f; }
float LoadPc(uint m) { return m == 3u || m == 4u ? 0.75f : 0.0f; }
float LoadPcr(uint m) { return 0.2f; }
float LoadPs(uint m) { return m == 4u ? 0.6f : 0.0f; }
bool LoadIsThinGlass(uint m) { return m == 6u; }
float3 LoadTf(uint m) { return float3(0.7f, 0.8f, 0.9f); }

#define SHEEN_LUT_INDEX 0
#define GGX_ESS_LUT_INDEX 1
struct MaterialFixtureLut
{
    float4 SampleLevel(uint samplerID, float3 uv, float lod)
    {
        return float4(0.55f + 0.35f * saturate(uv.y) * (1.0f - 0.5f * uv.x), 0, 0, 0);
    }
};
static MaterialFixtureLut g_LUT;
static const uint g_sampler_LUT = 0u;
struct MaterialFixtureInstance { uint opaqueTriCount; };
static MaterialFixtureInstance instanceProps[1];
#include "Fresnel_v8.hlsli"
#include "Material_Common_v8.hlsli"
#include "Material_GGX_v8.hlsli"
#include "Material_Lambertian_v8.hlsli"
#include "Material_Coat_v8.hlsli"
#include "Material_Sheen_v8.hlsli"
#include "BXDF_v8.hlsli"

float MaterialRelativeError(float4 a, float4 b)
{
    if (!all(isfinite(a)) || !all(isfinite(b))) return 1e10f;
    float4 e = abs(a - b) / max(1.0f, abs(a));
    return max(max(e.x, e.y), max(e.z, e.w));
}

[numthreads(64, 1, 1)]
void materialCheck(uint3 tid : SV_DispatchThreadID)
{
    uint seed = Hash32(tid.x);
    float worst = 0.0f;
    [loop] for (uint i = 0u; i < 256u; ++i)
    {
        uint m = i & 7u;
        float3 n = float3(0, 1, 0);
        float z = 0.001f + 0.998f * RandomFloatSingle(seed);
        float3 v = float3(sqrt(1.0f - z * z), z, 0);
        float3 l = normalize(float3(RandomFloatSingle(seed) * 2.0f - 1.0f,
            RandomFloatSingle(seed) * 2.0f - 1.0f, RandomFloatSingle(seed) * 2.0f - 1.0f));
        half rough = (half)(0.08f + 0.9f * RandomFloatSingle(seed));
        half metal = m == 2u ? (half)1.0f : (half)0.0f;
        half etaI = (i & 16u) != 0u ? (half)LoadNi(m) : (half)1.0f;
        half etaT = (i & 16u) != 0u ? (half)1.0f : (half)LoadNi(m);
        float3 kd = float3(0.25f, 0.5f, 0.75f);
        SharcSurface surface = (SharcSurface)0;
        SharcDescriptor descriptor = (SharcDescriptor)0;
        surface.normal = surface.geometricNormal = descriptor.normal = descriptor.geometricNormal = n;
        descriptor.demodulator = 1.0f;
        float3 ratio = exp2(float3(RandomFloatSingle(seed), RandomFloatSingle(seed), RandomFloatSingle(seed)) * 1.6f - 0.8f);
        surface.demodulator = ratio;
        float referenceWeight = 1.0f - smoothstep(0.3f, 0.7f,
            max(abs(log2(ratio.x)), max(abs(log2(ratio.y)), abs(log2(ratio.z)))));
        worst = max(worst, abs(referenceWeight - SharcSurfaceWeight(descriptor, surface, 0.0f, 1.0f)));
        SamplingP p = CalculateStrategyProbabilities(m, v, n, etaI, etaT, kd, metal);
        if ((i & 32u) != 0u) DropDiffuseLobe(p);
        BrdfData full = EvaluateAndPdf_COMBINED(p, m, n, n, l, v, kd, rough, metal, etaI, etaT);
        [unroll] for (uint strategy = 0u; strategy < 4u; ++strategy)
        {
            float3 val; float pdf;
            BrdfData fused = EvaluateAndPdf_COMBINED_L(p, strategy, m, n, n, l, v,
                kd, rough, metal, etaI, etaT, false, val, pdf);
            BrdfData separate = EvaluateLobePdf_COMBINED(p, strategy, m, n, n, l, v,
                kd, rough, metal, etaI, etaT);
            worst = max(worst, MaterialRelativeError(float4(full.val, full.pdf), float4(fused.val, fused.pdf)));
            worst = max(worst, MaterialRelativeError(float4(separate.val, separate.pdf), float4(val, pdf)));
        }
    }
    results.Store(tid.x * 4u, asuint(worst));
}

// Arithmetic throughput fixture. Material IDs are coherent within blocks;
// directions and roughness vary per lane. Analytic LUTs exclude texture cost.
[numthreads(64, 1, 1)]
void materialBenchmark(uint3 tid : SV_DispatchThreadID)
{
    uint seed = Hash32(tid.x);
    uint m = (tid.x >> 12u) & 7u;
    float3 n = float3(0, 1, 0);
    float3 v = normalize(float3(0.5f, 0.01f + RandomFloatSingle(seed), 0));
    float3 l = normalize(float3(RandomFloatSingle(seed) - 0.5f,
        0.01f + RandomFloatSingle(seed), RandomFloatSingle(seed) - 0.5f));
    half rough = (half)(0.08f + 0.9f * RandomFloatSingle(seed));
    half metal = m == 2u ? (half)1.0f : (half)0.0f;
    float3 kd = float3(0.25f, 0.5f, 0.75f);
    SamplingP p = CalculateStrategyProbabilities(m, v, n, (half)1.0f, (half)LoadNi(m), kd, metal);
    if (testMode == 34u) DropDiffuseLobe(p);
    BrdfData full;
    float3 diffuse = 0.0f;
    if (testMode == 35u)
    {
        float diffusePdf;
        full = EvaluateAndPdf_COMBINED_L(p, 0u, m, n, n, l, v, kd, rough, metal,
            (half)1.0f, (half)LoadNi(m), false, diffuse, diffusePdf);
    }
    else
    {
        full = EvaluateAndPdf_COMBINED(p, m, n, n, l, v, kd, rough, metal, (half)1.0f, (half)LoadNi(m));
        if (testMode == 36u)
            diffuse = EvaluateLobePdf_COMBINED(p, 0u, m, n, n, l, v, kd, rough, metal,
                (half)1.0f, (half)LoadNi(m)).val;
    }
    results.Store(tid.x * 4u, asuint(dot(full.val + diffuse, kd) + full.pdf));
}
