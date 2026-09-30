#include "Constants_v8.hlsli"
float LoadKd_w(uint m) { return m == 5u || m == 6u || m == 10u ? 0.0f : 1.0f; }
bool LoadIsOceanMaterial(uint m) { return m == 10u; }
float LoadNi(uint m) { return m == 0u ? 1.0f : 1.5f; }
float LoadDiffuseRoughness(uint m) { return testMode >= 100u ? testCamera.y : (m == 0u ? 0.0f : 0.5f); }
float LoadAniso(uint m) { return m == 7u ? 0.8f : 0.0f; }
float LoadAnisoRot(uint m) { return m == 7u ? 0.3f : 0.0f; }
float LoadPc(uint m) { return m == 3u || m == 4u || m == 8u || m == 9u ? 0.75f : 0.0f; }
float LoadPcr(uint m) { return testMode >= 100u ? testCamera.x : (m == 8u ? 0.06f : (m == 9u ? 0.0f : 0.2f)); }
float LoadPs(uint m) { return m == 4u ? 0.6f : 0.0f; }
bool LoadIsThinGlass(uint m) { return m == 6u; }
float3 LoadTf(uint m) { return float3(0.7f, 0.8f, 0.9f); }

#define SHEEN_LUT_INDEX 0
#define GGX_ESS_LUT_INDEX 1
struct MaterialFixtureLut
{
    float4 SampleLevel(uint samplerID, float3 uv, float lod)
    {
        if (testMode >= 100u)
            return uv.z == SHEEN_LUT_INDEX ? float4(testCamera.z, 0, 0, 0) : float4(asfloat(testPad), 0);
        return float4(0.55f + 0.35f * saturate(uv.y) * (1.0f - 0.5f * uv.x), .04f, .01f, 0);
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
#include "OceanGpuTests.hlsli"

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
    float worst = OceanRegressionError();
    [loop] for (uint i = 0u; i < 256u; ++i)
    {
        uint m = i % 9u == 8u ? 10u : (i & 7u);
        float3 n = float3(0, 1, 0);
        float z = 0.001f + 0.998f * RandomFloatSingle(seed);
        float3 v = float3(sqrt(1.0f - z * z), z, 0);
        float3 l = normalize(float3(RandomFloatSingle(seed) * 2.0f - 1.0f,
            RandomFloatSingle(seed) * 2.0f - 1.0f, RandomFloatSingle(seed) * 2.0f - 1.0f));
        half rough = (half)(0.08f + 0.9f * RandomFloatSingle(seed));
        if (m == 10u && (i & 1u) != 0u) rough = (half)0.0f;
        half metal = m == 2u ? (half)1.0f : (half)0.0f;
        half etaI = (i & 16u) != 0u ? (half)LoadNi(m) : (half)1.0f;
        half etaT = (i & 16u) != 0u ? (half)1.0f : (half)LoadNi(m);
        float3 kd = float3(0.25f, 0.5f, 0.75f);
        SharcSurface surface = (SharcSurface)0;
        SharcDescriptor descriptor = (SharcDescriptor)0;
        surface.normal = surface.geometricNormal = descriptor.normal = descriptor.geometricNormal = n;
        descriptor.demodulator = 1.0f;
        float3 ratio = exp2(float3(RandomFloatSingle(seed), RandomFloatSingle(seed), RandomFloatSingle(seed)) * 5.0f - 2.5f);
        surface.demodulator = ratio;
        float referenceWeight = 1.0f - smoothstep(SHARC_SIMILAR_ALBEDO.x, SHARC_SIMILAR_ALBEDO.y,
            max(abs(log2(ratio.x)), max(abs(log2(ratio.y)), abs(log2(ratio.z)))));
        worst = max(worst, abs(referenceWeight - SharcSurfaceWeight(descriptor, surface, 0.0f, 1.0f)));
        SamplingP p = CalculateStrategyProbabilities(m, v, n, etaI, etaT, kd, rough, metal);
        if ((i & 32u) != 0u) DropBroadLobes(p, rough);
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
        {
            float3 val; float pdf;
            BrdfData fused = EvaluateAndPdf_COMBINED_L(p, LOBE_BROAD, m, n, n, l, v,
                kd, rough, metal, etaI, etaT, false, val, pdf);
            BrdfData separate = EvaluateLobePdf_COMBINED(p, LOBE_BROAD, m, n, n, l, v,
                kd, rough, metal, etaI, etaT);
            worst = max(worst, MaterialRelativeError(float4(full.val, full.pdf), float4(fused.val, fused.pdf)));
            worst = max(worst, MaterialRelativeError(float4(separate.val, separate.pdf), float4(val, pdf)));
            BrdfData d = EvaluateLobePdf_COMBINED(p, 0u, m, n, n, l, v, kd, rough, metal, etaI, etaT);
            BrdfData g = EvaluateLobePdf_COMBINED(p, 1u, m, n, n, l, v, kd, rough, metal, etaI, etaT);
            const bool dif = p.Pdiff >= EPSILON;
            const bool ggx = p.Pspec >= EPSILON && IsBroadGGX(rough);
            const float pShare = (dif ? p.Pdiff : 0.0f) + (ggx ? p.Pspec : 0.0f);
            const float3 sumVal = (dif ? d.val : 0.0f) + (ggx ? g.val : 0.0f);
            const float  sumPdf = pShare > 0.0f
                ? ((dif ? p.Pdiff * d.pdf : 0.0f) + (ggx ? p.Pspec * g.pdf : 0.0f)) / pShare : 0.0f;
            worst = max(worst, MaterialRelativeError(float4(sumVal, sumPdf), float4(val, pdf)));
        }
    }
    results.Store(tid.x * 4u, asuint(worst));
}

[numthreads(64, 1, 1)]
void materialSamplingCheck(uint3 tid : SV_DispatchThreadID)
{
    const float3 n = float3(0, 1, 0);
    uint seed = Hash32(tid.x + 0x6d61746cu);
    if (testMode >= 100u)
    {
        float nv = asfloat(testIndex);
        float3 v = float3(sqrt(1.0f - nv * nv), nv, 0);
        half rough = (half)testCamera.x;
        float4 sum = 0.0f;
        const uint count = 32768u;
        uint m = testMode == 101u ? 1u : (testMode == 102u ? 3u : 4u);
        SamplingP p = CalculateStrategyProbabilities(m, v, n, (half)1.0f, (half)1.5f,
                                                     1.0f, rough, (half)0.0f);
        [loop] for (uint i = 0u; i < count; ++i)
        {
            float3 l = CosineUnitVectorInHemisphere(n, seed);
            if (testMode == 100u)
            {
                // Independent VNDF-sampled reference.
                float3 h = SampleVNDF_H((float)rough * (float)rough, v, n, seed);
                float3 reflected = reflect(-v, h);
                if (dot(n, reflected) > 0.0f)
                {
                    float weight = G1_SmithGGX(dot(n, reflected), (float)rough * (float)rough);
                    float f = pow(saturate(1.0f - dot(v, h)), 5.0f);
                    sum.xyz += weight * float3(1, f, f * f);
                }
                sum.w += EvaluateBRDF_SHEEN(4u, n, -l, v).x * PI / LoadPs(4u);
            }
            else if (testMode == 104u)
            {
                sum.x += EvaluateBRDF_Lambertian(m, n, n, -l, v, 1.0f, 1.5f, 1.0f).x * PI;
            }
            else
            {
                uint strategy;
                l = SampleBRDF(p, m, v, n, n, 1.0f, rough, (half)0.0f, seed,
                               (half)1.0f, (half)1.5f, false, strategy);
                if (dot(l, l) > 0.0f)
                {
                    BrdfData b = EvaluateAndPdf_COMBINED(p, m, n, n, l, v, 1.0f,
                                                       rough, (half)0.0f, (half)1.0f, (half)1.5f);
                    if (b.pdf > 0.0f) sum.x += b.val.x * l.y / b.pdf;
                }
            }
        }
        results.Store4(tid.x * 16u, asuint(sum / float(count)));
        return;
    }
    float accepted = 0.0f, energy = 0.0f, maxError = 0.0f;
    if (testMode == 0u)
    {
        [loop] for (uint i = 0u; i < 4096u; ++i)
        {
            const float3 l = SampleBRDF_WithStrategy(1u, 2u, n, n, n, 1.0f,
                (half)1.0f, (half)1.0f, seed, (half)1.0f, (half)1.5f);
            if (dot(l, l) > 0.0f)
            {
                ++accepted;
                const GGXResult value = EvalGGXAll(2u, n, n, n, l,
                    (half)1.0f, (half)1.5f, 1.0f, (half)1.0f, (half)1.0f);
                energy += value.f.x * max(0.0f, dot(n, l)) / value.pdf;
            }
        }
        accepted /= 4096.0f;
        energy /= 4096.0f;
    }
    else if (testMode >= 3u)
    {
        const float nv = testMode == 3u ? 0.8f : 0.15f;
        const float3 v = float3(sqrt(1.0f - nv * nv), nv, 0);
        const float3 kd = 0.953125f;
        const half rough = (half)(128.0f / 255.0f);
        const SamplingP p = CalculateStrategyProbabilities(1u, v, n,
            (half)1.0f, (half)1.5f, kd, rough, (half)0.0f);
        float3 sum = 0.0f;
        const uint sampleCount = 32768u;
        [loop] for (uint i = 0u; i < sampleCount; ++i)
        {
            uint strategy;
            const float3 l = SampleBRDF(p, 1u, v, n, n, kd, rough, (half)0.0f,
                seed, (half)1.0f, (half)1.5f, false, strategy);
            if (dot(l, l) > 0.0f)
            {
                float3 lobe; float lobePdf;
                const BrdfData value = EvaluateAndPdf_COMBINED_L(p, strategy,
                    1u, n, n, l, v, kd, rough, (half)0.0f,
                    (half)1.0f, (half)1.5f, false, lobe, lobePdf);
                const float light = testMode == 5u ? (l.y < 0.25f ? 8.0f : 0.0f) : 1.0f;
                if (value.pdf > 0.0f) sum.x += light * value.val.x * l.y / value.pdf;
                if (lobePdf > 0.0f) sum.y += light * lobe.x * l.y / (lobePdf * StrategyP(p, strategy));
            }
            const float y = RandomFloatSingle(seed);
            const float phi = 2.0f * PI * RandomFloatSingle(seed);
            const float xz = sqrt(1.0f - y * y);
            const float3 u = float3(xz * cos(phi), y, xz * sin(phi));
            const BrdfData value = EvaluateAndPdf_COMBINED(p, 1u, n, n, u, v,
                kd, rough, (half)0.0f, (half)1.0f, (half)1.5f);
            const float light = testMode == 5u ? (y < 0.25f ? 8.0f : 0.0f) : 1.0f;
            sum.z += light * value.val.x * y * (2.0f * PI);
        }
        results.Store4(tid.x * 16u, asuint(float4(sum / float(sampleCount), 0)));
        return;
    }
    else
    {
        const uint m = testMode == 1u ? 8u : 9u;
        const float rough = LoadPcr(m);
        const float alpha = max(EPSILON, rough * rough);
        [loop] for (uint i = 0u; i < 64u; ++i)
        {
            const float theta = float(tid.x * 64u + i) * (0.04f / 4095.0f);
            const float s = sin(theta), c = cos(theta);
            const float3 h = float3(s, c, 0);
            const float3 l = reflect(-n, h);
            const CoatResult value = EvalCoatAll(m, n, n, l, (half)1.0f, (half)1.5f);
            const float den = s * s + alpha * alpha * c * c;
            const float expectedPdf = alpha * alpha / (4.0f * PI * den * den);
            const float error = abs(value.pdf - expectedPdf) / expectedPdf;
            maxError = max(maxError, isfinite(error) ? error : 1e10f);
        }
    }
    results.Store4(tid.x * 16u, asuint(float4(accepted, energy, maxError, 0)));
}

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
    SamplingP p = CalculateStrategyProbabilities(m, v, n, (half)1.0f, (half)LoadNi(m), kd, rough, metal);
    if (testMode == 34u) DropBroadLobes(p, rough);
    BrdfData full;
    float3 share = 0.0f;
    if (testMode == 35u)
    {
        float sharePdf;
        full = EvaluateAndPdf_COMBINED_L(p, LOBE_BROAD, m, n, n, l, v, kd, rough, metal,
            (half)1.0f, (half)LoadNi(m), false, share, sharePdf);
    }
    else
    {
        full = EvaluateAndPdf_COMBINED(p, m, n, n, l, v, kd, rough, metal, (half)1.0f, (half)LoadNi(m));
        if (testMode == 36u)
            share = EvaluateLobePdf_COMBINED(p, LOBE_BROAD, m, n, n, l, v, kd, rough, metal,
                (half)1.0f, (half)LoadNi(m)).val;
    }
    results.Store(tid.x * 4u, asuint(dot(full.val + share, kd) + full.pdf));
}
