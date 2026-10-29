#pragma once
// In-place light samples of path vertices: run deferred by Pass_pt_nee_v8 from a queue the trace pass
// fills, or inline in the water callable (PtNeeWater, PtCallables_v8.hlsl). No ray-tracing shader
// intrinsics here, so the compute pass can include it.
#include "PtDefer_v8.hlsli"
#include "OceanVolume.hlsli"

// Light sample request of an in-place vertex, 44 bytes; the evaluation re-derives the strategy
// probabilities, seed and blue-noise index. Exact f16 copies of the HitContext material values.
struct PtNeeIO {
    float3 pos;
    uint   nPk;     // shading normal (PackNormal)
    uint   dirPk;   // incoming ray direction (PackNormal)
    uint   matID;   // | PT_NEE_REGULARIZE
    uint   kdRG;    // in: Kd.rg; out (PtNeeWater): the light sample, PvPackRadiance
    uint   kdBPr;   // in: Kd.b, Pr; out (PtNeeWater): its reuse-suffix share, PvPackRadiance
    uint   pmIorI;  // Pm, eta_i
    uint   iorTBits; // eta_t | PT_NEE_* bits << 16
    float  groupP;  // probability of the picked lobe group
};
#define PT_NEE_DEPTH_MASK  0x3Fu        // depth (bounces are clamped to 32)
#define PT_NEE_SAMPLE_SHIFT 6u          // 3 bits
#define PT_NEE_GROUP_SHIFT  9u          // 2 bits
#define PT_NEE_BLUE         (1u << 11u)
#define PT_NEE_LITE_BROAD   (1u << 12u) // the reuse suffix takes the broad lobe only
#define PT_NEE_LITE_PATH    (1u << 13u)
#define PT_NEE_IN_WATER     (1u << 14u)
#define PT_NEE_WATER_DIRECT (1u << 15u)
#define PT_NEE_REGULARIZE   (1u << 31u) // in matID: g_regularizeRoughness is on (path spread)

// Light-tree and sun sample of an in-place vertex, on the picked lobe group.
void PtInlineNee(HitContext ctx, SamplingP spPath, uint group, float groupP, float3 rayDir, uint pathSeed,
    uint depth, bool blue, uint2 pixel, uint blueIndex, bool liteBroadOnly, bool litePath, bool inWater,
    bool waterDirect, out float3 direct, out float3 liteDirect)
{
    direct = 0.0f;
    liteDirect = 0.0f;
    const bool useLearnedLights = LTC_UseSurfaceLearning();
    uint sNee = RcBounceSeed(pathSeed, depth, RC_STREAM_NEE);
    // The sun's blue-noise pair ahead of the loop, so pixel, depth and index do not live through it.
    float2 rSunBlue = 0.0f;
    if (blue) rSunBlue = PtBlue2(pixel, blueIndex, depth, BN_PAIR_SUN);
    [loop]
    for (uint tech = 0u; tech < 2u; ++tech)
    {
        // Widen the water's picked GGX lobe for the sun only.
        const bool sunTech = tech == 1u;
        const half neePr = (waterDirect && sunTech && group == LOBE_GROUP_SPEC)
            ? (half)OceanHighlightRoughness(ctx.hitLocalPr, LoadOceanSunLobeRoughness())
            : ctx.hitLocalPr;

        float3 L = 0.0f, visTarget = 0.0f, visTargetN = 0.0f, radiance = 0.0f;
        float  lightPdf = 0.0f, cosSurf = 0.0f;
        uint2  token = 0u;
        bool   sampled = false;
        OceanWaterSun waterSun = (OceanWaterSun)0;
        if (tech == 0u)
        {
            if ((rs_flags & RS_FLAG_NO_MESH_LIGHTS) != 0u || waterDirect) continue;
            const LT_Sample pick = LT_SampleLight(ctx.hitPos, ctx.hitNormal, sNee, useLearnedLights);
            token = pick.learningToken;
            const LT_LightSampleResult light = LT_SamplePointOnLightTree(ctx.hitPos, pick, sNee);
            const float3 toLight = light.position - ctx.hitPos;
            const float  dist    = sqrt(max(dot(toLight, toLight), 1e-20f));
            L = toLight / dist;
            cosSurf = dot(ctx.hitNormal, L);
            if (cosSurf > 1e-6f && dot(light.normal, -L) > 1e-6f && light.pdfSolidAngle > 1e-20f)
            {
                sampled    = true;
                visTarget  = light.position;
                visTargetN = light.normal;
                radiance   = light.emission;
                lightPdf   = light.pdfSolidAngle;
            }
        }
        else
        {
            float2 rSun = float2(RandomFloatSingle(sNee), RandomFloatSingle(sNee));
            if (blue) rSun = rSunBlue;
            if (inWater)
            {
                if (OceanSampleWaterSun(ctx.hitPos, rSun, waterSun))
                {
                    L = waterSun.L;
                    cosSurf = dot(ctx.hitNormal, L);
                    sampled  = cosSurf > 1e-6f;
                    radiance = waterSun.radiance;
                    lightPdf = waterSun.pdf;
                }
            }
            else
            {
                const SunSampleResult sun = SampleSun(rSun, ctx.hitPos + sceneOriginWorld);
                L = sun.direction;
                cosSurf = dot(ctx.hitNormal, L);
                if (cosSurf > 1e-6f && sun.pdf > 1e-20f)
                {
                    sampled    = true;
                    visTarget  = ctx.hitPos + sun.direction * RAY_TMAX_PLANET;
                    visTargetN = -sun.direction;
                    radiance   = sun.radiance;
                    lightPdf   = sun.pdf;
                }
            }
        }

        float reward = 0.0f;
        if (sampled)
        {
            const bool waterSunTech = inWater && sunTech;
            const float3 visT = waterSunTech ? OceanWaterSunReach(ctx.hitPos, ctx.hitNormal, waterSun)
                : VisibilityTransmittance(ctx.hitPos, ctx.hitNormal, visTarget, visTargetN, false, !waterDirect);
            if (any(visT > 0.0f))
            {
                const BrdfData lobe = EvaluateLobe(spPath, group, ctx.matID, ctx.hitNormal, ctx.hitNormal, L,
                    -rayDir, ctx.hitLocalKd, neePr, ctx.hitLocalPm, ctx.iors.x, ctx.iors.y);
                if (lobe.pdf > 0.0f)
                {
                    const float3 broad = group == LOBE_GROUP_BROAD ? lobe.val : 0.0f;
                    reward = dot(radiance * cosSurf * visT * LTC_TrainShare((float)ctx.hitLocalPr, ctx.matID,
                        lobe.val, group == LOBE_GROUP_BROAD) / lightPdf,
                        float3(0.2126f, 0.7152f, 0.0722f));
                    // Water direct light and the in-water sun have no BSDF partner.
                    const float  misWeight  = waterDirect || waterSunTech ? 1.0f : lightPdf / (lightPdf + lobe.pdf);
                    const float3 lightScale = radiance * cosSurf * visT * (misWeight / (lightPdf * groupP));
                    direct += lobe.val * lightScale;
                    if (litePath) liteDirect += (liteBroadOnly ? broad : lobe.val) * lightScale;
                }
            }
        }
        // Failed picks train too, with zero reward.
        if (tech == 0u) LT_TrainSample(token, reward);
    }
}

// A request's light sample (direct) and its reuse-suffix share. water: compile-time; false drops every
// water path (in-water sun, sea-surface highlight); the queue never holds water samples.
void PtNeeEvaluate(PtNeeIO p, uint2 pixel, bool water, out float3 direct, out float3 liteDirect)
{
    const uint bits   = p.iorTBits >> 16u;
    const uint depth  = bits & PT_NEE_DEPTH_MASK;
    const uint sample = (bits >> PT_NEE_SAMPLE_SHIFT) & 7u;
    g_regularizeRoughness = (p.matID & PT_NEE_REGULARIZE) != 0u ? PT_REGULARIZE_ROUGHNESS : 0.0f;

    HitContext ctx = (HitContext)0;
    ctx.hitPos      = p.pos;
    ctx.hitNormal   = UnpackNormal(p.nPk);
    ctx.matID       = p.matID & ~PT_NEE_REGULARIZE;
    ctx.hitLocalKd  = (half3)float3(f16tof32(p.kdRG), f16tof32(p.kdRG >> 16u), f16tof32(p.kdBPr));
    ctx.hitLocalPr  = (half)f16tof32(p.kdBPr >> 16u);
    ctx.hitLocalPm  = (half)f16tof32(p.pmIorI);
    ctx.iors        = (half2)float2(f16tof32(p.pmIorI >> 16u), f16tof32(p.iorTBits));
    ctx.mediumMatID = MEDIUM_INVALID;
    const float3 rayDir = UnpackNormal(p.dirPk);
    const SamplingP spPath = CalculateStrategyProbabilities(ctx.matID, -rayDir, ctx.hitNormal, ctx.iors.x,
        ctx.iors.y, ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm);

    PtInlineNee(ctx, spPath, (bits >> PT_NEE_GROUP_SHIFT) & 3u, p.groupP, rayDir, PtPathSeed(pixel, sample),
        depth, (bits & PT_NEE_BLUE) != 0u, pixel, (uint)time * PtSampleCount() + sample,
        (bits & PT_NEE_LITE_BROAD) != 0u, (bits & PT_NEE_LITE_PATH) != 0u,
        water && (bits & PT_NEE_IN_WATER) != 0u, water && (bits & PT_NEE_WATER_DIRECT) != 0u, direct, liteDirect);
}

// Callee side of the water callable (PtNeeWater): the two radiances packed back into the request.
void PtNeeShade(inout PtNeeIO p, uint2 pixel)
{
    float3 direct, liteDirect;
    PtNeeEvaluate(p, pixel, true, direct, liteDirect);
    p.kdRG  = PvPackRadiance(direct);
    p.kdBPr = PvPackRadiance(liteDirect);
}

// Caller side: the request of a shaded vertex. nPk: PackNormal(ctx.hitNormal), already packed for the
// payload; g_regularizeRoughness is the shading vertex's.
PtNeeIO PtNeeRequest(HitContext ctx, uint nPk, float3 rayDir, uint group, float groupP, uint depth, uint sample,
    uint bits)
{
    bits |= min(depth, PT_NEE_DEPTH_MASK) | (sample << PT_NEE_SAMPLE_SHIFT) | (group << PT_NEE_GROUP_SHIFT);
    PtNeeIO p;
    p.pos      = ctx.hitPos;
    p.nPk      = nPk;
    p.dirPk    = PackNormal(rayDir);
    p.matID    = ctx.matID | (g_regularizeRoughness > 0.0f ? PT_NEE_REGULARIZE : 0u);
    p.kdRG     = f32tof16(ctx.hitLocalKd.r) | (f32tof16(ctx.hitLocalKd.g) << 16u);
    p.kdBPr    = f32tof16(ctx.hitLocalKd.b) | (f32tof16(ctx.hitLocalPr) << 16u);
    p.pmIorI   = f32tof16(ctx.hitLocalPm) | (f32tof16(ctx.iors.x) << 16u);
    p.iorTBits = f32tof16(ctx.iors.y) | (bits << 16u);
    p.groupP   = groupP;
    return p;
}

// The deferred-sample queue (PtDefer_v8.hlsli): each request is its PtNeeIO and pixel (written by the
// shading vertex) and the path's weights there (written by the trace raygen).

// One atomic per wave; PT_NEE_NO_SLOT when full. Call only from the lanes that need a slot.
uint PtNeeQueueAppend()
{
    const uint lanes = WaveActiveCountBits(true);
    const uint rank  = WavePrefixCountBits(true);
    uint first = 0u;
    if (rank == 0u) g_pathStateBuffer.InterlockedAdd(PtNeeQueueBase(), lanes, first);
    const uint slot = WaveReadLaneFirst(first) + rank;
    return slot < PtNeeQueueCapacity() ? slot : PT_NEE_NO_SLOT;
}

void PtNeeQueueStoreRequest(uint slot, PtNeeIO p, uint2 pixel)
{
    const uint a = PtNeeRequestAddress(slot);
    g_pathStateBuffer.Store4(a,       uint4(asuint(p.pos), p.nPk));
    g_pathStateBuffer.Store4(a + 16u, uint4(p.dirPk, p.matID, p.kdRG, p.kdBPr));
    g_pathStateBuffer.Store4(a + 32u, uint4(p.pmIorI, p.iorTBits, asuint(p.groupP), pixel.x | (pixel.y << 16u)));
}

// weight: the throughput the light sample arrives with; liteWeight: the reuse suffix's, 0 if none.
void PtNeeQueueStoreWeights(uint slot, float3 weight, float3 liteWeight)
{
    g_pathStateBuffer.Store4(PtNeeRequestAddress(slot) + 48u,
        uint4(asuint(weight), PackRGB9E5(max(liteWeight, 0.0f))));
}

void PtNeeQueueLoad(uint slot, out PtNeeIO p, out uint2 pixel, out float3 weight, out float3 liteWeight)
{
    const uint  a  = PtNeeRequestAddress(slot);
    const uint4 w0 = g_pathStateBuffer.Load4(a);
    const uint4 w1 = g_pathStateBuffer.Load4(a + 16u);
    const uint4 w2 = g_pathStateBuffer.Load4(a + 32u);
    const uint4 w3 = g_pathStateBuffer.Load4(a + 48u);
    p.pos      = asfloat(w0.xyz);
    p.nPk      = w0.w;
    p.dirPk    = w1.x;
    p.matID    = w1.y;
    p.kdRG     = w1.z;
    p.kdBPr    = w1.w;
    p.pmIorI   = w2.x;
    p.iorTBits = w2.y;
    p.groupP   = asfloat(w2.z);
    pixel      = uint2(w2.w & 0xFFFFu, w2.w >> 16u);
    weight     = asfloat(w3.xyz);
    liteWeight = UnpackRGB9E5(w3.w);
}
