#ifndef SHARC_PATH_V8_HLSLI
#define SHARC_PATH_V8_HLSLI
#include "Sharc_v8.hlsli"

#ifndef SHARC_TEST

bool SharcScatterHasSpread(uint strategy, uint matID, half roughness)
{
    if (strategy == 0u || strategy == 3u) return true;
    if (strategy == 1u) return roughness >= SMOOTH_SPECULAR_THRESHOLD;
    if (strategy == 2u) return LoadPcr(matID) >= SMOOTH_SPECULAR_THRESHOLD;
    return false;
}

// Restrict cache updates to stable, sufficiently diffuse surfaces.
bool SharcMaterialEligible(HitContext ctx, SamplingP sp, float3 geometricNormal)
{
    if (ctx.mediumMatID != MEDIUM_INVALID || LoadIsSSS(ctx.matID)) return false;
    if (LoadKd_w(ctx.matID) < 1.0f - EPSILON) return false;
    if (!HasBroadShare(sp, ctx.hitLocalPr, ctx.hitLocalPm)) return false;
    return dot(ctx.hitNormal, geometricNormal) > 0.9f;
}

// Estimate the diffuse layer transmission used by cache reweighting.
float SharcLayerTransmission(SamplingP sp, HitContext ctx, float3 view)
{
    const float3 N = normalize(ctx.hitNormal);
    const float3 V = normalize(view);
    float gate = 1.0f;

    if (sp.Psheen >= EPSILON)
        gate *= Transmittance_SHEEN(ctx.matID, N, -N, V);
    if (sp.Pcoat >= EPSILON)
        gate *= CoatTransmittance(ctx.matID, N, V, N, ctx.iors.x, ctx.iors.y);
    if (sp.Pspec >= EPSILON && !IsBroadGGX(ctx.hitLocalPr))
        gate *= GGXTransmittance(ctx.matID, N, V, N, ctx.iors.x, ctx.iors.y, (float3)ctx.hitLocalKd,
            ctx.hitLocalPr, ctx.hitLocalPm);
    return saturate(gate);
}
// Build the packed cache surface from the current hit context.
SharcSurface SharcMakeSurface(HitContext ctx, float3 geometricNormal)
{
    SharcSurface s;
    s.position = ctx.hitPos;
    s.geometricNormal = geometricNormal;
    s.normal = ctx.hitNormal;
    s.demodulator = max((float3)ctx.hitLocalKd, 0.05f);
    s.instance = ctx.instID;
    s.material = ctx.matID;

    s.variant = ctx.backface ? 1u : 0u;
    s.roughness = ctx.hitLocalPr;
    return s;
}
#endif

#if SHARC_UPDATE_PASS

#define SHARC_PROPAGATION_DEPTH 2u

struct SharcTrainingState
{
    uint address[SHARC_PROPAGATION_DEPTH];
    float splat[SHARC_PROPAGATION_DEPTH];
#ifdef SHARC_TRAINING_SPILL
    uint spill;
#else
    float3 radiance[SHARC_PROPAGATION_DEPTH];
    float3 weight[SHARC_PROPAGATION_DEPTH];
#endif
    uint count;

    float suffixLuma;

    uint fresh;
};

#ifdef SHARC_TRAINING_SPILL
uint SharcTrainingSpillAddress(uint pixelIndex) { return pixelIndex * 48u; }
float3 SharcTrainingRadianceOf(SharcTrainingState s, uint i)
{
    return asfloat(g_pathStateBuffer.Load3(s.spill + i * 24u));
}
void SharcTrainingSetRadiance(inout SharcTrainingState s, uint i, float3 v)
{
    g_pathStateBuffer.Store3(s.spill + i * 24u, asuint(v));
}
float3 SharcTrainingWeightOf(SharcTrainingState s, uint i)
{
    return asfloat(g_pathStateBuffer.Load3(s.spill + i * 24u + 12u));
}
void SharcTrainingSetWeight(inout SharcTrainingState s, uint i, float3 v)
{
    g_pathStateBuffer.Store3(s.spill + i * 24u + 12u, asuint(v));
}
#else
float3 SharcTrainingRadianceOf(SharcTrainingState s, uint i) { return s.radiance[i]; }
void SharcTrainingSetRadiance(inout SharcTrainingState s, uint i, float3 v) { s.radiance[i] = v; }
float3 SharcTrainingWeightOf(SharcTrainingState s, uint i) { return s.weight[i]; }
void SharcTrainingSetWeight(inout SharcTrainingState s, uint i, float3 v) { s.weight[i] = v; }
#endif

void SharcTrainingInit(out SharcTrainingState state, uint spillPixel = 0u)
{
    [unroll] for (uint i = 0u; i < SHARC_PROPAGATION_DEPTH; ++i)
    {
        state.address[i] = SHARC_INVALID;
        state.splat[i] = 0.0f;
    }
#ifdef SHARC_TRAINING_SPILL
    state.spill = SharcTrainingSpillAddress(spillPixel);
#else
    [unroll] for (uint k = 0u; k < SHARC_PROPAGATION_DEPTH; ++k)
    {
        state.radiance[k] = 0.0f;
        state.weight[k] = 0.0f;
    }
#endif
    state.count = 0u;
    state.suffixLuma = 1.0f;
    state.fresh = SHARC_INVALID;
}

void SharcTrainingRadiance(inout SharcTrainingState state, float3 value)
{
    [unroll] for (uint i = 0u; i < SHARC_PROPAGATION_DEPTH; ++i)
        if (i < state.count)
            SharcTrainingSetRadiance(state, i, SharcTrainingRadianceOf(state, i) + SharcTrainingWeightOf(state, i) * value);
}

void SharcTrainingRadianceSplit(inout SharcTrainingState state, float3 full, float3 diffuseOnly)
{
    [unroll] for (uint i = 0u; i < SHARC_PROPAGATION_DEPTH; ++i)
        if (i < state.count)
            SharcTrainingSetRadiance(state, i, SharcTrainingRadianceOf(state, i) +
                SharcTrainingWeightOf(state, i) * (i == state.fresh ? diffuseOnly : full));
}

void SharcTrainingScatter(inout SharcTrainingState state, float3 weight)
{
    [unroll] for (uint i = 0u; i < SHARC_PROPAGATION_DEPTH; ++i)
        if (i < state.count) SharcTrainingSetWeight(state, i, SharcTrainingWeightOf(state, i) * weight);
}

void SharcTrainingVertex(inout SharcTrainingState state, SharcSurface surface, float layerTransmission, uint seed)
{
    if (state.count >= SHARC_PROPAGATION_DEPTH) return;
    uint address; float splat;
    if (!SharcAllocateDeposit(surface, seed, address, splat)) return;

    bool duplicate = false;
    [unroll] for (uint j = 0u; j < SHARC_PROPAGATION_DEPTH; ++j)
        if (j < state.count && state.address[j] == address) duplicate = true;
    if (duplicate) return;
    [unroll] for (uint i = 0u; i < SHARC_PROPAGATION_DEPTH; ++i)
    {
        if (i != state.count) continue;
        state.address[i] = address;
        state.splat[i] = splat;
        SharcTrainingSetRadiance(state, i, 0.0f);

        SharcTrainingSetWeight(state, i, rcp(surface.demodulator * max(layerTransmission, 1e-3f)));
    }
    state.fresh = state.count;
    ++state.count;
    state.suffixLuma = 1.0f;
}

float SharcTrainingSurvival(SharcTrainingState state, float3 scatterWeight)
{
    return clamp(state.suffixLuma * Luma(scatterWeight), 0.1f, 1.0f);
}

// Advance training throughput after BSDF and roulette weighting.
void SharcTrainingAdvance(inout SharcTrainingState state, float3 full, float3 diffuseOnly, float rrWeight)
{
    state.suffixLuma *= Luma(full);
    [unroll] for (uint i = 0u; i < SHARC_PROPAGATION_DEPTH; ++i)
        if (i < state.count)
            SharcTrainingSetWeight(state, i, SharcTrainingWeightOf(state, i) * (i == state.fresh ? diffuseOnly : full) * rrWeight);
    state.fresh = SHARC_INVALID;
}

void SharcTrainingCommit(SharcTrainingState state)
{
    [unroll] for (uint i = 0u; i < SHARC_PROPAGATION_DEPTH; ++i)
        if (i < state.count) SharcAccumulate(state.address[i], SharcTrainingRadianceOf(state, i), state.splat[i]);
}
#endif
#endif
