#ifndef SHARC_PATH_V8_HLSLI
#define SHARC_PATH_V8_HLSLI
#include "Sharc_v8.hlsli"

#ifndef SHARC_TEST
// Footprint growth follows the sampled distribution, not just its lobe ID.
// GGX and coat below the sampler's smooth cutoff are delta events; otherwise
// their directional PDF controls the spread, including rough transmission.
// The Charlie sheen sampler is continuous (it has no delta branch).
bool SharcScatterHasSpread(uint strategy, uint matID, half roughness)
{
    if (strategy == 0u || strategy == 3u) return true;
    if (strategy == 1u) return roughness >= SMOOTH_SPECULAR_THRESHOLD;
    if (strategy == 2u) return LoadPcr(matID) >= SMOOTH_SPECULAR_THRESHOLD;
    return false;
}

// Material eligibility shared by training deposits, the inspector and the
// rendering query. The cache holds the BROAD share: the diffuse lobe and a
// GGX lobe rough enough to be nearly view-independent (BROAD_GGX_ROUGHNESS),
// so a rough metal qualifies like a Lambertian surface. Sheen, coat and a
// glossy GGX lobe are always traced. What matters is that a broad share
// exists at all: opaque, not SSS, not inside a medium, with the shading
// normal within ~25 deg of the face normal.
bool SharcMaterialEligible(HitContext ctx, SamplingP sp, float3 geometricNormal)
{
    if (ctx.mediumMatID != MEDIUM_INVALID || LoadIsSSS(ctx.matID)) return false;
    if (LoadKd_w(ctx.matID) < 1.0f - EPSILON) return false;
    if (!HasBroadShare(sp, ctx.hitLocalPr, ctx.hitLocalPm)) return false;
    return dot(ctx.hitNormal, geometricNormal) > 0.9f;
}

// Transmission of the layers ABOVE the broad share for the OUTGOING direction
// at normal light incidence: sheen and coat, and the GGX gate only while the
// GGX lobe is not itself part of the share. Every layer's transmittance in
// this BXDF is a product of a view-only and a light-only factor (coat
// (1-pc Fi)(1-pc Fo), GGX gate (1-Fo)(1-Fi), sheen view only), so dividing a
// record by this and by Kd removes those layers' view dependence from the
// training view; a query multiplies its own factor back. A broad GGX lobe's
// own Fresnel stays inside the record (small for a rough lobe). The EPSILON
// skips match EvaluateAndPdf_COMBINED_L, which gates the share the same way.
float SharcLayerTransmission(SamplingP sp, HitContext ctx, float3 view)
{
    const float3 N = normalize(ctx.hitNormal);
    const float3 V = normalize(view);
    float gate = 1.0f;
    if (sp.Psheen >= EPSILON)
        gate *= Transmittance_SHEEN(ctx.matID, N, -N, V);
    if (sp.Pcoat >= EPSILON)
        gate *= EvalCoatAll(ctx.matID, N, V, N, ctx.iors.x, ctx.iors.y, true).t;
    if (sp.Pspec >= EPSILON && !IsBroadGGX(ctx.hitLocalPr))
        gate *= EvalGGXAll(ctx.matID, N, N, V, N, ctx.iors.x, ctx.iors.y, (float3)ctx.hitLocalKd,
            ctx.hitLocalPr, ctx.hitLocalPm, false, true).t;
    return saturate(gate);
}

SharcSurface SharcMakeSurface(HitContext ctx, float3 geometricNormal)
{
    SharcSurface s;
    s.position = ctx.hitPos;
    s.geometricNormal = geometricNormal;
    s.normal = ctx.hitNormal;
    s.demodulator = max((float3)ctx.hitLocalKd, 0.05f);
    s.instance = ctx.instID;
    s.material = ctx.matID;
    // Front/back is discrete; normals, roughness and reflectance use smooth
    // support rather than quantization boundaries in the rendered estimate.
    s.variant = ctx.backface ? 1u : 0u;
    s.roughness = ctx.hitLocalPr;
    return s;
}
#endif

#if SHARC_UPDATE_PASS
// Registered vertices per training path. With cache resampling a warm path
// ends at its second eligible vertex, so deeper registrations would rarely
// happen; two keeps the live state small.
#define SHARC_PROPAGATION_DEPTH 2u

// Live training state carried across every trace and reorder. Every loop over
// the registered vertices is unrolled with a count guard, so DXC keeps the
// arrays in registers instead of dynamically indexed local memory. With
// SHARC_TRAINING_SPILL (the update pass) the per-vertex radiance and
// demodulation weight, 48 bytes, live in the path-state buffer at the
// training pixel instead (planes no pass uses before the render pass), so
// only the addresses, splats and counters ride the reorders.
struct SharcTrainingState
{
    uint address[SHARC_PROPAGATION_DEPTH];
    float splat[SHARC_PROPAGATION_DEPTH];
#ifdef SHARC_TRAINING_SPILL
    uint spill; // byte address of this path's spilled vertices
#else
    float3 radiance[SHARC_PROPAGATION_DEPTH];
    float3 weight[SHARC_PROPAGATION_DEPTH]; // rcp(Kd x layer transmission) x BSDF/roulette weights since the vertex
#endif
    uint count;
    // Uncompensated BSDF throughput since the most recent registered vertex:
    // the roulette survival of the suffix. Reset to one at every registration.
    float suffixLuma;
    // Vertex registered at the CURRENT bounce, or SHARC_INVALID. Its record is
    // the diffuse lobe only, so this bounce's NEE and scatter reach it through
    // the diffuse lobe's value; earlier vertices see the full BSDF as usual.
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

// This bounce's NEE: the fresh vertex takes the diffuse lobe's share.
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
    // Each record receives at most one suffix observation from a training path,
    // including revisits at nonadjacent bounces.
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
        // Demodulate at registration by Kd and by the layer transmission for
        // this training view, avoiding another live float3 per vertex.
        SharcTrainingSetWeight(state, i, rcp(surface.demodulator * max(layerTransmission, 1e-3f)));
    }
    state.fresh = state.count;
    ++state.count;
    state.suffixLuma = 1.0f;
}

// Roulette survival for continuing past this vertex: the luminance of the
// uncompensated BSDF throughput accumulated since the last registered vertex,
// including the scatter about to be taken. Dark camera prefixes therefore do
// not kill training paths, and dark suffixes do not run to the depth cap.
float SharcTrainingSurvival(SharcTrainingState state, float3 scatterWeight)
{
    return clamp(state.suffixLuma * Luma(scatterWeight), 0.1f, 1.0f);
}

// This bounce's scatter: the fresh vertex takes the diffuse lobe's share of
// the same sampled direction (value of that lobe over the full strategy pdf).
void SharcTrainingAdvance(inout SharcTrainingState state, float3 full, float3 diffuseOnly, float rrWeight)
{
    state.suffixLuma *= Luma(full); // compensation stays out of the survival estimate
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
