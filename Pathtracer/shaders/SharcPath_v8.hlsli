#pragma once
#include "Sharc_v8.hlsli"


bool SharcScatterHasSpread(uint strategy, uint matID, half roughness)
{
    if (strategy == 0u || strategy == 3u) return true;
    if (strategy == 1u) return roughness >= SMOOTH_SPECULAR_THRESHOLD;
    if (strategy == 2u) return LoadPcr(matID) >= SMOOTH_SPECULAR_THRESHOLD;
    return false;
}

// The cone of a picked lobe for the cache test (SharcConeRamp): the lobe itself, never the density
// of the one direction drawn from it. Diffuse and sheen cover the hemisphere. A GGX lobe (the coat
// too) reflects half of its microfacet normals within 2 atan(alpha) of the mirror direction, a cone
// of solid angle 4 pi alpha^2 / (1 + alpha^2) that widens smoothly into the hemisphere at
// alpha = 1. Returned as the full angle of the circular cone with that solid angle
// (pi/4 angle^2), so the cones of successive lobes add up along the path.
float SharcLobeConeAngle(uint strategy, uint matID, half roughness)
{
    if (strategy == 0u || strategy == 3u) return SHARC_HEMISPHERE_CONE;
    const float r = strategy == 2u ? LoadPcr(matID) : (float)roughness;
    const float a = r * r;
    return 4.0f * a * rsqrt(1.0f + a * a);
}

// Restrict cache updates to stable, sufficiently diffuse surfaces.
bool SharcMaterialEligible(HitContext ctx, SamplingP sp, float3 geometricNormal)
{
    // Water's direct-highlight model differs from its continuation response.
    if (LoadIsOceanMaterial(ctx.matID)) return false;
    if (ctx.mediumMatID != MEDIUM_INVALID || LoadIsSSS(ctx.matID)) return false;
    if (LoadKd_w(ctx.matID) < 1.0f - EPSILON) return false;
    if (!HasBroadShare(sp, ctx.hitLocalPr, ctx.hitLocalPm)) return false;
    // Normal maps tilt the shading normal well away from the face on ordinary diffuse surfaces;
    // only a normal bent far over (grazing clamps, broken tangents) keeps a surface out.
    return dot(ctx.hitNormal, geometricNormal) > 0.5f;
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
