#pragma once
#include "Sharc_v8.hlsli"


bool SharcScatterHasSpread(uint strategy, uint matID, half roughness)
{
    if (strategy == 0u || strategy == 3u) return true;
    if (strategy == 1u) return roughness >= SMOOTH_SPECULAR_THRESHOLD;
    if (strategy == 2u) return LoadPcr(matID) >= SMOOTH_SPECULAR_THRESHOLD;
    return false;
}

// The cone of a picked lobe for the cache test (SharcConeRamp): the solid angle the lobe actually
// covers, one definition for every lobe, 1 / integral(p^2) with p its density over the directions
// it can reach, normalized to one (the inverse of the density a direction drawn from it expects).
// The kind of a lobe and its roughness enter only through that size. The cosine lobe covers
// 1.5 pi (SHARC_DIFFUSE_CONE); a GGX lobe (the coat's too) of roughness 0.7 about as much, 0.8
// 1.2 times as much and 1 the whole hemisphere, from any view. A glossy lobe narrows towards
// grazing, about with the cosine of the view down to a floor the horizon sets; a rough one hardly
// does. The GGX size is a fit to the numerically integrated VNDF reflection lobe, within 13% rms
// over roughness 0.1 to 1 and all views: 12 pi alpha^2 at a glossy lobe, 2 pi at roughness 1. The
// sheen fit is within 23%. Taken from the lobe itself, never from the density of the one direction
// drawn from it. Returned as the full angle of the circular cone with that solid angle
// (pi/4 angle^2), so the cones of successive lobes add up along the path.
float SharcLobeConeAngle(uint strategy, uint matID, half roughness, float cosView)
{
    if (strategy == 0u) return SHARC_DIFFUSE_CONE;
    float solidAngle;
    if (strategy == 3u)
        solidAngle = 3.1f + 5.6f * cosView * (1.0f - cosView);
    else
    {
        const float r = strategy == 2u ? LoadPcr(matID) : (float)roughness;
        const float a = r * r;
        const float b = 1.0f - a * a;
        const float horizon = 2.05f * a / (1.0f + 5.9f * a);
        const float s = 6.0f * a * a * sqrt(cosView * cosView + horizon * horizon) / max(b * b * b, 1e-6f);
        solidAngle = 2.0f * PI * s / (1.0f + s);
    }
    return 2.0f * sqrt(solidAngle * INV_PI);
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
