#pragma once
#include "Sharc_v8.hlsli"


bool SharcScatterHasSpread(uint strategy, uint matID, half roughness)
{
    if (strategy == 0u || strategy == 3u) return true;
    if (strategy == 1u) return roughness >= SMOOTH_SPECULAR_THRESHOLD;
    if (strategy == 2u) return LoadPcr(matID) >= SMOOTH_SPECULAR_THRESHOLD;
    return false;
}

// The solid angle of a GGX refraction lobe into or out of solid glass (SharcLobeConeAngle), eta the
// IOR ratio etai / etat across the surface. Snell turns a microfacet's tilt into (1 - eta) of it in
// deviation where a reflection turns it into twice it, so head-on the lobe is 3% of the reflection
// lobe entering glass and 6% leaving it, more toward grazing views. Smooth glass maps the normals'
// lobe (3 pi alpha^2) through the Jacobian of Snell's law at the macro normal; rough glass
// saturates at a size set by |1 - eta| alone (a lobe and its reverse are the same size) that halves
// toward grazing. A fit to the numerically integrated VNDF refraction lobe, given that the sampler
// refracted, over roughness 0.06 to 1, all views and eta 1/2.4 to 2.4: within 16% rms and never
// over by more than 1.63x; at grazing views it comes out up to 1.9x small, the side on which the
// cache answers later. Leaving glass past the critical angle, where only steep microfacets refract,
// it is mostly small too and at most 2x large. The reflection lobe used for refractions before was
// 27x too large rms, up to 484x, and let the cache answer right behind frosted glass.
float SharcRefractionSolidAngle(half roughness, float cosView, float eta)
{
    const float a = max((float)roughness * (float)roughness, 1e-3f);
    const float cosT = sqrt(max(1.0f - eta * eta * (1.0f - cosView * cosView), 0.34f * 0.34f));
    const float jacobian = (eta * cosView - cosT) * (eta * cosView - cosT) / cosT;
    const float smooth = jacobian * 3.0f * PI * a * a;
    const float rough = max(5.6f * pow(1.0f - min(eta, rcp(eta)), 1.7f), 1e-6f) *
        (0.5f + 0.5f * pow(cosView, 5.0f));
    return smooth * pow(1.0f + pow(smooth / rough, 1.5f), -1.0f / 1.5f);
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
// sheen fit is within 23%. A refraction into or out of solid glass (refractEta, the IOR ratio
// across the surface, 0 otherwise) has a lobe of its own (SharcRefractionSolidAngle); thin glass
// transmits the mirrored reflection lobe. Taken from the lobe itself, never from the density of
// the one direction drawn from it. Returned as the full angle of the circular cone with that solid
// angle (pi/4 angle^2), so the cones of successive lobes add up along the path.
float SharcLobeConeAngle(uint strategy, uint matID, half roughness, float cosView, float refractEta = 0.0f)
{
    if (strategy == 0u) return SHARC_DIFFUSE_CONE;
    float solidAngle;
    if (strategy == 3u)
        solidAngle = 3.1f + 5.6f * cosView * (1.0f - cosView);
    else if (refractEta > 0.0f)
        solidAngle = SharcRefractionSolidAngle(roughness, cosView, refractEta);
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
