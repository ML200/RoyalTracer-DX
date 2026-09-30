#pragma once
#include "Sharc_v8.hlsli"


bool SharcScatterHasSpread(uint strategy, uint matID, half roughness)
{
    if (strategy == 0u || strategy == 3u) return true;
    if (strategy == 1u) return roughness >= SMOOTH_SPECULAR_THRESHOLD;
    if (strategy == 2u) return LoadPcr(matID) >= SMOOTH_SPECULAR_THRESHOLD;
    return false;
}

// GGX refraction lobe solid angle, eta = etai / etat; fit to the VNDF lobe.
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

// Full cone angle of the lobe's solid angle 1 / int p^2 (fits); refractEta 0 = none.
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

bool SharcMaterialEligible(HitContext ctx, SamplingP sp, float3 geometricNormal)
{
    // Water's highlight model differs from its continuation response.
    if (LoadIsOceanMaterial(ctx.matID)) return false;
    if (ctx.mediumMatID != MEDIUM_INVALID || LoadIsSSS(ctx.matID)) return false;
    if (LoadKd_w(ctx.matID) < 1.0f - EPSILON) return false;
    if (!HasBroadShare(sp, ctx.hitLocalPr, ctx.hitLocalPm)) return false;
    // Loose: only rejects badly bent normals, not normal maps.
    return dot(ctx.hitNormal, geometricNormal) > 0.5f;
}

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
