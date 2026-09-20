#pragma once
// The light-tree receiver of a path vertex: the reflectance of the lobes its light sample
// stands for and the roughness the node importance filters. Narrow lobes bounce in place and
// take no light sample, so they carry no weight; the broad glossy layers merge into one lobe
// by averaging their roughness matrices, as the paper does for the two glossy layers of
// Standard Surface. The result is the packed form, which both the light sample of the vertex
// and the MIS weights of the emitter hits from it unpack.
uint4 LT_PackVertexReceiver(HitContext ctx, float3 view)
{
    const float3 n   = ctx.hitNormal;
    const float3 Kd  = (float3)ctx.hitLocalKd;
    const float  pr  = (float)ctx.hitLocalPr;
    const float  pm  = (float)ctx.hitLocalPm;
    const float  etai = (float)ctx.iors.x, etat = (float)ctx.iors.y;
    const float  rSheen = Sampling_Weight_SHEEN(ctx.matID, n, view);
    const float  rCoat  = Sampling_Weight_COAT(ctx.matID, n, view, etai, etat);
    const float  rGgx   = Sampling_Weight_GGX(ctx.matID, n, view, etai, etat, Kd, pm);
    const float  Fd     = FresnelDielectricTIR(view, n, etai, etat).x;
    const float  afterSheen = 1.0f - rSheen;
    const float  afterCoat  = afterSheen * (1.0f - rCoat);
    const float  pcr = LoadPcr(ctx.matID);
    const float  ws  = pr  >= SMOOTH_SPECULAR_THRESHOLD ? afterCoat * ((1.0f - pm) * Fd + pm * Luma(FresnelConductor(Kd, view, n))) : 0.0f;
    const float  wc  = (LoadPc(ctx.matID) > 0.0f && pcr >= SMOOTH_SPECULAR_THRESHOLD) ? afterSheen * rCoat : 0.0f;
    float  wg = ws + wc;
    float  wd = afterCoat * (1.0f - rGgx) * Luma(Kd) * (1.0f - pm) + rSheen;
    float ax, ay;
    ComputeAnisotropicAlphas(max(0.001f, pr * pr), LoadAniso(ctx.matID), ax, ay);
    float2 a2 = float2(ax * ax, ay * ay);
    if (wg > 0.0f)
    {
        const float ac = max(0.001f, pcr * pcr);
        a2 = (ws * a2 + wc * (ac * ac)) / wg;
    }
    if (wd + wg < 1e-4f) { wd = 1.0f; wg = 0.0f; }
    float3 T, B;
    BuildAnisotropicFrame(n, LoadAnisoRot(ctx.matID), T, B);
    return LT_PackReceiver(view, T, wd, wg, sqrt(a2.x), sqrt(a2.y));
}
