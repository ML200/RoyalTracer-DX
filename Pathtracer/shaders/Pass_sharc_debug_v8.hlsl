#include "Includes_v8.hlsli"
#include "PtDefer_v8.hlsli"
#include "SharcDebug_v8.hlsli"

// Cache and guide inspection: colors the primary surface by the selected cache view. Runs only
// while an inspection mode is active, so the path tracer itself carries no debug code.
[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    const uint2 imgSize = uint2(IMG_W, IMG_H);
    if (any(tid.xy >= imgSize)) return;
    const uint2 pixel    = tid.xy;
    const uint  pixelIdx = MapPixelID(imgSize, pixel);

    float4 debugColor = float4(0.015f, 0.02f, 0.03f, 0.0f);
    if ((load_flagsWord(g_sample_current, pixelIdx) & SD_FLAG_NOBOUNCE) == 0u)
    {
        const SDRecord primary = load_SD(g_sample_current, pixelIdx);
        float2 pIors; uint pMedium; float3 pAbsorb;
        load_rg_primaryExtra(pixelIdx, pIors, pMedium, pAbsorb);
        HitContext ctx;
        ctx.hitPos         = primary.x1;
        ctx.hitNormal      = primary.n1_s;
        ctx.matID          = primary.matID;
        ctx.instID         = primary.instID;
        ctx.backface       = (primary.flags & SD_FLAG_BACKFACE) != 0u;
        ctx.hitLocalKd     = (half3)primary.Kd;
        ctx.hitLocalPr     = (half)primary.Pr;
        ctx.hitLocalPm     = (half)primary.Pm;
        ctx.iors           = (half2)pIors;
        ctx.mediumMatID    = pMedium;
        ctx.absorptionTint = (half3)pAbsorb;
        const float3 geometricNormal = gScratchPing[uint3(pixel, SHARC_DEBUG_SCRATCH)].xyz;
        const SamplingP sp = CalculateStrategyProbabilities(ctx.matID, normalize(InitOrigin() - primary.x1),
            ctx.hitNormal, ctx.iors.x, ctx.iors.y, ctx.hitLocalKd, ctx.hitLocalPr, ctx.hitLocalPm);
        if (!SharcMaterialEligible(ctx, sp, geometricNormal))
        {
            debugColor = float4(0.05f, 0.07f, 0.12f, 0.0f);
        }
        else if (SHARC_DEBUG_MODE == SHARC_DEBUG_GUIDING)
        {
            debugColor = GuideDebugColor(primary.x1, geometricNormal, ctx.hitNormal,
                (sharc_enabled & SHARC_DEBUG_OTHER_LEVEL_BIT) != 0u);
        }
        else
        {
            const SharcSurface surface = SharcMakeSurface(ctx, geometricNormal);
            const float lod = SharcLevel(primary.x1);
            uint level = (uint)(lod + 0.5f);
            if ((sharc_enabled & SHARC_DEBUG_OTHER_LEVEL_BIT) != 0u)
                level = frac(lod) >= 0.5f ? level - 1u : level + 1u;
            level = min(level, SHARC_MAX_LEVEL - 1u);
            debugColor = SharcDebugColor(surface, level, SHARC_DEBUG_MODE);
        }
    }
    gScratchPing[uint3(pixel, SHARC_DEBUG_SCRATCH)] = debugColor;
}
