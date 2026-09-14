#define SPMIS_GRID_NONCOHERENT
#include "Includes_v8.hlsli"
#include "Raygen_Common_v8.hlsli"
#include "Hybrid_Replay_v8.hlsli"

[shader("raygeneration")]
void Pass_shift_v8()
{
    const uint2  launchIndex = DispatchRaysIndex().xy;
    const float2 dims        = float2(IMG_W, IMG_H);
    const uint   pixelIdx    = MapPixelID(dims, launchIndex);
    const uint   d           = DispatchRaysIndex().z;

    const uint resRaw = g_pathStateBuffer.Load(SPM_slotZ(pixelIdx, d));
    if (resRaw == SP_UNDEF) return;

    const uint  resPx      = resRaw & SPM_RESPX_MASK;
    const bool  resBufLast = (resRaw & SPM_BUF_LAST_BIT) != 0u;
    const bool  killPh     = (resRaw & SPM_KILLPH_BIT) != 0u;

    const bool  useHint    = false;

    const uint startRaw     = g_pathStateBuffer.Load(SPM_slotS(pixelIdx, d));
    const uint startPx      = startRaw & 0x7FFFFFFFu;
    const bool startBufLast = (startRaw & SPM_BUF_LAST_BIT) != 0u;

    const uint rcInfo = resBufLast ? load_rcInfo(g_Reservoirs_last, resPx)
                                   : load_rcInfo(g_Reservoirs_current, resPx);
    const bool needWalk = RcReplayLen(rcInfo) > 0u;

    ReplayResult rr;
    uint pathSeed = 0u;
    if (needWalk)
    {
        pathSeed = resBufLast ? load_seed(g_Reservoirs_last, resPx) : load_seed(g_Reservoirs_current, resPx);
        const uint pinMat = resBufLast ? load_matID_res(g_Reservoirs_last, resPx)
                                       : load_matID_res(g_Reservoirs_current, resPx);
        HitContext ctx;
        float3     camToX1;
        if (startBufLast)
            Replay_PrimaryCtx(g_sample_last, startPx, ctx, camToX1);
        else
            Replay_PrimaryCtx(g_sample_current, startPx, ctx, camToX1);
        rr = ReplayWalk(ctx, camToX1, pathSeed, rcInfo, pinMat);
        Replay_AnchorLoads();
    }
    else
    {

        const float3 camPos = InitOrigin();
        SurfaceVertex sv;
        if (startBufLast) sv = BuildVertex(g_sample_last,    startPx, load_x1(g_sample_last,    startPx), camPos);
        else               sv = BuildVertex(g_sample_current, startPx, load_x1(g_sample_current, startPx), camPos);
        rr = DirectShiftResult(sv);
    }

    const float gBaseHint = useHint
        ? (resBufLast ? load_gBase(g_Reservoirs_last, resPx) : load_gBase(g_Reservoirs_current, resPx))
        : -1.0f;

    Reservoir r = (Reservoir)0;
    if (!RcEnvReplay(rcInfo) && rr.ok && !killPh)
    {
        if (resBufLast) loadReservoirPayload(g_Reservoirs_last,    resPx, r);
        else            loadReservoirPayload(g_Reservoirs_current, resPx, r);
    }

    float Jn, cachedNew;
    bool  preVisDead;
    float3 envDir; float envMisPdf;
    const float3 c = HybridShiftEval_post_r(rr, r, pathSeed, rcInfo, true,
                                            gBaseHint, spmis_jacThreshold, killPh,
                                            Jn, cachedNew, preVisDead, envDir, envMisPdf);

    uint2 jWord;
    if (RcEnvReplay(rcInfo))
        jWord = uint2(PackNormal(envDir), asuint(envMisPdf));
    else
        jWord = uint2(asuint(cachedNew), asuint(preVisDead ? -Jn : Jn));
    g_pathStateBuffer.Store2(SPM_slotJ(pixelIdx, d), jWord);
    g_pathStateBuffer.Store3(SPM_slotC(pixelIdx, d), asuint(c));
}
