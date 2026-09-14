#ifndef TEMPORAL_MERGE_V8_HLSLI
#define TEMPORAL_MERGE_V8_HLSLI

#define TM_CAND_OK      0u
#define TM_CAND_DEAD    1u
#define TM_CAND_GEOMREJ 2u

// Validate a reprojected reservoir before temporal reuse.
inline uint TemporalResolveCand(
    int2 cand, float2 dims_f,
    float3 myPos, float3 myN1s, float3 cameraPos,
    out uint tempPixelIdx, out Reservoir rdi_r, out float3 rPos)
{
    tempPixelIdx = 0xFFFFFFFFu;
    rdi_r = (Reservoir)0;
    rPos  = (float3)0.0f;

    uint tpx = 0xFFFFFFFFu;
    if (!TestTemporalCandidate(cand, dims_f, g_sample_last, tpx))
        return TM_CAND_DEAD;

    Reservoir rr = loadReservoir(g_Reservoirs_last, tpx);
    if (!IsValidReservoir(rr))
        return TM_CAND_DEAD;

    if (HYBRID_SHIFT_ON)
    {
        if (!RcReusable(rr.rcInfo))
            return TM_CAND_DEAD;
    }
    else
    {
        if (RcReplayLen(rr.rcInfo) > 0u)
            return TM_CAND_DEAD;
        if (!IsSentinelMatID(rr.matID) && !IsVolumeVertex(rr.matID) &&
            rr.Pr < rs_reconnectRoughnessMin)
            return TM_CAND_DEAD;
    }

    rPos = load_x1(g_sample_last, tpx);
    {
        const float3 rN = load_n1_s(g_sample_last, tpx);
        if (temp_normalSimCos > -1.0f && dot(myN1s, rN) <= temp_normalSimCos)
            return TM_CAND_GEOMREJ;
        const float planeThresh = temp_planeDist * length(myPos - cameraPos);
        if (abs(dot(rPos - myPos, myN1s)) > planeThresh)
            return TM_CAND_GEOMREJ;
    }

    tempPixelIdx = tpx;
    rdi_r = rr;
    return TM_CAND_OK;
}

// Merge current and reprojected reservoirs after geometric checks.
void TemporalMergeBody(uint pixelIdx, uint2 launchIndex, int2 candCoord, bool dualIn)
{
    const float2 dims_f = float2(IMG_W, IMG_H);

    const uint   myInstID = load_instID(g_sample_current, pixelIdx);
    const float3 myPos    = load_x1_with_instID(g_sample_current, pixelIdx, myInstID);
    const float3 myN1s    = load_n1_s_with_instID(g_sample_current, pixelIdx, myInstID);
    const float3 cameraPos = InitOrigin();

    int2      cand = candCoord;
    bool      dual = dualIn;
    uint      tempPixelIdx;
    Reservoir rdi_r;
    float3    rPos;
    uint st = TemporalResolveCand(cand, dims_f, myPos, myN1s, cameraPos,
                                  tempPixelIdx, rdi_r, rPos);
    if (st == TM_CAND_GEOMREJ && DUAL_MV_ON && !dual)
    {

        const float2 occNow = GetCurrentFramePixelCoordinates_World(rPos, view, projection, dims_f);
        if (occNow.x > -1e8f)
        {
            const int2 dc = int2(round(float2(launchIndex) - (occNow - float2(cand))));
            if (dc.x >= 0 && dc.y >= 0 && dc.x < (int)IMG_W && dc.y < (int)IMG_H &&
                any(dc != cand))
            {
                st = TemporalResolveCand(dc, dims_f, myPos, myN1s, cameraPos,
                                         tempPixelIdx, rdi_r, rPos);
                if (st == TM_CAND_OK) { cand = dc; dual = true; }
            }
        }
    }
    if (st != TM_CAND_OK)
        return;

    const Reservoir rdi = loadReservoir(g_Reservoirs_current, pixelIdx);
    const bool revValid = IsValidReservoir(rdi) && rdi.W > 0.0f &&
                          (!HYBRID_SHIFT_ON || RcReusable(rdi.rcInfo));

    g_pathStateBuffer.Store(SPM_slotS(pixelIdx, 0u), pixelIdx);
    g_pathStateBuffer.Store(SPM_slotZ(pixelIdx, 0u), tempPixelIdx | SPM_BUF_LAST_BIT);

    g_pathStateBuffer.Store(SPM_slotS(pixelIdx, 1u), tempPixelIdx | SPM_BUF_LAST_BIT);
    g_pathStateBuffer.Store(SPM_slotZ(pixelIdx, 1u), revValid ? pixelIdx : SP_UNDEF);

    g_pathStateBuffer.Store(SPM_w1(pixelIdx),
                            (uint(cand.x) & 0xFFFFu) | ((uint(cand.y) & 0x7FFFu) << 16) |
                            (dual ? 0x80000000u : 0u));
    g_pathStateBuffer.Store(SPM_w0(pixelIdx), TEMP_STATUS_OK);
}

#endif
