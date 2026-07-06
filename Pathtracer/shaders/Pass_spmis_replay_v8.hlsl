#define SPMIS_GRID_NONCOHERENT   // read-only grid/scratch -> L1-cached (see Includes_v8.hlsli)
#include "Includes_v8.hlsli"
#include "Raygen_Common_v8.hlsli"   // HitContext (replayed bounce-loop ctx)
#include "Hybrid_Replay_v8.hlsli"

//====================================
//SPMIS SPATIAL REUSE - REPLAY  (pass 3/4: k>2 shifts) — COMPACTED INDIRECT, ROLE THREADS
//====================================
//Resolves everything Pass_spmis_shift deferred: non-canonical draws whose source
//pin needs a prefix replay at this pixel (SPM_Z_PENDING_BIT), and the canonical
//reverse-shift when the CENTRE pin needs a replay at the partner pixel (SPM_w1
//parked as partnerPx|bit31, entry bit31 mirrors it). Weights use the exact same
//SpmisDrawWeight / SpmisCanonicalMis algebra as the inline k==2 path, so merge
//cannot tell which pass produced a slot.
//
//ROLE THREADS: dispatched (queueCount x 1+SPMIS_SPLIT_MAXDRAWS) — .x picks the
//queue entry (g_raygenQueue counter at byte 8, entries from byte 16; the
//temporal entries they overwrite were consumed by Pass_temp_replay already),
//.y the ROLE: 0 = deferred canonical, 1+d = deferred draw slot d. ONE replay
//walk per thread max — the old per-pixel loop serialized up to 1+Ntn walks in
//a single thread, so every lane in a wave idled at the join for the wave's
//worst pixel. Roles with nothing pending exit after two loads; per-role stores
//are disjoint (SPM_w1 vs slotZ/P/J/C per d) and the pass draws no RNG, so the
//split is render-identical. Host pins Height = 1+SPMIS_SPLIT_MAXDRAWS in the
//args template (WriteRaysIndirectTemplate — keep in sync).
//
//All merge-side reservoir/aggregate state loads AFTER the walk; only the three
//ident words ride the traces (see Hybrid_Replay's live-state contract).

[shader("raygeneration")]
void Pass_spmis_replay_v8()
{
    const uint  entry    = g_raygenQueue.Load(16u + DispatchRaysIndex().x * 4u);
    const uint  role     = DispatchRaysIndex().y;
    const bool  canonDef = (entry & 0x80000000u) != 0u;
    const uint2 pixel    = uint2(entry & 0xFFFFu, (entry >> 16) & 0x7FFFu);
    const float2 dims    = float2(IMG_W, IMG_H);
    const uint  pixelIdx = MapPixelID(dims, pixel);

    const uint Ntn = min(max(spmis_reuseN, 1u), SPMIS_SPLIT_MAXDRAWS);

    SetSkyObserver(InitOrigin() + sceneOriginWorld);

    //==== role 0: deferred CANONICAL — replay my sample's prefix at the partner pixel ====
    if (role == 0u)
    {
        if (!canonDef)
            return;
        const uint partnerPx = g_pathStateBuffer.Load(SPM_w1(pixelIdx)) & 0x7FFFFFFFu;

        //cell aggregate gates the walk itself (mis_c degenerates to 1 without
        //neighbours), so it loads up front; everything else waits for the walk.
        const uint  w0        = g_pathStateBuffer.Load(SPM_w0(pixelIdx));
        const uint  reuseCell = SPM_hdrCell(w0);
        const uint4 agg       = g_spmisBuffer.Load4(SP_AGG(reuseCell));   // PIXCNT|NZ|CONF|OFF
        const float pixCount  = (float)agg.x;
        const float scaling   = (SPMIS_CONF_ADJUST && pixCount > 0.0f) ? min(1.0f, (float)Ntn / pixCount) : 1.0f;
        const float neighbors_conf_sum = (float)agg.z * scaling;

        float mis_c = 1.0f;
        if (neighbors_conf_sum > EPSILON && pixCount > 0.0f && partnerPx != (SP_UNDEF & 0x7FFFFFFFu))
        {
            //walk inputs: the centre reservoir's three ident words only
            const uint mySeed   = load_seed      (g_Reservoirs_current, pixelIdx);
            const uint myRcInfo = load_rcInfo    (g_Reservoirs_current, pixelIdx);
            const uint myPinMat = load_matID_res (g_Reservoirs_current, pixelIdx);

            float Jn_rev = 0.0f, cachedNew_rev = 0.0f;
            const float3 cr = HybridShiftEval(g_sample_current, partnerPx,
                                              g_Reservoirs_current, pixelIdx,
                                              mySeed, myRcInfo, myPinMat,
                                              true, Jn_rev, cachedNew_rev);

            //merge-side state, post-walk
            Reservoir rdi = (Reservoir)0;
            loadReservoirState(g_Reservoirs_current, pixelIdx, rdi);
            const float centerConf  = (float)rdi.M;                 // UNCAPPED
            const float visReuse_c  = (rdi.W > 0.0f) ? 1.0f : 0.0f;
            const float p_c         = GetPHat(rdi.F) * visReuse_c;
            const float partnerConf = (float)load_M(g_Reservoirs_current, partnerPx) * scaling;

            float revT = 0.0f;
            if (!RcGeomReject(Jn_rev, rdi.gBase, spmis_jacThreshold) && rdi.cachedJac > 0.0f)
                revT = GetPHat(cr) * Jn_rev / rdi.cachedJac;

            mis_c = SpmisCanonicalMis(revT, p_c, centerConf, neighbors_conf_sum, partnerConf, pixCount);
        }
        g_pathStateBuffer.Store(SPM_w1(pixelIdx), asuint(mis_c));
        return;
    }

    //==== roles 1..Ntn: deferred DRAW d — replay the source's prefix at THIS pixel ====
    const uint d = role - 1u;
    if (d >= Ntn)
        return;
    const uint zRaw = g_pathStateBuffer.Load(SPM_slotZ(pixelIdx, d));
    if (zRaw == SP_UNDEF || (zRaw & SPM_Z_PENDING_BIT) == 0u)
        return;
    const uint zPx = zRaw & ~SPM_Z_PENDING_BIT;

    //walk inputs: the source reservoir's three ident words only
    const uint srcSeed   = load_seed      (g_Reservoirs_current, zPx);
    const uint srcRcInfo = load_rcInfo    (g_Reservoirs_current, zPx);
    const uint srcPinMat = load_matID_res (g_Reservoirs_current, zPx);

    float Jn = 0.0f, cachedNew = 0.0f;
    const float3 c = HybridShiftEval(g_sample_current, pixelIdx,
                                     g_Reservoirs_current, zPx,
                                     srcSeed, srcRcInfo, srcPinMat,
                                     true, Jn, cachedNew);

    //source RIS/PSS state, post-walk (gBase feeds the reject band below)
    Reservoir pr = (Reservoir)0;
    loadReservoirState(g_Reservoirs_current, zPx, pr);

    if (GetPHat(c) <= 0.0f || cachedNew <= 0.0f ||
        RcGeomReject(Jn, pr.gBase, spmis_jacThreshold))
    {
        g_pathStateBuffer.Store(SPM_slotZ(pixelIdx, d), SP_UNDEF);
        return;
    }

    //weight inputs, post-walk
    const uint  w0        = g_pathStateBuffer.Load(SPM_w0(pixelIdx));
    const uint  reuseCell = SPM_hdrCell(w0);
    const uint4 agg       = g_spmisBuffer.Load4(SP_AGG(reuseCell));   // PIXCNT|NZ|CONF|OFF
    const float pixCount  = (float)agg.x;
    const float scaling   = (SPMIS_CONF_ADJUST && pixCount > 0.0f) ? min(1.0f, (float)Ntn / pixCount) : 1.0f;
    const float neighbors_conf_sum = (float)agg.z * scaling;
    const float centerConf = (float)load_M(g_Reservoirs_current, pixelIdx);   // UNCAPPED
    const float selProb    = asfloat(g_pathStateBuffer.Load(SPM_slotP(pixelIdx, d)));

    const float c_i_scaled = (float)pr.M * scaling;
    const float w_draw = SpmisDrawWeight(GetPHat(pr.F), pr.cachedJac,
                                         GetPHat(c), Jn, selProb, Ntn,
                                         neighbors_conf_sum, centerConf, c_i_scaled) * pr.W;

    const float3 Fshift = c * Jn / cachedNew;

    g_pathStateBuffer.Store (SPM_slotZ(pixelIdx, d), zPx);   //clear pending
    g_pathStateBuffer.Store (SPM_slotP(pixelIdx, d), asuint(w_draw));
    g_pathStateBuffer.Store2(SPM_slotJ(pixelIdx, d), uint2(asuint(cachedNew), asuint(Jn)));
    g_pathStateBuffer.Store3(SPM_slotC(pixelIdx, d), asuint(Fshift));
}
