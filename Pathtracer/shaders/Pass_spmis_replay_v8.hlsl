#define SPMIS_GRID_NONCOHERENT   // read-only grid/scratch -> L1-cached (see Includes_v8.hlsli)
#include "Includes_v8.hlsli"
#include "Raygen_Common_v8.hlsli"   // HitContext (replayed bounce-loop ctx)
#include "Hybrid_Replay_v8.hlsli"

//====================================
//SPMIS SPATIAL REUSE - REPLAY  (pass 3/4: k>2 shifts) — COMPACTED INDIRECT
//====================================
//Resolves everything Pass_spmis_shift deferred: non-canonical draws whose source
//pin needs a prefix replay at this pixel (SPM_Z_PENDING_BIT), and the canonical
//reverse-shift when the CENTRE pin needs a replay at the partner pixel (SPM_w1
//parked as partnerPx|bit31, entry bit31 mirrors it). Weights use the exact same
//SpmisDrawWeight / SpmisCanonicalMis algebra as the inline k==2 path, so merge
//cannot tell which pass produced a slot. Dispatched 1D over the spatial replay
//queue (g_raygenQueue counter at byte 8, entries from byte 16 — the temporal
//entries they overwrite were consumed by Pass_temp_replay already).

[shader("raygeneration")]
void Pass_spmis_replay_v8()
{
    const uint  entry    = g_raygenQueue.Load(16u + DispatchRaysIndex().x * 4u);
    const bool  canonDef = (entry & 0x80000000u) != 0u;
    const uint2 pixel    = uint2(entry & 0xFFFFu, (entry >> 16) & 0x7FFFu);
    const float2 dims    = float2(IMG_W, IMG_H);
    const uint  pixelIdx = MapPixelID(dims, pixel);

    SetSkyObserver(InitOrigin() + sceneOriginWorld);

    const uint w0        = g_pathStateBuffer.Load(SPM_w0(pixelIdx));
    const uint reuseCell = SPM_hdrCell(w0);

    const float3 camPos = InitOrigin();
    Reservoir    rdi    = loadReservoir(g_Reservoirs_current, pixelIdx);

    const uint  Ntn        = min(max(spmis_reuseN, 1u), SPMIS_SPLIT_MAXDRAWS);
    const float centerConf = (float)rdi.M;                 // UNCAPPED
    const float visReuse_c = (rdi.W > 0.0f) ? 1.0f : 0.0f;
    const float p_c        = GetPHat(rdi.F) * visReuse_c;

    const uint4 agg       = g_spmisBuffer.Load4(SP_AGG(reuseCell));   // PIXCNT|NZ|CONF|OFF
    const float pixCount  = (float)agg.x;
    const float scaling   = (SPMIS_CONF_ADJUST && pixCount > 0.0f) ? min(1.0f, (float)Ntn / pixCount) : 1.0f;
    const float neighbors_conf_sum = (float)agg.z * scaling;

    //==== deferred CANONICAL: replay my sample's prefix at the partner pixel ====
    if (canonDef)
    {
        const uint partnerPx = g_pathStateBuffer.Load(SPM_w1(pixelIdx)) & 0x7FFFFFFFu;
        float mis_c = 1.0f;
        if (neighbors_conf_sum > EPSILON && pixCount > 0.0f && partnerPx != (SP_UNDEF & 0x7FFFFFFFu))
        {
            const float  partnerConf = (float)load_M(g_Reservoirs_current, partnerPx) * scaling;

            float Jn_rev = 0.0f, cachedNew_rev = 0.0f;
            const float3 cr = HybridShiftEval(g_sample_current, partnerPx, rdi,
                                              true, Jn_rev, cachedNew_rev);
            float revT = 0.0f;
            if (!RcGeomReject(Jn_rev, rdi.gBase, spmis_jacThreshold) && rdi.cachedJac > 0.0f)
                revT = GetPHat(cr) * Jn_rev / rdi.cachedJac;

            mis_c = SpmisCanonicalMis(revT, p_c, centerConf, neighbors_conf_sum, partnerConf, pixCount);
        }
        g_pathStateBuffer.Store(SPM_w1(pixelIdx), asuint(mis_c));
    }

    //==== deferred DRAWS: replay each pending source's prefix at THIS pixel ====
    [loop]
    for (uint d = 0u; d < Ntn; ++d)
    {
        const uint zRaw = g_pathStateBuffer.Load(SPM_slotZ(pixelIdx, d));
        if (zRaw == SP_UNDEF || (zRaw & SPM_Z_PENDING_BIT) == 0u) continue;
        const uint zPx = zRaw & ~SPM_Z_PENDING_BIT;
        const float selProb = asfloat(g_pathStateBuffer.Load(SPM_slotP(pixelIdx, d)));

        Reservoir pr = loadReservoir(g_Reservoirs_current, zPx);

        float Jn = 0.0f, cachedNew = 0.0f;
        const float3 c = HybridShiftEval(g_sample_current, pixelIdx, pr,
                                         true, Jn, cachedNew);
        if (GetPHat(c) <= 0.0f || cachedNew <= 0.0f ||
            RcGeomReject(Jn, pr.gBase, spmis_jacThreshold))
        {
            g_pathStateBuffer.Store(SPM_slotZ(pixelIdx, d), SP_UNDEF);
            continue;
        }

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
}
