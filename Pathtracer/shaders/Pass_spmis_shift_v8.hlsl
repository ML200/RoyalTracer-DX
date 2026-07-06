#define SPMIS_GRID_NONCOHERENT   // read-only grid/scratch -> L1-cached (see Includes_v8.hlsli)
#include "Includes_v8.hlsli"

//====================================
//SPMIS SPATIAL REUSE - SHIFT  (pass 2/4: k==2 reconnections + visibility rays) — RAYGEN
//====================================
//Ray-heavy stage of the split reuse. A RAYGENERATION shader, not compute: measured ~1.4ms
//vs ~2.3ms for the equivalent compute dispatch - DispatchRays scheduling keeps far more
//rays in flight here. (An SER thread reorder was tried and removed: the state-movement cost
//outweighed any coherence gain for these short <=5-ray threads.) Visibility is the inline
//RayQuery IsVisible. Consumes select's candidate indices -> canonical mis_c + per-draw
//{w_draw, J8, F_shifted}; the WRS combine + finalize stay in the compute merge pass.
//
//HYBRID SHIFT: this pass keeps ZERO replay code (no TraceRay). Draws whose source pin
//sits at k>2, and the canonical reverse-shift when the CENTRE pin sits at k>2, are
//deferred: the draw slot gets SPM_Z_PENDING_BIT, the canonical parks partnerPx in
//SPM_w1 with bit31, and the pixel is appended ONCE to the spatial replay queue
//(g_raygenQueue counter at byte 8). Pass_spmis_replay resolves them before merge.
//
//Note: sv_me (the centre vertex) is built up-front with rdi/myPos so its G-buffer loads
//issue early and their latency overlaps the canonical work. Math mirrors the temporal
//merge body's PSS algebra (weight = lum(c*vis)*Jn/cachedJac_src*W; band on Jn/gBase).
//Layout: see HashGridHash_v8.hlsli. Export name MUST equal the filename base.

#define SAMPLE_PT_ROUGHNESS_MIN rs_reconnectRoughnessMin   // legacy reconnection-vertex (x2) roughness reject (hybrid OFF)

[shader("raygeneration")]
void Pass_spmis_shift_v8()
{
    if (!SPMIS_SPATIAL_MODE) return;

    const uint2  launchIndex = DispatchRaysIndex().xy;
    const float2 dims        = float2(IMG_W, IMG_H);
    const uint   pixelIdx    = MapPixelID(dims, launchIndex);

    const uint w0     = g_pathStateBuffer.Load(SPM_w0(pixelIdx));
    const uint status = SPM_hdrStatus(w0);
    if (status == SPM_STATUS_SKIP) return;   // emitter; merge writes the sentinel

    const float3 camPos = InitOrigin();
    Reservoir    rdi    = loadReservoir(g_Reservoirs_current, pixelIdx);
    const float3 myPos  = load_x1(g_sample_current, pixelIdx);
    const SurfaceVertex sv_me = BuildVertex(g_sample_current, pixelIdx, myPos, camPos);

    //==== passthrough: one deferred visibility ray, cached for merge ====
    //direct k==2 ONLY: a replay canonical's reconnection segment starts at its
    //(never re-derived here) replayed prefix vertex — x1 -> x_k is a
    //non-segment through geometry. Its F was generated fully shadowed, so
    //passVis stays 1.
    if (status == SPM_STATUS_PASS)
    {
        const float  W    = (rdi.W > 0.0f) ? rdi.W : 0.0f;
        const float3 outC = rdi.F * W;
        float passVis = 1.0f;
        if (GetPHat(outC) > 0.0f && !IsVolumeVertex(rdi.matID) &&
            RcK(rdi.rcInfo) == 2u && !RcEnvReplay(rdi.rcInfo))
            //deferred passthrough stores a single float -> achromatic transmittance here (the
            //per-channel thin-glass tint is carried by the inline RIS-target visibility).
            passVis = Luma(ReconnectVis(sv_me.x, sv_me.n_s, rdi.matID, rdi.x2, rdi.n2_s));
        g_pathStateBuffer.Store(SPM_w1(pixelIdx), asuint(passVis));
        return;
    }

    //==== normal path ====
    const uint  reuseCell  = SPM_hdrCell(w0);
    const uint  Ntn        = min(max(spmis_reuseN, 1u), SPMIS_SPLIT_MAXDRAWS);
    const float centerConf = (float)rdi.M;                 // UNCAPPED
    const float visReuse_c = (rdi.W > 0.0f) ? 1.0f : 0.0f;
    const float p_c        = GetPHat(rdi.F) * visReuse_c;

    const uint4 agg       = g_spmisBuffer.Load4(SP_AGG(reuseCell));   // PIXCNT|NZ|CONF|OFF
    const float pixCount  = (float)agg.x;   // PIXCNT
    const float scaling   = (SPMIS_CONF_ADJUST && pixCount > 0.0f) ? min(1.0f, (float)Ntn / pixCount) : 1.0f;
    const float neighbors_conf_sum = (float)agg.z * scaling;   // CONF

    bool queueCanonical = false;
    bool queueDraws     = false;

    //==== CANONICAL MIS WEIGHT (non-defensive, Nc=1 uniform pick over the cell) ====
    //Uses rdi (the centre sample) reconnected at the partner pixel.
    const uint partnerPx = g_pathStateBuffer.Load(SPM_w1(pixelIdx));
    float mis_c = 1.0f;
    const bool canonReusable = !HYBRID_SHIFT_ON || RcReusable(rdi.rcInfo);
    const bool canonEligible = neighbors_conf_sum > EPSILON && pixCount > 0.0f &&
                               partnerPx != SP_UNDEF && p_c > EPSILON && canonReusable;
    if (canonEligible && HYBRID_SHIFT_ON && RcReplayLen(rdi.rcInfo) > 0u)
    {
        //centre pin needs a replay at the partner -> defer (bit31 marks the
        //w1 payload as a parked partner index; mis_c >= 0 can never set bit31).
        g_pathStateBuffer.Store(SPM_w1(pixelIdx), partnerPx | 0x80000000u);
        queueCanonical = true;
    }
    else
    {
        if (canonEligible)
        {
            const float  partnerConf = (float)load_M(g_Reservoirs_current, partnerPx) * scaling;
            const float3 pPos = load_x1(g_sample_current, partnerPx);

            float revT = 0.0f;
            {
                const SurfaceVertex sv_p = BuildVertex(g_sample_current, partnerPx, pPos, camPos);
                float Jn_rev = 0.0f, cachedNew_rev = 0.0f;
                float3 cr = ReconnectPSS_sv(sv_p, rdi, Jn_rev, cachedNew_rev);
                float ph = GetPHat(cr);
                if (!HYBRID_SHIFT_ON &&
                    rdi.matID != MATID_LIGHT_TRI && rdi.matID != MATID_ENV_MISS &&
                    !IsVolumeVertex(rdi.matID) && rdi.Pr < SAMPLE_PT_ROUGHNESS_MIN)
                    ph = 0.0f;
                //A banded geometric ratio zeroes the shifted density regardless of
                //visibility, so fold it in BEFORE the shadow ray and skip the trace
                //when rejected. Identical result, one fewer ray on reject.
                const bool geomOk = !RcGeomReject(Jn_rev, rdi.gBase, spmis_jacThreshold);
                if (ph > 0.0f && geomOk && !IsVolumeVertex(rdi.matID))
                {
                    float3 visT;
                    if (rdi.matID == MATID_ENV_MISS)
                    { const float3 md = normalize(rdi.x2);
                      visT = VisibilityTransmittance(sv_p.x, sv_p.n_s, sv_p.x + md * RAY_TMAX_PLANET, -md); }
                    else
                      visT = VisibilityTransmittance(sv_p.x, sv_p.n_s, rdi.x2, rdi.n2_s);
                    ph = GetPHat(cr * visT);
                }
                if (geomOk && rdi.cachedJac > 0.0f)
                    revT = ph * Jn_rev / rdi.cachedJac;
            }
            mis_c = SpmisCanonicalMis(revT, p_c, centerConf, neighbors_conf_sum, partnerConf, pixCount);
        }
        g_pathStateBuffer.Store(SPM_w1(pixelIdx), asuint(mis_c));   // overwrite partnerPx
    }

    //==== NON-CANONICAL: reconnect + shadow ray per selected draw (k==2 only) ====
    [loop]
    for (uint d = 0u; d < Ntn; ++d)
    {
        const uint zPx = g_pathStateBuffer.Load(SPM_slotZ(pixelIdx, d));
        if (zPx == SP_UNDEF) continue;
        const float selProb = asfloat(g_pathStateBuffer.Load(SPM_slotP(pixelIdx, d)));

        Reservoir pr = loadReservoir(g_Reservoirs_current, zPx);

        //dead-reservoir gate + mode-specific reusability gates. NO roughness or
        //distance gating under the hybrid shift — full reuse across all
        //roughnesses; the shifted BSDF magnitude self-gates and the geometric
        //band below bounds the singularities.
        bool dead = (pr.W <= 0.0f);
        if (!dead)
        {
            if (HYBRID_SHIFT_ON)
            {
                if (!RcReusable(pr.rcInfo))
                    dead = true;
                else if (RcReplayLen(pr.rcInfo) > 0u)
                {
                    //prefix/env replay needed: defer to the replay pass
                    //(keeps selProb in slotP)
                    g_pathStateBuffer.Store(SPM_slotZ(pixelIdx, d), zPx | SPM_Z_PENDING_BIT);
                    queueDraws = true;
                    continue;
                }
            }
            else if (RcReplayLen(pr.rcInfo) > 0u)
            {
                dead = true;   //stale replay candidate after a hybrid-off toggle
            }
            else if (pr.matID != MATID_LIGHT_TRI && pr.matID != MATID_ENV_MISS &&
                     !IsVolumeVertex(pr.matID) && pr.Pr < SAMPLE_PT_ROUGHNESS_MIN)
            {
                dead = true;   //legacy roughness gate
            }
        }
        if (dead)
        {
            g_pathStateBuffer.Store(SPM_slotZ(pixelIdx, d), SP_UNDEF);
            continue;
        }

        float Jn = 0.0f, cachedNew = 0.0f;
        float3 c = ReconnectPSS_sv(sv_me, pr, Jn, cachedNew);
        if (RcGeomReject(Jn, pr.gBase, spmis_jacThreshold) || GetPHat(c) <= 0.0f || cachedNew <= 0.0f)
        {
            g_pathStateBuffer.Store(SPM_slotZ(pixelIdx, d), SP_UNDEF);
            continue;
        }

        //Shadowed resample: trace the reconnection shadow ray (inline RayQuery) so the
        //target + contribution carry visibility (occluded neighbours get target 0).
        float3 vis = 1.0.xxx;
        if (!IsVolumeVertex(pr.matID))   //volume vertex: in-medium, skip
        {
            if (pr.matID == MATID_ENV_MISS)
            { const float3 md = normalize(pr.x2);
              vis = VisibilityTransmittance(sv_me.x, sv_me.n_s, sv_me.x + md * RAY_TMAX_PLANET, -md); }
            else
              vis = VisibilityTransmittance(sv_me.x, sv_me.n_s, pr.x2, pr.n2_s);
        }
        c *= vis;

        const float c_i_scaled = (float)pr.M * scaling;       // UNCAPPED * SS4.3 scaling
        const float w_draw = SpmisDrawWeight(GetPHat(pr.F), pr.cachedJac,
                                             GetPHat(c), Jn, selProb, Ntn,
                                             neighbors_conf_sum, centerConf, c_i_scaled) * pr.W;

        //re-anchored shifted contribution (what the merge accepts as the new F)
        const float3 Fshift = c * Jn / cachedNew;

        g_pathStateBuffer.Store (SPM_slotP(pixelIdx, d), asuint(w_draw));
        g_pathStateBuffer.Store2(SPM_slotJ(pixelIdx, d), uint2(asuint(cachedNew), asuint(Jn)));
        g_pathStateBuffer.Store3(SPM_slotC(pixelIdx, d), asuint(Fshift));
    }

    //==== ONE queue entry per pixel when anything was deferred ====
    if (queueCanonical || queueDraws)
    {
        uint slot;
        g_raygenQueue.InterlockedAdd(8u, 1u, slot);
        g_raygenQueue.Store(16u + slot * 4u,
                            (launchIndex.x & 0xFFFFu) | (launchIndex.y << 16) |
                            (queueCanonical ? 0x80000000u : 0u));
    }
}
