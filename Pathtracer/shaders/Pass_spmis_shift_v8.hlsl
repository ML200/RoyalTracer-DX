#define SPMIS_GRID_NONCOHERENT   // read-only grid/scratch -> L1-cached (see Includes_v8.hlsli)
#include "Includes_v8.hlsli"

//====================================
//SPMIS SPATIAL REUSE - SHIFT  (pass 2/3: reconnections + visibility rays) — RAYGEN
//====================================
//Ray-heavy stage of the split reuse. A RAYGENERATION shader, not compute: measured ~1.4ms
//vs ~2.3ms for the equivalent compute dispatch - DispatchRays scheduling keeps far more
//rays in flight here. (An SER thread reorder was tried and removed: the state-movement cost
//outweighed any coherence gain for these short <=5-ray threads.) Visibility is the inline
//RayQuery IsVisible. Consumes select's candidate indices -> canonical mis_c + per-draw
//{w_draw, c}; the WRS combine + finalize stay in the compute merge pass.
//
//Note: sv_me (the centre vertex) is built up-front with rdi/myPos so its G-buffer loads
//issue early and their latency overlaps the canonical work. Sinking it to just before the
//draws to shave ~a SurfaceVertex of registers was tried and reverted - it measurably hurt
//latency hiding for no real occupancy gain. Math mirrors the monolithic Pass_spmis_reuse.
//Layout: see HashGridHash_v8.hlsli. Export name MUST equal the filename base.

#define SAMPLE_PT_ROUGHNESS_MIN rs_reconnectRoughnessMin   // editor-settable reconnection-vertex (x2) roughness reject (default 0.15)

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
    if (status == SPM_STATUS_PASS)
    {
        const float  W    = (rdi.W > 0.0f) ? rdi.W : 0.0f;
        const float3 outC = rdi.F * W;
        float passVis = 1.0f;
        if (GetPHat(outC) > 0.0f)
            passVis = ReconnectVis(sv_me.x, sv_me.n_s, rdi.matID, rdi.x2, rdi.n2_s);
        g_pathStateBuffer.Store(SPM_w1(pixelIdx), asuint(passVis));
        return;
    }

    //==== normal path ====
    const uint  reuseCell  = SPM_hdrCell(w0);
    const uint  Ntn        = min(max(spmis_reuseN, 1u), SPMIS_SPLIT_MAXDRAWS);
    const float centerConf = (float)rdi.M;                 // UNCAPPED
    const float my_Jc      = (rdi.matID == MATID_ENV_MISS) ? 1.0f : ComputeJc(myPos, rdi.x2, rdi.n2_s);
    const float visReuse_c = (rdi.W > 0.0f) ? 1.0f : 0.0f;
    const float p_c        = GetPHat(rdi.F) * visReuse_c;

    const uint4 agg       = g_spmisBuffer.Load4(SP_AGG(reuseCell));   // PIXCNT|NZ|CONF|OFF
    const float pixCount  = (float)agg.x;   // PIXCNT
    const float scaling   = (SPMIS_CONF_ADJUST && pixCount > 0.0f) ? min(1.0f, (float)Ntn / pixCount) : 1.0f;
    const float neighbors_conf_sum = (float)agg.z * scaling;   // CONF

    //==== CANONICAL MIS WEIGHT (non-defensive Eq.18, Nc=1 uniform pick over the cell) ====
    //Uses rdi (the centre sample) reconnected at the partner pixel.
    const uint partnerPx = g_pathStateBuffer.Load(SPM_w1(pixelIdx));
    float mis_c = 1.0f;
    if (neighbors_conf_sum > EPSILON && pixCount > 0.0f && partnerPx != SP_UNDEF)
    {
        const float  partnerConf = (float)load_M(g_Reservoirs_current, partnerPx) * scaling;
        const float3 pPos = load_x1(g_sample_current, partnerPx);
        const SurfaceVertex sv_p = BuildVertex(g_sample_current, partnerPx, pPos, camPos);

        float Jn_rev = 0.0f;
        float3 cr = Reconnect(sv_p.x, sv_p.n_s, sv_p.o, sv_p.matID,
                              sv_p.Kd, sv_p.Pr, sv_p.Pm, sv_p.etai, sv_p.etat,
                              rdi.matID, rdi.x2, rdi.n2_s, rdi.L2, rdi.V2,
                              rdi.Kd, rdi.Pr, rdi.Pm, rdi.eta, Jn_rev);
        float ph = GetPHat(cr);
        if (rdi.matID != MATID_LIGHT_TRI && rdi.matID != MATID_ENV_MISS && rdi.Pr < SAMPLE_PT_ROUGHNESS_MIN)
            ph = 0.0f;
        //A rejected Jacobian zeroes tf regardless of visibility, so fold it in BEFORE the
        //shadow ray and skip the IsVisible trace when jr_c==0 (mirrors the non-canonical
        //loop's shift_jac<=0 early-out below). Identical result, one fewer ray on reject.
        const float jr_c = JacobianRatioRej(Jn_rev, my_Jc, spmis_jacThreshold);
        //shadowed resample: visibility of the centre's sample from the partner pixel.
        if (ph > 0.0f && jr_c > 0.0f)
        {
            if (rdi.matID == MATID_ENV_MISS)
            { const float3 md = normalize(rdi.x2);
              ph *= IsVisible(sv_p.x, sv_p.n_s, sv_p.x + md * RAY_TMAX_PLANET, -md) ? 1.0f : 0.0f; }
            else
              ph *= IsVisible(sv_p.x, sv_p.n_s, rdi.x2, rdi.n2_s) ? 1.0f : 0.0f;
        }
        const float tf_center_at_neighbor = ph * jr_c;
        const float denom_mc = tf_center_at_neighbor * neighbors_conf_sum + p_c * centerConf;
        mis_c = (denom_mc > EPSILON)
              ? (pixCount) * (partnerConf / neighbors_conf_sum) * (p_c * centerConf) / denom_mc
              : 0.0f;
    }
    g_pathStateBuffer.Store(SPM_w1(pixelIdx), asuint(mis_c));   // overwrite partnerPx

    //==== NON-CANONICAL: reconnect + shadow ray per selected draw ====
    [loop]
    for (uint d = 0u; d < Ntn; ++d)
    {
        const uint sa  = SPM_slot(pixelIdx, d);
        const uint zPx = g_pathStateBuffer.Load(sa);
        if (zPx == SP_UNDEF) continue;
        const float selProb = asfloat(g_pathStateBuffer.Load(sa + 4u));

        Reservoir pr = loadReservoir(g_Reservoirs_current, zPx);
        //roughness gate + dead-reservoir gate -> mark the slot dead so merge skips it.
        if (pr.W <= 0.0f ||
            (pr.matID != MATID_LIGHT_TRI && pr.matID != MATID_ENV_MISS && pr.Pr < SAMPLE_PT_ROUGHNESS_MIN))
        {
            g_pathStateBuffer.Store(sa, SP_UNDEF);
            continue;
        }

        const float3 zPos = load_x1(g_sample_current, zPx);
        float Jn = 0.0f;
        float3 c = Reconnect(sv_me.x, sv_me.n_s, sv_me.o, sv_me.matID,
                             sv_me.Kd, sv_me.Pr, sv_me.Pm, sv_me.etai, sv_me.etat,
                             pr.matID, pr.x2, pr.n2_s, pr.L2, pr.V2,
                             pr.Kd, pr.Pr, pr.Pm, pr.eta, Jn);
        const float Jc_z      = (pr.matID == MATID_ENV_MISS) ? 1.0f : ComputeJc(zPos, pr.x2, pr.n2_s);
        const float shift_jac = JacobianRatioRej(Jn, Jc_z, spmis_jacThreshold);
        if (shift_jac <= 0.0f)
        {
            g_pathStateBuffer.Store(sa, SP_UNDEF);
            continue;
        }

        //Shadowed resample: trace the reconnection shadow ray (inline RayQuery) so the
        //target + contribution carry visibility (occluded neighbours get target 0).
        float vis = 1.0f;
        if (GetPHat(c) > 0.0f)
        {
            if (pr.matID == MATID_ENV_MISS)
            { const float3 md = normalize(pr.x2);
              vis = IsVisible(sv_me.x, sv_me.n_s, sv_me.x + md * RAY_TMAX_PLANET, -md) ? 1.0f : 0.0f; }
            else
              vis = IsVisible(sv_me.x, sv_me.n_s, pr.x2, pr.n2_s) ? 1.0f : 0.0f;
        }
        c *= vis;
        const float tf_center   = GetPHat(c);                  // shadowed (visibility folded in)
        const float tf_neighbor = GetPHat(pr.F) / shift_jac;   // p_hat_i / J
        const float c_i_scaled  = (float)pr.M * scaling;       // UNCAPPED * SS4.3 scaling

        //non-defensive Eq.16 non-canonical -> reservoir-combine weight
        const float denom  = tf_neighbor * neighbors_conf_sum + tf_center * centerConf;
        const float mi     = (denom > EPSILON) ? (tf_neighbor * c_i_scaled / denom) : 0.0f;
        const float spmis  = (selProb > 0.0f) ? (1.0f / ((float)Ntn * selProb)) : 0.0f;
        const float w_draw = spmis * mi * tf_center * pr.W * shift_jac;

        g_pathStateBuffer.Store (sa + 4u, asuint(w_draw));
        g_pathStateBuffer.Store3(sa + 8u, asuint(c));
    }
}
