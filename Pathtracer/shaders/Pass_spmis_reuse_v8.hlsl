#define COMPUTE_PASS
#include "Includes_v8.hlsli"

//====================================
//SPMIS SPATIAL REUSE  (non-defensive stochastic pairwise MIS)
//====================================
//Owns spatial reuse when RS_FLAG_SPMIS_SPATIAL (0x10) is set; the texture passes
//(select/shift/_v8_1) no-op in that case. Reads the global hash-grid cells built by
//Pass_spmis_{reset,count,offsets,sort} (g_spmisBuffer) plus the post-temporal
//reservoirs (g_Reservoirs_current). Output mirrors the texture path: F*W -> scratch
//slot 2 and the resampled reservoir -> g_Reservoirs_last.
//
//Defaults:
//  * cell search pre-seeds the CENTER cell, 12 iters / r0=20px / x1.25, square,
//    mirror-at-borders, WRS by the TRUE per-cell confidence sum (SP_CONF).
//  * SS4.3 scaling = Ntilde / cell pixel count.
//  * Ntilde non-canonical draws, each an inner RIS over a few uniform samples of the
//    cell's non-zero list (prop. UCW*target*M, P=1/UCW).
//  * canonical Nc=1: one uniform pick over the whole cell.
//  * NON-DEFENSIVE Eq.16/18; confidences UNCAPPED (cap only on the output M).
//  * resampling target is UNSHADOWED; visibility is one deferred ReconnectVis on the
//    selected sample.
//  * roughness gate rejects reconnecting to near-specular vertices.

static const float SP_SEARCH_R0      = 20.0f;
static const float SP_SEARCH_GROW    = 1.25f;
static const uint  SP_SEARCH_ITERS   = 12u;
#define SAMPLE_PT_ROUGHNESS_MIN rs_reconnectRoughnessMin   // editor-settable reconnection-vertex (x2) roughness reject (default 0.15)

//neighbor-similarity heuristic used in the cell search. Normal cone only here
//(plane-distance needs a scene-scale threshold; the hash cell already groups by screen
//tile + normal bucket, so the cone is the key guard).
inline bool sp_similar(uint npx, float3 myPos, float3 myN, float planeThresh)
{
    const uint ninst = load_instID(g_sample_current, npx);
    if (ninst == 0xFFFFFFFFu) return false;
    //normal cone. spmis_normalSimCos is a uniform, so this is a scalar branch: at the
    //default -1 (accept all normals) the test can only fire on an exactly-antiparallel
    //neighbour, so skip the scattered normal fetch entirely - the hash cell already
    //buckets by normal bucket. Only pay the load when the cone is actually enabled.
    if (spmis_normalSimCos > -1.0f)
    {
        const float3 nN = load_n1_s_with_instID(g_sample_current, npx, ninst);
        if (dot(myN, nN) <= spmis_normalSimCos) return false;
    }
    //plane-distance rejection (regular-ReSTIR geometry test rejecting bad cells based
    //on geometry): drop neighbours whose primary hit lies off the query's tangent plane
    //(a depth discontinuity - container edge vs background, etc.).
    const float3 nPos = load_x1_with_instID(g_sample_current, npx, ninst);
    return abs(dot(nPos - myPos, myN)) <= planeThresh;
}

[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (!SPMIS_SPATIAL_MODE) return;
    if (tid.x >= IMG_W || tid.y >= IMG_H) return;
    gDispatchIdx = tid;

    const float2 dims     = float2(IMG_W, IMG_H);
    const uint   pixelIdx = MapPixelID(dims, tid.xy);

    //emitter early-out (mirror the texture path's V2 sentinel)
    if (load_isEmitter(g_sample_current, pixelIdx))
    {
        g_Reservoirs_last.Store(addr_v2(pixelIdx), PROBE_DI_NORMAL_ZERO_CODE);
        return;
    }

    Reservoir rdi = loadReservoir(g_Reservoirs_current, pixelIdx);

    const uint  myInst = load_instID(g_sample_current, pixelIdx);
    const float3 myN   = load_n1_s_with_instID(g_sample_current, pixelIdx, myInst);
    const float3 camPos = InitOrigin();
    const float3 myPos  = load_x1_with_instID(g_sample_current, pixelIdx, myInst);
    const SurfaceVertex sv_me = BuildVertex(g_sample_current, pixelIdx, myPos, camPos);

    const uint cellCenter = g_spmisBuffer.Load(SP_A(SP_HASH, pixelIdx));

    //no center sample / no resolved cell -> passthrough. Also low-roughness glass: its
    //near-specular transmit/refract lobe is strongly view-dependent, so spatial resampling
    //across neighbours spikes p_hat -> fireflies; skip SPMIS and keep the (view-stable)
    //temporal + canonical reservoir. See SPMIS_GLASS_ROUGHNESS_MIN.
    const bool glassNoReuse = sv_me.Pr < SPMIS_GLASS_ROUGHNESS_MIN
                              && LoadKd_w(sv_me.matID) < 1.0f - EPSILON;
    if (rdi.M == 0u || cellCenter == SP_UNDEF || glassNoReuse)
    {
        const float  W    = (rdi.W > 0.0f) ? rdi.W : 0.0f;
        float3       outC = rdi.F * W;
        if (GetPHat(outC) > 0.0f && !IsVolumeVertex(rdi.matID))   //volume vertex is in-medium
            outC *= ReconnectVis(sv_me.x, sv_me.n_s, rdi.matID, rdi.x2, rdi.n2_s);
        gScratchPing[uint3(tid.xy, 2)] = float4(outC, 0);
        storeReservoir(g_Reservoirs_last, pixelIdx, rdi);
        return;
    }

    uint2 seed = GetSeed(pixelIdx, time, 6);
    seed.x = Hash32(seed.x);

    const uint  Ntilde     = max(spmis_reuseN, 1u);
    const float centerConf = (float)rdi.M;                 // UNCAPPED
    const float my_Jc      = (rdi.matID == MATID_ENV_MISS) ? 1.0f
                           : (IsVolumeVertex(rdi.matID) ? ComputeJcVol(myPos, rdi.x2)
                                                        : ComputeJc(myPos, rdi.x2, rdi.n2_s));
    const float visReuse_c = (rdi.W > 0.0f) ? 1.0f : 0.0f;
    const float p_c        = GetPHat(rdi.F) * visReuse_c;  // canonical target at center

    //====================================
    //CELL SEARCH - WRS, center pre-seeded
    //====================================
    float weight_sum    = (float)g_spmisBuffer.Load(SP_CONF_A(cellCenter));
    uint  selectedCell  = cellCenter;
    float radius        = SP_SEARCH_R0;
    //plane-distance reject threshold scaled by distance to camera (scene-scale-free).
    const float planeThresh = spmis_planeDist * length(myPos - camPos);
    [loop]
    for (uint i = 0u; i < SP_SEARCH_ITERS; ++i, radius *= SP_SEARCH_GROW)
    {
        int2 off = int2(round(radius * (RandomFloatPCG(seed.x) * 2.0f - 1.0f)),
                        round(radius * (RandomFloatPCG(seed.x) * 2.0f - 1.0f)));
        int2 nc  = int2(tid.xy) + off;
        //mirror at borders
        if (nc.x < 0) nc.x = -nc.x; else if (nc.x >= (int)IMG_W) nc.x = 2 * (int)IMG_W - nc.x - 1;
        if (nc.y < 0) nc.y = -nc.y; else if (nc.y >= (int)IMG_H) nc.y = 2 * (int)IMG_H - nc.y - 1;
        if (nc.x < 0 || nc.y < 0 || nc.x >= (int)IMG_W || nc.y >= (int)IMG_H) continue;

        const uint npx   = MapPixelID(dims, (uint2)nc);
        const uint ncell = g_spmisBuffer.Load(SP_A(SP_HASH, npx));
        if (ncell == SP_UNDEF || ncell == cellCenter) continue;   // center already seeded
        if (!sp_similar(npx, myPos, myN, planeThresh)) continue;

        const float w = (float)g_spmisBuffer.Load(SP_CONF_A(ncell));
        weight_sum += w;
        if (RandomFloatPCG(seed.x) < w / weight_sum) selectedCell = ncell;
    }

    const uint  reuseCell = selectedCell;
    const uint4 agg       = g_spmisBuffer.Load4(SP_AGG(reuseCell));   // PIXCNT|NZ|CONF|OFF
    const uint  cellBase  = agg.w;          // OFF
    const float pixCount  = (float)agg.x;   // PIXCNT
    const uint  nzCount   = agg.y;          // NZ
    //SS4.3 non-canonical confidence scaling.
    //OFF by default -> scaling = 1 -> canonical weight ~0, full neighbour reuse.
    //ON -> Ntilde/pixCount -> boosts canonical, weaker reuse.
    const float scaling   = (SPMIS_CONF_ADJUST && pixCount > 0.0f) ? min(1.0f, (float)Ntilde / pixCount) : 1.0f;
    const float neighbors_conf_sum = (float)agg.z * scaling;   // CONF (from the Load4 above)

    //====================================
    //CANONICAL MIS WEIGHT (non-defensive Eq.18, Nc=1 uniform pick over the whole cell)
    //====================================
    float mis_c = 1.0f;
    if (neighbors_conf_sum > EPSILON && pixCount > 0.0f)
    {
        uint k = (uint)(pixCount * RandomFloatPCG(seed.x));
        if (k >= (uint)pixCount) k = (uint)pixCount - 1u;
        const uint partnerPx = g_spmisBuffer.Load(SP_A(SP_SORTED, cellBase + k));
        if (partnerPx == SP_UNDEF)
        {
            mis_c = 1.0f;
        }
        else
        {
            const float partnerConf = (float)load_M(g_Reservoirs_current, partnerPx) * scaling;
            const float3 pPos = load_x1(g_sample_current, partnerPx);
            const SurfaceVertex sv_p = BuildVertex(g_sample_current, partnerPx, pPos, camPos);

            float Jn_rev = 0.0f;
            float3 cr = Reconnect(sv_p.x, sv_p.n_s, sv_p.o, sv_p.matID,
                                  sv_p.Kd, sv_p.Pr, sv_p.Pm, sv_p.etai, sv_p.etat,
                                  rdi.matID, rdi.x2, rdi.n2_s, rdi.L2, rdi.V2,
                                  rdi.Kd, rdi.Pr, rdi.Pm, rdi.eta, Jn_rev);
            float ph = GetPHat(cr);
            if (rdi.matID != MATID_LIGHT_TRI && rdi.matID != MATID_ENV_MISS &&
                !IsVolumeVertex(rdi.matID) && rdi.Pr < SAMPLE_PT_ROUGHNESS_MIN)
                ph = 0.0f;
            //shadowed resample: visibility of the centre's sample from the partner pixel,
            //keeping the canonical MIS weight consistent with the shadowed non-canonical.
            //Volume vertices are in-medium (entry surface self-occludes) -> skip the ray.
            //thin glass attenuates (1-F)*Tf -> fold the RGB transmittance into the target.
            if (ph > 0.0f && !IsVolumeVertex(rdi.matID))
            {
                float3 visT;
                if (rdi.matID == MATID_ENV_MISS)
                { const float3 md = normalize(rdi.x2);
                  visT = VisibilityTransmittance(sv_p.x, sv_p.n_s, sv_p.x + md * RAY_TMAX_PLANET, -md); }
                else
                  visT = VisibilityTransmittance(sv_p.x, sv_p.n_s, rdi.x2, rdi.n2_s);
                ph = GetPHat(cr * visT);
            }
            const float tf_center_at_neighbor = ph * JacobianRatioRej(Jn_rev, my_Jc, spmis_jacThreshold);
            const float denom_mc = tf_center_at_neighbor * neighbors_conf_sum + p_c * centerConf;
            //(1/(Nc*Pc)) * (c_partner/cSigma) * (p_c*M_c)/denom ; Pc = 1/pixCount, Nc = 1
            mis_c = (denom_mc > EPSILON)
                  ? (pixCount) * (partnerConf / neighbors_conf_sum) * (p_c * centerConf) / denom_mc
                  : 0.0f;
        }
    }

    //canonical is the initial reservoir content; non-canonical draws WRS into it.
    rdi.w_sum = mis_c * p_c * rdi.W;
    float3 contrib_final = rdi.F * visReuse_c;
    uint   effDraws      = 0u;      // materialized Ntilde draws counted into output confidence

    //====================================
    //NON-CANONICAL (Ntilde draws; each = inner RIS over the cell's non-zero list)
    //====================================
    if (nzCount > 0u)
    {
        [loop]
        for (uint nd = 0u; nd < Ntilde; ++nd)
        {
            //--- inner RIS: pick one non-zero reservoir prop. UCW*target*M, P=1/UCW ---
            float ris_wsum = 0.0f, ris_selTF = 0.0f;
            uint  ris_selK = SP_UNDEF;
            [loop]
            for (uint ri = 0u; ri < spmis_risN; ++ri)
            {
                uint k = (uint)((float)nzCount * RandomFloatPCG(seed.x));
                if (k >= nzCount) k = nzCount - 1u;
                //target (UCW*target*M) precomputed at sort time: ONE float load per
                //candidate instead of load_W + load_F + load_M across two SoA planes.
                const float tf = asfloat(g_spmisBuffer.Load(SP_A(SP_SORTEDW, cellBase + k)));
                const float w  = tf * (float)nzCount / (float)spmis_risN;       // (1/ris)*tf/(1/nz)
                ris_wsum += w;
                if (ris_wsum > 0.0f && RandomFloatPCG(seed.x) < w / ris_wsum) { ris_selK = k; ris_selTF = tf; }
            }
            if (ris_selK == SP_UNDEF || ris_selTF <= 0.0f) continue;
            //resolve the winner's pixel index (single load, winner only) now that the
            //inner RIS no longer touches the reservoir SoA per candidate.
            const uint zPx = g_spmisBuffer.Load(SP_A(SP_SORTED, cellBase + ris_selK));
            if (zPx == SP_UNDEF) continue;
            const float selProb = ris_selTF / ris_wsum;                          // 1/UCW

            Reservoir pr = loadReservoir(g_Reservoirs_current, zPx);
            if (pr.W <= 0.0f) continue;
            //roughness gate: reject reconnecting to a too-smooth GI vertex (volume exempt)
            if (pr.matID != MATID_LIGHT_TRI && pr.matID != MATID_ENV_MISS &&
                !IsVolumeVertex(pr.matID) && pr.Pr < SAMPLE_PT_ROUGHNESS_MIN) continue;

            const float3 zPos = load_x1(g_sample_current, zPx);
            float Jn = 0.0f;
            float3 c = Reconnect(sv_me.x, sv_me.n_s, sv_me.o, sv_me.matID,
                                 sv_me.Kd, sv_me.Pr, sv_me.Pm, sv_me.etai, sv_me.etat,
                                 pr.matID, pr.x2, pr.n2_s, pr.L2, pr.V2,
                                 pr.Kd, pr.Pr, pr.Pm, pr.eta, Jn);
            const float Jc_z      = (pr.matID == MATID_ENV_MISS) ? 1.0f
                                  : (IsVolumeVertex(pr.matID) ? ComputeJcVol(zPos, pr.x2)
                                                              : ComputeJc(zPos, pr.x2, pr.n2_s));
            const float shift_jac = JacobianRatioRej(Jn, Jc_z, spmis_jacThreshold);
            if (shift_jac <= 0.0f) continue;

            //Shadowed resample: trace the reconnection shadow ray now so the target +
            //contribution carry visibility and occluded neighbours (target 0) are never
            //selected. Volume vertices are in-medium, so skip the (self-occluding) ray.
            float3 vis = 1.0.xxx;
            if (GetPHat(c) > 0.0f && !IsVolumeVertex(pr.matID))
            {
                if (pr.matID == MATID_ENV_MISS)
                { const float3 md = normalize(pr.x2);
                  vis = VisibilityTransmittance(sv_me.x, sv_me.n_s, sv_me.x + md * RAY_TMAX_PLANET, -md); }
                else
                  vis = VisibilityTransmittance(sv_me.x, sv_me.n_s, pr.x2, pr.n2_s);
            }
            c *= vis;
            const float tf_center   = GetPHat(c);                  // shadowed (visibility folded in)
            const float tf_neighbor = GetPHat(pr.F) / shift_jac;   // p_hat_i / J
            const float c_i_scaled  = (float)pr.M * scaling;       // UNCAPPED * SS4.3 scaling

            //non-defensive Eq.16 non-canonical
            const float denom = tf_neighbor * neighbors_conf_sum + tf_center * centerConf;
            const float mi    = (denom > EPSILON) ? (tf_neighbor * c_i_scaled / denom) : 0.0f;
            const float spmis = (selProb > 0.0f) ? (1.0f / ((float)Ntilde * selProb)) : 0.0f;
            //reservoir combine: w = mis * target_at_center * UCW * jacobian
            const float w_draw = spmis * mi * tf_center * pr.W * shift_jac;

            rdi.w_sum += w_draw;
            effDraws++;
            if (rdi.w_sum > 0.0f && RandomFloatPCG(seed.x) < w_draw / rdi.w_sum)
            {
                rdi.x2 = pr.x2; rdi.n2_s = pr.n2_s; rdi.objID = pr.objID;
                rdi.matID = pr.matID; rdi.eta = pr.eta;
                rdi.Kd = pr.Kd; rdi.Pr = pr.Pr; rdi.Pm = pr.Pm;
                rdi.L2 = pr.L2; rdi.V2 = pr.V2;
                contrib_final = c;
            }
        }
    }

    //====================================
    //FINALIZE  (M = center.M + effective contributing draws; UCW = wsum / target(selected); M-cap last)
    //====================================
    rdi.F = contrib_final;
    rdi.M = (uint)centerConf + effDraws;
    const float F_mag = GetPHat(rdi.F);

    //bounded UCW (ucw_clampMax): unclamped w_sum/p_hat spikes near grazing/occluded
    //surfaces and feeds back through reuse into a diverging firefly. See FinalizeUCW.
    rdi.W = FinalizeUCW(rdi.w_sum, F_mag, ucw_clampMax);

    if (spmis_mcap > 0u) rdi.M = min(rdi.M, spmis_mcap);

    //Shadowed resample: the contribution already carries visibility, so no deferred ray.
    float3 outC = rdi.F * rdi.W;

    gScratchPing[uint3(tid.xy, 2)] = float4(outC, 0);
    storeReservoir(g_Reservoirs_last, pixelIdx, rdi);
}
