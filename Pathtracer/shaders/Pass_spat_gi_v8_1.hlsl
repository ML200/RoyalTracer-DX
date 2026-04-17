#include "Includes_v8.hlsli"

// Layout in Pass_spat_gi_select_v8.hlsl
static const uint GI_SEL_STRIDE = 8u;
uint gi_sel_addr(uint linearIdx) { return linearIdx * GI_SEL_STRIDE; }

[shader("raygeneration")]
void Pass_spat_gi_v8_1()
{
    // sortKey: 0=emitter, 1=disabled, 2=no-neighbor, 3=has-neighbor
    uint sortKey;
    {
        const uint2 li  = DispatchRaysIndex().xy;
        const uint  px  = MapPixelID(float2(IMG_W, IMG_H), li);
        const bool  emi = load_isEmitter(g_sample_current, px);

        if (emi || !(rs_flags & 8u))
        {
            sortKey = emi ? 0u : 1u;
        }
        else
        {
            const uint selBase = gi_sel_addr(li.y * IMG_W + li.x);
            sortKey = 2u + g_pathStateBuffer.Load(selBase); // validCount (0 or 1)
        }
    }

    dx::MaybeReorderThread(sortKey, 2);

    //================================================================================
    const uint2  launchIndex = DispatchRaysIndex().xy;
    const float2 dims        = float2(IMG_W, IMG_H);
    const uint   pixelIdx    = MapPixelID(dims, launchIndex);

    Reservoir_GI rdi = loadReservoirGI(g_Reservoirs_current_gi, pixelIdx);

    //Emitter early-out
    if (load_isEmitter(g_sample_current, pixelIdx))
    {
        storeReservoirGI(g_Reservoirs_last_gi, pixelIdx, rdi);
        return;
    }

    //Disabled early-out
    if (!(rs_flags & 8u))
    {
        float3 c = UnpackRGB9E5(rdi.F_gi) * rdi.F_mag_gi;
        float  W = (rdi.W_gi > 0.0f) ? rdi.W_gi : 0.0f;
        gScratchPing[uint3(launchIndex, 2)] = float4(c * W, 0);
        storeReservoirGI(g_Reservoirs_last_gi, pixelIdx, rdi);
        return;
    }

    //Re-read neighbor selection
    const uint linearIdx   = launchIndex.y * IMG_W + launchIndex.x;
    const uint selBase     = gi_sel_addr(linearIdx);
    const uint2 header     = g_pathStateBuffer.Load2(selBase);
    const uint  validCount = header.x;
    const uint  nID        = header.y;

    //Current pixel info
    const uint   myInstID = load_instID(g_sample_current, pixelIdx);
    const uint   myPrimID = load_primID(g_sample_current, pixelIdx);
    const float2 myBary   = load_bary(g_sample_current, pixelIdx);
    const float3 myPos    = ReconstructPosition(myInstID, myPrimID, myBary);

    //Canonical M cap
    const float M_c = min(SPAT_MCAP_GI, rdi.M_gi);
    rdi.M_gi = M_c;

    //Default contrib = canonical's stored integrand, gated by stored-sample validity
    const float  visReuse_c   = (rdi.W_gi > 0.0f) ? 1.0f : 0.0f;
    float3       contrib_final = UnpackRGB9E5(rdi.F_gi) * rdi.F_mag_gi * visReuse_c;

    //RNG for reservoir update
    uint2 seed = GetSeed(pixelIdx, time, 2);

    //No neighbor: canonical passes through unchanged (single-sample MIS == 1 is implicit)
    if (validCount == 0u)
    {
        gScratchPing[uint3(launchIndex, 2)] = float4(contrib_final * rdi.W_gi, 0);
        storeReservoirGI(g_Reservoirs_last_gi, pixelIdx, rdi);
        return;
    }

    //Load neighbor reservoir + pixel info
    Reservoir_GI rdi_r = loadReservoirGI(g_Reservoirs_current_gi, nID);

    const uint   rInstID = load_instID(g_sample_current, nID);
    const uint   rPrimID = load_primID(g_sample_current, nID);
    const float2 rBary   = load_bary(g_sample_current, nID);

    const float3 cameraPos = InitOrigin();

    //─────────────────────────────────────────────────────────────────────────
    //  Pairwise MIS
    //─────────────────────────────────────────────────────────────────────────

    // p_c
    const float p_c = rdi.F_mag_gi * visReuse_c;

    //Canonical Jc: jacobian at current x1 -> canonical x2
    const float Jc_canonical = ComputeJc(myPos, rdi.x2_gi, rdi.n2_s_gi);

    //p_n, reconnect from neighbors vertex to canonicals x2
    float Jnc = 0.0f;
    float p_n = 0.0f;
    {
        SurfaceVertex sv_r = BuildVertex(rInstID, rPrimID, rBary, cameraPos);

        float3 rcKd; float rcPr, rcPm;
        RefetchMaterial(rdi.matID_gi, rdi.uv_gi, rcKd, rcPr, rcPm);

        float3 c = ReconnectGI(
            sv_r.x, sv_r.n_s, sv_r.o, sv_r.matID,
            sv_r.Kd, sv_r.Pr, sv_r.Pm,
            rdi.matID_gi, rdi.x2_gi, rdi.n2_s_gi, rdi.L2_gi, rdi.V2_gi,
            rcKd, rcPr, rcPm, rdi.eta_gi,
            Jnc);

        const float  ph   = GetPHat(c);
        const float3 conn = rdi.x2_gi - sv_r.x;
        const float  cd   = length(conn);
        const float  vis  = (cd > EPSILON && IsVisible(sv_r.x, sv_r.n_s, conn / cd, cd * 0.999f)) ? 1.0f : 0.0f;
        p_n = ph * vis;
    }

    //n_c, reconnect from current vertex to neighbors x2  (+ keep contrib for RIS)
    float Jn = 0.0f;
    float J2 = 0.0f;
    float n_c = 0.0f;
    float3 contrib_n_from_me = 0.0.xxx;
    {
        SurfaceVertex sv_c = BuildVertex(myInstID, myPrimID, myBary, cameraPos);

        //Neighbors own Jc: jacobian at neighbors x1 -> neighbors x2
        const float Jc_neighbor = ComputeJc(
            ReconstructPosition(rInstID, rPrimID, rBary),
            rdi_r.x2_gi, rdi_r.n2_s_gi);

        float3 rrKd; float rrPr, rrPm;
        RefetchMaterial(rdi_r.matID_gi, rdi_r.uv_gi, rrKd, rrPr, rrPm);

        float3 c = ReconnectGI(
            sv_c.x, sv_c.n_s, sv_c.o, sv_c.matID,
            sv_c.Kd, sv_c.Pr, sv_c.Pm,
            rdi_r.matID_gi, rdi_r.x2_gi, rdi_r.n2_s_gi, rdi_r.L2_gi, rdi_r.V2_gi,
            rrKd, rrPr, rrPm, rdi_r.eta_gi,
            Jn);

        J2 = JacobianRatio(Jn, Jc_neighbor);

        const float  ph   = GetPHat(c);
        const float3 conn = rdi_r.x2_gi - sv_c.x;
        const float  cd   = length(conn);
        const float  vis  = (cd > EPSILON && IsVisible(sv_c.x, sv_c.n_s, conn / cd, cd * 0.999f)) ? 1.0f : 0.0f;
        n_c = ph * vis;
        contrib_n_from_me = c * vis;
    }

    //n_n, neighbor's own p_hat
    const float visReuse_n = (rdi_r.W_gi > 0.0f) ? 1.0f : 0.0f;
    const float n_n        = rdi_r.F_mag_gi * visReuse_n;

    //M caps
    const float M_n   = min(SPAT_MCAP_GI, rdi_r.M_gi);
    const float M_sum = M_c + M_n;

    //Jacobian-adjusted cross terms
    const float p_nJ1 = p_n * JacobianRatio(Jnc, Jc_canonical);
    const float n_cJ2 = n_c * J2;

    //Pairwise MIS
    const float mis_c = PairwiseMIS_Canonical_Temp(M_c, M_n, p_c, p_nJ1, M_sum);
    const float mis_n = PairwiseMIS_Neighbour_Temp(M_c, M_n, n_cJ2, n_n, M_sum);

    const float w_c = mis_c * p_c   * rdi.W_gi;
    const float w_n = mis_n * n_cJ2 * rdi_r.W_gi;

    rdi.w_sum_gi = w_c;

    //Track winners F payload + p_hat used for W normalization
    uint  F_gi_color_winner = rdi.F_gi;
    float F_gi_mag_winner   = rdi.F_mag_gi;
    float p_hat_final       = p_c;

    if (UpdateReservoirGI(
            rdi,
            w_n,
            M_n,
            rdi_r.x2_gi, rdi_r.n2_s_gi, rdi_r.L2_gi, rdi_r.V2_gi,
            rdi_r.uv_gi,
            rdi_r.matID_gi, rdi_r.objID_gi, rdi_r.eta_gi,
            rdi_r.F_gi, rdi_r.F_mag_gi,
            seed))
    {
        p_hat_final   = n_c;
        contrib_final = contrib_n_from_me;

        const float  n_c_mag  = GetPHat(contrib_n_from_me);
        const float3 n_c_norm = (n_c_mag > 1e-20f) ? contrib_n_from_me / n_c_mag : float3(0, 0, 0);
        F_gi_color_winner = PackRGB9E5(n_c_norm);
        F_gi_mag_winner   = n_c_mag;
    }

    //Normalize W
    if (p_hat_final > EPSILON && rdi.w_sum_gi > 0.0f && rdi.w_sum_gi < 1e10f)
    {
        float W = rdi.w_sum_gi / p_hat_final;
        if (isnan(W) || isinf(W) || (W < 0.0f)) W = 0.0f;
        rdi.W_gi = W;
    }
    else
    {
        rdi.W_gi = 0.0f;
    }

    rdi.F_gi     = F_gi_color_winner;
    rdi.F_mag_gi = F_gi_mag_winner;

    gScratchPing[uint3(launchIndex, 2)] = float4(contrib_final * rdi.W_gi, 0);

    //Store merged reservoir
    storeReservoirGI(g_Reservoirs_last_gi, pixelIdx, rdi);
}