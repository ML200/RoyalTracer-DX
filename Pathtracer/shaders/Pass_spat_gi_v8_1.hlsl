#include "Includes_raygen_v8.hlsli"

//─────────────────────────────────────────────────────────────────────────────
//  SPATIAL  GI  (raygen shader)
//─────────────────────────────────────────────────────────────────────────────
[shader("raygeneration")]
void Pass_spat_gi_v8_1()
{
    const uint2  launchIndex = DispatchRaysIndex().xy;
    const float2 dims        = float2(IMG_W, IMG_H);
    const uint   pixelIdx    = MapPixelID(dims, launchIndex);

    Reservoir_GI rdi = loadReservoirGI(g_Reservoirs_current_gi, pixelIdx);

    // Only do spatial GI when pixel is not an emitter
    if (load_isEmitter(g_sample_current, pixelIdx))
    {
        storeReservoirGI(g_Reservoirs_last_gi, pixelIdx, rdi);
        return;
    }

    // Lightweight loads
    const uint   myInstID = load_instID(g_sample_current, pixelIdx);
    const uint   myPrimID = load_primID(g_sample_current, pixelIdx);
    const float2 myBary   = load_bary(g_sample_current, pixelIdx);
    const uint   myMatID  = GetMatIDFast(myInstID, myPrimID);
    const float3 myPos    = ReconstructPosition(myInstID, myPrimID, myBary);
    const float3 myN1s    = load_n1_s_with_instID(g_sample_current, pixelIdx, myInstID);
    const float3 myN1g    = load_n1_g_with_instID(g_sample_current, pixelIdx, myInstID);
    const float2 myUV     = load_uv(g_sample_current, pixelIdx);
    float3 myKd; float myPr, myPm;
    RefetchMaterial(myMatID, myUV, myKd, myPr, myPm);

    // Specularity: reduce spatial samples for reflective surfaces
    const float3 camPos     = InitOrigin();
    const float  NoV        = saturate(dot(normalize(camPos - myPos), myN1s));
    const float  specularity = Luma(EnvBRDFApprox2(myKd, myPr, myPm, NoV));

    // If spatial GI is disabled, compute canonical output and store reservoir unchanged
    if (!(rs_flags & 8u))
    {
        const float vis = (rdi.W_gi > 0.0f) ? 1.0f : 0.0f;
        const float3 cameraPos = InitOrigin();
        SurfaceVertex sv1 = BuildVertex(myInstID, myPrimID, myBary, cameraPos);
        sv1.etai = load_etai(g_sample_current, pixelIdx);
        sv1.etat = load_etat(g_sample_current, pixelIdx);
        float3 rKd0; float rPr0, rPm0;
        RefetchMaterial(rdi.matID_gi, rdi.uv_gi, rKd0, rPr0, rPm0);
        SurfaceVertex sv2 = { rdi.x2_gi, rdi.n2_s_gi, rdi.n2_g_gi, rdi.V2_gi, rKd0, rPr0, rPm0, rdi.etai_gi, rdi.etat_gi, rdi.matID_gi, rdi.uv_gi };
        float Jd1 = 0.0f, Jd2 = 0.0f;
        float3 c = ReconnectGI(
            sv1.x, sv1.n_s, sv1.n_g, sv1.o, sv1.matID,
            sv1.Kd, sv1.Pr, sv1.Pm, sv1.etai, sv1.etat,
            sv2.matID, sv2.x, sv2.n_s, sv2.n_g, rdi.L2_gi, sv2.o,
            sv2.Kd, sv2.Pr, sv2.Pm, sv2.etai, sv2.etat,
            rdi.J_gi.x, 1.0f, false, Jd1, Jd2) * vis;
        gScratchPing[uint3(launchIndex, 2)] = float4(c * rdi.W_gi, 0);
        storeReservoirGI(g_Reservoirs_last_gi, pixelIdx, rdi);
        return;
    }

    // RNG
    uint2 seed = GetSeed(pixelIdx, time, 2);

    // Budgeting (keep semantics)
    const float conf = min(60.0f, rdi.M_gi) / max(1u, rs_tempMcapGI);

    const uint baseBudget =
        min(rs_spatCountMinGI, SPAT_COUNT_MAX_GI) +
        uint((1.0f - conf) * float(min(rs_spatCountMaxGI, SPAT_COUNT_MAX_GI) - min(rs_spatCountMinGI, SPAT_COUNT_MAX_GI)) + 0.5f);
    const uint nbrBudget = uint(float(baseBudget) * (1.0f - specularity) + 0.5f);

    const uint radiusBudget =
        rs_spatRadMinGI +
        uint((1.0f - conf) * float(rs_spatRadMaxGI - rs_spatRadMinGI) + 0.5f);

    // Neighbor ID list (kept compact: [0..validCount-1] valid)
    uint nIds[SPAT_COUNT_MAX_GI];

    // Initialize to invalid (non-unrolled to avoid code bloat)
    [loop]
    for (uint i = 0; i < SPAT_COUNT_MAX_GI; ++i)
        nIds[i] = 0xFFFFFFFFu;

    uint  validCount = 0;
    float M_sum      = 0.0f;

    //─────────────────────────────────────────────────────────────────────────
    // Candidate selection: compact list, delay expensive loads
    //─────────────────────────────────────────────────────────────────────────
    [loop]
    for (uint i = 0; i < nbrBudget; ++i)
    {
        uint chosen = 0xFFFFFFFFu;

        [loop]
        for (uint j = 0; j < SPAT_TRIS_GI; ++j)
        {
            const uint iID = GetRandomPixelCircleWeighted(
                radiusBudget, dims.x, dims.y,
                launchIndex.x, launchIndex.y,
                seed);

            // Cheap checks first (sample fetches) before reservoir fetch
            // Keep in tight scope to drop temps ASAP.
            bool ok = false;
            {
                if (!load_isEmitter(g_sample_current, iID))
                {
                    uint nInstID_t = load_instID(g_sample_current, iID);
                    uint nPrimID_t = load_primID(g_sample_current, iID);
                    if (GetMatIDFast(nInstID_t, nPrimID_t) == myMatID)
                    {
                        const float3 n1g_r = load_n1_g_with_instID(g_sample_current, iID, nInstID_t);
                        if (!RejectNormal_GI(myN1g, n1g_r, 0.36f))
                        {
                            float2 nBary_t = load_bary(g_sample_current, iID);
                            const float3 x1_r = ReconstructPosition(nInstID_t, nPrimID_t, nBary_t);
                            if (!RejectDistance_GI(myPos, x1_r, myN1g, 0.1f))
                            {
                                ok = true;
                            }
                        }
                    }
                }
            }

            if (!ok)
                continue;

            // Now load the reservoir only for plausible candidates
            {
                Reservoir_GI rdi_r = loadReservoirGI(g_Reservoirs_current_gi, iID);

                if (IsValidReservoir_GI_opt(rdi_r.n2_g_gi, rdi_r.M_gi))
                {
                    chosen = iID;

                    const float Mn = min(SPAT_MCAP_GI, rdi_r.M_gi);
                    M_sum     += Mn;

                    break;
                }
            }
        }

        if (chosen != 0xFFFFFFFFu)
            nIds[validCount++] = chosen;
    }

    // Canonical M cap + include in M_sum (same as your original intent)
    const float M_c = min(SPAT_MCAP_GI, rdi.M_gi);
    rdi.M_gi = M_c;
    M_sum += M_c;

    //─────────────────────────────────────────────────────────────────────────
    // Canonical contribution (stage everything to reduce live ranges)
    //─────────────────────────────────────────────────────────────────────────
    const float visReuse = (rdi.W_gi > 0.0f) ? 1.0f : 0.0f;

    float3 contrib_final = 0.0.xxx;
    float  p_c           = 0.0f;
    float  Jn_canonical  = 0.0f;

    // Build canonical vertex
    const float3 cameraPos2 = InitOrigin();
    SurfaceVertex sv1 = BuildVertex(myInstID, myPrimID, myBary, cameraPos2);
    sv1.etai = load_etai(g_sample_current, pixelIdx);
    sv1.etat = load_etat(g_sample_current, pixelIdx);

    // Build canonical x2 vertex from reservoir
    float3 rKd; float rPr, rPm;
    RefetchMaterial(rdi.matID_gi, rdi.uv_gi, rKd, rPr, rPm);
    SurfaceVertex sv2_c = { rdi.x2_gi, rdi.n2_s_gi, rdi.n2_g_gi, rdi.V2_gi, rKd, rPr, rPm, rdi.etai_gi, rdi.etat_gi, rdi.matID_gi, rdi.uv_gi };

    {
        float Jdummy = 0.0f;
        float3 contrib_c = ReconnectGI(
            sv1.x, sv1.n_s, sv1.n_g, sv1.o, sv1.matID,
            sv1.Kd, sv1.Pr, sv1.Pm, sv1.etai, sv1.etat,
            sv2_c.matID, sv2_c.x, sv2_c.n_s, sv2_c.n_g, rdi.L2_gi, sv2_c.o,
            sv2_c.Kd, sv2_c.Pr, sv2_c.Pm, sv2_c.etai, sv2_c.etat,
            rdi.J_gi.x, 1.0f, false, Jn_canonical, Jdummy);
        contrib_c *= visReuse;
        p_c = GetPHat(contrib_c);
        contrib_final = contrib_c;
    }

    // Set canonical jacobian (will be overwritten if neighbor wins in merge)
    rdi.J_gi.y = Jn_canonical;

    // MIS for canonical
    const float mis_c = PairwiseMIS_Canonical_Spat_GI(
        M_sum, p_c, M_c, nIds,
        rdi.x2_gi, rdi.n2_s_gi, rdi.n2_g_gi, rdi.L2_gi, rdi.V2_gi, rdi.matID_gi,
        rKd, rPr, rPm, rdi.etai_gi, rdi.etat_gi,
        rdi.J_gi.x, rdi.J_gi.y
    );

    // Adjust canonical weight
    rdi.w_sum_gi = mis_c * p_c * rdi.W_gi;

    //─────────────────────────────────────────────────────────────────────────
    // Merge neighbors (tight per-iteration scopes)
    //─────────────────────────────────────────────────────────────────────────
    [loop]
    for (uint k = 0; k < validCount; ++k)
    {
        const uint nID = nIds[k];
        // (nID is guaranteed valid here)

        // Load neighbor reservoir only for this iteration
        Reservoir_GI rdi_r = loadReservoirGI(g_Reservoirs_current_gi, nID);

        // Compute neighbor reconnection contribution, staged
        float3 contrib_n = 0.0.xxx;
        float  p_hat_from = 0.0f;
        float  Jn = 0.0f;
        float  Jnn = 1.0f;

        {
            // Refetch neighbor x2 material
            float3 rnKd; float rnPr, rnPm;
            RefetchMaterial(rdi_r.matID_gi, rdi_r.uv_gi, rnKd, rnPr, rnPm);

            SurfaceVertex sv2_n = { rdi_r.x2_gi, rdi_r.n2_s_gi, rdi_r.n2_g_gi, rdi_r.V2_gi, rnKd, rnPr, rnPm, rdi_r.etai_gi, rdi_r.etat_gi, rdi_r.matID_gi, rdi_r.uv_gi };

            contrib_n = ReconnectGI(
                sv1.x, sv1.n_s, sv1.n_g, sv1.o, sv1.matID,
                sv1.Kd, sv1.Pr, sv1.Pm, sv1.etai, sv1.etat,
                sv2_n.matID, sv2_n.x, sv2_n.n_s, sv2_n.n_g, rdi_r.L2_gi, sv2_n.o,
                sv2_n.Kd, sv2_n.Pr, sv2_n.Pm, sv2_n.etai, sv2_n.etat,
                rdi_r.J_gi.x, rdi_r.J_gi.y, true, Jn, Jnn);

            // Visibility after reconnection
            {
                float3 _conn = rdi_r.x2_gi - sv1.x; float _cd = length(_conn);
                float vis = (_cd > EPSILON && IsVisible(sv1.x, sv1.n_g, _conn / _cd, _cd * 0.999f)) ? 1.0f : 0.0f;
                contrib_n *= vis;
            }

            p_hat_from = GetPHat(contrib_n) * Jnn;
        }

        // MIS weight for neighbor
        const float Mn = min(SPAT_MCAP_GI, rdi_r.M_gi);
        const float mis_n = PairwiseMIS_Neighbor_Spat_GI(
            M_sum,
            M_c, Mn,
            p_hat_from,
            rdi_r.W_gi, rdi_r.F_gi
        );

        const float w_n = mis_n * p_hat_from * rdi_r.W_gi;

        // Update reservoir (preserve your behavior)
        if (UpdateReservoirGI(
                rdi,
                w_n,
                min(SPAT_MCAP_GI, rdi_r.M_gi),
                rdi_r.x2_gi, rdi_r.n2_s_gi, rdi_r.n2_g_gi, rdi_r.L2_gi, rdi_r.V2_gi,
                rdi_r.uv_gi,
                rdi_r.etai_gi, rdi_r.etat_gi,
                rdi_r.matID_gi, rdi_r.objID_gi,
                rdi_r.J_gi, rdi_r.F_gi,
                rdi_r.seed_gi,
                seed
            ))
        {
            contrib_final = contrib_n;
            rdi.J_gi.y = Jn;
        }
    }

    //─────────────────────────────────────────────────────────────────────────
    // Finalize W/F/J and output
    //─────────────────────────────────────────────────────────────────────────
    {
        const float p_hat_final = GetPHat(contrib_final);

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

        rdi.F_gi = p_hat_final;

        gScratchPing[uint3(launchIndex, 2)] = float4(contrib_final * rdi.W_gi, 0);
    }

    // Store merged reservoir
    storeReservoirGI(g_Reservoirs_last_gi, pixelIdx, rdi);
}