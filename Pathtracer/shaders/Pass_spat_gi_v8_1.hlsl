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

    // Load current sample + reservoir
    SampleData   sdata = loadSampleData(g_sample_current, pixelIdx);
    Reservoir_GI rdi   = loadReservoirGI(g_Reservoirs_current_gi, pixelIdx);

    // Only do spatial GI when there is no L1
    if (!all(sdata.L1 < EPSILON))
    {
        gScratchPing[uint3(launchIndex, 2)] = float4(sdata.L1, 0);
        storeReservoirGI(g_Reservoirs_last_gi, pixelIdx, rdi);
        return;
    }

    // RNG
    uint2 seed = GetSeed(pixelIdx, time, 2);

    // Budgeting (keep semantics)
    const float conf = min(60.0f, rdi.M_gi) / TEMP_MCAP_GI;

    const uint nbrBudget =
        SPAT_COUNT_MIN_GI +
        uint((1.0f - conf) * float(SPAT_COUNT_MAX_GI - SPAT_COUNT_MIN_GI) + 0.5f);

    const uint radiusBudget =
        SPAT_RAD_MIN_GI +
        uint((1.0f - conf) * float(SPAT_RAD_MAX_GI - SPAT_RAD_MIN_GI) + 0.5f);

    // Neighbor ID list (kept compact: [0..validCount-1] valid)
    uint nIds[SPAT_COUNT_MAX_GI];

    // Initialize to invalid (non-unrolled to avoid code bloat)
    [loop]
    for (uint i = 0; i < SPAT_COUNT_MAX_GI; ++i)
        nIds[i] = 0xFFFFFFFFu;

    uint  validCount = 0;
    float M_sum      = 0.0f;
    float M_sum_sym  = 1.0f; // matches your original init

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
                const float3 L1r = load_L1(g_sample_current, iID);
                if (all(L1r < EPSILON) && (load_matID(g_sample_current, iID) == sdata.matID))
                {
                    const float3 n1s_r = load_n1_s(g_sample_current, iID);
                    if (!RejectNormal_GI(sdata.n1_s, n1s_r, 0.36f))
                    {
                        const float3 x1_r = load_x1(g_sample_current, iID);
                        if (!RejectDistance_GI(sdata.x1, x1_r, sdata.n1_s, 0.1f))
                        {
                            ok = true;
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
                    M_sum_sym += 1.0f;

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

    // Refetch material for canonical x1 and x2 vertices (needed for MIS too)
    float3 sKd; float sPr, sPm;
    RefetchMaterial(sdata.matID, sdata.uv, sKd, sPr, sPm);
    float3 rKd; float rPr, rPm;
    RefetchMaterial(rdi.matID_gi, rdi.uv_gi, rKd, rPr, rPm);

    {
        float Jdummy = 0.0f;

        // applyJ = false; Jc is irrelevant in that case (use 1.0f)
        // Jn_canonical is always computed now and used to track J through merge
        float3 contrib_c = ReconnectGI(
            sdata.x1, sdata.n1_s, sdata.n1_g, sdata.o, sdata.matID,
            sKd, sPr, sPm, sdata.etai, sdata.etat,
            rdi.matID_gi, rdi.x2_gi, rdi.n2_s_gi, rdi.n2_g_gi, rdi.L2_gi, rdi.V2_gi,
            rKd, rPr, rPm, rdi.etai_gi, rdi.etat_gi,
            rdi.J_gi.x, 1.0f, false,
            Jn_canonical, Jdummy
        );

        contrib_c *= visReuse;

        p_c = GetPHat(contrib_c);
        contrib_final = contrib_c;
    }

    // Set canonical jacobian (will be overwritten if neighbor wins in merge)
    rdi.J_gi.y = Jn_canonical;

    // MIS for canonical
    float mis_c = 0.0f;
    {
        if (rdi.M_gi <= SPAT_MIN_M_GI)
        {
            mis_c = PairwiseMIS_Canonical_Spat_GI_Sym(
                M_sum_sym, p_c, M_c, nIds,
                rdi.x2_gi, rdi.n2_s_gi, rdi.n2_g_gi, rdi.L2_gi, rdi.V2_gi, rdi.matID_gi,
                rKd, rPr, rPm, rdi.etai_gi, rdi.etat_gi,
                rdi.J_gi.x, rdi.J_gi.y, SPAT_BETA_GI
            );
        }
        else
        {
            mis_c = PairwiseMIS_Canonical_Spat_GI(
                M_sum, p_c, M_c, nIds,
                rdi.x2_gi, rdi.n2_s_gi, rdi.n2_g_gi, rdi.L2_gi, rdi.V2_gi, rdi.matID_gi,
                rKd, rPr, rPm, rdi.etai_gi, rdi.etat_gi,
                rdi.J_gi.x, rdi.J_gi.y
            );
        }
    }

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

            contrib_n = ReconnectGI(
                sdata.x1, sdata.n1_s, sdata.n1_g, sdata.o, sdata.matID,
                sKd, sPr, sPm, sdata.etai, sdata.etat,
                rdi_r.matID_gi, rdi_r.x2_gi, rdi_r.n2_s_gi, rdi_r.n2_g_gi, rdi_r.L2_gi, rdi_r.V2_gi,
                rnKd, rnPr, rnPm, rdi_r.etai_gi, rdi_r.etat_gi,
                rdi_r.J_gi.x, rdi_r.J_gi.y, true,
                Jn, Jnn
            );

            // Visibility after reconnection (keep your normal choice: n1_s)
            const float vis = VisibilityCheckCP(sdata.x1, rdi_r.x2_gi, sdata.n1_s, 0u);
            contrib_n *= vis;

            // Your code multiplies phat by Jnn
            p_hat_from = GetPHat(contrib_n) * Jnn;
        }

        // MIS weight for neighbor
        float mis_n = 0.0f;
        {
            const float Mn = min(SPAT_MCAP_GI, rdi_r.M_gi);

            if (rdi.M_gi <= SPAT_MIN_M_GI)
            {
                mis_n = PairwiseMIS_Neighbor_Spat_GI_Sym(
                    M_sum_sym,
                    M_c, Mn,
                    p_hat_from,
                    rdi_r.W_gi, rdi_r.F_gi,
                    SPAT_BETA_GI
                );
            }
            else
            {
                mis_n = PairwiseMIS_Neighbor_Spat_GI(
                    M_sum,
                    M_c, Mn,
                    p_hat_from,
                    rdi_r.W_gi, rdi_r.F_gi
                );
            }
        }

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

        // J_gi.y already tracked: set to Jn_canonical or overwritten by neighbor Jn in merge

        gScratchPing[uint3(launchIndex, 2)] = float4(contrib_final * rdi.W_gi, 0);
    }

    // Store merged reservoir
    storeReservoirGI(g_Reservoirs_last_gi, pixelIdx, rdi);
}