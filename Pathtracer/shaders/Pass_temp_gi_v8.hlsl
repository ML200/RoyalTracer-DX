#include "Includes_v8.hlsli"

//─────────────────────────────────────────────────────────────────────────────
//  TEMPORAL  GI  (register-pressure optimized refactor)
//─────────────────────────────────────────────────────────────────────────────
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID, uint3 ltid : SV_GroupThreadID)
{
    // Optional bounds (left commented like yours)
    // if (tid.x >= IMG_W || tid.y >= IMG_H) return;

    const uint2 launchIndex = tid.xy;
    const float2 dims_f = float2(IMG_W, IMG_H);
    const uint pixelIdx = MapPixelID(dims_f, launchIndex);

    // Load current sample
    SampleData sdata = loadSampleData(g_sample_current, pixelIdx);

    // Only do temporal GI when there's no L1 (same as your "if(all(L1<EPS)) { ... }")
    if (!all(sdata.L1 < EPSILON))
        return;

    // Load current reservoir (must stay alive until final store)
    Reservoir_GI rdi = loadReservoirGI(g_Reservoirs_current_gi, pixelIdx);

    // Keep boilValue as a single scalar that survives until boil filter
    float boilValue = 0.0f;

    // --- Update canonical Jacobian cache for current stored GI sample ---
    // Put in its own scope so temps die before reprojection / reuse path.
    {
        float Jc = PSSJacobian(
            sdata.x1, sdata.n1_s, sdata.n1_g, sdata.o, sdata.matID,
            sdata.localKd, sdata.localPr, sdata.localPm, sdata.etai, sdata.etat,
            rdi.x2_gi, rdi.n2_s_gi, rdi.n2_g_gi, rdi.V2_gi, rdi.matID_gi,
            rdi.localKd_gi, rdi.localPr_gi, rdi.localPm_gi, rdi.etai_gi, rdi.etat_gi,
            rdi.J_gi.x
        );
        rdi.J_gi.y = Jc;
    }

    // RNG (keep as small as possible; preserve your semantics)
    uint2 seed = GetSeed(pixelIdx, time, 3);
    uint  permSeed = GetSeed(1, time, 3).x;

    // --- base reprojection (fallback) ---
    int2 baseCoord = GetBestReprojectedPixel_d(sdata.x1, prevView, prevProjection, dims_f, sdata.objID);
    if (baseCoord.x == -1 && baseCoord.y == -1)
        baseCoord = (int2)launchIndex;

    // --- permuted candidate ---
    int2 permCoord = baseCoord;
    bool permInBounds = false;
    {
        float u = RandomFloatSingle(permSeed);
        uint  permRnd = (uint)min(u * 16.0f, 15.0f);

        ApplyPermutationSampling(permCoord, permRnd);

        permInBounds =
            (permCoord.x >= 0 && permCoord.y >= 0 &&
             permCoord.x < (int)IMG_W && permCoord.y < (int)IMG_H);

        if (!permInBounds)
            permCoord = baseCoord;
    }

    bool valid = false;
    uint tempPixelIdx = 0xFFFFFFFFu;

    // --- try permuted first ---
    // Note: delay reservoir load until sample-level rejects pass, to reduce pressure on invalid path.
    SampleData sdata_r = (SampleData)0;

    if (permInBounds)
    {
        tempPixelIdx = MapPixelID(dims_f, (uint2)permCoord);
        sdata_r = loadSampleData(g_sample_last, tempPixelIdx);

        // Cheap rejects first (avoid loading rdi_r when not needed)
        valid =
            (all(sdata_r.L1 < EPSILON) &&
             (sdata_r.matID == sdata.matID) &&
             !RejectNormal_GI(sdata.n1_s, sdata_r.n1_s, 0.36f) &&
             (!RejectDistance_GI(sdata.x1, sdata_r.x1, sdata.n1_s, 0.05f)));
    }

    // --- heavy reuse path ---
    // Strongly isolate lifetime of rdi_r + all heavy temporaries.
    [branch]
    if (valid)
    {
        Reservoir_GI rdi_r = loadReservoirGI(g_Reservoirs_last_gi, tempPixelIdx);

        // Final validity check that requires reservoir
        if (!IsValidReservoir_GI(rdi_r))
        {
            valid = false;
        }
        else
        {
            // Everything below is scoped to minimize peak live values.
            float p_hat_final = 0.0f;

            {
                // Canonical / neighbour target terms
                float Jnc = 0.0f, Jn = 0.0f;
                float J1  = 1.0f, J2 = 1.0f;

                const float visReuse_c = (rdi.W_gi > 0.0f) ? 1.0f : 0.0f;
                const float p_c = rdi.F_gi * visReuse_c;

                float p_n = 0.0f;
                float n_c = 0.0f;

                // p_n = phat(ReconnectGI(sdata_r -> current rdi sample)) * visibility
                {
                    float3 c = ReconnectGI(
                        sdata_r.x1, sdata_r.n1_s, sdata_r.n1_g, sdata_r.o, sdata_r.matID,
                        sdata_r.localKd, sdata_r.localPr, sdata_r.localPm, sdata_r.etai, sdata_r.etat,
                        rdi.matID_gi, rdi.x2_gi, rdi.n2_s_gi, rdi.n2_g_gi, rdi.L2_gi, rdi.V2_gi,
                        rdi.localKd_gi, rdi.localPr_gi, rdi.localPm_gi, rdi.etai_gi, rdi.etat_gi,
                        rdi.J_gi.x, rdi.J_gi.y,
                        true, Jnc, J1
                    );

                    float ph = GetPHat(c);
                    float vis = VisibilityCheckCP(sdata_r.x1, rdi.x2_gi, sdata_r.n1_g, 0u);
                    p_n = ph * vis;
                }

                // n_c = phat(ReconnectGI(sdata -> neighbour rdi_r sample)) * visibility
                {
                    float3 c = ReconnectGI(
                        sdata.x1, sdata.n1_s, sdata.n1_g, sdata.o, sdata.matID,
                        sdata.localKd, sdata.localPr, sdata.localPm, sdata.etai, sdata.etat,
                        rdi_r.matID_gi, rdi_r.x2_gi, rdi_r.n2_s_gi, rdi_r.n2_g_gi, rdi_r.L2_gi, rdi_r.V2_gi,
                        rdi_r.localKd_gi, rdi_r.localPr_gi, rdi_r.localPm_gi, rdi_r.etai_gi, rdi_r.etat_gi,
                        rdi_r.J_gi.x, rdi_r.J_gi.y,
                        true, Jn, J2
                    );

                    float ph = GetPHat(c);
                    float vis = VisibilityCheckCP(sdata.x1, rdi_r.x2_gi, sdata.n1_g, 0u);
                    n_c = ph * vis;
                }

                const float visReuse_n = (rdi_r.W_gi > 0.0f) ? 1.0f : 0.0f;
                const float n_n = rdi_r.F_gi * visReuse_n;

                // Dynamic M caps
                const float minRoughTemp  = min(sdata.localPr, rdi_r.localPr_gi);
                const float tempMcapScale = smoothstep(REUSE_ROUGHNESS_MIN, REUSE_ROUGHNESS_MAX, minRoughTemp);
                const float dynTempMcap   = (minRoughTemp <= REUSE_ROUGHNESS_MIN) ? 0.0f
                                        : min(TEMP_MCAP_GI, TEMP_MCAP_GI * tempMcapScale);

                const float M_c   = min(TEMP_MCAP_GI, rdi.M_gi);
                const float M_n   = min(dynTempMcap,  rdi_r.M_gi);
                const float M_sum = M_c + M_n;

                // Precompute reused products to reduce recompute and (often) live ranges
                const float p_nJ1  = p_n * J1;
                const float n_cJ2  = n_c * J2;

                const float mis_c = PairwiseMIS_Canonical_Temp(M_c, M_n, p_c, p_nJ1, M_sum);
                const float mis_n = PairwiseMIS_Neighbour_Temp(M_c, M_n, n_cJ2, n_n, M_sum);

                const float w_c = mis_c * p_c * rdi.W_gi;
                const float w_n = mis_n * n_cJ2 * rdi_r.W_gi;

                // Adjust wsum of the existing reservoir
                rdi.w_sum_gi = w_c;

                // Update reservoir: if neighbour wins, canonical target becomes n_c (as in your code)
                p_hat_final = p_c;
                if (UpdateReservoirGI(
                        rdi,
                        w_n,
                        rdi_r.M_gi,
                        rdi_r.x2_gi, rdi_r.n2_s_gi, rdi_r.n2_g_gi, rdi_r.L2_gi, rdi_r.V2_gi,
                        rdi_r.localKd_gi, rdi_r.localPr_gi, rdi_r.localPm_gi,
                        rdi_r.etai_gi, rdi_r.etat_gi,
                        rdi_r.matID_gi, rdi_r.objID_gi,
                        rdi_r.J_gi,
                        rdi_r.F_gi,
                        seed
                    ))
                {
                    p_hat_final = n_c;
                    rdi.J_gi.y  = Jn;
                }

                // Compute new W (same logic)
                if (p_hat_final > EPSILON && rdi.w_sum_gi > 0.0f)
                {
                    float W = rdi.w_sum_gi / p_hat_final;
                    if (isnan(W) || isinf(W) || (W < 0.0f)) W = 0.0f;
                    rdi.W_gi = W;
                }
                else
                {
                    rdi.W_gi = 0.0f;
                }

                boilValue = p_hat_final * rdi.W_gi;

                // Recompute/update jacobian for the now-canonical stored sample (same as your code)
                rdi.J_gi.y = PSSJacobian(
                    sdata.x1, sdata.n1_s, sdata.n1_g, sdata.o, sdata.matID,
                    sdata.localKd, sdata.localPr, sdata.localPm, sdata.etai, sdata.etat,
                    rdi.x2_gi, rdi.n2_s_gi, rdi.n2_g_gi, rdi.V2_gi, rdi.matID_gi,
                    rdi.localKd_gi, rdi.localPr_gi, rdi.localPm_gi, rdi.etai_gi, rdi.etat_gi,
                    rdi.J_gi.x
                );

                rdi.F_gi = p_hat_final;
            }
        }
    }

    // --- boiling filter ---
    // Isolate to avoid carrying heavy temporaries into store.
    {
        float avgV, thrV;
        bool boil = BoilingFilter(ltid.xy, GI_BOIL_STRENGTH_TEMP, boilValue, avgV, thrV);

        if (avgV < GI_BOIL_MIN_AVG_TEMP)
            boil = false;

        if (boil && boilValue > 0.0f)
        {
            float scale = thrV / boilValue;
            rdi.W_gi *= scale;
            rdi.w_sum_gi = rdi.W_gi * max(rdi.F_gi, EPSILON);
            rdi.M_gi = min(rdi.M_gi, 1.0f);
        }
    }

    // Store merged reservoir
    storeReservoirGI(g_Reservoirs_current_gi, pixelIdx, rdi);
}