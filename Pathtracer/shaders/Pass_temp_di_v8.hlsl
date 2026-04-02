#include "Includes_v8.hlsli"

//─────────────────────────────────────────────────────────────────────────────
//  TEMPORAL  DI
//─────────────────────────────────────────────────────────────────────────────
[numthreads(16, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= IMG_W || tid.y >= IMG_H) return;
    gDispatchIdx = tid;

    uint2  launchIndex   = tid.xy;
    float2 dims = float2(IMG_W, IMG_H);
    uint   pixelIdx  = MapPixelID(dims, launchIndex);

    // Load the sample data
    SampleData sdata = loadSampleData(g_sample_current, pixelIdx);
    if (!all(sdata.L1 < EPSILON))
        return;

    // Load current reservoir
    Reservoir_DI rdi = loadReservoirDI(g_Reservoirs_current_di, pixelIdx);

    float boilValue = 0.0f;

    // RNG
    uint2 seed = GetSeed(pixelIdx, time, 2);
    uint  permSeed = GetSeed(1, time, 2).x;

    // --- base reprojection ---
    int2 baseCoord = GetBestReprojectedPixel_d(sdata.x1, prevView, prevProjection, dims, sdata.objID);
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
    SampleData sdata_r = (SampleData)0;

    // --- try permuted first ---
    if (permInBounds)
    {
        tempPixelIdx = MapPixelID(dims, (uint2)permCoord);
        sdata_r = loadSampleData(g_sample_last, tempPixelIdx);

        valid =
            (all(sdata_r.L1 < EPSILON) &&
             (sdata_r.matID == sdata.matID) &&
             !RejectNormal_DI(sdata.n1_s, sdata_r.n1_s, 0.9f) &&
             !RejectDistance_DI(sdata.x1, sdata_r.x1, sdata.n1_s, 0.05f));
    }

    // --- fallback to non-permuted base coordinate ---
    if (!valid)
    {
        int2 fallback = baseCoord;
        if (fallback.x >= 0 && fallback.y >= 0 &&
            fallback.x < (int)IMG_W && fallback.y < (int)IMG_H)
        {
            tempPixelIdx = MapPixelID(dims, (uint2)fallback);
            sdata_r = loadSampleData(g_sample_last, tempPixelIdx);

            valid =
                (all(sdata_r.L1 < EPSILON) &&
                 (sdata_r.matID == sdata.matID) &&
                 !RejectNormal_DI(sdata.n1_s, sdata_r.n1_s, 0.9f) &&
                 !RejectDistance_DI(sdata.x1, sdata_r.x1, sdata.n1_s, 0.05f));
        }
    }

    // --- fallback: 3x3 neighbourhood search around baseCoord (cheap loads) ---
    if (!valid)
    {
        [unroll]
        for (int dy = -1; dy <= 1 && !valid; ++dy)
        {
            [unroll]
            for (int dx = -1; dx <= 1 && !valid; ++dx)
            {
                if (dx == 0 && dy == 0) continue;

                int2 c = baseCoord + int2(dx, dy);
                if (c.x < 0 || c.y < 0 || c.x >= (int)IMG_W || c.y >= (int)IMG_H)
                    continue;

                uint cIdx = MapPixelID(dims, (uint2)c);

                if (!all(load_L1(g_sample_last, cIdx) < EPSILON))   continue;
                if (load_matID(g_sample_last, cIdx) != sdata.matID) continue;
                float3 cn1s = load_n1_s(g_sample_last, cIdx);
                if (RejectNormal_DI(sdata.n1_s, cn1s, 0.9f))       continue;
                float3 cx1 = load_x1(g_sample_last, cIdx);
                if (RejectDistance_DI(sdata.x1, cx1, sdata.n1_s, 0.05f)) continue;

                tempPixelIdx = cIdx;
                sdata_r = loadSampleData(g_sample_last, cIdx);
                valid = true;
            }
        }
    }

    // --- last resort: current pixel (no reprojection) ---
    if (!valid)
    {
        tempPixelIdx = pixelIdx;
        sdata_r = loadSampleData(g_sample_last, pixelIdx);

        valid =
            (all(sdata_r.L1 < EPSILON) &&
             (sdata_r.matID == sdata.matID) &&
             !RejectNormal_DI(sdata.n1_s, sdata_r.n1_s, 0.9f) &&
             !RejectDistance_DI(sdata.x1, sdata_r.x1, sdata.n1_s, 0.05f));
    }

    // --- heavy reuse path ---
    [branch]
    if (valid)
    {
        Reservoir_DI rdi_r = loadReservoirDI(g_Reservoirs_last_di, tempPixelIdx);

        if (!IsValidReservoir_DI(rdi_r))
        {
            valid = false;
        }
        else
        {
            // Refetch materials for both vertices
            float3 sKd; float sPr, sPm;
            RefetchMaterial(sdata.matID, sdata.uv, sKd, sPr, sPm);
            float3 srKd; float srPr, srPm;
            RefetchMaterial(sdata_r.matID, sdata_r.uv, srKd, srPr, srPm);

            // Calculate the canonical target function
            float visReuse_c = rdi.W_di > 0.0f ? 1.0f : 0.0f;
            float p_c = GetPHat(ReconnectDI(sdata.x1, sdata.n1_s, sdata.n1_g, sdata.o, sdata.matID, rdi.x2_di, rdi.n2_di, rdi.L2_di, sKd, sPr, sPm, sdata.etai, sdata.etat, rdi.objID_di)) * visReuse_c;
            float p_n = GetPHat(ReconnectDI(sdata_r.x1, sdata_r.n1_s, sdata_r.n1_g, sdata_r.o, sdata_r.matID, rdi.x2_di, rdi.n2_di, rdi.L2_di, srKd, srPr, srPm, sdata_r.etai, sdata_r.etat, rdi.objID_di));
            p_n *= VisibilityCheckCP(sdata_r.x1, rdi.x2_di, sdata_r.n1_s, rdi.objID_di);
            p_n *= JacobianDeterminantDI(sdata.x1, rdi.x2_di, sdata_r.x1, rdi.n2_di, rdi.objID_di);
            float n_c = GetPHat(ReconnectDI(sdata.x1, sdata.n1_s, sdata.n1_g, sdata.o, sdata.matID, rdi_r.x2_di, rdi_r.n2_di, rdi_r.L2_di, sKd, sPr, sPm, sdata.etai, sdata.etat, rdi_r.objID_di));
            n_c *= VisibilityCheckCP(sdata.x1, rdi_r.x2_di, sdata.n1_g, rdi_r.objID_di);
            n_c *= JacobianDeterminantDI(sdata_r.x1, rdi_r.x2_di, sdata.x1, rdi_r.n2_di, rdi_r.objID_di);
            float visReuse = rdi_r.W_di > 0.0f ? 1.0f : 0.0f;
            float n_n = GetPHat(ReconnectDI(sdata_r.x1, sdata_r.n1_s, sdata_r.n1_g, sdata_r.o, sdata_r.matID, rdi_r.x2_di, rdi_r.n2_di, rdi_r.L2_di, srKd, srPr, srPm, sdata_r.etai, sdata_r.etat, rdi_r.objID_di)) * visReuse;

            // Dynamic M caps (roughness-based, aligned with GI)
            float sdata_Pr = EvaluatePBRProperties(materials[sdata.matID], sdata.uv, 0).x;
            float rdi_r_Pr = EvaluatePBRProperties(materials[sdata_r.matID], sdata_r.uv, 0).x;
            const float minRoughTemp  = min(sdata_Pr, rdi_r_Pr);
            const float tempMcapScale = smoothstep(REUSE_ROUGHNESS_MIN, REUSE_ROUGHNESS_MAX, minRoughTemp);
            const float dynTempMcap   = (minRoughTemp <= REUSE_ROUGHNESS_MIN) ? 0.0f
                                    : min(TEMP_MCAP_DI, TEMP_MCAP_DI * tempMcapScale);

            const float M_c   = min(TEMP_MCAP_DI, rdi.M_di);
            const float M_n   = min(dynTempMcap,  rdi_r.M_di);
            const float M_sum = M_c + M_n;

            // Calculate the MIS weights
            float mis_c = PairwiseMIS_Canonical_Temp(M_c, M_n, p_c, p_n, M_sum);
            float mis_n = PairwiseMIS_Neighbour_Temp(M_c, M_n, n_c, n_n, M_sum);

            // Calculate the reservoirs weights
            float w_c = mis_c * p_c * rdi.W_di;
            float w_n = mis_n * n_c * rdi_r.W_di;

            // Adjust wsum of the existing reservoir
            rdi.w_sum_di = w_c;

            // Update the reservoir
            float p_hat_final = p_c;
            if(UpdateReservoirDI(rdi, w_n, rdi_r.M_di, rdi_r.x2_di, rdi_r.n2_di, rdi_r.L2_di, rdi_r.objID_di, seed)){
                p_hat_final = n_c;
            }

            // Calculate new W
            if (p_hat_final > 1e-5 && rdi.w_sum_di > EPSILON && rdi.w_sum_di < 1e10f) {
                float W = rdi.w_sum_di / p_hat_final;
                if (isnan(W) || isinf(W)) {
                    W = 0.0f;
                }
                rdi.W_di = W;
            }
            else
                rdi.W_di = 0.0f;

            boilValue = p_hat_final * rdi.W_di;
        }
    }

    // --- boiling filter ---
    {
        float avgV, thrV;
        bool boil = BoilingFilter_Wave(DI_BOIL_STRENGTH_TEMP, boilValue, avgV, thrV);

        if (avgV < DI_BOIL_MIN_AVG_TEMP)
            boil = false;

        if (boil && boilValue > 0.0f)
        {
            float scale = thrV / boilValue;
            rdi.W_di     *= scale;
            rdi.w_sum_di *= scale;
            rdi.M_di = min(rdi.M_di, 1u);
        }
    }

    // Store the merged reservoir
    storeReservoirDI(g_Reservoirs_current_di, pixelIdx, rdi);
}
