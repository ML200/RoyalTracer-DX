#include "Includes_v8.hlsli"

//─────────────────────────────────────────────────────────────────────────────
//  SPATIAL  DI
//─────────────────────────────────────────────────────────────────────────────
[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= IMG_W || tid.y >= IMG_H) return;
    gDispatchIdx = tid;

    const uint2  launchIndex = tid.xy;
    const float2 dims        = float2(IMG_W, IMG_H);
    const uint   pixelIdx    = MapPixelID(dims, launchIndex);

    Reservoir_DI rdi = loadReservoirDI(g_Reservoirs_current_di, pixelIdx);

    // Copy compact G-buffer for temporal reuse (must run for all pixels)
    copySampleData(g_sample_last, g_sample_current, pixelIdx);

    // Only do spatial DI when pixel is not an emitter
    if (load_isEmitter(g_sample_current, pixelIdx))
    {
        storeReservoirDI(g_Reservoirs_last_di, pixelIdx, rdi);
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

    // If spatial DI is disabled, compute canonical output and store reservoir unchanged
    if (!(rs_flags & 4u))
    {
        const float3 cameraPos = InitOrigin();
        SurfaceVertex sv_c = BuildVertex(myInstID, myPrimID, myBary, cameraPos);
        sv_c.etai = load_etai(g_sample_current, pixelIdx);
        sv_c.etat = load_etat(g_sample_current, pixelIdx);
        const float vis = (rdi.W_di > 0.0f) ? 1.0f : 0.0f;
        float3 c = ReconnectDI_v2(sv_c, rdi.x2_di, rdi.n2_di, rdi.L2_di, rdi.objID_di) * vis;
        gScratchPing[uint3(tid.xy, 1)] = float4(c * rdi.W_di, 0);
        storeReservoirDI(g_Reservoirs_last_di, pixelIdx, rdi);
        return;
    }

    // RNG
    uint2 seed = GetSeed(pixelIdx, time, 2);

    // Budgeting
    const float conf = min(60.0f, rdi.M_di) / max(1u, rs_tempMcapDI);

    const uint nbrBudget =
        min(rs_spatCountMinDI, SPAT_COUNT_MAX_DI) +
        uint((1.0f - conf) * float(min(rs_spatCountMaxDI, SPAT_COUNT_MAX_DI) -
                                   min(rs_spatCountMinDI, SPAT_COUNT_MAX_DI)) + 0.5f);

    const uint radiusBudget =
        rs_spatRadMinDI +
        uint((1.0f - conf) * float(rs_spatRadMaxDI - rs_spatRadMinDI) + 0.5f);

    // Neighbor ID list (compact: [0..validCount-1] valid)
    uint nIds[SPAT_COUNT_MAX_DI];

    [loop]
    for (uint i = 0; i < SPAT_COUNT_MAX_DI; ++i)
        nIds[i] = 0xFFFFFFFFu;

    uint  validCount = 0;
    float M_sum      = 0.0f;

    //─────────────────────────────────────────────────────────────────────────
    // Candidate selection: compact list, cheap checks before reservoir load
    //─────────────────────────────────────────────────────────────────────────
    [loop]
    for (uint i = 0; i < nbrBudget; ++i)
    {
        uint chosen = 0xFFFFFFFFu;

        [loop]
        for (uint j = 0; j < SPAT_TRIS_DI; ++j)
        {
            const uint iID = GetRandomPixelCircleWeighted(
                radiusBudget, dims.x, dims.y,
                launchIndex.x, launchIndex.y, seed);

            // Cheap sample-data checks first, before loading the reservoir
            bool ok = false;
            {
                if (!load_isEmitter(g_sample_current, iID))
                {
                    uint nInstID_t = load_instID(g_sample_current, iID);
                    uint nPrimID_t = load_primID(g_sample_current, iID);
                    if (GetMatIDFast(nInstID_t, nPrimID_t) == myMatID)
                    {
                        const float3 n1g_r = load_n1_g_with_instID(g_sample_current, iID, nInstID_t);
                        if (!RejectNormal_DI(myN1g, n1g_r, 0.9f))
                        {
                            float2 nBary_t = load_bary(g_sample_current, iID);
                            const float3 x1_r = ReconstructPosition(nInstID_t, nPrimID_t, nBary_t);
                            if (!RejectDistance_DI(myPos, x1_r, myN1g, 0.05f))
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
                Reservoir_DI rdi_r = loadReservoirDI(g_Reservoirs_current_di, iID);

                if (IsValidReservoir_DI_opt(rdi_r.n2_di, rdi_r.M_di))
                {
                    chosen = iID;
                    M_sum += min(SPAT_MCAP_DI, rdi_r.M_di);
                    break;
                }
            }
        }

        if (chosen != 0xFFFFFFFFu)
            nIds[validCount++] = chosen;
    }

    // Canonical M cap + include in M_sum
    const float M_c = min(SPAT_MCAP_DI, rdi.M_di);
    rdi.M_di = M_c;
    M_sum += M_c;

    //─────────────────────────────────────────────────────────────────────────
    // Canonical contribution
    //─────────────────────────────────────────────────────────────────────────
    const float3 cameraPos2 = InitOrigin();
    SurfaceVertex sv_c = BuildVertex(myInstID, myPrimID, myBary, cameraPos2);
    sv_c.etai = load_etai(g_sample_current, pixelIdx);
    sv_c.etat = load_etat(g_sample_current, pixelIdx);

    const float visReuse = (rdi.W_di > 0.0f) ? 1.0f : 0.0f;
    float3 contrib_c = ReconnectDI_v2(sv_c, rdi.x2_di, rdi.n2_di, rdi.L2_di, rdi.objID_di) * visReuse;
    float  p_c           = GetPHat(contrib_c);
    float3 contrib_final = contrib_c;

    // MIS for canonical
    float mis_c = PairwiseMIS_Canonical_Spat_DI(M_sum, p_c, M_c, nIds,
                                                sv_c.x, rdi.x2_di, rdi.n2_di, rdi.L2_di, rdi.objID_di);

    // Adjust canonical weight
    rdi.w_sum_di = mis_c * p_c * rdi.W_di;

    //─────────────────────────────────────────────────────────────────────────
    // Merge neighbors (tight per-iteration scopes)
    //─────────────────────────────────────────────────────────────────────────
    [loop]
    for (uint k = 0; k < validCount; ++k)
    {
        const uint nID = nIds[k];

        // Load neighbor reservoir only for this iteration
        Reservoir_DI rdi_r = loadReservoirDI(g_Reservoirs_current_di, nID);

        // Reconnect neighbor sample at canonical position
        float3 contrib_n = ReconnectDI_v2(sv_c, rdi_r.x2_di, rdi_r.n2_di, rdi_r.L2_di, rdi_r.objID_di);
        // Visibility check
        {
            float3 _vd; float _vt;
            if (rdi_r.objID_di >= 0xFFFFFFFEu) { _vd = normalize(rdi_r.x2_di); _vt = 10000.0f; }
            else { float3 _c = rdi_r.x2_di - sv_c.x; float _d = length(_c); _vd = _c / max(_d, EPSILON); _vt = _d * 0.999f; }
            contrib_n *= IsVisible(sv_c.x, sv_c.n_g, _vd, _vt) ? 1.0f : 0.0f;
        }

        float3 x1_neighbor = ReconstructPosition(load_instID(g_sample_current, nID), load_primID(g_sample_current, nID), load_bary(g_sample_current, nID));
        float p_hat_from = GetPHat(contrib_n * JacobianDeterminantDI(
            x1_neighbor, rdi_r.x2_di, sv_c.x, rdi_r.n2_di, rdi_r.objID_di));

        // MIS weight for neighbor
        float mis_n = PairwiseMIS_Neighbor_Spat_DI(M_sum, M_c, min(SPAT_MCAP_DI, rdi_r.M_di),
                                                   p_c, p_hat_from, nID,
                                                   rdi_r.x2_di, rdi_r.n2_di, rdi_r.L2_di, rdi_r.objID_di);

        const float w_n = mis_n * p_hat_from * rdi_r.W_di;

        // Update reservoir
        if (UpdateReservoirDI(rdi, w_n, min(SPAT_MCAP_DI, rdi_r.M_di),
                              rdi_r.x2_di, rdi_r.n2_di, rdi_r.L2_di, rdi_r.objID_di, seed))
        {
            contrib_final = contrib_n;
        }
    }

    //─────────────────────────────────────────────────────────────────────────
    // Finalize W and output
    //─────────────────────────────────────────────────────────────────────────
    {
        const float p_hat_final = GetPHat(contrib_final);

        if (p_hat_final > EPSILON && rdi.w_sum_di > EPSILON && rdi.w_sum_di < 1e10f)
        {
            float W = rdi.w_sum_di / p_hat_final;
            if (isnan(W) || isinf(W)) W = 0.0f;
            rdi.W_di = W;
        }
        else
        {
            rdi.W_di = 0.0f;
        }

        gScratchPing[uint3(tid.xy, 1)] = float4(contrib_final * rdi.W_di, 0);
    }

    // Store merged reservoir
    storeReservoirDI(g_Reservoirs_last_di, pixelIdx, rdi);
}
