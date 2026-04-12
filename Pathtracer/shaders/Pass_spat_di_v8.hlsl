#define COMPUTE_PASS
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

    // Copy SoA G-buffer current → last for temporal reuse (must run for all pixels)
    gb_copy(g_sample_last, g_gbuf_current, pixelIdx);

    // Only do spatial DI when pixel is not an emitter
    if (gb_load_isEmitter(g_gbuf_current, pixelIdx))
    {
        storeReservoirDI(g_Reservoirs_last_di, pixelIdx, rdi);
        return;
    }

    // Lightweight loads — only worldPos kept in register
    const uint   myInstID = gb_load_instID(g_gbuf_current, pixelIdx);
    const uint   myMatID  = gb_load_matID(g_gbuf_current, pixelIdx);
    const float3 myPos    = gb_load_worldPos(g_gbuf_current, pixelIdx, myInstID);
    const float3 myN1s    = gb_load_normal_world(g_gbuf_current, pixelIdx, myInstID);

    // If spatial DI is disabled, compute canonical output and store reservoir unchanged
    if (!(rs_flags & 4u))
    {
        const float3 cameraPos = InitOrigin();
        const float3 viewDir   = normalize(cameraPos - myPos);
        const float vis = (rdi.W_di > 0.0f) ? 1.0f : 0.0f;
        float3 c = ReconnectDI(myPos, myN1s, viewDir, myMatID,
                               rdi.x2_di, rdi.n2_di, rdi.L2_di,
                               gb_load_Kd(g_gbuf_current, pixelIdx),
                               gb_load_Pr(g_gbuf_current, pixelIdx),
                               gb_load_Pm(g_gbuf_current, pixelIdx),
                               1.0,
                               materials[myMatID].Ni,
                               rdi.objID_di) * vis;
        gScratchPing[uint3(tid.xy, 1)] = float4(c * rdi.W_di, 0);
        storeReservoirDI(g_Reservoirs_last_di, pixelIdx, rdi);
        return;
    }

    // RNG
    uint2 seed = GetSeed(pixelIdx, time, 2);

    // Single neighbor selection: rs_spatTriesDI attempts, radius shrinks linearly.
    uint  nIds[SPAT_COUNT_MAX_DI];
    nIds[0] = 0xFFFFFFFFu;

    uint  validCount = 0;
    float M_sum      = 0.0f;

    {
        const uint totalTries = max(1u, rs_spatTriesDI);

        [loop]
        for (uint attempt = 0; attempt < totalTries; ++attempt)
        {
            float t = (totalTries > 1) ? float(attempt) / float(totalTries - 1) : 0.0f;
            uint  radius = (uint)lerp(float(rs_spatRadMaxDI), float(rs_spatRadMinDI), t);

            const uint iID = GetRandomPixelCircleWeighted(
                radius, dims.x, dims.y,
                launchIndex.x, launchIndex.y, seed);

            bool ok = false;
            if (!gb_load_isEmitter(g_gbuf_current, iID))
            {
                uint nInstID_t = gb_load_instID(g_gbuf_current, iID);
                if (gb_load_matID(g_gbuf_current, iID) == myMatID)
                {
                    const float3 n1s_r = gb_load_normal_world(g_gbuf_current, iID, nInstID_t);
                    if (!RejectNormal_DI(myN1s, n1s_r, 0.9f))
                    {
                        const float3 x1_r = gb_load_worldPos(g_gbuf_current, iID, nInstID_t);
                        if (!RejectDistance_DI(myPos, x1_r, myN1s, 0.05f))
                            ok = true;
                    }
                }
            }

            if (ok)
            {
                Reservoir_DI rdi_r = loadReservoirDI(g_Reservoirs_current_di, iID);
                if (IsValidReservoir_DI_opt(rdi_r.n2_di, rdi_r.M_di))
                {
                    nIds[0] = iID;
                    validCount = 1;
                    M_sum = min(SPAT_MCAP_DI, rdi_r.M_di);
                    break;
                }
            }
        }
    }

    // Canonical M cap + include in M_sum
    const float M_c = min(SPAT_MCAP_DI, rdi.M_di);
    rdi.M_di = M_c;
    M_sum += M_c;

    //─────────────────────────────────────────────────────────────────────────
    // Canonical contribution
    //─────────────────────────────────────────────────────────────────────────
    const float3 cameraPos2 = InitOrigin();
    const float3 viewDir_c  = normalize(cameraPos2 - myPos);

    const float visReuse = (rdi.W_di > 0.0f) ? 1.0f : 0.0f;
    float3 contrib_c = ReconnectDI(myPos, myN1s, viewDir_c, myMatID,
                                   rdi.x2_di, rdi.n2_di, rdi.L2_di,
                                   gb_load_Kd(g_gbuf_current, pixelIdx),
                                   gb_load_Pr(g_gbuf_current, pixelIdx),
                                   gb_load_Pm(g_gbuf_current, pixelIdx),
                                   1.0,
                                   materials[myMatID].Ni,
                                   rdi.objID_di) * visReuse;
    float  p_c           = GetPHat(contrib_c);
    float3 contrib_final = contrib_c;

    // MIS for canonical
    float mis_c = PairwiseMIS_Canonical_Spat_DI(M_sum, p_c, M_c, nIds,
                                                myPos, rdi.x2_di, rdi.n2_di, rdi.L2_di, rdi.objID_di);

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
        float3 contrib_n = ReconnectDI(myPos, myN1s, viewDir_c, myMatID,
                                       rdi_r.x2_di, rdi_r.n2_di, rdi_r.L2_di,
                                       gb_load_Kd(g_gbuf_current, pixelIdx),
                                       gb_load_Pr(g_gbuf_current, pixelIdx),
                                       gb_load_Pm(g_gbuf_current, pixelIdx),
                                       1.0,
                                       materials[myMatID].Ni,
                                       rdi_r.objID_di);
        // Visibility check
        {
            float3 _vd; float _vt;
            if (rdi_r.objID_di >= 0xFFFFFFFEu) { _vd = normalize(rdi_r.x2_di); _vt = 10000.0f; }
            else { float3 _c = rdi_r.x2_di - myPos; float _d = length(_c); _vd = _c / max(_d, EPSILON); _vt = _d * 0.999f; }
            contrib_n *= IsVisible(myPos, myN1s, _vd, _vt) ? 1.0f : 0.0f;
        }

        // Load neighbor world position for Jacobian
        uint   nInstID = gb_load_instID(g_gbuf_current, nID);
        float3 x1_neighbor = gb_load_worldPos(g_gbuf_current, nID, nInstID);
        float p_hat_from = GetPHat(contrib_n * JacobianDeterminantDI(
            x1_neighbor, rdi_r.x2_di, myPos, rdi_r.n2_di, rdi_r.objID_di));

        // Inlined PairwiseMIS_Neighbor_Spat_DI — uses G-buffer loads
        float mis_n;
        {
            float M_sum_safe = max(M_sum, 1.0f);
            float M_n = min(SPAT_MCAP_DI, rdi_r.M_di);
            float visReuse_n = load_W_di(g_Reservoirs_current_di, nID) > 0.0f ? 1.0f : 0.0f;
            float3 nPos    = x1_neighbor;
            float3 nN1s    = gb_load_normal_world(g_gbuf_current, nID, nInstID);
            float3 nViewDir = normalize(cameraPos2 - nPos);
            uint   nMatID  = gb_load_matID(g_gbuf_current, nID);
            float p_n = visReuse_n * GetPHat(ReconnectDI(nPos, nN1s, nViewDir, nMatID,
                rdi_r.x2_di, rdi_r.n2_di, rdi_r.L2_di,
                gb_load_Kd(g_gbuf_current, nID),
                gb_load_Pr(g_gbuf_current, nID),
                gb_load_Pm(g_gbuf_current, nID),
                1.0,
                materials[nMatID].Ni,
                rdi_r.objID_di));
            float m_num = (M_sum_safe - M_c) * p_n;
            float m_den = m_num + M_c * p_hat_from;
            mis_n = (m_den > 1e-4) ? (M_n / M_sum_safe) * (m_num / m_den) : 0.0f;
        }

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
