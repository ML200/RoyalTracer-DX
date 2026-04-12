#include "Includes_raygen_v8.hlsli"

//─────────────────────────────────────────────────────────────────────────────
//  SPATIAL  GI  (raygen shader — merge only, neighbor selection done in pre-pass)
//─────────────────────────────────────────────────────────────────────────────

// Must match layout in Pass_spat_gi_select_v8.hlsl
static const uint GI_SEL_STRIDE = 40u;
uint gi_sel_addr(uint linearIdx) { return linearIdx * GI_SEL_STRIDE; }

[shader("raygeneration")]
void Pass_spat_gi_v8_1()
{
    //─────────────────────────────────────────────────────────────────────────
    // SER: classify thread with minimal live state across reorder
    //─────────────────────────────────────────────────────────────────────────
    uint sortKey;
    {
        const uint2 li  = DispatchRaysIndex().xy;
        const uint  px  = MapPixelID(float2(IMG_W, IMG_H), li);
        const bool  emi = gb_load_isEmitter(g_gbuf_current, px);

        if (emi || !(rs_flags & 8u))
        {
            sortKey = emi ? 0u : 1u;
        }
        else
        {
            const uint selBase = gi_sel_addr(li.y * IMG_W + li.x);
            sortKey = 2u + g_pathStateBuffer.Load(selBase); // validCount
        }
    }

    dx::MaybeReorderThread(sortKey, 3);

    //─────────────────────────────────────────────────────────────────────────
    // Post-reorder: recompute identity, then heavy loads on coherent warps
    //─────────────────────────────────────────────────────────────────────────
    const uint2  launchIndex = DispatchRaysIndex().xy;
    const float2 dims        = float2(IMG_W, IMG_H);
    const uint   pixelIdx    = MapPixelID(dims, launchIndex);

    Reservoir_GI rdi = loadReservoirGI(g_Reservoirs_current_gi, pixelIdx);

    // Emitter early-out
    if (gb_load_isEmitter(g_gbuf_current, pixelIdx))
    {
        storeReservoirGI(g_Reservoirs_last_gi, pixelIdx, rdi);
        return;
    }

    // Disabled early-out
    if (!(rs_flags & 8u))
    {
        float3 c = UnpackRGB9E5(rdi.F_gi);
        float  W = (rdi.W_gi > 0.0f) ? rdi.W_gi : 0.0f;
        gScratchPing[uint3(launchIndex, 2)] = float4(c * W, 0);
        storeReservoirGI(g_Reservoirs_last_gi, pixelIdx, rdi);
        return;
    }

    // Re-read neighbor selection (cheap buffer loads)
    const uint linearIdx = launchIndex.y * IMG_W + launchIndex.x;
    const uint selBase   = gi_sel_addr(linearIdx);
    uint2 header         = g_pathStateBuffer.Load2(selBase);
    const uint  validCount = header.x;
    const float M_sum_nbr  = asfloat(header.y);

    uint nIds[SPAT_COUNT_MAX_GI];
    [unroll]
    for (uint i = 0; i < SPAT_COUNT_MAX_GI; ++i)
        nIds[i] = g_pathStateBuffer.Load(selBase + 8u + i * 4u);

    // Lightweight loads
    const uint   myInstID = gb_load_instID(g_gbuf_current, pixelIdx);
    const uint   myMatID  = gb_load_matID(g_gbuf_current, pixelIdx);

    // Canonical M cap + include in M_sum
    const float M_c = min(SPAT_MCAP_GI, rdi.M_gi);
    rdi.M_gi = M_c;
    const float M_sum = M_sum_nbr + M_c;

    // RNG (for reservoir update only — neighbor selection already consumed its own)
    uint2 seed = GetSeed(pixelIdx, time, 2);

    //─────────────────────────────────────────────────────────────────────────
    // Canonical contribution
    //─────────────────────────────────────────────────────────────────────────
    const float visReuse = (rdi.W_gi > 0.0f) ? 1.0f : 0.0f;

    float3 contrib_final = 0.0.xxx;
    float  p_c           = 0.0f;

    // Build canonical vertex (needed for MIS and neighbor evaluation)
    const float3 cameraPos2 = InitOrigin();
    const float3 myPos  = gb_load_worldPos(g_gbuf_current, pixelIdx, myInstID);
    const float3 myN1s  = gb_load_normal_world(g_gbuf_current, pixelIdx, myInstID);
    const float3 viewDir = normalize(cameraPos2 - myPos);

    // Build canonical x2 vertex from reservoir (needed for MIS canonical)
    float3 rKd; float rPr, rPm;
    RefetchMaterial(rdi.matID_gi, rdi.uv_gi, rKd, rPr, rPm);

    // Use stored F_gi directly — skip canonical ReconnectGI
    {
        float3 contrib_c = UnpackRGB9E5(rdi.F_gi) * visReuse;
        p_c = GetPHat(contrib_c);
        contrib_final = contrib_c;
    }

    // Canonical Jc: jacobian at current pixel's x1 → canonical x2
    const float Jc_canonical = ComputeJc(myPos, rdi.x2_gi, rdi.n2_s_gi);

    // MIS for canonical
    const float mis_c = PairwiseMIS_Canonical_Spat_GI(
        M_sum, p_c, M_c, nIds,
        rdi.x2_gi, rdi.n2_s_gi, rdi.L2_gi, rdi.V2_gi, rdi.matID_gi,
        rKd, rPr, rPm,
        Jc_canonical
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

        // Load neighbor reservoir only for this iteration
        Reservoir_GI rdi_r = loadReservoirGI(g_Reservoirs_current_gi, nID);

        // Neighbor Jc: jacobian at neighbor's x1 → neighbor's x2
        const float Jc_neighbor = ComputeJc(
            gb_load_worldPos(g_gbuf_current, nID, gb_load_instID(g_gbuf_current, nID)),
            rdi_r.x2_gi, rdi_r.n2_s_gi);

        // Compute neighbor reconnection contribution
        float3 contrib_n = 0.0.xxx;
        float  p_hat_from = 0.0f;
        float  Jn = 0.0f;

        {
            // Refetch neighbor x2 material
            float3 rnKd; float rnPr, rnPm;
            RefetchMaterial(rdi_r.matID_gi, rdi_r.uv_gi, rnKd, rnPr, rnPm);

            contrib_n = ReconnectGI(
                myPos, myN1s, viewDir, myMatID,
                gb_load_Kd(g_gbuf_current, pixelIdx),
                gb_load_Pr(g_gbuf_current, pixelIdx),
                gb_load_Pm(g_gbuf_current, pixelIdx),
                1.0,
                materials[myMatID].Ni,
                rdi_r.matID_gi, rdi_r.x2_gi, rdi_r.n2_s_gi, rdi_r.L2_gi, rdi_r.V2_gi,
                rnKd, rnPr, rnPm,
                Jn);

            // Visibility after reconnection
            {
                float3 _conn = rdi_r.x2_gi - myPos; float _cd = length(_conn);
                float vis = (_cd > EPSILON && IsVisible(myPos, myN1s, _conn / _cd, _cd * 0.999f)) ? 1.0f : 0.0f;
                contrib_n *= vis;
            }

            p_hat_from = GetPHat(contrib_n) * JacobianRatio(Jn, Jc_neighbor);
        }

        // MIS weight for neighbor
        const float Mn = min(SPAT_MCAP_GI, rdi_r.M_gi);
        const float mis_n = PairwiseMIS_Neighbor_Spat_GI(
            M_sum,
            M_c, Mn,
            p_hat_from,
            rdi_r.W_gi, GetPHat(UnpackRGB9E5(rdi_r.F_gi))
        );

        const float w_n = mis_n * p_hat_from * rdi_r.W_gi;

        // Update reservoir
        if (UpdateReservoirGI(
                rdi,
                w_n,
                min(SPAT_MCAP_GI, rdi_r.M_gi),
                rdi_r.x2_gi, rdi_r.n2_s_gi, rdi_r.L2_gi, rdi_r.V2_gi,
                rdi_r.uv_gi,
                rdi_r.matID_gi, rdi_r.objID_gi,
                rdi_r.F_gi,
                seed
            ))
        {
            contrib_final = contrib_n;
        }
    }

    //─────────────────────────────────────────────────────────────────────────
    // Finalize W/F and output
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

        rdi.F_gi = PackRGB9E5(contrib_final);

        gScratchPing[uint3(launchIndex, 2)] = float4(contrib_final * rdi.W_gi, 0);
    }

    // Store merged reservoir
    storeReservoirGI(g_Reservoirs_last_gi, pixelIdx, rdi);
}
