#include "Includes_raygen_v8.hlsli"

//─────────────────────────────────────────────────────────────────────────────
//  TEMPORAL  GI  (raygen shader)
//─────────────────────────────────────────────────────────────────────────────
[shader("raygeneration")]
void Pass_temp_gi_v8()
{
    //─────────────────────────────────────────────────────────────────────────
    // SER: classify thread with minimal live state across reorder
    //─────────────────────────────────────────────────────────────────────────
    uint sortKey;
    {
        const uint2 li  = DispatchRaysIndex().xy;
        const uint  px  = MapPixelID(float2(IMG_W, IMG_H), li);
        const bool  emi = gb_load_isEmitter(g_gbuf_current, px);

        sortKey = emi ? 0u : (!(rs_flags & 2u) ? 1u : 2u);
    }

    dx::MaybeReorderThread(sortKey, 2);

    //─────────────────────────────────────────────────────────────────────────
    // Post-reorder: recompute identity, then heavy loads on coherent warps
    //─────────────────────────────────────────────────────────────────────────
    const uint2  launchIndex = DispatchRaysIndex().xy;
    const float2 dims_f      = float2(IMG_W, IMG_H);
    const uint   pixelIdx    = MapPixelID(dims_f, launchIndex);

    // Emitter early-out
    if (gb_load_isEmitter(g_gbuf_current, pixelIdx))
    {
        gScratchPing[uint3(launchIndex, 5)] = 0;
        return;
    }

    // Load current reservoir (must stay alive until final store)
    Reservoir_GI rdi = loadReservoirGI(g_Reservoirs_current_gi, pixelIdx);

    // Disabled early-out
    if (!(rs_flags & 2u)) {
        gScratchPing[uint3(launchIndex, 5)] = 0;
        storeReservoirGI(g_Reservoirs_current_gi, pixelIdx, rdi);
        return;
    }

    // Keep boilValue as a single scalar that survives until boil filter
    float boilValue = 0.0f;

    // RNG (keep as small as possible; preserve your semantics)
    uint2 seed = GetSeed(pixelIdx, time, 3);
    uint  permSeed = GetSeed(1, time, 3).x;

    // --- base reprojection ---
    // Lightweight loads for reprojection & rejection
    const uint   myInstID = gb_load_instID(g_gbuf_current, pixelIdx);
    const uint   myMatID  = gb_load_matID(g_gbuf_current, pixelIdx);
    const float3 myPos    = gb_load_worldPos(g_gbuf_current, pixelIdx, myInstID);
    const float3 myN1s    = gb_load_normal_world(g_gbuf_current, pixelIdx, myInstID);

    // Specularity (same computation DLSS-RR gets via EnvBRDFApprox2)
    const float3 camPos = InitOrigin();
    const float  NoV = saturate(dot(normalize(camPos - myPos), myN1s));
    const float  myPr = gb_load_Pr(g_gbuf_current, pixelIdx);
    const float  myPm = gb_load_Pm(g_gbuf_current, pixelIdx);
    const float  specularity = Luma(EnvBRDFApprox2(gb_load_Kd(g_gbuf_current, pixelIdx), myPr, myPm, NoV));

    // Stochastic reprojection: randomly choose specular vs diffuse MV
    // weighted by specularity. Over many frames this converges naturally.
    float4 reflData = gScratchPing[uint3(launchIndex, 4)];
    uint   reflInstID = asuint(reflData.w);
    bool   reflValid = (reflInstID < 0xFFFFFFFEu);
    float  rSpec = RandomFloatSingle(seed.x);
    bool   useSpecReproj = (rSpec < specularity) && reflValid;

    int2 baseCoord;
    if (useSpecReproj)
    {
        baseCoord = GetBestReprojectedPixel_d(reflData.xyz, prevView, prevProjection, dims_f, reflInstID);
        if (baseCoord.x == -1)
            baseCoord = GetBestReprojectedPixel_d(myPos, prevView, prevProjection, dims_f, myInstID);
    }
    else
    {
        baseCoord = GetBestReprojectedPixel_d(myPos, prevView, prevProjection, dims_f, myInstID);
    }
    if (baseCoord.x == -1 && baseCoord.y == -1)
        baseCoord = (int2)launchIndex;

    // Debug: slice 6 = specular reprojection (blue=spec, red=diff), slice 7 = specularity
    gOutput[uint3(launchIndex, 6)] = float4(useSpecReproj ? 0.0f : 1.0f, 0.0f, useSpecReproj ? 1.0f : 0.0f, 1.0f);
    gOutput[uint3(launchIndex, 7)] = float4(specularity, specularity, specularity, 1.0f);

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

    // Lightweight neighbor identifiers (survive rejection -> merge)
    uint   rInstID = 0;

    // 1) Try the permuted sample
    if (permInBounds)
        valid = TestTemporalCandidate_GI(permCoord, dims_f, g_gbuf_current, myMatID, myN1s, myPos,
                                         tempPixelIdx, rInstID);

    [branch]
    if (valid)
    {
        Reservoir_GI rdi_r = loadReservoirGI(g_Reservoirs_last_gi, tempPixelIdx);

        // Final validity check that requires reservoir
        if (!IsValidReservoir_GI(rdi_r))
        {
            valid = false;
        }
        if (valid)
        {
            float p_hat_final = 0.0f;

            {
                const float3 cameraPos = InitOrigin();
                float Jnc = 0.0f, Jn = 0.0f;

                const float visReuse_c = (rdi.W_gi > 0.0f) ? 1.0f : 0.0f;
                const float p_c = GetPHat(UnpackRGB9E5(rdi.F_gi)) * visReuse_c;

                // Canonical Jc: jacobian at current pixel's x1 → canonical x2
                const float Jc_canonical = ComputeJc(myPos, rdi.x2_gi, rdi.n2_s_gi);

                float p_n = 0.0f;
                float n_c = 0.0f;
                float3 contrib_n_from_me = 0;

                // p_n: reconnect from neighbor vertex to current GI reservoir sample
                {
                    const float3 rWorldPos = gb_load_worldPos(g_gbuf_current, tempPixelIdx, rInstID);
                    const float3 rN1s      = gb_load_normal_world(g_gbuf_current, tempPixelIdx, rInstID);
                    const float3 rO        = normalize(cameraPos - rWorldPos);
                    const uint   rMatID    = gb_load_matID(g_gbuf_current, tempPixelIdx);

                    float3 rcKd; float rcPr, rcPm;
                    RefetchMaterial(rdi.matID_gi, rdi.uv_gi, rcKd, rcPr, rcPm);

                    float3 c = ReconnectGI(
                        rWorldPos, rN1s, rO, rMatID,
                        gb_load_Kd(g_gbuf_current, tempPixelIdx),
                        gb_load_Pr(g_gbuf_current, tempPixelIdx),
                        gb_load_Pm(g_gbuf_current, tempPixelIdx),
                        1.0,
                        materials[rMatID].Ni,
                        rdi.matID_gi, rdi.x2_gi, rdi.n2_s_gi, rdi.L2_gi, rdi.V2_gi,
                        rcKd, rcPr, rcPm,
                        Jnc);

                    float ph = GetPHat(c);
                    { float3 _conn = rdi.x2_gi - rWorldPos; float _cd = length(_conn);
                      p_n = ph * ((_cd > EPSILON && IsVisible(rWorldPos, rN1s, _conn / _cd, _cd * 0.999f)) ? 1.0f : 0.0f); }
                }

                // n_c: reconnect from current vertex to neighbor GI reservoir sample
                float J2;
                {
                    const float3 myO = normalize(cameraPos - myPos);

                    // Neighbor Jc: jacobian at neighbor's x1 → neighbor's x2
                    const float Jc_neighbor = ComputeJc(
                        gb_load_worldPos(g_gbuf_current, tempPixelIdx, rInstID),
                        rdi_r.x2_gi, rdi_r.n2_s_gi);

                    float3 rrKd; float rrPr, rrPm;
                    RefetchMaterial(rdi_r.matID_gi, rdi_r.uv_gi, rrKd, rrPr, rrPm);

                    float3 c = ReconnectGI(
                        myPos, myN1s, myO, myMatID,
                        gb_load_Kd(g_gbuf_current, pixelIdx),
                        gb_load_Pr(g_gbuf_current, pixelIdx),
                        gb_load_Pm(g_gbuf_current, pixelIdx),
                        1.0,
                        materials[myMatID].Ni,
                        rdi_r.matID_gi, rdi_r.x2_gi, rdi_r.n2_s_gi, rdi_r.L2_gi, rdi_r.V2_gi,
                        rrKd, rrPr, rrPm,
                        Jn);

                    J2 = JacobianRatio(Jn, Jc_neighbor);
                    float ph = GetPHat(c);
                    { float3 _conn = rdi_r.x2_gi - myPos; float _cd = length(_conn);
                      float vis_n = (_cd > EPSILON && IsVisible(myPos, myN1s, _conn / _cd, _cd * 0.999f)) ? 1.0f : 0.0f;
                      n_c = ph * vis_n;
                      contrib_n_from_me = c * vis_n; }
                }

                const float visReuse_n = (rdi_r.W_gi > 0.0f) ? 1.0f : 0.0f;
                const float n_n = GetPHat(UnpackRGB9E5(rdi_r.F_gi)) * visReuse_n;

                // Dynamic M caps
                float sdata_Pr = gb_load_Pr(g_gbuf_current, pixelIdx);
                float rdi_r_Pr = EvaluatePBRProperties(materials[rdi_r.matID_gi], rdi_r.uv_gi, 0).x;
                const float minRoughTemp  = min(sdata_Pr, rdi_r_Pr);
                const float tempMcapScale = smoothstep(rs_reuseRoughnessMin, rs_reuseRoughnessMax, minRoughTemp);
                const float dynTempMcap   = (minRoughTemp <= rs_reuseRoughnessMin) ? 0.0f
                                        : min(rs_tempMcapGI, rs_tempMcapGI * tempMcapScale);

                const float M_c   = min(rs_tempMcapGI, rdi.M_gi);
                const float M_n   = min(dynTempMcap,  rdi_r.M_gi);
                const float M_sum = M_c + M_n;

                const float p_nJ1  = p_n * JacobianRatio(Jnc, Jc_canonical);
                const float n_cJ2  = n_c * J2;

                const float mis_c = PairwiseMIS_Canonical_Temp(M_c, M_n, p_c, p_nJ1, M_sum);
                const float mis_n = PairwiseMIS_Neighbour_Temp(M_c, M_n, n_cJ2, n_n, M_sum);

                const float w_c = mis_c * p_c * rdi.W_gi;
                const float w_n = mis_n * n_cJ2 * rdi_r.W_gi;

                rdi.w_sum_gi = w_c;

                uint F_gi_winner = rdi.F_gi;
                p_hat_final = p_c;
                if (UpdateReservoirGI(
                        rdi,
                        w_n,
                        rdi_r.M_gi,
                        rdi_r.x2_gi, rdi_r.n2_s_gi, rdi_r.L2_gi, rdi_r.V2_gi,
                        rdi_r.uv_gi,
                        rdi_r.matID_gi, rdi_r.objID_gi,
                        rdi_r.F_gi,
                        seed
                    ))
                {
                    p_hat_final = n_c;
                    F_gi_winner = PackRGB9E5(contrib_n_from_me);
                }

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

                rdi.F_gi = F_gi_winner;
            }
        }
    }

    // Write boilValue to scratch for the groupshared boiling post-pass
    gScratchPing[uint3(launchIndex, 5)] = float4(boilValue, 0, 0, 0);

    // Store merged reservoir
    storeReservoirGI(g_Reservoirs_current_gi, pixelIdx, rdi);
}