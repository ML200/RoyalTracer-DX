#include "Includes_raygen_v8.hlsli"

//─────────────────────────────────────────────────────────────────────────────
//  TEMPORAL  GI  (raygen shader)
//─────────────────────────────────────────────────────────────────────────────
[shader("raygeneration")]
void Pass_temp_gi_v8()
{
    const uint2 launchIndex = DispatchRaysIndex().xy;
    const float2 dims_f = float2(IMG_W, IMG_H);
    const uint pixelIdx = MapPixelID(dims_f, launchIndex);

    // ── Cheap early outs (no reorder for these — they exit immediately) ──
    if (load_isEmitter(g_sample_current, pixelIdx))
    {
        gScratchPing[uint3(launchIndex, 5)] = 0;
        return;
    }

    if (!(rs_flags & 2u))
    {
        // Temporal disabled: copy reservoir through, no shift work to coalesce.
        Reservoir_GI rdi_pass = loadReservoirGI(g_Reservoirs_current_gi, pixelIdx);
        gScratchPing[uint3(launchIndex, 5)] = 0;
        storeReservoirGI(g_Reservoirs_current_gi, pixelIdx, rdi_pass);
        return;
    }

    // ── SER reorder seed: load only what's needed to estimate shift work ──
    // Canonical reservoir's expected replay length (small loads only).
    const uint canonMethod = load_method_gi(g_Reservoirs_current_gi, pixelIdx);
    const uint canonK      = load_k_gi    (g_Reservoirs_current_gi, pixelIdx);

    // Reproject with the diffuse motion vector — needed for the hint and reused
    // by the actual reprojection path below (the spec path also falls back to it).
    const uint   myInstID = load_instID(g_sample_current, pixelIdx);
    const uint   myPrimID = load_primID(g_sample_current, pixelIdx);
    const float2 myBary   = load_bary(g_sample_current, pixelIdx);
    const float3 myPos    = ReconstructPosition(myInstID, myPrimID, myBary);

    const int2 diffReproj = GetBestReprojectedPixel_d(myPos, prevView, prevProjection, dims_f, myInstID);

    uint reprojMethod = 0u;
    uint reprojK      = 0u;
    if (diffReproj.x >= 0)
    {
        const uint reprojIdx = MapPixelID(dims_f, uint2(diffReproj));
        reprojMethod = load_method_gi(g_Reservoirs_last_gi, reprojIdx);
        reprojK      = load_k_gi    (g_Reservoirs_last_gi, reprojIdx);
    }

    // Pack: (canonical replay bounces : 4 bits) | (reproj replay bounces : 4 bits).
    // Threads with the same (canonK, reprojK) replay-length pair end up in the
    // same SER bucket → matched control flow through ShiftGI's replay loop.
    {
        const uint canRb = min(ShiftReplayBounceEstimate(canonMethod, canonK), 15u);
        const uint rprRb = min(ShiftReplayBounceEstimate(reprojMethod, reprojK), 15u);
        dx::MaybeReorderThread((canRb << 4) | rprRb, 8u);
    }

    // ── Heavy loads happen AFTER reorder so they land on coherent threads ──
    Reservoir_GI rdi = loadReservoirGI(g_Reservoirs_current_gi, pixelIdx);

    // Keep boilValue as a single scalar that survives until boil filter
    float boilValue = 0.0f;

    // RNG (keep as small as possible; preserve your semantics)
    uint2 seed = GetSeed(pixelIdx, time, 3);
    uint  permSeed = GetSeed(1, time, 3).x;

    // (Already loaded for the reorder hint: myInstID/myPrimID/myBary/myPos)
    const uint   myMatID  = GetMatIDFast(myInstID, myPrimID);
    const float3 myN1s    = load_n1_s_with_instID(g_sample_current, pixelIdx, myInstID);
    const float2 myUV     = load_uv(g_sample_current, pixelIdx);
    float3 myKd; float myPr, myPm;
    RefetchMaterial(myMatID, myUV, myKd, myPr, myPm);

    // Specularity (same computation DLSS-RR gets via EnvBRDFApprox2)
    const float3 camPos = InitOrigin();
    const float  NoV = saturate(dot(normalize(camPos - myPos), myN1s));
    const float  specularity = Luma(EnvBRDFApprox2(myKd, myPr, myPm, NoV));

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
            baseCoord = diffReproj;   // reuse the one already computed for the SER hint
    }
    else
    {
        baseCoord = diffReproj;
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
    uint   rPrimID = 0;
    float2 rBary   = float2(0, 0);

    // 1) Try the permuted sample
    if (permInBounds)
        valid = TestTemporalCandidate_GI(permCoord, dims_f, g_sample_last, myMatID, myN1s, myPos,
                                         tempPixelIdx, rInstID, rPrimID, rBary);

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
                float J1  = 1.0f, J2 = 1.0f;

                const float visReuse_c = (rdi.W_gi > 0.0f) ? 1.0f : 0.0f;
                const float p_c = GetPHat(UnpackRGB9E5(rdi.F_gi)) * visReuse_c;

                float p_n = 0.0f;
                float n_c = 0.0f;
                float3 contrib_n_from_me = 0;

                // p_n: shift CANONICAL's sample into the neighbor's domain (sv_r is neighbor's x_1).
                // Uses the canonical reservoir's method/k/seed.
                {
                    SurfaceVertex sv_r = BuildVertex(rInstID, rPrimID, rBary, cameraPos);
                    sv_r.etai = load_etai(g_sample_last, tempPixelIdx);
                    sv_r.etat = load_etat(g_sample_last, tempPixelIdx);

                    float3 rcKd; float rcPr, rcPm;
                    RefetchMaterial(rdi.matID_gi, rdi.uv_gi, rcKd, rcPr, rcPm);

                    float3 c = ShiftGI(
                        sv_r,
                        rdi.method_gi, rdi.k_gi, rdi.seed_gi,
                        rdi.x2_gi, rdi.n2_s_gi, rdi.n2_g_gi,
                        rdi.matID_gi, rdi.objID_gi, rdi.uv_gi,
                        rdi.etai_gi, rdi.etat_gi,
                        rcKd, rcPr, rcPm,
                        rdi.L2_gi, rdi.V2_gi, rdi.J_gi.y,
                        rdi.J_gi.x,   // pdfx2_atrc (= stored lightPdf for *_ATRC, 0 otherwise)
                        Jnc, J1);
                    p_n = GetPHat(c);
                }

                // n_c: shift NEIGHBOR's sample into the canonical domain (sv_c is canonical's x_1).
                // Uses the neighbor reservoir's method/k/seed.
                {
                    SurfaceVertex sv_c = BuildVertex(myInstID, myPrimID, myBary, cameraPos);
                    sv_c.etai = load_etai(g_sample_current, pixelIdx);
                    sv_c.etat = load_etat(g_sample_current, pixelIdx);

                    float3 rrKd; float rrPr, rrPm;
                    RefetchMaterial(rdi_r.matID_gi, rdi_r.uv_gi, rrKd, rrPr, rrPm);

                    float3 c = ShiftGI(
                        sv_c,
                        rdi_r.method_gi, rdi_r.k_gi, rdi_r.seed_gi,
                        rdi_r.x2_gi, rdi_r.n2_s_gi, rdi_r.n2_g_gi,
                        rdi_r.matID_gi, rdi_r.objID_gi, rdi_r.uv_gi,
                        rdi_r.etai_gi, rdi_r.etat_gi,
                        rrKd, rrPr, rrPm,
                        rdi_r.L2_gi, rdi_r.V2_gi, rdi_r.J_gi.y,
                        rdi_r.J_gi.x,   // pdfx2_atrc (= stored lightPdf for *_ATRC, 0 otherwise)
                        Jn, J2);
                    n_c = GetPHat(c);
                    contrib_n_from_me = c;
                }

                const float visReuse_n = (rdi_r.W_gi > 0.0f) ? 1.0f : 0.0f;
                const float n_n = GetPHat(UnpackRGB9E5(rdi_r.F_gi)) * visReuse_n;

                // Dynamic M caps
                float sdata_Pr = myPr;
                float rdi_r_Pr = EvaluatePBRProperties(materials[rdi_r.matID_gi], rdi_r.uv_gi, 0).x;
                const float minRoughTemp  = min(sdata_Pr, rdi_r_Pr);
                const float tempMcapScale = smoothstep(rs_reuseRoughnessMin, rs_reuseRoughnessMax, minRoughTemp);
                const float dynTempMcap   = rs_tempMcapGI;//(minRoughTemp <= rs_reuseRoughnessMin) ? 0.0f
                                        //: min(rs_tempMcapGI, rs_tempMcapGI * tempMcapScale);

                const float M_c   = min(rs_tempMcapGI, rdi.M_gi);
                const float M_n   = min(dynTempMcap,  rdi_r.M_gi);
                const float M_sum = M_c + M_n;

                const float p_nJ1  = p_n * J1;
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
                        rdi_r.x2_gi, rdi_r.n2_s_gi, rdi_r.n2_g_gi, rdi_r.L2_gi, rdi_r.V2_gi,
                        rdi_r.uv_gi,
                        rdi_r.etai_gi, rdi_r.etat_gi,
                        rdi_r.matID_gi, rdi_r.objID_gi,
                        rdi_r.J_gi,
                        rdi_r.F_gi,
                        rdi_r.seed_gi, rdi_r.k_gi, rdi_r.method_gi,
                        seed
                    ))
                {
                    p_hat_final = n_c;
                    rdi.J_gi.y  = Jn;
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