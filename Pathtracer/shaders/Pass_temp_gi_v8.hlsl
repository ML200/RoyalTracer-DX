#include "Includes_v8.hlsli"

//  TEMPORAL  GI
[shader("raygeneration")]
void Pass_temp_gi_v8()
{
    uint sortKey;
    {
        const uint2 li  = DispatchRaysIndex().xy;
        const uint  px  = MapPixelID(float2(IMG_W, IMG_H), li);
        const bool  emi = load_isEmitter(g_sample_current, px);

        sortKey = emi ? 0u : (!(rs_flags & 2u) ? 1u : 2u);
    }

    dx::MaybeReorderThread(sortKey, 2);

    const uint2  launchIndex = DispatchRaysIndex().xy;
    const float2 dims_f      = float2(IMG_W, IMG_H);
    const uint   pixelIdx    = MapPixelID(dims_f, launchIndex);

    // Emitter early-out
    if (load_isEmitter(g_sample_current, pixelIdx))
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

    float boilValue = 0.0f;
    uint2 seed = GetSeed(pixelIdx, time, 3);
    uint  permSeed = GetSeed(1, time, 3).x;

    //base reprojection
    //Lightweight loads
    const uint   myInstID = load_instID(g_sample_current, pixelIdx);
    const uint   myPrimID = load_primID(g_sample_current, pixelIdx);
    const float2 myBary   = load_bary(g_sample_current, pixelIdx);
    const uint   myMatID  = GetMatIDFast(myInstID, myPrimID);
    const float3 myPos    = ReconstructPosition(myInstID, myPrimID, myBary);
    const float3 myN1s    = load_n1_s_with_instID(g_sample_current, pixelIdx, myInstID);
    const float2 myUV = load_uv(g_sample_current, pixelIdx);
    float3 myKd; float myPr, myPm;
    RefetchMaterial(myMatID, myUV, myKd, myPr, myPm);

    //Specularity (same computation DLSS-RR gets via EnvBRDFApprox2)
    const float3 camPos = InitOrigin();
    const float  NoV = saturate(dot(normalize(camPos - myPos), myN1s));
    const float  specularity = Luma(EnvBRDFApprox2(myKd, myPr, myPm, NoV));

    //Stochastic reprojection: randomly choose specular vs diffuse MV
    //weighted by specularity
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

    //permuted candidate
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

    //Lightweight neighbor identifiers (survive rejection -> merge)
    uint   rInstID = 0;
    uint   rPrimID = 0;
    float2 rBary   = float2(0, 0);

    //Candidate search ladder: permuted -> base reprojection -> 3x3 around base.
    //Each step is only attempted if the prior one didn't find a valid sample.

    //permuted sample
    if (permInBounds)
        valid = TestTemporalCandidate_GI(permCoord, dims_f, g_sample_last, myMatID, myN1s, myPos,
                                         tempPixelIdx, rInstID, rPrimID, rBary);

    [branch]
    if (valid)
    {
        Reservoir_GI rdi_r = loadReservoirGI(g_Reservoirs_last_gi, tempPixelIdx);

        //Final validity check that requires reservoir
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
                const float p_c = rdi.F_mag_gi * visReuse_c;

                //Canonical Jc: jacobian at current pixel's x1 -> canonical x2.
                //Env/miss samples preserve direction under shift, so Jc = 1.
                const float Jc_canonical = IsSentinelMatID(rdi.matID_gi)
                    ? ((rdi.matID_gi == MATID_ENV_MISS) ? 1.0f : ComputeJc(myPos, rdi.x2_gi, rdi.n2_s_gi))
                    : ComputeJc(myPos, rdi.x2_gi, rdi.n2_s_gi);

                float p_n = 0.0f;
                float n_c = 0.0f;
                float3 contrib_n_from_me = 0;

                //p_n: reconnect from neighbor vertex to current GI reservoir sample
                {
                    SurfaceVertex sv_r = BuildVertex(rInstID, rPrimID, rBary, cameraPos);

                    float3 rcKd = 0.0f; float rcPr = 0.0f, rcPm = 0.0f;
                    if (!IsSentinelMatID(rdi.matID_gi))
                        RefetchMaterial(rdi.matID_gi, rdi.uv_gi, rcKd, rcPr, rcPm);

                    float3 c = ReconnectGI(
                        sv_r.x, sv_r.n_s, sv_r.o, sv_r.matID,
                        sv_r.Kd, sv_r.Pr, sv_r.Pm,
                        rdi.matID_gi, rdi.x2_gi, rdi.n2_s_gi, rdi.L2_gi, rdi.V2_gi,
                        rcKd, rcPr, rcPm, rdi.eta_gi,
                        Jnc);

                    float ph = GetPHat(c);
                    // Env/miss: rdi.x2_gi is a DIRECTION — cast to far distance.
                    // Other: position-based connection + self-length shadow ray.
                    {
                        float vis;
                        if (rdi.matID_gi == MATID_ENV_MISS)
                        {
                            vis = IsVisible(sv_r.x, sv_r.n_s, normalize(rdi.x2_gi), 10000.0f) ? 1.0f : 0.0f;
                        }
                        else
                        {
                            float3 _conn = rdi.x2_gi - sv_r.x; float _cd = length(_conn);
                            vis = (_cd > EPSILON && IsVisible(sv_r.x, sv_r.n_s, _conn / _cd, _cd * 0.999f)) ? 1.0f : 0.0f;
                        }
                        p_n = ph * vis;
                    }
                }

                //n_c: reconnect from current vertex to neighbor GI reservoir sample
                float J2;
                {
                    SurfaceVertex sv_c = BuildVertex(myInstID, myPrimID, myBary, cameraPos);

                    // Neighbor Jc: jacobian at neighbor's x1 -> neighbor's x2
                    // Env/miss samples preserve direction under shift, Jc = 1.
                    const float Jc_neighbor = (rdi_r.matID_gi == MATID_ENV_MISS)
                        ? 1.0f
                        : ComputeJc(ReconstructPosition(rInstID, rPrimID, rBary),
                                    rdi_r.x2_gi, rdi_r.n2_s_gi);

                    float3 rrKd = 0.0f; float rrPr = 0.0f, rrPm = 0.0f;
                    if (!IsSentinelMatID(rdi_r.matID_gi))
                        RefetchMaterial(rdi_r.matID_gi, rdi_r.uv_gi, rrKd, rrPr, rrPm);

                    float3 c = ReconnectGI(
                        sv_c.x, sv_c.n_s, sv_c.o, sv_c.matID,
                        sv_c.Kd, sv_c.Pr, sv_c.Pm,
                        rdi_r.matID_gi, rdi_r.x2_gi, rdi_r.n2_s_gi, rdi_r.L2_gi, rdi_r.V2_gi,
                        rrKd, rrPr, rrPm, rdi_r.eta_gi,
                        Jn);

                    J2 = JacobianRatio(Jn, Jc_neighbor);
                    float ph = GetPHat(c);
                    //Env/miss
                    {
                        float vis_n;
                        if (rdi_r.matID_gi == MATID_ENV_MISS)
                        {
                            vis_n = IsVisible(sv_c.x, sv_c.n_s, normalize(rdi_r.x2_gi), 10000.0f) ? 1.0f : 0.0f;
                        }
                        else
                        {
                            float3 _conn = rdi_r.x2_gi - sv_c.x; float _cd = length(_conn);
                            vis_n = (_cd > EPSILON && IsVisible(sv_c.x, sv_c.n_s, _conn / _cd, _cd * 0.999f)) ? 1.0f : 0.0f;
                        }
                        n_c = ph * vis_n;
                        contrib_n_from_me = c * vis_n;
                    }
                }

                const float visReuse_n = (rdi_r.W_gi > 0.0f) ? 1.0f : 0.0f;
                const float n_n = rdi_r.F_mag_gi * visReuse_n;

                //M caps
                float sdata_Pr = myPr;
                float rdi_r_Pr = IsSentinelMatID(rdi_r.matID_gi)
                    ? 1.0f
                    : EvaluatePBRProperties(materials[rdi_r.matID_gi], rdi_r.uv_gi, 0).x;
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

                uint  F_gi_color_winner = rdi.F_gi;
                float F_gi_mag_winner   = rdi.F_mag_gi;
                p_hat_final = p_c;
                if (UpdateReservoirGI(
                        rdi,
                        w_n,
                        rdi_r.M_gi,
                        rdi_r.x2_gi, rdi_r.n2_s_gi, rdi_r.L2_gi, rdi_r.V2_gi,
                        rdi_r.uv_gi,
                        rdi_r.matID_gi, rdi_r.objID_gi, rdi_r.eta_gi,
                        rdi_r.F_gi, rdi_r.F_mag_gi,
                        seed
                    ))
                {
                    p_hat_final = n_c;
                    float  n_c_mag  = GetPHat(contrib_n_from_me);
                    float3 n_c_norm = (n_c_mag > 1e-20f) ? contrib_n_from_me / n_c_mag : float3(0,0,0);
                    F_gi_color_winner = PackRGB9E5(n_c_norm);
                    F_gi_mag_winner   = n_c_mag;
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

                rdi.F_gi     = F_gi_color_winner;
                rdi.F_mag_gi = F_gi_mag_winner;
            }
        }
    }

    //Write boilValue to scratch for the groupshared boiling post-pass
    gScratchPing[uint3(launchIndex, 5)] = float4(boilValue, 0, 0, 0);

    //Store merged reservoir
    storeReservoirGI(g_Reservoirs_current_gi, pixelIdx, rdi);
}