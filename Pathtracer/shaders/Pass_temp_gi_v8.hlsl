#include "Includes_v8.hlsli"

//====================================
//TEMPORAL GI
//====================================
[shader("raygeneration")]
void Pass_temp_gi_v8()
{
    /*uint sortKey;
    {
        const uint2 li  = DispatchRaysIndex().xy;
        const uint  px  = MapPixelID(float2(IMG_W, IMG_H), li);
        const bool  emi = load_isEmitter(g_sample_current, px);

        sortKey = emi ? 0u : (!(rs_flags & 2u) ? 1u : 2u);
    }
    dx::MaybeReorderThread(sortKey, 2);*/

    const uint2  launchIndex = DispatchRaysIndex().xy;
    const float2 dims_f      = float2(IMG_W, IMG_H);
    const uint   pixelIdx    = MapPixelID(dims_f, launchIndex);

    //emitter early terrain
    if (load_isEmitter(g_sample_current, pixelIdx))
    {
        gScratchPing[uint3(launchIndex, 5)] = 0;
        return;
    }

    //keep current reservoir alive until final store
    Reservoir rdi = loadReservoir(g_Reservoirs_current, pixelIdx);

    //disabled early terrain
    if (!(rs_flags & 2u)) {
        gScratchPing[uint3(launchIndex, 5)] = 0;
        storeReservoir(g_Reservoirs_current, pixelIdx, rdi);
        return;
    }

    float boilValue = 0.0f;
    uint2 seed = GetSeed(pixelIdx, time, 3);
    uint  permSeed = GetSeed(1, time, 3).x;

    //====================================
    //BASE REPROJECTION
    //====================================
    //lightweight loads only - all baked, no per-triangle / texture access
    const uint   myInstID = load_instID(g_sample_current, pixelIdx);
    const uint   myMatID  = load_matID(g_sample_current, pixelIdx);
    const float3 myPos    = load_x1(g_sample_current, pixelIdx);
    const float3 myN1s    = load_n1_s_with_instID(g_sample_current, pixelIdx, myInstID);
    float3 myKd = load_kd(g_sample_current, pixelIdx);
    float  myPr, myPm;
    load_prpm(g_sample_current, pixelIdx, myPr, myPm);

    //specularity matches DLSS RR EnvBRDFApprox2
    const float3 camPos = InitOrigin();
    const float  NoV = saturate(dot(normalize(camPos - myPos), myN1s));
    const float  specularity = Luma(EnvBRDFApprox2(myKd, myPr, myPm, NoV));

    //stochastic reprojection, specular vs diffuse MV weighted by specularity
    float4 reflData = gScratchPing[uint3(launchIndex, 4)];
    uint   reflInstID = asuint(reflData.w);
    //valid unless the reflection probe missed (0xFFFFFFFF sentinel).
    bool   reflValid = (reflInstID != 0xFFFFFFFFu);
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

    //====================================
    //PERMUTED CANDIDATE
    //====================================
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

    if (permInBounds)
        valid = TestTemporalCandidate(permCoord, dims_f, g_sample_last, tempPixelIdx);
    //PLANET: no terrain staleness check needed - the surface is rebuilt from
    //the stored world hitT, which LOD re-tessellation cannot invalidate.

    [branch]
    if (valid)
    {
        Reservoir rdi_r = loadReservoir(g_Reservoirs_last, tempPixelIdx);

        if (!IsValidReservoir(rdi_r))
        {
            valid = false;
        }

        if (valid)
        {
            float p_hat_final = 0.0f;

            {
                const float3 cameraPos = InitOrigin();
                //neighbour primary hit + instID, reconstructed from the stored
                //hitT - no per-triangle data, identical for terrain and meshes.
                //rInstID feeds IsVisibleEnvMiss' offset_ray (terrain/mesh branch).
                const uint   rInstID = load_instID(g_sample_last, tempPixelIdx);
                const float3 rPos = load_x1(g_sample_last, tempPixelIdx);
                float Jnc = 0.0f, Jn = 0.0f;

                const float visReuse_c = (rdi.W > 0.0f) ? 1.0f : 0.0f;
                const float p_c = GetPHat(rdi.F) * visReuse_c;

                //canonical Jc, env/miss preserves direction so Jc=1
                const float Jc_canonical = IsSentinelMatID(rdi.matID)
                    ? ((rdi.matID == MATID_ENV_MISS) ? 1.0f : ComputeJc(myPos, rdi.x2, rdi.n2_s))
                    : ComputeJc(myPos, rdi.x2, rdi.n2_s);

                float p_n = 0.0f;
                float n_c = 0.0f;
                float3 contrib_n_from_me = 0;

                //p_n, reconnect from neighbor to current GI sample
                {
                    SurfaceVertex sv_r = BuildVertex(g_sample_last, tempPixelIdx, rPos, cameraPos);

                    float3 c = Reconnect(
                        sv_r.x, sv_r.n_s, sv_r.o, sv_r.matID,
                        sv_r.Kd, sv_r.Pr, sv_r.Pm, sv_r.etai, sv_r.etat,
                        rdi.matID, rdi.x2, rdi.n2_s, rdi.L2, rdi.V2,
                        rdi.Kd, rdi.Pr, rdi.Pm, rdi.eta,
                        Jnc);

                    float ph = GetPHat(c);
                    //env/miss casts far, others use self length shadow ray
                    {
                        float vis;
                        if (rdi.matID == MATID_ENV_MISS)
                        {
                            vis = IsVisibleEnvMiss(sv_r.x, sv_r.n_s, normalize(rdi.x2), RAY_TMAX_PLANET, rInstID) ? 1.0f : 0.0f;
                        }
                        else
                        {
                            float3 _conn = rdi.x2 - sv_r.x; float _cd = length(_conn);
                            vis = (_cd > EPSILON && IsVisible(sv_r.x, sv_r.n_s, _conn / _cd, _cd * 0.999f)) ? 1.0f : 0.0f;
                        }
                        p_n = ph * vis;
                    }
                }

                //n_c, reconnect from current to neighbor GI sample
                float J2;
                {
                    SurfaceVertex sv_c = BuildVertex(g_sample_current, pixelIdx, myPos, cameraPos);

                    const float Jc_neighbor = (rdi_r.matID == MATID_ENV_MISS)
                        ? 1.0f
                        : ComputeJc(rPos, rdi_r.x2, rdi_r.n2_s);

                    float3 c = Reconnect(
                        sv_c.x, sv_c.n_s, sv_c.o, sv_c.matID,
                        sv_c.Kd, sv_c.Pr, sv_c.Pm, sv_c.etai, sv_c.etat,
                        rdi_r.matID, rdi_r.x2, rdi_r.n2_s, rdi_r.L2, rdi_r.V2,
                        rdi_r.Kd, rdi_r.Pr, rdi_r.Pm, rdi_r.eta,
                        Jn);

                    J2 = JacobianRatio(Jn, Jc_neighbor);
                    float ph = GetPHat(c);
                    {
                        float vis_n;
                        if (rdi_r.matID == MATID_ENV_MISS)
                        {
                            vis_n = IsVisibleEnvMiss(sv_c.x, sv_c.n_s, normalize(rdi_r.x2), RAY_TMAX_PLANET, myInstID) ? 1.0f : 0.0f;
                        }
                        else
                        {
                            float3 _conn = rdi_r.x2 - sv_c.x; float _cd = length(_conn);
                            vis_n = (_cd > EPSILON && IsVisible(sv_c.x, sv_c.n_s, _conn / _cd, _cd * 0.999f)) ? 1.0f : 0.0f;
                        }
                        n_c = ph * vis_n;
                        contrib_n_from_me = c * vis_n;
                    }
                }

                const float visReuse_n = (rdi_r.W > 0.0f) ? 1.0f : 0.0f;
                const float n_n = GetPHat(rdi_r.F) * visReuse_n;

                //correlation reduction cCap, dup count D refreshes the chain (2026 paper)
                const float D       = saturate(gScratchPing[uint3(uint2(permCoord), 6)].x);
                const float effMcap = (rdi_r.matID == MATID_ENV_MISS) ? (float)rs_tempMcap : lerp((float)rs_tempMcap, 1.0f, pow(D, 0.1f));

                //M caps, roughness dependent
                //gate ONLY on myPr, the current pixel's primary roughness, which is
                //deterministic G-buffer data fixed regardless of which sample wins the
                //RIS. Pulling rdi_r_Pr (roughness of the resampled neighbor vertex) into
                //M_n makes the confidence weight depend on the sample being resampled,
                //which biases the temporal estimator: energy gain that scales with the cap.
                const float tempMcapScale = smoothstep(rs_reuseRoughnessMin, rs_reuseRoughnessMax, myPr);
                const float dynTempMcap   = (myPr <= rs_reuseRoughnessMin) ? 0.0f : effMcap * tempMcapScale;

                const float M_c   = min(effMcap, rdi.M);
                const float M_n   = min(dynTempMcap,  rdi_r.M);
                const float M_sum = M_c + M_n;

                const float p_nJ1  = p_n * JacobianRatio(Jnc, Jc_canonical);
                const float n_cJ2  = n_c * J2;

                const float mis_c = PairwiseMIS_Canonical_Temp(M_c, M_n, p_c, p_nJ1, M_sum);
                const float mis_n = PairwiseMIS_Neighbour_Temp(M_c, M_n, n_cJ2, n_n, M_sum);

                const float w_c = mis_c * p_c * rdi.W;
                const float w_n = mis_n * n_cJ2 * rdi_r.W;

                rdi.w_sum = w_c;

                p_hat_final = p_c;
                //on acceptance F=contrib_n_from_me, GetPHat(F)=n_c
                if (UpdateReservoir(
                        rdi,
                        w_n,
                        rdi_r.M,
                        rdi_r.x2, rdi_r.n2_s, rdi_r.L2, rdi_r.V2,
                        rdi_r.Kd, rdi_r.Pr, rdi_r.Pm,
                        rdi_r.matID, rdi_r.objID, rdi_r.eta,
                        contrib_n_from_me,
                        seed
                    ))
                {
                    p_hat_final = n_c;
                }

                if (p_hat_final > EPSILON && rdi.w_sum > 0.0f)
                {
                    float W = rdi.w_sum / p_hat_final;
                    if (isnan(W) || isinf(W) || (W < 0.0f)) W = 0.0f;
                    rdi.W = W;
                }
                else
                {
                    rdi.W = 0.0f;
                }

                boilValue = p_hat_final * rdi.W;
            }
        }
    }

    //handoff to groupshared boiling pass
    gScratchPing[uint3(launchIndex, 5)] = float4(boilValue, 0, 0, 0);

    storeReservoir(g_Reservoirs_current, pixelIdx, rdi);
}
