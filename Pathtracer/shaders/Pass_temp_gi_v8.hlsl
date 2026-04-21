#include "Includes_v8.hlsli"

//====================================================================
//TEMPORAL GI
//====================================================================
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

    //Emitter early-out
    if (load_isEmitter(g_sample_current, pixelIdx))
    {
        gScratchPing[uint3(launchIndex, 5)] = 0;
        return;
    }

    //Load current reservoir, must stay alive until final store
    Reservoir rdi = loadReservoir(g_Reservoirs_current, pixelIdx);

    //Disabled early-out
    if (!(rs_flags & 2u)) {
        gScratchPing[uint3(launchIndex, 5)] = 0;
        storeReservoir(g_Reservoirs_current, pixelIdx, rdi);
        return;
    }

    float boilValue = 0.0f;
    uint2 seed = GetSeed(pixelIdx, time, 3);
    uint  permSeed = GetSeed(1, time, 3).x;

    //====================================================================
    //BASE REPROJECTION
    //====================================================================
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

    //Specularity, same computation DLSS-RR gets via EnvBRDFApprox2
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

    //====================================================================
    //PERMUTED CANDIDATE
    //====================================================================
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

    //Lightweight neighbor identifiers, survive rejection for merge
    uint   rInstID = 0;
    uint   rPrimID = 0;
    float2 rBary   = float2(0, 0);

    //Permuted sample
    if (permInBounds)
        valid = TestTemporalCandidate(permCoord, dims_f, g_sample_last, myMatID, myN1s, myPos,
                                         tempPixelIdx, rInstID, rPrimID, rBary);

    [branch]
    if (valid)
    {
        Reservoir rdi_r = loadReservoir(g_Reservoirs_last, tempPixelIdx);

        //Final validity check that requires reservoir
        if (!IsValidReservoir(rdi_r))
        {
            valid = false;
        }

        if (valid)
        {
            float p_hat_final = 0.0f;

            {
                const float3 cameraPos = InitOrigin();
                float Jnc = 0.0f, Jn = 0.0f;

                const float visReuse_c = (rdi.W > 0.0f) ? 1.0f : 0.0f;
                const float p_c = GetPHat(rdi.F) * visReuse_c;

                //Canonical Jc: jacobian at current pixel's x1 to canonical x2.
                //Env/miss samples preserve direction under shift, so Jc = 1.
                const float Jc_canonical = IsSentinelMatID(rdi.matID)
                    ? ((rdi.matID == MATID_ENV_MISS) ? 1.0f : ComputeJc(myPos, rdi.x2, rdi.n2_s))
                    : ComputeJc(myPos, rdi.x2, rdi.n2_s);

                float p_n = 0.0f;
                float n_c = 0.0f;
                float3 contrib_n_from_me = 0;

                //p_n: reconnect from neighbor vertex to current GI reservoir sample
                {
                    SurfaceVertex sv_r = BuildVertex(rInstID, rPrimID, rBary, cameraPos);

                    float3 rcKd = 0.0f; float rcPr = 0.0f, rcPm = 0.0f;
                    if (!IsSentinelMatID(rdi.matID))
                        RefetchMaterial(rdi.matID, rdi.uv, rcKd, rcPr, rcPm);

                    float3 c = Reconnect(
                        sv_r.x, sv_r.n_s, sv_r.o, sv_r.matID,
                        sv_r.Kd, sv_r.Pr, sv_r.Pm, sv_r.etai, sv_r.etat,
                        rdi.matID, rdi.x2, rdi.n2_s, rdi.L2, rdi.V2,
                        rcKd, rcPr, rcPm, rdi.eta,
                        Jnc);

                    float ph = GetPHat(c);
                    //Env/miss: rdi.x2 is a DIRECTION, cast to far distance.
                    //Other: position-based connection + self-length shadow ray.
                    {
                        float vis;
                        if (rdi.matID == MATID_ENV_MISS)
                        {
                            vis = IsVisible(sv_r.x, sv_r.n_s, normalize(rdi.x2), 10000.0f) ? 1.0f : 0.0f;
                        }
                        else
                        {
                            float3 _conn = rdi.x2 - sv_r.x; float _cd = length(_conn);
                            vis = (_cd > EPSILON && IsVisible(sv_r.x, sv_r.n_s, _conn / _cd, _cd * 0.999f)) ? 1.0f : 0.0f;
                        }
                        p_n = ph * vis;
                    }
                }

                //n_c: reconnect from current vertex to neighbor GI reservoir sample
                float J2;
                {
                    SurfaceVertex sv_c = BuildVertex(myInstID, myPrimID, myBary, cameraPos);

                    //Neighbor Jc: jacobian at neighbor's x1 to neighbor's x2.
                    //Env/miss samples preserve direction under shift, Jc = 1.
                    const float Jc_neighbor = (rdi_r.matID == MATID_ENV_MISS)
                        ? 1.0f
                        : ComputeJc(ReconstructPosition(rInstID, rPrimID, rBary),
                                    rdi_r.x2, rdi_r.n2_s);

                    float3 rrKd = 0.0f; float rrPr = 0.0f, rrPm = 0.0f;
                    if (!IsSentinelMatID(rdi_r.matID))
                        RefetchMaterial(rdi_r.matID, rdi_r.uv, rrKd, rrPr, rrPm);

                    float3 c = Reconnect(
                        sv_c.x, sv_c.n_s, sv_c.o, sv_c.matID,
                        sv_c.Kd, sv_c.Pr, sv_c.Pm, sv_c.etai, sv_c.etat,
                        rdi_r.matID, rdi_r.x2, rdi_r.n2_s, rdi_r.L2, rdi_r.V2,
                        rrKd, rrPr, rrPm, rdi_r.eta,
                        Jn);

                    J2 = JacobianRatio(Jn, Jc_neighbor);
                    float ph = GetPHat(c);
                    //Env/miss
                    {
                        float vis_n;
                        if (rdi_r.matID == MATID_ENV_MISS)
                        {
                            vis_n = IsVisible(sv_c.x, sv_c.n_s, normalize(rdi_r.x2), 10000.0f) ? 1.0f : 0.0f;
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

                //Correlation-reduction cCap (Lin et al. 2026 §5). The
                //previous frame's duplication map at the temporal
                //neighbor's pixel counts shifted-copies in its 17x17
                //window, normalized to D in [0, 1]. High D means this
                //sample has already spread across many neighbors, so
                //we lower the temporal cCap to refresh the chain and
                //stop firefly persistence. alpha = 0.1 gives a quick
                //ramp even at small D; cMin = 1 matches the paper.
                //Introduces a small bounded bias in highly correlated
                //regions (paper measured ~3% mean absolute in Kitchen).
                //
                //Env/miss (sky, depth=0 sun NEE) is direction-preserving
                //under the reconnection shift — propagation across
                //pixels is benign (bounded radiance, no firefly
                //amplification), and applying the reduction there caps
                //effMcap ~ 1 in sky regions regardless of rs_tempMcap,
                //defeating temporal accumulation on the sky. Skip the
                //reduction for env/miss and let the full rs_tempMcap
                //drive the history length.
                const float D       = saturate(gScratchPing[uint3(uint2(permCoord), 6)].x);
                const float effMcap = (rdi_r.matID == MATID_ENV_MISS)
                    ? (float)rs_tempMcap
                    : lerp((float)rs_tempMcap, 1.0f, pow(D, 0.1f));

                //M caps
                float sdata_Pr = myPr;
                float rdi_r_Pr = IsSentinelMatID(rdi_r.matID)
                    ? 1.0f
                    : EvaluatePBRProperties(rdi_r.matID, rdi_r.uv, 0).x;
                const float minRoughTemp  = min(sdata_Pr, rdi_r_Pr);
                const float tempMcapScale = smoothstep(rs_reuseRoughnessMin, rs_reuseRoughnessMax, minRoughTemp);
                const float dynTempMcap   = (minRoughTemp <= rs_reuseRoughnessMin) ? 0.0f
                                        : min(effMcap, effMcap * tempMcapScale);

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
                //UpdateReservoir writes F = contrib_n_from_me on acceptance,
                //the shifted-to-current-pixel contribution, which is exactly
                //what we want stored for the winner, GetPHat of it is n_c.
                if (UpdateReservoir(
                        rdi,
                        w_n,
                        rdi_r.M,
                        rdi_r.x2, rdi_r.n2_s, rdi_r.L2, rdi_r.V2,
                        rdi_r.uv,
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

    //Write boilValue to scratch for the groupshared boiling post-pass
    gScratchPing[uint3(launchIndex, 5)] = float4(boilValue, 0, 0, 0);

    //Store merged reservoir
    storeReservoir(g_Reservoirs_current, pixelIdx, rdi);
}
