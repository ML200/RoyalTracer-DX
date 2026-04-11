#define COMPUTE_PASS
#include "Includes_v8.slang"

//---------------------------------------------------------------------
//  TEMPORAL  DI
//---------------------------------------------------------------------
[numthreads(16, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= IMG_W || tid.y >= IMG_H) return;
    gDispatchIdx = tid;

    uint2  launchIndex   = tid.xy;
    float2 dims = float2(IMG_W, IMG_H);
    uint   pixelIdx  = MapPixelID(dims, launchIndex);

    // Emitter check
    if (load_isEmitter(g_sample_current, pixelIdx))
    {
        gScratchPing[uint3(launchIndex, 0)] = 0;
        return;
    }

    // Load current reservoir
    Reservoir_DI rdi = loadReservoirDI(g_Reservoirs_current_di, pixelIdx);

    // Early-out if temporal DI is disabled
    if (!(rs_flags & 1u)) {
        gScratchPing[uint3(launchIndex, 0)] = 0;
        storeReservoirDI(g_Reservoirs_current_di, pixelIdx, rdi);
        return;
    }

    // Lightweight loads for reprojection & rejection (defer BuildVertex to merge)
    const uint   myInstID = load_instID(g_sample_current, pixelIdx);
    const uint   myPrimID = load_primID(g_sample_current, pixelIdx);
    const float2 myBary   = load_bary(g_sample_current, pixelIdx);
    const uint   myMatID  = GetMatIDFast(myInstID, myPrimID);
    const float3 myPos    = ReconstructPosition(myInstID, myPrimID, myBary);
    const float3 myN1g    = load_n1_g_with_instID(g_sample_current, pixelIdx, myInstID);

    float boilValue = 0.0f;

    // RNG
    uint2 seed = GetSeed(pixelIdx, time, 2);
    uint  permSeed = GetSeed(1, time, 2).x;

    // --- base reprojection ---
    int2 baseCoord = GetBestReprojectedPixel_d(myPos, prevView, prevProjection, dims, myInstID);
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
    // Lightweight neighbor identifiers (survive rejection -> merge)
    uint   rInstID = 0;
    uint   rPrimID = 0;
    float2 rBary   = float2(0, 0);

    // 1) Try the permuted sample
    if (permInBounds)
        valid = TestTemporalCandidate_DI(permCoord, dims, g_sample_last, myMatID, myN1g, myPos,
                                         tempPixelIdx, rInstID, rPrimID, rBary);


    [branch]
    if (valid)
    {
        Reservoir_DI rdi_r = loadReservoirDI(g_Reservoirs_last_di, tempPixelIdx);

        if (!IsValidReservoir_DI(rdi_r))
        {
            valid = false;
        }

        if (valid)
        {
            const float3 cameraPos = InitOrigin();
            const float visReuse_c = rdi.W_di > 0.0f ? 1.0f : 0.0f;
            const float visReuse_r = rdi_r.W_di > 0.0f ? 1.0f : 0.0f;

            float p_c, n_c, p_n, n_n;
            float3 x_c, n_g_c, x_r, n_g_r;
            float Pr_c, Pr_r;

            // -- Phase 1: sv_c scope (build, reconnect, extract, drop) --
            {
                SurfaceVertex sv = BuildVertex(myInstID, myPrimID, myBary, cameraPos);
                sv.etai = load_etai(g_sample_current, pixelIdx);
                sv.etat = load_etat(g_sample_current, pixelIdx);

                p_c = GetPHat(ReconnectDI(sv.x, sv.n_s, sv.n_g, sv.o, sv.matID, rdi.x2_di, rdi.n2_di, rdi.L2_di, sv.Kd, sv.Pr, sv.Pm, sv.etai, sv.etat, rdi.objID_di)) * visReuse_c;
                n_c = GetPHat(ReconnectDI(sv.x, sv.n_s, sv.n_g, sv.o, sv.matID, rdi_r.x2_di, rdi_r.n2_di, rdi_r.L2_di, sv.Kd, sv.Pr, sv.Pm, sv.etai, sv.etat, rdi_r.objID_di));
                x_c   = sv.x;
                n_g_c = sv.n_g;
                Pr_c  = sv.Pr;

                // visibility for n_c
                { float3 _vd; float _vt;
                  if (rdi_r.objID_di >= 0xFFFFFFFEu) { _vd = normalize(rdi_r.x2_di); _vt = 10000.0f; }
                  else { float3 _c = rdi_r.x2_di - x_c; float _d = length(_c); _vd = _c / max(_d, EPSILON); _vt = _d * 0.999f; }
                  n_c *= IsVisible(x_c, n_g_c, _vd, _vt) ? 1.0f : 0.0f; }
            }

            // -- Phase 2: sv_r scope (build, reconnect, extract, drop) --
            {
                SurfaceVertex sv = BuildVertex(rInstID, rPrimID, rBary, cameraPos);
                sv.etai = load_etai(g_sample_last, tempPixelIdx);
                sv.etat = load_etat(g_sample_last, tempPixelIdx);

                p_n = GetPHat(ReconnectDI(sv.x, sv.n_s, sv.n_g, sv.o, sv.matID, rdi.x2_di, rdi.n2_di, rdi.L2_di, sv.Kd, sv.Pr, sv.Pm, sv.etai, sv.etat, rdi.objID_di));
                n_n = GetPHat(ReconnectDI(sv.x, sv.n_s, sv.n_g, sv.o, sv.matID, rdi_r.x2_di, rdi_r.n2_di, rdi_r.L2_di, sv.Kd, sv.Pr, sv.Pm, sv.etai, sv.etat, rdi_r.objID_di)) * visReuse_r;
                x_r   = sv.x;
                n_g_r = sv.n_g;
                Pr_r  = sv.Pr;

                // visibility for p_n
                { float3 _vd; float _vt;
                  if (rdi.objID_di >= 0xFFFFFFFEu) { _vd = normalize(rdi.x2_di); _vt = 10000.0f; }
                  else { float3 _c = rdi.x2_di - x_r; float _d = length(_c); _vd = _c / max(_d, EPSILON); _vt = _d * 0.999f; }
                  p_n *= IsVisible(x_r, n_g_r, _vd, _vt) ? 1.0f : 0.0f; }
            }

            // -- Phase 3: Jacobians (only extracted positions) --
            p_n *= JacobianDeterminantDI(x_c, rdi.x2_di, x_r, rdi.n2_di, rdi.objID_di);
            n_c *= JacobianDeterminantDI(x_r, rdi_r.x2_di, x_c, rdi_r.n2_di, rdi_r.objID_di);

            // Dynamic M caps (roughness-based, aligned with GI)
            float sdata_Pr = Pr_c;
            float rdi_r_Pr = Pr_r;
            const float minRoughTemp  = min(sdata_Pr, rdi_r_Pr);
            const float tempMcapScale = smoothstep(rs_reuseRoughnessMin, rs_reuseRoughnessMax, minRoughTemp);
            const float dynTempMcap   = (minRoughTemp <= rs_reuseRoughnessMin) ? 0.0f
                                    : min(rs_tempMcapDI, rs_tempMcapDI * tempMcapScale);

            const float M_c   = min(rs_tempMcapDI, rdi.M_di);
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

    // Write boilValue to scratch for the groupshared boiling post-pass
    gScratchPing[uint3(launchIndex, 0)] = float4(boilValue, 0, 0, 0);

    // Store the merged reservoir
    storeReservoirDI(g_Reservoirs_current_di, pixelIdx, rdi);
}
