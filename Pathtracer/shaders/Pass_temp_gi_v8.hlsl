#include "Includes_v8.hlsli"

//─────────────────────────────────────────────────────────────────────────────
//  TEMPORAL  GI
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
    if(all(sdata.L1 < EPSILON)){
        // Load current reservoir
        Reservoir_GI rdi = loadReservoirGI(g_Reservoirs_current_gi, pixelIdx);

        // Get a random seed
        uint2 seed = GetSeed(pixelIdx, time, 2);

        SampleData sdata_r;
        Reservoir_GI rdi_r;
        //uint tempPixelIdx = 0xFFFFFFFF;

        int2 tempPixelCoordinate = GetBestReprojectedPixel_d(sdata.x1, prevView, prevProjection, dims, sdata.objID);
        if(tempPixelCoordinate.x == -1 && tempPixelCoordinate.y == -1)
            tempPixelCoordinate = launchIndex;

        uint tempPixelIdx = MapPixelID(dims, tempPixelCoordinate);
        // Get the reprojected sample data
        sdata_r = loadSampleData(g_sample_last, tempPixelIdx);
        // Get the reprojected reservoir
        rdi_r = loadReservoirGI(g_Reservoirs_last_gi, tempPixelIdx);

        // Weight the current sample - is it valid? And select the one closest in world space
        bool valid =
            (all(sdata_r.L1 < EPSILON) &&
            IsValidReservoir_GI(rdi_r) &&
            !RejectNormal_GI(sdata.n1_s, sdata_r.n1_s, 0.36f)&&
            (!RejectDistance_GI(sdata.x1, sdata_r.x1, sdata.n1_s, 0.1f))&&
            (sdata_r.matID == sdata.matID));


        if(tempPixelIdx != 0xFFFFFFFF && valid){
            // Calculate the canonical target function
            float visReuse_c = rdi.W_gi > 0.0f ? 1.0f : 0.0f;
            float Jnc = 0.0f;
            float Jn = 0.0f;
            float J1 = 1.0f;
            float J2 = 1.0f;
            float J = 1.0f;
            float3 f_c = ReconnectGI(sdata.x1, sdata.n1_s, sdata.n1_g, sdata.o, sdata.matID, sdata.localKd, sdata.localPr, sdata.localPm, sdata.etai, sdata.etat, rdi.matID_gi, rdi.x2_gi, rdi.n2_s_gi, rdi.n2_g_gi, rdi.L2_gi, rdi.V2_gi, rdi.localKd_gi, rdi.localPr_gi, rdi.localPm_gi, rdi.etai_gi, rdi.etat_gi, rdi.J_gi.x, 1.0f, false, Jnc, J) * visReuse_c;
            float p_c = GetPHat(f_c);
            float p_n = GetPHat(ReconnectGI(sdata_r.x1, sdata_r.n1_s, sdata_r.n1_g, sdata_r.o, sdata_r.matID, sdata_r.localKd, sdata_r.localPr, sdata_r.localPm, sdata_r.etai, sdata_r.etat, rdi.matID_gi, rdi.x2_gi, rdi.n2_s_gi, rdi.n2_g_gi, rdi.L2_gi, rdi.V2_gi, rdi.localKd_gi, rdi.localPr_gi, rdi.localPm_gi, rdi.etai_gi, rdi.etat_gi, rdi.J_gi.x, rdi.J_gi.y, true, Jnc, J1)) * VisibilityCheckCP(sdata_r.x1, rdi.x2_gi, sdata_r.n1_s, 0u); // would require last frame AS and we dont store it, can be ommited for minimal added bias
            float3 fn_c = ReconnectGI(sdata.x1, sdata.n1_s, sdata.n1_g, sdata.o, sdata.matID, sdata.localKd, sdata.localPr, sdata.localPm, sdata.etai, sdata.etat, rdi_r.matID_gi, rdi_r.x2_gi, rdi_r.n2_s_gi, rdi_r.n2_g_gi, rdi_r.L2_gi, rdi_r.V2_gi, rdi_r.localKd_gi, rdi_r.localPr_gi, rdi_r.localPm_gi, rdi_r.etai_gi, rdi_r.etat_gi, rdi_r.J_gi.x, rdi_r.J_gi.y, true, Jn, J2) * VisibilityCheckCP(sdata.x1, rdi_r.x2_gi, sdata.n1_s, 0u);
            float n_c = GetPHat(fn_c);
            float visReuse = rdi_r.W_gi > 0.0f ? 1.0f : 0.0f;
            float n_n = GetPHat(ReconnectGI(sdata_r.x1, sdata_r.n1_s, sdata_r.n1_g, sdata_r.o, sdata_r.matID, sdata_r.localKd, sdata_r.localPr, sdata_r.localPm, sdata_r.etai, sdata_r.etat, rdi_r.matID_gi, rdi_r.x2_gi, rdi_r.n2_s_gi, rdi_r.n2_g_gi, rdi_r.L2_gi, rdi_r.V2_gi, rdi_r.localKd_gi, rdi_r.localPr_gi, rdi_r.localPm_gi, rdi_r.etai_gi, rdi_r.etat_gi, rdi_r.J_gi.x, 1.0f, false, Jnc, J)) * visReuse;
            float M_c = min(TEMP_MCAP_GI,rdi.M_gi);
            float M_n = min(TEMP_MCAP_GI,rdi_r.M_gi);
            float M_sum = M_c + M_n;
            // Calculate the MIS weights
            float mis_c = PairwiseMIS_Canonical_Temp(M_c, M_n, p_c, p_n * J1, M_sum);
            float mis_n = PairwiseMIS_Neighbour_Temp(M_c, M_n, n_c * J2, n_n, M_sum);

            // Calculate the reservoirs weights
            float w_c = mis_c * p_c * rdi.W_gi;
            float w_n = mis_n * n_c * J2 * rdi_r.W_gi;

            // Adjust wsum of the existing reservoir
            rdi.w_sum_gi = w_c;

            // Update the reservoir
            float p_hat_final = p_c;
            if(UpdateReservoirGI(rdi, w_n, rdi_r.M_gi, rdi_r.x2_gi, rdi_r.n2_s_gi, rdi_r.n2_g_gi, rdi_r.L2_gi, rdi_r.V2_gi, rdi_r.localKd_gi, rdi_r.localPr_gi, rdi_r.localPm_gi, rdi_r.etai_gi, rdi_r.etat_gi, rdi_r.matID_gi, rdi_r.objID_gi, rdi_r.J_gi, rdi_r.F_gi, seed)){
                p_hat_final = n_c;
                rdi.J_gi.y = Jn;
            }

            // Calculate new W
            //float p_hat = GetPHat(ReconnectDI(sdata.x1, sdata.n1, sdata.o, sdata.matID, rdi.x2_di, rdi.n2_di, rdi.L2_di));
            if (p_hat_final > EPSILON && rdi.w_sum_gi > 0.0f) {
                float W = rdi.w_sum_gi / p_hat_final;
                // NaN/Inf protection
                if (isnan(W) || isinf(W)) {
                    W = 0.0f;
                }
                rdi.W_gi = W;
            }
            else
                rdi.W_gi = 0.0f;

            // Recompute and update the jacobian for the now-canonical stored sample
            rdi.J_gi.y = PSSJacobian(sdata.x1, sdata.n1_s, sdata.n1_g, sdata.o, sdata.matID, sdata.localKd, sdata.localPr, sdata.localPm, sdata.etai, sdata.etat, rdi.x2_gi, rdi.n2_s_gi, rdi.n2_g_gi, rdi.V2_gi, rdi.matID_gi, rdi.localKd_gi, rdi.localPr_gi, rdi.localPm_gi, rdi.etai_gi, rdi.etat_gi, rdi.J_gi.x);
            rdi.F_gi = p_hat_final;

            // Store the merged reservoir
            storeReservoirGI(g_Reservoirs_current_gi, pixelIdx, rdi);
        }
    }
}
