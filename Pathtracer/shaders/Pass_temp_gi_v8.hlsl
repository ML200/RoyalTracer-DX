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

    // Load the sample data (contains x1, n1, material props, etc.)
    SampleData sdata = loadSampleData(g_sample_current, pixelIdx);

    // Only process if valid sample
    if(all(sdata.L1 < EPSILON)){

        // Load current reservoir
        Reservoir_GI rdi = loadReservoirGI(g_Reservoirs_current_gi, pixelIdx);

        // Get a random seed
        uint2 seed = GetSeed(pixelIdx, time, 2);

        SampleData sdata_r;
        Reservoir_GI rdi_r;

        int2 tempPixelCoordinate = GetBestReprojectedPixel_d(sdata.x1, prevView, prevProjection, dims, sdata.objID);
        if(tempPixelCoordinate.x == -1 && tempPixelCoordinate.y == -1)
            tempPixelCoordinate = launchIndex;

        uint tempPixelIdx = MapPixelID(dims, tempPixelCoordinate);

        // Get the reprojected sample data and reservoir
        sdata_r = loadSampleData(g_sample_last, tempPixelIdx);
        rdi_r   = loadReservoirGI(g_Reservoirs_last_gi, tempPixelIdx);

        // Validation: Check validity, normals, position, and material ID
        bool valid =
            (all(sdata_r.L1 < EPSILON) &&
            IsValidReservoir_GI(rdi_r) &&
            !RejectNormal_GI(sdata.n1_s, sdata_r.n1_s, 0.36f) &&
            (!RejectDistance_GI(sdata.x1, sdata_r.x1, sdata.n1_s, 0.1f)) &&
            (sdata_r.matID == sdata.matID));

        if(tempPixelIdx != 0xFFFFFFFF && valid){

            // --- 1. Canonical Target Function (Current Surface -> Current Res Sample) ---
            float visReuse_c = rdi.W_gi > 0.0f ? 1.0f : 0.0f;
            float Jnc = 0.0f;
            float Jn = 0.0f;
            float J1 = 1.0f;
            float J2 = 1.0f;
            float J = 1.0f;

            float3 f_c = ReconnectGI(
                sdata.x1, sdata.n1_s, sdata.o, sdata.matID,
                sdata.localKd, sdata.localPr, sdata.localPm, sdata.etai, sdata.etat,

                rdi.x2_gi, rdi.n2_gi, rdi.L2_gi, rdi.V2_gi, rdi.matID_gi,
                rdi.localKd_gi, rdi.localPr_gi, rdi.localPm_gi, rdi.etai_gi, rdi.etat_gi,

                rdi.J_gi.x, 1.0f, false, Jnc, J
            ) * visReuse_c;

            float p_c = GetPHat(f_c);

            // --- 2. Neighbor Path -> Current Res Sample (Previous Surface -> Current Res Sample) ---
            // Used for MIS weight of current reservoir
            float3 f_p_n = ReconnectGI(
                sdata_r.x1, sdata_r.n1_s, sdata_r.o, sdata_r.matID,
                sdata_r.localKd, sdata_r.localPr, sdata_r.localPm, sdata_r.etai, sdata_r.etat,

                rdi.x2_gi, rdi.n2_gi, rdi.L2_gi, rdi.V2_gi, rdi.matID_gi,
                rdi.localKd_gi, rdi.localPr_gi, rdi.localPm_gi, rdi.etai_gi, rdi.etat_gi,

                rdi.J_gi.x, rdi.J_gi.y, true, Jnc, J1
            );
            float p_n = GetPHat(f_p_n);
            // * VisibilityCheckCP(...) omitted for perf

            // --- 3. Canonical Path -> Neighbor Res Sample (Current Surface -> Previous Res Sample) ---
            // The candidate sample we might merge
            float3 fn_c = ReconnectGI(
                sdata.x1, sdata.n1_s, sdata.o, sdata.matID,
                sdata.localKd, sdata.localPr, sdata.localPm, sdata.etai, sdata.etat,

                rdi_r.x2_gi, rdi_r.n2_gi, rdi_r.L2_gi, rdi_r.V2_gi, rdi_r.matID_gi,
                rdi_r.localKd_gi, rdi_r.localPr_gi, rdi_r.localPm_gi, rdi_r.etai_gi, rdi_r.etat_gi,

                rdi_r.J_gi.x, rdi_r.J_gi.y, true, Jn, J2
            ) * VisibilityCheckCP(sdata.x1, rdi_r.x2_gi, sdata.n1_s);

            float n_c = GetPHat(fn_c);

            // --- 4. Neighbor Target Function (Previous Surface -> Previous Res Sample) ---
            float visReuse = rdi_r.W_gi > 0.0f ? 1.0f : 0.0f;
            float3 f_n_n = ReconnectGI(
                sdata_r.x1, sdata_r.n1_s, sdata_r.o, sdata_r.matID,
                sdata_r.localKd, sdata_r.localPr, sdata_r.localPm, sdata_r.etai, sdata_r.etat,

                rdi_r.x2_gi, rdi_r.n2_gi, rdi_r.L2_gi, rdi_r.V2_gi, rdi_r.matID_gi,
                rdi_r.localKd_gi, rdi_r.localPr_gi, rdi_r.localPm_gi, rdi_r.etai_gi, rdi_r.etat_gi,

                rdi_r.J_gi.x, 1.0f, false, Jnc, J
            ) * visReuse;

            float n_n = GetPHat(f_n_n);

            // --- MIS Calculation ---
            float M_c = min(TEMP_MCAP_GI, rdi.M_gi);
            float M_n = min(TEMP_MCAP_GI, rdi_r.M_gi);
            float M_sum = M_c + M_n;

            float mis_c = PairwiseMIS_Canonical_Temp(M_c, M_n, p_c, p_n * J1, M_sum);
            float mis_n = PairwiseMIS_Neighbour_Temp(M_c, M_n, n_c * J2, n_n, M_sum);

            // Calculate reservoir weights
            float w_c = mis_c * p_c * rdi.W_gi;
            float w_n = mis_n * n_c * J2 * rdi_r.W_gi;

            // Adjust wsum of the existing reservoir
            rdi.w_sum_gi = w_c;

            // --- Update Reservoir ---
            float p_hat_final = p_c;

            // Pass all new material properties to the Update function
            if(UpdateReservoirGI(
                rdi, w_n, rdi_r.M_gi,
                rdi_r.x2_gi, rdi_r.n2_gi, rdi_r.L2_gi, rdi_r.V2_gi,
                rdi_r.localKd_gi, rdi_r.localPr_gi, rdi_r.localPm_gi, rdi_r.etai_gi, rdi_r.etat_gi, // New params
                rdi_r.matID_gi, rdi_r.objID_gi, rdi_r.rSeed_gi, rdi_r.J_gi, rdi_r.rIndex_gi,
                rdi_r.F_gi, rdi_r.lobe0_gi, rdi_r.lobe1_gi, seed))
            {
                p_hat_final = n_c;
                rdi.J_gi.y = Jn; // Update Jacobian
            }

            // --- Finalize W ---
            if (p_hat_final > EPSILON && rdi.w_sum_gi > 0.0f) {
                float W = rdi.w_sum_gi / p_hat_final;
                if (isnan(W) || isinf(W)) { W = 0.0f; }
                rdi.W_gi = W;
            }
            else {
                rdi.W_gi = 0.0f;
            }

            // --- Update Jacobian for next frame (PSS) ---
            // Calculates Jacobian of (Surface 1 -> Selected Sample)
            rdi.J_gi.y = PSSJacobian(
                sdata.x1, sdata.n1_s, sdata.o, sdata.matID,
                sdata.localKd, sdata.localPr, sdata.localPm, sdata.etai, sdata.etat,

                rdi.x2_gi, rdi.n2_gi, rdi.V2_gi, rdi.matID_gi,
                rdi.localKd_gi, rdi.localPr_gi, rdi.localPm_gi, rdi.etai_gi, rdi.etat_gi,

                rdi.J_gi.x
            );

            rdi.F_gi = p_hat_final;

            // Store the merged reservoir
            storeReservoirGI(g_Reservoirs_current_gi, pixelIdx, rdi);
        }
    }
}