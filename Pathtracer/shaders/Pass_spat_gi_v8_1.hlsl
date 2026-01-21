#include "Includes_v8.hlsli"

//─────────────────────────────────────────────────────────────────────────────
//  SPATIAL  GI
//─────────────────────────────────────────────────────────────────────────────
[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= IMG_W || tid.y >= IMG_H) return;
    gDispatchIdx = tid;

    uint2  launchIndex   = tid.xy;
    float2 dims = float2(IMG_W, IMG_H);
    uint   pixelIdx  = MapPixelID(dims, launchIndex);

    // Load the sample data (stored for next temporal samples already done spat_di)
    SampleData sdata = loadSampleData(g_sample_current, pixelIdx);
    // Load current reservoir
    Reservoir_GI rdi = loadReservoirGI(g_Reservoirs_current_gi, pixelIdx);

    if(all(sdata.L1 < EPSILON)){
        // Get a random seed
        uint2 seed = GetSeed(pixelIdx, time, 2);

        // ########################################### NODE #############################################################
        // Based on the quality of the current canonical sample, reduce the number of spatial reuses.
        float conf = min(60.0f, rdi.M_gi) / TEMP_MCAP_GI;
        uint nbrBudget = SPAT_COUNT_MIN_GI +
                 uint((1.0f - conf) * float(SPAT_COUNT_MAX_GI - SPAT_COUNT_MIN_GI) + 0.5f);
        uint radiusBudget = SPAT_RAD_MIN_GI +
                         uint((1.0f - conf) * float(SPAT_RAD_MAX_GI - SPAT_RAD_MIN_GI) + 0.5f);


        // Array to hold valid neighbor IDs
        uint nIds[SPAT_COUNT_MAX_GI];
        float M_sum = 0.0f;
        float M_sum_sym = 1.0f;
        // loop over all candidates and select those that are best
        [unroll(SPAT_COUNT_MAX_GI)]
        for(uint i = 0; i < SPAT_COUNT_MAX_GI; i++){
            if(i < nbrBudget){
                // loop until we find a valid candidate
                nIds[i] = 0xFFFFFFFF; // Set the id to invalid for that neighbor, until a valid one is found
                [unroll(SPAT_TRIS_GI)]
                for(uint j = 0; j < SPAT_TRIS_GI; j++){
                    // Get the candidate ID
                    uint iID = GetRandomPixelCircleWeighted(radiusBudget, dims.x, dims.y, launchIndex.x, launchIndex.y, seed);
                    // Only load data required for the comparison (L2 + M)
                    Reservoir_GI rdi_r = loadReservoirGI(g_Reservoirs_current_gi, iID);

                    // Check wether the reservoir is valid for merge (Later, replace this with a weight -> the neighbor with the highest weight is selected)
                    bool candidateAcceptedGI =
                        IsValidReservoir_GI_opt(rdi_r.n2_g_gi, rdi_r.M_gi) &&
                        (all(load_L1(g_sample_current, iID) < EPSILON) &&
                        !RejectNormal_GI(sdata.n1_s, load_n1_s(g_sample_current, iID), 0.36f) &&
                        !RejectDistance_GI(sdata.x1, load_x1(g_sample_current, iID), sdata.n1_s, 0.02f) &&
                        !RejectLength_GI(rdi.x2_gi, rdi.n2_g_gi, sdata.x1, load_x1(g_sample_current, iID), 0.1f) &&
                        !RejectLength_GI(rdi_r.x2_gi, rdi_r.n2_g_gi, load_x1(g_sample_current, iID), sdata.x1, 0.5f) &&
                        (load_matID(g_sample_current, iID) == sdata.matID));
                    if(candidateAcceptedGI){
                        nIds[i] = iID;
                        M_sum += min(SPAT_MCAP_GI, rdi_r.M_gi);
                        M_sum_sym += 1.0f;
                        break;
                    }
                }
            }
            else
                nIds[i] = 0xFFFFFFFF; // Fill the rest with invalid pixels -> fast loop over it (divergence, but fuck it)
        }
        // Reorganize the list of samples for better thread coherency
        uint writeIdx = 0;
        [unroll(SPAT_COUNT_MAX_GI)]
        for (uint readIdx = 0; readIdx < SPAT_COUNT_MAX_GI; ++readIdx)
        {
            uint id = nIds[readIdx];
            if (id != 0xFFFFFFFF)
                nIds[writeIdx++] = id;
        }
        [unroll(SPAT_COUNT_MAX_GI)]
        for (uint i = writeIdx; i < SPAT_COUNT_MAX_GI; ++i)
            nIds[i] = 0xFFFFFFFF;
        // ########################################### NODE #############################################################

        // Calculate M_sum for all valid candidates
        M_sum += min(SPAT_MCAP_GI, rdi.M_gi);
        float M_c = min(SPAT_MCAP_GI, rdi.M_gi);
        rdi.M_gi = M_c;

        float debug = 0.0f;

        // Calculate canonical pixel p_hat before loading the expensive data
        float visReuse = rdi.W_gi > 0.0f ? 1.0f : 0.0f;
        float Jnc = 0.0f;
        float Jcc = 0.0f;
        float3 contrib_c = ReconnectGI(sdata.x1, sdata.n1_s, sdata.n1_g, sdata.o, sdata.matID, sdata.localKd, sdata.localPr, sdata.localPm, sdata.etai, sdata.etat, rdi.matID_gi, rdi.x2_gi, rdi.n2_s_gi, rdi.n2_g_gi, rdi.L2_gi, rdi.V2_gi, rdi.localKd_gi, rdi.localPr_gi, rdi.localPm_gi, rdi.etai_gi, rdi.etat_gi, rdi.J_gi.x, 1.0f, false, Jnc, Jcc) * visReuse;
        float p_c = GetPHat(contrib_c);
        float3 contrib_final = contrib_c;
        // Compute the pairwise MIS weight for the canonical sample
        float mis_c = 0.0f;
        if(rdi.M_gi <= SPAT_MIN_M_GI){
            mis_c = PairwiseMIS_Canonical_Spat_GI_Sym(M_sum_sym, p_c, M_c, nIds, rdi.x2_gi, rdi.n2_s_gi, rdi.n2_g_gi, rdi.L2_gi, rdi.V2_gi, rdi.matID_gi, rdi.localKd_gi, rdi.localPr_gi, rdi.localPm_gi, rdi.etai_gi, rdi.etat_gi, rdi.J_gi.x, rdi.J_gi.y, SPAT_BETA_GI);
        }
        else{
            mis_c = PairwiseMIS_Canonical_Spat_GI(M_sum, p_c, M_c, nIds, rdi.x2_gi, rdi.n2_s_gi, rdi.n2_g_gi, rdi.L2_gi, rdi.V2_gi, rdi.matID_gi, rdi.localKd_gi, rdi.localPr_gi, rdi.localPm_gi, rdi.etai_gi, rdi.etat_gi, rdi.J_gi.x, rdi.J_gi.y);
        }
        debug += mis_c;
        // Adjust the weight in the canonical reservoir
        rdi.w_sum_gi = mis_c * p_c * rdi.W_gi;

        // ########################################### NODE #############################################################
        // Iterate through all valid neighbors and update the canonical reservoir with them
        [unroll(SPAT_COUNT_MAX_GI)]
        for(int i = 0; i < SPAT_COUNT_MAX_GI; i++){
            if(nIds[i] != 0xFFFFFFFF){
                // Calculate p_hat for the neighbor using the canonical sample position
                Reservoir_GI rdi_r = loadReservoirGI(g_Reservoirs_current_gi, nIds[i]);
                float Jn = 0.0f;
                float Jnn = 1.0f;
                float3 contrib_n = ReconnectGI(sdata.x1, sdata.n1_s, sdata.n1_g, sdata.o, sdata.matID, sdata.localKd, sdata.localPr, sdata.localPm, sdata.etai, sdata.etat, rdi_r.matID_gi, rdi_r.x2_gi, rdi_r.n2_s_gi, rdi_r.n2_g_gi, rdi_r.L2_gi, rdi_r.V2_gi, rdi_r.localKd_gi, rdi_r.localPr_gi, rdi_r.localPm_gi, rdi_r.etai_gi, rdi_r.etat_gi, rdi_r.J_gi.x, rdi_r.J_gi.y, true, Jn, Jnn);
                contrib_n *= VisibilityCheckCP(sdata.x1, rdi_r.x2_gi, sdata.n1_s, 0u);
                float p_hat_from = GetPHat(contrib_n) * Jnn;
                // Calculate the samples MIS weight - low canonical M: use symmetric ratio, high M: use pairwise MIS. Why? Because if M is low, the image is more likely to contain correlations
                float mis_n = 0.0f;
                if(rdi.M_gi <= SPAT_MIN_M_GI)
                    mis_n = PairwiseMIS_Neighbor_Spat_GI_Sym(M_sum_sym, M_c, min(SPAT_MCAP_GI ,rdi_r.M_gi), p_c, p_hat_from, nIds[i], rdi_r.x2_gi, rdi_r.n2_s_gi, rdi_r.n2_g_gi, rdi_r.L2_gi, rdi_r.V2_gi, rdi_r.matID_gi, rdi_r.localKd_gi, rdi_r.localPr_gi, rdi_r.localPm_gi, rdi_r.etai_gi, rdi_r.etat_gi, rdi_r.J_gi.x, SPAT_BETA_GI);
                else
                    mis_n = PairwiseMIS_Neighbor_Spat_GI(M_sum, M_c, min(SPAT_MCAP_GI ,rdi_r.M_gi), p_c, p_hat_from, nIds[i], rdi_r.x2_gi, rdi_r.n2_s_gi, rdi_r.n2_g_gi, rdi_r.L2_gi, rdi_r.V2_gi, rdi_r.matID_gi, rdi_r.localKd_gi, rdi_r.localPr_gi, rdi_r.localPm_gi, rdi_r.etai_gi, rdi_r.etat_gi, rdi_r.J_gi.x);
                debug += mis_n;
                // Calculate the sample weight
                float w_n = mis_n * p_hat_from * rdi_r.W_gi;

                // Update the reservoir
                if(UpdateReservoirGI(rdi, w_n, min(SPAT_MCAP_GI ,rdi_r.M_gi), rdi_r.x2_gi, rdi_r.n2_s_gi, rdi_r.n2_g_gi, rdi_r.L2_gi, rdi_r.V2_gi, rdi_r.localKd_gi, rdi_r.localPr_gi, rdi_r.localPm_gi, rdi_r.etai_gi, rdi_r.etat_gi, rdi_r.matID_gi, rdi_r.objID_gi, rdi_r.J_gi, rdi_r.F_gi, seed)){
                    contrib_final = contrib_n;
                    rdi.J_gi.y = Jn;
                }

            }
        }
        // ########################################### NODE #############################################################

        // Calculate new W
        float p_hat_final = GetPHat(contrib_final) /* VisibilityCheckCP(sdata.x1, rdi.x2_gi, sdata.n1)*/;
        if (p_hat_final > EPSILON && rdi.w_sum_gi > 0.0f && rdi.w_sum_gi < 1e10f) {
            float W = rdi.w_sum_gi / p_hat_final;
            // NaN/Inf protection
            if (isnan(W) || isinf(W)) {
                W = 0.0f;
            }
            rdi.W_gi = W;
        }
        else
            rdi.W_gi = 0.0f;

        // Store the final output
        gScratchPing[uint3(tid.xy, 2)] = float4(contrib_final * rdi.W_gi, 0);

        // DEBUG
        /*float3 heat;
        heat.r = step(debug, 0.9);          // red when <1
        heat.g = saturate(1 - abs(debug-1)); // green at exactly 1
        heat.b = step(1.1, debug);          // blue when >1
        gOutput[uint3(tid.xy, 0)] = float4(heat, 1);*/
        //gOutput[uint3(tid.xy, 0)] = float4(rdi.W_gi * 0.1f, rdi.W_gi* 0.1f, rdi.W_gi* 0.1f, 1.0f);

    }
    else
        gScratchPing[uint3(tid.xy, 2)] = float4(sdata.L1, 0);

    rdi.J_gi.y = PSSJacobian(sdata.x1, sdata.n1_s, sdata.n1_g, sdata.o, sdata.matID, sdata.localKd, sdata.localPr, sdata.localPm, sdata.etai, sdata.etat, rdi.x2_gi, rdi.n2_s_gi, rdi.n2_g_gi, rdi.V2_gi, rdi.matID_gi, rdi.localKd_gi, rdi.localPr_gi, rdi.localPm_gi, rdi.etai_gi, rdi.etat_gi, rdi.J_gi.x);

    // Store the merged reservoir
    storeReservoirGI(g_Reservoirs_last_gi, pixelIdx, rdi);
}