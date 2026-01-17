#include "Includes_v8.hlsli"

//─────────────────────────────────────────────────────────────────────────────
//  TEMPORAL  DI
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
        Reservoir_DI rdi = loadReservoirDI(g_Reservoirs_current_di, pixelIdx);
        // Get a random seed
        uint2 seed = GetSeed(pixelIdx, time, 2);
        SampleData sdata_r;
        Reservoir_DI rdi_r;
        //uint tempPixelIdx = 0xFFFFFFFF;

        int2 tempPixelCoordinate = GetBestReprojectedPixel_d(sdata.x1, prevView, prevProjection, dims, sdata.objID);
        if(tempPixelCoordinate.x == -1 && tempPixelCoordinate.y == -1)
            tempPixelCoordinate = launchIndex;

        uint tempPixelIdx = MapPixelID(dims, tempPixelCoordinate);
        sdata_r = loadSampleData(g_sample_last, tempPixelIdx);
        rdi_r = loadReservoirDI(g_Reservoirs_last_di, tempPixelIdx);
        bool valid =
                (all(sdata_r.L1 < EPSILON) &&
                IsValidReservoir_DI(rdi_r) &&
                !RejectNormal_DI(sdata.n1_s, sdata_r.n1_s, 0.36f) &&
                (!RejectDistance_DI(sdata.x1, sdata_r.x1, sdata.n1_s, 0.05f))  &&
                (sdata_r.matID == sdata.matID));

        if(tempPixelIdx != 0xFFFFFFFF /*&& valid_history*/ && valid){
            // Calculate the canonical target function
            float visReuse_c = rdi.W_di > 0.0f ? 1.0f : 0.0f;
            float p_c = GetPHat(ReconnectDI(sdata.x1, sdata.n1_s, sdata.n1_g, sdata.o, sdata.matID, rdi.x2_di, rdi.n2_di, rdi.L2_di, sdata.localKd, sdata.localPr, sdata.localPm, sdata.etai, sdata.etat)) * visReuse_c;
            float p_n = GetPHat(ReconnectDI(sdata_r.x1, sdata_r.n1_s, sdata_r.n1_g, sdata_r.o, sdata_r.matID, rdi.x2_di, rdi.n2_di, rdi.L2_di, sdata_r.localKd, sdata_r.localPr, sdata_r.localPm, sdata_r.etai, sdata_r.etat));// * VisibilityCheckCP(sdata_r.x1, rdi.x2_di, sdata_r.n1); // would require last frame AS and we dont store it, can be ommited for minimal added bias
            float n_c = GetPHat(ReconnectDI(sdata.x1, sdata.n1_s, sdata.n1_g, sdata.o, sdata.matID, rdi_r.x2_di, rdi_r.n2_di, rdi_r.L2_di, sdata.localKd, sdata.localPr, sdata.localPm, sdata.etai, sdata.etat)) * VisibilityCheckCP(sdata.x1, rdi_r.x2_di, sdata.n1_g);
            float visReuse = rdi_r.W_di > 0.0f ? 1.0f : 0.0f;
            float n_n = GetPHat(ReconnectDI(sdata_r.x1, sdata_r.n1_s, sdata_r.n1_g, sdata_r.o, sdata_r.matID, rdi_r.x2_di, rdi_r.n2_di, rdi_r.L2_di, sdata_r.localKd, sdata_r.localPr, sdata_r.localPm, sdata_r.etai, sdata_r.etat)) * visReuse;
            float M_c = min(TEMP_MCAP_DI,rdi.M_di);
            float M_n = min(TEMP_MCAP_DI,rdi_r.M_di);
            float M_sum = M_c + M_n;
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
            if (p_hat_final > EPSILON && rdi.w_sum_di > EPSILON && rdi.w_sum_di < 1e10f) {
                float W = rdi.w_sum_di / p_hat_final;
                // NaN/Inf protection
                if (isnan(W) || isinf(W)) {
                    W = 0.0f;
                }
                rdi.W_di = W;
            }
            else
                rdi.W_di = 0.0f;

            // Store the merged reservoir
            storeReservoirDI(g_Reservoirs_current_di, pixelIdx, rdi);
        }
    }
}
