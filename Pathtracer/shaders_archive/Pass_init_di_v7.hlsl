#include "Includes_v7.hlsli"

//─────────────────────────────────────────────────────────────────────────────
//  Initial Sampling DI
//─────────────────────────────────────────────────────────────────────────────
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= IMG_W || tid.y >= IMG_H) return;
    gDispatchIdx = tid;

    uint2  launchIndex   = tid.xy;
    float2 dims = float2(IMG_W, IMG_H);
    uint   pixelIdx  = MapPixelID(dims, launchIndex);

    //SampleData sdata = SampleCameraRay(pixelIdx);
    SampleData sdata = SampleCameraRay(pixelIdx, launchIndex, dims);
    //gOutput[uint3(tid.xy, 0)] = float4(abs(sdata.n1),1);
    //gOutput[uint3(tid.xy, 0)] = float4(sdata.L1,1);

    if(sdata.matID != 4294967294 && all(sdata.L1 < EPSILON)){
        // Get a random seed
        uint2 seed = GetSeed(pixelIdx, time, 1);
        uint waveSeed = GetWaveSeed(pixelIdx, uint2(8,8), time, 1);

        Reservoir_DI reservoir = (Reservoir_DI)0;
        float phat_final = 0.0f;
        uint n_nee_eff = NEE_SAMPLES_DI;
        // NEE sample(s)
        for (int i = 0; i < NEE_SAMPLES_DI; i++) {
            // Get the sample result
            SampleReturn result = SampleNEE(sdata, waveSeed, seed);
            if (any(result.L2 > 0.0f)) {
                // Calculate contribution and p_hat.
                float3 c = ReconnectDI(sdata.x1, sdata.n1, sdata.o, sdata.matID, result.x2, result.n2, result.L2);
                float p_hat = GetPHat(c);
                float w_mis = MIS_Initial_NEE(result.pdf_nee, result.pdf_bsdf, NEE_SAMPLES_DI, BSDF_SAMPLES_DI) * p_hat / result.pdf_nee;
                if (isnan(w_mis)) w_mis = 0.0f;
                // Update reservoir
                if (UpdateReservoirDI(reservoir, w_mis, 0, result.x2, result.n2, result.L2, result.objID, seed))
                    phat_final = p_hat;
            }
        }

        bool requires_shadow_ray = true;
        // BSDF sample(s)
        for(int j = 0; j<BSDF_SAMPLES_DI; j++){
            // Get the sample result
            SampleReturn result = SampleBSDF(sdata, waveSeed, seed);
            // Calculate contribution and p_hat.
            if(any(result.L2 > 0.0f)){
                float3 c = ReconnectDI(sdata.x1, sdata.n1, sdata.o, sdata.matID, result.x2, result.n2, result.L2);
                float p_hat = GetPHat(c);
                float w_mis = MIS_Initial_BSDF(result.pdf_nee, result.pdf_bsdf, NEE_SAMPLES_DI, BSDF_SAMPLES_DI) * p_hat / result.pdf_bsdf;
                if(isnan(w_mis) || isinf(w_mis))
                    w_mis = 0.0f;
                // Update reservoir
                if( UpdateReservoirDI(reservoir, w_mis, 0, result.x2, result.n2, result.L2, result.objID, seed)){
                    requires_shadow_ray = false;
                    phat_final = p_hat;
                }
                store_n2_init(float3(0,0,0), g_InitialBSDFRays, pixelIdx);
            }
            else{
                if(length(result.n2) > 0.0f){
                    // Store the sample for the GI pass!
                    store_dir_init(normalize(result.x2-sdata.x1), g_InitialBSDFRays, pixelIdx);
                    store_x2_init(result.x2, g_InitialBSDFRays, pixelIdx);
                    store_n2_init(result.n2, g_InitialBSDFRays, pixelIdx);
                    store_objID_init(result.objID, g_InitialBSDFRays, pixelIdx);
                    store_matID_init(result.matID, g_InitialBSDFRays, pixelIdx);
                    store_pdfB_init(result.pdf_bsdf, g_InitialBSDFRays, pixelIdx);
                    store_pdfN_init(result.pdf_nee, g_InitialBSDFRays, pixelIdx);
                }
            }
        }

        // Visbility check for the stored sample, if fail, set W to 0
        float V = 1.0f;
        if(requires_shadow_ray){
            V = VisibilityCheckCP(sdata.x1, reservoir.x2_di, sdata.n1);
        }
        // Calculate W
        reservoir.W_di = 0.0f;
        reservoir.L2_di *= V;
        reservoir.M_di = 1u;
        if (phat_final > EPSILON) {
            reservoir.W_di = V * reservoir.w_sum_di / phat_final;
            // Protect against NaN/Inf
            if (isnan(reservoir.W_di) || isinf(reservoir.W_di)) {
                reservoir.W_di = 0.0f;
                reservoir.L2_di = 0.0f;
            }
        }

        // Save the resulting reservoir to memory
        storeReservoirDI(g_Reservoirs_current_di, pixelIdx, reservoir);
        /*store_x2_di(reservoir.x2_di, g_Reservoirs_current_di, pixelIdx, reservoir.objID_di);
        store_n2_di(reservoir.n2_di, g_Reservoirs_current_di, pixelIdx, reservoir.objID_di);
        store_L2_di(reservoir.L2_di, g_Reservoirs_current_di, pixelIdx);
        store_W_di(reservoir.W_di, g_Reservoirs_current_di, pixelIdx);
        store_objID_di(reservoir.objID_di, g_Reservoirs_current_di, pixelIdx);
        store_M_di(1, g_Reservoirs_current_di, pixelIdx);*/
    }
}