#include "Includes_v7.hlsli"

//─────────────────────────────────────────────────────────────────────────────
//  Initial Sampling GI
//─────────────────────────────────────────────────────────────────────────────
[numthreads(16, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= IMG_W || tid.y >= IMG_H) return;
    gDispatchIdx = tid;

    uint2  launchIndex   = tid.xy;
    float2 dims = float2(IMG_W, IMG_H);
    uint   pixelIdx  = MapPixelID(dims, launchIndex);

    // Initialize the GI reservoir and trace the path (2-4 consecutive bsdf rays)
    Reservoir_GI reservoir = (Reservoir_GI)0;
    // Get a random seed
    uint2 seed = GetSeed(pixelIdx, time, 1);
    uint waveSeed = GetWaveSeed(pixelIdx, time, 1);

    // Load sample data
    SampleData sdata = loadSampleData(g_sample_current, pixelIdx);

    // Probs:
    float3 sProbs1 = BD_MethodProbsFromRoughness(1.0f);
    uint pickID1 = BD_PickMethod(sProbs1, seed);
    // Sample the selected method:
    BDReturn bdreturn1 = BD_SamplePicked(sdata.x1, sdata.n1, sdata.matID, sdata.o, waveSeed, seed, pickID1);
    // Get the combined pdf
    float bdpdf1 = BD_OneSamplePDF(bdreturn1.pdf, pickID1, sdata, bdreturn1, sProbs1);

    float3 segment_throughput = 0.0f;
    float3 debug = ReconnectGIBD_Simple(sdata.x1, sdata.n1, sdata.o, sdata.matID, bdreturn1.x2, bdreturn1.n2, bdreturn1.matID, bdreturn1.x3, bdreturn1.n3,bdreturn1.L2, bdpdf1, segment_throughput);

if(bdreturn1.pdf_seg > EPSILON && pickID1!=1u && length(bdreturn1.n2) > EPSILON){
    // More path variables
    float3 position_cache = bdreturn1.x2;
    float3 normal_cache = normalize(bdreturn1.n2);
    float3 outgoing_cache = normalize(sdata.x1 - bdreturn1.x2);
    uint matID_cache = bdreturn1.matID;

    float3 throughput_2 = segment_throughput;
    float pdf_2 = bdreturn1.pdf_seg;

    // Probs:
    float3 sProbs2 = BD_MethodProbsFromRoughness(1.0f);
    uint pickID2 = BD_PickMethod(sProbs2, seed);
    // Sample the selected method:
    BDReturn bdreturn2 = BD_SamplePicked(position_cache, normal_cache, matID_cache, outgoing_cache, waveSeed, seed, pickID2);

    SampleData sdata2;
    sdata2.x1    = position_cache;   // x2 of the previous step
    sdata2.n1    = normal_cache;     // n2
    sdata2.matID = matID_cache;
    sdata2.o     = outgoing_cache;   // normalized (x1 - x2)
    // Get the combined pdf
    float bdpdf2 = BD_OneSamplePDF(bdreturn2.pdf, pickID2, sdata2, bdreturn2, sProbs2);

    float3 segment_throughput2;
    debug += throughput_2/pdf_2 * ReconnectGIBD_Simple(position_cache, normal_cache, outgoing_cache, matID_cache, bdreturn2.x2, bdreturn2.n2, bdreturn2.matID, bdreturn2.x3, bdreturn2.n3, bdreturn2.L2, bdpdf2, segment_throughput2);
}

    // Get a sample
    //SampleReturn result_init = SampleBSDF_gen(sdata.x1, sdata.n1, sdata.matID, sdata.o, waveSeed, seed);
    /*if(length(result_init.n2) > 0.0f && length(result_init.L2) == 0.0f){
        bool requires_shadow_ray = true; // Set to false whenever a bsdf ray wins beeing added to the reservoir in the end
        float p_hat_final = 0.0f; // P hat cache from the iteration -> no recompute required.
        // Postponed shadow ray
        float3 s_x1 = (float3)0;
        float3 s_x2 = (float3)0;
        float3 s_n1 = (float3)0;

        // Store path variables that are used to fill the reservoirs
        float3 V2 = (float3)0;

        // Store path variables
        // Variables to cache path data
        float3 position = result_init.x2;
        float3 normal = result_init.n2;
        float3 outgoing = normalize(sdata.x1 - position);
        uint matID = result_init.matID;
        // full path throughput
        float3 tp_full = ReconnectGISingle(sdata.x1, sdata.n1, sdata.o, sdata.matID, position, normal, result_init.pdf_bsdf); // reconnect without L

        // partial path throughput (from x3 onward, used to set L2 in the reservoir)
        float3 tp_partial = float3(1,1,1);

        // Full path pdf of a given subpath
        float3 ldir = position - sdata.x1;
        float dist2 = length(ldir) * length(ldir);
        float pdf_full = result_init.pdf_bsdf;

        // For reconnection shift, this is easy
        float J_prefix = result_init.pdf_bsdf * dot(normalize(-ldir), normalize(result_init.n2)) / dist2;
        for(int i = 0; i < BSDF_SAMPLES_GI; i++){
            {
                // NEE samples
                for(int j = 0; j<NEE_SAMPLES_GI; j++){
                    // Get the sample result
                    SampleReturn result = SampleNEE_gen(position, normal, matID, outgoing, waveSeed, seed);
                    if(any(result.L2 > 0.0f)){
                        // Calculate contribution and p_hat.
                        float3 c = ReconnectGISingle(position, normal, outgoing, matID, result.x2, result.n2, result.pdf_nee);
                        float p_hat = GetPHat(c * tp_full * result.L2);
                        float w_mis = MIS_Initial_NEE(result.pdf_nee, result.pdf_bsdf, NEE_SAMPLES_GI, 1) * p_hat;

                        //length(result.x2 - position) < 1.0f ? w_mis = 0.0f : w_mis = w_mis;
debug += c * tp_full * result.L2 * VisibilityCheckCP(position, result.x2, normal);

                        if(isnan(w_mis))
                            w_mis = 0.0f;

                        float3 L2 = result.L2;
                        float3 V2_temp = V2;
                        if(i != 0)
                            L2 *= c * tp_partial;
                        else {
                            V2_temp = position - result.x2;
                            float3 V2_norm = normalize(V2_temp);
                            L2 *= G_term(normal, V2_norm);
                        }
                        // Update reservoir
                        if(UpdateReservoirGI(reservoir, w_mis, 0, 0, 0, L2, normalize(V2_temp), 0, 0, 0, 0, 0, 0, 0, 0, seed)){
                            p_hat_final = p_hat;
                            s_x1 = position;
                            s_x2 = result.x2;
                            s_n1 = normal;

                            // Also store the Jacobian part for this canonical sample, which is pdfxk * pdfxk+1 * cosx2 / dist^2
                            reservoir.J_gi.y = J_prefix;
                            if(i==0)
                                reservoir.J_gi.y *= result.pdf_nee;

                            // Store the NEE pdf in the reservoir as well in Jx -> its not dependant on the outgoing direction
                            if(i == 0)
                                reservoir.J_gi.x = result.pdf_nee; // Only if i == 0, we perform a reconnection with nee pdf, otherwise theres another ray inbetween and we use bsdf
                        }
                    }
                }
            }

            // BSDF advancement
            {
                // Get a sample direction
                SampleReturn result = SampleBSDF_gen(position, normal, matID, outgoing, waveSeed, seed);
                float3 c = ReconnectGISingle(position, normal, outgoing, matID, result.x2, result.n2, result.pdf_bsdf);
                // Calculate contribution and p_hat.
                if(any(result.L2 > 0.0f)){
                    tp_full *= c;
                    float p_hat = GetPHat(tp_full * result.L2);
debug += tp_full * result.L2;
                    float w_mis = MIS_Initial_BSDF(result.pdf_nee, result.pdf_bsdf, NEE_SAMPLES_GI, 1) * p_hat;
                    if(isnan(w_mis) || isinf(w_mis))
                        w_mis = 0.0f;

                    float3 L2 = result.L2;
                    float3 V2_temp = V2;
                    if(i != 0)
                        L2 *= tp_partial * c;
                    else {
                        V2_temp = position - result.x2;
                        float3 V2_norm = normalize(V2_temp);
                        L2 *= G_term(normal, V2_norm);
                    }
                    // Update reservoir with the sub path
                    if(UpdateReservoirGI(reservoir, w_mis, 0, 0, 0, L2, normalize(V2_temp), 0, 0, 0, 0, 0, 0, 0, 0, seed)){
                        requires_shadow_ray = false;
                        p_hat_final = p_hat;
                        // Also store the Jacobian part for this canonical sample
                        reservoir.J_gi.y = J_prefix;
                        if(i==0)
                            reservoir.J_gi.y *= result.pdf_bsdf;
                    }
                    break;
                }
                else{
                    if(any(result.n2 > 0.0f)){
                        if(i == 0){
                            V2 = normalize(position - result.x2);
                            tp_partial *= G_term(normal, normalize(V2));
                            J_prefix *= result.pdf_bsdf;
                        }
                        else
                            tp_partial *= c;

                        // Advance ray
                        pdf_full *= result.pdf_bsdf;
                        tp_full *= c;
                        outgoing = normalize(position - result.x2);
                        position = result.x2;
                        normal = result.n2;
                        matID = result.matID;
                    }
                    else
                        break;
                }
            }
        }

       // Visbility check for the stored sample, if fail, set W to 0
        float V = 1.0f;
        if(requires_shadow_ray && length(s_n1)>EPSILON){
            V = VisibilityCheckCP(s_x1, s_x2, s_n1);
        }
        reservoir.L2_gi *= V;
        // Calculate W
        float W = 0.0f;
        if (p_hat_final > EPSILON) {
            W = V * reservoir.w_sum_gi / p_hat_final;
            // Protect against NaN/Inf
            if (isnan(W) || isinf(W)) {
                reservoir.n2_gi = 0; // Dont pick this sample, its nan!
            }
        }
        reservoir.W_gi = W;

        reservoir.objID_gi = result_init.objID;
        reservoir.matID_gi = result_init.matID;
        reservoir.x2_gi = result_init.x2;
        reservoir.n2_gi = result_init.n2;
        reservoir.M_gi = 1;

        SampleData sdata = loadSampleData(g_sample_current, pixelIdx);
        reservoir.J_gi.y = PSSJacobian(sdata.x1, sdata.n1, sdata.o, sdata.matID, reservoir.x2_gi, reservoir.n2_gi, reservoir.V2_gi, reservoir.matID_gi, reservoir.J_gi.x);
        reservoir.F_gi = p_hat_final;
        reservoir.J_gi.z = result_init.pdf_nee;
    }
    storeReservoirGI(g_Reservoirs_current_gi, pixelIdx, reservoir);*/
gScratchPing[uint3(tid.xy, 2)] = float4(debug, 0.0f);
}
