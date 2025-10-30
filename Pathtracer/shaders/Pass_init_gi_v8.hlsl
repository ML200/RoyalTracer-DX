#include "Includes_v7.hlsli"

//─────────────────────────────────────────────────────────────────────────────
//  Initial Sampling GI
//─────────────────────────────────────────────────────────────────────────────
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= IMG_W || tid.y >= IMG_H) return;
    gDispatchIdx = tid;

    uint2  launchIndex   = tid.xy;
    float2 dims = float2(IMG_W, IMG_H);
    uint   pixelIdx  = MapPixelID(dims, launchIndex);

    // Load sample data & get a random seed to prepare the tracing
    SampleData sdata = loadSampleData(g_sample_current, pixelIdx);
    uint2 seed = GetSeed(pixelIdx, time, 1);
    uint waveSeed = GetWaveSeed(pixelIdx, uint2(8,8), time, 1);

    // Probs:
    float3 sProbs1 = BD_MethodProbsFromRoughness(1.0f);
    uint pickID1 = BD_PickMethod(sProbs1, waveSeed);
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
        uint pickID2 = BD_PickMethod(sProbs2, waveSeed);
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

    gScratchPing[uint3(tid.xy, 2)] = float4(debug, 0.0f);
}
