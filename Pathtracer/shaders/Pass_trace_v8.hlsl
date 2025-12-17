#include "Includes_v8.hlsli"

[numthreads(8, 4, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= IMG_W || tid.y >= IMG_H) return;
    gDispatchIdx = tid;

    uint2  launchIndex   = tid.xy;
    float2 dims = float2(IMG_W, IMG_H);
    uint   pixelIdx  = MapPixelID(dims, launchIndex);

    // Load path state
    PathState pstate = loadPathState(g_pathStateBuffer, pixelIdx);
    if(pstate.matID == 0x0000FFFF){return;}

    // Sample a direction and hitpoint using bsdf sampling
    SampleState sstate = Sample_BSDF_BW_S(pstate);


    // Is the sample well defined?
    if(!ValidSampleState(sstate)){
        pstate.matID = 0x0000FFFF;
        storePathState(g_pathStateBuffer, pixelIdx, pstate);
        gScratchPing[uint3(tid.xy, 1)] = float4(0,0,0,0);
        return;
    }

    // Evaluate throughput and adjust path throughput
    float2 iors = GetIORs(pstate);
    float3 tpp = EvaluateBRDF_COMBINED(pstate, sstate, iors.x, iors.y) * abs(dot(pstate.n_s, sstate.s));
    float pdfp = BRDF_PDF_COMBINED(pstate, sstate, iors.x, iors.y);

    // If the pdf is invalid (==0), break;
    if(pdfp == 0.0f || isnan(pdfp) || isinf(pdfp)){
        pstate.matID = 0x0000FFFF;
        storePathState(g_pathStateBuffer, pixelIdx, pstate);
        gScratchPing[uint3(tid.xy, 1)] = float4(0,0,0,0);
        return;
    }

    // Update the stack
    if (dot(pstate.n_g, sstate.s) < 0.0f) { // Did we enter/exit a substrate?
        UpdateIORStack(pstate);
    }

    uint currentMatID = GetCurrentMediumMaterialID(pstate);
    float3 tint = currentMatID!=0x0000FFFF? CalculateAbsorptionThroughput(materials[pstate.matID].Tf, length(pstate.x - sstate.x)) : (float3)1.0f;
    tpp *= tint;

    // Did we hit a light? If so, terminate path, write color and return
    if(any(sstate.L > 0.0f)){
        pstate.matID = 0x0000FFFF;
        float3 tp_final = tpp * pstate.t * sstate.L / pdfp;
        float3 c_final = tp_final;
        if(any(isnan(c_final)) || any(isinf(c_final))){
            pstate.matID = 0x0000FFFF;
            storePathState(g_pathStateBuffer, pixelIdx, pstate);
            gScratchPing[uint3(tid.xy, 1)] = float4(0,0,0,0);
            return;
        }
        pstate.matID = 0x0000FFFF;
        storePathState(g_pathStateBuffer, pixelIdx, pstate);
        gScratchPing[uint3(tid.xy, 1)] = float4(c_final, 0);
        return;
    }

    pstate.t *= tpp/pdfp;

    // Advance the path state
    AdvancePathState(sstate, pstate);

    gScratchPing[uint3(tid.xy, 1)] = float4(0,0,0,0); // TODO remove this

    // Take a sample
    float2 iors2 = GetIORs(pstate);
    float3 s = SampleBRDF(pstate, iors2.x, iors2.y);

    // update the path state object
    pstate.s = s;

    if(length(s) == 0.0f)
        pstate.matID = 0x0000FFFF;

    // Store the new path state
    storePathState(g_pathStateBuffer, pixelIdx, pstate);
}