#include "Includes_v8.hlsli"

//─────────────────────────────────────────────────────────────────────────────
//  Initial Sampling DI
//─────────────────────────────────────────────────────────────────────────────
[numthreads(8, 4, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= IMG_W || tid.y >= IMG_H) return;
    gDispatchIdx = tid;

    uint2  launchIndex   = tid.xy;
    float2 dims = float2(IMG_W, IMG_H);
    uint   pixelIdx  = MapPixelID(dims, launchIndex);
    // Seeding
    RandomData rdata = initRandomData(launchIndex, uint2(8,8), time, 1u);

    // Trace the initial camera rays and store the sdata.
    SampleData sdata = SampleCameraRay(pixelIdx, launchIndex, dims);

    // If we hit an emitter (or the sky), color the pixel and return
    if(any(sdata.L1 > 0.0f)){
        gScratchPing[uint3(tid.xy, 1)] = float4(sdata.L1, 0);
        return;
    }

    // Create the path state + throughput state object from the sdata information
    PathState pstate = InitPathState(sdata.x1, sdata.n1_g, sdata.n1_s, sdata.o, sdata.objID, sdata.matID);
    ThroughputState tstate = InitThroughputState();

    for(int i = 0; i < 10; i++){
        // Sample a direction and hitpoint using bsdf sampling
        SampleState sstate = Sample_BSDF_BW_S(pstate, rdata);

        // Is the sample well defined?
        if(!ValidSampleState(sstate)){
            gScratchPing[uint3(tid.xy, 1)] = float4(0,0,0,0);
            return;
        }

        // Evaluate throughput and adjust path throughput
        float3 tpp = EvaluateBRDF_COMBINED(pstate.matID, pstate.n_g, pstate.n_s, -sstate.s, pstate.o) * dot(pstate.n_s, sstate.s);
        float pdfp = BRDF_PDF_COMBINED(pstate.matID, pstate.n_g, pstate.n_s, -sstate.s, pstate.o);

        // If the pdf is invalid (==0), break;
        if(pdfp == 0.0f && !isnan(pdfp)){
            gScratchPing[uint3(tid.xy, 1)] = float4(0,0,0,0);
            return;
        }

        // Did we hit a light? If so, terminate path, write color and return
        if(any(sstate.L > 0.0f)){
            float3 tp_final = tpp * tstate.t * sstate.L;
            float pdf_final = pdfp * tstate.pdf;
            gScratchPing[uint3(tid.xy, 1)] = float4(tp_final/pdf_final, 0);
            return;
        }

        tstate.t *= tpp;
        tstate.pdf *= pdfp;

        // Advance the path state
        AdvancePathState(sstate, pstate);
    }

    gScratchPing[uint3(tid.xy, 1)] = float4(0,0,0,0);

    //Debug
    //gScratchPing[uint3(tid.xy, 1)] = float4((sdata.n1_s+1.0f)/2.0f, 0.0f); // Norm
    //gScratchPing[uint3(tid.xy, 1)] = float4(sdata.x1, 0.0f); // Pos
    //gScratchPing[uint3(tid.xy, 1)] = float4(materials[sdata.matID].Kd.xyz, 0.0f); // Material
    //gScratchPing[uint3(tid.xy, 1)] = float4(sdata.L1, 0.0f); // Emission

}