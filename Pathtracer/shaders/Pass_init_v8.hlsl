#include "Includes_v8.hlsli"

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= IMG_W || tid.y >= IMG_H) return;
    gDispatchIdx = tid;

    uint2  launchIndex = tid.xy;
    float2 dims = float2(IMG_W, IMG_H);
    uint   pixelIdx  = MapPixelID(dims, launchIndex);

    // Seeding
    uint seed = initRandomData(launchIndex, uint2(8,8), time, 1u);

    // 1. Trace Camera Ray
    SampleData sdata = SampleCameraRay(pixelIdx, launchIndex, dims, seed);

    // 2. Camera Ray Hit Light? (Primary Visibility)
    if(any(sdata.L1 > 0.0f)){
        gScratchPing[uint3(tid.xy, 1)] = float4(sdata.L1, 0); // Accumulate direct emission
        return;
    }

    // Initialize Path State data
    PathPayload pload = InitPathPayload(seed);
    VolumeIOR vior = InitVolumeIOR();
    VolumeAux aior = InitVolumeAux();

    // Sample material and store direction
    float2 iors = GetIORs(vior, aior, sdata.matID, sdata.objID);

    SamplingP p1 = CalculateStrategyProbabilities(sdata.matID, sdata.o, sdata.n1_s, iors.x, iors.y, sdata.localKd, sdata.localPm);
    float3 s = SampleBRDF(p1, sdata.matID, sdata.o, sdata.n1_s, sdata.n1_g, sdata.localKd, sdata.localPr, sdata.localPm, pload.seed, iors.x, iors.y, vior.pointer);

    // Update the ior stack with the new sample
    if (dot(sdata.n1_g, s) < 0.0f) { // Did we enter/exit a substrate?
        UpdateIORStack(vior, aior, sdata.matID, sdata.objID);
    }

    // Store raygeom
    RayGeometry rgeom = InitRayGeometry(sdata.x1, s);
    StoreRayGeometry(g_pathStateBuffer, pixelIdx, rgeom);
    //UpdateOriginBounds(sdata.x1);

    StoreVolumeIOR(g_pathStateBuffer, pixelIdx, vior);
    StoreVolumeAux(g_pathStateBuffer, pixelIdx, aior);

    SamplingP p2 = CalculateStrategyProbabilities(sdata.matID, sdata.o, sdata.n1_s, iors.x, iors.y, sdata.localKd, sdata.localPm);
    float  pdf   = BRDF_PDF_COMBINED(p2, sdata.matID, sdata.n1_s, sdata.n1_g, s, sdata.o, sdata.localKd, sdata.localPr, sdata.localPm, iors.x, iors.y);

    // evaluate BSDF immediately
    float3 f_val = EvaluateBRDF_COMBINED(sdata.matID, sdata.n1_s, sdata.n1_g, s, sdata.o, sdata.localKd, sdata.localPr, sdata.localPm, iors.x, iors.y);
    float  cosTheta = abs(dot(sdata.n1_s, s));

    // Validation & Throughput Update
    if(length(s) == 0.0f || pdf <= 1e-6f || any(isnan(f_val))) {
        // Kill path immediately if sample is invalid
        gScratchPing[uint3(tid.xy, 1)] = float4(0,0,0,0);
        return;
    }

    // Update Throughput for the ray we are about to shoot
    pload.throughput *= (f_val * cosTheta) / pdf;

    // Store the payload
    StorePathPayload(g_pathStateBuffer, pixelIdx, pload);

    PushStack(g_InputStackIdx, uint2(pixelIdx,0));
}