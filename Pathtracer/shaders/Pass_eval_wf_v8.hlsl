#include "Includes_v8.hlsli"

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    // 1. Stack Management
    uint waveIdx = tid.x;
    uint activeCount = g_GlobalCounters.Load(g_InputStackIdx * 4);
    if (waveIdx >= activeCount) return;

    uint pixelIdx = PopStack(g_InputStackIdx, waveIdx);
    uint2 coord = UnmapPixelID(pixelIdx, gImageSize);

    // Load the hitstate data from the trace
    HitState hstate = LoadHitState(g_pathStateBuffer, pixelIdx);

    // Load ray origin first
    float3 origin = LoadRayOrigin(g_pathStateBuffer, pixelIdx);
    HitInfo hinfo = EvalSurfaceState(hstate.instanceID, hstate.primitiveID, hstate.bary, origin);

    // Load Path State data
    PathPayload pload = LoadPathPayload(g_pathStateBuffer, pixelIdx);
    VolumeIOR vior = LoadVolumeIOR(g_pathStateBuffer, pixelIdx);
    VolumeAux aior = LoadVolumeAux(g_pathStateBuffer, pixelIdx);

    // Sample material and store direction
    float2 iors = GetIORs(vior, aior, hinfo.materialID, hinfo.objID);
    float3 o = normalize(origin - hinfo.hitPosition);

    // Calculate sampling probabilites
    SamplingP p1 = CalculateStrategyProbabilities(hinfo.materialID, o, hinfo.hitNormal, iors.x, iors.y, hinfo.localKd, hinfo.localPm);
    float3 s = SampleBRDF(p1, hinfo.materialID, o, hinfo.hitNormal, hinfo.hitGNormal, hinfo.localKd, hinfo.localPr, hinfo.localPm, pload.seed, iors.x, iors.y, vior.pointer);

    // Update the ior stack with the new sample
    if (dot(hinfo.hitGNormal, s) < 0.0f) { // Did we enter/exit a substrate?
        UpdateIORStack(vior, aior, hinfo.materialID, hinfo.objID);
    }

    // Store raygeom
    RayGeometry rgeom = InitRayGeometry(hinfo.hitPosition, s);
    StoreRayGeometry(g_pathStateBuffer, pixelIdx, rgeom);

    StoreVolumeIOR(g_pathStateBuffer, pixelIdx, vior);
    StoreVolumeAux(g_pathStateBuffer, pixelIdx, aior);

    // evaluate BSDF immediately
    //float pdf   = BRDF_PDF_COMBINED(p, hinfo.materialID, hinfo.hitNormal, hinfo.hitGNormal, s, o, hinfo.localKd, hinfo.localPr, hinfo.localPm, iors.x, iors.y);
    //float3 f_val = EvaluateBRDF_COMBINED(hinfo.materialID, hinfo.hitNormal, hinfo.hitGNormal, s, o, hinfo.localKd, hinfo.localPr, hinfo.localPm, iors.x, iors.y);
    SamplingP p2 = CalculateStrategyProbabilities(hinfo.materialID, o, hinfo.hitNormal, iors.x, iors.y, hinfo.localKd, hinfo.localPm);
    BrdfData bdata = EvaluateAndPdf_COMBINED(p2, hinfo.materialID, hinfo.hitNormal, hinfo.hitGNormal, s, o, hinfo.localKd, hinfo.localPr, hinfo.localPm, iors.x, iors.y);

    float  cosTheta = abs(dot(hinfo.hitNormal, s));

    // Validation & Throughput Update
    if(length(s) == 0.0f || bdata.pdf <= 1e-6f || any(isnan(bdata.val))) {
        // Kill path immediately if sample is invalid
        gScratchPing[uint3(tid.xy, 1)] = float4(0,0,0,0);
        return;
    }

    // Update Throughput for the ray we are about to shoot
    pload.throughput *= (bdata.val * cosTheta) / bdata.pdf;

    // Store the payload
    StorePathPayload(g_pathStateBuffer, pixelIdx, pload);

    PushStack(g_OutputStackIdx, pixelIdx);
}