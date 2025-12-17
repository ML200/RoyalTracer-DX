#include "Includes_v8.hlsli"

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    // Stack Management
    uint waveIdx = tid.x;
    uint activeCount = g_GlobalCounters.Load(g_InputStackIdx * 4);
    if (waveIdx >= activeCount) return;

    uint pixelIdx = PopStack(g_InputStackIdx, waveIdx).x;
    uint2 coord = UnmapPixelID(pixelIdx, gImageSize);

    // Load Path State data needed for trace
    RayGeometry rgeom = LoadRayGeometry(g_pathStateBuffer, pixelIdx);

    HitState hstate = (HitState)0.0f;

    {
        // Trace Ray
        RayDesc ray;
        ray.Origin = rgeom.origin;
        ray.Direction = rgeom.dir;
        ray.TMin = 0.00001f;
        ray.TMax = 10000.0f;
        TraceRayInline_HitState(SceneBVH, ray, hstate, RAY_FLAG_NONE, 0xFF);
    }

    {
        // Did we hit the sky?
        if(hstate.instanceID == 0xFFFFFFFF){
            // We hit the skybox
                // Load the throughput up to this point
            float3 t = LoadPathThroughput(g_pathStateBuffer, pixelIdx);
            gScratchPing[uint3(coord, 1)] = float4(t * EvalMissState(), 0);
            return;
        }
        // Did we hit a light?
        float3 emission = GetEmissionFast(hstate.instanceID, hstate.primitiveID);
        if(length(emission > 0.0f)){
            float3 t = LoadPathThroughput(g_pathStateBuffer, pixelIdx);
            gScratchPing[uint3(coord, 1)] = float4(t * emission, 0);
            return;
        }
    }

    // Material eval and new ray
    {
        HitInfo hinfo = EvalSurfaceState(hstate.instanceID, hstate.primitiveID, hstate.bary, rgeom.origin);

        // Load Path State data
        PathPayload pload = LoadPathPayload(g_pathStateBuffer, pixelIdx);

        // Russian roulette
        float survivalProb = Luma(pload.throughput);
        if (survivalProb > 1.0f) survivalProb = 1.0f;
        float rnd = RandomFloatSingle(pload.seed);
        if (rnd >= survivalProb){
            gScratchPing[uint3(coord, 1)] = float4(0,0,0,0);
            return;
        }
        pload.throughput /= max(survivalProb, 0.0001f);

        VolumeIOR vior = LoadVolumeIOR(g_pathStateBuffer, pixelIdx);
        VolumeAux aior = LoadVolumeAux(g_pathStateBuffer, pixelIdx);

        uint currentMatID = GetCurrentMediumMaterialID(vior, aior);
        float3 tint = currentMatID!=0x0000FFFF? CalculateAbsorptionThroughput(materials[currentMatID].Tf, length(rgeom.origin - hinfo.hitPosition)) : (float3)1.0f;
        pload.throughput *= tint;

        // Sample material and store direction
        float2 iors = GetIORs(vior, aior, hinfo.materialID, hinfo.objID);

        // Phantom surface:
        if(iors.y == 0.0f){
            RayGeometry next_rgeom = InitRayGeometry(hinfo.hitPosition, rgeom.dir);
            StoreRayGeometry(g_pathStateBuffer, pixelIdx, next_rgeom);
            StorePathPayload(g_pathStateBuffer, pixelIdx, pload);

            UpdateIORStack(vior, aior, hinfo.materialID, hinfo.objID);

            StoreVolumeIOR(g_pathStateBuffer, pixelIdx, vior);
            StoreVolumeAux(g_pathStateBuffer, pixelIdx, aior);

            PushStack(g_OutputStackIdx, uint2(pixelIdx, 0));
            return;
        }

        float3 o = normalize(rgeom.origin - hinfo.hitPosition);

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
        //UpdateOriginBounds(hinfo.hitPosition);

        StoreVolumeIOR(g_pathStateBuffer, pixelIdx, vior);
        StoreVolumeAux(g_pathStateBuffer, pixelIdx, aior);

        // evaluate BSDF immediately
        SamplingP p2 = CalculateStrategyProbabilities(hinfo.materialID, o, hinfo.hitNormal, iors.x, iors.y, hinfo.localKd, hinfo.localPm);
        BrdfData bdata = EvaluateAndPdf_COMBINED(p2, hinfo.materialID, hinfo.hitNormal, hinfo.hitGNormal, s, o, hinfo.localKd, hinfo.localPr, hinfo.localPm, iors.x, iors.y);

        float  cosTheta = abs(dot(hinfo.hitNormal, s));

        // Validation & Throughput Update
        if(length(s) == 0.0f || bdata.pdf <= 1e-6f || any(isnan(bdata.val))) {
            // Kill path immediately if sample is invalid
            gScratchPing[uint3(coord, 1)] = float4(0,0,0,0);
            return;
        }

        // Update Throughput for the ray we are about to shoot
        pload.throughput *= (bdata.val * cosTheta) / bdata.pdf;

        // Store the payload
        StorePathPayload(g_pathStateBuffer, pixelIdx, pload);
    }

    // Push the surviving data on the new stack
    gScratchPing[uint3(coord, 1)] = (float4)0.0f; // Make sure the current pixel is 0 if nothing was hit
    PushStack(g_OutputStackIdx, uint2(pixelIdx,0));
}