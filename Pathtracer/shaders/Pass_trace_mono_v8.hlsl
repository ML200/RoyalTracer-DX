#include "Includes_v8.hlsli"

// Define max depth for the loop (adjust as needed for your scene)
#ifndef MAX_BOUNCES
#define MAX_BOUNCES 100
#endif

[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    // -------------------------------------------------------------------------
    // 1. INIT & SETUP (Replaces Shader 1)
    // -------------------------------------------------------------------------

    if (tid.x >= IMG_W || tid.y >= IMG_H) return;

    // Local Register State (Replaces g_pathStateBuffer)
    float3 accumulatedRadiance = float3(0, 0, 0);
    float3 throughput = float3(1, 1, 1);
    float3 rayOrigin = float3(0, 0, 0);
    float3 rayDir = float3(0, 0, 0);

    uint2  launchIndex = tid.xy;
    float2 dims = float2(IMG_W, IMG_H);
    uint   pixelIdx  = MapPixelID(dims, launchIndex);

    // Seeding
    uint seed = initRandomData(launchIndex, uint2(8,8), time, 1u);

    // -- Bounce 0: Camera Ray Generation --
    // Note: SampleCameraRay likely encapsulates the primary trace inside it based on the original code
    SampleData sdata = SampleCameraRay(pixelIdx, launchIndex, dims, seed);

    // Primary Visibility Check
    if(any(sdata.L1 > 0.0f)){
        // Directly hit a light/emissive surface from camera
        gScratchPing[uint3(tid.xy, 1)] = float4(sdata.L1, 0);
        return;
    }

    // Initialize Volume & Path State (Local Registers now)
    // Note: InitPathPayload usually sets throughput to 1, which we did manually above
    VolumeIOR vior = InitVolumeIOR();
    VolumeAux aior = InitVolumeAux();

    // -- Bounce 0: Material Sampling --
    float2 iors = GetIORs(vior, aior, sdata.matID, sdata.objID);

    SamplingP p1 = CalculateStrategyProbabilities(sdata.matID, sdata.o, sdata.n1_s, iors.x, iors.y, sdata.localKd, sdata.localPm);
    float3 s = SampleBRDF(p1, sdata.matID, sdata.o, sdata.n1_s, sdata.n1_g, sdata.localKd, sdata.localPr, sdata.localPm, seed, iors.x, iors.y, vior.pointer);

    // Update IOR stack
    if (dot(sdata.n1_g, s) < 0.0f) {
        UpdateIORStack(vior, aior, sdata.matID, sdata.objID);
    }

    // Set up ray for the first iteration of the loop
    rayOrigin = sdata.x1;
    rayDir    = s;

    // Evaluate BSDF for Bounce 0
    SamplingP p2 = CalculateStrategyProbabilities(sdata.matID, sdata.o, sdata.n1_s, iors.x, iors.y, sdata.localKd, sdata.localPm);
    float  pdf   = BRDF_PDF_COMBINED(p2, sdata.matID, sdata.n1_s, sdata.n1_g, s, sdata.o, sdata.localKd, sdata.localPr, sdata.localPm, iors.x, iors.y);
    float3 f_val = EvaluateBRDF_COMBINED(sdata.matID, sdata.n1_s, sdata.n1_g, s, sdata.o, sdata.localKd, sdata.localPr, sdata.localPm, iors.x, iors.y);
    float  cosTheta = abs(dot(sdata.n1_s, s));

    // Validation
    if(length(s) == 0.0f || pdf <= 1e-6f || any(isnan(f_val))) {
        gScratchPing[uint3(tid.xy, 1)] = float4(0,0,0,0);
        return;
    }

    // Update Throughput
    throughput *= (f_val * cosTheta) / pdf;

    // -------------------------------------------------------------------------
    // 2. PATH LOOP (Replaces Shader 2)
    // -------------------------------------------------------------------------

    // We start at 1 because Bounce 0 (Camera -> First Hit) was handled above
    [loop]
    for (int depth = 1; depth < MAX_BOUNCES; ++depth)
    {
        HitState hstate = (HitState)0.0f;

        // A. Trace Ray
        RayDesc ray;
        ray.Origin = rayOrigin;
        ray.Direction = rayDir;
        ray.TMin = 0.00001f;
        ray.TMax = 10000.0f;

        TraceRayInline_HitState(SceneBVH, ray, hstate, RAY_FLAG_NONE, 0xFF);

        // B. Handle Miss (Skybox)
        if(hstate.instanceID == 0xFFFFFFFF){
            accumulatedRadiance += throughput * EvalMissState();
            break; // End of path
        }

        // C. Handle Light Hit
        float3 emission = GetEmissionFast(hstate.instanceID, hstate.primitiveID);
        if(length(emission) > 0.0f){
            accumulatedRadiance += throughput * emission;
            break; // End of path (unless you support transparent lights)
        }

        // D. Surface Logic
        HitInfo hinfo = EvalSurfaceState(hstate.instanceID, hstate.primitiveID, hstate.bary, rayOrigin);

        // Russian Roulette
        float survivalProb = Luma(throughput);
        if (survivalProb > 1.0f) survivalProb = 1.0f;
        float rnd = RandomFloatSingle(seed); // Assuming RandomFloatSingle updates seed or hashes based on it

        if (rnd >= survivalProb) {
            break; // Path terminated
        }
        throughput /= max(survivalProb, 0.0001f);

        // Absorption (Beer's Law)
        uint currentMatID = GetCurrentMediumMaterialID(vior, aior);
        float3 tint = currentMatID != 0x0000FFFF ?
                      CalculateAbsorptionThroughput(materials[currentMatID].Tf, length(rayOrigin - hinfo.hitPosition)) :
                      (float3)1.0f;
        throughput *= tint;

        // Get IORs for new hit
        iors = GetIORs(vior, aior, hinfo.materialID, hinfo.objID);

        // E. Phantom Surface Handling (Transparent boundaries)
        if(iors.y == 0.0f){
            // Pass through without scattering
            rayOrigin = hinfo.hitPosition;
            // rayDir remains the same

            UpdateIORStack(vior, aior, hinfo.materialID, hinfo.objID);

            // Loop continues immediately to trace next segment
            depth--; // Do not count this as a bounce depth
            continue;
        }

        float3 o = normalize(rayOrigin - hinfo.hitPosition); // View Vector

        // F. Calculate Scattering
        SamplingP pLoop = CalculateStrategyProbabilities(hinfo.materialID, o, hinfo.hitNormal, iors.x, iors.y, hinfo.localKd, hinfo.localPm);
        float3 s_loop = SampleBRDF(pLoop, hinfo.materialID, o, hinfo.hitNormal, hinfo.hitGNormal, hinfo.localKd, hinfo.localPr, hinfo.localPm, seed, iors.x, iors.y, vior.pointer);

        // Update IOR Stack
        if (dot(hinfo.hitGNormal, s_loop) < 0.0f) {
            UpdateIORStack(vior, aior, hinfo.materialID, hinfo.objID);
        }

        // Evaluate BSDF
        SamplingP p2_loop = CalculateStrategyProbabilities(hinfo.materialID, o, hinfo.hitNormal, iors.x, iors.y, hinfo.localKd, hinfo.localPm);
        BrdfData bdata = EvaluateAndPdf_COMBINED(p2_loop, hinfo.materialID, hinfo.hitNormal, hinfo.hitGNormal, s_loop, o, hinfo.localKd, hinfo.localPr, hinfo.localPm, iors.x, iors.y);

        float cosThetaLoop = abs(dot(hinfo.hitNormal, s_loop));

        // Validation
        if(length(s_loop) == 0.0f || bdata.pdf <= 1e-6f || any(isnan(bdata.val))) {
            break; // Kill path
        }

        // Update Throughput
        throughput *= (bdata.val * cosThetaLoop) / bdata.pdf;

        // Prepare for next bounce
        rayOrigin = hinfo.hitPosition;
        rayDir = s_loop;
    }

    // -------------------------------------------------------------------------
    // 3. FINAL WRITE
    // -------------------------------------------------------------------------
    gScratchPing[uint3(launchIndex, 1)] = float4(accumulatedRadiance, 0);
}