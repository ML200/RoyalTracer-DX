using namespace dx;

#include "Includes_raygen_v8.hlsli"

#ifndef MAX_BOUNCES
#define MAX_BOUNCES 30
#endif

[shader("raygeneration")]
void Pass_raygen_v8()
{
    // Reset output texture
    gScratchPing[uint3(DispatchRaysIndex().xy, 1)] = float4(0,0,0,0);
    uint  seed        = initRandomData(DispatchRaysIndex().xy, uint2(8, 4), time, 1u);

    float3 rayOrigin = InitOrigin();
    float3 rayDir    = InitDirection(DispatchRaysIndex().xy, float2(DispatchRaysDimensions().xy), seed);

    // path state
    half3 accumulatedRadiance = float3(0, 0, 0);
    half3 throughput          = float3(1, 1, 1);
    VolumeIOR vior = InitVolumeIOR();
    VolumeAux aior = InitVolumeAux();

    float  prev_pdf = 1.0f;
    float3 prev_x;
    float3 prev_n;


    [loop]
    for (int depth = 0; depth < MAX_BOUNCES; ++depth)
    {
        bool badDir = any(isnan(rayDir)) || any(isinf(rayDir)) || length(rayDir) < 1e-6f;
        bool badOrg = any(isnan(rayOrigin))    || any(isinf(rayOrigin));
        if (badDir || badOrg) break;
        RayDesc ray;
        ray.Origin    = rayOrigin;
        ray.Direction = rayDir;
        ray.TMin      = 0.00001f;
        ray.TMax      = 10000.0f;
        dx::HitObject hitObj = TraceRay_Custom(SceneBVH, ray, RAY_FLAG_NONE, 0xFF);

        if (!hitObj.IsHit()) { gScratchPing[uint3(DispatchRaysIndex().xy, 1)] += throughput * EvalMissState(); break; }

        // Get material id of our current
        uint matID = GetMatIDFast(hitObj.GetInstanceIndex(), hitObj.GetPrimitiveIndex());

        // Check iors, maybe for phantom surface
        half2 iors = GetIORs(vior, aior, matID, hitObj.GetInstanceIndex());
        if (iors.y == 0.0f) // Phantom surfaces
        {
            rayOrigin += hitObj.GetRayTCurrent() * rayDir;
            UpdateIORStack(vior, aior, matID, hitObj.GetInstanceIndex());
            continue;
        }
        uint currentMatID = GetCurrentMediumMaterialID(vior, aior);

        PathRayPayload payload = InitPayload_Raygen_payload(matID, currentMatID, rayDir, seed, iors, depth, vior.pointer);
        dx::HitObject::Invoke(hitObj, payload);
        // Partial loaders loading the rest of the required data later (normal, updateior condition etc so we save registers)

        // Terminate on invalid samples
        if (length(s) == 0.0f || bdata.pdf <= 1e-6f || any(isnan(bdata.val)))
        {
            break;
        }

        // Update throughput
        float cosThetaLoop = abs(dot(hinfo.hitNormal, s));
        throughput *= (bdata.val * cosThetaLoop) / bdata.pdf;

        // Aggressive russian roulette termination to leverage max SER
        if(depth > 0){
            float survivalProb = Luma(throughput);
            if (survivalProb > 1.0f) survivalProb = 1.0f;

            if (RandomFloatSingle(seed) >= survivalProb)
            {
                break;
            }
            throughput /= max(survivalProb, 0.1f);
        }

        // Update IOR stack on transmission
        if (dot(hinfo.hitGNormal, s) < 0.0f) // TODO replace the condition with flag return from the hit shader
        {
            UpdateIORStack(vior, aior, matID, hinfo.objID);
        }

        // Setup for next bounce
        rayOrigin += hinfo.hitT * rayDir;
        rayDir    = s;

        prev_pdf = bdata.pdf;
        prev_x  = rayOrigin;
        prev_n  = hinfo.hitNormal;
    }
}