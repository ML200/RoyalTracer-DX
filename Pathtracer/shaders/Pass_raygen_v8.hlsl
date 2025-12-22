using namespace dx;

#include "Includes_raygen_v8.hlsli"

#ifndef MAX_BOUNCES
#define MAX_BOUNCES 30
#endif


[shader("raygeneration")]
void Pass_raygen_v8()
{
    const uint2 pix = DispatchRaysIndex().xy;

    // Reset output texture
    gScratchPing[uint3(pix, 1)] = float4(0, 0, 0, 0);

    uint seed = initRandomData(pix, uint2(8, 4), time, 1u);

    float3 rayOrigin = InitOrigin();

    // Keep direction compressed across bounces: oct float2
    float3 dir0   = InitDirection(pix, float2(DispatchRaysDimensions().xy), seed);
    float2 rayDir2 = OctEncodeFloat2_payload(dir0);

    // Keep throughput compressed across bounces: RGB9E5
    uint packedThroughput = PackRGB9E5(float3(1.0f, 1.0f, 1.0f));

    // Keep previous normal compressed across bounces: octsnorm16x2
    uint prevNPacked = PackOctSnorm16_payload(float3(0.0f, 1.0f, 0.0f));

    // Path state (still unpacked for now; biggest future win is packing these)
    VolumeIOR vior = InitVolumeIOR();
    VolumeAux aior = InitVolumeAux();

    float prev_pdf = 1.0f;

    [loop]
    for (int depth = 0; depth < MAX_BOUNCES; ++depth)
    {
        // Decode direction only for the work that needs float3
        float3 rayDir = OctDecodeFloat2_payload(rayDir2);

        // Avoid sqrt; keep debug only
        float d2 = dot(rayDir, rayDir);
        bool badDir = any(isnan(rayDir)) || any(isinf(rayDir)) || (d2 < 1e-12f);
        bool badOrg = any(isnan(rayOrigin)) || any(isinf(rayOrigin));
        if (badDir || badOrg) break;

        RayDesc ray;
        ray.Origin    = rayOrigin;
        ray.Direction = rayDir;
        ray.TMin      = 0.00001f;
        ray.TMax      = 10000.0f;

        dx::HitObject hitObj = TraceRay_Custom(SceneBVH, ray, RAY_FLAG_NONE, 0xFF);

        if (!hitObj.IsHit())
        {
            // Unpack throughput only when needed
            float3 T = UnpackRGB9E5(packedThroughput);
            gScratchPing[uint3(pix, 1)] += float4(T * EvalMissState(), 0);
            break;
        }

        // --------------------------------------------------------------------
        // Pre-invoke setup (phantom check & payload init)
        // --------------------------------------------------------------------
        uint matID = GetMatIDFast(hitObj.GetInstanceIndex(), hitObj.GetPrimitiveIndex());

        // Get IORs (currently returns half2); pack immediately and keep packed
        half2 iors = GetIORs(vior, aior, matID, hitObj.GetInstanceIndex());
        uint  iorsPacked = PackHalf2_payload(float2(iors.x, iors.y));

        // Phantom surface: if ior.y == 0, advance and continue
        // (ior.y is stored in upper 16 bits in your PackHalf2)
        if ((iorsPacked >> 16) == 0u)
        {
            rayOrigin += hitObj.GetRayTCurrent() * rayDir;
            UpdateIORStack(vior, aior, matID, hitObj.GetInstanceIndex());
            continue;
        }

        uint currentMediumMatID = GetCurrentMediumMaterialID(vior, aior);

        // Metadata packed once
        uint meta0 = PackMeta0_payload((uint)depth, vior.pointer, matID, currentMediumMatID);
        uint meta1 = PackMeta1_payload(currentMediumMatID, (depth == 0) ? PF_FIRST_BOUNCE : 0);

        // --------------------------------------------------------------------
        // Invoke ClosestHit: keep payload strictly short-lived
        // --------------------------------------------------------------------
        float2 nextDir2;
        uint   nextNPacked;
        uint   nextPackedThroughput; // CHS currently returns updateWeight in this slot
        float  nextPdf;
        uint   nextSeed;
        uint   outMeta1;

        {
            PathRayPayload payload = InitPayload_Raygen_packed_payload(
                rayDir2,               // current ray dir in oct form
                prevNPacked,           // previous shading normal packed
                meta0,
                meta1,
                seed,
                iorsPacked,
                packedThroughput,      // path throughput up to this vertex (packed)
                prev_pdf
            );

            dx::HitObject::Invoke(hitObj, payload);

            nextDir2            = payload.dir2;
            nextNPacked         = payload.packedNs;
            nextPackedThroughput= payload.packedThroughput;
            nextPdf             = payload.bsdfPdf;
            nextSeed            = payload.seed;
            outMeta1            = payload.meta1;

            // Transmission flag update uses meta1 from CHS
            if ((GetFlags_payload(outMeta1) & PF_TRANSMIT_BACK) != 0)
            {
                UpdateIORStack(vior, aior, matID, hitObj.GetInstanceIndex());
            }
        } // payload dies here

        // --------------------------------------------------------------------
        // Validate and update state (keep unpacked locals short-lived)
        // --------------------------------------------------------------------
        // Decode next direction only for validation / tracing next bounce
        float3 s = OctDecodeFloat2_payload(nextDir2);

        // Weight is returned in packedThroughput slot (per your current CHS design)
        float3 weight = UnpackRGB9E5(nextPackedThroughput);

        // Validate
        if (dot(s, s) < 1e-12f || nextPdf <= 1e-6f || any(isnan(weight)) || any(isinf(weight)))
            break;

        // Apply throughput update in a tight scope
        {
            float3 T = UnpackRGB9E5(packedThroughput);
            T *= weight;
            packedThroughput = PackRGB9E5(T);
        }

        // Update RNG
        seed = nextSeed;

        // Russian Roulette (unpack only for luma)
        if (depth > 0)
        {
            float3 T = UnpackRGB9E5(packedThroughput);
            float survivalProb = min(1.0f, Luma(T));
            if (RandomFloatSingle(seed) >= survivalProb) break;
            T /= max(survivalProb, 0.1f);
            packedThroughput = PackRGB9E5(T);
        }

        // --------------------------------------------------------------------
        // Advance ray
        // --------------------------------------------------------------------
        rayOrigin += hitObj.GetRayTCurrent() * rayDir;

        // Carry compressed state to next bounce
        rayDir2         = nextDir2;
        prevNPacked     = nextNPacked;
        prev_pdf        = nextPdf;
    }
}
