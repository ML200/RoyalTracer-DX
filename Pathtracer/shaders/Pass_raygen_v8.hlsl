using namespace dx;

#include "Includes_raygen_v8.hlsli"

#ifndef MAX_BOUNCES
#define MAX_BOUNCES 30
#endif

[shader("raygeneration")]
void Pass_raygen_v8()
{
    uint  seed        = initRandomData(DispatchRaysIndex().xy, uint2(8, 8), time, 1u);

    float3 rayOrigin = InitOrigin();
    float3 rayDir    = InitDirection(DispatchRaysIndex().xy, float2(DispatchRaysDimensions().xy), seed);

    // path state
    float3 accumulatedRadiance = float3(0, 0, 0);
    float3 throughput          = float3(1, 1, 1);
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

        if (!hitObj.IsHit()) { accumulatedRadiance += throughput * EvalMissState(); break; }

        HitInfo hinfo = EvalSurfaceState(hitObj.GetInstanceIndex(), hitObj.GetPrimitiveIndex(), hitObj.GetAttributes<BuiltInTriangleIntersectionAttributes>().barycentrics, rayOrigin);

        {
            float3 emission = GetEmissionFast(hitObj.GetInstanceIndex(), hitObj.GetPrimitiveIndex());

            if (any(emission != 0) && hinfo.lightID != 0xFFFFFFFFu) {
                if(depth == 0){
                    accumulatedRadiance = emission;
                    break;
                }
                else {
                    float lightPdfArea = LT_Pdf_LightTree_Area(prev_x, prev_n, hinfo.lightID, hinfo.objID);
                    float cosLight = max(dot(hinfo.hitNormal, -rayDir), 0.0f);
                    float lightPdfSA = (cosLight > 1e-6f) ? (lightPdfArea * hinfo.hitT * hinfo.hitT / cosLight) : 0.0f;
                    float misWeight = prev_pdf / max(prev_pdf + lightPdfSA, 1e-20f);

                    accumulatedRadiance += throughput * emission * misWeight;
                }
            }
        }

        // Absorption
        uint currentMatID = GetCurrentMediumMaterialID(vior, aior);
        if (currentMatID != 0x0000FFFF)
        {
             float3 tint = CalculateAbsorptionThroughput(materials[currentMatID].Tf, hinfo.hitT);
             throughput *= tint;
        }

        // Check iors, maybe for phantom surface
        float2 iors = GetIORs(vior, aior, hinfo.materialID, hinfo.objID);
        if (iors.y == 0.0f) // Phantom surfaces
        {
            rayOrigin += hinfo.hitT * rayDir;
            UpdateIORStack(vior, aior, hinfo.materialID, hinfo.objID);
            continue;
        }

        // Only if we are not inside a medium, perform NEE. Also, the surface should have a diffuse component
        if(!(currentMatID != 0x0000FFFF || materials[hinfo.materialID].Kd.w < EPSILON)){
            LT_LightSampleResult light = LT_SamplePointOnLight(rayOrigin + rayDir * hinfo.hitT, hinfo.hitNormal, seed);

            float3 toLight = light.position - (rayOrigin + rayDir * hinfo.hitT);
            float  distSq  = dot(toLight, toLight);
            float  dist    = sqrt(distSq);
            float3 L       = toLight / dist;

            float cosSurf  = dot(hinfo.hitNormal, L);
            float cosLight = dot(light.normal, -L);

            if (cosSurf > 1e-6f && cosLight > 1e-6f)
            {
                RayDesc shadowRay;
                shadowRay.Origin    = rayOrigin + rayDir * hinfo.hitT;
                shadowRay.Direction = L;
                shadowRay.TMin      = 0.001f;
                shadowRay.TMax      = dist - 0.001f;

                RayQuery<RAY_FLAG_CULL_NON_OPAQUE | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
                q.TraceRayInline(SceneBVH, RAY_FLAG_NONE, 0xFF, shadowRay);
                q.Proceed();

                if (q.CommittedStatus() == COMMITTED_NOTHING)
                {
                    SamplingP sp_nee = CalculateStrategyProbabilities(
                        hinfo.materialID, -rayDir, hinfo.hitNormal,
                        iors.x, iors.y, hinfo.localKd, hinfo.localPm
                    );

                    BrdfData bdataNEE = EvaluateAndPdf_COMBINED(
                        sp_nee,
                        hinfo.materialID, hinfo.hitNormal, hinfo.hitGNormal, L, -rayDir,
                        hinfo.localKd, hinfo.localPr, hinfo.localPm, iors.x, iors.y
                    );

                    float cosSurf = dot(hinfo.hitNormal, L);

                    float lightPdf = light.pdfSolidAngle;
                    float bsdfPdf  = bdataNEE.pdf;

                    if (lightPdf > 0.0f && bsdfPdf > 0.0f)
                    {
                        float misWeight = lightPdf / (lightPdf + bsdfPdf);
                        accumulatedRadiance += throughput * cosSurf * light.emission * bdataNEE.val * (misWeight / lightPdf);
                    }
                }
            }
        }

        SamplingP sp = CalculateStrategyProbabilities(
            hinfo.materialID, -rayDir, hinfo.hitNormal,
            iors.x, iors.y, hinfo.localKd, hinfo.localPm
        );

        float3 s = SampleBRDF(
            sp, hinfo.materialID, -rayDir, hinfo.hitNormal, hinfo.hitGNormal,
            hinfo.localKd, hinfo.localPr, hinfo.localPm,
            seed, iors.x, iors.y, vior.pointer
        );

        // Update IOR stack on transmis
        if (dot(hinfo.hitGNormal, s) < 0.0f)
        {
            UpdateIORStack(vior, aior, hinfo.materialID, hinfo.objID);
        }

        // Evaluate and PDF
        BrdfData bdata = EvaluateAndPdf_COMBINED(
            sp, hinfo.materialID, hinfo.hitNormal, hinfo.hitGNormal, s, -rayDir,
            hinfo.localKd, hinfo.localPr, hinfo.localPm, iors.x, iors.y
        );

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
            throughput /= max(survivalProb, 0.0001f);
        }

        // Setup for next bounce
        rayOrigin += hinfo.hitT * rayDir;
        rayDir    = s;

        prev_pdf = bdata.pdf;
        prev_x  = rayOrigin;
        prev_n  = hinfo.hitNormal;
    }

    gScratchPing[uint3(DispatchRaysIndex().xy, 1)] = float4(accumulatedRadiance, 0);
}