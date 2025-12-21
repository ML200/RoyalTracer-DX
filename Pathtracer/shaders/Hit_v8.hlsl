#include "Includes_raygen_v8.hlsli"

[shader("closesthit")] void ClosestHit(inout PathRayPayload payload, in BuiltInTriangleIntersectionAttributes attr) {
    uint   instanceIdx = InstanceID();
    uint   primIdx     = PrimitiveIndex();
    float3 rayOrigin   = WorldRayOrigin();
    float2 bary = attr.barycentrics;

    // Load data from raygen shader payload
    PathPayloadUncompressed data = UnpackPayload_payload(payload);

    HitInfo hinfo = EvalSurfaceState(InstanceID(), PrimitiveIndex(), attr.barycentrics, WorldRayOrigin(), data.depth);

    // Absorption
    if (data.mediumMatID != 0x0000FFFF)
    {
         float3 tint = CalculateAbsorptionThroughput(materials[data.mediumMatID].Tf, RayTCurrent());
         throughput *= tint;
    }

    // Only if we are not inside a medium, perform NEE. Also, the surface should have a diffuse component
    if(!(data.mediumMatID != 0x0000FFFF || materials[data.matID].Kd.w < EPSILON)){
        LT_LightSampleResult light = LT_SamplePointOnLight(rayOrigin + rayDir * RayTCurrent(), hinfo.hitNormal, seed);

        float3 toLight = light.position - (rayOrigin + rayDir * RayTCurrent());
        float  distSq  = dot(toLight, toLight);
        float  dist    = sqrt(distSq);
        float3 L       = toLight / dist;

        float cosSurf  = dot(hinfo.hitNormal, L);
        float cosLight = dot(light.normal, -L);

        if (cosSurf > 1e-6f && cosLight > 1e-6f)
        {
            RayDesc shadowRay;
            shadowRay.Origin    = rayOrigin + rayDir * RayTCurrent();
            shadowRay.Direction = L;
            shadowRay.TMin      = 0.001f;
            shadowRay.TMax      = dist - 0.001f;

            RayQuery<RAY_FLAG_CULL_NON_OPAQUE | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
            q.TraceRayInline(SceneBVH, RAY_FLAG_NONE, 0xFF, shadowRay);
            q.Proceed();

            if (q.CommittedStatus() == COMMITTED_NOTHING)
            {
                SamplingP sp_nee = CalculateStrategyProbabilities(
                    data.matID, -rayDir, hinfo.hitNormal,
                    iors.x, iors.y, hinfo.localKd, hinfo.localPm
                );

                BrdfData bdataNEE = EvaluateAndPdf_COMBINED(
                    sp_nee,
                    data.matID, hinfo.hitNormal, hinfo.hitGNormal, L, -rayDir,
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
                float lightPdfSA = (cosLight > 1e-6f) ? (lightPdfArea * RayTCurrent() * RayTCurrent() / cosLight) : 0.0f;
                float misWeight = prev_pdf / max(prev_pdf + lightPdfSA, 1e-20f);
                accumulatedRadiance += throughput * emission * misWeight;
            }
        }
    }


    SamplingP sp = CalculateStrategyProbabilities(
        data.matID, -rayDir, hinfo.hitNormal,
        iors.x, iors.y, hinfo.localKd, hinfo.localPm
    );

    float3 s = SampleBRDF(
        sp, data.matID, -rayDir, hinfo.hitNormal, hinfo.hitGNormal,
        hinfo.localKd, hinfo.localPr, hinfo.localPm,
        seed, iors.x, iors.y, vior.pointer
    );

    // Evaluate and PDF
    BrdfData bdata = EvaluateAndPdf_COMBINED(
        sp, data.matID, hinfo.hitNormal, hinfo.hitGNormal, s, -rayDir,
        hinfo.localKd, hinfo.localPr, hinfo.localPm, iors.x, iors.y
    );

    if (dot(hinfo.hitGNormal, s) < 0.0f)
        // TODO set below surface flag bit

    // Update the payload and compress it for returning
    payload = PackPayload_payload(data);
}