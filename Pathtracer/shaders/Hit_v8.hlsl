#include "Includes_raygen_v8.hlsli"

[shader("closesthit")]
void ClosestHit(inout PathRayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    // 1. Unpack Payload (Load state from RayGen)
    PathPayloadUncompressed data = UnpackPayload_payload(payload);

    // 3. Evaluate Surface
    HitInfo hinfo = EvalSurfaceState(InstanceID(), PrimitiveIndex(), attr.barycentrics, WorldRayOrigin(), data.depth);

    // 4. Handle Absorption (Volume)
    // We calculate it here and apply it to the BSDF return value later so RayGen applies it to throughput
    float3 absorptionTint = float3(1, 1, 1);
    if (data.mediumMatID != MEDIUM_INVALID_15)
    {
         absorptionTint = CalculateAbsorptionThroughput(materials[data.mediumMatID].Tf, RayTCurrent());
    }

    // 5. Emission (Direct Leed) & MIS
    {
        float3 emission = GetEmissionFast(InstanceID(), PrimitiveIndex());
        if (any(emission > 0.0f) && hinfo.lightID != 0xFFFFFFFFu)
        {
            if (data.depth == 0)
            {
                gScratchPing[uint3(DispatchRaysIndex().xy, 1)] += float4(emission,0); // TODO: UPDATE RESERVOIR HERE
            }
            else
            {
                // MIS: Balance Heuristic
                // prev_x is rayOrigin, prev_n is data.normal (from previous bounce), prev_pdf is data.bsdfPdf
                float lightPdfArea = LT_Pdf_LightTree_Area(WorldRayOrigin(), data.normal, hinfo.lightID, InstanceID());

                float cosLight   = max(dot(hinfo.hitNormal, -WorldRayDirection()), 0.0f);
                float lightPdfSA = (cosLight > 1e-6f) ? (lightPdfArea * RayTCurrent() * RayTCurrent() / cosLight) : 0.0f;

                float prev_pdf   = data.bsdfPdf;
                float misWeight  = prev_pdf / max(prev_pdf + lightPdfSA, 1e-20f);

                gScratchPing[uint3(DispatchRaysIndex().xy, 1)] += float4(data.throughput * emission * misWeight, 0); // TODO: UPDATE RESERVOIR HERE
            }
        }
    }

    // 6. Next Event Estimation (NEE)
    // Only perform if not in medium (simplified) and surface is not purely specular
    bool performNEE = !(data.mediumMatID != MEDIUM_INVALID_15 || materials[data.matID].Kd.w < EPSILON);

    if (performNEE)
    {
        float3 hitPos    = WorldRayOrigin() + WorldRayDirection() * RayTCurrent(); // Current intersection point
        LT_LightSampleResult light = LT_SamplePointOnLight(hitPos, hinfo.hitNormal, data.seed);

        float3 toLight = light.position - hitPos;
        float  distSq  = dot(toLight, toLight);
        float  dist    = sqrt(distSq);
        float3 L       = toLight / dist;

        float cosSurf  = dot(hinfo.hitNormal, L);
        float cosLight = dot(light.normal, -L);

        if (cosSurf > 1e-6f && cosLight > 1e-6f)
        {
            RayDesc shadowRay;
            shadowRay.Origin    = hitPos;
            shadowRay.Direction = L;
            shadowRay.TMin      = 0.001f;
            shadowRay.TMax      = dist - 0.001f;

            RayQuery<RAY_FLAG_CULL_NON_OPAQUE | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
            q.TraceRayInline(SceneBVH, RAY_FLAG_NONE, 0xFF, shadowRay);
            q.Proceed();

            if (q.CommittedStatus() == COMMITTED_NOTHING)
            {
                // Evaluate Surface for Light Direction
                SamplingP sp_nee = CalculateStrategyProbabilities(
                    data.matID, -WorldRayDirection(), hinfo.hitNormal,
                    data.iors.x, data.iors.y, hinfo.localKd, hinfo.localPm
                );

                BrdfData bdataNEE = EvaluateAndPdf_COMBINED(
                    sp_nee,
                    data.matID, hinfo.hitNormal, hinfo.hitGNormal, L, -WorldRayDirection(),
                    hinfo.localKd, hinfo.localPr, hinfo.localPm, data.iors.x, data.iors.y
                );

                float lightPdf = light.pdfSolidAngle;
                float bsdfPdf  = bdataNEE.pdf;

                if (lightPdf > 0.0f && bsdfPdf > 0.0f)
                {
                    float misWeight = lightPdf / (lightPdf + bsdfPdf);
                    // data.throughput contains path throughput up to this vertex
                    gScratchPing[uint3(DispatchRaysIndex().xy, 1)] += float4(data.throughput * cosSurf * light.emission * bdataNEE.val * (misWeight / lightPdf), 0); // TODO: UPDATE RESERVOIR HERE
                }
            }
        }
    }

    // Directional light
    if (performNEE)
    {
        float3 hitPos = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
        {
            // 1. Sample direction on the sun cone
            float2 rSun = float2(RandomFloatSingle(data.seed), RandomFloatSingle(data.seed));
            SunSampleResult sun = SampleSun(rSun);

            float NdotL = dot(hinfo.hitNormal, sun.direction);

            // Only trace if light is above horizon
            if (NdotL > 1e-6f)
            {
                RayDesc shadowRay;
                shadowRay.Origin    = hitPos;
                shadowRay.Direction = sun.direction;
                shadowRay.TMin      = 0.001f;
                shadowRay.TMax      = sun.dist; // Effectively infinite

                RayQuery<RAY_FLAG_CULL_NON_OPAQUE | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
                q.TraceRayInline(SceneBVH, RAY_FLAG_NONE, 0xFF, shadowRay);
                q.Proceed();

                if (q.CommittedStatus() == COMMITTED_NOTHING)
                {
                    // Evaluate BSDF for this Light Direction
                    SamplingP sp_nee = CalculateStrategyProbabilities(
                        data.matID, -WorldRayDirection(), hinfo.hitNormal,
                        data.iors.x, data.iors.y, hinfo.localKd, hinfo.localPm
                    );

                    BrdfData bdataNEE = EvaluateAndPdf_COMBINED(
                        sp_nee,
                        data.matID, hinfo.hitNormal, hinfo.hitGNormal, sun.direction, -WorldRayDirection(),
                        hinfo.localKd, hinfo.localPr, hinfo.localPm, data.iors.x, data.iors.y
                    );

                    float lightPdf = sun.pdf;
                    float bsdfPdf  = bdataNEE.pdf;

                    if (lightPdf > 0.0f && bsdfPdf > 0.0f)
                    {
                        float3 contrib = data.throughput * NdotL * sun.radiance * bdataNEE.val / lightPdf;

                        gScratchPing[uint3(DispatchRaysIndex().xy, 1)] += float4(contrib, 0); // TODO: UPDATE RESERVOIR HERE
                    }
                }
            }
        }
    }

    // 7. Sample Next Direction (BSDF)
    SamplingP sp = CalculateStrategyProbabilities(
        data.matID, -WorldRayDirection(), hinfo.hitNormal,
        data.iors.x, data.iors.y, hinfo.localKd, hinfo.localPm
    );

    float3 s = SampleBRDF(
        sp, data.matID, -WorldRayDirection(), hinfo.hitNormal, hinfo.hitGNormal,
        hinfo.localKd, hinfo.localPr, hinfo.localPm,
        data.seed, data.iors.x, data.iors.y, data.iorPointer
    );

    BrdfData bdata = EvaluateAndPdf_COMBINED(
        sp, data.matID, hinfo.hitNormal, hinfo.hitGNormal, s, -WorldRayDirection(),
        hinfo.localKd, hinfo.localPr, hinfo.localPm, data.iors.x, data.iors.y
    );

    // 8. Update Payload for Return

    // Direction & PDF
    data.dir     = s;
    data.bsdfPdf = bdata.pdf;

    // Normal: Update to current shading normal (for next bounce cosine term in RayGen)
    data.normal  = hinfo.hitNormal;

    // Calculate the cosine term here
    float cosTheta = abs(dot(hinfo.hitNormal, s));

    // Calculate the full throughput update weight
    // Weight = (BSDF * Absorption * Cos) / PDF
    // This value is typically <= 1.0, so it packs safely into RGB9E5
    float3 updateWeight = float3(0, 0, 0);
    if (bdata.pdf > 1e-6f)
    {
        updateWeight = (bdata.val * absorptionTint * cosTheta) / bdata.pdf;
    }

    // Store the WEIGHT in the throughput slot, not the raw BSDF value
    data.throughput = updateWeight;

    // Flags: Update Transmission bit
    if (dot(hinfo.hitGNormal, s) < 0.0f)
        data.flags |= PF_TRANSMIT_BACK;
    else
        data.flags &= ~PF_TRANSMIT_BACK;

    // 9. Pack and Return
    payload = PackPayload_payload(data);
}