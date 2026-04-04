#include "Includes_raygen_v8.hlsli"

[shader("closesthit")]
void ClosestHit(inout PathRayPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    // 1. Unpack Payload (Load state from RayGen)
    PathPayloadUncompressed data = UnpackPayload_payload(payload);
    // 3. Evaluate Surface
    HitInfo hinfo = EvalSurfaceState(InstanceIndex(), PrimitiveIndex(), attr.barycentrics, WorldRayOrigin(), data.depth);

    // Refetch material from UV (HitInfo no longer carries localKd/Pr/Pm)
    float3 hitLocalKd; float hitLocalPr, hitLocalPm;
    RefetchMaterial(data.matID, hinfo.uv, hitLocalKd, hitLocalPr, hitLocalPm);

    // 4. Handle Absorption (Volume)
    // We calculate it here and apply it to the BSDF return value later so RayGen applies it to throughput
    float3 absorptionTint = float3(1, 1, 1);
    if (data.mediumMatID != MEDIUM_INVALID_15)
    {
         absorptionTint = CalculateAbsorptionThroughput(materials[data.mediumMatID].Tf, RayTCurrent());
    }

    // 5. Emission (Direct Light) & MIS
    {
        float3 emission = GetEmissionFast(InstanceIndex(), PrimitiveIndex());

        // initialize the sample data structure here and store it.
        if(data.depth == 0){
            SampleData sdata = (SampleData)0;
            sdata.x1 = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
            sdata.n1_s = hinfo.hitNormal;
            sdata.n1_g = hinfo.hitGNormal;
            sdata.L1 = emission;
            sdata.o = -WorldRayDirection();
            sdata.objID = InstanceIndex();
            sdata.matID = data.matID;
            sdata.uv = hinfo.uv;
            sdata.etai = data.iors.x;
            sdata.etat = data.iors.y;
            uint idx = MapPixelID(float2(DispatchRaysDimensions().xy), DispatchRaysIndex().xy);
            storeSampleData(g_sample_current, idx, sdata);
        }

        if (any(emission > 0.0f) && hinfo.lightID != 0xFFFFFFFFu)
        {
            // MIS: Balance Heuristic
            // prev_x is rayOrigin, prev_n is data.normal (from previous bounce), prev_pdf is data.bsdfPdf
            float lightPdfArea = LT_Pdf_LightTree_Area(WorldRayOrigin(), data.normal, hinfo.lightID, InstanceIndex());

            float cosLight = max(dot(hinfo.hitNormal, -WorldRayDirection()), 0.0f);
            float dist2 = max(RayTCurrent() * RayTCurrent(), EPSILON);
            float lightPdfSA = (cosLight > EPSILON) ? (lightPdfArea * dist2 / cosLight) : 0.0f;

            float jacobian = (dist2 > EPSILON) ? (cosLight / dist2) : 0.0f;
            float prev_pdf = data.bsdfPdf;
            float misWeight = prev_pdf / max(prev_pdf + lightPdfSA, EPSILON);

            float pdfAM = jacobian * prev_pdf;

            // TODO: UPDATE RESERVOIR HERE
            if(data.depth == 1){
                uint idx = MapPixelID(float2(DispatchRaysDimensions().xy), DispatchRaysIndex().xy);
                float p_hat = GetPHat(data.throughput * emission * prev_pdf); // We need to remove the previous pdf; Cancel it my multiplying with it
                float wi = misWeight * p_hat / prev_pdf;//pdfAM;
                float3 x2 = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
                float3 n2 = hinfo.hitNormal;
                bool update = UpdateReservoirDI_Fast(g_Reservoirs_current_di, idx, wi, x2, n2, emission, InstanceIndex(), data.seed);
                if(update)store_phat_di(g_Reservoirs_current_di, idx, p_hat);
            }
            if (data.depth >= 2)
            {
                uint idx_gi = MapPixelID(float2(DispatchRaysDimensions().xy), DispatchRaysIndex().xy);

                float3 V2_new = (data.depth == 2) ? (-WorldRayDirection()) : load_Vpost_gi(g_Reservoirs_current_gi, idx_gi);
                float2 J_new  = float2(0.0f, 0.0f); // direct hit: no cached pdfx2, and posterior requires J=0

                // PSS: contribution already includes /pdf along the path, so wi = p_hat
                float3 contrib = data.throughput * emission;
                float  p_hat   = GetPHat(contrib);
                float  wi      = p_hat * misWeight;

                float3 tpostgi = load_Tpost_gi(g_Reservoirs_current_gi, idx_gi);

                bool update = UpdateReservoirGI_Fast(g_Reservoirs_current_gi, idx_gi,wi,emission * tpostgi, J_new, V2_new,data.seed);
                if (update) store_F_gi(g_Reservoirs_current_gi, idx_gi, p_hat);
            }
            data.bsdfPdf = 0.0f;
            payload = PackPayload_payload(data);
            return;
        }

        if (data.depth == 1)
        {
            uint idx_gi = MapPixelID(float2(DispatchRaysDimensions().xy), DispatchRaysIndex().xy);

            float3 x2 = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();

            SetReservoirGI_ConstHit(g_Reservoirs_current_gi, idx_gi,
                                    x2, hinfo.hitNormal, hinfo.hitGNormal,
                                    data.matID, InstanceIndex());

            SetReservoirGI_UVAndIOR(g_Reservoirs_current_gi, idx_gi,
                                    hinfo.uv, data.iors.x, data.iors.y);
        }

        if (data.depth == 2)
        {
            uint idx_gi = MapPixelID(float2(DispatchRaysDimensions().xy), DispatchRaysIndex().xy);
            store_Vpost_gi(g_Reservoirs_current_gi, idx_gi, -WorldRayDirection());
        }
    }

    // 6. Next Event Estimation (NEE)
    // Only perform if not in medium (simplified) and surface is not purely specular
    bool performNEE = !(data.mediumMatID != MEDIUM_INVALID_15 || materials[data.matID].Kd.w < EPSILON);

    if (performNEE)
    {
        float3 hitPos = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
        LT_LightSampleResult light = LT_SamplePointOnLight(hitPos, hinfo.hitNormal, data.seed);

        float3 toLight = light.position - hitPos;
        float  distSq  = dot(toLight, toLight);
        float  dist    = sqrt(distSq);
        float3 L       = toLight / dist;

        float cosSurf  = dot(hinfo.hitNormal, L);
        float cosLight = dot(light.normal, -L);

        // Check geometry and backfacing lights
        if (cosSurf > 1e-6f && cosLight > 1e-6f)
        {
            RayDesc shadowRay;
            shadowRay.Origin    = hitPos;
            shadowRay.Direction = L;
            shadowRay.TMin      = 0.00001f;
            shadowRay.TMax      = dist - 0.001f;

            RayQuery<RAY_FLAG_CULL_NON_OPAQUE | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
            q.TraceRayInline(SceneBVH, RAY_FLAG_NONE, 0xFF, shadowRay);
            q.Proceed();

            if (q.CommittedStatus() == COMMITTED_NOTHING)
            {
                // Evaluate BSDF
                SamplingP sp_nee = CalculateStrategyProbabilities(
                    data.matID, -WorldRayDirection(), hinfo.hitNormal,
                    data.iors.x, data.iors.y, hitLocalKd, hitLocalPm
                );

                BrdfData bdataNEE = EvaluateAndPdf_COMBINED(
                    sp_nee,
                    data.matID, hinfo.hitNormal, hinfo.hitGNormal, L, -WorldRayDirection(),
                    hitLocalKd, hitLocalPr, hitLocalPm, data.iors.x, data.iors.y
                );

                float lightPdf = light.pdfSolidAngle;
                float bsdfPdf  = bdataNEE.pdf;

                if (lightPdf > 0.0f && bsdfPdf > 0.0f)
                {
                    // MIS Weight (using Solid Angle measure for ratios)
                    float misWeight = lightPdf / (lightPdf + bsdfPdf);

                    // TODO UPDATE RESERVOIR (NEE Area Light)
                    if(data.depth == 0)
                    {
                        uint idx = MapPixelID(float2(DispatchRaysDimensions().xy), DispatchRaysIndex().xy);
                        float3 targetRadiance = data.throughput * light.emission * bdataNEE.val * cosSurf;
                        float p_hat = GetPHat(targetRadiance);
                        float wi = (lightPdf > 1e-20f) ? (misWeight * p_hat / lightPdf) : 0.0f;
                        bool update = UpdateReservoirDI_Fast(g_Reservoirs_current_di, idx, wi, light.position, light.normal, light.emission, light.objID, data.seed);
                        if(update) store_phat_di(g_Reservoirs_current_di, idx, p_hat);
                    }
                    // GI init: prior sample at x2 (depth==1) via NEE
                    if (data.depth >= 1)
                    {
                        uint idx_gi = MapPixelID(float2(DispatchRaysDimensions().xy), DispatchRaysIndex().xy);

                        float3 V2_new = (data.depth == 1) ? (-L) : load_Vpost_gi(g_Reservoirs_current_gi, idx_gi);

                        // Prior keeps pdf cache; posterior forces it off.
                        float2 J_new  = (data.depth == 1) ? float2(lightPdf, 0.0f)
                                                          : float2(0.0f, 0.0f);

                        // PSS: contribution includes /lightPdf; wi = p_hat directly
                        float3 contrib = data.throughput * light.emission * bdataNEE.val * cosSurf / lightPdf;
                        float  p_hat = GetPHat(contrib);
                        float  wi    = p_hat * misWeight;

                        float3 tpostgi = load_Tpost_gi(g_Reservoirs_current_gi, idx_gi);
                        if(data.depth > 1) tpostgi *= bdataNEE.val * cosSurf / lightPdf;

                        bool update = UpdateReservoirGI_Fast(g_Reservoirs_current_gi, idx_gi,wi,light.emission * tpostgi, J_new, V2_new,data.seed);
                        if (update) store_F_gi(g_Reservoirs_current_gi, idx_gi, p_hat);
                    }
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
                shadowRay.TMin      = 0.00001f;
                shadowRay.TMax      = 10000.0f;

                RayQuery<RAY_FLAG_CULL_NON_OPAQUE | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
                q.TraceRayInline(SceneBVH, RAY_FLAG_NONE, 0xFF, shadowRay);
                q.Proceed();

                if (q.CommittedStatus() == COMMITTED_NOTHING)
                {
                    // Evaluate BSDF for this Light Direction
                    SamplingP sp_nee = CalculateStrategyProbabilities(
                        data.matID, -WorldRayDirection(), hinfo.hitNormal,
                        data.iors.x, data.iors.y, hitLocalKd, hitLocalPm
                    );

                    BrdfData bdataNEE = EvaluateAndPdf_COMBINED(
                        sp_nee,
                        data.matID, hinfo.hitNormal, hinfo.hitGNormal, sun.direction, -WorldRayDirection(),
                        hitLocalKd, hitLocalPr, hitLocalPm, data.iors.x, data.iors.y
                    );

                    float lightPdf = sun.pdf;
                    float bsdfPdf  = bdataNEE.pdf;

                    if (lightPdf > 0.0f && bsdfPdf > 0.0f)
                    {
                        float3 contrib = data.throughput * NdotL * sun.radiance * bdataNEE.val / lightPdf;

                        // DI: sun at depth 0
                        if (data.depth == 0)
                        {
                            uint idx = MapPixelID(float2(DispatchRaysDimensions().xy), DispatchRaysIndex().xy);
                            float3 target = data.throughput * bdataNEE.val * NdotL * sun.radiance;
                            float  p_hat  = GetPHat(target);
                            float  wi = (lightPdf > 1e-20f) ? (p_hat / lightPdf) : 0.0f;
                            bool update = UpdateReservoirDI_Infinite(g_Reservoirs_current_di, idx, wi, normalize(sun.direction), sun.radiance, 0xFFFFFFFEu, data.seed);
                            if (update) store_phat_di(g_Reservoirs_current_di, idx, p_hat);
                        }
                        // GI init: prior sample at x2 (depth==1) via sun NEE
                        if (data.depth >= 1)
                        {
                            uint idx_gi = MapPixelID(float2(DispatchRaysDimensions().xy), DispatchRaysIndex().xy);

                            float3 V2_new = (data.depth == 1) ? (-sun.direction) : load_Vpost_gi(g_Reservoirs_current_gi, idx_gi);

                            float2 J_new  = (data.depth == 1) ? float2(lightPdf, 0.0f)
                                                              : float2(0.0f, 0.0f);

                            // If you want MIS against BSDF at the vertex, compute it here; otherwise keep as-is.
                            float3 contrib = data.throughput * bdataNEE.val * NdotL * sun.radiance / lightPdf;
                            float  p_hat   = GetPHat(contrib);
                            float  wi      = p_hat;

                            float3 tpostgi = load_Tpost_gi(g_Reservoirs_current_gi, idx_gi);
                            if(data.depth > 1) tpostgi *= bdataNEE.val * NdotL / lightPdf;

                            bool update = UpdateReservoirGI_Fast(g_Reservoirs_current_gi, idx_gi,wi,sun.radiance * tpostgi, J_new, V2_new, data.seed);
                            if (update) store_F_gi(g_Reservoirs_current_gi, idx_gi, p_hat);
                        }
                    }
                }
            }
        }
    }

    // 7. Sample Next Direction (BSDF)
    SamplingP sp = CalculateStrategyProbabilities(
        data.matID, -WorldRayDirection(), hinfo.hitNormal,
        data.iors.x, data.iors.y, hitLocalKd, hitLocalPm
    );

    float3 s = SampleBRDF(
        sp, data.matID, -WorldRayDirection(), hinfo.hitNormal, hinfo.hitGNormal,
        hitLocalKd, hitLocalPr, hitLocalPm,
        data.seed, data.iors.x, data.iors.y, data.iorPointer
    );

    BrdfData bdata = EvaluateAndPdf_COMBINED(
        sp, data.matID, hinfo.hitNormal, hinfo.hitGNormal, s, -WorldRayDirection(),
        hitLocalKd, hitLocalPr, hitLocalPm, data.iors.x, data.iors.y
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

    if(data.depth >= 1){
        uint idx_gi = MapPixelID(float2(DispatchRaysDimensions().xy), DispatchRaysIndex().xy);
        float3 tpostgi = load_Tpost_gi(g_Reservoirs_current_gi, idx_gi);
        tpostgi *= updateWeight;
        store_Tpost_gi(g_Reservoirs_current_gi, idx_gi,tpostgi);
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