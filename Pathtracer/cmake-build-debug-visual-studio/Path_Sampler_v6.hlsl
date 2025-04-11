// Traces a simple path using only BSDF sampling.
// Accumulates and returns emission via the provided reservoir.
float3 SamplePathSimple(
    inout Reservoir_GI reservoir,
    const float3       initPoint,
    const float3       initNormal,
    const float3       initOutgoing,
    const MaterialOptimized initMaterial,
    inout uint2        seed
)
{
    // Throughputs and PDFs
    float3 acc_f               = float3(1, 1, 1);  // Throughput up to the current vertex (f(x))
    float3 acc_f_reconnection  = float3(1, 1, 1);  // Throughput from the reconnection vertex
    float  acc_pdf             = 1.0f;             // Accumulated PDF up to the current vertex

    // Accumulated radiance
    float3 acc_L = float3(0, 0, 0);

    // Primary path variables
    float3 origin  = initPoint;
    float3 normal  = initNormal;
    float3 outgoing= normalize(initOutgoing);

    MaterialOptimized material = initMaterial;

    // Reconnection point cache
    float3 xn, nn, Vn;
    uint   mID2;

    //
    // 1) Perform an initial BSDF sampling to get the first indirect intersection
    //
    {
        float p_strategy;
        uint strategy = SelectSamplingStrategy(material, outgoing, normal, seed, p_strategy);

        // Calculate sample direction and trace the ray
        float3 sample, adjustedOrigin;
        SampleBRDF(strategy, material, outgoing, normal, normal, sample, adjustedOrigin, origin, seed);

        RayDesc ray;
        ray.Origin    = origin;
        ray.Direction = sample;
        ray.TMin      = s_bias;
        ray.TMax      = 10000;

        HitInfo samplePayload;
        TraceRay(SceneBVH, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, samplePayload);

        // Check if a light is hit
        if (length(materials[samplePayload.materialID].Ke) > 0.0f)
        {
            // If a direct light is immediately hit, do nothing further;
            // the reservoir is effectively invalid for this sample.
            return 0.0f;
        }
        else
        {
            float3 incoming = normalize(-sample);

            // Evaluate both BRDF lobes, combine, and apply the cosine factor
            float2 probs  = CalculateStrategyProbabilities(material, outgoing, normal);
            float3 brdf0  = EvaluateBRDF(0, material, normal, incoming, outgoing);
            float3 brdf1  = EvaluateBRDF(1, material, normal, incoming, outgoing);
            float pdf0   = BRDF_PDF(0, material, normal, incoming, outgoing);
            float pdf1   = BRDF_PDF(1, material, normal, incoming, outgoing);

            float3 F1 = SafeMultiply(probs.x, brdf0);
            float3 F2 = SafeMultiply(probs.y, brdf1);
            float P1 = SafeMultiply(probs.x, pdf0);
            float P2 = SafeMultiply(probs.y, pdf1);

            float3 F = F1 + F2;
            float P = P1 + P2;

            float NdotL = dot(normal, sample);

            // Update PDF/product terms
            acc_pdf *= P;
            acc_f   *= (F * NdotL);

            // Advance path
            outgoing = incoming;

            uint mID = samplePayload.materialID;
            material.Kd             = materials[mID].Kd;
            material.Pr_Pm_Ps_Pc    = materials[mID].Pr_Pm_Ps_Pc;
            material.Ks             = materials[mID].Ks;
            material.Ke             = materials[mID].Ke;
            material.mID            = mID;

            normal = samplePayload.hitNormal;
            origin = samplePayload.hitPosition;
        }
    }

    //
    // 2) Set up reconnection data (first valid indirect vertex)
    //
    xn   = origin;
    nn   = normalize(normal);
    mID2 = material.mID;

    //
    // 3) Perform multiple bounces with MIS‐weighted NEE and BSDF sampling
    //
    [loop]
    for (int i = 0; i < bounces; i++)
    {
        bool isReconnection = (i == 0);

        // Select a sampling strategy
        float p_strategy = 1.0f;
        uint  strategy   = SelectSamplingStrategy(material, outgoing, normal, seed, p_strategy);

        //
        // 3a) NEE sampling
        //
        for (int j = 0; j < nee_samples; j++)
        {
            float   pdf_light      = 1.0f;
            float   pdf_bsdf       = 1.0f;
            float3  throughput_NEE = float3(1, 1, 1);
            float   pdf_NEE        = 1.0f;
            float3  emission_NEE   = float3(0, 0, 0);
            float3  incoming_NEE;

            float3 contribution = SampleLightNEE_GI(
                pdf_light,       // out
                pdf_bsdf,
                incoming_NEE,
                seed,            // in
                strategy,
                origin,
                normal,
                outgoing,
                acc_f,
                acc_pdf,
                throughput_NEE,
                pdf_NEE,
                emission_NEE,
                material,
                true,
                isReconnection
            );
            float3 local_throughput = float3(1,1,1);

            if (isReconnection)
            {
                Vn = normalize(incoming_NEE);
            }
            //else{
                local_throughput = throughput_NEE;
            //}

            // MIS weight
            float mi = pdf_light / (nee_samples * pdf_light + pdf_bsdf);

            float3 E_reconnection = acc_f_reconnection * mi * emission_NEE * local_throughput;
            float3 E_path         = mi * contribution;

            float wi = LinearizeVector(E_path);
            acc_L += mi * contribution;

            if(isnan(wi) || isinf(wi))
                wi = 0.0f;

            // Reservoir update
            UpdateReservoir_GI(
                reservoir,
                wi,
                0.0f,
                xn,
                normalize(nn),
                Vn,
                E_reconnection,
                mID2,
                seed
            );
        }

        //
        // 3b) BSDF sampling
        //
        float   pdf_light       = 1.0f;
        float   pdf_bsdf        = 1.0f;
        float3  throughput_BSDF = float3(1, 1, 1);
        float   pdf_BSDF        = 1.0f;
        float3  emission_BSDF   = float3(0, 0, 0);
        float3  incoming_BSDF;
        float3  new_origin;
        float3  new_normal;
        float3  new_outgoing;
        MaterialOptimized new_material;

        strategy   = SelectSamplingStrategy(material, outgoing, normal, seed, p_strategy);

        float3 contribution = SampleLightBSDF_GI(
            pdf_light,       // out
            pdf_bsdf,
            incoming_BSDF,
            new_origin,
            new_normal,
            new_outgoing,
            new_material,
            seed,            // in
            strategy,
            origin,
            normal,
            outgoing,
            acc_f,
            acc_pdf,
            throughput_BSDF,
            pdf_BSDF,
            emission_BSDF,
            material,
            isReconnection
        );

        if (isReconnection)
        {
            Vn = normalize(incoming_BSDF);
        }
        //else{
            acc_f_reconnection *= throughput_BSDF;
        //}

        if (length(contribution) > 0.0f)
        {
            float mi = pdf_bsdf / (nee_samples * pdf_light + pdf_bsdf);

            float3 E_reconnection = acc_f_reconnection * mi * emission_BSDF;
            float3 E_path         = mi * contribution;

            float wi = LinearizeVector(E_path);

            // DEBUG
            acc_L += E_path;

            if(isnan(wi) || isinf(wi))
                wi = 0.0f;

            // Reservoir update
            UpdateReservoir_GI(
                reservoir,
                wi,
                0.0f,
                xn,
                normalize(nn),
                Vn,
                E_reconnection,
                mID2,
                seed
            );
            break;
        }
        else
        {
            origin  = new_origin;
            material= new_material;
            outgoing= new_outgoing;
            normal  = new_normal;
        }
    }
    return acc_L;
}