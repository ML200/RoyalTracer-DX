// Traces a simple path using only BSDF sampling.
// Accumulates and returns emission via the provided reservoir.
void SamplePathSimple(
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
    float  acc_pdf_reconnection= 1.0f;

    // Accumulated radiance
    float3 acc_L = float3(0, 0, 0);

    // Primary path variables
    float3 origin  = initPoint;
    float3 normal  = initNormal;
    float3 outgoing= normalize(initOutgoing);

    MaterialOptimized material = initMaterial;

    // Reconnection point cache
    float3 xn, nn, Vn;
    uint   mID2, s, k;
    uint2  sample_seed = seed;

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
            // (If you want to terminate here, uncomment below)
            return;
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
    k    = 1;

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

            if (isReconnection)
            {
                Vn = normalize(incoming_NEE);
                s  = strategy;
            }

            // MIS weight
            float mi = pdf_light / (nee_samples * pdf_light + pdf_bsdf);

            acc_f_reconnection   *= throughput_NEE;
            acc_pdf_reconnection *= pdf_NEE;

            float3 E_reconnection = acc_f_reconnection * mi * emission_NEE;
            float3 E_path         = mi * contribution;

            float wi = LinearizeVector(E_path);
            acc_L += mi * contribution;

            // Reservoir update
            UpdateReservoir_GI(
                reservoir,
                wi,
                0.0f,
                xn,
                normalize(nn),
                Vn,
                E_reconnection,
                s,
                k,
                mID2,
                emission_NEE * acc_f,
                1.0f,
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

        if (length(contribution) > 10000.0f)
        {
            // Valid contribution (non‐zero)
            if (contribution.x != -1.0f)
            {
                if (isReconnection)
                {
                    Vn = normalize(incoming_BSDF);
                    s  = strategy;
                }

                float mi = pdf_bsdf / (nee_samples * pdf_light + pdf_bsdf);

                acc_f_reconnection   *= throughput_BSDF;
                acc_pdf_reconnection *= pdf_BSDF;

                float3 E_reconnection = acc_f_reconnection * mi * emission_BSDF;
                float3 E_path         = mi * contribution;

                float wi = LinearizeVector(E_path);
                acc_L += mi * contribution;

                // Reservoir update
                UpdateReservoir_GI(
                    reservoir,
                    wi,
                    0.0f,
                    xn,
                    normalize(nn),
                    Vn,
                    E_reconnection,
                    s,
                    k,
                    mID2,
                    emission_BSDF * acc_f,
                    1.0f,
                    seed
                );
            }
            // Optionally terminate path here if desired
            // break;
        }
        else
        {
            // Continue the path
            if (isReconnection)
            {
                Vn = normalize(incoming_BSDF);
                s  = strategy;
            }

            acc_f_reconnection   *= throughput_BSDF;
            acc_pdf_reconnection *= pdf_BSDF;

            origin  = new_origin;
            material= new_material;
            outgoing= new_outgoing;
            normal  = new_normal;
        }
    }

    //
    // 4) Final reservoir adjustment
    //
    float p_hat = LinearizeVector(reservoir.f);
    if (p_hat > 0.0f)
        reservoir.W = reservoir.w_sum / LinearizeVector(reservoir.f);
    else
        reservoir.W = 0.0f;

    if (!any(isnan(acc_L)) && !any(isinf(acc_L)))
        reservoir.f = acc_L;
    else
        reservoir.f = float3(1, 0, 0);
}