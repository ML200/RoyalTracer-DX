// Function to trace a simple path using only BSDF sampling.
// Return the resulting emission as float3
void SamplePathSimple(inout Reservoir_GI reservoir, const float3 initPoint, const float3 initNormal, const float3 initOutgoing, const MaterialOptimized initMaterial, inout uint2 seed){
    float3 acc_f = float3(1,1,1); // Throughput up the the current vertex WITHOUT the pdf (= f(x))
    float3 acc_f_reconnection = float3(1,1,1); // Throughput from the reconnection vertex
    float acc_pdf = 1.0f; // Accumulated pdf up to the current vertex
    float acc_pdf_reconnection = 1.0f;
    float3 acc_L = float3(0,0,0);

    float3 origin = initPoint;
    float3 normal = initNormal;
    float3 outgoing = normalize(initOutgoing);
    MaterialOptimized material = initMaterial; // path forward variables

    // Reconnection point cache
    float3 xn;
    float3 nn;
    float3 Vn;
    uint mID2;
    uint2 sample_seed = seed;
    uint s;
    uint k;

    // Perform sampling
    /*
    From the start position, perform one bsdf sample to get first indirect sampling pos.
    If a light is hit, the sample is invalid and the reservoir therefore as well (keep it 0).
    */
    {
        // Select sampling strategy based on material
        float p_strategy;
        uint strategy = SelectSamplingStrategy(material, outgoing, normal, seed, p_strategy);

        // Calculate sample direction vector
        float3 sample;
        float3 adjustedOrigin;
        SampleBRDF(strategy, material, outgoing, normal, normal, sample, adjustedOrigin, origin, seed);

        RayDesc ray;
        ray.Origin = origin;
        ray.Direction = sample;
        ray.TMin = s_bias;
        ray.TMax = 10000;
        HitInfo samplePayload;
        TraceRay(SceneBVH, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, samplePayload);

        if(length(materials[samplePayload.materialID].Ke) > 0.0f){
            return;
        }
        else{
            float3 incoming = normalize(-sample);

            // Calculate the bsdf evaluation for both lobes, weigh them and adjust f
            float2 probs = CalculateStrategyProbabilities(material, outgoing, normal);

            float3 brdf0 = EvaluateBRDF(0, material, normal, incoming, outgoing);
            float3 brdf1 = EvaluateBRDF(1, material, normal, incoming, outgoing);

            float3 pdf0 = BRDF_PDF(0, material, normal, incoming, outgoing);
            float3 pdf1 = BRDF_PDF(1, material, normal, incoming, outgoing);

            float3 F1 = SafeMultiply(probs.x, brdf0);
            float3 F2 = SafeMultiply(probs.y, brdf1);
            float3 P1 = SafeMultiply(probs.x, pdf0);
            float3 P2 = SafeMultiply(probs.y, pdf1);
            float3 F = F1 + F2;
            float3 P = P1 + P2;

            // cosine term
            float NdotL = dot(normal, sample);
            //-----------------------------------Update accumulators-----------------------------------
            // Accumulate the pdf
            acc_pdf *= P;
            acc_f *= F * NdotL;

            //-----------------------------------Update path variables-----------------------------------
            outgoing = incoming;
            // Update material (optimized)
            uint mID = samplePayload.materialID;
            material.Kd = materials[mID].Kd;
            material.Pr_Pm_Ps_Pc = materials[mID].Pr_Pm_Ps_Pc;
            material.Ks = materials[mID].Ks;
            material.Ke = materials[mID].Ke;
            material.mID = mID;
            //____________________
            normal = samplePayload.hitNormal;
            origin = samplePayload.hitPosition;
        }
    }

    // Fill in the reconnection data (in the GI case, it is constantly x2)
    xn = origin;
    nn = normalize(normal);
    mID2 = material.mID;
    k = 1;

    /*
    After getting the first indirect hit position, we can perform unrestricted MIS weighted NEE/BSDF sampling.
    Each strategy produces an independant path that we use to update out pathracing reservoir (later).
    Importantly, this is also the reconnection vertex. So for the first iteration, set reconnection to true
    */
    [loop]
    for(int i = 0; i < bounces; i++){
        bool isReconnection = false;
        if(i == 0)
            isReconnection = true;
        // Select the sampling strategy for this path vertex
        float p_strategy = 1.0f;
        uint strategy = SelectSamplingStrategy(material, outgoing, normal, seed, p_strategy);
        // NEE n times
        for(int j = 0; j < nee_samples; j++){
            float pdf_light = 1.0f;
            float pdf_bsdf = 1.0f;
            float3 throughput_NEE = float3(1,1,1);
            float pdf_NEE = 1.0f;
            float3 emission_NEE = float3(0,0,0);
            float3 incoming_NEE;
            float3 contribution = SampleLightNEE_GI(
                pdf_light, // Outputs
                pdf_bsdf,
                incoming_NEE,
                seed, // Inputs
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
            if(isReconnection){
                Vn = normalize(incoming_NEE);
                s = strategy;
            }
            // MIS weight:
            float mi = pdf_light / (nee_samples * pdf_light + pdf_bsdf);
            acc_f_reconnection *= throughput_NEE;
            acc_pdf_reconnection *= pdf_NEE;

            float3 E_reconnection = acc_f_reconnection * mi * emission_NEE;
            float3 E_path = mi * contribution;

            float wi = LinearizeVector(E_path);

            acc_L += mi * contribution;

            // Add this path to the reservoir
            UpdateReservoir_GI(
                reservoir,
                wi,
                0.0f,
                xn,
                nn,
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
        // BSDF
        float pdf_light = 1.0f;
        float pdf_bsdf = 1.0f;
        float3 throughput_BSDF = float3(1,1,1);
        float pdf_BSDF = 1.0f;
        float3 emission_BSDF = float3(0,0,0);
        float3 incoming_BSDF;
        float3 new_origin;
        float3 new_normal;
        float3 new_outgoing;
        MaterialOptimized new_material;
        float3 contribution = SampleLightBSDF_GI(
            pdf_light, // Outputs (this time both in solid angle measure)
            pdf_bsdf,
            incoming_BSDF, // light direction
            new_origin,
            new_normal,
            new_outgoing,
            new_material,
            seed, // Inputs
            strategy,
            origin,
            normal,
            outgoing,
            acc_f, // Inout as this might be changed in case no light is hit
            acc_pdf, // Same here
            throughput_BSDF,
            pdf_BSDF,
            emission_BSDF,
            material,
            isReconnection
            );
        // MIS weight:
        if(length(contribution) > 0.0f){
            if(contribution.x != -1.0f){
                if(isReconnection){
                    Vn = normalize(incoming_BSDF);
                    s = strategy;
                }
                float mi = pdf_bsdf / (nee_samples * pdf_light + pdf_bsdf);
                acc_f_reconnection *= throughput_BSDF;
                acc_pdf_reconnection *= pdf_BSDF;

                float3 E_reconnection = acc_f_reconnection * mi * emission_BSDF;
                float3 E_path = mi * contribution;

                float wi = LinearizeVector(E_path);

                acc_L += mi * contribution;

                // Add this path to the reservoir
                UpdateReservoir_GI(
                    reservoir,
                    wi,
                    0.0f,
                    xn,
                    nn,
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
            // terminate path
            break;
        }
        else{
            if(isReconnection){
                Vn = normalize(incoming_BSDF);
                s = strategy;
            }
            // Continue the path
            acc_f_reconnection *= throughput_BSDF;
            acc_pdf_reconnection *= pdf_BSDF;

            origin = new_origin;
            material = new_material;
            outgoing = new_outgoing;
            normal = new_normal;
        }
    }
    float p_hat = LinearizeVector(reservoir.f);
    if(p_hat > 0.0f)
        reservoir.W = reservoir.w_sum / LinearizeVector(reservoir.f);
    else
        reservoir.W = 0.0f;
    /*if(!any(isnan(acc_L)))
        reservoir.f = acc_L;
    else
        reservoir.f = float3(0,0,0);*/
}