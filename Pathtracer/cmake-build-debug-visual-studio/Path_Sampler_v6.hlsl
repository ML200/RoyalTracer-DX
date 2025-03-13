// Function to trace a simple path using only BSDF sampling.
// Return the resulting emission as float3
void SamplePathSimple(inout Reservoir_GI reservoir, const float3 initPoint, const float3 initNormal, const float3 initOutgoing, const MaterialOptimized initMaterial, inout uint2 seed){
    float3 acc_f = float3(1,1,1); // Throughput up the the current vertex WITHOUT the pdf (= f(x))
    float3 acc_f_reconnection = float3(1,1,1); // Throughput from the reconnection vertex
    float acc_pdf = 1.0f; // Accumulated pdf up to the current vertex
    float3 acc_L = float3(0,0,0);

    float3 origin = initPoint;
    float3 normal = initNormal;
    float3 outgoing = normalize(initOutgoing);
    MaterialOptimized material = initMaterial; // path forward variables


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
    /*
    After getting the first indirect hit position, we can perform unrestricted MIS weighted NEE/BSDF sampling.
    Each strategy produces an independant path that we use to update out pathracing reservoir (later).
    */
    [loop]
    for(int i = 0; i < bounces; i++){
        // Select the sampling strategy for this path vertex
        float p_strategy = 1.0f;
        uint strategy = SelectSamplingStrategy(material, outgoing, normal, seed, p_strategy);
        // NEE n times
        for(int j = 0; j < nee_samples; j++){
            float pdf_light = 1.0f;
            float pdf_bsdf = 1.0f;
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
                material,
                true
                );
            // MIS weight:
            float mi = pdf_light / (nee_samples * pdf_light + pdf_bsdf);
            acc_L += mi * contribution;
        }
        // BSDF
        float pdf_light = 1.0f;
        float pdf_bsdf = 1.0f;
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
            material
            );
        // MIS weight:
        if(length(contribution) > 0.0f){
            if(contribution.x != -1.0f){
                float mi = pdf_bsdf / (nee_samples * pdf_light + pdf_bsdf);
                acc_L += mi * contribution;
            }
            // terminate ray
            break;
        }
        else{
            origin = new_origin;
            material = new_material;
            outgoing = new_outgoing;
            normal = new_normal;
        }
    }
    if(!any(isnan(acc_L)))
        reservoir.E3 = acc_L;
}