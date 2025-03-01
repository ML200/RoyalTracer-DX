
// Function to trace a simple path using only BSDF sampling.
// Return the resulting emission as float3
float3 SamplePathSimple(const float3 initPoint, const float3 initNormal, const float3 initOutgoing, const Material initMaterial, inout uint2 seed){
    float3 accumulation = float3(0,0,0);
    for(int j = 0; j < 5; j++){
        float3 throughput = float3(1,1,1);

        float3 origin = initPoint;
        float3 normal = initNormal;
        float3 outgoing = normalize(initOutgoing);
        Material material = initMaterial;

        // Perform sampling as long as we dont hit a light source or reach max bounces
        for(int i = 0; i < 3; i++){
            // Select sampling strategy based on material
            float p_strategy;
            uint strategy = SelectSamplingStrategy(material, outgoing, normal, seed, p_strategy);

            // Calculate sample direction vector
            float3 sample;
            float3 adjustedOrigin;
            SampleBRDF(strategy, material, outgoing, normal, normal, sample, adjustedOrigin, origin, seed);
            float pdf = BRDF_PDF(strategy, material, normal, -sample, outgoing);
            if (isnan(pdf) || pdf < 1e-12f) {
                return float3(0,0,0);
            }

            float3 brdf = EvaluateBRDF(strategy, material, normal, -sample, outgoing);
            float ndot = dot(normal, sample);

            // Trace the ray
            RayDesc ray;
            ray.Origin = origin;
            ray.Direction = sample;
            ray.TMin = s_bias;
            ray.TMax = 10000;
            HitInfo samplePayload;
            TraceRay(SceneBVH, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, samplePayload);

            Material materialHit = materials[samplePayload.materialID];
            // Did we hit a light?
            if(length(materialHit.Ke) > 0.0f){
                // Calculate contribution
                if(i != 0){
                    //return throughput * brdf * ndot * materialHit.Ke / pdf;
                    accumulation += throughput * brdf * ndot * materialHit.Ke / pdf;
                }
                //else
                    //return float3(0,0,0);
            }
            else{
                // Adjust throughput
                throughput *= brdf * ndot / pdf;
                // Adjust sampling parameters
                outgoing = normalize(-sample);
                material = materialHit;
                normal = samplePayload.hitNormal;
                origin = samplePayload.hitPosition;
            }
        }
        // path hit no light
        //return float3(0,0,0);
    }
    return accumulation/5.0f;
}