float LinearizeVector(float3 v){
    //return (v.x + v.y + v.z)/3.0f;
    return length(v);
}

// The remaining functions remain unchanged.
float VisibilityCheck(
    float3 x1,
    float3 n1,
    float3 dir,
    float dist
)
{
    float V = 0.0f;
    RayDesc ray;
    ray.Origin = x1 + normalize(n1) * s_bias;
    ray.Direction = dir;
    ray.TMin = 0.0f;
    ray.TMax = max(dist - 10.0f * s_bias, 2.0f * s_bias);
    ShadowHitInfo shadowPayload;
    shadowPayload.isHit = false;
    TraceRay(SceneBVH, RAY_FLAG_NONE, 0xFF, 1, 0, 1, ray, shadowPayload);
    V = shadowPayload.isHit ? 0.0f : 1.0f;
    return V;
}

float3 ReconnectDI(
    float3 x1,
    float3 n1,
    float3 x2,
    float3 n2,
    float3 L,
    float3 outgoing,
    uint strategy,
    Material material
)
{
    float3 dir = x2 - x1;
    float dist = length(dir);
    if(length(L) == 0.0f || dist < MIN_DIST)
        return float3(0,0,0);

    float cosThetaX1 = max(0, dot(n1, normalize(dir)));
    if(dot(n2, normalize(-dir)) < 0.0f)
        n2 = -n2;
    float cosThetaX2 = max(0, dot(n2, normalize(-dir)));
    float2 probs = CalculateStrategyProbabilities(material, normalize(outgoing), n1);
    float3 F1 = probs.x * EvaluateBRDF(0, material, n1, normalize(-dir), normalize(outgoing));
    float3 F2 = probs.y * EvaluateBRDF(1, material, n1, normalize(-dir), normalize(outgoing));
    float3 F = F1 + F2;

    return F * L * cosThetaX1 * cosThetaX2 / (dist * dist);
}

float GetP_Hat(Reservoir_DI s, Reservoir_DI t, SampleData sample_s, bool use_visibility){
    float f_g = LinearizeVector(ReconnectDI(sample_s.x1, sample_s.n1, t.x2, t.n2, t.L2, sample_s.o, s.s, materials[sample_s.mID]));
    float v = 1.0f;

    if(use_visibility){
        v = VisibilityCheck(sample_s.x1, sample_s.n1, normalize(t.x2-sample_s.x1), length(t.x2-sample_s.x1));
    }
    return f_g * v;
}

float GetW(Reservoir_DI r, float p_hat){
    if(p_hat > EPSILON)
        return r.w_sum / p_hat;
    else
        return 0.0f;
}


// Use visibility is mandatory
void SampleLightBSDF(
    inout float pdf_light, // Outputs
    inout float pdf_bsdf,
    inout float3 incoming,
    inout float p_hat,

    inout uint2 seed, // Inputs
    float3 worldOrigin,
    float3 normal,
    float3 outgoing,
    Material material,
    float fresnel,
    uint strategy,
    inout float3 emission,
    inout float3 x2,
    inout float3 n2
    ){

    // Sample a BSDF direction
    float3 sample;
    float3 origin = worldOrigin;
    SampleBRDF(strategy, material, outgoing, normal, normal, sample, origin, worldOrigin, seed);

    // Trace the ray
    RayDesc ray;
    ray.Origin = worldOrigin;
    ray.Direction = sample;
    ray.TMin = s_bias;
    ray.TMax = 10000;
    HitInfo samplePayload;
    TraceRay(SceneBVH, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, samplePayload);

    // Evaluate the contribution
    Material material_ke = materials[samplePayload.materialID];
    float Ke = material_ke.Ke.x + material_ke.Ke.y + material_ke.Ke.z;
    emission = material_ke.Ke;
    x2 = samplePayload.hitPosition;
    n2 = samplePayload.hitNormal;

    if(Ke > EPSILON){
        float3 L = samplePayload.hitPosition - worldOrigin;
        float dist = length(L);
        float dist2 = dist * dist;
        float cos_theta = dot(samplePayload.hitNormal, -sample);

        pdf_light = ((Ke / 3.0f) / g_EmissiveTriangles[0].total_weight);
        pdf_bsdf = max(EPSILON, BRDF_PDF(strategy, material, normal, -sample, outgoing) * cos_theta / dist2);
        incoming = -sample;

        float2 probs = CalculateStrategyProbabilities(material, normalize(outgoing), normal);
        float3 brdf1 = probs.x * EvaluateBRDF(0, material, normal, -sample, outgoing);
        float3 brdf2 = probs.y * EvaluateBRDF(1, material, normal, -sample, outgoing);
        float3 brdf = brdf1 + brdf2;
        float ndot = dot(normal, sample);

        // Compute p_hat defensively.
        p_hat = LinearizeVector(brdf * material_ke.Ke * ndot * cos_theta / dist2);
    }
    else{
        p_hat = 0.0f;
    }
}


void SampleLightNEE(
    inout float pdf_light, // Outputs
    inout float pdf_bsdf,
    inout float3 incoming,
    inout float p_hat,

    inout uint2 seed, // Inputs
    float3 worldOrigin,
    float3 normal,
    float3 outgoing,
    Material material,
    uint strategy,
    float fresnel,
    inout float3 emission,
    inout float3 x2,
    inout float3 n2,
    bool useVisibility
    ){

    // Sample a Light Triangle
    float randomValue = RandomFloat(seed);
    int left = 0;
    int right = g_EmissiveTriangles[0].triCount - 1;
    int selectedIndex = 0;

    while (left <= right) {
        int mid = left + (right - left) / 2;
        float midCdf = g_EmissiveTriangles[mid].cdf;
        if (randomValue < midCdf) {
            selectedIndex = mid;
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    LightTriangle sampleLight = g_EmissiveTriangles[selectedIndex];

    // Calculate the current world coordinates of the triangle
    float4x4 conversionMatrix = instanceProps[sampleLight.instanceID].objectToWorld;
    float3 x_v = mul(conversionMatrix, float4(sampleLight.x, 1.f)).xyz;
    float3 y_v = mul(conversionMatrix, float4(sampleLight.y, 1.f)).xyz;
    float3 z_v = mul(conversionMatrix, float4(sampleLight.z, 1.f)).xyz;

    // Generate barycentric coordinates
    float xi1 = RandomFloat(seed);
    float xi2 = RandomFloat(seed);
    if (xi1 + xi2 > 1.0f) {
        xi1 = 1.0f - xi1;
        xi2 = 1.0f - xi2;
    }
    float u = 1.0f - xi1 - xi2;
    float v = xi1;
    float w = xi2;
    float3 samplePoint = u * x_v + v * y_v + w * z_v;
    x2 = samplePoint;

    // Get the sample direction and compute distance
    float3 L = samplePoint - worldOrigin;
    float dist2 = dot(L, L);
    float dist = sqrt(max(dist2, EPSILON));
    float3 L_norm = normalize(L);

    // Compute the light's surface normal from triangle geometry
    float3 edge1 = y_v - x_v;
    float3 edge2 = z_v - x_v;
    float3 cross_l = cross(edge1, edge2);
    float3 normal_l = normalize(cross_l);
    if(dot(normal_l, -L_norm) < 0.0f){
        normal_l = -normal_l;
    }
    n2 = normal_l;

    float area_l = abs(length(cross_l) * 0.5f);
    float pdf_l = sampleLight.weight / max(area_l, EPSILON);
    float pdf_brdf_light = max(BRDF_PDF(strategy, material, normal, -L_norm, normalize(outgoing)), EPSILON);

    // Compute cosine factors
    float cos_theta_x = dot(normal, L_norm);

    float cos_theta_y = dot(normal_l, -L_norm);


    // Compute the geometry term
    float G = max(cos_theta_y * cos_theta_x / dist2, EPSILON);
    float3 emission_l = sampleLight.emission;
    emission = emission_l;

    // Sample the BSDF for the light's direction
    float2 probs = CalculateStrategyProbabilities(material, normalize(outgoing), normal);
    float3 brdf_light1 = probs.x * EvaluateBRDF(0, material, normal, -L_norm, normalize(outgoing));
    float3 brdf_light2 = probs.y * EvaluateBRDF(1, material, normal, -L_norm, normalize(outgoing));
    float3 brdf_light = brdf_light1 + brdf_light2;
    // Optional visibility check
    float V = 1.0f;
    if(useVisibility == true){
        RayDesc ray;
        ray.Origin = worldOrigin + s_bias * normal;
        ray.Direction = L_norm;
        ray.TMin = 0.0f;
        ray.TMax = dist - s_bias * 2.0f;
        ShadowHitInfo shadowPayload;
        shadowPayload.isHit = false;
        TraceRay(SceneBVH, RAY_FLAG_NONE, 0xFF, 1, 0, 1, ray, shadowPayload);
        V = shadowPayload.isHit ? 0.0f : 1.0f;
    }

    // Evaluate the contribution defensively.
    p_hat = LinearizeVector(emission_l * brdf_light * G * V);

    pdf_light = max(EPSILON, pdf_l);
    pdf_bsdf = pdf_brdf_light * cos_theta_y / dist2;
    incoming = -L_norm;
}


// Sample RIS for direct Lights (for now) on a given point
// M1: Number NEE samples
// M2: Number BSDF samples
void SampleRIS(
    uint M1,
    uint M2,
    float3 outgoing,
    inout Reservoir_DI reservoir,
    HitInfo payload,
    inout uint2 seed
    ){

    Material material = materials[payload.materialID];
    float p_strategy = 1.0f;
    uint strategy = SelectSamplingStrategy(material, outgoing, payload.hitNormal, seed, p_strategy);

    // Iterate through M1 NEE samples and fill up the reservoir
    [unroll]
    for(int i = 0; i < M1; i++){
        float pdf_light = 0.0f;
        float pdf_bsdf = 0.0f;
        float3 incoming;
        float p_hat;
        float3 emission;
        float3 x2;
        float3 n2;

        SampleLightNEE(
            pdf_light,
            pdf_bsdf,
            incoming,
            p_hat,
            seed,
            payload.hitPosition,
            payload.hitNormal,
            outgoing,
            material,
            strategy,
            p_strategy,
            emission,
            x2,
            n2,
            false // Test with visibility for now
            );

        // Calculate MIS weight and update reservoir only if contribution is valid
        float mi = pdf_light / (M1 * pdf_light + M2 * pdf_bsdf);
        float wi = mi * p_hat / pdf_light;
        if(p_hat > 0.0f)
            UpdateReservoir(reservoir, wi, 0.0f, x2, n2, emission, strategy, seed);
    }

    // Iterate through M2 BSDF samples and fill the reservoir
    [unroll]
    for(int j = 0; j < M2; j++){
        float pdf_light = 0.0f;
        float pdf_bsdf = 0.0f;
        float3 incoming;
        float p_hat;
        float3 emission;
        float3 x2;
        float3 n2;

        SampleLightBSDF(
            pdf_light,
            pdf_bsdf,
            incoming,
            p_hat,
            seed,
            payload.hitPosition,
            payload.hitNormal,
            outgoing,
            material,
            p_strategy,
            strategy,
            emission,
            x2,
            n2
            );
        float mi = pdf_bsdf / (M1 * pdf_light + M2 * pdf_bsdf);
        float wi = mi * p_hat / pdf_bsdf;
        if(p_hat > 0.0f)
            UpdateReservoir(reservoir, wi, 0.0f, x2, n2, emission, strategy, seed);
    }
    // Set canonical weight
    reservoir.M = 1.0f;
}

int2 GetBestReprojectedPixel_d(
    float3 worldPos,
    float4x4 prevView,
    float4x4 prevProjection,
    float2 resolution)
{
    float2 subPixelCoord = GetLastFramePixelCoordinates_Float(worldPos, prevView, prevProjection, resolution);
    int2 basePixel = int2(floor(subPixelCoord));
    int2 candidates[4];
    candidates[0] = basePixel;
    candidates[1] = basePixel + int2(1, 0);
    candidates[2] = basePixel + int2(0, 1);
    candidates[3] = basePixel + int2(1, 1);
    float minDistance = 1e7f;
    int2 bestPixel = int2(-1, -1);
    for (int i = 0; i < 4; i++)
    {
        int2 candidate = candidates[i];
        uint index = candidate.y * uint(resolution.x) + candidate.x;
        float3 candidateWorldPos = g_sample_last[index].x1;
        float dist = length(candidateWorldPos - worldPos);
        if (dist < minDistance)
        {
            minDistance = dist;
            bestPixel = candidate;
        }
    }
    return bestPixel;
}