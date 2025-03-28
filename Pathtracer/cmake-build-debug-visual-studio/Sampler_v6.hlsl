inline float LinearizeVector(float3 v){
    //return (v.x + v.y + v.z)/3.0f;
    return length(v);
}

inline bool IsValidReservoir_GI(Reservoir_GI r){
    bool valid =
        length(r.nn) > 0.0f &&
        length(r.Vn) > 0.0f &&
        r.M > 0.0f;
    return valid;
}


inline float Jacobian_Reconnection(float3 x1r, float3 x1q, float3 x2q, float3 n2q)
{
    // Direction vectors from x2 up to x1
    float3 vq = x2q - x1q;
    float3 vr = x2q - x1r;

    // Cosines of incidence angles
    float cosPhi2q = abs(dot(normalize(-vq), normalize(n2q)));
    float cosPhi2r = abs(dot(normalize(-vr), normalize(n2q)));

    // Squared lengths of vq, vr
    float len2_vq = dot(vq, vq);
    float len2_vr = dot(vr, vr);

    // Final Jacobian
    float J = (cosPhi2q / cosPhi2r) * (len2_vr / len2_vq);
    if(J > 10.0f || J < 1.0f/10.0f || isnan(J) || isinf(J))
        return 0.0f;
    return J;
}


MaterialOptimized CreateMaterialOptimized(in Material mat, uint materialID)
{
    MaterialOptimized optMat;

    optMat.Kd               = half4(mat.Kd);
    optMat.Pr_Pm_Ps_Pc      = half4(mat.Pr_Pm_Ps_Pc);
    optMat.Ks               = half3(mat.Ks);
    optMat.Ke               = half3(mat.Ke);
    optMat.mID              = materialID;

    return optMat;
}


// The remaining functions remain unchanged.
inline float VisibilityCheck(
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

inline float3 ReconnectDI(
    float3 x1,
    float3 n1,
    float3 x2,
    float3 n2,
    float3 L,
    float3 outgoing,
    uint strategy,
    MaterialOptimized material
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
    float3 brdf0 = EvaluateBRDF(0, material, n1, normalize(-dir), normalize(outgoing));
    float3 brdf1 = EvaluateBRDF(1, material, n1, normalize(-dir), normalize(outgoing));
    float3 F1 = SafeMultiply(probs.x, brdf0);
    float3 F2 = SafeMultiply(probs.y, brdf1);
    float3 F = F1 + F2;

    return F * L * cosThetaX1 * cosThetaX2 / (dist * dist);
}


inline float3 ReconnectGI(
    float3 x1,
    float3 n1,
    float3 x2,
    float3 n2,
    float3 L, // contribution
    float3 V, // incoming direction from path
    float3 outgoing,
    MaterialOptimized material1,
    MaterialOptimized material2
)
{
    float3 dir = x2 - x1; // The reconnection direction

    float cosThetaX1 = dot(n1, normalize(dir));
    float cosThetaX2 = dot(n2, normalize(-V));

    float2 probs = CalculateStrategyProbabilities(material1, normalize(outgoing), n1);
    float3 brdf0 = EvaluateBRDF(0, material1, n1, normalize(-dir), normalize(outgoing));
    float3 brdf1 = EvaluateBRDF(1, material1, n1, normalize(-dir), normalize(outgoing));
    float3 F1 = SafeMultiply(probs.x, brdf0);
    float3 F2 = SafeMultiply(probs.y, brdf1);
    float3 Fx1 = F1 + F2;

    float2 probs_2 = CalculateStrategyProbabilities(material2, normalize(-dir), n2);
    float3 brdf0_2 = EvaluateBRDF(0, material2, n2, normalize(V), normalize(-dir));
    float3 brdf1_2 = EvaluateBRDF(1, material2, n2, normalize(V), normalize(-dir));
    float3 F1_2 = SafeMultiply(probs_2.x, brdf0_2);
    float3 F2_2 = SafeMultiply(probs_2.y, brdf1_2);
    float3 Fx2 = F1_2 + F2_2;

    return Fx1 * Fx2 * cosThetaX1 * cosThetaX2 * L;
}

float GetP_Hat(float3 x1, float3 n1, float3 x2, float3 n2, float3 L2, float3 o, uint s, MaterialOptimized matOpt, bool use_visibility){
    float f_g = LinearizeVector(ReconnectDI(x1, n1, x2, n2, L2, o, s, matOpt));
    float v = 1.0f;

    if(use_visibility){
        v = VisibilityCheck(x1, n1, normalize(x2-x1), length(x2-x1));
    }
    return f_g * v;
}

float3 GetP_Hat_GI(float3 x1, float3 n1, float3 x2, float3 n2, float3 L2, float3 V2, float3 o, MaterialOptimized matOpt1, MaterialOptimized matOpt2, bool use_visibility){
    float3 f_g = ReconnectGI(x1, n1, x2, n2, L2, V2, o, matOpt1, matOpt2);
    float v = 1.0f;

    if(use_visibility){
        v = VisibilityCheck(x1, n1, normalize(x2-x1), length(x2-x1));
    }
    return f_g * v;
}

inline float GetW(Reservoir_DI r, float p_hat){
    if(p_hat > EPSILON)
        return r.w_sum / p_hat;
    else
        return 0.0f;
}

inline float GetW_GI(Reservoir_GI r, float p_hat){
    if(p_hat > 0.0f && r.w_sum <= 1.0f)
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
    MaterialOptimized material,
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
        incoming = -sample;

        // Sample the BSDF for the light's direction
        float2 probs = CalculateStrategyProbabilities(material, normalize(outgoing), normal);

        float3 brdf0 = EvaluateBRDF(0, material, normal, -sample, normalize(outgoing));
        float3 brdf1 = EvaluateBRDF(1, material, normal, -sample, normalize(outgoing));

        float3 pdf0 = BRDF_PDF(0, material, normal, -sample, outgoing) * cos_theta / dist2;
        float3 pdf1 = BRDF_PDF(1, material, normal, -sample, outgoing) * cos_theta / dist2;

        float3 F1 = SafeMultiply(probs.x, brdf0);
        float3 F2 = SafeMultiply(probs.y, brdf1);
        float3 P1 = SafeMultiply(probs.x, pdf0);
        float3 P2 = SafeMultiply(probs.y, pdf1);
        float3 brdf = F1 + F2;
        pdf_bsdf = P1 + P2;

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
    MaterialOptimized material,
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

    float3 brdf0 = EvaluateBRDF(0, material, normal, -L_norm, normalize(outgoing));
    float3 brdf1 = EvaluateBRDF(1, material, normal, -L_norm, normalize(outgoing));

    float3 pdf0 = BRDF_PDF(0, material, normal, -L_norm, normalize(outgoing)) * cos_theta_y / dist2;
    float3 pdf1 = BRDF_PDF(1, material, normal, -L_norm, normalize(outgoing)) * cos_theta_y / dist2;

    float3 F1 = SafeMultiply(probs.x, brdf0);
    float3 F2 = SafeMultiply(probs.y, brdf1);
    float3 P1 = SafeMultiply(probs.x, pdf0);
    float3 P2 = SafeMultiply(probs.y, pdf1);
    float3 brdf_light = F1 + F2;
    float3 P = P1 + P2;

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
    pdf_bsdf = P;
    incoming = -L_norm;
}

// Use visibility is mandatory
float3 SampleLightBSDF_GI(
    inout float pdf_light, // Outputs (this time both in solid angle measure)
    inout float pdf_bsdf,
    inout float3 incoming, // light direction
    inout float3 new_origin,
    inout float3 new_normal,
    inout float3 new_outgoing,
    inout MaterialOptimized new_material,

    inout uint2 seed, // Inputs
    uint strategy,
    float3 origin,
    float3 normal,
    float3 outgoing,
    inout float3 acc_l, // Inout as this might be changed in case no light is hit
    inout float acc_pdf, // Same here
    inout float3 throughput, // Throughput change for this bounce
    inout float pdf, // Same here for pdf
    inout float3 emission, // Subpath emission
    MaterialOptimized material,
    bool isReconnection
    ){

    // Sample a BSDF direction
    float3 sample;
    float3 adjustedOrigin = float3(0,0,0);
    SampleBRDF(strategy, material, outgoing, normal, normal, sample, adjustedOrigin, origin, seed);

    // Trace the ray
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = sample;
    ray.TMin = s_bias;
    ray.TMax = 10000;
    HitInfo samplePayload;
    TraceRay(SceneBVH, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, samplePayload);

    // Evaluate the contribution
    MaterialOptimized mat_ke = {
        materials[samplePayload.materialID].Kd, materials[samplePayload.materialID].Pr_Pm_Ps_Pc,
        materials[samplePayload.materialID].Ks, materials[samplePayload.materialID].Ke, samplePayload.materialID
    };

    // Sample the BSDF for the light's direction
    float2 probs = CalculateStrategyProbabilities(material, normalize(outgoing), normal);

    float3 brdf0 = EvaluateBRDF(0, material, normal, -sample, normalize(outgoing));
    float3 brdf1 = EvaluateBRDF(1, material, normal, -sample, normalize(outgoing));

    float pdf0 = BRDF_PDF(0, material, normal, -sample, outgoing);
    float pdf1 = BRDF_PDF(1, material, normal, -sample, outgoing);

    float3 F1 = SafeMultiply(probs.x, brdf0);
    float3 F2 = SafeMultiply(probs.y, brdf1);
    float P1 = SafeMultiply(probs.x, pdf0);
    float P2 = SafeMultiply(probs.y, pdf1);
    float3 brdf = F1 + F2;
    float P = P1 + P2;

    pdf_bsdf = P;
    float NdotL = dot(normal, sample);

    if(length(mat_ke.Ke) > 0.0f){
        // If we hit a light, treat as a light sample
        float3 L = samplePayload.hitPosition - origin;
        float dist = length(L);
        float dist2 = dist * dist;
        float cos_theta = dot(samplePayload.hitNormal, -sample);

        pdf_light = (((mat_ke.Ke.x + mat_ke.Ke.y + mat_ke.Ke.z) / 3.0f) / g_EmissiveTriangles[0].total_weight) * dist2 / cos_theta;

        incoming = -sample;

        acc_pdf *= pdf_bsdf;
        acc_l *= brdf * NdotL;

        //----------------------------subpath parameters------------------------------
        //if(!isReconnection) // If this is the reconnection vertex, the brdf is evalutated later in the reconnection step.
        throughput = brdf * NdotL;

        pdf = pdf_bsdf;
        emission = mat_ke.Ke;

        return mat_ke.Ke * acc_l / acc_pdf;
    }
    else{

        incoming = -sample;
        // If we hit no light, continue the path. This means adjusting throughput and accumulated pdf accordingly
        acc_pdf *= pdf_bsdf;
        acc_l *= brdf * NdotL;

        // Also set the returned new path parameters
        new_origin = samplePayload.hitPosition;
        new_normal = samplePayload.hitNormal;
        new_outgoing = -sample;
        new_material = mat_ke;

        //----------------------------subpath parameters------------------------------
        //if(!isReconnection) // If this is the reconnection vertex, the brdf is evalutated later in the reconnection step.
        throughput = brdf * NdotL;

        pdf = pdf_bsdf;
        emission = float3(0,0,0);
        return float3(0,0,0);
    }
}


float3 SampleLightNEE_GI(
    inout float pdf_light, // Outputs (this time both in solid angle measure)
    inout float pdf_bsdf,
    inout float3 incoming, // light direction

    inout uint2 seed, // Inputs
    uint strategy,
    float3 origin,
    float3 normal,
    float3 outgoing,
    float3 acc_l, // Inout as this might be changed in case no light is hit
    float acc_pdf, // Same here
    inout float3 throughput,
    inout float pdf,
    inout float3 emission, // Subpath emission
    MaterialOptimized material,
    bool useVisibility,
    bool isReconnection
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

    // Generate random barycentric coordinates
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

    // Get the sample direction and compute distance
    float3 L = samplePoint - origin;
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

    float area_l = abs(length(cross_l) * 0.5f);
    float pdf_l = sampleLight.weight / max(area_l, EPSILON);

    // Compute cosine factors
    float cos_theta_x = max(dot(normal, L_norm), EPSILON);

    float cos_theta_y = max(dot(normal_l, -L_norm), EPSILON);


    // Compute the geometry term
    float G = cos_theta_x;
    float3 emission_l = sampleLight.emission;

    // Sample the BSDF for the light's direction
    float2 probs = CalculateStrategyProbabilities(material, normalize(outgoing), normal);

    float3 brdf0 = EvaluateBRDF(0, material, normal, -L_norm, normalize(outgoing));
    float3 brdf1 = EvaluateBRDF(1, material, normal, -L_norm, normalize(outgoing));

    float pdf0 = BRDF_PDF(0, material, normal, -L_norm, normalize(outgoing));
    float pdf1 = BRDF_PDF(1, material, normal, -L_norm, normalize(outgoing));

    float3 F1 = SafeMultiply(probs.x, brdf0);
    float3 F2 = SafeMultiply(probs.y, brdf1);
    float P1 = SafeMultiply(probs.x, pdf0);
    float P2 = SafeMultiply(probs.y, pdf1);
    float3 brdf_light = F1 + F2;
    float P = P1 + P2;

    // Optional visibility check
    float V = 1.0f;
    if(useVisibility == true){
        RayDesc ray;
        ray.Origin = origin + s_bias * normal;
        ray.Direction = L_norm;
        ray.TMin = 0.0f;
        ray.TMax = dist - s_bias * 2.0f;
        ShadowHitInfo shadowPayload;
        shadowPayload.isHit = false;
        TraceRay(SceneBVH, RAY_FLAG_NONE, 0xFF, 1, 0, 1, ray, shadowPayload);
        V = shadowPayload.isHit ? 0.0f : 1.0f;
    }

    pdf_light = max(EPSILON, pdf_l) * dist2 / cos_theta_y;
    pdf_bsdf = P;
    incoming = -L_norm;

    acc_pdf *= pdf_light;
    acc_l *= brdf_light * G * V;

    //----------------------------subpath parameters------------------------------
    if(!isReconnection) // If this is the reconnection vertex, the brdf is evalutated later in the reconnection step.
        throughput = brdf_light * G * V;
    else
        throughput = G;

    pdf = pdf_light;
    emission = emission_l;

    if(acc_pdf > 0.0f)
        return emission_l * acc_l / acc_pdf;
    else
        return float3(0,0,0);
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
    MaterialOptimized matOpt,
    inout uint2 seed
    ){

    float p_strategy = 1.0f;
    uint strategy = SelectSamplingStrategy(matOpt, outgoing, payload.hitNormal, seed, p_strategy);

    // Iterate through M1 NEE samples and fill up the reservoir
    [loop]
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
            matOpt,
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
    [loop]
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
            matOpt,
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
    reservoir.M = 1;
}

inline int2 GetBestReprojectedPixel_d(
    float3 worldPos,
    float4x4 prevView,
    float4x4 prevProjection,
    float2 resolution)
{
    float2 subPixelCoord = GetLastFramePixelCoordinates_Float(worldPos, prevView, prevProjection, resolution);
    int2 pixel = int2(round(subPixelCoord));
    return pixel;
}
