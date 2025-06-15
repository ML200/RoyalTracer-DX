inline float LinearizeVector(float3 v){
    //return (v.x + v.y + v.z)/3.0f;

    return length(v);
}

inline bool IsValidReservoir(Reservoir_DI r){
    bool valid =
        length(r.n2) > 0.0f &&
        length(r.L2) > 0.0f &&
        r.w_sum > 0.0f &&
        r.M > 0.0f;
    return valid;
}


inline bool IsValidReservoir_GI(Reservoir_GI r){
    bool valid =
        r.w_sum > 0.0f &&
        r.M > 0.0f;
    return valid;
}


/*inline float Jacobian_Reconnection(
    SampleData sdata_r,
    SampleData sdata_q,
    float3 x2q, float3 n2q,
    float3 i2)
{
    // Direction vectors from x2 up to x1
    float3 vq = x2q - sdata_q.x1;
    float3 vr = x2q - sdata_r.x1;

    // Cosines of incidence angles
    float cosPhi2q = abs(dot(normalize(-vq), normalize(n2q)));
    float cosPhi2r = abs(dot(normalize(-vr), normalize(n2q)));

    // Squared lengths of vq, vr
    float len2_vq = dot(vq, vq);
    float len2_vr = dot(vr, vr);

    // Final Jacobian
    float J = (cosPhi2q / cosPhi2r) * (len2_vr / len2_vq);
    return J;
}*/

inline float Jacobian_Reconnection(
    SampleData sdata_r,
    SampleData sdata_q,
    float3 x2q, float3 n2q)
{
    // Direction vectors from x2 up to x1
    float3 vq = x2q - sdata_q.x1;
    float3 vr = x2q - sdata_r.x1;

    // Cosines of incidence angles
    float cosPhi2q = abs(dot(normalize(-vq), normalize(n2q)));
    float cosPhi2r = abs(dot(normalize(-vr), normalize(n2q)));

    // Squared lengths of vq, vr
    float len2_vq = dot(vq, vq);
    float len2_vr = dot(vr, vr);

    // Final Jacobian
    float J = (cosPhi2q / cosPhi2r) * (len2_vr / len2_vq);
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
    MaterialOptimized material
)
{
    float3 dir = x2 - x1;
    float dist = length(dir);

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
    float3 outgoing,
    MaterialOptimized material1
)
{
    float3 dir = x2 - x1; // The reconnection direction

    float cosThetaX1 = abs(dot(n1, normalize(dir)));

    float2 probs = CalculateStrategyProbabilities(material1, normalize(outgoing), n1);
    float3 brdf0 = EvaluateBRDF(0, material1, n1, normalize(-dir), normalize(outgoing));
    float3 brdf1 = EvaluateBRDF(1, material1, n1, normalize(-dir), normalize(outgoing));
    float3 F1 = SafeMultiply(probs.x, brdf0);
    float3 F2 = SafeMultiply(probs.y, brdf1);
    float3 Fx1 = F1 + F2;

    float3 fr = Fx1 * cosThetaX1 * L;

    if(any(isnan(fr)) || any(isinf(fr)))
        return float3(0,0,0);

    return fr;
}

float GetP_Hat(float3 x1, float3 n1, float3 x2, float3 n2, float3 L2, float3 o, MaterialOptimized matOpt, bool use_visibility){
    float f_g = LinearizeVector(ReconnectDI(x1, n1, x2, n2, L2, o, matOpt));
    float v = 1.0f;

    if(use_visibility){
        v = VisibilityCheck(x1, n1, normalize(x2-x1), length(x2-x1));
    }
    return f_g * v;
}

float3 GetP_Hat_GI(float3 x1, float3 n1, float3 x2, float3 n2, float3 L2, float3 o, MaterialOptimized matOpt1, bool use_visibility){
    float3 f_g = ReconnectGI(x1, n1, x2, n2, L2, o, matOpt1);
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
    inout float3 x2_pos,

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

    x2_pos = samplePoint;

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
    float cos_theta_x = abs(dot(normal, L_norm));
    if(cos_theta_x < EPSILON)
        cos_theta_x = 0.0f;

    float cos_theta_y = abs(dot(normal_l, -L_norm));
    if(cos_theta_y < EPSILON)
        cos_theta_y = 0.0f;


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
        ray.Origin = origin + s_bias * normalize(normal);
        ray.Direction = L_norm;
        ray.TMin = 0.5f * s_bias;
        ray.TMax = max(s_bias, dist - (s_bias * 5.0f));
        ShadowHitInfo shadowPayload;
        shadowPayload.isHit = false;
        TraceRay(SceneBVH, RAY_FLAG_NONE, 0xFF, 1, 0, 1, ray, shadowPayload);
        V = shadowPayload.isHit ? 0.0f : 1.0f;
    }
    if(cos_theta_y > 0.f)
        pdf_light = max(EPSILON, pdf_l) * dist2 / cos_theta_y;
    pdf_bsdf = P;

    incoming = -L_norm;

    acc_pdf *= pdf_light;
    acc_l *= brdf_light * G * V;

    throughput = brdf_light * G * V;

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
            UpdateReservoir(reservoir, wi, 0.0f, x2, n2, emission, seed);
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
            UpdateReservoir(reservoir, wi, 0.0f, x2, n2, emission, seed);
    }
    // Set canonical weight
    reservoir.M = 1;
}

inline float2 GetLastFramePixelCoordinates_Float(
    float3 worldPos,
    float4x4 prevView,
    float4x4 prevProjection,
    float2 resolution,
    uint objID)
{
    // 1. Convert current world-space position back into the local space of this object:
    float4 localPos = mul(instanceProps[objID].objectToWorldInverse, float4(worldPos, 1.0f));

    // 2. Transform that local position by the *previous* frame's object-to-world matrix:
    float4 prevWorldPos = mul(instanceProps[objID].prevObjectToWorld, localPos);

    // 3. Project it into clip space using the previous frame’s view and projection:
    float4 clipPos = mul(prevProjection, mul(prevView, prevWorldPos));

    // If the clip-space w is not positive, it means the position was behind the camera last frame:
    if (clipPos.w <= 0.0f)
    {
        // Return some sentinel value that indicates it's off-screen or invalid:
        return float2(-1.0f, -1.0f);
    }

    // 4. Convert clip space to normalized device coordinates:
    float2 ndc = clipPos.xy / clipPos.w;

    // 5. Transform NDC (-1..1) to screen UV (0..1):
    float2 screenUV = ndc * 0.5f + 0.5f;

    // 6. Flip Y if needed (common in many rendering APIs):
    screenUV.y = 1.0f - screenUV.y;

    // 7. Finally convert to actual pixel coordinates:
    return screenUV * resolution;
}

inline float2 GetLastFramePixelCoordinates_Float(
    float3 worldPos,
    float4x4 prevView,
    float4x4 prevProjection,
    float2 resolution,
    uint objID)
{
    // 1. Convert current world-space position back into the local space of this object:
    float4 localPos = mul(instanceProps[objID].objectToWorldInverse, float4(worldPos, 1.0f));

    // 2. Transform that local position by the *previous* frame's object-to-world matrix:
    float4 prevWorldPos = mul(instanceProps[objID].prevObjectToWorld, localPos);

    // 3. Project it into clip space using the previous frame’s view and projection:
    float4 clipPos = mul(prevProjection, mul(prevView, prevWorldPos));

    // If the clip-space w is not positive, it means the position was behind the camera last frame:
    if (clipPos.w <= 0.0f)
    {
        // Return some sentinel value that indicates it's off-screen or invalid:
        return float2(-1.0f, -1.0f);
    }

    // 4. Convert clip space to normalized device coordinates:
    float2 ndc = clipPos.xy / clipPos.w;

    // 5. Transform NDC (-1..1) to screen UV (0..1):
    float2 screenUV = ndc * 0.5f + 0.5f;

    // 6. Flip Y if needed (common in many rendering APIs):
    screenUV.y = 1.0f - screenUV.y;

    // 7. Finally convert to actual pixel coordinates:
    return screenUV * resolution;
}

inline int2 GetBestReprojectedPixel_d(
    float3 worldPos,
    float4x4 prevView,
    float4x4 prevProjection,
    float2 resolution,
    uint objID
    )
{
    float2 subPixelCoord = GetLastFramePixelCoordinates_Float(worldPos, prevView, prevProjection, resolution, objID);
    int2 pixel = int2(round(subPixelCoord));
    return pixel;
}

struct BilinearResult
{
    int2  coords[4];      // The four pixel coordinates
    float distances[4];   // Distance from subPixelCoord
    float weights[4];     // Bilinear weights that sum to ~1.0
};

inline BilinearResult GetBestReprojectedPixelBilinear_CenterBased(
    float3   worldPos,
    float4x4 prevView,
    float4x4 prevProjection,
    float2   resolution,
    uint     objID
)
{
    BilinearResult result;

    // Initialize everything to sentinel values:
    [unroll]
    for (int i = 0; i < 4; i++)
    {
        result.coords[i]    = int2(-1, -1);
        result.distances[i] = -1.0f;
        result.weights[i]   = 0.0f;
    }

    // (A) Get the sub-pixel coordinate in "center-based" space from prev frame:
    float2 subPixelCoord = GetLastFramePixelCoordinates_Float(
        worldPos, prevView, prevProjection, resolution, objID
    );

    // If off-screen or invalid, return the sentinel data:
    if (subPixelCoord.x < 0.0f || subPixelCoord.y < 0.0f ||
        subPixelCoord.x >= resolution.x || subPixelCoord.y >= resolution.y)
    {
        return result;
    }

    // (B) The integer pixel index if each pixel center = integer coordinate.
    //     e.g. if subPixelCoord.x = 2.0 => that's exactly the center of pixel #2 in X.
    //     We'll floor() here, but you could also cast to int if your pipeline requires rounding.
    int2 basePixel;
    basePixel.x = (int)floor(subPixelCoord.x);
    basePixel.y = (int)floor(subPixelCoord.y);

    // Clamp to screen bounds (avoid out-of-range):
    basePixel.x = clamp(basePixel.x, 0, (int)resolution.x - 1);
    basePixel.y = clamp(basePixel.y, 0, (int)resolution.y - 1);

    // (C) The next pixel in X and Y, also clamped:
    int2 nextX = int2(min(basePixel.x + 1, (int)resolution.x - 1), basePixel.y);
    int2 nextY = int2(basePixel.x, min(basePixel.y + 1, (int)resolution.y - 1));
    int2 nextXY = int2(nextX.x, nextY.y);

    // (D) The fractional offset inside the "center-based" pixel:
    //     e.g. if subPixelCoord.x = 2.3 => basePixel.x=2 => fracX=0.3 => we are 0.3 right of the center of pixel #2.
    float fracX = subPixelCoord.x - (float)basePixel.x;
    float fracY = subPixelCoord.y - (float)basePixel.y;

    // Ensure 0..1 range if user’s math might push it slightly out of range:
    fracX = saturate(fracX);
    fracY = saturate(fracY);

    // The four corners around basePixel:
    //   c0 = basePixel
    //   c1 = (basePixel.x+1, basePixel.y)
    //   c2 = (basePixel.x, basePixel.y+1)
    //   c3 = (basePixel.x+1, basePixel.y+1)
    int2 c0 = basePixel;
    int2 c1 = nextX;
    int2 c2 = nextY;
    int2 c3 = nextXY;

    // (E) Distances from the subpixel center (for reference or debug):
    float2 fCenter = subPixelCoord;
    float2 fC0 = float2(c0);
    float2 fC1 = float2(c1);
    float2 fC2 = float2(c2);
    float2 fC3 = float2(c3);

    float d0 = distance(fCenter, fC0);
    float d1 = distance(fCenter, fC1);
    float d2 = distance(fCenter, fC2);
    float d3 = distance(fCenter, fC3);

    // (F) Bilinear weights:
    float w0 = (1.0f - fracX) * (1.0f - fracY);
    float w1 = fracX * (1.0f - fracY);
    float w2 = (1.0f - fracX) * fracY;
    float w3 = fracX * fracY;

    // (G) Store final results:
    result.coords[0]    = c0;
    result.coords[1]    = c1;
    result.coords[2]    = c2;
    result.coords[3]    = c3;

    result.distances[0] = d0;
    result.distances[1] = d1;
    result.distances[2] = d2;
    result.distances[3] = d3;

    result.weights[0]   = w0;
    result.weights[1]   = w1;
    result.weights[2]   = w2;
    result.weights[3]   = w3;

    return result;
}

