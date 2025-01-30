// This provides relevant wrapper functions for tracing a ray.

// Use visibility is mandatory
void SampleLightBSDF(
    inout float pdf_light, // Outputs
    inout float pdf_light_sa,
    inout float pdf_bsdf,
    inout float3 f,
    inout float3 incoming,
    inout float p_hat,
    inout float3 dir_l,
    inout float dist_l,

    inout uint2 seed, // Inputs
    float3 worldOrigin,
    float3 normal,
    float3 outgoing,
    Material material,
    uint strategy
    ){

    // Sample a BSDF direction
    float3 sample;
    float3 origin = worldOrigin;
    SampleBRDF(strategy, material, outgoing, normal,normal, sample, origin, worldOrigin, seed);

    // Trace the ray
    //____________________________________________________________
    RayDesc ray;
    ray.Origin = worldOrigin + s_bias * normal; // Offset origin along the normal
    ray.Direction = sample;
    ray.TMin = s_bias;
    ray.TMax = 10000;
    bool hit = true;
    // Initialize the ray payload
    HitInfo samplePayload;
    // Trace the ray
    TraceRay(SceneBVH, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, samplePayload);
    //____________________________________________________________

    // Evaluate the contribution
    // Did we hit an emissive surface?
    Material material_ke = materials[samplePayload.materialID];
    float Ke = material_ke.Ke.x + material_ke.Ke.y + material_ke.Ke.z;


    pdf_bsdf = max(EPSILON,BRDF_PDF(strategy, material, normal, -sample, outgoing));

    // If yes, get the samples triangle information
    if(Ke > EPSILON){
        float area_l = samplePayload.area;
        float3 L = samplePayload.hitPosition - worldOrigin;
        float dist = length(L);
        float dist2 = dist * dist;
        float cos_theta = max(EPSILON, dot(samplePayload.hitNormal, -sample));

        pdf_light = ((Ke / 3.0f) / g_EmissiveTriangles[0].total_weight);
        pdf_light_sa = max(EPSILON, pdf_light * dist2 / cos_theta);
        incoming = -sample;

        float3 brdf = EvaluateBRDF(strategy, material, normal, -sample, outgoing);

        p_hat = (brdf.x * material_ke.Ke.x + brdf.y * material_ke.Ke.y + brdf.z * material_ke.Ke.z) / 3.0f * dot(normal, -incoming);
        f = brdf * material_ke.Ke * dot(normal, -incoming);
        dist_l = dist;
        dir_l = normalize(L);
    }
    else{
        pdf_light = 0.0f;
        pdf_light_sa = 0.0f;
        dist_l = 0.0f;
        dir_l = float3(0,0,0);
        p_hat = 0.0f;
        f = float3(0,0,0);
    }
}



void SampleLightNEE(
    inout float pdf_light, // Outputs
    inout float pdf_light_sa,
    inout float pdf_bsdf,
    inout float3 f,
    inout float3 incoming,
    inout float p_hat,
    inout float3 dir_l,
    inout float dist_l,

    inout uint2 seed, // Inputs
    float3 worldOrigin,
    float3 normal,
    float3 outgoing,
    Material material,
    uint strategy,
    bool useVisibility
    ){

    // Sample a Light Triangle
    // Generate a random float in [0,1)
    float randomValue = RandomFloat(seed);

    // Get a random light source based on the weights
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
    //____________________________________________________________

    //Calculate the current world coordinates of the given triangle
    //Get the conversion matrix:
    float4x4 conversionMatrix = instanceProps[sampleLight.instanceID].objectToWorld;
    float3 x_v = mul(conversionMatrix, float4(sampleLight.x, 1.f)).xyz;
    float3 y_v = mul(conversionMatrix, float4(sampleLight.y, 1.f)).xyz;
    float3 z_v = mul(conversionMatrix, float4(sampleLight.z, 1.f)).xyz;
    //____________________________________________________________

    // Generate two random numbers in [0,1)
    float xi1 = RandomFloat(seed);
    float xi2 = RandomFloat(seed);

    // Ensure the sample lies within the triangle
    if (xi1 + xi2 > 1.0f) {
        xi1 = 1.0f - xi1;
        xi2 = 1.0f - xi2;
    }

    // Compute the barycentric coordinates
    float u = 1.0f - xi1 - xi2;
    float v = xi1;
    float w = xi2;

    // Calculate the sample point in world coordinates
    float3 samplePoint = u * x_v + v * y_v + w * z_v;
    //____________________________________________________________

    // Get the sample direction
    float3 L = samplePoint - worldOrigin;
    float dist2 = max(abs(dot(L, L)), EPSILON);
    float dist = max(sqrt(dist2), EPSILON);
    float3 L_norm = normalize(L); // Sampling direction

    // Compute the normal vector using the cross product of the edge vectors
    float3 edge1 = y_v - x_v;
    float3 edge2 = z_v - x_v;
    float3 cross_l = cross(edge1, edge2);
    float3 normal_l = normalize(cross_l);

    if(dot(normal_l, -L_norm) < 0.0f){
        normal_l = -normal_l;
    }

    float area_l = abs(length(cross_l) * 0.5f);


    // Compute the cosine of the angles at x and y
    float cos_theta_x = max(EPSILON, dot(normal, L_norm));       // Cosine at shading point
    float cos_theta_y = max(EPSILON, dot(normal_l, -L_norm));      // Cosine at light sample

    // Compute the geometry term
    float G = max(cos_theta_y * cos_theta_x / dist2, EPSILON);
    float pdf_l = sampleLight.weight / max(area_l, EPSILON);
    float3 emission_l =  sampleLight.emission;

    // Get the light BRDF
    //____________________________________________________________
    //We sample the BSDF for the selected importance light.
    float3 brdf_light = EvaluateBRDF(strategy, material, normal, -L_norm, normalize(outgoing));
    // We also need to get the pdf for that sample
    float pdf_brdf_light = max(BRDF_PDF(strategy, material, normal, -L_norm, normalize(outgoing)), EPSILON);


    // Trace the ray if using visibility term, otherwise V = 1
    float V = 1.0f;
    if(useVisibility == true){
        //Shadow ray
        //____________________________________________________________
        RayDesc ray;
        ray.Origin = worldOrigin + s_bias * normal; // Offset origin along the normal
        ray.Direction = L_norm;
        ray.TMin = 0.0f;
        ray.TMax = dist - s_bias * 2.0f;
        bool hit = true;
        // Initialize the ray payload
        ShadowHitInfo shadowPayload;
        shadowPayload.isHit = false;
        // Trace the ray
        TraceRay(SceneBVH, RAY_FLAG_NONE, 0xFF, 1, 0, 1, ray, shadowPayload);
        V = shadowPayload.isHit ? 0.0 : 1.0;
        //____________________________________________________________
    }


    // Evaluate the contribution
    p_hat = (emission_l.x * brdf_light.x + emission_l.y * brdf_light.y + emission_l.z * brdf_light.z) / 3.0f * G * V;
    f = emission_l * brdf_light * G * V;
    pdf_light = max(EPSILON,pdf_l);
    pdf_light_sa = pdf_light * dist2/cos_theta_y;
    pdf_bsdf = pdf_brdf_light;
    incoming = -L_norm;
    dist_l = dist;
    dir_l = L_norm;
}


// Sample RIS for direct Lights (for now) on a given point
//M1: Number NEE samples
//M2: Number BSDF samples
void SampleRIS(
    uint M1,
    uint M2,
    float3 outgoing,
    inout Reservoir reservoir,
    HitInfo payload,
    inout uint2 seed
    ){

    Material material = materials[payload.materialID];
    float p_strategy = 1.0f;
    uint strategy = SelectSamplingStrategy(material, outgoing, payload.hitNormal, seed, p_strategy);

    // Iterate through M1 NEE samples and fill up the reservoir
    for(int i=0; i<M1; i++){
        float pdf_light = 1.0f;
        float pdf_light_sa = 1.0f;
        float pdf_bsdf = 1.0f;
        float3 f;
        float3 incoming;
        float p_hat;
        float dist;
        float3 dir;

        SampleLightNEE(
            pdf_light,
            pdf_light_sa,
            pdf_bsdf,
            f,
            incoming,
            p_hat,
            dir,
            dist,
            seed,
            payload.hitPosition,
            payload.hitNormal,
            outgoing,
            material,
            strategy,
            false // Test with visibility for now
            );

        // Calculate Resampling weight
        // MIS weight
        float mi = pdf_light_sa / (M1 * pdf_light_sa + M2 * pdf_bsdf);
        float wi = mi * p_hat / pdf_light;

        // Update the reservoir with this entry
        UpdateReservoir(reservoir, wi, 1.0f/(M1 + M2), seed, f, p_hat, true, 0.0f, dir, dist, payload.hitPosition, payload.hitNormal);
    }

    // Iterate through M2 BSDF samples and fill them in the reservoir
    for(int i=0; i<M2; i++){
        float pdf_light = 1.0f;
        float pdf_light_sa = 1.0f;
        float pdf_bsdf = 1.0f;
        float3 f;
        float3 incoming;
        float p_hat;
        float dist;
        float3 dir;

        SampleLightBSDF(
            pdf_light,
            pdf_light_sa,
            pdf_bsdf,
            f,
            incoming,
            p_hat,
            dir,
            dist,
            seed,
            payload.hitPosition,
            payload.hitNormal,
            outgoing,
            material,
            strategy
            );
        // Calculate Resampling weight
        // MIS weight
        float mi = pdf_bsdf / (M1 * pdf_light_sa + M2 * pdf_bsdf);
        float wi = mi * p_hat / pdf_bsdf;

        // Update the reservoir with this entry
        UpdateReservoir(reservoir, wi, 1.0f/(M1 + M2), seed, f, p_hat, true, 0.0f, dir, dist, payload.hitPosition, payload.hitNormal);
    }
}