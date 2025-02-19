// This provides relevant wrapper functions for tracing a ray.


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
    SampleBRDF(strategy, material, outgoing, normal,normal, sample, origin, worldOrigin, seed);

    // Trace the ray
    //____________________________________________________________
    RayDesc ray;
    ray.Origin = worldOrigin; // Offset origin along the normal
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
	emission = material_ke.Ke;
	x2 = samplePayload.hitPosition;
	n2 = samplePayload.hitNormal;

    // If yes, get the samples triangle information
    if(Ke > EPSILON){
        float area_l = samplePayload.area;
        float3 L = samplePayload.hitPosition - worldOrigin;
        float dist = length(L);
        float dist2 = dist * dist;
        float cos_theta = max(EPSILON, dot(samplePayload.hitNormal, -sample));

        pdf_light = ((Ke / 3.0f) / g_EmissiveTriangles[0].total_weight);
		pdf_bsdf = max(EPSILON,BRDF_PDF(strategy, material, normal, -sample, outgoing) * cos_theta / dist2);
        incoming = -sample;

		float3 brdf = EvaluateBRDF(strategy, material, normal, -sample, outgoing);

        p_hat = length(brdf * material_ke.Ke * dot(normal, -incoming) * cos_theta / dist2);
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
	x2 = samplePoint;
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
	n2 = normal_l;

    float area_l = abs(length(cross_l) * 0.5f);


    // Compute the cosine of the angles at x and y
    float cos_theta_x = max(EPSILON, dot(normal, L_norm));       // Cosine at shading point
    float cos_theta_y = max(EPSILON, dot(normal_l, -L_norm));      // Cosine at light sample

    // Compute the geometry term
    float G = max(cos_theta_y * cos_theta_x / dist2, EPSILON);
    float pdf_l = sampleLight.weight / max(area_l, EPSILON);
    float3 emission_l =  sampleLight.emission;
	emission = emission_l;

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
    p_hat = length(emission_l * brdf_light * G * V);
    pdf_light = max(EPSILON,pdf_l);
    pdf_bsdf = pdf_brdf_light * cos_theta_y/dist2;
    incoming = -L_norm;
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
        float pdf_bsdf = 1.0f;
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

        // Calculate Resampling weight
        // MIS weight
        float mi =  pdf_light / (M1 * pdf_light + M2 * pdf_bsdf);
        float wi = mi * p_hat / pdf_light;

        // Update the reservoir with this entry
		if(p_hat > 0.0f)
			UpdateReservoir(reservoir, wi, 0.0f, p_hat, x2, n2, emission, strategy, seed);
    }

    // Iterate through M2 BSDF samples and fill them in the reservoir
    for(int j=0; j<M2; j++){
        float pdf_light = 1.0f;
        float pdf_bsdf = 1.0f;
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
        // Calculate Resampling weight
        // MIS weight
        float mi = pdf_bsdf / (M1 * pdf_light + M2 * pdf_bsdf);
        float wi = mi * p_hat / pdf_bsdf;

        // Update the reservoir with this entry
		if(p_hat > 0.0f)
        	UpdateReservoir(reservoir, wi, 0.0f, p_hat, x2, n2, emission, strategy, seed);
    }

    // First selected sample is canonical
    reservoir.M = 1.0f;
}

// Shoot a shadow ray to check for occlusion
float VisibilityCheck(
    float3 x1,
    float3 n1,
    float3 dir,
    float dist
)
{
    float V = 0.0f;
    //Shadow ray
    //____________________________________________________________
    RayDesc ray;
    ray.Origin = x1 + normalize(n1) * s_bias; // Offset origin along the normal
    ray.Direction = dir;
    ray.TMin = 0.0f;
    ray.TMax = max(dist - 2.0f*s_bias, 2.0f * s_bias);
    bool hit = true;
    // Initialize the ray payload
    ShadowHitInfo shadowPayload;
    shadowPayload.isHit = false;
    // Trace the ray
    TraceRay(SceneBVH, RAY_FLAG_NONE, 0xFF, 1, 0, 1, ray, shadowPayload);
    V = shadowPayload.isHit ? 0.0 : 1.0;
    //____________________________________________________________
    return V;
}

// Return light contribution without W: F * V * L * cosx * cosy / dist^2
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
    // Calculate light vector
    float3 dir = x2 - x1;
    // Calculate distance to reconnection point
    float dist = length(dir);
    if(length(L) == 0.0f)
        return float3(0,0,0);
	if (dist < 2.0f * s_bias) {
    	return float3(0, 0, 0); // or some other fallback value
	}

    // Calculate the angles (n1 o dir, n2 o -dir)
    float cosThetaX1 = max(0, dot(n1, normalize(dir)));
    float cosThetaX2 = max(0, dot(n2, normalize(-dir)));

    // Calculate F
    float3 F = EvaluateBRDF(strategy, material, n1, normalize(-dir), normalize(outgoing));

    return F * L * cosThetaX1 * cosThetaX2 / (dist * dist);
}

// This function selects among the 4 candidates the pixel with the smallest world space error.
int2 GetBestReprojectedPixel(
    float3 worldPos,
    float4x4 prevView,
    float4x4 prevProjection,
    float2 resolution)
{
    // Compute the sub-pixel coordinate.
    float2 subPixelCoord = GetLastFramePixelCoordinates_Float(worldPos, prevView, prevProjection, resolution);

    // Get the base (floor) coordinate.
    int2 basePixel = int2(floor(subPixelCoord));

    // Define the 4 candidate pixels.
    int2 candidates[4];
    candidates[0] = basePixel;
    candidates[1] = basePixel + int2(1, 0);
    candidates[2] = basePixel + int2(0, 1);
    candidates[3] = basePixel + int2(1, 1);

    float minDistance = 10000000.0f;
    int2 bestPixel = int2(-1, -1);

    // For each candidate, we need to fetch its stored world-space position.
    for (int i = 0; i < 4; i++)
    {
        int2 candidate = candidates[i];
        // Convert candidate to a 1D index if your reservoir is in a buffer:
        uint index = candidate.y * uint(resolution.x) + candidate.x;
        // Retrieve the candidate world position.
        float3 candidateWorldPos = g_Reservoirs_last[index].x1;

        // Compute the world space error.
        float dist = length(candidateWorldPos - worldPos);
        if (dist < minDistance)
        {
            minDistance = dist;
            bestPixel = candidate;
        }
    }

    return bestPixel;
}
