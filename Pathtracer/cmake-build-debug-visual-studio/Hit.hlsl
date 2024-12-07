#include "BRDF.hlsl"

// #DXR Extra - Another ray type
struct ShadowHitInfo {
  bool isHit;
};

struct InstanceProperties
{
  float4x4 objectToWorld;
  float4x4 prevObjectToWorld;
  float4x4 objectToWorldNormal;
  float4x4 prevObjectToWorldNormal;
};

struct LightTriangle {
    float3 x;
    float  pad0;
    float3 y;
    float  pad1;
    float3 z;
    float  pad2;
    uint   instanceID;
    float  weight;
    uint   triCount;
    float  total_weight;
    float3 emission;
    float  cdf;
};


struct STriVertex {
  float3 vertex;
  float4 normal;
};

// #DXR Extra: Per-Instance Data
cbuffer Colors : register(b0) {
  float3 A;
  float3 B;
  float3 C;
}

float3 SchlickFresnel_DEBUG(float3 F0, float cosTheta)
{
    return  F0 + (1.0f - F0) * pow(abs(1.0f - cosTheta), 5.0f);
}


StructuredBuffer<STriVertex> BTriVertex : register(t2);
StructuredBuffer<int> indices : register(t1);
RaytracingAccelerationStructure SceneBVH : register(t0);
StructuredBuffer<InstanceProperties> instanceProps : register(t3);
StructuredBuffer<uint> materialIDs : register(t4);
StructuredBuffer<Material> materials : register(t5);
StructuredBuffer<LightTriangle> g_EmissiveTriangles : register(t6);


[shader("closesthit")] void ClosestHit(inout HitInfo payload,
                                       Attributes attrib) {
    //Get information about the surface hit
    uint vertId = 3 * PrimitiveIndex();
    uint materialID = materialIDs[vertId+BTriVertex[indices[vertId]].normal.w];
    float3 barycentrics = float3(1.f - attrib.bary.x - attrib.bary.y, attrib.bary.x, attrib.bary.y);

    // Calculate the position of the intersection point
    float3 hitPosition = BTriVertex[indices[vertId]].vertex * barycentrics.x +
           BTriVertex[indices[vertId + 1]].vertex * barycentrics.y +
           BTriVertex[indices[vertId + 2]].vertex * barycentrics.z;


    //_____________________________________________________________________________________________________________________________________________
    //Determine the impact normal. Apply interpolation if necessary.
    // Determine if any vertex of the triangle requests flat shading
    float3 normal = float3(0, 0, 0);
    // Always calculate the flat shading normal
    float3 e1 = BTriVertex[indices[vertId + 1]].vertex - BTriVertex[indices[vertId]].vertex;
    float3 e2 = BTriVertex[indices[vertId + 2]].vertex - BTriVertex[indices[vertId]].vertex;
    float3 flatNormal = normalize(cross(e1, e2));

    // Initialize the smooth shading normal to zero
    //____________________________________________________________
    float3 smoothNormal = float3(0, 0, 0);

    // Check each vertex normal; accumulate if not zero, otherwise use flat normal
    for (int i = 0; i < 3; i++) {
        if (all(BTriVertex[indices[vertId + i]].normal.xyz != float3(0, 0, 0))) {
            // Accumulate the weighted normals for smooth shading
            smoothNormal += BTriVertex[indices[vertId + i]].normal.xyz * barycentrics[i];
        } else {
            // One of the vertex normals is (0,0,0), use flat shading for this vertex
            smoothNormal += flatNormal * barycentrics[i];
        }
    }

    // Normalize the smooth normal if it's not near-zero
    if (length(smoothNormal) > 0.0001) {
        normal = normalize(smoothNormal);
    } else {
        // Fallback to flat shading if the smooth normal is near-zero
        normal = flatNormal;
    }

    // Transform normal to world space and adjust for ray direction
    //____________________________________________________________
    normal = normalize(mul(instanceProps[InstanceID()].objectToWorldNormal, float4(normal, 0.f)).xyz);
    flatNormal = normalize(mul(instanceProps[InstanceID()].objectToWorldNormal, float4(flatNormal, 0.f)).xyz);
    if (dot(normal, -payload.direction) < 0.0f)
        normal = -normal; // Flip the normal to match the ray direction
    //_____________________________________________________________________________________________________________________________________________

    // # MIS - Multiple Importance Sampling for lights
    float3 worldOrigin = WorldRayOrigin() + RayTCurrent() * WorldRayDirection();

    // Set variables needed for later evaluation
    float3 direct = 0.0f;
    float3 emissive = 0.0f;
    float pdf_sample = 1.0f; // Set to 1 to ensure no NaN values
    float3 brdf_sample = float3(0.0f,0.0f,0.0f);
    float3 origin = payload.origin;
    float3 incoming = -payload.direction;

    //If we hit a emissive surface, calculate MIS weights and end the path
    if(length(materials[materialID].Ke) > 0.0f){
        // If its the first bounce, we didnt sample the lightsource, so the weight is 1.0
        if(payload.util.y == 0.0f){
            emissive = materials[materialID].Ke;
            payload.util.x = 1.0f;
        }
        else{
            //Calculate the G term
            float3 L_emissive = worldOrigin - origin;
            float dist2_emissive = max(dot(L_emissive, L_emissive), EPSILON);
            float dist_emissive = max(sqrt(dist2_emissive), EPSILON);
            float3 L_emissive_norm = L_emissive / dist_emissive;

            // Cosine terms for shading point and emissive surface
            float cos_theta_shading = max(EPSILON, dot(payload.hitNormal, L_emissive_norm)); // Cosine at shading point
            float cos_theta_emissive = max(EPSILON, dot(normal, -L_emissive_norm)); // Cosine at emissive surface

            float G_emissive = (cos_theta_shading * cos_theta_emissive) / dist2_emissive;

            //Calculate the relative pdf_l using the area and weight

            // Get the vertices
            float4x4 conversionMatrix = instanceProps[InstanceID()].objectToWorld;
            float3 x_v = mul(conversionMatrix, float4(BTriVertex[indices[vertId]].vertex, 1.f)).xyz;
            float3 y_v = mul(conversionMatrix, float4(BTriVertex[indices[vertId + 1]].vertex, 1.f)).xyz;
            float3 z_v = mul(conversionMatrix, float4(BTriVertex[indices[vertId + 2]].vertex, 1.f)).xyz;

            // Calculate the triangle area
            float3 edge1 = y_v - x_v;
            float3 edge2 = z_v - x_v;
            float3 cross_l = cross(edge1, edge2);
            float area_l = abs(length(cross_l) * 0.5f);

            // Calculate the emissive triangle weight: area * emissiveness / total weight
            float s_weight = area_l * ((materials[materialID].Ke.x + materials[materialID].Ke.y + materials[materialID].Ke.z) / 3.0f);
            float t_weight = g_EmissiveTriangles[0].total_weight;
            float weight = s_weight/t_weight;

            // Calculate the pdf for sampling this exact light
            float pdf_l = max(EPSILON, weight * dist2_emissive / cos_theta_emissive); // Adjust for solid angle space

            //Calculate the MIS weight
            float weight_emissive = payload.pdf  / (payload.pdf + pdf_l); // Convert the pdf to solid angle space
            // Calculate the MIS-weighted illumation
            emissive = materials[materialID].Ke * payload.colorAndDistance.xyz * weight_emissive;
            payload.util.x = 1.0f;
        }

    }
    else{ // Else we perform NEE, using RIS to select the most optimal light to sample

        //As materials might combine lobes, we have to select one:
        float p_strategy = 1.0f;
        float3 outgoing = -payload.direction; //Outgoing is the direction to the camera
        uint strategy = SelectSamplingStrategy(materials[materialID], outgoing, normal, payload.seed, p_strategy);

        // Temporary buffers for RIS
        float ris_weights[RIS_M];
        uint ris_indices[RIS_M];
        float3 ris_f[RIS_M];
        float ris_dist[RIS_M];
        float ris_cos_theta[RIS_M];
        float ris_pdf_brdf_light[RIS_M];
        float3 ris_LDir[RIS_M];
        float ris_cdf[RIS_M]; // Unordered cdf (unnormalized) for semi-fast access (perform reservoir sampling later)
        float ris_pdf_l[RIS_M];
        float ris_total_weight = 0.0f;

        // Iterate through the lights M times and save their RIS weight (in this case wi = (1/M) * (p^/p) = f/(p*M)) as well as their index in the lights array

        for(uint i = 0; i < RIS_M; i++){
            // Generate a random float in [0,1)
            float randomValue = RandomFloat(payload.seed);

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
            float xi1 = RandomFloat(payload.seed);
            float xi2 = RandomFloat(payload.seed);

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
            float3 L = samplePoint - (worldOrigin + s_bias * flatNormal);
            float dist2 = max(dot(L, L), EPSILON);
            float dist = max(sqrt(dist2), EPSILON);
            float3 L_norm = L / dist; // Sampling direction

            // Compute the normal vector using the cross product of the edge vectors
            float3 edge1 = y_v - x_v;
            float3 edge2 = z_v - x_v;
            float3 cross_l = cross(edge1, edge2);
            float3 normal_l = normalize(cross_l);
            float area_l = abs(length(cross_l) * 0.5f);


            // Compute the cosine of the angles at x and y
            float cos_theta_x = max(EPSILON, dot(normal, L_norm));       // Cosine at shading point
            float cos_theta_y = max(EPSILON, dot(normal_l, -L_norm));      // Cosine at light sample

            // Compute the geometry term
            float G = max((cos_theta_x * cos_theta_y) / dist2, EPSILON);
            float pdf_l = sampleLight.weight / max(area_l, EPSILON);
            float3 emission_l =  sampleLight.emission;

            // Get the light BRDF
            //____________________________________________________________
            //We sample the BSDF for the selected importance light.
            float3 brdf_light = EvaluateBRDF(strategy, materials[materialID], normal, -L_norm, -payload.direction);
            // We also need to get the pdf for that sample
            float pdf_brdf_light = max(BRDF_PDF(strategy, materials[materialID], normal, -L_norm, -payload.direction), EPSILON);

            // Save the important information in our resampling list
            // RIS weights: mi*p^/p
            ris_f[i] = emission_l * brdf_light * G;
            ris_weights[i] = (1.0f/RIS_M) * (((emission_l.x + emission_l.y + emission_l.z) / 3.0f * brdf_light * G) / pdf_l); // Use luminance to get a scalar value
            ris_indices[i] = selectedIndex;
            ris_LDir[i] = L_norm;
            ris_dist[i] = dist;
            ris_cos_theta[i] = cos_theta_y;
            ris_pdf_brdf_light[i] = pdf_brdf_light;
            ris_pdf_l[i] = pdf_l;


        }

        // Select a sample from the list based on weights
        // Create CDF
        ris_cdf[0] = ris_weights[0];
        for (uint i = 1; i < RIS_M; i++) {
            ris_cdf[i] = ris_cdf[i-1] + ris_weights[i];
        }
        ris_total_weight = ris_cdf[RIS_M-1];

        //Random number
        float randomSelectRIS = RandomFloat(payload.seed);
        float threshold = randomSelectRIS * ris_total_weight;

        uint selectedCandidate = 0;
        for (uint i = 0; i < RIS_M; i++) {
            if (threshold < ris_cdf[i]) {
                selectedCandidate = i;
                break;
            }
        }

        // Calculate the unbiased contribution weight WX = 1/p^ * ris_total_weight
        float WX = 1.0f/ris_f[selectedCandidate] * ris_total_weight;

        //Shadow ray
        //____________________________________________________________
        RayDesc ray;
        ray.Origin = worldOrigin + s_bias * flatNormal; // Offset origin along the normal
        ray.Direction = ris_LDir[selectedCandidate];
        ray.TMin = s_bias;
        ray.TMax = length(ris_dist[selectedCandidate]) - s_bias;
        bool hit = true;
        // Initialize the ray payload
        ShadowHitInfo shadowPayload;
        shadowPayload.isHit = false;
        // Trace the ray
        TraceRay(SceneBVH, RAY_FLAG_NONE, 0xFF, 1, 0, 1, ray, shadowPayload);
        float visible = shadowPayload.isHit ? 0.0 : 1.0;
        //____________________________________________________________

        //Direct lighting:
        direct  = ris_f[selectedCandidate] * visible * WX;
        // MIS Weight: Use G to convert from area space to solid-angle space. Fuck me, took forever to find this.
        float pdf_l_sa = max(EPSILON, (1.0f/WX) * ris_dist[selectedCandidate] * ris_dist[selectedCandidate] / ris_cos_theta[selectedCandidate]);
        float weight_light = pdf_l_sa / (pdf_l_sa + ris_pdf_brdf_light[selectedCandidate]);
        // Also adjust for the throughput
        direct  *= payload.colorAndDistance.xyz * weight_light;
        //____________________________________________________________


        //Get the sample BRDF:
        //____________________________________________________________
        //First, we sample the BSDF of the given material - payload.direction and payload.origin will be altered. We need the incoming ray direction later, so we buffer it.
        SampleBRDF(strategy, materials[materialID], outgoing, normal,flatNormal, payload.direction, payload.origin, worldOrigin, payload.seed);
        incoming = -payload.direction;

        //Check if we generated an invalid sample, if so set throughput to 0 and terminate ray here, effectively discarding the sample
        if(length(payload.direction) < 0.01f){
            //payload.emission = 0.0f;
            payload.util.x = 1.1f; //Terminate ray
        }
        else{
            //PDF
            pdf_sample = max(BRDF_PDF(strategy, materials[materialID], normal, incoming, outgoing),0.0001f);
            //Now, we evaluate the BRDF for the new sample. Our incident vector is the former ray direction vector and our outgoing vector is the new ray direction.
            brdf_sample = EvaluateBRDF(strategy, materials[materialID], normal, incoming, outgoing);
        }
    }

    //____________________________________________________________

    payload.emission += abs(direct) + abs(emissive);
    //payload.u_emission += materials[materialID].Ke * payload.colorAndDistance.xyz;
    //payload.emission += materials[materialID].Ke * payload.colorAndDistance.xyz;

    //Adjust the throughput
    payload.colorAndDistance = float4(payload.colorAndDistance.xyz * brdf_sample * dot(normal, -incoming) / pdf_sample, RayTCurrent());

    //Save the pdf_sample in case the next ray hits a light source
    payload.pdf = pdf_sample;

    //_________DEBUG___________
    //float cosTheta = dot(normal, outgoing);
    //float3 fresnel = SchlickFresnel_DEBUG(materials[materialID].Ks, cosTheta);
    //payload.emission = 1.0f/pdf_sample;
    //_________DEBUG___________

    //Misc
    //____________________________________________________________
    payload.hitNormal = normal;
    payload.reflectiveness = 1.0f-materials[materialID].Pr_Pm_Ps_Pc.x;
}




// #DXR Extra - Another ray type (unused for now)
[shader("closesthit")] void PlaneClosestHit(inout HitInfo payload,
                                                Attributes attrib) {
    payload.emission = float3(InstanceID(),1,1);

}