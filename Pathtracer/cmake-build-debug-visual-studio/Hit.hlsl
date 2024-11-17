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
    float  pad3;
    float3 emission;
    float  pad4;
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
    if (dot(payload.direction, normal) > 0.0f) {
        normal = -normal; // Flip the normal if hitting from behind
    }
    if (dot(payload.direction, flatNormal) > 0.0f) {
        flatNormal = -flatNormal; // Flip the normal if hitting from behind
    }
    //_____________________________________________________________________________________________________________________________________________

    // # MIS - Multiple Importance Sampling for lights
    float3 worldOrigin = WorldRayOrigin() + RayTCurrent() * WorldRayDirection();

    // Get a random light source based on the weights (Replace later with binary search)
    int random = int(RandomFloatLCG(payload.seed.x) * g_EmissiveTriangles[0].triCount);
    LightTriangle sampleLight = g_EmissiveTriangles[random];
    //____________________________________________________________

    //Calculate the current world coordinates of the given triangle
    //Get the conversion matrix:
    float4x4 conversionMatrix = instanceProps[sampleLight.instanceID].objectToWorld;
    float3 x_v = mul(conversionMatrix, float4(sampleLight.x, 1.f)).xyz;
    float3 y_v = mul(conversionMatrix, float4(sampleLight.y, 1.f)).xyz;
    float3 z_v = mul(conversionMatrix, float4(sampleLight.z, 1.f)).xyz;
    //____________________________________________________________


    // Generate two random numbers in [0,1)
    float xi1 = RandomFloatLCG(payload.seed.x);
    float xi2 = RandomFloatLCG(payload.seed.x);

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
    float dist2 = dot(L, L);
    float dist = sqrt(dist2);
    float3 L_norm = L / dist; // Sampling direction

    // Compute the normal vector using the cross product of the edge vectors
    float3 edge1 = y_v - x_v;
    float3 edge2 = z_v - x_v;
    float3 cross_l = cross(edge1, edge2);
    float3 normal_l = normalize(cross_l);
    float area_l = abs(length(cross_l) * 0.5f);


    // Compute the cosine of the angles at x and y
    float cos_theta_x = max(0.0, dot(normal, L_norm));       // Cosine at shading point
    float cos_theta_y = max(0.0, dot(normal_l, -L_norm));      // Cosine at light sample

    // Compute the geometry term
    float G = (cos_theta_x * cos_theta_y) / dist2;
    float pdf_l = 1.0 / (area_l * g_EmissiveTriangles[0].triCount);
    float3 emission_l =  sampleLight.emission;


    //Shadow ray
    //____________________________________________________________
    RayDesc ray;
    ray.Origin = worldOrigin + s_bias * flatNormal; // Offset origin along the normal
    ray.Direction = L_norm;
    ray.TMin = s_bias;
    ray.TMax = length(dist) - s_bias;
    bool hit = true;
    // Initialize the ray payload
    ShadowHitInfo shadowPayload;
    shadowPayload.isHit = false;
    // Trace the ray
    TraceRay(SceneBVH, RAY_FLAG_NONE, 0xFF, 1, 0, 1, ray, shadowPayload);
    float visible = shadowPayload.isHit ? 0.0 : 1.0;
    //____________________________________________________________



    // Get the light BRDF
    //____________________________________________________________
    //We sample the BSDF for the selected importance light.
    float3 brdf_light = EvaluateBRDF_Combined(materials[materialID], normal, -L_norm, -payload.direction);
    // We also need to get the pdf for that sample
    float pdf_brdf_light = BRDF_PDF_Combined(materials[materialID], normal, -L_norm, -payload.direction);

    //Direct lighting:
    float3 direct  = emission_l * visible * brdf_light * dot(normal, L_norm) * G / pdf_l;
    // MIS Weight: Use G to convert from area space to solid-angle space. Fuck me, took forever to find this.
    float weight_light = pdf_l / (pdf_l + pdf_brdf_light * G);
    // Also adjust for the throughput
    direct  *= payload.colorAndDistance.xyz * weight_light;
    //____________________________________________________________


    //Get the sample BRDF:
    //____________________________________________________________
    //First, we sample the BSDF of the given material - payload.direction and payload.origin will be altered. We need the incoming ray direction later, so we buffer it.
    float3 outgoing = -payload.direction; //Outgoing is the direction to the camera
    float3 origin = payload.origin;
    //As materials might combine lobes, we have to select one:
    uint strategy = SelectSamplingStrategy(materials[materialID], outgoing, normal, payload.seed);
    SampleBRDF(strategy, materials[materialID], outgoing, normal,flatNormal, payload.direction, payload.origin, worldOrigin, payload.seed);
    float3 incoming = -payload.direction;

    //PDF
    float pdf_sample = BRDF_PDF(strategy, materials[materialID], normal, incoming, outgoing);
    //Now, we evaluate the BRDF for the new sample. Our incident vector is the former ray direction vector and our outgoing vector is the new ray direction.
    float3 brdf_sample = EvaluateBRDF(strategy, materials[materialID], normal, incoming, outgoing);
    //____________________________________________________________


    float3 emissive = float3(0.0f,0.0f,0.0f);
    //If we hit a emissive surface, calculate MIS weights and end the path
    if(length(materials[materialID].Ke) > 0.0f){
        // If its the first bounce, we didnt sample the lightsource, so the weight is 1.0
        if(payload.util.y == 0.0f){
            emissive = materials[materialID].Ke;
        }
        else{
            //Calculate the G term
            float3 L_emissive = worldOrigin - origin;
            float dist2_emissive = dot(L_emissive, L_emissive);
            float dist_emissive = sqrt(dist2_emissive);
            float3 L_emissive_norm = L_emissive / dist_emissive;

            // Cosine terms for shading point and emissive surface
            float cos_theta_shading = max(0.0f, dot(payload.hitNormal, L_emissive_norm)); // Cosine at shading point
            float cos_theta_emissive = max(0.0f, dot(normal, -L_emissive_norm)); // Cosine at emissive surface

            float G_emissive = (cos_theta_shading * cos_theta_emissive) / dist2_emissive;


            //Calculate the MIS weight
            float weight_emissive = payload.pdf * G_emissive  / (payload.pdf * G_emissive  + pdf_l);
            // Calculate the MIS-weighted illumation
            emissive = materials[materialID].Ke * payload.colorAndDistance.xyz * weight_emissive;
            payload.util.x = 1.1f;
        }

    }

    payload.emission += abs(direct) + abs(emissive);
    //payload.emission += materials[materialID].Ke * payload.colorAndDistance.xyz;

    //_________DEBUG___________
    //float cosTheta = dot(normal, outgoing);
    //float3 fresnel = SchlickFresnel_DEBUG(materials[materialID].Ks, cosTheta);
    //payload.emission = 1.0f/pdf_sample;
    //payload.emission = brdf_sample;
    //_________DEBUG___________

    //Adjust the throughput
    payload.colorAndDistance = float4(payload.colorAndDistance.xyz * brdf_sample * dot(normal, -incoming) / pdf_sample, RayTCurrent());

    //Save the pdf_sample in case the next ray hits a light source
    payload.pdf = pdf_sample;


    //Misc
    //____________________________________________________________
    payload.hitNormal = normal;
    payload.reflectiveness = 1.0f-materials[materialID].Pr_Pm_Ps_Pc.x;
    payload.currentOTW = instanceProps[InstanceID()].objectToWorld;
    payload.prevOTW = instanceProps[InstanceID()].prevObjectToWorld;
}




// #DXR Extra - Another ray type (unused for now)
[shader("closesthit")] void PlaneClosestHit(inout HitInfo payload,
                                                Attributes attrib) {
    payload.emission = float3(InstanceID(),1,1);

}