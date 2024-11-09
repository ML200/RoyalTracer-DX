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
    normal = normalize(mul(instanceProps[InstanceID()].objectToWorldNormal, float4(normal, 0.f)).xyz);
    flatNormal = normalize(mul(instanceProps[InstanceID()].objectToWorldNormal, float4(flatNormal, 0.f)).xyz);
    if (dot(payload.direction, normal) > 0.0f) {
        normal = -normal; // Flip the normal if hitting from behind
    }
    if (dot(payload.direction, flatNormal) > 0.0f) {
        flatNormal = -flatNormal; // Flip the normal if hitting from behind
    }


    //_____________________________________________________________________________________________________________________________________________

    // # DXR Extra - Simple Lighting
    float3 worldOrigin = WorldRayOrigin() + RayTCurrent() * WorldRayDirection();

    // Get a random light source based on the weights
    int random = int(RandomFloatLCG(payload.seed.x) * g_EmissiveTriangles[0].triCount);
    LightTriangle sampleLight = g_EmissiveTriangles[random];

    //Calculate the current world coordinates of the given triangle
    //Get the conversion matrix:
    float4x4 conversionMatrix = instanceProps[sampleLight.instanceID].objectToWorld;
    float3 x_v = mul(conversionMatrix, float4(sampleLight.x, 1.f)).xyz;
    float3 y_v = mul(conversionMatrix, float4(sampleLight.y, 1.f)).xyz;
    float3 z_v = mul(conversionMatrix, float4(sampleLight.z, 1.f)).xyz;

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

    float pdf;
    bool recieveDir;
    float3 brdf = evaluateBRDF(materials[materialID], normalize(WorldRayDirection()), normal,flatNormal, L_norm, payload.direction, payload.origin, worldOrigin, pdf, payload.seed, recieveDir);

    //Direct lighting:
    float3 direct = float3(0,0,0);

    if(recieveDir){
        if(payload.util.y != 0)
            direct  = g_EmissiveTriangles[0].triCount* emission_l * payload.colorAndDistance.xyz * max(0.f, dot(normal, L_norm)) * visible * brdf * G  * materials[materialID].Kd / pdf;
        else
            direct  = g_EmissiveTriangles[0].triCount* emission_l * payload.colorAndDistance.xyz * max(0.f, dot(normal, L_norm)) * visible * brdf * G * materials[materialID].Kd / pdf;
    }

    //_____________________________________________________________________________________________________________________________________________

    float3 emissive = materials[materialID].Ke * payload.colorAndDistance.xyz;
    payload.emission += (direct + emissive) / 2.0f;

    payload.colorAndDistance = float4(payload.colorAndDistance.xyz * materials[materialID].Kd, RayTCurrent());
    //payload.colorAndDistance = float4(payload.colorAndDistance.xyz / pdf, RayTCurrent());
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