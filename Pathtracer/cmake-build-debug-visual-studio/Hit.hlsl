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
    float3 lightPos = float3(3, 2, -3);
    float3 toLight = lightPos - worldOrigin;
    float3 centerLightDir = normalize(toLight);
    float distanceToLight = length(toLight);

    // Inverse Square Law for Attenuation
    float attenuation = 1.0 / (distanceToLight * distanceToLight);


    //Shadow ray
    RayDesc ray;
    ray.Origin = worldOrigin + s_bias * flatNormal; // Offset origin along the normal
    ray.Direction = centerLightDir;
    ray.TMin = s_bias;
    ray.TMax = length(toLight) - s_bias;
    bool hit = true;
    // Initialize the ray payload
    ShadowHitInfo shadowPayload;
    shadowPayload.isHit = false;
    // Trace the ray
    TraceRay(SceneBVH, RAY_FLAG_NONE, 0xFF, 1, 0, 1, ray, shadowPayload);
    float factor = shadowPayload.isHit ? 0.0 : 1.0;

    float pdf;
    bool recieveDir;
    float3 brdf = evaluateBRDF(materials[materialID], normalize(WorldRayDirection()), normal,flatNormal, normalize(centerLightDir), payload.direction, payload.origin, worldOrigin, pdf, payload.seed, recieveDir);


    payload.colorAndDistance = float4(payload.colorAndDistance.xyz * materials[materialID].Kd, RayTCurrent());
    //Direct lighting: (Later, take a random sample from all available point lights

    //Reduce fireflies:
    float3 direct = float3(0,0,0);

    if(recieveDir){
        if(payload.util.y != 0)
            direct  = float3(10,10,10) * payload.colorAndDistance.xyz * attenuation * max(0.f, dot(normal, centerLightDir)) * factor * brdf * materials[materialID].Pr_Pm_Ps_Pc.x * materials[materialID].Pr_Pm_Ps_Pc.x;
        else
            direct  = float3(10,10,10) * payload.colorAndDistance.xyz * attenuation * max(0.f, dot(normal, centerLightDir)) * factor * brdf;
    }

    float3 emissive = materials[materialID].Ke * payload.colorAndDistance.xyz;
    payload.emission += direct + emissive;
    payload.hitNormal = normal;
    payload.reflectiveness = 1.0f-materials[materialID].Pr_Pm_Ps_Pc.x;
    payload.currentOTW = instanceProps[InstanceID()].objectToWorld;
    payload.prevOTW = instanceProps[InstanceID()].prevObjectToWorld;
    //payload.colorAndDistance = float4(payload.colorAndDistance.xyz / pdf, RayTCurrent());
}




// #DXR Extra - Another ray type (unused for now)
[shader("closesthit")] void PlaneClosestHit(inout HitInfo payload,
                                                Attributes attrib) {
    payload.emission = float3(InstanceID(),1,1);

}