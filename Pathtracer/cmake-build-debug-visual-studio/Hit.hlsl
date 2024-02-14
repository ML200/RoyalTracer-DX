#include "Common.hlsl"

// #DXR Extra - Another ray type
struct ShadowHitInfo {
  bool isHit;
};

struct InstanceProperties
{
  float4x4 objectToWorld;
  // # DXR Extra - Simple Lighting
  float4x4 objectToWorldNormal;
};

struct Material
{
     float4 Kd;
     float3 Ks;
     float3 Ke;
     float4 Pr_Pm_Ps_Pc;
     float2 aniso_anisor;
};


struct STriVertex {
  float3 vertex;
  float3 normal;
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
    uint vertId = 3 * PrimitiveIndex();
    uint materialID = materialIDs[vertId];

    // Modulate the color by the light's influence
    float3 hitColor = materials[materialID].Kd.xyz;
    float3 barycentrics = float3(1.f - attrib.bary.x - attrib.bary.y, attrib.bary.x, attrib.bary.y);


    // Calculate the position of the intersection point
    float3 hitPosition = BTriVertex[indices[vertId]].vertex * barycentrics.x +
           BTriVertex[indices[vertId + 1]].vertex * barycentrics.y +
           BTriVertex[indices[vertId + 2]].vertex * barycentrics.z;


    //_____________________________________________________________________________________________________________________________________________
    //Determine the impact normal. Apply interpolation if necessary.
    // Determine if any vertex of the triangle requests flat shading
    bool isFlatShading = all(BTriVertex[indices[vertId]].normal == float3(0, 0, 0)) ||
                         all(BTriVertex[indices[vertId + 1]].normal == float3(0, 0, 0)) ||
                         all(BTriVertex[indices[vertId + 2]].normal == float3(0, 0, 0));

    // Initialize the normal to zero
    float3 normal = float3(0, 0, 0);

    // Accumulate the weighted normals for smooth shading
    if (!isFlatShading) {
        normal +=
                (all(BTriVertex[indices[vertId]].normal != float3(0, 0, 0)) ? BTriVertex[indices[vertId]].normal * barycentrics.x : float3(0, 0, 0)) +
                (all(BTriVertex[indices[vertId + 1]].normal != float3(0, 0, 0)) ? BTriVertex[indices[vertId + 1]].normal * barycentrics.y : float3(0, 0, 0)) +
                (all(BTriVertex[indices[vertId + 2]].normal != float3(0, 0, 0)) ? BTriVertex[indices[vertId + 2]].normal * barycentrics.z : float3(0, 0, 0));

        // Normalize the normal if it's not near-zero
        if (length(normal) > 0.0001) {
            normal = normalize(normal);
        } else {
            // Fallback to flat shading if the smooth normal is near-zero
            float3 e1 = BTriVertex[indices[vertId + 1]].vertex - BTriVertex[indices[vertId]].vertex;
            float3 e2 = BTriVertex[indices[vertId + 2]].vertex - BTriVertex[indices[vertId]].vertex;
            normal = normalize(cross(e1, e2));
        }
    } else {
        // Flat shading
        float3 e1 = BTriVertex[indices[vertId + 1]].vertex - BTriVertex[indices[vertId]].vertex;
        float3 e2 = BTriVertex[indices[vertId + 2]].vertex - BTriVertex[indices[vertId]].vertex;
        normal = normalize(cross(e1, e2));
    }

// Transform normal to world space and adjust for ray direction
    normal = mul(instanceProps[InstanceID()].objectToWorldNormal, float4(normal, 0.f)).xyz;
    if (dot(payload.direction, normal) > 0.0f) {
        normal = -normal; // Flip the normal if hitting from behind
    }
    //_____________________________________________________________________________________________________________________________________________

    // # DXR Extra - Simple Lighting
    float3 worldOrigin = WorldRayOrigin() + RayTCurrent() * WorldRayDirection();
    float3 lightPos = float3(5, 5, -5);
    float3 toLight = lightPos - worldOrigin;
    float3 centerLightDir = normalize(toLight);
    float distanceToLight = length(toLight);

    // Inverse Square Law for Attenuation
    float attenuation = 1.0 / (distanceToLight * distanceToLight);

    float nDotL = max(0.f, dot(normal, centerLightDir));
    nDotL *= attenuation;


    //Shadow ray
    RayDesc ray;
    ray.Origin = worldOrigin;
    ray.Direction = centerLightDir;
    ray.TMin = 0.0001;
    ray.TMax = length(toLight) - 0.001f;
    bool hit = true;
    // Initialize the ray payload
    ShadowHitInfo shadowPayload;
    shadowPayload.isHit = false;
    // Trace the ray
    TraceRay(SceneBVH, RAY_FLAG_NONE, 0xFF, 1, 0, 1, ray, shadowPayload);
    float factor = shadowPayload.isHit ? 0.0 : 1.0;


    //Now, adjust the payload to the new origin and direction:
    payload.direction = RandomUnitVectorInHemisphere(normal,payload.seed);
    payload.origin = worldOrigin;

    payload.colorAndDistance = float4(payload.colorAndDistance.xyz *hitColor, RayTCurrent());
    payload.emission += nDotL * factor *float3(120,120,120) * payload.colorAndDistance.xyz * payload.pdf; //Hardcoded intensity
    payload.pdf = max(dot(normal, payload.direction), 0.0);
}




// #DXR Extra - Another ray type (unused for now)
[shader("closesthit")] void PlaneClosestHit(inout HitInfo payload,
                                                Attributes attrib) {
    payload.util = 1;

}