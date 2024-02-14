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

    // Normal world space
    float3 e1 = BTriVertex[indices[vertId + 1]].vertex - BTriVertex[indices[vertId + 0]].vertex;
    float3 e2 = BTriVertex[indices[vertId + 2]].vertex - BTriVertex[indices[vertId + 0]].vertex;
    float3 normal = normalize(cross(e2, e1));
    normal = mul(instanceProps[InstanceID()].objectToWorldNormal, float4(normal, 0.f)).xyz;

    if (dot(payload.direction, normal) > 0.0f) {
        normal = -normal; // Flip the normal if hitting from behind
    }


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

    payload.colorAndDistance = float4(payload.colorAndDistance.xyz * hitColor, RayTCurrent());
    payload.emission += nDotL * factor *float3(120,120,120) * payload.colorAndDistance.xyz * payload.pdf; //Hardcoded intensity
    payload.pdf = max(dot(normal, payload.direction), 0.0);

}




// #DXR Extra - Another ray type
[shader("closesthit")] void PlaneClosestHit(inout HitInfo payload,
                                                Attributes attrib) {
    payload.util = 1;
    // Modulate the color by the light's influence
    /*float3 hitColor = float3(0.0,0.95,0.0);
    float3 barycentrics = float3(1.f - attrib.bary.x - attrib.bary.y, attrib.bary.x, attrib.bary.y);

    uint vertId = 3 * PrimitiveIndex();

    // Calculate the position of the intersection point
    float3 hitPosition = BTriVertex[indices[vertId]].vertex * barycentrics.x +
                         BTriVertex[indices[vertId + 1]].vertex * barycentrics.y +
                         BTriVertex[indices[vertId + 2]].vertex * barycentrics.z;

    // Normal world space
    float3 e1 = BTriVertex[indices[vertId + 1]].vertex - BTriVertex[indices[vertId + 0]].vertex;
    float3 e2 = BTriVertex[indices[vertId + 2]].vertex - BTriVertex[indices[vertId + 0]].vertex;
    float3 normal = normalize(cross(e2, e1));
    normal = mul(instanceProps[InstanceID()].objectToWorldNormal, float4(normal, 0.f)).xyz;



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

    float pdf = max(dot(normal, payload.direction), 0.0);
    payload.colorAndDistance = float4(payload.colorAndDistance.xyz * hitColor, RayTCurrent());
    payload.emission += nDotL * factor *float3(8,8,8) * payload.colorAndDistance.xyz / pdf; //Hardcoded intensity*/
}