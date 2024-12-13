#include "Common_v6.hlsl"

StructuredBuffer<STriVertex> BTriVertex : register(t2);
StructuredBuffer<int> indices : register(t1);
RaytracingAccelerationStructure SceneBVH : register(t0);
StructuredBuffer<InstanceProperties> instanceProps : register(t3);
StructuredBuffer<uint> materialIDs : register(t4);
StructuredBuffer<Material> materials : register(t5);
StructuredBuffer<LightTriangle> g_EmissiveTriangles : register(t6);


[shader("closesthit")] void ClosestHit(inout HitInfo payload, Attributes attrib) {
    // Get information about the surface hit
    float3 worldOrigin = WorldRayOrigin() + RayTCurrent() * WorldRayDirection();
    uint vertId = 3 * PrimitiveIndex();
    uint materialID = materialIDs[vertId+BTriVertex[indices[vertId]].normal.w];
    float3 barycentrics = float3(1.f - attrib.bary.x - attrib.bary.y, attrib.bary.x, attrib.bary.y);

    // Calculate the position of the intersection point
    float3 hitPosition = BTriVertex[indices[vertId]].vertex * barycentrics.x +
           BTriVertex[indices[vertId + 1]].vertex * barycentrics.y +
           BTriVertex[indices[vertId + 2]].vertex * barycentrics.z;

    //Determine the impact normal. Apply interpolation.
    float3 normal = float3(0, 0, 0);
    // Always calculate the flat shading normal
    float3 e1 = BTriVertex[indices[vertId + 1]].vertex - BTriVertex[indices[vertId]].vertex;
    float3 e2 = BTriVertex[indices[vertId + 2]].vertex - BTriVertex[indices[vertId]].vertex;
    float3 cross_a = cross(e1, e2);
    float area_l = abs(length(cross_a) * 0.5f);
    float3 flatNormal = normalize(cross_a);

    payload.area = area_l;

    // Smooth shading normal
    float3 smoothNormal = float3(0, 0, 0);

    // Check each vertex normal; accumulate if not zero, otherwise use flat normal
    for (int i = 0; i < 3; i++) {
        if (all(BTriVertex[indices[vertId + i]].normal.xyz != float3(0, 0, 0))) {
            smoothNormal += BTriVertex[indices[vertId + i]].normal.xyz * barycentrics[i];
        } else {
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

    normal = normalize(mul(instanceProps[InstanceID()].objectToWorldNormal, float4(normal, 0.f)).xyz);     // Transform normal to world space

    payload.hitNormal = normal;
    payload.materialID = materialID;
    payload.hitPosition = worldOrigin;
}

//Unused for now
[shader("closesthit")] void PlaneClosestHit(inout HitInfo payload, Attributes attrib) {
}