#include "Structures_misc.hlsli"

StructuredBuffer<STriVertex> BTriVertex      : register(t2);
StructuredBuffer<int>        indices         : register(t1);
RaytracingAccelerationStructure SceneBVH      : register(t0);
StructuredBuffer<InstanceProperties> instanceProps : register(t3);
StructuredBuffer<uint>       materialIDs     : register(t4);
StructuredBuffer<Material>   materials       : register(t5);
StructuredBuffer<LightTriangle> g_EmissiveTriangles : register(t6);

[shader("closesthit")]
void ClosestHit(inout HitInfo payload, Attributes attrib)
{
    uint   primIdx     = PrimitiveIndex();
    uint   baseIdx     = 3 * primIdx;
    int    i0          = indices[baseIdx + 0];
    int    i1          = indices[baseIdx + 1];
    int    i2          = indices[baseIdx + 2];

    // Cache all triangle vertices & normals in one go
    STriVertex v0     = BTriVertex[i0];
    STriVertex v1     = BTriVertex[i1];
    STriVertex v2     = BTriVertex[i2];

    float3 p0         = v0.vertex;
    float3 p1         = v1.vertex;
    float3 p2         = v2.vertex;

    float3 n0         = v0.normal.xyz;
    float3 n1         = v1.normal.xyz;
    float3 n2         = v2.normal.xyz;

    // Material lookup: the original code did `vertId + normal.w`, but
    // w is stored per-vertex so we only need it once. Take it from v0.
    uint materialID   = materialIDs[baseIdx + v0.normal.w];

    // Compute hit position in world‐space
    float tHit         = RayTCurrent();
    float3 rayOrig     = WorldRayOrigin();
    float3 rayDir      = WorldRayDirection();
    float3 hitPosWorld = rayOrig + tHit * rayDir;

    // Barycentrics
    float3 b = float3(1 - attrib.bary.x - attrib.bary.y,
                      attrib.bary.x,
                      attrib.bary.y);

    // Precompute edges & flat normal
    float3 e1          = p1 - p0;
    float3 e2          = p2 - p0;
    float3 flatN       = normalize(cross(e1, e2));
    float  triArea     = 0.5 * length(cross(e1, e2));

    // Smooth‐shading normal (fall back per-vertex if zero)
    float3 smoothN = n0 * b.x + n1 * b.y + n2 * b.z;
    if (length(smoothN) < 1e-4)
    {
        // if all vertex normals zero, just use flat
        smoothN = flatN;
    }
    else
    {
        smoothN = normalize(smoothN);
    }

    // Transform to world‐space only once
    float3 worldN = normalize(
        mul(instanceProps[InstanceID()].objectToWorldNormal, float4(smoothN, 0)).xyz
    );

    // Fill payload
    payload.objID      = InstanceID();
    payload.area       = triArea;
    payload.hitNormal  = worldN;
    payload.materialID = materialID;
    payload.hitPosition= hitPosWorld;
}
