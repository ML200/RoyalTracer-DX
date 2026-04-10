#include "Includes_raygen_v8.hlsli"

[shader("anyhit")]
void AlphaTestAnyHit(inout TracePayload payload,
                     in BuiltInTriangleIntersectionAttributes attr)
{
    uint instID = InstanceID();
    uint primID = FlatPrimID(instID, GeometryIndex(), PrimitiveIndex());

    uint matID = materialIDs[instanceProps[instID].materialBase + primID];
    Material mat = materials[matID];

    // No albedo texture → fully opaque, accept hit
    if (mat.albedoTexID < 0)
        return;

    // Interpolate UVs
    uint baseI = instanceProps[instID].indexBase;
    uint i0 = indices[baseI + 3u * primID + 0u];
    uint i1 = indices[baseI + 3u * primID + 1u];
    uint i2 = indices[baseI + 3u * primID + 2u];

    float2 uv0 = (float2)BTriVertex[i0].texCoord;
    float2 uv1 = (float2)BTriVertex[i1].texCoord;
    float2 uv2 = (float2)BTriVertex[i2].texCoord;

    float b0 = 1.0f - attr.barycentrics.x - attr.barycentrics.y;
    float2 uv = uv0 * b0 + uv1 * attr.barycentrics.x + uv2 * attr.barycentrics.y;

    Texture2D<float4> tex = ResourceDescriptorHeap[mat.albedoTexID];
    float alpha = tex.SampleLevel(g_sampler, uv * mat.albedoUVScale, 0).a;

    if (alpha < mat.alphaThreshold)
        IgnoreHit();
}