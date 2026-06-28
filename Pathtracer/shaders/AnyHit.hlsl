#include "Includes_v8.hlsli"

//====================================
//ANY-HIT ALPHA TEST
//====================================
[shader("anyhit")]
void AlphaTestAnyHit(inout TracePayload payload,
                     in BuiltInTriangleIntersectionAttributes attr)
{
#if DISABLE_ALPHA_TEST
    //TEMP: alpha testing off - accept every hit as opaque (never IgnoreHit).
    return;
#else
    uint instID = InstanceID();
    uint primID = FlatPrimID(instID, GeometryIndex(), PrimitiveIndex());

    uint matID = materialIDs[instanceProps[instID].materialBase + primID];
    const int texID = LoadAlbedoTexID(matID);

    //no albedo means fully opaque
    if (texID < 0)
        return;

    //interpolate UVs
    uint baseI = instanceProps[instID].indexBase;
    uint i0 = indices[baseI + 3u * primID + 0u];
    uint i1 = indices[baseI + 3u * primID + 1u];
    uint i2 = indices[baseI + 3u * primID + 2u];

    float2 uv0 = (float2)BTriVertex[i0].texCoord;
    float2 uv1 = (float2)BTriVertex[i1].texCoord;
    float2 uv2 = (float2)BTriVertex[i2].texCoord;

    float b0 = 1.0f - attr.barycentrics.x - attr.barycentrics.y;
    float2 uv = uv0 * b0 + uv1 * attr.barycentrics.x + uv2 * attr.barycentrics.y;

    Texture2D<float4> tex = ResourceDescriptorHeap[texID];
    float alpha = SampleMaterialTex(tex, uv * LoadAlbedoUVScale(matID), 0).a;

    //flip when the sample channel is transparency (1=transparent) instead of opacity.
    //Set by the loader heuristics or the editor override.
    if (LoadInvertAlpha(matID)) alpha = 1.0f - alpha;

    if (alpha < LoadAlphaThreshold(matID))
        IgnoreHit();
#endif
}
