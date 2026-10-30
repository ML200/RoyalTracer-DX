#pragma once
// The virtual geometry view: the primary hit coloured by cluster, hierarchy level, triangle, projected triangle size,
// instance or residency, in place of the image (GEOMETRY_VIEW_*).

#define GEOMETRY_VIEW ((dbg_dlssLayer >> GEOMETRY_VIEW_SHIFT) & GEOMETRY_VIEW_MASK)

float3 GeometryViewHash(uint v)
{
    v ^= v >> 16; v *= 0x7feb352du; v ^= v >> 15; v *= 0x846ca68bu; v ^= v >> 16;
    return float3(v & 0xFFu, (v >> 8) & 0xFFu, (v >> 16) & 0xFFu) * (0.8f / 255.0f) + 0.2f;
}

// Blue, green, yellow, red at 0, 1/3, 2/3 and 1.
float3 GeometryViewRamp(float t)
{
    const float3 c0 = float3(0.10f, 0.30f, 0.95f), c1 = float3(0.15f, 0.85f, 0.25f);
    const float3 c2 = float3(0.95f, 0.85f, 0.10f), c3 = float3(0.95f, 0.15f, 0.10f);
    t = saturate(t) * 3.0f;
    return t < 1.0f ? lerp(c0, c1, t) : (t < 2.0f ? lerp(c1, c2, t - 1.0f) : lerp(c2, c3, t - 2.0f));
}

// The triangle's size in pixels where the ray met it: the side of a square of its area over the pixel footprint.
float GeometryViewTrianglePixels(uint instID, uint flatPrim, float3x4 hitToWorld, float hitT)
{
    const uint base = instanceProps[instID].indexBase + 3u * flatPrim;
    const float3x4 M = IS_SET_INSTANCE(instID) ? hitToWorld : instanceProps[instID].objectToWorld;
    const float3 p0 = mul(M, float4(BTriVertex[indices[base + 0u]].vertex, 1.0f));
    const float3 p1 = mul(M, float4(BTriVertex[indices[base + 1u]].vertex, 1.0f));
    const float3 p2 = mul(M, float4(BTriVertex[indices[base + 2u]].vertex, 1.0f));
    const float area = 0.5f * length(cross(p1 - p0, p2 - p0));
    return sqrt(area) / max(PixelConeAngle() * hitT, 1e-9f);
}

// instanceIndex, hitToWorld: the hit's TLAS instance and its transform (an instance set's instances share instID).
float3 GeometryViewColor(uint mode, uint instID, uint instanceIndex, uint clusterID, uint primIdx, uint flatPrim,
                         float3x4 hitToWorld, float hitT, float3 geometricNormal, float3 rayDir)
{
    const bool clustered = RDN_CLUSTERS && clusterID != CLUSTER_ID_INVALID;
    const uint info = clustered ? clusterRecords[clusterID].info : 0u;
    const uint kind = info >> CLUSTER_INFO_KIND_SHIFT, level = info & CLUSTER_INFO_LEVEL_MASK;
    const bool hierarchy = kind >= CLUSTER_KIND_RESIDENT;
    float3 c = float3(0.35f, 0.35f, 0.35f);
    switch (mode)
    {
    case GEOMETRY_VIEW_CLUSTERS:
        if (clustered) c = GeometryViewHash(clusterID);
        break;
    case GEOMETRY_VIEW_LOD_LEVEL:
        // Full detail blue, each level up towards red; outside any hierarchy grey, kept out of one magenta.
        if (hierarchy) c = GeometryViewRamp(level / 6.0f);
        else if (kind == CLUSTER_KIND_EXCLUDED) c = float3(0.75f, 0.25f, 0.75f);
        break;
    case GEOMETRY_VIEW_TRIANGLES:
        c = GeometryViewHash((clustered ? clusterID : instID) * 0x9E3779B1u + primIdx);
        break;
    case GEOMETRY_VIEW_TRIANGLE_SIZE:
        // Blue at a quarter pixel, green at one, yellow at four, red at sixteen and more.
        c = GeometryViewRamp((log2(max(GeometryViewTrianglePixels(instID, flatPrim, hitToWorld, hitT), 1e-6f)) + 2.0f) / 6.0f);
        break;
    case GEOMETRY_VIEW_INSTANCES:
        c = GeometryViewHash((IS_SET_INSTANCE(instID) ? instanceIndex : instID) * 0x85EBCA6Bu + 0x165667B1u);
        break;
    case GEOMETRY_VIEW_RESIDENCY:
        c = kind == CLUSTER_KIND_EXCLUDED ? float3(0.75f, 0.25f, 0.75f)  // kept out of the hierarchy
          : kind == CLUSTER_KIND_RESIDENT ? float3(0.20f, 0.45f, 0.95f)  // every level in memory
          : kind == CLUSTER_KIND_ROOT     ? float3(0.95f, 0.70f, 0.15f)  // root page
          : kind == CLUSTER_KIND_STREAMED ? float3(0.20f, 0.85f, 0.30f)  // streamed page
          : c;
        break;
    default:
        break;
    }
    return c * (0.35f + 0.65f * abs(dot(geometricNormal, rayDir)));
}
