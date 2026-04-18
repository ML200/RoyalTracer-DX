#ifndef SURFACE_VERTEX_V8_HLSLI
#define SURFACE_VERTEX_V8_HLSLI

// SurfaceVertex: clean wrapper for reconnection functions

struct SurfaceVertex {
    float3 x;
    float3 n_s;
    float3 o;
    float3 Kd;
    float  Pr;
    float  Pm;
    float  etai;
    float  etat;
    uint   matID;
    float2 uv;
};

inline float3 ReconstructPosition(uint instID, uint primID, float2 bary)
{
    const uint baseI = instanceProps[instID].indexBase;
    const uint i0 = indices[baseI + 3u * primID + 0u];
    const uint i1 = indices[baseI + 3u * primID + 1u];
    const uint i2 = indices[baseI + 3u * primID + 2u];
    const float3 p0 = BTriVertex[i0].vertex;
    const float3 p1 = BTriVertex[i1].vertex;
    const float3 p2 = BTriVertex[i2].vertex;
    const float b0 = 1.0f - bary.x - bary.y;
    float3 pLocal = p0 * b0 + p1 * bary.x + p2 * bary.y;
    return mul(instanceProps[instID].objectToWorld, float4(pLocal, 1.0f)).xyz;
}

inline SurfaceVertex BuildVertex(uint instID, uint primID, float2 bary, float3 viewOrigin)
{
    SurfaceVertex v;
    HitInfo h = EvalSurfaceState(instID, primID, bary, viewOrigin, 0);
    v.x     = h.hitPos;
    v.n_s   = h.hitNormal;
    v.o     = normalize(viewOrigin - h.hitPos);
    v.matID = GetMatIDFast(instID, primID);
    v.uv    = h.uv;
    RefetchMaterial(v.matID, h.uv, v.Kd, v.Pr, v.Pm);

    // IOR pair mirrors raygen: entering → (1, matNi), exiting → (matNi, 1).
    // Pass-through (matNi ≈ 1) yields (1, 1) and disables medium logic naturally.
    const float matNi = materials[v.matID].Ni;
    v.etai = h.backface ? matNi : 1.0f;
    v.etat = h.backface ? 1.0f  : matNi;
    return v;
}

inline SurfaceVertex BuildVertexLight(
    uint instID, uint primID, float2 bary,
    float3 n1s_world, float2 uv, float3 viewOrigin)
{
    SurfaceVertex v;

    // Inline the triangle-vertex fetch so the geometric normal falls out
    // for free — backface detection then matches EvalSurfaceState without
    // a second round of index/vertex loads.
    const uint baseI = instanceProps[instID].indexBase;
    const uint i0 = indices[baseI + 3u * primID + 0u];
    const uint i1 = indices[baseI + 3u * primID + 1u];
    const uint i2 = indices[baseI + 3u * primID + 2u];
    const float3 p0 = BTriVertex[i0].vertex;
    const float3 p1 = BTriVertex[i1].vertex;
    const float3 p2 = BTriVertex[i2].vertex;
    const float  b0 = 1.0f - bary.x - bary.y;
    const float3 pLocal    = p0 * b0 + p1 * bary.x + p2 * bary.y;
    const float3 geoNormOb = cross(p1 - p0, p2 - p0);

    v.x     = mul(instanceProps[instID].objectToWorld, float4(pLocal, 1.0f)).xyz;
    v.n_s   = n1s_world;
    v.o     = normalize(viewOrigin - v.x);
    v.matID = GetMatIDFast(instID, primID);
    v.uv    = uv;
    RefetchMaterial(v.matID, uv, v.Kd, v.Pr, v.Pm);

    // Same backface test as EvalSurfaceState: view direction (hit→camera) vs
    // geometric normal. length of geoNormOb doesn't matter, only sign of dot.
    const float3 geoNormW = ObjectToWorldNrm(instID, geoNormOb);
    const bool   backface = dot(v.o, geoNormW) < 0.0f;

    const float matNi = materials[v.matID].Ni;
    v.etai = backface ? matNi : 1.0f;
    v.etat = backface ? 1.0f  : matNi;
    return v;
}

#endif // SURFACE_VERTEX_V8_HLSLI
