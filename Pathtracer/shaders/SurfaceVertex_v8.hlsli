#ifndef SURFACE_VERTEX_V8_HLSLI
#define SURFACE_VERTEX_V8_HLSLI

// =====================================================================================================================
// SurfaceVertex: clean wrapper for reconnection functions
// =====================================================================================================================

struct SurfaceVertex {
    float3 x;           // world position
    float3 n_s;         // shading normal (world)
    float3 n_g;         // geometric normal (world)
    float3 o;           // outgoing/view direction (world, normalized)
    float3 Kd;          // albedo (refetched from texture)
    float  Pr;          // roughness
    float  Pm;          // metallic
    float  etai;        // IOR (incident side)
    float  etat;        // IOR (transmitted side)
    uint   matID;       // material ID
    float2 uv;          // texture coordinates
};

// Lightweight position-only reconstruction (~6 loads + 20 ALU)
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

// Full reconstruction via EvalSurfaceState + RefetchMaterial (~20 scattered loads)
inline SurfaceVertex BuildVertex(uint instID, uint primID, float2 bary, float3 viewOrigin)
{
    SurfaceVertex v;
    HitInfo h = EvalSurfaceState(instID, primID, bary, viewOrigin, 0);
    v.x     = h.hitPos;
    v.n_s   = h.hitNormal;
    v.n_g   = h.hitGNormal;
    v.o     = normalize(viewOrigin - h.hitPos);
    v.matID = GetMatIDFast(instID, primID);
    v.uv    = h.uv;
    RefetchMaterial(v.matID, h.uv, v.Kd, v.Pr, v.Pm);
    v.etai  = 1.0f;
    v.etat  = 1.0f;
    return v;
}

// Lightweight reconstruction using cached G-buffer data (~8 loads vs 20)
// Skips vertex normal/UV loads by using pre-cached values from the G-buffer
inline SurfaceVertex BuildVertexLight(
    uint instID, uint primID, float2 bary,
    float3 n1s_world, float3 n1g_world, float2 uv,
    float etai, float etat, float3 viewOrigin)
{
    SurfaceVertex v;
    v.x     = ReconstructPosition(instID, primID, bary);
    v.n_s   = n1s_world;
    v.n_g   = n1g_world;
    v.o     = normalize(viewOrigin - v.x);
    v.matID = GetMatIDFast(instID, primID);
    v.uv    = uv;
    RefetchMaterial(v.matID, uv, v.Kd, v.Pr, v.Pm);
    v.etai  = etai;
    v.etat  = etat;
    return v;
}

#endif // SURFACE_VERTEX_V8_HLSLI
