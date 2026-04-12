#ifndef SURFACE_VERTEX_V8_HLSLI
#define SURFACE_VERTEX_V8_HLSLI

// =====================================================================================================================
// SurfaceVertex: clean wrapper for reconnection functions
// =====================================================================================================================

struct SurfaceVertex {
    float3 x;           // world position
    float3 n_s;         // shading normal (world)
    float3 o;           // outgoing/view direction (world, normalized)
    float3 Kd;          // albedo
    float  Pr;          // roughness
    float  Pm;          // metallic
    float  etai;        // IOR (incident side)
    float  etat;        // IOR (transmitted side)
    uint   matID;       // material ID
    float2 uv;          // texture coordinates
};

// Lightweight position-only reconstruction (~6 loads + 20 ALU)
// Still needed for Jacobian computation on GI reservoir x2 vertices
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

#endif // SURFACE_VERTEX_V8_HLSLI
