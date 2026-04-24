#ifndef SURFACE_VERTEX_V8_HLSLI
#define SURFACE_VERTEX_V8_HLSLI

//====================================
//SURFACE VERTEX
//====================================
//wrapper for reconnection functions

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

//====================================
//POSITION RECONSTRUCTION
//====================================
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

//====================================
//VERTEX BUILDERS
//====================================
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

    //only flip IOR for transmissive backfaces, opaque backface has no interior
    const float matNi = LoadNi(v.matID);
    const float Kd_w  = LoadKd_w(v.matID);
    const bool transmissive = (matNi > 1.0f + EPSILON) && (Kd_w < 1.0f - EPSILON);
    const bool flipIOR = h.backface && transmissive;
    v.etai = flipIOR ? matNi : 1.0f;
    v.etat = flipIOR ? 1.0f  : matNi;
    return v;
}

inline SurfaceVertex BuildVertexLight(
    uint instID, uint primID, float2 bary,
    float3 n1s_world, float2 uv, float3 viewOrigin)
{
    SurfaceVertex v;

    //inline tri fetch so geometric normal falls out, matches EvalSurfaceState without second loads
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

    //same backface test as EvalSurfaceState, only sign of dot matters
    const float3 geoNormW = ObjectToWorldNrm(instID, geoNormOb);
    const bool   backface = dot(v.o, geoNormW) < 0.0f;

    //opaque backface must NOT flip IOR
    const float matNi = LoadNi(v.matID);
    const float Kd_w  = LoadKd_w(v.matID);
    const bool transmissive = (matNi > 1.0f + EPSILON) && (Kd_w < 1.0f - EPSILON);
    const bool flipIOR = backface && transmissive;
    v.etai = flipIOR ? matNi : 1.0f;
    v.etat = flipIOR ? 1.0f  : matNi;
    return v;
}

#endif
