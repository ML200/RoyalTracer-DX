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
//VERTEX BUILDERS
//====================================
//ReconstructPosition removed - all post-raygen passes recover x1 from the
//stored hitT via ReconstructPositionFromHitT (Camera_Ray_v8.hlsli).
//Assemble a SurfaceVertex purely from the baked G-buffer - no per-triangle
//data, no texture sample, no instanceProps. Works identically for scene meshes
//and procedural planet terrain (terrain just has no per-triangle data to miss).
//'x1' is the world hit position, reconstructed by the caller via
//ReconstructPositionFromHitT (that helper lives later in the include chain).
inline SurfaceVertex BuildVertex(RWByteAddressBuffer sampleBuf, uint pixelIdx,
                                 float3 x1, float3 viewOrigin)
{
    SurfaceVertex v;
    v.x     = x1;
    v.n_s   = load_n1_s(sampleBuf, pixelIdx);
    v.o     = normalize(viewOrigin - v.x);
    v.matID = load_matID(sampleBuf, pixelIdx);
    v.Kd    = load_kd(sampleBuf, pixelIdx);
    load_prpm(sampleBuf, pixelIdx, v.Pr, v.Pm);
    v.uv    = float2(0.0f, 0.0f);   // uv no longer stored - unused downstream

    //only flip IOR for transmissive backfaces, opaque backface has no interior
    const float matNi = LoadNi(v.matID);
    const float Kd_w  = LoadKd_w(v.matID);
    const bool transmissive = (matNi > 1.0f + EPSILON) && (Kd_w < 1.0f - EPSILON);
    const bool flipIOR = load_backface(sampleBuf, pixelIdx) && transmissive;
    v.etai = flipIOR ? matNi : 1.0f;
    v.etat = flipIOR ? 1.0f  : matNi;
    return v;
}

//BuildVertexLight removed - post-raygen passes no longer reconstruct geometry
//from (instID, primID, bary); BuildVertex assembles purely from baked data.

#endif
