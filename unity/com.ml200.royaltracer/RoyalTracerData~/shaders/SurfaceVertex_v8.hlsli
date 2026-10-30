#pragma once

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
};

inline SurfaceVertex MakeVertex(float3 x1, float3 n_s, float3 viewOrigin,
                                uint matID, float3 Kd, float Pr, float Pm,
                                bool backface)
{
    SurfaceVertex v;
    v.x     = x1;
    v.n_s   = n_s;
    v.o     = normalize(viewOrigin - v.x);
    v.matID = matID;
    v.Kd    = Kd;
    v.Pr    = Pr;
    v.Pm    = Pm;

    const float matNi = LoadNi(matID);
    const float Kd_w  = LoadKd_w(matID);
    const bool transmissive = (matNi > 1.0f + EPSILON) && (Kd_w < 1.0f - EPSILON);
    const bool flipIOR = backface && transmissive && !LoadIsThinGlass(matID);
    v.etai = flipIOR ? matNi : 1.0f;
    v.etat = flipIOR ? 1.0f  : matNi;
    return v;
}

inline SurfaceVertex BuildVertex(RWByteAddressBuffer sampleBuf, uint pixelIdx,
                                 float3 x1, float3 viewOrigin)
{
    float pr, pm;
    load_prpm(sampleBuf, pixelIdx, pr, pm);
    return MakeVertex(x1,
                      load_n1_s(sampleBuf, pixelIdx),
                      viewOrigin,
                      load_matID(sampleBuf, pixelIdx),
                      load_kd(sampleBuf, pixelIdx),
                      pr, pm,
                      load_backface(sampleBuf, pixelIdx));
}

