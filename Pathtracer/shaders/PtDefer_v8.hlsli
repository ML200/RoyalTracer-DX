#pragma once
// Deferred shading records for the cache-driven path tracer. The trace pass walks the path and
// leaves the first wide vertex (the one light sampling applies to) together with everything the
// light and material passes need to finish it. The planes live in PathStateLayout.h.
#include "SharcPath_v8.hlsli"
#include "SharcGuide_v8.hlsli"
#include "RestirLite_v8.hlsli"

// Path sampling flags (identical to the training pass so seeds and reuse stay bit-compatible).
#define PT_PS_DEPTH_SHIFT   0u
#define PT_PS_MIS_NONE      (1u << 7u)
#define PT_PS_DIFF_SHIFT    8u
#define PT_PS_SAMPLE_SHIFT  16u
#define PT_PS_GUIDE_SHIFT   20u
#define PT_PS_SSS           (1u << 24u)
#define PT_PS_LITE_VERTEX   (1u << 25u)
#define PT_PS_LITE_X2       (1u << 26u)
#define PT_PS_SPREAD        (1u << 29u)
#define PT_REGULARIZE_ROUGHNESS ((float)((guide_params >> GUIDE_PARAM_REGULARIZE_SHIFT) & 63u) * 0.01f)
uint PtPsInit(uint s)       { return (s << PT_PS_SAMPLE_SHIFT) | (1u << PT_PS_DEPTH_SHIFT) | PT_PS_MIS_NONE; }
uint PtPsDepth(uint ps)     { return (ps >> PT_PS_DEPTH_SHIFT) & 0x7Fu; }
uint PtPsDiffDepth(uint ps) { return (ps >> PT_PS_DIFF_SHIFT) & 0x7Fu; }
uint PtPsSample(uint ps)    { return (ps >> PT_PS_SAMPLE_SHIFT) & 0xFu; }
uint PtPsGuideDepth(uint ps){ return (ps >> PT_PS_GUIDE_SHIFT) & 0xFu; }
uint PtPsWith(uint ps, uint flag, bool on) { return on ? (ps | flag) : (ps & ~flag); }

// Blue-noise dimensions per vertex.
#define BN_DIMS_PER_VERTEX 16u
#define BN_DIM_STRATEGY    0u
#define BN_DIM_GUIDE_TEST  1u
#define BN_PAIR_COSINE     1u
#define BN_DIM_GUIDE_PICK  4u
#define BN_PAIR_GUIDE_CAP  3u
#define BN_PAIR_SUN        4u
float  PtBlue1(uint2 px, uint blueIndex, uint depth, uint dim)  { return BlueNoise (px, blueIndex, dim  + BN_DIMS_PER_VERTEX * (depth - 1u)); }
float2 PtBlue2(uint2 px, uint blueIndex, uint depth, uint pair) { return BlueNoise2(px, blueIndex, pair + (BN_DIMS_PER_VERTEX / 2u) * (depth - 1u)); }

uint PtSampleCount() { return max(PT_SAMPLE_COUNT, 1u); }
uint PtPathSeed(uint2 pixel, uint sample)
{
    return Hash32(initRandomData(pixel, uint2(8, 4), time, sample + 1u) ^ 0x9E3779B9u);
}
// The point on the picked light triangle and the sun sample are drawn from this stream.
#define PT_STREAM_LIGHT_POINT 0x4c505453u

// Radiance packed into one word; the scale keeps bright emitters inside the RGB9E5 range.
static const float PV_RAD_PACK = 1.0f / 1024.0f;
uint   PvPackRadiance(float3 L) { return PackRGB9E5(max(L, 0.0f) * PV_RAD_PACK); }
float3 PvUnpackRadiance(uint p) { return UnpackRGB9E5(p) / PV_RAD_PACK; }

uint dv_vert   (uint px) { return ps_plane(DV_VERT_PLANE,    DV_VERT_BYTES,    px); }
uint dv_scatter(uint px) { return ps_plane(DV_SCATTER_PLANE, DV_SCATTER_BYTES, px); }
uint dv_path   (uint px) { return ps_plane(DV_PATH_PLANE,    DV_PATH_BYTES,    px); }
uint dv_lite   (uint px) { return ps_plane(DV_LITE_PLANE,    DV_LITE_BYTES,    px); }
uint dv_end    (uint px) { return ps_plane(DV_END_PLANE,     DV_END_BYTES,     px); }
uint dv_light  (uint px) { return ps_plane(DV_LIGHT_PLANE,   DV_LIGHT_BYTES,   px); }

// ---------------------------------------------------------------------------------------------
// Vertex record (48 bytes): the deferred shading vertex. The sheen share is rebuilt from the
// other three lobes.
// ---------------------------------------------------------------------------------------------
#define DVF_BACKFACE     1u
#define DVF_FLIP_IOR     2u
#define DVF_PERFORM_NEE  4u
#define DVF_UNIT_IOR     8u   // subsurface exit: no interface, a white Lambertian vertex
#define DVF_GROUP_SHIFT  4u   // 2 bits: the picked lobe group (LOBE_GROUP_*); its probability is in the path record
#define DVF_REGULARIZE   64u  // the path had spread before this vertex: its roughness is regularized
#define DVF_BROAD_NEE    128u // the broad group takes the light sample on every pick that defers the vertex
#define DVF_LITE_NEE     256u // a reuse primary: its broad light sample feeds the reservoir, whatever the pick

struct DvVertex {
    float3    pos;
    float3    n;
    float3    dirIn;    // direction the path arrived from
    uint      matID;
    uint      instID;
    float3    Kd;
    float     Pr;
    float     Pm;
    SamplingP sp;
    float3    absorb;   // absorption through the entered medium
    uint      flags;
};

DvVertex DvLoadVertex(uint px)
{
    const uint a = dv_vert(px);
    const uint4 w0 = g_pathStateBuffer.Load4(a);
    const uint4 w1 = g_pathStateBuffer.Load4(a + 16u);
    const uint4 w2 = g_pathStateBuffer.Load4(a + 32u);
    DvVertex v;
    v.pos    = asfloat(w0.xyz);
    v.n      = UnpackNormal(w0.w);
    v.dirIn  = UnpackNormal(w1.x);
    v.matID  = w1.y;
    v.instID = w1.z;
    v.absorb = UnpackRGB9E5(w1.w);
    v.Kd     = UnpackRGB9E5(w2.x);
    UnpackFloat2x16(w2.y, v.Pr, v.Pm);
    UnpackFloat2x16(w2.z, v.sp.Pdiff, v.sp.Pspec);
    v.sp.Pcoat  = f16tof32(w2.w & 0xFFFFu);
    v.sp.Psheen = saturate(1.0f - v.sp.Pdiff - v.sp.Pspec - v.sp.Pcoat);
    v.flags  = w2.w >> 16u;
    return v;
}
void DvStoreVertex(uint px, DvVertex v)
{
    const uint a = dv_vert(px);
    g_pathStateBuffer.Store4(a,       uint4(asuint(v.pos), PackNormal(v.n)));
    g_pathStateBuffer.Store4(a + 16u, uint4(PackNormal(v.dirIn), v.matID, v.instID, PackRGB9E5(saturate(v.absorb))));
    g_pathStateBuffer.Store4(a + 32u, uint4(PackRGB9E5(v.Kd), PackFloat2x16(v.Pr, v.Pm),
        PackFloat2x16(v.sp.Pdiff, v.sp.Pspec), f32tof16(v.sp.Pcoat) | (v.flags << 16u)));
}
uint DvLoadVertexFlags(uint px) { return g_pathStateBuffer.Load(dv_vert(px) + 44u) >> 16u; }
// The deferred vertex is evaluated on the lobe group its sample picked, and so is the light sample
// of a picked group other than the broad one (DVF_BROAD_NEE).
uint DvLobeGroup(uint flags) { return (flags >> DVF_GROUP_SHIFT) & 3u; }
// Probability that a sample at the vertex defers it: every pick of a wide lobe does
// (SharcScatterHasSpread), a mirror-like GGX or coat pick bounces in place instead.
float DvDeferP(DvVertex v)
{
    float narrow = 0.0f;
    if (v.Pr < SMOOTH_SPECULAR_THRESHOLD) narrow += v.sp.Pspec;
    if (LoadPcr(v.matID) < SMOOTH_SPECULAR_THRESHOLD) narrow += v.sp.Pcoat;
    return max(1.0f - narrow, 1e-6f);
}
void DvLoadVertexSurface(uint px, out float3 pos, out float3 n)
{
    const uint4 w = g_pathStateBuffer.Load4(dv_vert(px));
    pos = asfloat(w.xyz);
    n   = UnpackNormal(w.w);
}

// Build the material context the shared BSDF helpers expect.
HitContext DvContext(DvVertex v)
{
    HitContext c;
    c.hitPos     = v.pos;
    c.hitNormal  = v.n;
    c.matID      = v.matID;
    c.instID     = v.instID;
    c.backface   = (v.flags & DVF_BACKFACE) != 0u;
    c.hitLocalKd = (half3)v.Kd;
    c.hitLocalPr = (half)v.Pr;
    c.hitLocalPm = (half)v.Pm;
    const bool  flip = (v.flags & DVF_FLIP_IOR) != 0u;
    const float ni   = (v.flags & DVF_UNIT_IOR) != 0u ? 1.0f : LoadNi(v.matID);
    c.iors           = (half2)(flip ? float2(ni, 1.0f) : float2(1.0f, ni));
    c.mediumMatID    = flip ? v.matID : MEDIUM_INVALID;
    c.absorptionTint = (half3)v.absorb;
    return c;
}

// ---------------------------------------------------------------------------------------------
// Scatter record (16 bytes): the continuation sampled at the deferred vertex, with the sampling
// density the trace pass already knows; the material pass supplies the value of the picked lobe
// group. info is the path's DV_* word (DvLoadInfo/DvStoreInfo), which the trace pass writes last.
// ---------------------------------------------------------------------------------------------
#define DV_KIND_NONE      0u  // light sampling only (budget reached, invalid ray)
#define DV_KIND_BSDF      1u  // BSDF-sampled continuation, value pending
#define DV_KIND_MASK      3u
#define DV_STRATEGY_SHIFT 2u
#define DV_END_SHIFT      4u
#define DV_END_NONE       0u
#define DV_END_EMITTER    1u  // path ended on an emitter right after the deferred scatter
#define DV_END_MISS       2u  // path escaped (anywhere after the deferred vertex)
#define DV_END_MASK       3u
#define DV_VALID          (1u << 6u)
#define DV_MIS_NONE       (1u << 7u)  // MIS state after the deferred scatter
#define DV_LITE_MISS      (1u << 8u)  // escape happened directly after a reuse primary
#define DV_LITE_X2        (1u << 9u)  // the reuse suffix parked a point; its radiance is in the lite record
#define DV_LITE_GEN       (1u << 10u) // the deferred vertex is a reuse primary

struct DvScatter {
    float3 dirOut;
    uint   info;       // the path's DV_* word, shared with DvLoadInfo/DvStoreInfo
    float  pdfTotal;   // sampling density: pick probability times the group density with guiding
    float  bsdfPdf;    // group density without guiding, the MIS partner of the light sample
};
uint  DvLoadInfo(uint px) { return g_pathStateBuffer.Load(dv_scatter(px) + 4u); }
void  DvStoreInfo(uint px, uint info) { g_pathStateBuffer.Store(dv_scatter(px) + 4u, info); }
float DvLoadScatterPdf(uint px) { return asfloat(g_pathStateBuffer.Load(dv_scatter(px) + 8u)); }
DvScatter DvLoadScatter(uint px)
{
    const uint4 w = g_pathStateBuffer.Load4(dv_scatter(px));
    DvScatter s;
    s.dirOut   = UnpackNormal(w.x);
    s.info     = w.y;
    s.pdfTotal = asfloat(w.z);
    s.bsdfPdf  = asfloat(w.w);
    return s;
}
void DvStoreScatter(uint px, DvScatter s)
{
    g_pathStateBuffer.Store4(dv_scatter(px), uint4(PackNormal(s.dirOut), s.info, asuint(s.pdfTotal), asuint(s.bsdfPdf)));
}

// ---------------------------------------------------------------------------------------------
// Path record (32 bytes): throughput arriving at the deferred vertex, the probability of the lobe
// pick there (the light sample is divided by it), the radiance gathered after the deferred scatter
// (relative to that scatter) and the sampling flags.
// ---------------------------------------------------------------------------------------------
struct DvPath {
    float3 T;
    float  pickP;
    float3 relL;
    uint   ps;
};
DvPath DvLoadPath(uint px)
{
    const uint a = dv_path(px);
    const uint4 w0 = g_pathStateBuffer.Load4(a);
    const uint4 w1 = g_pathStateBuffer.Load4(a + 16u);
    DvPath p;
    p.T = asfloat(w0.xyz);
    p.pickP = asfloat(w0.w);
    p.relL = asfloat(w1.xyz);
    p.ps = w1.w;
    return p;
}
// The trace pass writes the record in parts as each is known, so none stays live through the rest
// of the path: the pick probability while shading the deferred vertex, the throughput and flags
// when the raygen applies it, the radiance gathered after it at the end.
void DvStorePickP(uint px, float pickP) { g_pathStateBuffer.Store(dv_path(px) + 12u, asuint(pickP)); }
void DvStorePathHead(uint px, float3 T, uint ps)
{
    const uint a = dv_path(px);
    g_pathStateBuffer.Store3(a, asuint(T));
    g_pathStateBuffer.Store(a + 28u, ps);
}
void DvStorePathRelL(uint px, float3 relL) { g_pathStateBuffer.Store3(dv_path(px) + 16u, asuint(relL)); }
uint DvLoadPs(uint px) { return g_pathStateBuffer.Load(dv_path(px) + 28u); }
float3 DvLoadPathRelL(uint px) { return asfloat(g_pathStateBuffer.Load3(dv_path(px) + 16u)); }

// Reuse suffix radiance (4 bytes), packed like the reuse reservoir packs radiance.
static const float DV_LITE_L_SCALE = 64.0f;
void   DvStoreLiteL(uint px, float3 L) { g_pathStateBuffer.Store(dv_lite(px), PackRGB9E5(max(L, 0.0f) / DV_LITE_L_SCALE)); }
float3 DvLoadLiteL(uint px) { return UnpackRGB9E5(g_pathStateBuffer.Load(dv_lite(px))) * DV_LITE_L_SCALE; }

// ---------------------------------------------------------------------------------------------
// Terminal event (24 bytes): the emitter hit whose MIS weight is pending, or the escape whose
// reuse candidate is pending.
// ---------------------------------------------------------------------------------------------
struct DvEmitter { uint lightID; uint inst; float3 pos; float3 n; };
void DvStoreEmitter(uint px, DvEmitter e)
{
    const uint a = dv_end(px);
    g_pathStateBuffer.Store4(a, uint4(e.lightID, e.inst, asuint(e.pos.xy)));
    g_pathStateBuffer.Store2(a + 16u, uint2(asuint(e.pos.z), PackNormal(e.n)));
}
DvEmitter DvLoadEmitter(uint px)
{
    const uint a = dv_end(px);
    const uint4 w0 = g_pathStateBuffer.Load4(a);
    const uint2 w1 = g_pathStateBuffer.Load2(a + 16u);
    DvEmitter e;
    e.lightID = w0.x;
    e.inst = w0.y;
    e.pos = float3(asfloat(w0.zw), asfloat(w1.x));
    e.n = UnpackNormal(w1.y);
    return e;
}
struct DvMiss { float3 dir; float3 skySun; float candMis; };
void DvStoreMiss(uint px, DvMiss m)
{
    g_pathStateBuffer.Store3(dv_end(px), uint3(PackNormal(m.dir), PackRGB9E5(m.skySun), asuint(m.candMis)));
}
DvMiss DvLoadMiss(uint px)
{
    const uint3 w = g_pathStateBuffer.Load3(dv_end(px));
    DvMiss m;
    m.dir = UnpackNormal(w.x);
    m.skySun = UnpackRGB9E5(w.y);
    m.candMis = asfloat(w.z);
    return m;
}

// ---------------------------------------------------------------------------------------------
// Light record (60 bytes): the resolved light sample of the deferred vertex, its sun sample, the
// transmittance towards both, and the tree pdf of a pending emitter hit. The light pass writes it
// in pieces as each part is resolved; the material pass reads each technique on its own.
//   0 token  8 objID  12 pdfSolidAngle  16 position  28 normal  32 emission
//   36 sunDir  40 sunRadiance  44 sunPdf  48 emitterPdfArea  52 visLight  56 visSun
// ---------------------------------------------------------------------------------------------
void DvStoreLightPick(uint px, uint2 token, float emitterPdfArea)
{
    const uint a = dv_light(px);
    g_pathStateBuffer.Store2(a, token);
    g_pathStateBuffer.Store(a + 48u, asuint(emitterPdfArea));
}
void DvStoreLightMesh(uint px, uint objID, float pdfSolidAngle, float3 position, float3 normal, float3 emission)
{
    const uint a = dv_light(px);
    g_pathStateBuffer.Store2(a + 8u, uint2(objID, asuint(pdfSolidAngle)));
    g_pathStateBuffer.Store4(a + 16u, uint4(asuint(position), PackNormal(normal)));
    g_pathStateBuffer.Store(a + 32u, PvPackRadiance(emission));
}
void DvStoreLightSun(uint px, float3 sunDir, float3 sunRadiance, float sunPdf)
{
    g_pathStateBuffer.Store3(dv_light(px) + 36u,
        uint3(PackNormal(sunDir), PvPackRadiance(sunRadiance), asuint(sunPdf)));
}
// which: 0 the light point, 1 the sun.
void DvStoreLightVisibility(uint px, uint which, float3 vis)
{
    g_pathStateBuffer.Store(dv_light(px) + 52u + which * 4u, PackRGB9E5(saturate(vis)));
}
uint2 DvLoadLightToken(uint px) { return g_pathStateBuffer.Load2(dv_light(px)); }
float DvLoadEmitterPdfArea(uint px) { return asfloat(g_pathStateBuffer.Load(dv_light(px) + 48u)); }
void DvLoadLightMesh(uint px, out uint objID, out float pdfSolidAngle, out float3 position, out float3 normal,
    out float3 emission, out float3 vis)
{
    const uint  a  = dv_light(px);
    const uint2 w0 = g_pathStateBuffer.Load2(a + 8u);
    const uint4 w1 = g_pathStateBuffer.Load4(a + 16u);
    const uint  w2 = g_pathStateBuffer.Load(a + 32u);
    const uint  w3 = g_pathStateBuffer.Load(a + 52u);
    objID         = w0.x;
    pdfSolidAngle = asfloat(w0.y);
    position      = asfloat(w1.xyz);
    normal        = UnpackNormal(w1.w);
    emission      = PvUnpackRadiance(w2);
    vis           = UnpackRGB9E5(w3);
}
void DvLoadLightSun(uint px, out float3 sunDir, out float3 sunRadiance, out float sunPdf, out float3 vis)
{
    const uint  a  = dv_light(px);
    const uint3 w0 = g_pathStateBuffer.Load3(a + 36u);
    const uint  w1 = g_pathStateBuffer.Load(a + 56u);
    sunDir      = UnpackNormal(w0.x);
    sunRadiance = PvUnpackRadiance(w0.y);
    sunPdf      = asfloat(w0.z);
    vis         = UnpackRGB9E5(w1);
}

// Add one contribution of the current sample to the pixel.
void PtAddRadiance(uint2 pixel, float3 L)
{
    L /= (float)PtSampleCount();
    if (!any(L != 0.0f) || any(isnan(L)) || any(isinf(L))) return;
    gScratchPing[uint3(pixel, 2)] += float4(L, 0.0f);
}
