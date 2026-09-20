#pragma once

bool LT_CompactNodes() { return (rs_flags & RS_FLAG_COMPACT_LIGHT_TREE) != 0u; }

float3 LT_DecodeAxis(uint packed) {
    float2 p = float2(packed & 65535u, packed >> 16u) * (2.0f / 65535.0f) - 1.0f;
    float3 n = float3(p, 1.0f-abs(p.x)-abs(p.y));
    float t = max(-n.z, 0.0f);
    n.xy += float2(n.x >= 0.0f ? -t : t, n.y >= 0.0f ? -t : t);
    return normalize(n);
}
// vMF sharpness of a mean resultant length (Banerjee et al. 2005).
float LT_KappaOf(float R) {
    R = clamp(R, 0.0f, 0.9999f);
    return (3.0f - R * R) * R / (1.0f - R * R);
}
float3 LT_AxisOf(float3 rbar, out float R) {
    const float l2 = dot(rbar, rbar);
    R = sqrt(l2);
    return l2 > 1e-12f ? rbar / R : float3(0.0f, 0.0f, 1.0f);
}
LightTLASNodeGpu LT_DecodeFullTLAS(LightTLASNodeFull f) {
    LightTLASNodeGpu n;
    n.mean=f.mean; n.variance=f.variance; n.power=f.power; n.radius=f.radius; n.cosTheta_o=f.cosTheta_o;
    float R; n.axis=LT_AxisOf(f.rbar,R); n.kappa=LT_KappaOf(R);
    n.firstChild=f.firstChild; n.childCount=f.childCount; n.slot=f.slot;
    return n;
}
LightBLASNodeGpu LT_DecodeFullBLAS(LightBLASNodeFull f) {
    LightBLASNodeGpu n;
    n.mean=f.mean; n.variance=f.variance; n.power=f.power; n.radius=f.radius; n.cosTheta_o=f.cosTheta_o;
    float R; n.axis=LT_AxisOf(f.rbar,R); n.kappa=LT_KappaOf(R);
    n.firstChild=f.firstChild; n.childCount=f.childCount; n.triFirst=f.triFirst; n.triCount=f.triCount;
    return n;
}
LightTLASNodeGpu LT_LoadTLAS(uint index) {
    if(!LT_CompactNodes()) {
        uint base=index*4u;
        uint4 a=gLT_TLAS[base],b=gLT_TLAS[base+1u],c=gLT_TLAS[base+2u],d=gLT_TLAS[base+3u];
        LightTLASNodeFull f;
        f.mean=asfloat(a.xyz); f.variance=asfloat(a.w); f.rbar=asfloat(b.xyz); f.power=asfloat(b.w);
        f.radius=asfloat(c.x); f.cosTheta_o=asfloat(c.y); f.firstChild=c.z; f.childCount=c.w;
        f.slot=d.x; f._pad=d.yzw;
        return LT_DecodeFullTLAS(f);
    }
    uint base=index*3u;
    uint4 a=gLT_TLAS[base],b=gLT_TLAS[base+1u],c=gLT_TLAS[base+2u];
    LightTLASNodeGpu n;
    n.mean=asfloat(a.xyz); n.power=asfloat(a.w);
    n.variance=asfloat(b.x); n.radius=asfloat(b.y);
    n.axis=LT_DecodeAxis(b.z);
    n.kappa=LT_KappaOf(f16tof32(b.w & 0xFFFFu)); n.cosTheta_o=f16tof32(b.w >> 16u);
    n.firstChild=c.y ? c.x : 0xffffffffu; n.childCount=c.y;
    n.slot=c.y ? 0xffffffffu : c.x;
    return n;
}
// Compact meshes store their standard deviation and radius relative to the unit in their
// header (word 0: twice the root radius).
struct LT_BlasFrame { float diag; };
LT_BlasFrame LT_LoadBlasFrame(uint offset) {
    LT_BlasFrame f=(LT_BlasFrame)0;
    if(LT_CompactNodes()) f.diag=asfloat(gLT_BLAS[offset*2u].x);
    return f;
}
LightBLASNodeGpu LT_LoadBLAS(uint offset,uint index,LT_BlasFrame f) {
    if(!LT_CompactNodes()) {
        uint base=(offset+index)*4u;
        uint4 a=gLT_BLAS[base],b=gLT_BLAS[base+1u],c=gLT_BLAS[base+2u],d=gLT_BLAS[base+3u];
        LightBLASNodeFull full;
        full.mean=asfloat(a.xyz); full.variance=asfloat(a.w); full.rbar=asfloat(b.xyz); full.power=asfloat(b.w);
        full.radius=asfloat(c.x); full.cosTheta_o=asfloat(c.y); full.firstChild=c.z; full.childCount=c.w;
        full.triFirst=d.x; full.triCount=d.y; full._pad=d.zw;
        return LT_DecodeFullBLAS(full);
    }
    uint base=(offset+1u+index)*2u;
    uint4 a=gLT_BLAS[base],b=gLT_BLAS[base+1u];
    LightBLASNodeGpu n;
    n.mean=asfloat(a.xyz); n.power=asfloat(a.w);
    n.axis=LT_DecodeAxis(b.x);
    const float sigma=f16tof32(b.y & 0xFFFFu)*f.diag;
    n.variance=sigma*sigma; n.radius=f16tof32(b.y >> 16u)*f.diag;
    n.kappa=LT_KappaOf(f16tof32(b.z & 0xFFFFu)); n.cosTheta_o=f16tof32(b.z >> 16u);
    const uint count=b.w >> LT_PACKED_INDEX_BITS, index2=b.w & LT_PACKED_INDEX_MASK;
    n.firstChild=count ? index2 : 0xffffffffu; n.childCount=count;
    n.triFirst=count ? 0u : index2; n.triCount=count ? 0u : 1u;
    return n;
}
LightBLASNodeGpu LT_LoadBLAS(uint offset,uint index) {
    return LT_LoadBLAS(offset,index,LT_LoadBlasFrame(offset));
}
