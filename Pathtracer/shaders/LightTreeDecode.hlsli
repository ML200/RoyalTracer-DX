#ifndef LIGHT_TREE_DECODE_HLSLI
#define LIGHT_TREE_DECODE_HLSLI

bool LT_CompactNodes() { return (rs_flags & RS_FLAG_COMPACT_LIGHT_TREE) != 0u; }

float3 LT_DecodeAxis(uint packed) {
    float2 p = float2(packed & 65535u, packed >> 16u) * (2.0f / 65535.0f) - 1.0f;
    float3 n = float3(p, 1.0f-abs(p.x)-abs(p.y));
    float t = max(-n.z, 0.0f);
    n.xy += float2(n.x >= 0.0f ? -t : t, n.y >= 0.0f ? -t : t);
    return normalize(n);
}
LightTLASNodeGpu LT_LoadTLAS(uint index) {
    bool compact=LT_CompactNodes();
    uint base=index*(compact?3u:4u);
    uint4 a=gLT_TLAS[base],b=gLT_TLAS[base+1u],c=gLT_TLAS[base+2u];
    LightTLASNodeGpu n;
    n.bmin=asfloat(a.xyz); n.bmax=asfloat(b.xyz); n.power=asfloat(a.w); n.cosTheta_o=asfloat(b.w);
    if(compact) {
        n.axis=LT_DecodeAxis(c.x); n.sinTheta_o=asfloat(c.y);
        n.firstChild=c.w ? c.z : 0xffffffffu; n.childCount=c.w;
        n.slot=c.w ? 0xffffffffu : c.z; n._pad=0u;
    } else {
        uint4 d=gLT_TLAS[base+3u];
        n.axis=asfloat(c.xyz);n.sinTheta_o=asfloat(c.w);
        n.firstChild=d.x;n.childCount=d.y;n.slot=d.z;n._pad=d.w;
    }
    return n;
}
struct LT_BlasFrame { float3 lo,hi; };
LT_BlasFrame LT_LoadBlasFrame(uint offset) {
    LT_BlasFrame f=(LT_BlasFrame)0;
    if(LT_CompactNodes()) {
        uint4 a=gLT_BLAS[offset*2u],b=gLT_BLAS[offset*2u+1u];
        f.lo=asfloat(a.xyz);f.hi=asfloat(uint3(a.w,b.xy));
    }
    return f;
}
float LT_BoundRound(float value,bool upper) {
    // Cover interpolation rounding when a mesh extent is small relative to its
    // local-coordinate offset. The integer quantizer also expands one grid unit.
    if(value==0.0f) return asfloat(upper?2u:0x80000002u);
    return asfloat(asuint(value)+((value>0.0f)==upper?2u:0xfffffffeu));
}
float3 LT_DecodeBounds(uint3 q, LT_BlasFrame f,bool upper) {
    float3 t=float3(q)*(1.0f/65535.0f);
    float3 v=(1.0f-t)*f.lo+t*f.hi;
    v=clamp(float3(LT_BoundRound(v.x,upper),LT_BoundRound(v.y,upper),LT_BoundRound(v.z,upper)),f.lo,f.hi);
    return float3(q.x==0u?f.lo.x:(q.x==65535u?f.hi.x:v.x),
                  q.y==0u?f.lo.y:(q.y==65535u?f.hi.y:v.y),
                  q.z==0u?f.lo.z:(q.z==65535u?f.hi.z:v.z));
}
LightBLASNodeGpu LT_LoadBLAS(uint offset,uint index,LT_BlasFrame f) {
    if(!LT_CompactNodes()) {
        uint base=(offset+index)*4u;
        uint4 a=gLT_BLAS[base],b=gLT_BLAS[base+1u],c=gLT_BLAS[base+2u],d=gLT_BLAS[base+3u];
        LightBLASNodeGpu full;
        full.bmin=asfloat(a.xyz);full.power=asfloat(a.w);full.bmax=asfloat(b.xyz);full.cosTheta_o=asfloat(b.w);
        full.axis=asfloat(c.xyz);full.sinTheta_o=asfloat(c.w);
        full.firstChild=d.x;full.childCount=d.y;full.triFirst=d.z;full.triCount=d.w;
        return full;
    }
    uint base=(offset+1u+index)*2u;
    uint4 a=gLT_BLAS[base],b=gLT_BLAS[base+1u];
    LightBLASNodeGpu n;
    uint3 q=a.xyz;
    n.bmin=LT_DecodeBounds(q&65535u,f,false); n.bmax=LT_DecodeBounds(q>>16u,f,true);
    n.power=asfloat(a.w); n.axis=LT_DecodeAxis(b.x); n.cosTheta_o=asfloat(b.y);
    n.sinTheta_o=sqrt(max(0.0f,1.0f-n.cosTheta_o*n.cosTheta_o));
    n.firstChild=b.w?b.z:0xffffffffu; n.childCount=b.w;
    n.triFirst=b.w?0u:b.z; n.triCount=b.w?0u:1u;
    return n;
}
LightBLASNodeGpu LT_LoadBLAS(uint offset,uint index) {
    return LT_LoadBLAS(offset,index,LT_LoadBlasFrame(offset));
}
#endif
