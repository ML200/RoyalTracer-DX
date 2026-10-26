#pragma once

LightTLASNodeGpu LT_LoadTLAS(uint index) {
    uint base=index*4u;
    uint4 a=gLT_TLAS[base],b=gLT_TLAS[base+1u],c=gLT_TLAS[base+2u],d=gLT_TLAS[base+3u];
    LightTLASNodeGpu n;
    n.bmin=asfloat(a.xyz); n.bmax=asfloat(b.xyz); n.power=asfloat(a.w); n.cosTheta_o=asfloat(b.w);
    n.axis=asfloat(c.xyz);n.sinTheta_o=asfloat(c.w);
    n.firstChild=d.x;n.childCount=d.y;n.slot=d.z;n._pad=d.w;
    return n;
}
LightBLASNodeGpu LT_LoadBLAS(uint offset,uint index) {
    uint base=(offset+index)*4u;
    uint4 a=gLT_BLAS[base],b=gLT_BLAS[base+1u],c=gLT_BLAS[base+2u],d=gLT_BLAS[base+3u];
    LightBLASNodeGpu n;
    n.bmin=asfloat(a.xyz);n.power=asfloat(a.w);n.bmax=asfloat(b.xyz);n.cosTheta_o=asfloat(b.w);
    n.axis=asfloat(c.xyz);n.sinTheta_o=asfloat(c.w);
    n.firstChild=d.x;n.childCount=d.y;n.triFirst=d.z;n.triCount=d.w;
    return n;
}
