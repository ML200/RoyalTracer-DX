#define COMPUTE_PASS
#include "Includes_v8.hlsli"
cbuffer TestConstants : register(b0, space1) { uint workCount; uint testMode; uint triangleCount; uint testSeed; }
RWStructuredBuffer<float4> results : register(u0, space1);
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if(tid.x>=workCount) return;
    if(testMode==5u) {
        float3 bmin=float3(-.1f,-.1f,0),bmax=float3(.1f,.1f,0);
        float front=LT_NodeImportance_Common(float3(0,0,-2),float3(0,0,1),bmin,bmax,float3(0,0,-1),1,0,2);
        float back=LT_NodeImportance_Common(float3(0,0,-2),float3(0,0,1),bmin,bmax,float3(0,0,1),1,0,2);
        float near1=LT_NodeImportance_Common(float3(0,0,-.01f),float3(0,0,1),bmin,bmax,float3(0,0,-1),1,0,2);
        float near2=LT_NodeImportance_Common(float3(0,0,-.02f),float3(0,0,1),2*bmin,2*bmax,float3(0,0,-1),1,0,2);
        if(tid.x==0) results[tid.x]=float4(front,back,near1,near2);
        else {
            float4x4 inverse=float4x4(.5f,0,0,0,0,1.0f/3.0f,0,0,0,0,.25f,0,0,0,0,1);
            results[tid.x]=float4(LT_LocalReceiverNormal(inverse,normalize(float3(1,1,1))),LT_SGIntegral(0));
        }
        return;
    }
    float3 x=float3(0,0,-10),n=float3(0,0,testMode==0u?-1:1);
    uint tri=testMode==0u?tid.x:tid.x%triangleCount;
    uint rng=LTC_Hash(tid.x+testSeed)+1u;
    if(testMode==2u) {
        uint cell;uint matches=0,count=0;float sum=0;
        if(LTC_Find(x,n,cell)) {
            count=g_sharc.Load(cell+4u);sum=LTC_Sum(cell,count);
            for(uint i=0;i<count;++i) if(LTC_Contains(LTC_LoadNode(LTC_Cluster(cell,i)),tri)) ++matches;
        }
        results[tid.x]=float4(LT_PdfSelectTriangle(x,n,tri),matches,count,sum);return;
    }
    LT_Sample sample=LT_SampleLight(x,n,rng);
    float pdf=(rs_flags & RS_FLAG_NO_MESH_LIGHTS)!=0u?0:LT_PdfSelectTriangle(x,n,tri);
    float evaluated=LT_PdfSelectTriangle(x,n,sample.id);
    if(testMode==0u) {
        results[tid.x]=float4(pdf,sample.pdf,evaluated,sample.id==LT_SENTINEL?-1.0f:float(sample.id));return;
    }
    // A known discrete integral: one visible unit emitter, all others blocked.
    // Training includes the zero observations and never includes MIS weights.
    float estimate=sample.id==0u && sample.pdf>0u && testMode!=4u?1.0f/sample.pdf:0;
    if(testMode==1u || testMode==4u) LT_Train(x,n,sample.id,estimate);
    results[tid.x]=float4(pdf,sample.pdf,evaluated,estimate);
}
