#ifndef LIGHT_TREE_LEARNING_HLSLI
#define LIGHT_TREE_LEARNING_HLSLI
#include "PersistentSamplingBuffer_v8.hlsli"
#ifdef LT_TEST_NO_CAMERA
#define LT_WORLD_ORIGIN float3(0,0,0)
#else
#define LT_WORLD_ORIGIN sceneOriginWorld
#endif
// Wang et al. 2021, Learning to Cluster for Rendering with Many Lights.
// Frozen per-frame cuts; rendering writes only requests and batch statistics.
// The prepare dispatch consumes them behind a UAV barrier on the next frame.
// See docs/LIGHT_TREE_LEARNING.md for the estimator and implementation choices.
uint LTC_Hash(uint v) { v^=v>>16;v*=0x7feb352du;v^=v>>15;v*=0x846ca68bu;return v^(v>>16); }
uint LTC_Cell(uint slot) { return LT_BUFFER_OFFSET+slot*LT_CELL_BYTES; }
uint LTC_Cluster(uint cell,uint slot) { return cell+LT_CELL_HEADER+slot*LT_CLUSTER_BYTES; }
uint4 LTC_Key(float3 x,float3 n) {
    // World positions keep keys stable under floating-origin rebases. Root
    // constant 24 carries cell size under PT and normal fuzz under legacy ReSTIR.
    float size=max(spmis_normalFuzz,1e-4f);
    int3 grid=(int3)floor((x+LT_WORLD_ORIGIN)/size);
    uint3 normal=(uint3)clamp(floor(n*3.0f+3.5f),0.0f,6.0f);
    return uint4(asuint(grid),normal.x+7u*normal.y+49u*normal.z);
}
uint LTC_Slot(uint4 k) { return LTC_Hash(k.x^LTC_Hash(k.y)^LTC_Hash(k.z)^LTC_Hash(k.w))&(LT_CELL_CAPACITY-1u); }
bool LTC_Enabled() { return (rs_flags & (LT_FLAG_LEARNING|RS_FLAG_NO_MESH_LIGHTS))==LT_FLAG_LEARNING; }
bool LTC_Find(float3 x,float3 n,out uint cell) {
    cell=0u;
    if(!LTC_Enabled()) return false;
    uint4 key=LTC_Key(x,n);cell=LTC_Cell(LTC_Slot(key));
    return g_sharc.Load(cell)==1u && all(g_sharc.Load4(cell+16u)==key);
}
void LTC_Request(float3 x,float3 n) {
    if(!LTC_Enabled()) return;
    uint4 key=LTC_Key(x,n);uint cell=LTC_Cell(LTC_Slot(key));
    uint old;g_sharc.InterlockedCompareExchange(cell+64u,0u,1u,old);
    if(old!=0u) return;
    g_sharc.Store4(cell+80u,key);
    g_sharc.Store3(cell+96u,asuint(x+LT_WORLD_ORIGIN));
    g_sharc.Store3(cell+112u,asuint(n));
}
struct LTC_Node { uint node,blas;uint2 trail;uint depth; };
LTC_Node LTC_Root() { LTC_Node n;n.node=0;n.blas=LT_SENTINEL;n.trail=0;n.depth=0;return n; }
LTC_Node LTC_LoadNode(uint address) {
    uint4 a=g_sharc.Load4(address);LTC_Node n;n.node=a.x;n.blas=a.y;n.trail=a.zw;n.depth=g_sharc.Load(address+16u);return n;
}
void LTC_StoreNode(uint address,LTC_Node n) {
    g_sharc.Store4(address,uint4(n.node,n.blas,n.trail));g_sharc.Store(address+16u,n.depth);
}
bool LTC_Prefix(uint2 path,uint2 prefix,uint depth) {
    if(depth==0u) return true;
    if(depth<16u) return ((path.x^prefix.x)&((1u<<(2u*depth))-1u))==0u;
    if(path.x!=prefix.x) return false;
    if(depth==16u) return true;
    if(depth==32u) return path.y==prefix.y;
    return ((path.y^prefix.y)&((1u<<(2u*(depth-16u)))-1u))==0u;
}
bool LTC_Contains(LTC_Node c,uint triIndex) {
    uint blas=gLT_TriToBLAS[triIndex];
    if(blas==LT_SENTINEL) return false;
    return c.blas==LT_SENTINEL?LTC_Prefix(gLT_BLASBitTrail[blas],c.trail,c.depth)
        :(c.blas==blas && LTC_Prefix(gLT_TriBitTrail[triIndex],c.trail,c.depth));
}
float LTC_Weight(uint address) { float q=asfloat(g_sharc.Load(address+20u));return isfinite(q)?max(0.0f,q):0.0f; }
float LTC_Sum(uint cell,uint count) {
    float sum=0;
    [loop] for(uint i=0;i<count;++i) sum+=LTC_Weight(LTC_Cluster(cell,i));
    return isfinite(sum)?sum:0.0f;
}
float LTC_Probability(uint address,uint count,float sum) {
    // Exploration prevents a noisy zero observation from permanently starving
    // a contributing cluster. The actual mixture is used by sampling and MIS.
    return sum>0?0.95f*LTC_Weight(address)/sum+0.05f/float(count):1.0f/float(count);
}
float LTC_Prior(LTC_Node c,float3 x,float3 n) {
    if(c.blas==LT_SENTINEL) return LT_NodeImportance_TLAS(gLT_TLAS[c.node],x,n);
    BlasRangeGpu r=gLT_Range[c.blas];
    return LT_NodeImportance_BLAS(gLT_BLAS[r.nodeOffset+c.node],mul(r.worldToLocal,float4(x,1)).xyz,LT_LocalReceiverNormal(r.worldToLocal,n));
}
uint LTC_Children(inout LTC_Node c,out uint first) {
    first=0;
    if(c.blas==LT_SENTINEL) {
        LightTLASNodeGpu t=gLT_TLAS[c.node];
        if(t.childCount>0u) { first=t.firstChild;return t.childCount; }
        // Crossing the TLAS leaf introduces a new, independent BLAS trail.
        c.node=0u;c.blas=t.blasIndex;c.trail=0u;c.depth=0u;
    }
    LightBLASNodeGpu b=gLT_BLAS[gLT_Range[c.blas].nodeOffset+c.node];
    first=b.firstChild;return b.childCount;
}
LTC_Node LTC_Child(LTC_Node parent,uint first,uint child) {
    parent.node=first+child;
    if(parent.depth<16u) parent.trail.x|=child<<(2u*parent.depth);
    else parent.trail.y|=child<<(2u*(parent.depth-16u));
    ++parent.depth;return parent;
}
LT_Sample LT_SampleLight(float3 x,float3 n,inout uint rng) {
    uint cell;
    if(!LTC_Find(x,n,cell)) { LTC_Request(x,n);return LT_SampleSubtree(x,n,rng); }
    uint count=g_sharc.Load(cell+4u);
    if(count==0u || count>LT_CUT_MAX) return LT_SampleSubtree(x,n,rng);
    float sum=LTC_Sum(cell,count), target=RandomFloatSingle(rng), accum=0,probability=0;
    uint chosen=count-1u;
    [loop] for(uint i=0;i<count;++i) {
        float p=LTC_Probability(LTC_Cluster(cell,i),count,sum);
        if(target<accum+p || i+1u==count) { chosen=i;probability=p;break; }
        accum+=p;
    }
    LTC_Node c=LTC_LoadNode(LTC_Cluster(cell,chosen));
    LT_Sample sample=LT_SampleSubtree(x,n,rng,c.node,c.blas);
    sample.pdf*=probability;return sample;
}
float LT_PdfSelectTriangle(float3 x,float3 n,uint tri) {
    if(tri==LT_SENTINEL || (rs_flags & RS_FLAG_NO_MESH_LIGHTS)!=0u) return 0;
    uint cell;
    if(!LTC_Find(x,n,cell)) return LT_PdfSubtree(x,n,tri);
    uint count=g_sharc.Load(cell+4u);
    if(count==0u || count>LT_CUT_MAX) return LT_PdfSubtree(x,n,tri);
    float sum=LTC_Sum(cell,count);
    [loop] for(uint i=0;i<count;++i) {
        uint a=LTC_Cluster(cell,i);LTC_Node c=LTC_LoadNode(a);
        if(LTC_Contains(c,tri)) return LTC_Probability(a,count,sum)*LT_PdfSubtree(x,n,tri,c.node,c.blas,c.depth);
    }
    return 0; // A valid cut partitions every leaf exactly once.
}
void LTC_AtomicAddFloat(uint address,float value) {
    if(value==0.0f) return;
    uint observed=g_sharc.Load(address);
    [allow_uav_condition] while(true) {
        uint expected=observed;
        float next=asfloat(expected)+value;
        // This protects learning metadata only, never the radiance estimator.
        next=min(next,3.0e38f);
        g_sharc.InterlockedCompareExchange(address,expected,asuint(next),observed);
        if(observed==expected) break;
    }
}
void LT_Train(float3 x,float3 n,uint tri,float contributionOverPdf) {
    if(tri==LT_SENTINEL) return;
    uint cell;if(!LTC_Find(x,n,cell)) return;
    uint count=g_sharc.Load(cell+4u);
    if(count==0u || count>LT_CUT_MAX) return;
    float sum=LTC_Sum(cell,count);
    [loop] for(uint i=0;i<count;++i) {
        uint a=LTC_Cluster(cell,i);
        if(!LTC_Contains(LTC_LoadNode(a),tri)) continue;
        float p=LTC_Probability(a,count,sum);
        float reward=max(0.0f,contributionOverPdf)*p;
        if(!isfinite(reward)) reward=0.0f;
        LTC_AtomicAddFloat(a+48u,reward);
        LTC_AtomicAddFloat(a+52u,min(reward*reward,3.0e38f));
        uint ignored;g_sharc.InterlockedAdd(a+56u,1u,ignored);
        return;
    }
}
#endif

