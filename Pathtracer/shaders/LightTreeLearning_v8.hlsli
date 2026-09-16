#ifndef LIGHT_TREE_LEARNING_HLSLI
#define LIGHT_TREE_LEARNING_HLSLI
#include "PersistentSamplingBuffer_v8.hlsli"
#ifdef LT_TEST_NO_CAMERA
cbuffer LightTreeTestConstants : register(b0, space1) {
    uint workCount,testMode,triangleCount,testSeed;
    float3 testCameraPosition;float testRewardScale;
}
#define LT_WORLD_ORIGIN float3(0,0,0)
#else
#define LT_WORLD_ORIGIN sceneOriginWorld
#endif

// Hash cell keys before bucket probing.
uint LTC_Hash(uint v) { v^=v>>16;v*=0x7feb352du;v^=v>>15;v*=0x846ca68bu;return v^(v>>16); }

uint LTC_KeyAddress(uint slot) { return LT_BUFFER_OFFSET+slot*LT_KEY_BYTES; }
uint LTC_Cell(uint slot) { return LT_BUFFER_OFFSET+LT_CELL_CAPACITY*LT_KEY_BYTES+slot*LT_CELL_HEADER; }
uint LTC_Index(uint cell) { return (cell-LTC_Cell(0u))/LT_CELL_HEADER; }

uint LTC_FrozenBase() { return LT_BUFFER_OFFSET+LT_CELL_CAPACITY*(LT_KEY_BYTES+LT_CELL_HEADER); }
uint LTC_StatsBase() { return LTC_FrozenBase()+LT_CELL_CAPACITY*LT_CUT_MAX*LT_FROZEN_BYTES; }
uint LTC_Cluster(uint cell,uint slot) { return LTC_FrozenBase()+(LTC_Index(cell)*LT_CUT_MAX+slot)*LT_FROZEN_BYTES; }
uint LTC_ClusterIndex(uint cluster) { return (cluster-LTC_FrozenBase())/LT_FROZEN_BYTES; }
uint LTC_Stats(uint cluster) { return LTC_StatsBase()+LTC_ClusterIndex(cluster)*LT_STATS_BYTES; }
uint LTC_Token(uint cell,uint cluster) { return LTC_Index(cell)*LT_CUT_MAX+cluster+1u; }
uint LTC_TokenAddress(uint token) { return LTC_FrozenBase()+(token-1u)*LT_FROZEN_BYTES; }
uint LTC_NormalFace(float3 n) {
    float3 a=abs(n);uint axis=a.x>=a.y && a.x>=a.z?0u:(a.y>=a.z?1u:2u);
    return axis*2u+(n[axis]<0.0f?1u:0u);
}
float3 LTC_FaceNormal(uint face) { float3 n=0;n[face/2u]=(face&1u)!=0u?-1.0f:1.0f;return n; }
float3 LTC_CameraPosition() {
#ifdef LT_TEST_NO_CAMERA
    return testCameraPosition;
#else
    return mul(viewI,float4(0,0,0,1)).xyz;
#endif
}
// Map camera distance to a clamped adaptive grid level.
float LTC_ContinuousLevel(float3 x) {
    float width=max(spmis_normalFuzz,1e-4f);
    float growth=max(asfloat(spmis_normalBits),0.001f);
    return clamp(log2(max(1.0f,length(x-LTC_CameraPosition())*growth/width)),0.0f,float(LT_MAX_LEVEL));
}
uint LTC_Level(float3 x) { return (uint)LTC_ContinuousLevel(x); }
uint LTC_Now() { return asuint(spmis_searchR0); }
uint4 LTC_KeyAtLevel(float3 x,float3 n,uint level) {
    float width=max(spmis_normalFuzz,1e-4f)*exp2(float(level));
    int3 grid=(int3)floor((x+LT_WORLD_ORIGIN)/width);
    uint3 normal=(uint3)clamp(floor(n*3.0f+3.5f),0.0f,6.0f);
    return uint4(asuint(grid),normal.x+7u*normal.y+49u*normal.z+((level+1u)<<9u));
}
uint4 LTC_Key(float3 x,float3 n) { return LTC_KeyAtLevel(x,n,LTC_Level(x)); }
uint LTC_KeyLevel(uint4 key) { return (key.w>>9u)-1u; }
uint LTC_Slot(uint4 k) { return LTC_Hash(k.x^LTC_Hash(k.y)^LTC_Hash(k.z)^LTC_Hash(k.w))&(LT_GRID_CAPACITY-1u); }
uint LTC_Probe(uint4 key,uint probe) {
    uint h=probe<LT_BUCKET_SIZE?LTC_Slot(key):LTC_Hash(key.y^LTC_Hash(key.z+0x9e3779b9u)^LTC_Hash(key.x+key.w));
    return (h & (LT_GRID_CAPACITY-LT_BUCKET_SIZE))+(probe&(LT_BUCKET_SIZE-1u));
}
bool LTC_Enabled() { return (rs_flags & (LT_FLAG_LEARNING|RS_FLAG_NO_MESH_LIGHTS))==LT_FLAG_LEARNING; }

bool LTC_UseSurfaceLearning() { return LTC_Enabled(); }

float3 LTC_TrainShare(float roughness,uint matID,float3 full,float3 broad) {
    const float lo=spmis_searchGrow;
    const bool smoothCoat=LoadPc(matID)>0.01f && LoadPcr(matID)<lo;
    return (roughness>=lo && !smoothCoat)?full:broad;
}
static const uint LTC_EXPIRED_SCORE=0x40000000u;
bool LTC_Replaceable(uint cell,bool underPressure=false) {
    return LTC_Index(cell)<LT_GRID_CAPACITY && (g_sharc.Load(cell)!=1u ||
        g_sharc.Load(cell+20u)>=(underPressure?1u:LTC_EXPIRED_SCORE));
}
void LTC_UpdateRetention(uint cell) {
    if(g_sharc.Load(cell+68u)==sharc_frame-1u) {
        g_sharc.Store(cell+12u,sharc_frame-1u);g_sharc.Store(cell+24u,LTC_Now());
    }
    uint score=0u;
    if(LTC_Index(cell)<LT_GRID_CAPACITY) {
        uint level=LTC_KeyLevel(g_sharc.Load4(LTC_KeyAddress(LTC_Index(cell))));
        float3 x=asfloat(g_sharc.Load3(cell+32u))-LT_WORLD_ORIGIN;
        uint desired=LTC_Level(x),age=LTC_Now()-g_sharc.Load(cell+24u);
        bool distant=level+1u<desired;

        if(age>LT_CELL_PRESSURE_MS) score=min(age,LTC_EXPIRED_SCORE-1u);

        if(age>(distant?LT_CELL_DISTANT_MS:LT_CELL_HISTORY_MS))
            score=min(age,LTC_EXPIRED_SCORE-1u)|(distant?0x80000000u:LTC_EXPIRED_SCORE);
    }
    g_sharc.Store(cell+20u,score);
}
// Probe the fixed-size hash table for an exact key.
bool LTC_FindExact(uint4 key,out uint cell) {
    cell=0u;
    [unroll] for(uint probe=0;probe<LT_CELL_PROBES;++probe) {
        uint index=LTC_Probe(key,probe);
        if(all(g_sharc.Load4(LTC_KeyAddress(index))==key)) {cell=LTC_Cell(index);return true;}
    }
    return false;
}
bool LTC_FindFrom(float3 x,float3 n,uint level,out uint cell) {
    [loop] for(uint parent=0;parent<LT_PARENT_LEVELS && level+parent<=LT_MAX_LEVEL;++parent)
        if(LTC_FindExact(LTC_KeyAtLevel(x,n,level+parent),cell)) return true;
    cell=LTC_Cell(LT_GRID_CAPACITY+LTC_NormalFace(n));
    return g_sharc.Load(cell)==1u;
}
bool LTC_Find(float3 x,float3 n,out uint cell) {
    cell=0u;return LTC_Enabled() && LTC_FindFrom(x,n,LTC_Level(x),cell);
}
struct LTC_Proposal { uint fine,coarse;float blend; };
// Select fine and ancestor cells for a blended proposal.
bool LTC_GetProposal(float3 x,float3 n,out LTC_Proposal proposal,bool warmup=false) {
    proposal=(LTC_Proposal)0;if(!LTC_Enabled()) return false;
    float level=LTC_ContinuousLevel(x);
    uint selectedLevel=0u;

    [loop] for(uint pass=0u;pass<2u;++pass) {
        uint found;
        bool ok=LTC_FindFrom(x,n,pass==0u?(uint)level:selectedLevel+1u,found);
        if(pass!=0u) {proposal.coarse=found;break;}
        if(!ok) return false;
        proposal.fine=found;proposal.coarse=found;
        uint index=LTC_Index(proposal.fine);
        if(index>=LT_GRID_CAPACITY) return true;
        selectedLevel=LTC_KeyLevel(g_sharc.Load4(LTC_KeyAddress(index)));

        float fraction=level-float(selectedLevel);
        proposal.blend=smoothstep(LT_LOD_BLEND_START,1.0f,fraction);

        if(!(proposal.blend>0.0f || (warmup && fraction>=LT_LOD_WARMUP_START))) break;
    }
    if(proposal.coarse==proposal.fine) proposal.blend=0.0f;
    return true;
}
// Queue refinement without blocking the sampling path.
void LTC_RequestCell(float3 x,float3 n,uint sourceCell,uint desired,uint alternatives) {
    uint index=LTC_Index(sourceCell);
    uint4 key=LTC_KeyAtLevel(x,n,desired);uint candidate=LT_SENTINEL;

    uint4 mask=WaveMatch(key);
    int4 highest=(int4)(firstbithigh(mask)|uint4(0,32,64,96));
    uint leader=(uint)max(max(highest.x,highest.y),max(highest.z,highest.w));
    if(WaveGetLaneIndex()!=leader) return;

    uint idleCandidate=LT_SENTINEL;
    [loop] for(uint attempt=0u;attempt<(alternatives>0u?2u:1u);++attempt) {
        if(attempt!=0u) key=LTC_KeyAtLevel(x,n,desired+1u+sharc_frame%alternatives);
        uint best=0u;

        [unroll] for(uint probe=0;probe<LT_CELL_PROBES;++probe) {
            uint slot=LTC_Probe(key,probe);uint4 stored=g_sharc.Load4(LTC_KeyAddress(slot));
            if(all(stored==key)) return;
            uint c=LTC_Cell(slot);
            uint score=g_sharc.Load(c)!=1u?0xffffffffu:g_sharc.Load(c+20u);
            if(score>best) {candidate=c;best=score;}
        }

        if(best<LTC_EXPIRED_SCORE) {
            if(attempt==0u) idleCandidate=candidate;
            candidate=LT_SENTINEL;
        }
        if(candidate!=LT_SENTINEL) break;
    }
    if(candidate==LT_SENTINEL) {
        if(idleCandidate==LT_SENTINEL) return;
        candidate=idleCandidate;key=LTC_KeyAtLevel(x,n,desired);
    }
    uint old;g_sharc.InterlockedCompareExchange(candidate+64u,0u,1u,old);
    if(old!=0u) return;
    g_sharc.Store4(candidate+80u,key);
    g_sharc.Store3(candidate+96u,asuint(x+LT_WORLD_ORIGIN));
    g_sharc.Store3(candidate+112u,asuint(n));
    g_sharc.Store(candidate+76u,index);

    g_sharc.InterlockedExchange(sourceCell+68u,sharc_frame,old);
}
void LTC_RequestRefinement(float3 x,float3 n,uint parentCell,uint desired) {
    uint index=LTC_Index(parentCell);
    uint parentLevel=index>=LT_GRID_CAPACITY?min(desired+LT_PARENT_LEVELS,LT_MAX_LEVEL+1u):LTC_KeyLevel(g_sharc.Load4(LTC_KeyAddress(index)));
    if(parentLevel<=desired) return;
    LTC_RequestCell(x,n,parentCell,desired,min(parentLevel-desired-1u,LT_PARENT_LEVELS-1u));
}
void LTC_RequestRefinement(float3 x,float3 n,uint parentCell) {
    LTC_RequestRefinement(x,n,parentCell,LTC_Level(x));
}
void LTC_Request(float3 x,float3 n) {
    uint cell;if(LTC_Find(x,n,cell)) LTC_RequestRefinement(x,n,cell);
}
float3 LTC_DebugColor(float3 x,float3 n) {
    LTC_Proposal proposal;if(!LTC_GetProposal(x,n,proposal)) return float3(1,0,0);
    uint cell=proposal.blend>=0.5f?proposal.coarse:proposal.fine;
    uint index=LTC_Index(cell),updates=g_sharc.Load(cell+8u);
    float brightness=0.2f+0.8f*saturate(log2(1.0f+float(updates))/6.0f);
    if(index>=LT_GRID_CAPACITY) return brightness*float3(1,.4f,.05f);
    uint level=LTC_KeyLevel(g_sharc.Load4(LTC_KeyAddress(index)));
    return brightness*(level==LTC_Level(x)?float3(.1f,1,.2f):float3(.1f,.4f,1));
}

struct LTC_Node { uint node,slot;uint2 trail;uint depth,parent; };
LTC_Node LTC_Root() { LTC_Node n;n.node=0;n.slot=LT_SENTINEL;n.trail=0;n.depth=0;n.parent=LT_SENTINEL;return n; }
LTC_Node LTC_LoadNode(uint address) {

    uint4 a=g_sharc.Load4(address+LT_FZ_NODE),b=g_sharc.Load4(address+LT_FZ_DEPTH);
    LTC_Node n;n.node=a.x;n.slot=a.y;n.trail=a.zw;n.depth=b.x;n.parent=b.y;return n;
}

float LTC_NodePower(LTC_Node c) {
    if(c.slot==LT_SENTINEL) return LT_LoadTLAS(c.node).power;
    LightSlotGpu s=gLT_Slot[c.slot];
    return LT_LoadBLAS(s.nodeOffset,c.node).power*s.powerScale;
}
void LTC_StoreNode(uint address,LTC_Node n) {
    g_sharc.Store4(address+LT_FZ_NODE,uint4(n.node,n.slot,n.trail));g_sharc.Store2(address+LT_FZ_DEPTH,uint2(n.depth,n.parent));

    float power=LTC_NodePower(n);
    g_sharc.Store(LTC_Stats(address)+LT_ST_POWER,asuint(max(0.0f,power)));
}
bool LTC_Prefix(uint2 path,uint2 prefix,uint depth) {
    if(depth==0u) return true;
    if(depth<16u) return ((path.x^prefix.x)&((1u<<(2u*depth))-1u))==0u;
    if(path.x!=prefix.x) return false;
    if(depth==16u) return true;
    if(depth==32u) return path.y==prefix.y;
    return ((path.y^prefix.y)&((1u<<(2u*(depth-16u)))-1u))==0u;
}

bool LTC_Contains(LTC_Node c,uint triIndex,uint slot) {
    if(slot==LT_SENTINEL || triIndex==LT_SENTINEL) return false;
    return c.slot==LT_SENTINEL?LTC_Prefix(gLT_BLASBitTrail[slot],c.trail,c.depth)
        :(c.slot==slot && LTC_Prefix(gLT_TriBitTrail[triIndex],c.trail,c.depth));
}
float LTC_Weight(uint address) { float q=asfloat(g_sharc.Load(LTC_Stats(address)+LT_ST_Q));return isfinite(q)?max(0.0f,q):0.0f; }
float LTC_Sum(uint cell,uint count) {
    float sum=0;
    [loop] for(uint i=0;i<count;++i) sum+=LTC_Weight(LTC_Cluster(cell,i));
    return isfinite(sum)?sum:0.0f;
}
float LTC_Probability(uint address,uint count,float sum,float powerSum) {
    float prior=powerSum>0.0f?asfloat(g_sharc.Load(LTC_Stats(address)+LT_ST_POWER))/powerSum:1.0f/float(count);

    float explore=0.8f*prior+0.2f/float(count);
    return sum>0?0.95f*LTC_Weight(address)/sum+0.05f*explore:explore;
}
float LTC_FrozenProbability(uint address) { return asfloat(g_sharc.Load(address+LT_FZ_PROBABILITY)); }
float LTC_Prior(LTC_Node c,float3 x,float3 n) {
    if(c.slot==LT_SENTINEL) return LT_NodeImportance_TLAS(LT_LoadTLAS(c.node),x,n);
    LightSlotGpu s=gLT_Slot[c.slot];
    return LT_NodeImportance_BLAS(LT_LoadBLAS(s.nodeOffset,c.node),x,n,s.worldToLocal)*s.powerScale;
}
uint LTC_Children(inout LTC_Node c,out uint first) {
    first=0;
    if(c.slot==LT_SENTINEL) {
        LightTLASNodeGpu t=LT_LoadTLAS(c.node);
        if(t.childCount>0u) { first=t.firstChild;return t.childCount; }

        c.node=0u;c.slot=t.slot;c.trail=0u;c.depth=0u;c.parent=LT_SENTINEL;
    }
    LightBLASNodeGpu b=LT_LoadBLAS(gLT_Slot[c.slot].nodeOffset,c.node);
    first=b.firstChild;return b.childCount;
}
LTC_Node LTC_Child(LTC_Node parent,uint first,uint child) {
    parent.parent=parent.node;parent.node=first+child;
    if(parent.depth<16u) parent.trail.x|=child<<(2u*parent.depth);
    else parent.trail.y|=child<<(2u*(parent.depth-16u));
    ++parent.depth;return parent;
}
uint2 LTC_OrderedTrail(uint2 trail) {
    uint2 r=reversebits(trail);
    return ((r&0x55555555u)<<1u)|((r>>1u)&0x55555555u);
}
uint4 LTC_OrderKey(LTC_Node c) {
    uint2 t=c.slot==LT_SENTINEL?c.trail:gLT_BLASBitTrail[c.slot];
    return uint4(LTC_OrderedTrail(t),c.slot==LT_SENTINEL?uint2(0,0):LTC_OrderedTrail(c.trail));
}
bool LTC_OrderLE(uint4 a,uint4 b) {
    return a.x!=b.x?a.x<b.x:(a.y!=b.y?a.y<b.y:(a.z!=b.z?a.z<b.z:a.w<=b.w));
}
uint LTC_ClusterForTriangle(uint cell,uint count,uint tri,uint slot) {
    if(slot==LT_SENTINEL || tri==LT_SENTINEL) return LT_SENTINEL;
    uint4 key=uint4(LTC_OrderedTrail(gLT_BLASBitTrail[slot]),LTC_OrderedTrail(gLT_TriBitTrail[tri]));
    uint lo=0u,hi=count;
    [loop] while(lo<hi) {
        uint mid=(lo+hi)/2u;
        if(LTC_OrderLE(LTC_OrderKey(LTC_LoadNode(LTC_Cluster(cell,mid))),key)) lo=mid+1u;
        else hi=mid;
    }
    if(lo==0u) return LT_SENTINEL;
    uint a=LTC_Cluster(cell,lo-1u);
    return LTC_Contains(LTC_LoadNode(a),tri,slot)?a:LT_SENTINEL;
}

// Evaluate a triangle PDF inside one frozen cut.
float LTC_CellPdf(float3 x,float3 n,uint cell,uint tri,uint slot,out uint token) {
    token=0u;
    uint node=0u,start=LT_SENTINEL,depth=0u;float probability=1.0f;
    uint count=cell==LT_SENTINEL?0u:g_sharc.Load(cell+4u);
    if(count!=0u && count<=LT_CUT_MAX) {
        uint a=LTC_ClusterForTriangle(cell,count,tri,slot);if(a==LT_SENTINEL) return 0;
        token=LTC_ClusterIndex(a)+1u;
        LTC_Node c=LTC_LoadNode(a);
        node=c.node;start=c.slot;depth=c.depth;probability=LTC_FrozenProbability(a);
    }
    return probability*LT_PdfSubtree(x,n,tri,slot,node,start,depth);
}
uint LTC_RefreshToken(float3 x,float3 n,uint cell,uint tri,uint slot,uint ticket) {
    if(ticket>=4u || LTC_Index(cell)>=LT_GRID_CAPACITY) return 0u;
    uint parent=LTC_Cell(LT_GRID_CAPACITY+LTC_NormalFace(n));
    uint level=LTC_KeyLevel(g_sharc.Load4(LTC_KeyAddress(LTC_Index(cell))));
    uint found;
    if(ticket<3u && level+ticket+1u<=LT_MAX_LEVEL &&
        LTC_FindExact(LTC_KeyAtLevel(x,n,level+ticket+1u),found)) parent=found;
    uint count=g_sharc.Load(parent+4u),a=LTC_ClusterForTriangle(parent,count,tri,slot);
    return a==LT_SENTINEL?0u:LTC_ClusterIndex(a)+1u;
}
// Sample the learned mixture, then attach training tokens.
LT_Sample LT_SampleLight(float3 x,float3 n,inout uint rng,bool useLearning=true) {

    LTC_Proposal proposal=(LTC_Proposal)0;
    const bool learned=useLearning && LTC_GetProposal(x,n,proposal,true);
    uint cell=LT_SENTINEL,chosen=0u,node=0u,start=LT_SENTINEL;
    float probability=1.0f;
    bool useCoarse=false;uint refreshTicket=0u;
    if(learned) {

        [loop] for(uint request=0u;request<2u;++request) {
            uint desired,alternatives;
            if(request==0u) {
                desired=LTC_Level(x);
                uint index=LTC_Index(proposal.fine);
                uint parentLevel=index>=LT_GRID_CAPACITY?min(desired+LT_PARENT_LEVELS,LT_MAX_LEVEL+1u):LTC_KeyLevel(g_sharc.Load4(LTC_KeyAddress(index)));
                if(parentLevel<=desired) continue;
                alternatives=min(parentLevel-desired-1u,LT_PARENT_LEVELS-1u);
            } else {
                if(proposal.coarse==proposal.fine) break;
                desired=min(LTC_Level(x)+1u,LT_MAX_LEVEL);
                uint index=LTC_Index(proposal.coarse);
                if(!(index>=LT_GRID_CAPACITY || LTC_KeyLevel(g_sharc.Load4(LTC_KeyAddress(index)))>desired)) break;
                alternatives=0u;
            }
            LTC_RequestCell(x,n,proposal.fine,desired,alternatives);
        }

        refreshTicket=min((uint)(RandomFloatSingle(rng)*float(4u*LT_PARENT_FEEDBACK_RATE)),4u*LT_PARENT_FEEDBACK_RATE-1u);
        refreshTicket=WaveReadLaneFirst(refreshTicket);
        useCoarse=proposal.blend>0.0f && RandomFloatSingle(rng)<proposal.blend;
        cell=useCoarse?proposal.coarse:proposal.fine;
        uint count=g_sharc.Load(cell+4u);
        if(count!=0u && count<=LT_CUT_MAX) {
            float target=RandomFloatSingle(rng);
            uint lo=0u,hi=count-1u;
            [loop] while(lo<hi) {
                uint mid=(lo+hi)/2u;
                if(target<asfloat(g_sharc.Load(LTC_Cluster(cell,mid)+LT_FZ_CDF))) hi=mid;
                else lo=mid+1u;
            }
            chosen=lo;probability=LTC_FrozenProbability(LTC_Cluster(cell,chosen));
            LTC_Node c=LTC_LoadNode(LTC_Cluster(cell,chosen));
            node=c.node;start=c.slot;
        } else cell=LT_SENTINEL;
    }

    uint sampleSlot;
    LT_Sample sample=LT_SampleSubtree(x,n,rng,node,start,sampleSlot);
    if(cell!=LT_SENTINEL) {
        sample.pdf*=probability;
        sample.learningToken=uint2(sample.id==LT_SENTINEL?0u:LTC_Token(cell,chosen),0u);
    }
    if(!learned || sample.id==LT_SENTINEL) return sample;

    uint pdfCell=LT_SENTINEL;
    if(proposal.blend>0.0f) pdfCell=useCoarse?proposal.fine:proposal.coarse;
    else if(proposal.coarse!=proposal.fine) pdfCell=proposal.coarse;
    if(pdfCell!=LT_SENTINEL) {
        uint otherToken;
        float other=LTC_CellPdf(x,n,pdfCell,sample.id,sampleSlot,otherToken);
        if(proposal.blend>0.0f) {
            sample.pdf=useCoarse?lerp(other,sample.pdf,proposal.blend):lerp(sample.pdf,other,proposal.blend);
            sample.learningToken.y=useCoarse?sample.learningToken.x:otherToken;
            if(useCoarse) sample.learningToken.x=otherToken;
        } else {

            sample.learningToken.y=otherToken;
        }
    } else sample.learningToken.y=LTC_RefreshToken(x,n,proposal.fine,sample.id,sampleSlot,refreshTicket);
    return sample;
}

// Match learned sampling with its mixture PDF.
float LT_PdfSelectTriangle(float3 x,float3 n,uint tri,uint inst,bool useLearning=true) {
    const uint slot=LT_SlotOfInstance(inst);
    if(tri==LT_SENTINEL || slot==LT_SENTINEL || (rs_flags & RS_FLAG_NO_MESH_LIGHTS)!=0u) return 0;
    LTC_Proposal proposal=(LTC_Proposal)0;
    const bool learned=useLearning && LTC_GetProposal(x,n,proposal);

    float pdf=0.0f;
    [loop] for(uint pass=0u;pass<2u;++pass) {
        uint ignored;
        float p=LTC_CellPdf(x,n,!learned?LT_SENTINEL:(pass==0u?proposal.fine:proposal.coarse),tri,slot,ignored);
        if(pass==0u) {pdf=p;if(!learned || !(proposal.blend>0.0f)) break;}
        else pdf=lerp(pdf,p,proposal.blend);
    }
    return pdf;
}
uint LTC_BatchAddress(uint cluster) {
    uint index=LTC_ClusterIndex(cluster);
    return LT_BUFFER_OFFSET+LT_CELL_CAPACITY*LT_CELL_BYTES+index*LT_BATCH_BYTES;
}
void LTC_ClearBatch(uint cluster) {
    g_sharc.Store(LTC_Stats(cluster)+LT_ST_SELECTED,0u);
    uint address=LTC_BatchAddress(cluster);
    [unroll] for(uint i=0u;i<LT_BATCH_BYTES;i+=16u) g_sharc.Store4(address+i,0u);
}

void LTC_Accumulate(uint address,float value) {
    uint bits=asuint(value),exponent=(bits>>23u)&255u;
    uint mantissa=(bits&0x7fffffu)|(exponent!=0u?0x800000u:0u);
    if(mantissa==0u) return;
    uint shift=max(exponent,1u)-1u,word=shift>>6u,offset=shift&63u;
    uint64_t low=(uint64_t)mantissa<<offset;
    uint64_t high=offset==0u?0ull:(uint64_t)mantissa>>(64u-offset);
    [loop] for(uint i=word;i<LT_ACCUMULATOR_WORDS;++i) {
        uint64_t previous=0ull,carry=0ull;
        if(low!=0ull) {
            g_sharc.InterlockedAdd64(address+i*8u,low,previous);
            carry=previous+low<previous?1ull:0ull;
        }
        low=high+carry;high=0ull;
        if(low==0ull) break;
    }
}
float LTC_ReadAccumulator(uint address) {

    precise float low=(float)g_sharc.Load<uint64_t>(address)*exp2(-126.0f);
    precise float sum=low*exp2(-23.0f);
    [unroll] for(uint i=1u;i<LT_ACCUMULATOR_WORDS;++i)
        sum=min(sum+(float)g_sharc.Load<uint64_t>(address+i*8u)*exp2(float(int(i)*64-149)),3.0e38f);
    return sum;
}
float2 LTC_BatchMoments(uint cluster) {
    uint address=LTC_BatchAddress(cluster);
    return float2(LTC_ReadAccumulator(address),LTC_ReadAccumulator(address+LT_ACCUMULATOR_BYTES));
}
// Accumulate bounded reward moments for a selected cluster.
void LT_TrainToken(uint token,float contributionOverPdf) {
    if(token==0u || !LTC_Enabled()) return;
    uint a=LTC_TokenAddress(token);
    float p=LTC_FrozenProbability(a);
    float reward=max(0.0f,contributionOverPdf)*p;
    if(!isfinite(reward)) reward=0.0f;

    uint4 mask=WaveMatch(a);
    float2 observation=float2(reward,min(reward*reward,3.0e38f));
    float2 totals=min(WaveMultiPrefixSum(observation,mask)+observation,3.0e38f);
    uint selected=WaveMultiPrefixCountBits(true,mask)+1u;
    int4 highest=(int4)(firstbithigh(mask)|uint4(0,32,64,96));
    uint leader=(uint)max(max(highest.x,highest.y),max(highest.z,highest.w));
    if(WaveGetLaneIndex()==leader) {

        uint cell=LTC_Cell((token-1u)/LT_CUT_MAX),ignored;
        if(g_sharc.Load(cell+68u)!=sharc_frame)
            g_sharc.InterlockedExchange(cell+68u,sharc_frame,ignored);
        uint batch=LTC_BatchAddress(a);
        LTC_Accumulate(batch,totals.x);
        LTC_Accumulate(batch+LT_ACCUMULATOR_BYTES,totals.y);
        g_sharc.InterlockedAdd(LTC_Stats(a)+LT_ST_SELECTED,selected,ignored);
    }
}
void LT_TrainSample(uint2 tokens,float contributionOverPdf) {

    [loop] for(uint k=0u;k<2u;++k) LT_TrainToken(k==0u?tokens.x:tokens.y,contributionOverPdf);
}

void LT_Train(float3 x,float3 n,uint tri,uint inst,float contributionOverPdf) {
    const uint slot=LT_SlotOfInstance(inst);
    if(tri==LT_SENTINEL || slot==LT_SENTINEL) return;
    uint cell;if(!LTC_Find(x,n,cell)) return;
    uint count=g_sharc.Load(cell+4u);
    if(count==0u || count>LT_CUT_MAX) return;
    uint a=LTC_ClusterForTriangle(cell,count,tri,slot);if(a==LT_SENTINEL) return;
    uint token=LTC_ClusterIndex(a)+1u;
    LT_TrainToken(token,contributionOverPdf);
}
#endif
