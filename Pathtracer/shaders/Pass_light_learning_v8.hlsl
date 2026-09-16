#define COMPUTE_PASS
#include "Includes_v8.hlsli"

void LTC_ClearCluster(uint address) {
    uint stats=LTC_Stats(address);
    [unroll] for(uint i=0;i<LT_FROZEN_BYTES;i+=16u) g_sharc.Store4(address+i,0u);
    [unroll] for(uint j=0;j<LT_STATS_BYTES;j+=16u) g_sharc.Store4(stats+j,0u);
    LTC_ClearBatch(address);
}
void LTC_CopyCluster(uint destination,uint source) {
    uint destinationStats=LTC_Stats(destination),sourceStats=LTC_Stats(source);
    [unroll] for(uint i=0u;i<LT_FROZEN_BYTES;i+=16u) g_sharc.Store4(destination+i,g_sharc.Load4(source+i));
    [unroll] for(uint j=0u;j<LT_STATS_BYTES;j+=16u) g_sharc.Store4(destinationStats+j,g_sharc.Load4(sourceStats+j));
}

LTC_Node LTC_ParentNode(LTC_Node child) {
    LTC_Node parent=LTC_Root();parent.slot=child.slot;
    [loop] for(uint d=0u;d+1u<child.depth;++d) {
        uint first=parent.slot==LT_SENTINEL?LT_LoadTLAS(parent.node).firstChild:LT_LoadBLAS(gLT_Slot[parent.slot].nodeOffset,parent.node).firstChild;
        uint digit=(d<16u?child.trail.x>>(2u*d):child.trail.y>>(2u*(d-16u)))&3u;
        parent=LTC_Child(parent,first,digit);
    }
    if(parent.slot!=LT_SENTINEL && parent.depth==0u) {

        uint2 trail=gLT_BLASBitTrail[parent.slot];parent=LTC_Root();
        [loop] for(uint d=0u;d<LT_TRAIL_MAX_DEPTH;++d) {
            LightTLASNodeGpu node=LT_LoadTLAS(parent.node);if(node.childCount==0u) break;
            uint digit=(d<16u?trail.x>>(2u*d):trail.y>>(2u*(d-16u)))&3u;
            parent=LTC_Child(parent,node.firstChild,digit);
        }
    }
    return parent;
}
// Merge the weakest neighboring cut entries before splitting.
bool LTC_MergeForSplit(uint cell,inout uint count,inout uint target,float targetQ) {
    uint best=LT_SENTINEL,bestSize=0u;float bestQ=0.5f*targetQ;

    [loop] for(uint j=0u;j<count;++j) {
        LTC_Node child=LTC_LoadNode(LTC_Cluster(cell,j));
        if(child.parent==LT_SENTINEL) continue;
        uint first,children;
        if(child.slot==LT_SENTINEL) {
            LightTLASNodeGpu p=LT_LoadTLAS(child.parent);first=p.firstChild;children=p.childCount;
        } else {
            LightBLASNodeGpu p=LT_LoadBLAS(gLT_Slot[child.slot].nodeOffset,child.parent);first=p.firstChild;children=p.childCount;
        }
        if(child.node!=first || children<2u || j+children>count || (target>=j && target<j+children)) continue;
        bool complete=true;float q=0.0f;
        [unroll] for(uint k=0u;k<4u;++k) if(k<children) {
            uint a=LTC_Cluster(cell,j+k);LTC_Node sibling=LTC_LoadNode(a);
            complete=complete && sibling.slot==child.slot && sibling.parent==child.parent && sibling.node==first+k;
            q+=LTC_Weight(a);
        }
        if(complete && q<=bestQ) {best=j;bestSize=children;bestQ=q;}
    }
    if(best==LT_SENTINEL) return false;
    uint a=LTC_Cluster(cell,best);LTC_Node parent=LTC_ParentNode(LTC_LoadNode(a));
    LTC_ClearCluster(a);LTC_StoreNode(a,parent);g_sharc.Store(LTC_Stats(a)+LT_ST_Q,asuint(bestQ));
    [loop] for(uint j=best+1u;j+bestSize-1u<count;++j)
        LTC_CopyCluster(LTC_Cluster(cell,j),LTC_Cluster(cell,j+bestSize-1u));
    count-=bestSize-1u;if(target>best) target-=bestSize-1u;
    g_sharc.Store(cell+4u,count);
    return true;
}
// Sort frozen clusters by descending learned mass.
void LTC_SortCut(uint cell) {
    uint count=g_sharc.Load(cell+4u);
    [loop] for(uint i=1;i<count;++i) {
        uint from=LTC_Cluster(cell,i),fromStats=LTC_Stats(from);uint4 key=LTC_OrderKey(LTC_LoadNode(from));
        uint4 a=g_sharc.Load4(from),b=g_sharc.Load4(from+16u),c=g_sharc.Load4(fromStats),d=g_sharc.Load4(fromStats+16u);
        uint j=i;
        [loop] while(j>0u) {
            uint prev=LTC_Cluster(cell,j-1u);
            if(LTC_OrderLE(LTC_OrderKey(LTC_LoadNode(prev)),key)) break;
            LTC_CopyCluster(LTC_Cluster(cell,j),prev);--j;
        }
        uint dest=LTC_Cluster(cell,j),destStats=LTC_Stats(dest);
        g_sharc.Store4(dest,a);g_sharc.Store4(dest+16u,b);g_sharc.Store4(destStats,c);g_sharc.Store4(destStats+16u,d);
    }
}
// Publish CDF and probabilities after a complete cut update.
void LTC_PublishDistribution(uint cell) {
    uint count=g_sharc.Load(cell+4u);float sum=LTC_Sum(cell,count),cdf=0.0f;
    float powerSum=0.0f;
    [loop] for(uint j=0u;j<count;++j) powerSum+=asfloat(g_sharc.Load(LTC_Stats(LTC_Cluster(cell,j))+LT_ST_POWER));
    [loop] for(uint j=0;j<count;++j) {
        uint a=LTC_Cluster(cell,j);
        float next=j+1u==count?1.0f:min(cdf+LTC_Probability(a,count,sum,powerSum),1.0f);

        g_sharc.Store2(a+LT_FZ_PROBABILITY,asuint(float2(next-cdf,next)));cdf=next;
    }
}
void LTC_InitializeRoot(uint cell,uint face) {
    uint4 key=uint4(0,0,0,((LT_MAX_LEVEL+2u)<<9u)|face);
    g_sharc.Store(cell+16u,0u);
    float3 x=LT_WORLD_ORIGIN,n=LTC_FaceNormal(face);
    g_sharc.Store3(cell+32u,asuint(x));g_sharc.Store3(cell+48u,asuint(n));
    x-=LT_WORLD_ORIGIN;
    [loop] for(uint i=0;i<LT_CUT_MAX;++i) LTC_ClearCluster(LTC_Cluster(cell,i));
    LTC_StoreNode(LTC_Cluster(cell,0u),LTC_Root());
    uint count=1u;

    [loop] for(uint step=0;step<LT_CUT_INITIAL;++step) {
        if(count>=LT_CUT_INITIAL) break;
        uint best=LT_SENTINEL;float score=-1;
        [loop] for(uint j=0;j<count;++j) {
            LTC_Node c=LTC_LoadNode(LTC_Cluster(cell,j));uint first;
            uint children=LTC_Children(c,first);
            if(children==0u || c.depth>=LT_TRAIL_MAX_DEPTH || count+children-1u>LT_CUT_MAX) continue;
            float p=LTC_NodePower(c);
            if(p>score) { score=p;best=j; }
        }
        if(best==LT_SENTINEL) break;
        LTC_Node parent=LTC_LoadNode(LTC_Cluster(cell,best));uint first;
        uint children=LTC_Children(parent,first);
        [unroll] for(uint j=0;j<4u;++j) if(j<children) {
            uint destination=j==0u?best:count+j-1u;
            LTC_StoreNode(LTC_Cluster(cell,destination),LTC_Child(parent,first,j));
        }
        count+=children-1u;
    }
    [loop] for(uint j=0;j<count;++j) {
        uint a=LTC_Cluster(cell,j);LTC_Node c=LTC_LoadNode(a);
        float q=LTC_NodePower(c);
        g_sharc.Store(LTC_Stats(a)+LT_ST_Q,asuint(q));
    }
    g_sharc.Store4(cell,uint4(1u,count,0u,sharc_frame));
    g_sharc.Store(cell+44u,count);g_sharc.Store(cell+60u,0u);
    g_sharc.Store(cell+72u,sharc_frame);
    g_sharc.Store(cell+24u,LTC_Now());
    LTC_SortCut(cell);
    LTC_PublishDistribution(cell);
    g_sharc.Store4(LTC_KeyAddress(LTC_Index(cell)),key);
}
void LTC_Initialize(uint cell) {
    uint parent=g_sharc.Load(cell+76u);
    if(parent>=LT_CELL_CAPACITY || g_sharc.Load(LTC_Cell(parent))!=1u) return;
    uint source=LTC_Cell(parent),count=g_sharc.Load(source+4u);
    if(count==0u || count>LT_CUT_MAX) return;

    [loop] for(uint j=0;j<LT_CUT_MAX;++j) {
        uint a=LTC_Cluster(cell,j);LTC_ClearCluster(a);
        if(j<count) {
            uint p=LTC_Cluster(source,j);LTC_StoreNode(a,LTC_LoadNode(p));
            float q=LTC_Weight(p);g_sharc.Store(LTC_Stats(a)+LT_ST_Q,asuint(q));
        }
    }

    uint4 key=g_sharc.Load4(cell+80u);g_sharc.Store(cell+16u,0u);
    g_sharc.Store3(cell+32u,g_sharc.Load3(cell+96u));g_sharc.Store3(cell+48u,g_sharc.Load3(cell+112u));
    g_sharc.Store4(cell,uint4(1u,count,0u,sharc_frame));
    g_sharc.Store(cell+44u,count);g_sharc.Store(cell+60u,0u);g_sharc.Store(cell+68u,0u);g_sharc.Store(cell+72u,sharc_frame);
    g_sharc.Store(cell+20u,0u);g_sharc.Store(cell+24u,LTC_Now());
    LTC_PublishDistribution(cell);g_sharc.Store4(LTC_KeyAddress(LTC_Index(cell)),key);
}
// Consume accumulated moments and refine one adaptive cell.
bool LTC_Update(uint cell,uint cellSlot) {
    uint count=g_sharc.Load(cell+4u),samples=0u;
    if(count==0u || count>LT_CUT_MAX) return false;
    float estimatedTotal=0.0f;
    bool calibrating=g_sharc.Load(cell+16u)==0u;
    [loop] for(uint j=0;j<count;++j) {
        uint a=LTC_Cluster(cell,j),selected=g_sharc.Load(LTC_Stats(a)+LT_ST_SELECTED);samples+=selected;
        if(calibrating && selected!=0u)
            estimatedTotal+=min(LTC_ReadAccumulator(LTC_BatchAddress(a))/max(LTC_FrozenProbability(a),1e-20f),1e30f);
    }
    if(samples==0u) return false;
    uint initial=max(g_sharc.Load(cell+44u),1u);
    uint budget=4u*max((count+initial-1u)/initial,2u);
    if(samples<budget) return false;
    uint iteration=g_sharc.Load(cell+8u)+1u;
    float alpha=max(LT_LEARNING_ALPHA_MIN,0.25f*pow(float(iteration),-6.0f/7.0f));
    float priorScale=1.0f;
    if(calibrating && estimatedTotal>0.0f) {

        float priorSum=LTC_Sum(cell,count);
        if(priorSum>0.0f) priorScale=(estimatedTotal/float(samples))/priorSum;
        g_sharc.Store(cell+16u,1u);
    }
    float totalVariance=0;
    [loop] for(uint j=0;j<count;++j) {
        uint a=LTC_Cluster(cell,j),stats=LTC_Stats(a);
        uint selected=g_sharc.Load(stats+LT_ST_SELECTED);
        float2 batch=selected!=0u?LTC_BatchMoments(a):0.0f;
        float sum=batch.x,sum2=batch.y;
        float probability=LTC_FrozenProbability(a);

        float target=sum/(float(samples)*probability);
        float q=lerp(LTC_Weight(a)*priorScale,min(target,3.0e30f),alpha);
        g_sharc.Store(stats+LT_ST_Q,asuint(max(0.0f,q)));
        float mean=asfloat(g_sharc.Load(stats+LT_ST_MEAN)),m2=asfloat(g_sharc.Load(stats+LT_ST_M2));
        uint visits=g_sharc.Load(stats+LT_ST_VISITS);

        uint history=visits==0u?0u:max(1u,(uint)(float(min(visits,LT_MOMENT_HISTORY))*(1.0f-alpha)));
        m2*=visits>1u?float(history>0u?history-1u:0u)/float(visits-1u):0.0f;
        visits=history;
        if(selected>0u) {
            float batchMean=sum/float(selected);
            float batchM2=max(0.0f,sum2-sum*batchMean);
            float delta=batchMean-mean;
            float total=float(visits)+float(selected);
            float cross=min(delta*delta,3.0e38f)*(float(visits)/total);
            m2=min(m2+batchM2+min(cross*float(selected),3.0e38f),3.0e38f);
            mean+=delta*(float(selected)/total);

            visits=(uint)min(total,float(LT_MOMENT_HISTORY));
            m2=min(m2,3.0e38f)*float(visits-1u)/max(total-1.0f,1.0f);
        }
        g_sharc.Store3(stats+LT_ST_MEAN,uint3(asuint(mean),asuint(m2),visits));
        totalVariance=min(totalVariance+(visits>1u?m2/float(visits-1u):0.0f),3.0e38f);
        if(selected!=0u) LTC_ClearBatch(a);
    }
    g_sharc.Store(cell+8u,iteration);
    float3 x=asfloat(g_sharc.Load3(cell+32u))-LT_WORLD_ORIGIN;
    float3 n=asfloat(g_sharc.Load3(cell+48u));

    [loop] for(uint j=0;j<count;++j) {
        uint a=LTC_Cluster(cell,j);LTC_Node parent=LTC_LoadNode(a);uint first;
        uint children=LTC_Children(parent,first);
        uint visits=g_sharc.Load(LTC_Stats(a)+LT_ST_VISITS);
        if(children==0u || parent.depth>=LT_TRAIL_MAX_DEPTH || visits<=1u) continue;
        float variance=asfloat(g_sharc.Load(LTC_Stats(a)+LT_ST_M2))/float(visits-1u);
        float split=(1.0f/(1.0f+float(count)/float(initial)*exp(-min(variance,80.0f))))
            *(variance/max(totalVariance,1e-20f))*(1.0f-1.0f/float(visits));
        uint random=LTC_Hash(cellSlot^LTC_Hash(iteration)^LTC_Hash(parent.node+parent.slot));
        if(float(random>>8u)*(1.0f/16777216.0f)>=split) continue;
        float parentQ=LTC_Weight(a);

        [loop] for(uint merge=0u;merge<3u && count+children-1u>LT_CUT_MAX;++merge)
            if(!LTC_MergeForSplit(cell,count,j,parentQ)) break;
        if(count+children-1u>LT_CUT_MAX) break;
        float priors[4];float priorSum=0;
        [unroll] for(uint k=0;k<4u;++k) { priors[k]=k<children?LTC_Prior(LTC_Child(parent,first,k),x,n):0;priorSum+=priors[k]; }
        [unroll] for(uint k=0;k<4u;++k) if(k<children) {
            uint destination=LTC_Cluster(cell,k==0u?j:count+k-1u);
            float ratio=priorSum>0?priors[k]/priorSum:1.0f/float(children);

            float A=pow(1.0f-alpha,ratio*float(visits));
            float calibratedPrior=ratio*parentQ;
            float q=A*calibratedPrior+(1.0f-A)*parentQ;
            LTC_ClearCluster(destination);LTC_StoreNode(destination,LTC_Child(parent,first,k));
            g_sharc.Store(LTC_Stats(destination)+LT_ST_Q,asuint(q));
        }
        g_sharc.Store(cell+4u,count+children-1u);g_sharc.Store(cell+60u,iteration);
        LTC_SortCut(cell);
        break;
    }
    return true;
}

bool LTC_CutValid(uint cell) {
    uint count=g_sharc.Load(cell+4u);
    if(count==0u || count>LT_CUT_MAX) return false;
    [loop] for(uint j=0u;j<count;++j) {
        LTC_Node c=LTC_LoadNode(LTC_Cluster(cell,j));
        uint node=0u,parent=LT_SENTINEL;
        if(c.slot==LT_SENTINEL) {
            [loop] for(uint d=0u;d<c.depth;++d) {
                LightTLASNodeGpu t=LT_LoadTLAS(node);
                uint digit=(d<16u?c.trail.x>>(2u*d):c.trail.y>>(2u*(d-16u)))&3u;
                if(digit>=t.childCount) return false;
                parent=node;node=t.firstChild+digit;
            }
            if(node!=c.node || parent!=c.parent) return false;
        } else {
            uint2 trail=gLT_BLASBitTrail[c.slot];
            [loop] for(uint d=0u;d<LT_TRAIL_MAX_DEPTH;++d) {
                LightTLASNodeGpu t=LT_LoadTLAS(node);
                if(t.childCount==0u) break;
                uint digit=(d<16u?trail.x>>(2u*d):trail.y>>(2u*(d-16u)))&3u;
                if(digit>=t.childCount) return false;
                node=t.firstChild+digit;
            }
            if(LT_LoadTLAS(node).slot!=c.slot) return false;
        }
    }
    return true;
}
[numthreads(64,1,1)]
void main(uint3 tid:SV_DispatchThreadID) {
    if(tid.x>=LT_CELL_CAPACITY) return;
    uint cell=LTC_Cell(tid.x);
    if((sharc_reset & LT_RESET_BIT)!=0u) {

        [unroll] for(uint b=0;b<LT_CELL_HEADER;b+=16u) g_sharc.Store4(cell+b,0u);
        g_sharc.Store4(LTC_KeyAddress(tid.x),0u);
        if(tid.x>=LT_GRID_CAPACITY) LTC_InitializeRoot(cell,tid.x-LT_GRID_CAPACITY);
        return;
    }
    if((sharc_reset & LT_REVALIDATE_BIT)!=0u && (sharc_reset & LT_INITIALIZE_BIT)==0u && g_sharc.Load(cell)==1u) {
        if(!LTC_CutValid(cell)) {
            [unroll] for(uint b=0;b<LT_CELL_HEADER;b+=16u) g_sharc.Store4(cell+b,0u);
            g_sharc.Store4(LTC_KeyAddress(tid.x),0u);
            if(tid.x>=LT_GRID_CAPACITY) LTC_InitializeRoot(cell,tid.x-LT_GRID_CAPACITY);
            return;
        }

        LTC_SortCut(cell);
    }
    if((sharc_reset & LT_INITIALIZE_BIT)!=0u) {

        if(g_sharc.Load(cell+64u)!=0u && LTC_Replaceable(cell,true)) LTC_Initialize(cell);
        g_sharc.Store(cell+64u,0u);
    } else if(g_sharc.Load(cell)==1u) {
        LTC_UpdateRetention(cell);
        if(LTC_Update(cell,tid.x)) LTC_PublishDistribution(cell);
    }
}
