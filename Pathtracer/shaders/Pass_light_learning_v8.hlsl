#define COMPUTE_PASS
#include "Includes_v8.hlsli"

void LTC_ClearCluster(uint address) {
    [unroll] for(uint i=0;i<LT_CLUSTER_BYTES;i+=16u) g_sharc.Store4(address+i,0u);
}
void LTC_Initialize(uint cell) {
    g_sharc.Store4(cell+16u,g_sharc.Load4(cell+80u));
    float3 x=asfloat(g_sharc.Load3(cell+96u));
    float3 n=asfloat(g_sharc.Load3(cell+112u));
    g_sharc.Store3(cell+32u,asuint(x));g_sharc.Store3(cell+48u,asuint(n));
    x-=LT_WORLD_ORIGIN;
    [loop] for(uint i=0;i<LT_CUT_MAX;++i) LTC_ClearCluster(LTC_Cluster(cell,i));
    LTC_StoreNode(LTC_Cluster(cell,0u),LTC_Root());
    uint count=1u;
    // Start with a coarse, power-prioritized cut; crossing a TLAS leaf may
    // immediately expose a BLAS subtree, including single-instance scenes.
    [loop] for(uint step=0;step<LT_CUT_INITIAL;++step) {
        if(count>=LT_CUT_INITIAL) break;
        uint best=LT_SENTINEL;float score=-1;
        [loop] for(uint j=0;j<count;++j) {
            LTC_Node c=LTC_LoadNode(LTC_Cluster(cell,j));uint first;
            uint children=LTC_Children(c,first);
            if(children==0u || c.depth>=LT_TRAIL_MAX_DEPTH || count+children-1u>LT_CUT_MAX) continue;
            float p=c.blas==LT_SENTINEL?gLT_TLAS[c.node].power:gLT_BLAS[gLT_Range[c.blas].nodeOffset+c.node].power;
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
        uint a=LTC_Cluster(cell,j);float q=LTC_Prior(LTC_LoadNode(a),x,n);
        g_sharc.Store(a+20u,asuint(q));g_sharc.Store(a+36u,asuint(q));
    }
    g_sharc.Store4(cell,uint4(1u,count,0u,sharc_frame));
    g_sharc.Store(cell+44u,count);g_sharc.Store(cell+60u,0u);
}
void LTC_Update(uint cell,uint cellSlot) {
    uint count=g_sharc.Load(cell+4u),samples=0u;
    if(count==0u || count>LT_CUT_MAX) return;
    [loop] for(uint j=0;j<count;++j) samples+=g_sharc.Load(LTC_Cluster(cell,j)+56u);
    if(samples==0u) return;
    g_sharc.Store(cell+12u,sharc_frame);
    uint initial=max(g_sharc.Load(cell+44u),1u);
    uint budget=4u*max((count+initial-1u)/initial,2u);
    if(samples<budget) return; // retain the batch across frames
    uint iteration=g_sharc.Load(cell+8u)+1u;
    float alpha=0.25f*pow(float(iteration),-6.0f/7.0f);
    float oldSum=LTC_Sum(cell,count);
    float totalVariance=0;
    [loop] for(uint j=0;j<count;++j) {
        uint a=LTC_Cluster(cell,j);
        uint4 batch=g_sharc.Load4(a+48u);
        float sum=asfloat(batch.x),sum2=asfloat(batch.y);
        uint selected=batch.z;
        float probability=LTC_Probability(a,count,oldSum);
        // Includes implicit zeros for every sample that chose another cluster.
        // E[sum(F / p(light|cluster)) / (N*p(cluster))] = cluster contribution.
        float target=sum/(float(samples)*probability);
        float q=lerp(LTC_Weight(a),min(target,3.0e30f),alpha);
        g_sharc.Store(a+20u,asuint(max(0.0f,q)));
        float mean=asfloat(g_sharc.Load(a+24u)),m2=asfloat(g_sharc.Load(a+28u));
        uint visits=g_sharc.Load(a+32u);
        if(selected>0u) {
            float batchMean=sum/float(selected);
            float batchM2=max(0.0f,sum2-sum*batchMean);
            float delta=batchMean-mean;
            float total=float(visits)+float(selected);
            m2+=batchM2+delta*delta*(float(visits)/total)*float(selected);
            mean+=delta*(float(selected)/total);
            visits+=selected;
            g_sharc.Store3(a+24u,uint3(asuint(mean),asuint(min(m2,3.0e38f)),visits));
        }
        totalVariance+=visits>1u?m2/float(visits-1u):0;
        g_sharc.Store4(a+48u,0u);
    }
    g_sharc.Store(cell+8u,iteration);
    if(count>=LT_CUT_MAX || iteration-g_sharc.Load(cell+60u)>128u*count) return;
    float3 x=asfloat(g_sharc.Load3(cell+32u))-LT_WORLD_ORIGIN;
    float3 n=asfloat(g_sharc.Load3(cell+48u));
    // One stochastic refinement per batch bounds update work. Generalization
    // of Eq. 7 to this renderer's up-to-four-way nodes.
    [loop] for(uint j=0;j<count;++j) {
        uint a=LTC_Cluster(cell,j);LTC_Node parent=LTC_LoadNode(a);uint first;
        uint children=LTC_Children(parent,first);
        uint visits=g_sharc.Load(a+32u);
        if(children==0u || parent.depth>=LT_TRAIL_MAX_DEPTH || visits<=1u || count+children-1u>LT_CUT_MAX) continue;
        float variance=asfloat(g_sharc.Load(a+28u))/float(visits-1u);
        float split=(1.0f/(1.0f+float(count)/float(initial)*exp(-min(variance,80.0f))))
            *(variance/max(totalVariance,1e-20f))*(1.0f-1.0f/float(visits));
        uint random=LTC_Hash(cellSlot^LTC_Hash(iteration)^LTC_Hash(parent.node+parent.blas));
        if(float(random>>8u)*(1.0f/16777216.0f)>=split) continue;
        float priors[4];float priorSum=0;
        [unroll] for(uint k=0;k<4u;++k) { priors[k]=k<children?LTC_Prior(LTC_Child(parent,first,k),x,n):0;priorSum+=priors[k]; }
        float parentQ=LTC_Weight(a);
        [unroll] for(uint k=0;k<4u;++k) if(k<children) {
            uint destination=LTC_Cluster(cell,k==0u?j:count+k-1u);
            float ratio=priorSum>0?priors[k]/priorSum:1.0f/float(children);
            // Eq. 8: initialize from the prior and the learned parent; child
            // observation/variance histories restart at zero after splitting.
            float A=pow(1.0f-alpha,ratio*float(visits));
            float q=A*priors[k]+(1.0f-A)*parentQ;
            LTC_ClearCluster(destination);LTC_StoreNode(destination,LTC_Child(parent,first,k));
            g_sharc.Store(destination+20u,asuint(q));g_sharc.Store(destination+36u,asuint(priors[k]));
        }
        g_sharc.Store(cell+4u,count+children-1u);g_sharc.Store(cell+60u,iteration);
        break;
    }
}
[numthreads(64,1,1)]
void main(uint3 tid:SV_DispatchThreadID) {
    if(tid.x>=LT_CELL_CAPACITY) return;
    uint cell=LTC_Cell(tid.x);
    if((sharc_reset & LT_RESET_BIT)!=0u) {
        [loop] for(uint b=0;b<LT_CELL_BYTES;b+=16u) g_sharc.Store4(cell+b,0u);
        return;
    }
    bool active=g_sharc.Load(cell)==1u;
    if(active) LTC_Update(cell,tid.x);
    if(g_sharc.Load(cell+64u)!=0u && (!active || sharc_frame-g_sharc.Load(cell+12u)>256u)) LTC_Initialize(cell);
    g_sharc.Store(cell+64u,0u);
}
