#include "Includes_v8.hlsli"
cbuffer LightTreeTestConstants : register(b0, space1) {
    uint workCount,testMode,triangleCount,testSeed;
    float3 testCameraPosition;float testRewardScale;
}
RWStructuredBuffer<float4> results : register(u0, space1);
StructuredBuffer<float4> testReceivers : register(t19);
StructuredBuffer<LightBLASNodeFull> unpackedLightNodes : register(t20);
uint TestInstOf(uint tri) { return g_EmissiveTriangles[tri].meshID; }
uint TestSlotOf(uint tri) { return LT_SlotOfInstance(TestInstOf(tri)); }
#define LT_TEST_Q(a) (LTC_Stats(a)+LT_ST_Q)
#define LT_TEST_MOMENTS(a) (LTC_Stats(a)+LT_ST_MEAN)
#define LT_TEST_VISITS(a) (LTC_Stats(a)+LT_ST_VISITS)
#define LT_TEST_SELECTED(a) (LTC_Stats(a)+LT_ST_SELECTED)
#define LT_TEST_POWER(a) (LTC_Stats(a)+LT_ST_POWER)
#define LT_TEST_PROBABILITY(a) ((a)+LT_FZ_PROBABILITY)
void LT_TEST_ClearRecord(uint a) {
    for(uint b=0u;b<LT_FROZEN_BYTES;b+=16u) g_sharc.Store4(a+b,0u);
    for(uint c=0u;c<LT_STATS_BYTES;c+=16u) g_sharc.Store4(LTC_Stats(a)+c,0u);
}
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if(tid.x>=workCount) return;
    if(testMode==47u || testMode==48u) {
        LightBLASNodeGpu expected=LT_DecodeFullBLAS(unpackedLightNodes[tid.x]);
        LightBLASNodeGpu actual;
        float unit=0.0f;
        if(testMode==47u) {actual=LT_LoadBLAS(testSeed,tid.x);unit=LT_LoadBlasFrame(testSeed).diag;}
        else {
            LightTLASNodeGpu t=LT_LoadTLAS(tid.x);
            actual.mean=t.mean;actual.variance=t.variance;actual.axis=t.axis;actual.kappa=t.kappa;
            actual.power=t.power;actual.radius=t.radius;actual.cosTheta_o=t.cosTheta_o;
            actual.firstChild=t.firstChild;actual.childCount=t.childCount;actual.triFirst=t.slot;actual.triCount=1u;
        }
        // The mean keeps FP32 in both layouts; a packed mesh widens its extent by at most the half
        // rounding (relative, or one subnormal step of the mesh unit), never narrows it.
        const float slack=4e-7f*unit;
        bool extent=all(asuint(actual.mean)==asuint(expected.mean)) &&
            actual.radius>=expected.radius && actual.radius<=expected.radius*1.002f+slack &&
            actual.variance>=expected.variance*0.999999f &&
            sqrt(actual.variance)<=sqrt(expected.variance)*1.002f+slack;
        bool topology=actual.childCount==expected.childCount && (actual.childCount?
            actual.firstChild==expected.firstChild:actual.triFirst==expected.triFirst);
        float delta=acos(clamp(dot(actual.axis,expected.axis),-1.0f,1.0f));
        bool cone=actual.cosTheta_o==-1.0f || delta+acos(clamp(expected.cosTheta_o,-1.0f,1.0f))<=acos(actual.cosTheta_o)+1e-6f;
        // The lobe sharpness rounds towards the blurrier side; the axis keeps the octahedral precision.
        bool lobe=actual.kappa<=expected.kappa*1.0001f+1e-6f && actual.kappa>=expected.kappa*0.99f-1e-3f &&
            (expected.kappa<1e-3f || delta<=2e-3f);
        if ((rs_flags & RS_FLAG_COMPACT_LIGHT_TREE) == 0u) {
            extent=all(asuint(actual.mean)==asuint(expected.mean)) && asuint(actual.radius)==asuint(expected.radius) &&
                asuint(actual.variance)==asuint(expected.variance);
            cone=asuint(actual.cosTheta_o)==asuint(expected.cosTheta_o);
            lobe=all(asuint(actual.axis)==asuint(expected.axis)) && asuint(actual.kappa)==asuint(expected.kappa);
        }
        results[tid.x]=float4(extent,asuint(actual.power)==asuint(expected.power),topology,cone && lobe);return;
    }
    if(testMode==49u) {
        // Node importance for a receiver and cluster given as records: [0] x, wd; [1] n, wg;
        // [2] view, ax; [3] tangent, ay; [4] mean, variance; [5] axis, resultant length;
        // [6] power, radius, cone cosine.
        const uint base=tid.x*7u;
        const float4 r0=testReceivers[base],r1=testReceivers[base+1u],r2=testReceivers[base+2u],r3=testReceivers[base+3u];
        const float4 c0=testReceivers[base+4u],c1=testReceivers[base+5u],c2=testReceivers[base+6u];
        const LT_Receiver R=LT_MakeReceiver(r0.xyz,normalize(r1.xyz),normalize(r2.xyz),r3.xyz,r0.w,r1.w,r2.w,r3.w);
        results[tid.x]=float4(LT_NodeImportance(R,c0.xyz,c0.w,normalize(c1.xyz),LT_KappaOf(c1.w),c2.x,c2.y,c2.z),0,0,0);
        return;
    }
    if(testMode==45u || testMode==46u) {
        float4 receiver=testReceivers[0],coat=testReceivers[1];
        float3 x=receiver.xyz,n=float3(0,0,1);
        // A glossy receiver seen from a slant, through the packed form the passes use. Roughness
        // one is a diffuse receiver; a clearcoat merges its roughness into the glossy lobe.
        const float rough=max(receiver.w,0.001f),alpha=rough*rough;
        float a2=alpha*alpha,wg=rough<1.0f?0.6f:0.0f,wd=0.4f;
        if(coat.x>0.0f) {const float ac=max(coat.y*coat.y,0.001f);a2=0.5f*(a2+ac*ac);wg+=0.3f*coat.x;}
        const float3 view=normalize(float3(0.6f,0.1f,0.8f));
        const LT_Receiver R=LT_UnpackReceiver(LT_PackReceiver(view,float3(1,0,0),wd,wg,sqrt(a2),sqrt(a2)),x,n);
        uint pathSeed=LTC_Hash(tid.x^testSeed),rng=RcBounceSeed(pathSeed,1u,RC_STREAM_NEE);
        uint bsdfRng=RcBounceSeed(pathSeed,1u,RC_STREAM_BSDF);
        const bool learned=LTC_UseSurfaceLearning();
        LT_Sample s=LT_SampleLight(R,rng,learned);
        float evaluated=LT_PdfSelectTriangle(R,s.id,s.inst,learned);
        float estimate=s.id==0u?1.0f/(s.pdf+0.75f):0.0f;
        if(RandomFloatSingle(bsdfRng)<0.75f) {
            float q=LT_PdfSelectTriangle(R,0u,TestInstOf(0u),testMode==46u?false:learned);
            estimate+=1.0f/(q+0.75f);
        }
        bool clean=learned || (all(s.learningToken==0u) &&
            abs(s.pdf-LT_PdfSubtree(R,s.id,LT_SlotOfInstance(s.inst)))<=s.pdf*3e-5f);
        results[tid.x]=float4(s.pdf,evaluated,estimate,clean?(learned?1:0):-1);return;
    }
    if(testMode==44u) {
        uint requests=0u;
        uint4 key=LTC_KeyAtLevel(testReceivers[0].xyz,float3(0,0,1),0u);
        for(uint i=1u;i<=LT_CELL_PROBES;++i) {
            uint cell=LTC_Cell((uint)testReceivers[i].w);
            requests+=g_sharc.Load(cell+64u)!=0u && all(g_sharc.Load4(cell+80u)==key)?1u:0u;
        }
        results[tid.x]=float4(requests,0,0,0);return;
    }
    if(testMode==39u || testMode==40u || testMode==43u) {
        bool valid=true;
        for(uint i=1u;i<=LT_CELL_PROBES;++i) {
            float4 receiver=testReceivers[i];uint cell=LTC_Cell((uint)receiver.w);
            uint4 key=LTC_KeyAtLevel(receiver.xyz,float3(0,0,1),testSeed);
            if(testMode==43u) {
                g_sharc.Store(cell,0u);g_sharc.Store4(LTC_KeyAddress((uint)receiver.w),0u);continue;
            }
            if(testMode==39u) {
                for(uint b=0u;b<LT_CELL_HEADER;b+=16u) g_sharc.Store4(cell+b,0u);
                uint a=LTC_Cluster(cell,0u);
                LT_TEST_ClearRecord(a);
                LTC_ClearBatch(a);LTC_StoreNode(a,LTC_Root());
                g_sharc.Store(LT_TEST_Q(a),asuint(1.0f));g_sharc.Store2(LT_TEST_PROBABILITY(a),asuint(float2(1,1)));
                g_sharc.Store4(cell,uint4(1u,1u,1u,sharc_frame));
                g_sharc.Store(cell+44u,1u);g_sharc.Store(cell+24u,LTC_Now());
                g_sharc.Store3(cell+32u,asuint(receiver.xyz));g_sharc.Store3(cell+48u,asuint(float3(0,0,1)));
                g_sharc.Store4(LTC_KeyAddress((uint)receiver.w),key);
            }
            valid=valid && LTC_Level(receiver.xyz)==testSeed && g_sharc.Load(cell)==1u &&
                all(g_sharc.Load4(LTC_KeyAddress((uint)receiver.w))==key);
            LT_Train(receiver.xyz,float3(0,0,1),0u,TestInstOf(0u),0.0f);
        }
        results[tid.x]=float4(valid?1:0,0,0,0);return;
    }
    if(testMode==41u || testMode==42u) {
        float3 x=testReceivers[0].xyz,n=float3(0,0,1);uint cell;
        LTC_Find(x,n,cell);uint count=g_sharc.Load(cell+4u);
        if(testMode==41u) {
            for(uint j=0u;j<count;++j) {
                uint a=LTC_Cluster(cell,j);
                g_sharc.Store(LT_TEST_Q(a),asuint(LTC_Contains(LTC_LoadNode(a),0u,TestSlotOf(0u))?1e20f:1e18f));
            }
            g_sharc.Store(cell+28u,1u);
            float sum=LTC_Sum(cell,count),powerSum=0,cdf=0;
            for(uint j=0u;j<count;++j) powerSum+=asfloat(g_sharc.Load(LT_TEST_POWER(LTC_Cluster(cell,j))));
            for(uint j=0u;j<count;++j) {
                uint a=LTC_Cluster(cell,j);
                float next=j+1u==count?1.0f:cdf+LTC_Probability(a,count,sum,powerSum);
                g_sharc.Store2(LT_TEST_PROBABILITY(a),asuint(float2(next-cdf,next)));cdf=next;
            }
        }
        results[tid.x]=float4(LTC_Sum(cell,count),LTC_Index(cell)>=LT_GRID_CAPACITY?25u:
            LTC_KeyLevel(g_sharc.Load4(LTC_KeyAddress(LTC_Index(cell)))),g_sharc.Load(cell+8u),g_sharc.Load(cell+28u));return;
    }
    if(testMode==38u) {
        float3 x=testReceivers[0].xyz,n=float3(0,0,1);
        for(uint level=1u;level<=3u;++level) {
            uint cell;
            if(LTC_FindExact(LTC_KeyAtLevel(x,n,level),cell)) {
                g_sharc.Store(cell,0u);g_sharc.Store4(LTC_KeyAddress(LTC_Index(cell)),0u);
            }
        }
        results[tid.x]=0;return;
    }
    if(testMode==36u || testMode==37u) {
        float3 x=testReceivers[0].xyz,n=float3(0,0,1);uint cell;
        if(!LTC_Find(x,n,cell)) {results[tid.x]=-1;return;}
        uint count=g_sharc.Load(cell+4u),matches=0u;bool valid=true;
        for(uint j=0u;j<count;++j) {
            uint a=LTC_Cluster(cell,j);
            if(testMode==37u) {
                g_sharc.Store3(LT_TEST_MOMENTS(a),uint3(asuint(1.0f),asuint(4.0e9f),0xfffffff0u));
            }
            matches+=LTC_Contains(LTC_LoadNode(a),tid.x,TestSlotOf(tid.x))?1u:0u;
            valid=valid && all(isfinite(asfloat(g_sharc.Load3(LT_TEST_Q(a)))));
            if(testMode==36u) valid=valid && g_sharc.Load(LT_TEST_VISITS(a))<=LT_MOMENT_HISTORY;
        }
        if(testMode==37u) {g_sharc.Store(cell+8u,1000000u);g_sharc.Store(cell+60u,0u);}
        results[tid.x]=float4(LT_PdfSelectTriangle(x,n,tid.x,TestInstOf(tid.x)),matches,count,valid?1:0);return;
    }
    if(testMode==34u || testMode==35u) {
        float3 x=testReceivers[0].xyz,n=float3(0,0,1);
        uint rng=LTC_Hash(tid.x^LTC_Hash(testSeed&0x7fffffffu))+1u;
        LT_Sample light=LT_SampleLight(x,n,rng);
        bool changed=(testSeed&0x80000000u)!=0u;
        bool visible=changed?light.id==triangleCount-1u:(light.id<triangleCount/4u && (light.id&15u)==0u);
        float estimate=visible?testRewardScale/light.pdf:0.0f;
        if(testMode==34u) LT_TrainSample(light.learningToken,estimate);
        results[tid.x]=float4(light.pdf,LT_PdfSelectTriangle(x,n,light.id,light.inst),estimate,light.id);return;
    }
    if(testMode>=29u && testMode<=33u) {
        uint cluster=LTC_TokenAddress(1u),address=LTC_BatchAddress(cluster);
        if(testMode==29u || testMode==33u) {
            LTC_ClearBatch(cluster);
            if(testMode==33u) for(uint stat=0;stat<2u;++stat)
                for(uint word=0;word<4u;++word) g_sharc.Store2(address+stat*LT_ACCUMULATOR_BYTES+word*8u,~0u);
            results[tid.x]=0;return;
        }
        if(testMode==30u) {
            float2 observation=testReceivers[tid.x%triangleCount].xy;
            LTC_Accumulate(address,observation.x);
            LTC_Accumulate(address+LT_ACCUMULATOR_BYTES,observation.y);
            uint ignored;g_sharc.InterlockedAdd(LT_TEST_SELECTED(cluster),1u,ignored);
            results[tid.x]=0;return;
        }
        if(testMode==31u) {
            uint2 word=g_sharc.Load2(address+tid.x*8u);
            results[tid.x]=float4(word.x&65535u,word.x>>16u,word.y&65535u,word.y>>16u);return;
        }
        results[tid.x]=float4(LTC_BatchMoments(cluster),g_sharc.Load(LT_TEST_SELECTED(cluster)),0);return;
    }
    if(testMode>=22u && testMode<=28u) {
        float3 x=testReceivers[0].xyz,n=float3(0,0,1);uint tri=tid.x%triangleCount;
        if(testMode==22u) {
            LTC_Proposal p;uint ignored;LTC_GetProposal(x,n,p);
            results[tid.x]=float4(LT_PdfSelectTriangle(x,n,tri,TestInstOf(tri)),LTC_CellPdf(x,n,p.fine,tri,TestSlotOf(tri),ignored),
                LTC_CellPdf(x,n,p.coarse,tri,TestSlotOf(tri),ignored),p.blend);return;
        }
        if(testMode==23u || testMode>=26u) {
            uint cell=0u,level=testSeed&31u,light=testSeed>>8u;
            bool found=level==25u;
            if(found) cell=LTC_Cell(LT_GRID_CAPACITY+LTC_NormalFace(n));
            else found=LTC_FindExact(LTC_KeyAtLevel(x,n,level),cell);
            if(!found) {results[tid.x]=float4(-1,0,0,0);return;}
            uint ignored;
            if(testMode==23u) {
                results[tid.x]=float4(LTC_CellPdf(x,n,cell,light,TestSlotOf(light),ignored),g_sharc.Load(cell+8u),LTC_Index(cell),g_sharc.Load(cell+24u));return;
            }
            if(testMode>=27u) {
                float sum=0;uint samples=0;
                for(uint j=0;j<g_sharc.Load(cell+4u);++j) {
                    uint a=LTC_Cluster(cell,j);
                    sum+=LTC_BatchMoments(a).x/LTC_FrozenProbability(a);samples+=g_sharc.Load(LT_TEST_SELECTED(a));
                    if(testMode==28u) LTC_ClearBatch(a);
                }
                results[tid.x]=float4(sum,samples,LTC_Index(cell),0);return;
            }
            uint count=g_sharc.Load(cell+4u);float total=0,cdf=0;
            for(uint j=0;j<count;++j) total+=LTC_Contains(LTC_LoadNode(LTC_Cluster(cell,j)),light,TestSlotOf(light))?100.0f:1.0f;
            for(uint j=0;j<count;++j) {
                uint a=LTC_Cluster(cell,j);float q=LTC_Contains(LTC_LoadNode(a),light,TestSlotOf(light))?100.0f:1.0f;
                float next=j+1u==count?1.0f:cdf+q/total;
                g_sharc.Store2(LT_TEST_PROBABILITY(a),asuint(float2(next-cdf,next)));cdf=next;
            }
            results[tid.x]=0;return;
        }
        uint rng=LTC_Hash(tid.x+(testSeed&65535u))+1u;LT_Sample s=LT_SampleLight(x,n,rng);
        float estimate=s.id==(testSeed>>16u)?testRewardScale/s.pdf:0.0f;
        if(testMode==24u) LT_TrainSample(s.learningToken,estimate);
        results[tid.x]=float4(s.pdf,LT_PdfSelectTriangle(x,n,s.id,s.inst),estimate,
            s.learningToken.y!=0u?float((s.learningToken.y-1u)/LT_CUT_MAX):-1.0f);return;
    }
    if((testMode>=11u && testMode<=19u) || testMode==50u || testMode==51u) {
        uint receiver=(testMode>=13u && testMode<=16u) || testMode>=50u?
            ((testSeed&0x80000000u)!=0u?LTC_Hash(tid.x)%triangleCount:(tid.x/256u)%triangleCount):tid.x%triangleCount;
        float3 x=testReceivers[receiver].xyz,n=float3(0,0,1);
        if(testMode==19u) {
            uint cell;bool found=LTC_Find(x,n,cell);
            uint level=found?(LTC_Index(cell)>=LT_GRID_CAPACITY?LT_MAX_LEVEL+1u:LTC_KeyLevel(g_sharc.Load4(LTC_KeyAddress(LTC_Index(cell))))):99u;
            results[tid.x]=float4(LTC_Level(x),level,found?g_sharc.Load(cell+8u):0,found?float(LTC_Index(cell)):-1);return;
        }
        if(testMode==17u || testMode==18u) {
            if(testMode==17u) {LT_Train(x,n,0u,TestInstOf(0u),float(tid.x%7u));results[tid.x]=0;return;}
            uint cell;results[tid.x]=0;
            if(LTC_Find(x,n,cell)) for(uint i=0;i<g_sharc.Load(cell+4u);++i) {
                uint a=LTC_Cluster(cell,i);if(!LTC_Contains(LTC_LoadNode(a),0u,TestSlotOf(0u))) continue;
                results[tid.x]=float4(LTC_BatchMoments(a),g_sharc.Load(LT_TEST_SELECTED(a)),LTC_FrozenProbability(a));return;
            }
            return;
        }
        if(testMode>=14u) {
            uint rng=LTC_Hash(tid.x+testSeed)+1u;
            // Modes 50/51 time a glossy receiver seen at a slant (the sample and sample+PDF cases).
            LT_Receiver R=LT_DiffuseReceiver(x,n);
            if(testMode>=50u) R=LT_MakeReceiver(x,n,normalize(float3(0.6f,0.1f,0.8f)),float3(1,0,0),0.4f,0.6f,0.3f,0.3f);
            LT_Sample light=LT_SampleLight(R,rng);
            if(testMode==15u) {
                LT_TrainSample(light.learningToken,testRewardScale*float(light.id%7u)/light.pdf);
            }
            results[tid.x]=float4(light.id,light.pdf,(testMode==16u || testMode==51u)?LT_PdfSelectTriangle(R,light.id,light.inst):0,0);return;
        }
        uint cell;bool found=LTC_Find(x,n,cell);
        if(testMode!=12u) {
            LTC_Request(x,n);
            if(found) LT_Train(x,n,tid.x%128u,TestInstOf(tid.x%128u),float(tid.x%7u));
        }
        if(testMode==12u && testSeed==1u) {
            results[tid.x]=float4(found?g_sharc.Load(cell+12u):0,found && LTC_Replaceable(cell)?1:0,0,0);return;
        }
        results[tid.x]=float4(found?1:0,found?g_sharc.Load(cell+8u):0,
            found?g_sharc.Load(cell+4u):0,found?float(cell):0);return;
    }

    if(testMode==5u) {
        // A small emitter at the origin facing -z; receivers on its -z side.
        const float3 mean=0,axis=float3(0,0,-1);const float variance=1e-4f,kappa=LT_KappaOf(0.5f),power=2,radius=.1f;
        const LT_Receiver up=LT_DiffuseReceiver(float3(0,0,-2),float3(0,0,1));
        float front=LT_NodeImportance(up,mean,variance,axis,kappa,power,radius,1.0f);
        float back=LT_NodeImportance(up,mean,variance,-axis,kappa,power,radius,1.0f);
        float below=LT_NodeImportance(LT_DiffuseReceiver(float3(0,0,-2),float3(0,0,-1)),mean,variance,axis,kappa,power,radius,1.0f);
        float far=LT_NodeImportance(LT_DiffuseReceiver(float3(0,0,-4),float3(0,0,1)),mean,variance,axis,kappa,power,radius,1.0f);
        if(tid.x==0) results[tid.x]=float4(front,back,below,front/max(far,1e-30f));
        else {
            float4x4 inverse=float4x4(.5f,0,0,0,0,1.0f/3.0f,0,0,0,0,.25f,0,0,0,0,1);
            results[tid.x]=float4(LT_LocalReceiverNormal((float3x4)inverse,normalize(float3(1,1,1))),0);
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
            for(uint i=0;i<count;++i) if(LTC_Contains(LTC_LoadNode(LTC_Cluster(cell,i)),tri,TestSlotOf(tri))) ++matches;
        }
        results[tid.x]=float4(LT_PdfSelectTriangle(x,n,tri,TestInstOf(tri)),matches,count,sum);return;
    }
    LT_Sample sample=LT_SampleLight(x,n,rng);
    float pdf=(rs_flags & RS_FLAG_NO_MESH_LIGHTS)!=0u?0:LT_PdfSelectTriangle(x,n,tri,TestInstOf(tri));
    float evaluated=LT_PdfSelectTriangle(x,n,sample.id,sample.inst);
    if(testMode==0u) {
        results[tid.x]=float4(pdf,sample.pdf,evaluated,sample.id==LT_SENTINEL?-1.0f:float(sample.id));return;
    }
    float estimate=sample.id==0u && sample.pdf>0u && testMode!=4u?testRewardScale/sample.pdf:0;
    if(testMode==1u || testMode==4u) {
        LT_TrainSample(sample.learningToken,estimate);
    }
    results[tid.x]=float4(pdf,sample.pdf,evaluated,estimate);
}
