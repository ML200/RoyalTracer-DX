#define COMPUTE_PASS
#include "Includes_v8.hlsli"
#include "CumulusDensity_v8.hlsli"
RWTexture3D<float> g_densityOut : register(u45);
RWTexture3D<int4> g_densityTagsOut : register(u46);
groupshared uint g_densityNeedsBake;
[numthreads(128,1,1)]
void main(uint3 group:SV_GroupID,uint lane:SV_GroupIndex)
{
    if(!CumulusDensityCacheActive() || cloudEnabled<.5f || cloudCoverage<=0)return;
    uint index=(uint(time)%CUMULUS_DENSITY_PHASES)*CUMULUS_DENSITY_BATCH+group.x;
    if(index>=CUMULUS_DENSITY_BRICK_COUNT)return;
    int3 n=CumulusDensityBrickCounts();
    float3 center=normalize(CumulusCameraPlanet())*(ATMOS_BOTTOM_RADIUS+cloudBaseKm+cloudThicknessKm*.5f);
    int3 origin=int3(floor(center/(CumulusDensityVoxelSize()*CUMULUS_DENSITY_BRICK_CELLS)))-n/2;
    int3 brick=origin+int3(index%uint(n.x),(index/uint(n.x))%uint(n.y),index/uint(n.x*n.y));
    int3 slot=CumulusDensityBrickSlot(brick);
    if(lane==0) {
        int4 tag=g_densityTagsOut[slot];
        g_densityNeedsBake=any(tag.xyz!=brick)||asuint(tag.w)!=cloudDensityEpoch;
    }
    GroupMemoryBarrierWithGroupSync();
    if(!g_densityNeedsBake)return;
    for(uint v=lane;v<729u;v+=128u) {
        int3 local=int3(v%9u,(v/9u)%9u,v/81u);
        precise float3 P=float3(brick*int(CUMULUS_DENSITY_BRICK_CELLS)+local)*CumulusDensityVoxelSize();
        // Direct material call: the bake must never read its own cache.
        // Keep identical vertices identical across independently compiled bakes
        // and probes: reassociation is visible at planet-scale coordinates.
        precise float density=CumulusSampleMaterial(P,0,true).density;
        g_densityOut[slot*CUMULUS_DENSITY_BRICK_VERTICES+local]=density;
    }
    DeviceMemoryBarrierWithGroupSync();
    if(lane==0)g_densityTagsOut[slot]=int4(brick,asint(cloudDensityEpoch));
}
