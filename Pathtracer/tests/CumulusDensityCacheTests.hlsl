#define COMPUTE_PASS
#include "Includes_v8.hlsli"
#include "CumulusDensity_v8.hlsli"
RWByteAddressBuffer probeOutput : register(u63);
[numthreads(8,8,1)]
void densityCacheProbe(uint3 p:SV_DispatchThreadID)
{
    if(any(p.xy>=gImageSize))return;
    uint i=p.y*IMG_W+p.x,rng=Hash32(i^0xB515574Du);
    int3 n=CumulusDensityBrickCounts();
    float3 center=normalize(CumulusCameraPlanet())*(ATMOS_BOTTOM_RADIUS+cloudBaseKm+cloudThicknessKm*.5f);
    int3 origin=int3(floor(center/(CumulusDensityVoxelSize()*CUMULUS_DENSITY_BRICK_CELLS)))-n/2;
    int3 brick=origin+int3(i%uint(n.x),(i/uint(n.x))%uint(n.y),(i/uint(n.x*n.y))%uint(n.z));
    if(cloudDebugView==20)brick.x+=n.x; // Same physical slot, different world block.
    // Interior points avoid rounding a nominal outside-edge point into the
    // neighbouring, unbaked window. Exact border vertices are checked below.
    float3 local=.125f+float3(RandomFloatPCG(rng),RandomFloatPCG(rng),RandomFloatPCG(rng))*7.75f;
    float3 P=(float3(brick*int(CUMULUS_DENSITY_BRICK_CELLS))+local)*CumulusDensityVoxelSize();
    float density;float footprint=cloudDebugView==21?.1f:0;
    bool hit=CumulusReadDensityCache(P,footprint,true,density);
    float reference=CumulusSampleMaterial(P,0,true).density;
    int3 vertex=int3(i%9u,(i/9u)%9u,(i/81u)%9u);
    precise float3 node=float3(brick*int(CUMULUS_DENSITY_BRICK_CELLS)+vertex)*CumulusDensityVoxelSize();
    float stored=hit?g_cumulusDensityCache.Load(int4(CumulusDensityBrickSlot(brick)*int(CUMULUS_DENSITY_BRICK_VERTICES)+vertex,0)):0;
    precise float exactNode=CumulusSampleMaterial(node,0,true).density;
    uint a=i*64;
    probeOutput.Store4(a,asuint(float4(hit?1:0,density,reference,abs(density-reference))));
    probeOutput.Store4(a+16,asuint(float4(stored,exactNode,abs(stored-exactNode),float(any(brick.xz<0)))));
    probeOutput.Store4(a+32,asuint(float4(P,footprint)));
    probeOutput.Store4(a+48,0);
}
