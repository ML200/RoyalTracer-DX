#ifndef CUMULUS_DENSITY_CACHE_V8
#define CUMULUS_DENSITY_CACHE_V8
Texture3D<float> g_cumulusDensityCache : register(t57);
Texture3D<int4> g_cumulusDensityTags : register(t58);
int3 CumulusDensityBrickCounts()
{
    return int3(CUMULUS_DENSITY_BRICKS_XZ,CUMULUS_DENSITY_BRICKS_Y,CUMULUS_DENSITY_BRICKS_XZ);
}
float CumulusDensityVoxelSize() { return CUMULUS_DENSITY_VOXEL_KM*max(cloudScale,.2f); }
int3 CumulusDensityBrickSlot(int3 brick)
{
    int3 n=CumulusDensityBrickCounts();return (brick%n+n)%n;
}
bool CumulusDensityCacheActive()
{
    // Spherical height/foot projection is not a rigid translation. Never
    // advect this snapshot as if it were an exact animated-wind material.
    return cloudDensityCache>.5f && cloudDensityEpoch!=0 && cloudWindX==0 && cloudWindZ==0;
}
bool CumulusReadDensityCache(float3 P,float footprintKm,bool fine,out float density)
{
    density=0;
    // Only the full-detail plateau is baked. Filtered material keeps its path.
    if(!CumulusDensityCacheActive() || !fine || footprintKm>.008f*max(cloudScale,.2f))return false;
    float3 grid=P/CumulusDensityVoxelSize();
    int3 brick=int3(floor(grid/CUMULUS_DENSITY_BRICK_CELLS));
    int3 slot=CumulusDensityBrickSlot(brick);
    int4 tag=g_cumulusDensityTags.Load(int4(slot,0));
    if(any(tag.xyz!=brick) || asuint(tag.w)!=cloudDensityEpoch)return false;
    float3 local=grid-float3(brick*int(CUMULUS_DENSITY_BRICK_CELLS));
    float3 uv=(float3(slot*CUMULUS_DENSITY_BRICK_VERTICES)+local+.5f)
        /float3(CumulusDensityBrickCounts()*CUMULUS_DENSITY_BRICK_VERTICES);
    density=g_cumulusDensityCache.SampleLevel(g_sampler_LUT,uv,0);
    return true;
}
#endif
