#define COMPUTE_PASS
#include "Includes_v8.hlsli"
#include "CumulusDensity_v8.hlsli"
#include "CumulusDensityReference.hlsli"
RWByteAddressBuffer probeOutput : register(u63);
// Separate entry points keep the same call context for both material graphs.
// Evaluating both in one kernel allows different floating-point contractions,
// even when the candidate is an unmodified copy of the reference.
void writeDensity(uint3 p,bool reference)
{
    if(any(p.xy>=gImageSize))return;
    uint i=p.y*IMG_W+p.x;
    uint rng=Hash32(i^asuint(cloudSeed)^0x75ED45BFu);
    float3 C,U,E,N;CumulusFrame(C,U,E,N);
    float2 xz=float2(RandomFloatPCG(rng),RandomFloatPCG(rng))*2000-1000;
    float height=lerp(-.05f,1.05f,RandomFloatPCG(rng));
    float3 P=normalize(C+xz.x*E+xz.y*N)*(ATMOS_BOTTOM_RADIUS+cloudBaseKm+height*cloudThicknessKm);
    // Include both sides and the exact endpoints of every footprint fade.
    const float footprints[16]={0,.002f,.008f,.012f,.02f,.06499f,.065f,.06501f,
        .07f,.08999f,.09f,.09001f,.12f,.3f,.7f,2};
    float footprint=footprints[i%16]*max(cloudScale,.2f);
    uint kind=(i/16)%4;
    bool fine=(kind&1u)==0,coarse=(kind&2u)!=0;
    CumulusMaterial a;
    if(reference)a=CumulusReferenceMaterial(P,footprint,fine,coarse);
    else a=CumulusSampleMaterial(P,footprint,fine,coarse);
    uint address=i*64;
    probeOutput.Store4(address,asuint(float4(a.density,a.profile,a.height,0)));
    probeOutput.Store4(address+16,0);
    probeOutput.Store4(address+32,asuint(float4(P,footprint)));
    probeOutput.Store4(address+48,asuint(float4(height,kind,0,0)));
}
[numthreads(8,8,1)]
void densityParity(uint3 p:SV_DispatchThreadID) { writeDensity(p,false); }
[numthreads(8,8,1)]
void densityReference(uint3 p:SV_DispatchThreadID) { writeDensity(p,true); }
