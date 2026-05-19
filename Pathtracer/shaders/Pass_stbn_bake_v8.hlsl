// =====================================================================
//  Pass_stbn_bake_v8.hlsl — One-shot spatiotemporal blue noise bake
// =====================================================================
//
//  Run ONCE during renderer init. Fills a 128x128x64 RGBA8 Texture2DArray
//  with approximate spatiotemporal blue noise. The runtime cloud shader
//  (CloudRand4 in Clouds_v8.hlsli) samples this array instead of evaluating
//  white noise hashes per pixel, giving the cone shadow taps and step
//  jitter a blue noise spatial spectrum that DLSS RR cleanly removes via
//  its low pass spatial filter (white noise leaves visible per pixel grain
//  because its energy reaches into the frequencies the filter passes).
//
//  Tile size: 128 x 128 per slice. Power of two so the runtime sample
//  uses AND masking instead of modulo. 64 temporal slices means a cycle
//  length of 64 frames at 60 fps = ~1 s, longer than any temporal pooling
//  window DLSS RR uses, so the loop is invisible.
//
//  Algorithm: Jiminez interleaved gradient noise (IGN). The IGN constants
//  (0.06711056, 0.00583715, 52.9829189) produce a per pixel output whose
//  spectral energy clusters in the high frequencies — the defining
//  property of blue noise. Originally derived for COD Advanced Warfare
//  (SIGGRAPH 2014) and now standard in UE5 and Frostbite for any pass
//  that wants blue noise behaviour at O(1) cost.
//
//  Per slice shift uses Jiminez's recommended 5.588238 frame stride so
//  the temporal sequence decorrelates across slices. Per channel shifts
//  (large primes) move each channel to a different lattice region so the
//  4 RGBA values within a single pixel are approximately independent —
//  important because the cloud shader uses (R, G) pairs as 2D cone disc
//  samples in SampleConeAroundDir.
//
//  This is a single dispatch approximation, not the iterative void and
//  cluster algorithm Christensen and Wolfe published. Quality is good
//  enough for DLSS RR to read the noise as removable; replace with true
//  STBN (Wolfe 2022) for the last factor of two of quality.
//
//  Dispatch: (128/8) x (128/8) x 64 = 16 x 16 x 64 = 16384 thread groups.

RWTexture2DArray<float4> g_stbn : register(u0);

#define STBN_W       128
#define STBN_H       128
#define STBN_SLICES  64

inline float IGN(float2 p)
{
    return frac(52.9829189f * frac(0.06711056f * p.x + 0.00583715f * p.y));
}

[numthreads(8, 8, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    if (any(dtid >= uint3(STBN_W, STBN_H, STBN_SLICES))) return;

    float2 p  = float2(dtid.xy);
    float  ts = 5.588238f * float(dtid.z);

    //Per channel offsets in the shared IGN lattice. Primes large enough
    //that adjacent pixels' (R, G) and (B, A) channel pairs don't align to
    //the lattice's primary directions, so the runtime can treat the four
    //values within a single fetch as approximately decorrelated.
    float r = IGN(p + ts + float2(  0.0f,   0.0f));
    float g = IGN(p + ts + float2( 47.0f, 109.0f));
    float b = IGN(p + ts + float2(199.0f, 313.0f));
    float a = IGN(p + ts + float2(457.0f, 617.0f));

    g_stbn[dtid] = float4(r, g, b, a);
}
