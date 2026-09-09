#define COMPUTE_PASS
#include "Includes_v8.hlsli"
RWTexture3D<float2> g_noiseRG : register(u28);
RWTexture3D<float2> g_noiseBA : register(u33);
RWTexture3D<int4> g_densityTagsOut : register(u46);
uint BakeHash3D(int3 p, int period)
{
    int3 m = (p % period + period) % period;
    uint h = uint(m.x) * 73856093u
           ^ uint(m.y) * 19349663u
           ^ uint(m.z) * 83492791u;
    h ^= h >> 16; h *= 0x7feb352du;
    h ^= h >> 15; h *= 0x846ca68bu;
    h ^= h >> 16;
    return h;
}

float BakeHashFloat(int3 p, int period)
{
    return float(BakeHash3D(p, period)) * (1.0f / 4294967296.0f);
}

float3 BakeHashVec3(int3 p, int period)
{
    uint h = BakeHash3D(p, period);
    return float3((h        & 1023u),
                  ((h >> 10) & 1023u),
                  ((h >> 20) & 1023u)) * (1.0f / 1023.0f);
}

float BakeValueNoise(float3 p, int period)
{
    int3   i = int3(floor(p));
    float3 f = frac(p);
    float3 u = f * f * f * (f * (f * 6.0f - 15.0f) + 10.0f);

    float c000 = BakeHashFloat(i + int3(0, 0, 0), period);
    float c100 = BakeHashFloat(i + int3(1, 0, 0), period);
    float c010 = BakeHashFloat(i + int3(0, 1, 0), period);
    float c110 = BakeHashFloat(i + int3(1, 1, 0), period);
    float c001 = BakeHashFloat(i + int3(0, 0, 1), period);
    float c101 = BakeHashFloat(i + int3(1, 0, 1), period);
    float c011 = BakeHashFloat(i + int3(0, 1, 1), period);
    float c111 = BakeHashFloat(i + int3(1, 1, 1), period);

    float x00 = lerp(c000, c100, u.x);
    float x10 = lerp(c010, c110, u.x);
    float x01 = lerp(c001, c101, u.x);
    float x11 = lerp(c011, c111, u.x);
    float y0  = lerp(x00,  x10,  u.y);
    float y1  = lerp(x01,  x11,  u.y);
    return lerp(y0, y1, u.z);
}

float BakeWorley(float3 p, int period)
{
    int3   i = int3(floor(p));
    float3 f = frac(p);
    float  minD2 = 1.0e10f;

    [unroll] for (int x = -1; x <= 1; ++x)
    [unroll] for (int y = -1; y <= 1; ++y)
    [unroll] for (int z = -1; z <= 1; ++z)
    {
        int3   cell    = i + int3(x, y, z);
        float3 feature = float3(x, y, z) + BakeHashVec3(cell, period) - f;
        float  d2      = dot(feature, feature);
        minD2          = min(minD2, d2);
    }
    return saturate(sqrt(minD2) * 1.15f);
}

float BakeWorleyFBM(float3 p, int basePeriod)
{
    float w0 = 1.0f - BakeWorley(p,                                          basePeriod);
    float w1 = 1.0f - BakeWorley(p * 2.0f + float3(13.31f, -7.13f, 19.77f),  basePeriod * 2);
    float w2 = 1.0f - BakeWorley(p * 4.0f + float3(-3.47f, 21.97f,  5.13f),  basePeriod * 4);
    return saturate(w0 * 0.55f + w1 * 0.30f + w2 * 0.15f);
}

float3 BakePerlinGrad(int3 p, int period)
{
    uint h=BakeHash3D(p,period);
    float3 g=float3(h&1023u,(h>>10)&1023u,(h>>20)&1023u)/511.5f-1;
    return g*rsqrt(max(dot(g,g),.001f));
}

float BakePerlin(float3 p, int period)
{
    int3   i = int3(floor(p));
    float3 f = frac(p);
    float3 u = f * f * f * (f * (f * 6.0f - 15.0f) + 10.0f);

    float3 g000 = BakePerlinGrad(i + int3(0, 0, 0), period);
    float3 g100 = BakePerlinGrad(i + int3(1, 0, 0), period);
    float3 g010 = BakePerlinGrad(i + int3(0, 1, 0), period);
    float3 g110 = BakePerlinGrad(i + int3(1, 1, 0), period);
    float3 g001 = BakePerlinGrad(i + int3(0, 0, 1), period);
    float3 g101 = BakePerlinGrad(i + int3(1, 0, 1), period);
    float3 g011 = BakePerlinGrad(i + int3(0, 1, 1), period);
    float3 g111 = BakePerlinGrad(i + int3(1, 1, 1), period);

    float c000 = dot(g000, f - float3(0, 0, 0));
    float c100 = dot(g100, f - float3(1, 0, 0));
    float c010 = dot(g010, f - float3(0, 1, 0));
    float c110 = dot(g110, f - float3(1, 1, 0));
    float c001 = dot(g001, f - float3(0, 0, 1));
    float c101 = dot(g101, f - float3(1, 0, 1));
    float c011 = dot(g011, f - float3(0, 1, 1));
    float c111 = dot(g111, f - float3(1, 1, 1));

    float x00 = lerp(c000, c100, u.x);
    float x10 = lerp(c010, c110, u.x);
    float x01 = lerp(c001, c101, u.x);
    float x11 = lerp(c011, c111, u.x);
    float y0  = lerp(x00,  x10,  u.y);
    float y1  = lerp(x01,  x11,  u.y);
    return lerp(y0, y1, u.z);
}
float BakePerlinFBM(float3 p, int period)
{
    float v = BakePerlin(p,                                       period)     * 0.50f
            + BakePerlin(p * 2.0f + float3(5.7f, -2.3f, 9.1f),    period * 2) * 0.25f
            + BakePerlin(p * 4.0f + float3(-1.3f, 11.7f, -4.2f),  period * 4) * 0.125f;
    return saturate(v * (1.0f / 1.75f) + 0.5f);
}

float BakePerlinWorley(float3 p, int period)
{
    float perlin    = saturate(BakePerlinFBM(p, period) + .30f);
    float worleyInv = BakeWorleyFBM(p, period);   // 3-octave; cauliflower edges
    return saturate((perlin - (1.0f - worleyInv)) / max(worleyInv, 1e-3f));
}

[numthreads(512,1,1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint i=id.x;
    if(i<CUMULUS_DENSITY_BRICK_COUNT) {
        uint3 p=uint3(i%CUMULUS_DENSITY_BRICKS_XZ,
            (i/CUMULUS_DENSITY_BRICKS_XZ)%CUMULUS_DENSITY_BRICKS_Y,
            i/(CUMULUS_DENSITY_BRICKS_XZ*CUMULUS_DENSITY_BRICKS_Y));
        g_densityTagsOut[p]=0;
    }
    if(i>=CUMULUS_NOISE_SIZE*CUMULUS_NOISE_SIZE*CUMULUS_NOISE_SIZE) return;
    uint3 p=uint3(i%CUMULUS_NOISE_SIZE,(i/CUMULUS_NOISE_SIZE)%CUMULUS_NOISE_SIZE,i/(CUMULUS_NOISE_SIZE*CUMULUS_NOISE_SIZE));
    float3 uv=(float3(p)+.5f)/CUMULUS_NOISE_SIZE;
    float4 value=float4(BakePerlinWorley(uv*16,16),BakeWorleyFBM(uv*8,8),BakeValueNoise(uv*48,48),BakeWorley(uv*16,16));
    g_noiseRG[p]=value.rg;g_noiseBA[p]=value.ba;
}
