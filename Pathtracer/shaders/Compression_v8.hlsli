// RGB9E5 constants
static const uint RGB9E5_MANTISSA_BITS = 9;
static const uint RGB9E5_EXP_BITS       = 5;
static const int  RGB9E5_EXP_BIAS       = 15;
static const uint RGB9E5_MANT_MASK      = (1u << RGB9E5_MANTISSA_BITS) - 1;
static const uint RGB9E5_EXP_MASK       = (1u << RGB9E5_EXP_BITS) - 1;

uint PackRGB9E5(float3 v)
{
    // clamp to [0, sharedexp_max]
    const float sharedexp_max = (float(RGB9E5_MANT_MASK) / float(RGB9E5_MANT_MASK + 1u))
                              * exp2((RGB9E5_EXP_MASK - RGB9E5_EXP_BIAS));
    float3 c = clamp(v, 0.0f, sharedexp_max);

    // pick the largest channel
    float m = max(max(c.x, c.y), c.z);

    // get IEEE exponent and mantissa
    uint bits    = asuint(m);
    int  exp_unb = int((bits >> 23) & 0xFF) - 127;       // floor(log2(m))
    uint frac    = bits & 0x7FFFFF;                     // mantissa bits

    // compute shared biased exponent = ceil(log2(m)) + B
    // ceil = floor + (mantissa>0?1:0)
    int sharedExp = exp_unb + int(frac != 0) + RGB9E5_EXP_BIAS;
    sharedExp = clamp(sharedExp, 0, int(RGB9E5_EXP_MASK));

    // denominator = 2^(sharedExp − B − N)
    float denom = exp2(float(sharedExp - RGB9E5_EXP_BIAS - int(RGB9E5_MANTISSA_BITS)));

    // quantize each channel
    uint rm = uint(floor(c.x / denom + 0.5f));
    uint gm = uint(floor(c.y / denom + 0.5f));
    uint bm = uint(floor(c.z / denom + 0.5f));

    uint maxMant = max(max(rm, gm), bm);
    if (maxMant > RGB9E5_MANT_MASK)          // ==512 after rounding?
    {
        rm >>= 1;  gm >>= 1;  bm >>= 1;      // divide all by 2
        sharedExp = min(sharedExp + 1, int(RGB9E5_EXP_MASK));
    }
    rm &= RGB9E5_MANT_MASK;
    gm &= RGB9E5_MANT_MASK;
    bm &= RGB9E5_MANT_MASK;

    // 7) pack: [ r:9 | g:9 | b:9 | exp:5 ]
    return (rm <<  0) |
           (gm <<  9) |
           (bm << 18) |
           (uint(sharedExp) << 27);
}

float3 UnpackRGB9E5(uint p)
{
    // extract fields
    uint rm = (p >>  0) & RGB9E5_MANT_MASK;
    uint gm = (p >>  9) & RGB9E5_MANT_MASK;
    uint bm = (p >> 18) & RGB9E5_MANT_MASK;
    int  e  = int((p >> 27) & RGB9E5_EXP_MASK);

    // compute scale = 2^(e − B − N)
    float scale = exp2(float(e - RGB9E5_EXP_BIAS - int(RGB9E5_MANTISSA_BITS)));

    return float3(rm * scale, gm * scale, bm * scale);
}

static const uint  kMax16 = 65535;

float2 signNotZero(float2 v)
{
    return step(0.0f, v) * 2.0f - 1.0f;
}

static const uint PROBE_DI_NORMAL_ZERO_CODE = ~0u;

uint PackNormal(float3 n)
{
    if (dot(n, n) < 1e-6f)
    {
        return PROBE_DI_NORMAL_ZERO_CODE;
    }

    // 2. Standard Octahedral Encoding
    n = normalize(n);
    float3 a = abs(n);
    float2 p = n.xy / (a.x + a.y + a.z);

    if (n.z < 0.0f)
        p = (1.0f - abs(p.yx)) * signNotZero(p);

    uint2 q = uint2(round((p * 0.5f + 0.5f) * kMax16));

    uint packed = q.x | (q.y << 16);
    if (packed == PROBE_DI_NORMAL_ZERO_CODE)
        packed--;

    return packed;
}

float3 UnpackNormal(uint bits)
{
    // 1. Handle Zero Vector explicitly
    if (bits == PROBE_DI_NORMAL_ZERO_CODE)
    {
        return float3(0.0f, 0.0f, 0.0f);
    }

    // 2. Standard Octahedral Decoding
    float2 f = (float2(bits & 0xFFFF, bits >> 16) / kMax16) * 2.0f - 1.0f;

    float3 n = float3(f.x, f.y, 1.0f - abs(f.x) - abs(f.y));

    float  t  = saturate(-n.z);
    n.xy     += -t * signNotZero(n.xy);

    return normalize(n);
}

float3 UnpackNormal_INT(uint packed)
{
    int ix = (int)(packed << 16) >> 16;
    int iy = (int)packed >> 16;
    float2 f = float2(ix, iy) / 32767.0f;

    float3 n = float3(f.x, f.y, 1.0f - abs(f.x) - abs(f.y));

    if (n.z < 0.0)
    {
        float oldX = n.x;
        float oldY = n.y;
        n.x = (1.0f - abs(oldY)) * (oldX >= 0.0f ? 1.0f : -1.0f);
        n.y = (1.0f - abs(oldX)) * (oldY >= 0.0f ? 1.0f : -1.0f);
    }

    return normalize(n);
}

// Packs two [0..1] floats into one uint (16 bits each)
uint PackScalars16(float a, float b)
{
    uint ua = (uint)(saturate(a) * 65535.0f);
    uint ub = (uint)(saturate(b) * 65535.0f);
    return ua | (ub << 16);
}

float2 UnpackScalars16(uint p)
{
    float a = float(p & 0xFFFF) / 65535.0f;
    float b = float(p >> 16) / 65535.0f;
    return float2(a, b);
}



