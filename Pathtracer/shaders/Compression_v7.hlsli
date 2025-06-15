// Constants per EXT_texture_shared_exponent spec
static const uint RGB9E5_MANTISSA_BITS = 9;
static const uint RGB9E5_EXP_BITS       = 5;
static const int  RGB9E5_EXP_BIAS       = 15;
static const uint RGB9E5_MANT_MASK      = (1u << RGB9E5_MANTISSA_BITS) - 1;   // 0x1FF
static const uint RGB9E5_EXP_MASK       = (1u << RGB9E5_EXP_BITS) - 1;       // 0x1F



uint PackRGB9E5(float3 v)
{
    // 1) clamp to [0, sharedexp_max]:
    //    sharedexp_max = ((2^N - 1)/2^N) * 2^(Emax−B), where N=9, Emax=31, B=15
    const float sharedexp_max = (float(RGB9E5_MANT_MASK) / float(RGB9E5_MANT_MASK + 1u))
                              * exp2((RGB9E5_EXP_MASK - RGB9E5_EXP_BIAS));
    float3 c = clamp(v, 0.0f, sharedexp_max);

    // 2) pick the largest channel
    float m = max(max(c.x, c.y), c.z);

    // 3) get IEEE exponent and mantissa
    uint bits    = asuint(m);
    int  exp_unb = int((bits >> 23) & 0xFF) - 127;       // floor(log2(m))
    uint frac    = bits & 0x7FFFFF;                     // mantissa bits

    // 4) compute shared biased exponent = ceil(log2(m)) + B
    //    ceil = floor + (mantissa>0?1:0)
    int sharedExp = exp_unb + int(frac != 0) + RGB9E5_EXP_BIAS;
    sharedExp = clamp(sharedExp, 0, int(RGB9E5_EXP_MASK));

    // 5) denominator = 2^(sharedExp − B − N)
    float denom = exp2(float(sharedExp - RGB9E5_EXP_BIAS - int(RGB9E5_MANTISSA_BITS)));

    // 6) quantize each channel (compute *unclamped* mantissas first)
    uint rm = uint(floor(c.x / denom + 0.5f));
    uint gm = uint(floor(c.y / denom + 0.5f));
    uint bm = uint(floor(c.z / denom + 0.5f));

    uint maxMant = max(max(rm, gm), bm);
    if (maxMant > RGB9E5_MANT_MASK)          // ==512 after rounding?
    {
        rm >>= 1;  gm >>= 1;  bm >>= 1;      // divide all by 2
        sharedExp = min(sharedExp + 1, int(RGB9E5_EXP_MASK));
    }

    // **now** clamp into 9 bits
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

// ------------------------------------------------------------------ helpers
float2 signNotZero(float2 v)       // branch‑free, works on any GPU
{
    return step(0.0f, v) * 2.0f - 1.0f;
}

// ------------------------------------------------------------------ pack
uint PackNormal(float3 n)
{
    n = normalize(n);
    float3 a = abs(n);
    float2 p = n.xy / (a.x + a.y + a.z);

    if (n.z < 0.0f)
        p = (1.0f - abs(p.yx)) * signNotZero(p);

    uint2 q = uint2(round((p * 0.5f + 0.5f) * kMax16));
    return q.x | (q.y << 16);
}

// ------------------------------------------------------------------ unpack
float3 UnpackNormal(uint bits)
{
    float2 f = (float2(bits & 0xFFFF, bits >> 16) / kMax16) * 2.0f - 1.0f;

    float3 n = float3(f.x, f.y, 1.0f - abs(f.x) - abs(f.y));

    float  t  = saturate(-n.z);          // Rune Stubbe “clamp‑back”
    n.xy     += -t * signNotZero(n.xy);  // no vector‑ternary

    return normalize(n);
}


