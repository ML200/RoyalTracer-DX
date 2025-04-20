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

    // 6) quantize each channel
    uint rm = uint(floor(c.x/denom + 0.5f)) & RGB9E5_MANT_MASK;
    uint gm = uint(floor(c.y/denom + 0.5f)) & RGB9E5_MANT_MASK;
    uint bm = uint(floor(c.z/denom + 0.5f)) & RGB9E5_MANT_MASK;

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



//— Pack 3×float Normal into a 32‑bit uint via octahedral mapping —//
// Projects the unit‑vector onto the octahedron, folds negative Z,
// remaps XY into [0..1], quantizes to 16 bits each, and bit‑fields.
// ~15 arithmetic ops + 2 shifts/or per pack; ~12 ops per unpack.

uint PackNormal(float3 n)
{
    // 1) normalize (if not already)
    n = normalize(n);

    // 2) project onto octahedron
    float3  a = abs(n);
    float2  p = n.xy / (a.x + a.y + a.z);

    // 3) fold lower hemisphere
    //    if n.z < 0: reflect p across diagonal
    p = (n.z >= 0.0f)
        ? p
        : (1.0f - abs(p.yx)) * float2(sign(p.x), sign(p.y));

    // 4) remap to [0..1] and quantize to 16 bits
    float2 m = p * 0.5f + 0.5f;
    uint2  q = uint2(m * 65535.0f + 0.5f);

    // 5) pack
    return q.x | (q.y << 16);
}

float3 UnpackNormal(uint pack)
{
    // 1) extract & remap to [-1..1]
    float2 p = float2(
        float((pack       & 0xFFFF)) / 65535.0f,
        float((pack >> 16 & 0xFFFF)) / 65535.0f
    ) * 2.0f - 1.0f;

    // 2) unfold Z
    float3 n;
    n.z = 1.0f - abs(p.x) - abs(p.y);

    // 3) reflect if negative hemisphere
    float2 folded = (n.z >= 0.0f)
        ? p
        : (1.0f - abs(p.yx)) * float2(sign(p.x), sign(p.y));

    // 4) reconstruct and normalize
    n.x = folded.x;
    n.y = folded.y;
    return normalize(n);
}
