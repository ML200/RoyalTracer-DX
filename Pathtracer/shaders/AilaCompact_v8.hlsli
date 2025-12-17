uint Part1By2_10(uint x) // assumes x <= 1023
{
    x &= 1023u;
    x = (x | (x << 16)) & 0x030000FFu;
    x = (x | (x <<  8)) & 0x0300F00Fu;
    x = (x | (x <<  4)) & 0x030C30C3u;
    x = (x | (x <<  2)) & 0x09249249u;
    return x;
}

uint Morton3_10(uint3 v) // 30-bit morton
{
    return (Part1By2_10(v.x) << 0) | (Part1By2_10(v.y) << 1) | (Part1By2_10(v.z) << 2);
}

uint Quantize01To10(float x)
{
    x = saturate(x);
    return (uint)round(x * 1023.0f);
}

uint AilaCompactKey32(float3 originWS, float3 dirWS,
                      float3 sceneMinWS, float3 sceneMaxWS)
{
    // Origin -> [0,1] in scene bounds (you can clamp or wrap depending on your use case)
    float3 o01 = (originWS - sceneMinWS) / max(sceneMaxWS - sceneMinWS, 1e-6.xxx);
    uint3  oq  = uint3(Quantize01To10(o01.x), Quantize01To10(o01.y), Quantize01To10(o01.z));
    uint   O   = Morton3_10(oq); // 30 bits

    // Direction -> map [-1,1] to [0,1] then morton
    float3 dn  = normalize(dirWS);
    float3 d01 = dn * 0.5f + 0.5f;
    uint3  dq  = uint3(Quantize01To10(d01.x), Quantize01To10(d01.y), Quantize01To10(d01.z));
    uint   D   = Morton3_10(dq); // 30 bits

    // "Compact Aila-style": emit a few MSBs of origin first (coarse origin),
    // then interleave remaining bits of origin and direction.
    // Tune ORIGIN_PREFIX for your content; 6–12 is a reasonable sweep.
    const uint ORIGIN_PREFIX = 8;

    uint key = 0;
    uint outBit = 31;

    // Take ORIGIN_PREFIX MSBs of O
    [unroll]
    for (uint i = 0; i < ORIGIN_PREFIX; ++i)
    {
        uint b = (O >> (29 - i)) & 1u;
        key |= (b << outBit);
        outBit--;
    }

    // Interleave remaining bits: O then D
    uint oIdx = ORIGIN_PREFIX;
    uint dIdx = 0;

    [unroll]
    for (; outBit != 0xFFFFFFFFu; )
    {
        if (oIdx < 30)
        {
            uint b = (O >> (29 - oIdx)) & 1u;
            key |= (b << outBit);
            outBit--; oIdx++;
            if (outBit == 0xFFFFFFFFu) break;
        }
        if (dIdx < 30)
        {
            uint b = (D >> (29 - dIdx)) & 1u;
            key |= (b << outBit);
            outBit--; dIdx++;
        }
        else
        {
            // no more direction bits; continue draining origin bits
            if (oIdx >= 30) break;
        }
    }

    return key;
}
