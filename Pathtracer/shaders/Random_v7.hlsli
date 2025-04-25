/*
Random Number generation functions.
*/

// c is a constant
uint2 GetSeed(uint2 idx, uint t, uint c)
{
    return
        idx.yx * uint2(73856093u, 37623481u)
      ^ idx   * uint2(19349663u, 51964263u)
      ^ uint2(83492791u, 68250729u) * c
      ^ uint2(293803u,    423977u)   * t;
}

// Optimized random function
float RandomFloat(inout uint2 s)
{
    uint sum = 0u;
    for(uint i = 0u; i < 4u; ++i)
    {
        sum += 0x9E3779B9u;
        s.x += ((s.y << 4u) + 0xA341316Cu)
             ^ (s.y + sum)
             ^ ((s.y >> 5u) + 0xC8013EA4u);
        s.y += ((s.x << 4u) + 0xAD90777Du)
             ^ (s.x + sum)
             ^ ((s.x >> 5u) + 0x7E95761Eu);
    }

    // reinterpret s.x’s low 23 bits as mantissa → [1,2), subtract 1 → [0,1)
    return asfloat((s.x & 0x007FFFFFu) | 0x3F800000u) - 1.0;
}

// Cheap random function for single seed
float RandomFloatSingle(inout uint s)
{
    s = s * 747796405u + 2891336453u;      // LCG step
    uint x = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    x = (x >> 22u) ^ x;
    return asfloat(0x3F800000u | (x >> 9u)) - 1.0;
}


