#ifndef PROCEDURAL_TERRAIN_V8_HLSLI
#define PROCEDURAL_TERRAIN_V8_HLSLI

#define PT_TOP_PERIOD_M     16384.0f

#define PT_BOTTOM_PERIOD_M  16.0f

#define PT_TOP_AMP_M        60.0f
#define PT_LACUNARITY       2.0f

#define PT_GAIN             0.55f
#define PT_MAX_OCTAVES      15

#define PT_SMOOTH_K         24.0f
#define PT_SEED             0xA5C9E1B7u

inline uint pt_hash_u32(uint x) {
    x ^= x >> 17;
    x *= 0xed5ad4bbu;
    x ^= x >> 11;
    x *= 0xac4c1b51u;
    x ^= x >> 15;
    x *= 0x31848babu;
    x ^= x >> 14;
    return x;
}

inline float3 pt_cell_feature(int ix, int iy, int iz) {
    uint h = pt_hash_u32(
        pt_hash_u32(pt_hash_u32(uint(ix) + PT_SEED)
                              + uint(iy))
                              + uint(iz));
    const float inv = 1.0f / 16777216.0f;
    float3 o;
    o.x = float(h & 0xFFFFFFu) * inv;
    h *= 1664525u;
    o.y = float(h & 0xFFFFFFu) * inv;
    h *= 1664525u;
    o.z = float(h & 0xFFFFFFu) * inv;
    return o;
}

inline float pt_worley_smooth(int3 cellIdx, float3 frac, out float3 gradOut) {
    float  sumExp = 0.0f;
    float3 sumWeightedDir = float3(0, 0, 0);
    const float k = PT_SMOOTH_K;

    [unroll]
    for (int dz = -1; dz <= 1; ++dz) {
        [unroll]
        for (int dy = -1; dy <= 1; ++dy) {
            [unroll]
            for (int dx = -1; dx <= 1; ++dx) {
                float3 o  = pt_cell_feature(cellIdx.x + dx,
                                            cellIdx.y + dy,
                                            cellIdx.z + dz);
                float3 dv = float3(float(dx) + o.x - frac.x,
                                    float(dy) + o.y - frac.y,
                                    float(dz) + o.z - frac.z);
                float  d  = sqrt(dot(dv, dv) + 1e-12f);
                float  w  = exp(-k * d);
                sumExp += w;
                float3 gd = dv * (-1.0f / d);
                sumWeightedDir += gd * w;
            }
        }
    }

    const float denom = sumExp + 1e-30f;

    const float raw = -log(denom) / k;
    if (raw <= 0.0f) {
        gradOut = float3(0, 0, 0);
        return 0.0f;
    }
    gradOut = sumWeightedDir * (1.0f / denom);
    return raw;
}

inline float pt_fbm_split(int3 POrigin, float3 PLocal, float footprintM,
                          out float3 gradOut) {
    // Keep planet-scale noise split into exact and local parts.
    const float3 POriginF = float3(POrigin);

    float  value = 0.0f;
    gradOut      = float3(0, 0, 0);

    float       period = PT_TOP_PERIOD_M;
    float       amp    = PT_TOP_AMP_M;
    float       freq   = 1.0f / period;
    const float nyquistThresh = 2.0f * footprintM;

    [loop]
    for (int i = 0; i < PT_MAX_OCTAVES; ++i) {
        if (period < nyquistThresh)   break;
        if (period < PT_BOTTOM_PERIOD_M) break;

        const float3 scaledOrigin     = POriginF * freq;
        const int3   scaledOrigin_int = int3(floor(scaledOrigin));
        const float3 scaledOrigin_frac = scaledOrigin - float3(scaledOrigin_int);

        const float3 scaledLocal = PLocal * freq;

        const float3 totalLocal = scaledLocal + scaledOrigin_frac;
        const int3   extra_int  = int3(floor(totalLocal));
        const float3 frac       = totalLocal - float3(extra_int);

        const int3   cellIdx    = scaledOrigin_int + extra_int;

        float3 octGrad;
        const float octVal = pt_worley_smooth(cellIdx, frac, octGrad);

        value   += amp * octVal;
        gradOut += amp * freq * octGrad;

        period /= PT_LACUNARITY;
        amp    *= PT_GAIN;
        freq   *= PT_LACUNARITY;
    }
    return value;
}

// Evaluate layered terrain noise with footprint-aware octaves.
inline float pt_fbm(float3 P, float footprintM, out float3 gradOut) {
    const int3   POrigin = int3(round(P));
    const float3 PLocal  = P - float3(POrigin);
    return pt_fbm_split(POrigin, PLocal, footprintM, gradOut);
}

#define PTD_TOP_PERIOD_M    4.0f
#define PTD_BOTTOM_PERIOD_M 0.03f
#define PTD_TOP_AMP_M       0.18f
#define PTD_GAIN            0.55f
#define PTD_MAX_OCTAVES     8
#define PTD_MAX_DIST        180.0f
#define PTD_PIXEL_ANGLE     0.0013f
#define PTD_STRENGTH        1.0f

inline float3 pt_detail_gradient(int3 POrigin, float3 PLocal, float footprintM) {
    const float3 POriginF = float3(POrigin);
    float3 grad   = float3(0.0f, 0.0f, 0.0f);
    float  period = PTD_TOP_PERIOD_M;
    float  amp    = PTD_TOP_AMP_M;
    float  freq   = 1.0f / period;
    const float nyquistThresh = 2.0f * footprintM;

    [loop]
    for (int i = 0; i < PTD_MAX_OCTAVES; ++i) {
        if (period < nyquistThresh)       break;
        if (period < PTD_BOTTOM_PERIOD_M) break;

        const float3 scaledOrigin      = POriginF * freq;
        const int3   scaledOrigin_int  = int3(floor(scaledOrigin));
        const float3 scaledOrigin_frac = scaledOrigin - float3(scaledOrigin_int);
        const float3 totalLocal        = PLocal * freq + scaledOrigin_frac;
        const int3   extra_int         = int3(floor(totalLocal));
        const float3 frac              = totalLocal - float3(extra_int);
        const int3   cellIdx           = scaledOrigin_int + extra_int;

        float3 octGrad;
        pt_worley_smooth(cellIdx, frac, octGrad);
        grad += amp * freq * octGrad;

        period /= PT_LACUNARITY;
        amp    *= PTD_GAIN;
        freq   *= PT_LACUNARITY;
    }
    return grad;
}

// Perturb a terrain normal using the local noise gradient.
inline float3 TerrainDetailNormal(int3 POrigin, float3 PLocal, float footprintM,
                                  float3 N, float strength) {
    if (strength <= 0.0f) return N;
    float3 g = pt_detail_gradient(POrigin, PLocal, footprintM) * strength;
    g -= dot(g, N) * N;
    return normalize(N - g);
}

#endif
