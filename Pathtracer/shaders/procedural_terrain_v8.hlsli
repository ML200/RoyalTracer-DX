#ifndef PROCEDURAL_TERRAIN_V8_HLSLI
#define PROCEDURAL_TERRAIN_V8_HLSLI

//====================================
//PROCEDURAL TERRAIN NOISE (GPU twin)
//====================================
//Mirror of Pathtracer/include/procedural_terrain.h. The CPU tessellator
//bakes pt_fbm() into the mesh height; this file evaluates the SAME noise's
//analytic gradient at shade time so the BRDF normal agrees with what the
//shadow rays trace against the mesh.
//
//Every constant and the hash function MUST stay in sync with the CPU
//header. The hash is integer-only (no sin tricks) so CPU and GPU produce
//byte-identical output for matching inputs.
//
//Planet-scale precision. CPU samples noise with double precision at world
//positions like `dir * R_planet` (~6.5e6 m). Naively done on the GPU in
//fp32 that drops the sub-metre fractional cell position - the smallest
//octave (1m period) ends up with random fractional positions per pixel
//because fp32 LSB at 6.5e6 is ~0.5 m. Mesh has crisp bumps, shading reads
//garbage gradients, result is pixelation.
//
//Fix: split each per-octave input into (integer-metres part, [-0.5, 0.5]
//fractional part). The integer is exact in fp32 up to 2^23 = 8.4M (comfy
//for 6.5M planet scale). With every period a power of 2 (PT_TOP_PERIOD_M
//= 16384 below), every frequency is a clean exponent shift and every
//`int_metres * freq` is exact. The cell-index math then never forms a
//lossy big+small sum.

//====================================
//Tuning (mirror of procedural_terrain.h)
//====================================
//PT_TOP_PERIOD_M is a power of 2 so freq_i = 2^(i-14) is exact in fp32
//and `int_metres * freq_i` does not lose precision (required by the
//split-position math in pt_fbm).
#define PT_TOP_PERIOD_M     16384.0f
//BOTTOM_PERIOD 16: Mars sandy-plains tuning, raised 1 -> 16 m to cut the
//1..8 m gravel octaves that read as fractured rock everywhere. Mirrors the
//CPU header's PROCTERRAIN_BOTTOM_PERIOD_M.
#define PT_BOTTOM_PERIOD_M  16.0f
//TOP_AMP 60: Mars sandy-plains tuning, 120 -> 60 m for a flatter, low-relief
//silhouette. Mirrors the CPU header's PROCTERRAIN_TOP_AMP_M.
#define PT_TOP_AMP_M        60.0f
#define PT_LACUNARITY       2.0f
//GAIN 0.55: Mars sandy-plains tuning, 0.70 -> 0.55 (near-classical FBM) so
//fine octaves stay gentle rather than reading as cobbles. Raise back toward
//0.70 for a rockier surface. Mirrors the CPU header's PROCTERRAIN_GAIN.
#define PT_GAIN             0.55f
#define PT_MAX_OCTAVES      15
//SMOOTH_K 24: Mars sandy-plains tuning, 48 -> 24 for rounded dune-like
//hummocks instead of fractured-rock cell edges, still C1 continuous. Raise
//back toward 48 for a rockier surface. Mirrors the CPU header.
#define PT_SMOOTH_K         24.0f
#define PT_SEED             0xA5C9E1B7u

//====================================
//Integer hash (PCG-derived) - deterministic and identical to the CPU
//pt_hash_u32. uint arithmetic wraps modulo 2^32 in both HLSL and C++.
//====================================
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
    const float inv = 1.0f / 16777216.0f;  // 1 / 2^24
    float3 o;
    o.x = float(h & 0xFFFFFFu) * inv;
    h *= 1664525u;
    o.y = float(h & 0xFFFFFFu) * inv;
    h *= 1664525u;
    o.z = float(h & 0xFFFFFFu) * inv;
    return o;
}

//Smooth Worley F1 over the 3x3x3 cell neighbourhood, taking a pre-split
//(cell, frac) so the caller can supply precision-preserving cell indices
//computed from the planet-scale input. cellIdx is the cell that owns frac
//(frac is in [0, 1) within that cell); see pt_fbm below for how the split
//is computed without losing fp32 precision.
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
                float3 gd = dv * (-1.0f / d);   // d(d)/d(frac) = (frac - feature)/d
                sumWeightedDir += gd * w;
            }
        }
    }

    const float denom = sumExp + 1e-30f;
    //Smooth-min distance drifts negative when feature points cluster near
    //the sample (sumExp > 1). True nearest-feature distance is >= 0, so
    //clamp here. Mirrors the CPU side in procedural_terrain.h - without
    //this, the mesh dips INWARD at deterministic positions, and the
    //shading gradient at those positions points into the surface.
    const float raw = -log(denom) / k;
    if (raw <= 0.0f) {
        gradOut = float3(0, 0, 0);
        return 0.0f;
    }
    gradOut = sumWeightedDir * (1.0f / denom);
    return raw;
}

//Worley FBM with Nyquist gate, taking a PRE-SPLIT planet-scale position.
//
//The noise samples the position P = POrigin + PLocal (metres, planet-centre
//relative). POrigin is the integer-metre part (exact in fp32 up to 2^24),
//PLocal is the metre-scale remainder. Keeping them separate is the whole
//point: the caller must NEVER form the planet-scale float `POrigin+PLocal`
//and pass it in, because at ~6.4e6 m the fp32 LSB is ~0.5 m and the finest
//octave (1 m cells) loses ~half a cell -> per-pixel noise jitter, the
//"pixelated / smooth-strip / detail-won't-track-the-camera" symptoms. The
//renderer builds (POrigin, PLocal) in a CAMERA-LOCAL frame: POrigin is the
//camera's surface point (a per-frame integer from the FP64 CPU side) and
//PLocal is the hit's small, precise tangential offset from it. PLocal may
//exceed 1 m for hits far from the camera - the split math below handles any
//magnitude; precision just follows fp32 at PLocal's scale, which is exactly
//what we want (far hits gate their fine octaves anyway).
//
//PT_TOP_PERIOD_M = 16384 makes all freq = 2^k, so POriginF * freq is an
//exact exponent shift (no mantissa rounding) and the integer/fractional
//split below is bit-clean.
inline float pt_fbm_split(int3 POrigin, float3 PLocal, float footprintM,
                          out float3 gradOut) {
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

        //POriginF * freq: exact for freq = power of 2 and POriginF < 2^24.
        //Splitting it into integer + [0, 1) fractional preserves all bits.
        const float3 scaledOrigin     = POriginF * freq;
        const int3   scaledOrigin_int = int3(floor(scaledOrigin));
        const float3 scaledOrigin_frac = scaledOrigin - float3(scaledOrigin_int);
        //scaledLocal stays in fp32's precise zone for |PLocal| up to the
        //tens-of-km range (fine octaves gate out before precision bites).
        const float3 scaledLocal = PLocal * freq;
        //Combine the two fractional parts (still small / precise), then
        //extract the residual cell offset and the [0, 1) within-cell frac.
        const float3 totalLocal = scaledLocal + scaledOrigin_frac;
        const int3   extra_int  = int3(floor(totalLocal));
        const float3 frac       = totalLocal - float3(extra_int);
        //Total cell index = integer-part-of-origin + residual. Equivalent
        //to floor(P * freq) computed in double, but never lost precision.
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

//Convenience wrapper that splits a planet-scale float P internally. Retained
//for any caller that genuinely has a small / near-origin P; AVOID for
//planet-scale inputs - prefer pt_fbm_split with a camera-local (POrigin,
//PLocal) so the planet-scale magnitude never touches an fp32 float.
inline float pt_fbm(float3 P, float footprintM, out float3 gradOut) {
    const int3   POrigin = int3(round(P));
    const float3 PLocal  = P - float3(POrigin);
    return pt_fbm_split(POrigin, PLocal, footprintM, gradOut);
}

//====================================
//RUNTIME DETAIL NORMAL (pebbles + sand) - normal-only micro-bump
//====================================
//The baked mesh stops at PT_BOTTOM_PERIOD_M (16 m). This adds shading-normal
//detail BELOW that - 4 m pebble clusters down to ~3 cm sand grain - WITHOUT
//touching geometry (no BLAS cost). Same smooth-Worley primitive as the mesh
//so the look stays consistent. Octaves gate out by the pixel footprint
//(Nyquist) so distant terrain stays smooth / alias-free, and the whole effect
//is distance-capped for perf.
//
//Precision: POrigin must be an exact integer (<= 2^24) and every octave freq a
//power of 2, so POrigin*freq is an exact fp32 exponent shift even at the
//~6.4e6 m planet scale (same split-precision trick as pt_fbm_split). Keep
//PTD_TOP_PERIOD_M a power of 2 and rely on PT_LACUNARITY = 2.
#define PTD_TOP_PERIOD_M    4.0f      // coarsest detail octave (m), power of 2
#define PTD_BOTTOM_PERIOD_M 0.03f     // ~3 cm finest grain (cutoff)
#define PTD_TOP_AMP_M       0.18f     // height of the coarsest detail octave (m) - tune to taste
#define PTD_GAIN            0.55f
#define PTD_MAX_OCTAVES     8
#define PTD_MAX_DIST        180.0f    // skip detail beyond this (m) - perf
#define PTD_PIXEL_ANGLE     0.0013f   // ~vertical radians/pixel (~60 deg FOV / 800 px); tune for FOV/res
#define PTD_STRENGTH        1.0f      // master detail strength; 0 disables

//World-space gradient (rise/run) of the fine detail height field at absolute
//position (POrigin + PLocal) metres. footprintM gates the fine octaves.
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

        const float3 scaledOrigin      = POriginF * freq;          // exact: freq is 2^k
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

//Perturb unit world normal N with the detail bump. POrigin/PLocal are the
//split absolute world position; strength scales the effect (0 = no-op).
inline float3 TerrainDetailNormal(int3 POrigin, float3 PLocal, float footprintM,
                                  float3 N, float strength) {
    if (strength <= 0.0f) return N;
    float3 g = pt_detail_gradient(POrigin, PLocal, footprintM) * strength;
    g -= dot(g, N) * N;                          // keep only the tangential slope
    return normalize(N - g);                     // tilt the normal against the slope
}

#endif
