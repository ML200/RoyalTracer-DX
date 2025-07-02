// Simple rejection checks

//NVIDIAs plane distance rejection
inline float DDistanceWeight(
    float3 currentPos,
    float3 otherPos,
    float3 currentNormal)
{
    float dist = abs(dot(currentPos - otherPos, currentNormal));
    float w    = 1.0 - dist * (1.0 / PLANE_DISTANCE_THRESHOLD);
    return saturate(w);
}

/*inline float DNormalWeight(float3 N0, float3 N1)
{
    float c = clamp(dot(N0, N1), 0.0, 1.0);
    return pow(c, NORMAL_POWER);
}*/

inline float DNormalWeight(float3 N0, float3 N1)
{
    float cosAngle = dot(N0, N1);
    return (cosAngle >= 0.99f) ? 1.0f : 0.0f;
}


// The more differnet the albedo, the lower the weight (exponential)
inline float DAlbedoWeight(float3 albedo0, float3 albedo1)
{
    // Luminance is slightly cheaper than full Euclidean length
    float  d  = abs(dot(albedo0 - albedo1, float3(0.299, 0.587, 0.114)));
    return exp(-d / ALBEDO_SIGMA);
}

// Anything with emission != 0 should have 0 weight
inline float DEmissionWeight(uint emission){
    return emission == 0 ? 1.0f : 0.0f;
}

// Exponential weight reduction proportional on distance to the center pixel and roughness of the material
// -> the rougher, the stronger the center pixel
inline float DRoughnessWeight(float  distancePx,
                              float  roughC,
                              float  roughN)
{
    if (distancePx <= 0.0f)        // also catches very first neighbour = 0
        return 1.0f;

    // 1.  Distance-dependent attenuation (shinier ⇒ sharper)
    float  roughAvg   = 0.5f * (roughC + roughN);
    float  sigma      = lerp(SIGMA_SMOOTH, SIGMA_ROUGH, roughAvg);
    float  inv2Sigma2 = 1.0f / (2.0f * sigma * sigma);
    float  w_dist     = exp(-distancePx * distancePx * inv2Sigma2);

    // 2.  Roughness-mismatch attenuation
    float  dR         = abs(roughC - roughN);
    float  inv2SigR2  = 1.0f / (2.0f * ROUGH_DIFF_SIGMA * ROUGH_DIFF_SIGMA);
    float  w_mismatch = exp(-dR * dR * inv2SigR2);

    // 3.  Combined weight
    return w_dist * w_mismatch;
}

// Everything with a different object id should have 0 weight
inline float DObjIDWeight(uint objID0, uint objID1){
    return objID0!=objID1? 0.0f:1.0f;
}

// Everything with a different object id should have 0 weight
inline float DMWeight(float M)
{
    return saturate(M / (2.0f * TEMP_MCAP_DI));
}

#define LUMA_REL_SIGMA  0.15f
#define LUMA_KNEE       0.6f       // controls how much ever gets *truly* muted

inline float DLumaWeight(float3 colC, float3 colN)
{
    float lumC = dot(colC, float3(0.2126,0.7152,0.0722));
    float lumN = dot(colN, float3(0.2126,0.7152,0.0722));
    float d    = abs(lumC - lumN) / max(max(lumC, lumN), 1e-3f);

    float w    = exp(-d / LUMA_REL_SIGMA);   // classic bilateral core
    return w / (w + LUMA_KNEE);              // compress instead of kill
}



// Pre-computed B-spline kernel, Σ = 16  (centre=4, cross=2, corners=1)
static const float KERNEL[9] = {
    4.0f/16,  // center
    2.0f/16,  // left
    2.0f/16,  // right
    2.0f/16,  // up
    2.0f/16,  // down
    1.0f/16,  // upper-left
    1.0f/16,  // upper-right
    1.0f/16,  // lower-left
    1.0f/16   // lower-right
};

// Pixel offsets (matches KERNEL order)
static const int2  OFF[9] = {
    int2( 0,  0),
    int2(-1,  0), int2( 1,  0),
    int2( 0, -1), int2( 0,  1),
    int2(-1, -1), int2( 1, -1),
    int2(-1,  1), int2( 1,  1)
};


inline void LoadGBuffer(
    int2     pix,
    out float3 albedo,
    out float  emission,
    out float  roughness,
    out uint   objID,
    out float3 normal,
    out float3 position,
    out float  M)
{
    // Slice 2
    albedo = gScratchPing.Load(int3(pix, 2)).xyz;

    // Slice 3
    float4 emRouID = gScratchPing.Load(int3(pix, 3));
    emission  = emRouID.x;
    roughness = emRouID.y;
    objID     = asuint(emRouID.z);

    // Slice 4
    normal = gScratchPing.Load(int3(pix, 4)).xyz;

    // Slice 5
    float4 posMot = gScratchPing.Load(int3(pix, 5));
    position = posMot.xyz;
    M   = posMot.w;
}


float3 AtrousKernel(int2 pixelPos, int step, uint slice)
{
    /* ── Load centre sample (same as before) ─────────── */
    float3 albC, norC, posC;
    float  emiC, roughC, motC;
    uint   idC;
    LoadGBuffer(pixelPos,
                albC, emiC, roughC,
                idC , norC, posC, motC);

    float3 colC = gScratchPing[uint3(pixelPos, slice)].xyz;
    if (emiC != 0.0f)                  // emitters stay crisp
        return colC;

    /* --- Accumulate --------------------------------------------------- */
    float3 accum = 0.0f;

    [unroll]
    for (uint i = 0; i < 9; ++i)
    {
        const int2 uv = pixelPos + OFF[i] * step;

        /* Neighbour data -------------------------- */
        float3 albN, norN, posN;
        float  emiN, roughN, motN;
        uint   idN;
        LoadGBuffer(uv,
                    albN, emiN, roughN,
                    idN , norN, posN, motN);

        float3 colN  = gScratchPing[uint3(uv, slice)].xyz;

        /* Distance in pixel space (for roughness term) */
        const float distPx = length(float2(OFF[i])) * step;

        /* Feature-guided scalar in [0,1] ------------ */
        float guide =
              DDistanceWeight (posC, posN, norC)       *
              DNormalWeight   (norC, norN)             *
              DAlbedoWeight   (albC, albN)             *
              DEmissionWeight (emiN)                   *
              DRoughnessWeight(distPx, roughC, roughN) *
              DObjIDWeight    (idC,  idN)              *
              DMWeight        (motC)                   *
              DMWeight        (motN)                   *
              DLumaWeight     (colC, colN);

        /* Bias-free contribution -------------------- */
        float3 sample = lerp(colC, colN, guide);   // if guide==0 → centre
        accum += KERNEL[i] * sample;               // kernel sums to 1
    }

    return accum;            // no need for explicit normalisation!
}


