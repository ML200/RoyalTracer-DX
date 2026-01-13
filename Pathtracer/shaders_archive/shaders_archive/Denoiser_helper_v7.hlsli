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
    if (distancePx <= 0.0f)
        return 1.0f;
    float  roughAvg   = 0.5f * (roughC + roughN);
    float  sigma      = lerp(SIGMA_SMOOTH, SIGMA_ROUGH, roughAvg);
    float  inv2Sigma2 = 1.0f / (2.0f * sigma * sigma);
    float  w_dist     = exp(-distancePx * distancePx * inv2Sigma2);

    float  dR         = abs(roughC - roughN);
    float  inv2SigR2  = 1.0f / (2.0f * ROUGH_DIFF_SIGMA * ROUGH_DIFF_SIGMA);
    float  w_mismatch = exp(-dR * dR * inv2SigR2);

    return w_dist * w_mismatch;
}

// Everything with a different object id should have 0 weight
inline float DObjIDWeight(uint objID0, uint objID1){
    return objID0!=objID1? 0.0f:1.0f;
}

// how converged is the restir pixel?
inline float DMWeight(float M)
{
    return saturate(M / (2.0f * TEMP_MCAP_DI));
}

#define LUMA_REL_SIGMA   0.2f    // bilateral fall-off
#define LUMA_KNEE        0.1f    // compression knee
#define LUMA_EPS         1e-2f    // protects deep-shadow division
#define M_REF            (2.0f * TEMP_MCAP_DI)

inline float DLumaWeight(float3 colC,
                         float3 colN,
                         float   M)
{
    float lumC = dot(colC, float3(0.2126, 0.7152, 0.0722));
    float lumN = dot(colN, float3(0.2126, 0.7152, 0.0722));

    float relMax   = max(max(lumC, lumN), LUMA_EPS);
    float d        = abs(lumC - lumN) / relMax;

    float wLum     = exp(-d / LUMA_REL_SIGMA);
    wLum           = wLum / (wLum + LUMA_KNEE);

    float invM     = 1.0f - saturate(M / M_REF);
    float wMerged  = lerp(wLum, 1.0f, invM);

    return wMerged;
}


static const float KERNEL[9] = {
    4.0f,  // center
    2.0f,  // left
    2.0f,  // right
    2.0f,  // up
    2.0f,  // down
    1.0f,  // upper-left
    1.0f,  // upper-right
    1.0f,  // lower-left
    1.0f   // lower-right
};

// Pixel offsets
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
    // center pixel
    float3 albC, norC, posC;
    float  emiC, roughC, motC;
    uint   idC;
    LoadGBuffer(pixelPos,
                albC, emiC, roughC,
                idC , norC, posC, motC);

    float3 colC = gScratchPing[uint3(pixelPos, slice)].xyz;

    // Keep emissive surfaces untouched!
    if (emiC != 0.0f)
        return colC;

    // weights
    float   w[9];          // per-tap weights
    float   wSum = 0.0f;   // total (for normalisation)

    [unroll]
    for (uint i = 0; i < 9; ++i)
    {
        const int2 uv = pixelPos + OFF[i] * step;

        float3 albN, norN, posN;
        float  emiN, roughN, motN;
        uint   idN;
        LoadGBuffer(uv,
                    albN, emiN, roughN,
                    idN , norN, posN, motN);

        // ignore emissives completely
        if (emiN != 0.0f)
        {
            w[i] = 0.0f;
            continue;
        }

        const float distPx = length(float2(OFF[i])) * step;

        float3 colN = gScratchPing[uint3(uv, slice)].xyz;
        const float guide =
              DDistanceWeight (posC, posN, norC)       *
              DNormalWeight   (norC, norN)             *
              DAlbedoWeight   (albC, albN)             *
              DEmissionWeight (emiN)                   *
              DRoughnessWeight(distPx, roughC, roughN) *
              DObjIDWeight    (idC,  idN)              *
              DLumaWeight     (colC, colN, motN);

        w[i] = KERNEL[i] * guide;

        wSum += w[i];
    }

    // Bad neigbors? fallback to base
    if (wSum == 0.0f)
        return colC;

    // accumulate
    float3 accum = 0.0f;

    [unroll]
    for (uint i = 0; i < 9; ++i)
    {
        if (w[i] == 0.0f)
            continue;

        const int2 uv = pixelPos + OFF[i] * step;
        float3 colN = gScratchPing[uint3(uv, slice)].xyz;

        float3 sample = lerp(colC, colN,
                             w[i] / KERNEL[i]);

        accum += (w[i] / wSum) * sample;
    }

    return accum;
}

// Reprojection
inline float2 GetLastFrameUV(
    float3   worldPos,
    float4x4 prevView,
    float4x4 prevProjection,
    uint     objID)
{
    // current world -> local
    float4 localPos     = mul(instanceProps[objID].objectToWorldInverse, float4(worldPos, 1.0f));
    // local -> previous world
    float4 prevWorldPos = mul(instanceProps[objID].prevObjectToWorld, localPos);

    // previous world -> previous clip
    float4 clipPos = mul(prevProjection, mul(prevView, prevWorldPos));
    if (clipPos.w <= 0.0f)
        return float2(-1.0f, -1.0f);

    // clip -> NDC -> UV
    float2 ndc = clipPos.xy / clipPos.w;
    float2 uv  = ndc * 0.5f + 0.5f;
    uv.y       = 1.0f - uv.y;

    // bounds check
    if (any(uv < 0.0f) || any(uv > 1.0f))
        return float2(-1.0f, -1.0f);

    return uv;
}


//  Bilinear reprojection with inverse-distance weighting
inline void GetBilinearReprojectedPixels_d(
    float3      worldPos,
    float4x4    prevView,
    float4x4    prevProjection,
    float2      resolution,
    uint        objID,
    out WeightedPixel outPx[4]
)
{
    // Reproject
    float2 uv = GetLastFrameUV(worldPos, prevView, prevProjection, objID);

    // Early out on failure
    if (uv.x < 0.0f)
    {
        [unroll] for (int i = 0; i < 4; ++i)
        {
            outPx[i].pix = kInvalidPixel;
            outPx[i].w   = 0.0f;
        }
        return;
    }

    float2 posF   = uv * resolution;   // continuous pixel position
    int2   pix00  = int2(floor(posF));        // top-left integer pixel
    float2 frac   = posF - pix00;

    // all possible neighbors
    const int2 offs[4] = { int2(0,0), int2(1,0), int2(0,1), int2(1,1) };
    float      basis[4] =
    {
        (1.0f - frac.x) * (1.0f - frac.y),
        frac.x          * (1.0f - frac.y),
        (1.0f - frac.x) * frac.y,
        frac.x          * frac.y
    };

    float  sumW = 0.0f;
    [unroll] for (int i = 0; i < 4; ++i)
    {
        int2 p = pix00 + offs[i];

        // Cull neighbours that ended up off-screen
        if (any(p < 0) || any(p >= int2(resolution)))
        {
            outPx[i].pix = kInvalidPixel;
            outPx[i].w   = 0.0f;
            continue;
        }

        float3 prevWorld = gScratchPing[uint3(p, 10)].xyz;
        float  dist      = length(prevWorld - worldPos);
        float  w         = basis[i] / (dist + 1e-4f);

        outPx[i].pix = p;
        outPx[i].w   = w;
        sumW        += w;
    }

    // normalize
    if (sumW > 0.0f)
    {
        float invSum = rcp(sumW);        // ← keep using the intrinsic
        [unroll] for (int i = 0; i < 4; ++i)
            outPx[i].w *= invSum;
    }
    else
    {
        [unroll] for (int i = 0; i < 4; ++i)
            outPx[i].w = 0.0f;
    }
}