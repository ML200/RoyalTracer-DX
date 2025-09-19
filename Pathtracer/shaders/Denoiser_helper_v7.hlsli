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

//--------------------------------------------------------------------
// Luminance-guided bilateral weight with
//  • inverse scaling by ReSTIR weight M   (low-M  ⇒   weight→1)
//  • gentler behaviour in very dark tones
//--------------------------------------------------------------------
#define LUMA_REL_SIGMA   0.2f    // bilateral fall-off (unchanged)
#define LUMA_KNEE        0.1f    // compression knee    (unchanged)
#define LUMA_EPS         1e-2f    // protects deep-shadow division
#define M_REF            (2.0f * TEMP_MCAP_DI)   // same normalisation you use elsewhere

inline float DLumaWeight(float3 colC,
                         float3 colN,
                         float   M)      // ← NEW PARAM
{
    //----------------------------------------------------------------
    // 1. Luminance in *linear* space (Rec.-709 coeffs are fine here)
    //----------------------------------------------------------------
    float lumC = dot(colC, float3(0.2126, 0.7152, 0.0722));
    float lumN = dot(colN, float3(0.2126, 0.7152, 0.0722));

    //----------------------------------------------------------------
    // 2. Relative difference – robust in deep shadows
    //----------------------------------------------------------------
    float relMax   = max(max(lumC, lumN), LUMA_EPS);     // avoid blow-up
    float d        = abs(lumC - lumN) / relMax;          //   ∈ [0,∞)

    //----------------------------------------------------------------
    // 3. Classic bilateral ⇒ compressed (same as before)
    //----------------------------------------------------------------
    float wLum     = exp(-d / LUMA_REL_SIGMA);           // bilateral core
    wLum           = wLum / (wLum + LUMA_KNEE);          // knee compression

    //----------------------------------------------------------------
    // 4. Inverse scaling by ReSTIR weight M
    //    • normalise M to [0,1]   (≈ confidence in the *current* sample)
    //    • blend towards 1 as confidence drops
    //----------------------------------------------------------------
    float invM     = 1.0f - saturate(M / M_REF);         // low-M ⇒ invM→1
    float wMerged  = lerp(wLum, 1.0f, invM);             // low-M relaxes test

    return wMerged;
}



// Pre-computed B-spline kernel, Σ = 16  (centre=4, cross=2, corners=1)
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
    /* ── 1. Centre sample ─────────────────────────────────────────── */
    float3 albC, norC, posC;
    float  emiC, roughC, motC;
    uint   idC;
    LoadGBuffer(pixelPos,
                albC, emiC, roughC,
                idC , norC, posC, motC);

    float3 colC = gScratchPing[uint3(pixelPos, slice)].xyz;

    // Keep emissive surfaces untouched
    if (emiC != 0.0f)
        return colC;

    /* ── 2. First pass: compute weights only ─────────────────────── */
    float   w[9];          // per-tap weights
    float   wSum = 0.0f;   // total (for normalisation)

    [unroll]
    for (uint i = 0; i < 9; ++i)
    {
        const int2 uv = pixelPos + OFF[i] * step;

        /* Neighbour attributes ------------------------------------ */
        float3 albN, norN, posN;
        float  emiN, roughN, motN;
        uint   idN;
        LoadGBuffer(uv,
                    albN, emiN, roughN,
                    idN , norN, posN, motN);

        /* Early-out: skip emissive neighbours – they never contribute */
        if (emiN != 0.0f)
        {
            w[i] = 0.0f;
            continue;
        }

        /* Pixel-space distance for roughness falloff */
        const float distPx = length(float2(OFF[i])) * step;

        /* Feature-guided bilateral term (∈ [0,1]) */
        float3 colN = gScratchPing[uint3(uv, slice)].xyz;
        const float guide =
              DDistanceWeight (posC, posN, norC)       *
              DNormalWeight   (norC, norN)             *
              DAlbedoWeight   (albC, albN)             *
              DEmissionWeight (emiN)                   *
              DRoughnessWeight(distPx, roughC, roughN) *
              DObjIDWeight    (idC,  idN)              *
              DLumaWeight     (colC, colN, motN);

        /* Spatial kernel (e.g. Gaussian) --------------------------- */
        w[i] = KERNEL[i] * guide;

        wSum += w[i];
    }

    /* Robust fallback: if all weights vanished, keep centre sample */
    if (wSum == 0.0f)
        return colC;

    /* ── 3. Second pass: accumulate with **normalised** weights ─── */
    float3 accum = 0.0f;

    [unroll]
    for (uint i = 0; i < 9; ++i)
    {
        if (w[i] == 0.0f)          // cheap branch; avoid useless reads
            continue;

        const int2 uv = pixelPos + OFF[i] * step;
        float3 colN = gScratchPing[uint3(uv, slice)].xyz;

        /* Bias-free lerp toward C as in original */
        float3 sample = lerp(colC, colN,           // if guide==0 → centre
                             w[i] / KERNEL[i]);    // undo spatial kernel part

        accum += (w[i] / wSum) * sample;           // **normalised** weight
    }

    return accum;  // fully normalised, no extra division needed
}

// Returns UV in [0,1]^2 with a TOP-LEFT origin (y flipped), or (-1,-1) on failure.
// Use this if you later convert to pixel coords the same way your other code does.
inline float2 GetLastFrameUV(
    float3   worldPos,
    float4x4 prevView,
    float4x4 prevProjection,   // include previous frame's jitter
    uint     objID)
{
    // current world -> local (using provided inverse)
    float4 localPos     = mul(instanceProps[objID].objectToWorldInverse, float4(worldPos, 1.0f));
    // local -> previous world (using previous instance transform)
    float4 prevWorldPos = mul(instanceProps[objID].prevObjectToWorld, localPos);

    // previous world -> previous clip
    float4 clipPos = mul(prevProjection, mul(prevView, prevWorldPos));
    if (clipPos.w <= 0.0f)
        return float2(-1.0f, -1.0f);

    // clip -> NDC -> UV
    float2 ndc = clipPos.xy / clipPos.w;          // [-1,1]
    float2 uv  = ndc * 0.5f + 0.5f;               // [0,1], bottom-left origin
    uv.y       = 1.0f - uv.y;                     // flip to top-left origin

    // bounds check
    if (any(uv < 0.0f) || any(uv > 1.0f))
        return float2(-1.0f, -1.0f);

    return uv;
}




//--------------------------------------------------------------------
//  Bilinear reprojection with inverse-distance weighting
//  • returns four neighbours + weight, already normalised so Σw = 1
//  • if GetLastFrameUV() fails → all weights are 0 and pix = kInvalidPixel
//--------------------------------------------------------------------
inline void GetBilinearReprojectedPixels_d(
    float3      worldPos,
    float4x4    prevView,
    float4x4    prevProjection,     // must already contain last frame’s jitter
    float2      resolution,         // (width, height)
    uint        objID,
    out WeightedPixel outPx[4]      // order: (0,0) (1,0) (0,1) (1,1)
)
{
    //----------------------------------------------------------------
    // 1. Reproject to previous-frame UV ------------------------------
    //----------------------------------------------------------------
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

    //----------------------------------------------------------------
    // 2. Continuous pixel position & fractional part -----------------
    //----------------------------------------------------------------
    //   Our earlier helper rounded with “+0.5” → pixel centres live
    //   at integer coordinates.  Undo that so that (0,0) sits at the
    //       centre of the **first** pixel and we can do bilinear maths.
    //----------------------------------------------------------------
    float2 posF   = uv * resolution;   // continuous pixel position
    int2   pix00  = int2(floor(posF));        // top-left integer pixel
    float2 frac   = posF - pix00;             // (= uv in [0,1) inside cell)

    //----------------------------------------------------------------
    // 3. Prepare the 4 neighbours ------------------------------------
    //----------------------------------------------------------------
    const int2 offs[4] = { int2(0,0), int2(1,0), int2(0,1), int2(1,1) };
    float      basis[4] =               // pure bilinear kernel
    {
        (1.0f - frac.x) * (1.0f - frac.y),   // w00
        frac.x          * (1.0f - frac.y),   // w10
        (1.0f - frac.x) * frac.y,            // w01
        frac.x          * frac.y             // w11
    };

    //----------------------------------------------------------------
    // 4. Combine bilinear kernel with inverse-distance fall-off ------
    //----------------------------------------------------------------
    //   gScratchPing stores previous-frame *world* positions in layer 10
    //   (same layout you hinted at).  We weight each neighbour by
    //
    //        w = bilinearKernel / (ε + ‖Δworld‖)
    //
    //   so that:
    //   • far-away re-projected pixels contribute less
    //   • all four weights are renormalised to Σw = 1
    //----------------------------------------------------------------
    float  sumW = 0.0f;
    [unroll] for (int i = 0; i < 4; ++i)
    {
        int2 p = pix00 + offs[i];

        // Cull neighbours that ended up off-screen (can happen at borders)
        if (any(p < 0) || any(p >= int2(resolution)))
        {
            outPx[i].pix = kInvalidPixel;
            outPx[i].w   = 0.0f;
            continue;
        }

        float3 prevWorld = gScratchPing[uint3(p, 10)].xyz;
        float  dist      = length(prevWorld - worldPos);     // metres (or whatever unit)
        float  w         = basis[i] / (dist + 1e-4f);        // ε avoids div-by-zero

        outPx[i].pix = p;
        outPx[i].w   = w;
        sumW        += w;
    }

    //----------------------------------------------------------------
    // 5. Normalise so that Σw = 1 ------------------------------------
    //----------------------------------------------------------------
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