//====================================================================
// Nrc_v8.hlsli — NRC shared bindings + record helpers.
//
// Offsets and strides mirror rdn/NRC/NrcLayout.h. Raygen, the resolve
// compute shader, and any future NRC pass all include this header.
// Does NOT modify Includes_v8.hlsli's existing bindings.
//====================================================================

#ifndef NRC_V8_HLSLI
#define NRC_V8_HLSLI

//====================================================================
// CONSTANTS — keep in sync with NrcLayout.h
//====================================================================
static const uint NRC_RAW_INPUT_DIM         = 14u;
static const uint NRC_OUTPUT_DIM            = 3u;
static const uint NRC_INFERENCE_IN_STRIDE   = NRC_RAW_INPUT_DIM * 4u;  // 56
static const uint NRC_INFERENCE_OUT_STRIDE  = NRC_OUTPUT_DIM   * 4u;   // 12

static const uint NRC_TRAINING_TILE_SIDE    = 8u;
static const uint NRC_UNBIASED_DENOM        = 16u;

static const uint NRC_MAX_TRAINING_PATHS    = 65536u;
static const uint NRC_MAX_VERTICES_PER_PATH = 8u;

static const uint NRC_INVALID_SLOT          = 0xFFFFFFFFu;
static const uint NRC_INVALID_PATH          = 0xFFFFFFFFu;

// Tail kind enum — must match nrc::TailKind in NrcLayout.h
static const uint NRC_TAIL_INVALID = 0u;
static const uint NRC_TAIL_RR      = 1u;
static const uint NRC_TAIL_EMITTER = 2u;
static const uint NRC_TAIL_MISS    = 3u;
static const uint NRC_TAIL_CACHE   = 4u;

// TrainingPathMeta: one per pathId, in the first bytes of the buffer.
static const uint NRC_TPM_STRIDE       = 16u;
static const uint NRC_TPM_OFF_NUMVERTS = 0u;
static const uint NRC_TPM_OFF_TAILKIND = 4u;
static const uint NRC_TPM_OFF_INFSLOT  = 8u;
static const uint NRC_TPM_OFF_TAILRAD  = 12u;

// Bytes reserved for the full meta table before per-vertex records begin.
static const uint NRC_PATH_META_TOTAL  = NRC_MAX_TRAINING_PATHS * NRC_TPM_STRIDE;  // 1048576 at 65536

// TrainingVertex: 56 B features + 4 B L_neePk + 4 B betaLocalPk = 64 B.
static const uint NRC_TV_STRIDE        = 64u;
static const uint NRC_TV_OFF_RAW       = 0u;
static const uint NRC_TV_OFF_LNEE      = 56u;
static const uint NRC_TV_OFF_BETA      = 60u;

// PendingGI offsets (bytes)
static const uint NRC_PG_STRIDE             = 16u;
static const uint NRC_PG_OFF_SLOT           = 0u;
static const uint NRC_PG_OFF_THROUGHPUT     = 4u;
static const uint NRC_PG_OFF_TPOST          = 8u;
static const uint NRC_PG_OFF_PDF            = 12u;

// Counters offsets (bytes) — keep in sync with nrc::Counters.
static const uint NRC_C_OFF_INFERENCE_COUNT = 0u;
static const uint NRC_C_OFF_TRAINING_PATH   = 4u;

// Path class tags for the SER sort key in raygen.
static const uint NRC_CLASS_RENDER          = 0u;
static const uint NRC_CLASS_TRAIN_BIASED    = 1u;
static const uint NRC_CLASS_TRAIN_UNBIASED  = 2u;

// Push-constant flag bits — mirrors nrc::flags in NrcLayout.h.
static const uint NRC_FLAG_ENABLED    = 1u;
static const uint NRC_FLAG_TRAIN      = 2u;
static const uint NRC_FLAG_DEBUG_VIEW = 4u;

inline bool NrcIsEnabled()   { return (nrc_flags & NRC_FLAG_ENABLED)    != 0u; }
inline bool NrcIsTrainOn()   { return (nrc_flags & NRC_FLAG_TRAIN)      != 0u; }
inline bool NrcIsDebugView() { return (nrc_flags & NRC_FLAG_DEBUG_VIEW) != 0u; }

//====================================================================
// BUFFER BINDINGS
//====================================================================
// u40 .. u44 — reserved for NRC. The training buffer (u43) interleaves
// the TrainingPathMeta table at the head with per-vertex bucket storage
// after — helpers below add the meta offset automatically.
RWByteAddressBuffer g_NrcInferenceIn  : register(u40);
RWByteAddressBuffer g_NrcInferenceOut : register(u41);
RWByteAddressBuffer g_NrcPendingGI    : register(u42);
RWByteAddressBuffer g_NrcTrainRecords : register(u43);
RWByteAddressBuffer g_NrcCounters     : register(u44);

//====================================================================
// SPHERICAL DIRECTION ENCODING
//====================================================================
inline float2 NrcEncodeSphDir(float3 d)
{
    const float theta = acos(clamp(d.z, -1.0f, 1.0f));
    const float phi   = atan2(d.y, d.x);
    const float u = theta * (1.0f / 3.14159265358979f);
    const float v = (phi + 3.14159265358979f) * (1.0f / 6.28318530717959f);
    return float2(saturate(u), saturate(v));
}

inline float NrcEncodeRoughness(float r)
{
    return saturate(1.0f - exp(-max(r, 0.0f)));
}

//====================================================================
// FEATURE PACK
//====================================================================
inline void NrcBuildFeatures(
    float3 x, float3 o, float3 n,
    float  r,
    float3 alpha, float3 beta,
    out float features[14])
{
    const float2 sphO = NrcEncodeSphDir(o);
    const float2 sphN = NrcEncodeSphDir(n);

    features[0]  = x.x;
    features[1]  = x.y;
    features[2]  = x.z;
    features[3]  = sphO.x;
    features[4]  = sphO.y;
    features[5]  = sphN.x;
    features[6]  = sphN.y;
    features[7]  = NrcEncodeRoughness(r);
    features[8]  = alpha.x;
    features[9]  = alpha.y;
    features[10] = alpha.z;
    features[11] = beta.x;
    features[12] = beta.y;
    features[13] = beta.z;
}

//====================================================================
// AREA-SPREAD TERMINATION (paper §3.4)
//====================================================================
inline float NrcComputeA0(float hitT, float cosPrimary)
{
    const float denom = max(4.0f * 3.14159265358979f * cosPrimary, 1e-20f);
    return (hitT * hitT) / denom;
}

inline void NrcAccumulateA(inout float a, float hitT, float prevPdf, float cosHit)
{
    const float denom = max(prevPdf * cosHit, 1e-20f);
    a += sqrt((hitT * hitT) / denom);
}

inline bool NrcShouldCacheTerminate(int hitIdx, float a0, float a, bool cacheEligible, float kGate)
{
    return cacheEligible && hitIdx >= 3 && (a * a) > (kGate * a0);
}

//====================================================================
// INFERENCE INPUT / OUTPUT
//====================================================================
inline uint NrcAppendInference(uint capacity, float features[14])
{
    uint slot;
    g_NrcCounters.InterlockedAdd(NRC_C_OFF_INFERENCE_COUNT, 1u, slot);
    if (slot >= capacity) return NRC_INVALID_SLOT;

    const uint base = slot * NRC_INFERENCE_IN_STRIDE;
    [unroll]
    for (uint i = 0; i < 14u; ++i) {
        g_NrcInferenceIn.Store(base + i * 4u, asuint(features[i]));
    }
    return slot;
}

inline float3 NrcLoadInferenceOutput(uint slot)
{
    if (slot == NRC_INVALID_SLOT) return float3(0, 0, 0);
    const uint base = slot * NRC_INFERENCE_OUT_STRIDE;
    const uint3 raw = g_NrcInferenceOut.Load3(base);
    return asfloat(raw);
}

//====================================================================
// PENDING GI RECORD
//====================================================================
inline void NrcClearPendingGI(uint pixelIdx)
{
    g_NrcPendingGI.Store(pixelIdx * NRC_PG_STRIDE + NRC_PG_OFF_SLOT, NRC_INVALID_SLOT);
}

inline void NrcStorePendingGI(
    uint  pixelIdx,
    uint  inferenceSlot,
    uint  throughputPk,
    uint  tpostPk,
    float pdfProduct)
{
    const uint base = pixelIdx * NRC_PG_STRIDE;
    g_NrcPendingGI.Store4(base, uint4(
        inferenceSlot,
        throughputPk,
        tpostPk,
        asuint(pdfProduct)
    ));
}

inline void NrcLoadPendingGI(
    uint pixelIdx,
    out uint  inferenceSlot,
    out uint  throughputPk,
    out uint  tpostPk,
    out float pdfProduct)
{
    const uint base = pixelIdx * NRC_PG_STRIDE;
    const uint4 raw = g_NrcPendingGI.Load4(base);
    inferenceSlot = raw.x;
    throughputPk  = raw.y;
    tpostPk       = raw.z;
    pdfProduct    = asfloat(raw.w);
}

//====================================================================
// CACHE-TERMINATION RECORD (Pending GI)
//====================================================================
// Returns the inference slot on success, NRC_INVALID_SLOT on failure.
// The slot is useful to the caller for class-1 training paths, which
// also record it as the cache-tail slot in their path meta.
//
// Reflectance factorisation (paper §5): the network predicts irradiance
// = L_s / (α+β). To recover radiance at resolve time we'd need to
// multiply by (α+β), but we can hoist that factor onto `throughput`
// and `tpost` here instead — the resolve shader then treats the MLP
// output AS radiance, no shader-side multiply required.
inline uint NrcWriteTerminationRecord(
    uint   pixelIdx,
    uint   inferenceCapacity,
    float3 hitPos,
    float3 viewDir,
    float3 hitNormal,
    float  roughness,
    float3 kd,
    float  metallic,
    float3 throughput,
    float3 tpost,
    float  pdfProduct)
{
    const float3 alpha   = kd * (1.0f - metallic);
    const float3 betaC   = lerp(float3(0.04f, 0.04f, 0.04f), kd, metallic);
    const float3 reflSum = alpha + betaC;

    float features[14];
    NrcBuildFeatures(hitPos, viewDir, hitNormal, roughness, alpha, betaC, features);

    const uint slot = NrcAppendInference(inferenceCapacity, features);
    if (slot == NRC_INVALID_SLOT) return NRC_INVALID_SLOT;

    NrcStorePendingGI(
        pixelIdx,
        slot,
        PackRGB9E5(throughput * reflSum),
        PackRGB9E5(tpost      * reflSum),
        pdfProduct);
    return slot;
}

//====================================================================
// TRAINING PATH META
//====================================================================
// Allocate a new training path id. Returns NRC_INVALID_PATH if the per-
// frame quota is exhausted — caller should skip training for this pixel.
inline uint NrcAllocateTrainingPath()
{
    uint pathId;
    g_NrcCounters.InterlockedAdd(NRC_C_OFF_TRAINING_PATH, 1u, pathId);
    if (pathId >= NRC_MAX_TRAINING_PATHS) return NRC_INVALID_PATH;
    return pathId;
}

inline void NrcStorePathMeta(
    uint pathId,
    uint numVertices,
    uint tailKind,
    uint inferenceSlot,
    uint tailRadiancePk)
{
    if (pathId >= NRC_MAX_TRAINING_PATHS) return;
    const uint base = pathId * NRC_TPM_STRIDE;
    g_NrcTrainRecords.Store4(base, uint4(numVertices, tailKind, inferenceSlot, tailRadiancePk));
}

//====================================================================
// TRAINING VERTEX
//====================================================================
// Direct indexed write — no atomic, caller tracks vIdx locally.
inline void NrcStoreTrainingVertex(
    uint pathId,
    uint vIdx,
    float features[14],
    uint L_neePk,
    uint betaLocalPk)
{
    if (pathId >= NRC_MAX_TRAINING_PATHS || vIdx >= NRC_MAX_VERTICES_PER_PATH) return;
    const uint base = NRC_PATH_META_TOTAL +
                      (pathId * NRC_MAX_VERTICES_PER_PATH + vIdx) * NRC_TV_STRIDE;
    [unroll]
    for (uint i = 0; i < 14u; ++i) {
        g_NrcTrainRecords.Store(base + NRC_TV_OFF_RAW + i * 4u, asuint(features[i]));
    }
    g_NrcTrainRecords.Store(base + NRC_TV_OFF_LNEE, L_neePk);
    g_NrcTrainRecords.Store(base + NRC_TV_OFF_BETA, betaLocalPk);
}

// Second-pass update: raygen samples the BSDF after emitting the record,
// so the transport factor β = BSDF·cos·T / pdf is known only after.
inline void NrcUpdateTrainingVertexBeta(uint pathId, uint vIdx, uint betaLocalPk)
{
    if (pathId >= NRC_MAX_TRAINING_PATHS || vIdx >= NRC_MAX_VERTICES_PER_PATH) return;
    const uint base = NRC_PATH_META_TOTAL +
                      (pathId * NRC_MAX_VERTICES_PER_PATH + vIdx) * NRC_TV_STRIDE;
    g_NrcTrainRecords.Store(base + NRC_TV_OFF_BETA, betaLocalPk);
}

//====================================================================
// PATH-CLASS CLASSIFICATION
//====================================================================
inline uint NrcClassifyPixel(
    uint2 pixel,
    uint  frameIndex,
    uint2 perFrameTileOffset,
    out bool isTraining,
    out bool isUnbiased)
{
    const uint2 tileOrigin = (pixel / NRC_TRAINING_TILE_SIDE) * NRC_TRAINING_TILE_SIDE;
    const uint2 local      = pixel - tileOrigin;
    isTraining = all(local == perFrameTileOffset);

    isUnbiased = false;
    if (isTraining)
    {
        uint h = (tileOrigin.x * 0x9E3779B9u) ^ (tileOrigin.y * 0x85EBCA6Bu) ^ frameIndex;
        h ^= h >> 16; h *= 0xC2B2AE35u;
        h ^= h >> 13; h *= 0x27D4EB2Fu;
        h ^= h >> 16;
        isUnbiased = (h % NRC_UNBIASED_DENOM) == 0u;
    }

    if (!isTraining)           return NRC_CLASS_RENDER;
    else if (isUnbiased)       return NRC_CLASS_TRAIN_UNBIASED;
    else                       return NRC_CLASS_TRAIN_BIASED;
}

#endif // NRC_V8_HLSLI
