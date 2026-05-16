//====================================
//NRC SHARED BINDINGS AND RECORD HELPERS
//====================================
//offsets mirror rdn/NRC/NrcLayout.h

#ifndef NRC_V8_HLSLI
#define NRC_V8_HLSLI

//====================================
//CONSTANTS
//====================================
//bumped to 17 to add an explicit side bit (+1 front / -1 back) at index 16
//for thin-sheet front/back disambiguation, fed through Identity encoding
static const uint NRC_RAW_INPUT_DIM         = 17u;
static const uint NRC_OUTPUT_DIM            = 3u;
static const uint NRC_INFERENCE_IN_STRIDE   = NRC_RAW_INPUT_DIM * 4u;
static const uint NRC_INFERENCE_OUT_STRIDE  = NRC_OUTPUT_DIM   * 4u;

//fallback until renderer packs adaptive tile side into nrc_flags
static const uint NRC_TRAINING_TILE_SIDE    = 8u;
static const uint NRC_TRAINING_TILE_MIN     = 4u;
static const uint NRC_TRAINING_TILE_MAX     = 32u;
//1/denom of training paths stay fully unbiased, (denom-1)/denom short-
//circuit deep bounces into the cache (cache-as-tail). Set to 16 (paper's
//value, Müller et al. 2021 §3.2). With biased paths the cache's own
//prediction feeds back into training targets -- a self-reinforcement loop
//that an earlier session saw cause brightness creep on the old config.
//Re-enabled because the current config (5x128 MLP, log2=19, 1.5x training
//data, roughness emit-gate) should keep the cache accurate enough for the
//loop to stay stable, and 15/16 paths short-circuiting is a big raygen
//cost saving. kTargetMax=10 is the backstop; watch GI for slow drift.
//Mirror: kUnbiasedDenom in rdn/NRC/NrcLayout.h must match.
static const uint NRC_UNBIASED_DENOM        = 16u;

static const uint NRC_MAX_TRAINING_PATHS    = 131072u;
static const uint NRC_MAX_VERTICES_PER_PATH = 8u;

static const uint NRC_INVALID_SLOT          = 0xFFFFFFFFu;
static const uint NRC_INVALID_PATH          = 0xFFFFFFFFu;

//tail kind, mirrors nrc::TailKind
static const uint NRC_TAIL_INVALID = 0u;
static const uint NRC_TAIL_RR      = 1u;
static const uint NRC_TAIL_EMITTER = 2u;
static const uint NRC_TAIL_MISS    = 3u;
static const uint NRC_TAIL_CACHE   = 4u;

//TrainingPathMeta, one per pathId, at head of buffer
static const uint NRC_TPM_STRIDE       = 16u;
static const uint NRC_TPM_OFF_NUMVERTS = 0u;
static const uint NRC_TPM_OFF_TAILKIND = 4u;
static const uint NRC_TPM_OFF_INFSLOT  = 8u;
static const uint NRC_TPM_OFF_TAILRAD  = 12u;

//meta table byte size before per-vertex records start
static const uint NRC_PATH_META_TOTAL  = NRC_MAX_TRAINING_PATHS * NRC_TPM_STRIDE;

//TrainingVertex, 68B features (17 floats) + 4B L_neePk + 4B betaLocalPk
static const uint NRC_TV_STRIDE        = 76u;
static const uint NRC_TV_OFF_RAW       = 0u;
static const uint NRC_TV_OFF_LNEE      = 68u;
static const uint NRC_TV_OFF_BETA      = 72u;

//PendingGI offsets in bytes
static const uint NRC_PG_STRIDE             = 16u;
static const uint NRC_PG_OFF_SLOT           = 0u;
static const uint NRC_PG_OFF_THROUGHPUT     = 4u;
static const uint NRC_PG_OFF_TPOST          = 8u;
static const uint NRC_PG_OFF_PDF            = 12u;

//Counters offsets in bytes
static const uint NRC_C_OFF_INFERENCE_COUNT = 0u;
static const uint NRC_C_OFF_TRAINING_PATH   = 4u;

//path class tags for SER sort key
static const uint NRC_CLASS_RENDER          = 0u;
static const uint NRC_CLASS_TRAIN_BIASED    = 1u;
static const uint NRC_CLASS_TRAIN_UNBIASED  = 2u;

//push-constant flag bits, mirrors nrc::flags
//bits 0..3, behavior toggles, bits 8..15, training tile side
static const uint NRC_FLAG_ENABLED          = 1u;
static const uint NRC_FLAG_TRAIN            = 2u;
static const uint NRC_FLAG_DEBUG_VIEW       = 4u;
static const uint NRC_FLAG_SHARP_REFL       = 8u;
static const uint NRC_FLAG_TILE_SHIFT       = 8u;
static const uint NRC_FLAG_TILE_MASK        = 0xFFu;

inline bool NrcIsEnabled()           { return (nrc_flags & NRC_FLAG_ENABLED)          != 0u; }
inline bool NrcIsTrainOn()           { return (nrc_flags & NRC_FLAG_TRAIN)            != 0u; }
inline bool NrcIsDebugView()         { return (nrc_flags & NRC_FLAG_DEBUG_VIEW)       != 0u; }
inline bool NrcIsSharpReflectionsOn(){ return (nrc_flags & NRC_FLAG_SHARP_REFL)       != 0u; }

//effective tile side, falls back to static default when flags unpacked
inline uint NrcTrainingTileSide()
{
    const uint packed = (nrc_flags >> NRC_FLAG_TILE_SHIFT) & NRC_FLAG_TILE_MASK;
    return (packed == 0u) ? NRC_TRAINING_TILE_SIDE : packed;
}

//====================================
//BUFFER BINDINGS
//====================================
//u40..u44 reserved for NRC, training buffer interleaves meta then vertices
RWByteAddressBuffer g_NrcInferenceIn  : register(u40);
RWByteAddressBuffer g_NrcInferenceOut : register(u41);
RWByteAddressBuffer g_NrcPendingGI    : register(u42);
RWByteAddressBuffer g_NrcTrainRecords : register(u43);
RWByteAddressBuffer g_NrcCounters     : register(u44);

//====================================
//DIRECTION ENCODING
//====================================
//SH encoder takes [0,1]^3, remaps internally to unit sphere
inline float3 NrcEncodeDir(float3 d)
{
    return d * 0.5f + 0.5f;
}

inline float NrcEncodeRoughness(float r)
{
    return saturate(1.0f - exp(-max(r, 0.0f)));
}

//====================================
//NAN INF SANITIZATION
//====================================
//one non-finite value kills every Adam moment permanently
inline float NrcCleanFinite(float v)
{
    return (isnan(v) || isinf(v)) ? 0.0f : v;
}
inline float3 NrcCleanFinite3(float3 v)
{
    return float3(NrcCleanFinite(v.x), NrcCleanFinite(v.y), NrcCleanFinite(v.z));
}
inline float3 NrcCleanRadiance(float3 v)
{
    return max(NrcCleanFinite3(v), float3(0, 0, 0));
}

//====================================
//FEATURE PACK
//====================================
//maps scene AABB to [0,1]^3 for HashGrid
inline float3 NrcNormalizePosition(float3 x)
{
    return (x - nrc_scene_center) * nrc_scene_scale_inv + 0.5f;
}

//sole sanitization checkpoint between path tracer and tcnn
inline void NrcBuildFeatures(
    float3 x, float3 o, float3 n,
    float  r,
    float3 alpha, float3 beta,
    bool   backface,
    out float features[17])
{
    const float3 xN   = NrcNormalizePosition(x);
    const float3 shO  = NrcEncodeDir(o);
    const float3 shN  = NrcEncodeDir(n);

    features[0]  = saturate(NrcCleanFinite(xN.x));
    features[1]  = saturate(NrcCleanFinite(xN.y));
    features[2]  = saturate(NrcCleanFinite(xN.z));
    features[3]  = saturate(NrcCleanFinite(shO.x));
    features[4]  = saturate(NrcCleanFinite(shO.y));
    features[5]  = saturate(NrcCleanFinite(shO.z));
    features[6]  = saturate(NrcCleanFinite(shN.x));
    features[7]  = saturate(NrcCleanFinite(shN.y));
    features[8]  = saturate(NrcCleanFinite(shN.z));
    features[9]  = saturate(NrcCleanFinite(NrcEncodeRoughness(r)));
    features[10] = clamp(NrcCleanFinite(alpha.x), 0.0f, 1e3f);
    features[11] = clamp(NrcCleanFinite(alpha.y), 0.0f, 1e3f);
    features[12] = clamp(NrcCleanFinite(alpha.z), 0.0f, 1e3f);
    features[13] = clamp(NrcCleanFinite(beta.x),  0.0f, 1e3f);
    features[14] = clamp(NrcCleanFinite(beta.y),  0.0f, 1e3f);
    features[15] = clamp(NrcCleanFinite(beta.z),  0.0f, 1e3f);
    //+1 front / -1 back, centered around 0 so the first hidden layer's
    //weighted sum has a clean zero crossing on indeterminate inputs
    features[16] = backface ? -1.0f : 1.0f;
}

//====================================
//AREA-SPREAD TERMINATION
//====================================
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

//====================================
//INFERENCE INPUT OUTPUT
//====================================
//raw-input variant: builds features inline and streams them directly to the
//inference buffer. The previous form took a float[17] which DXC allocates at
//raygen function entry and counts as 68B of live state across every reorder
//point even when logically dead. Stay raw, never materialise the array.
inline uint NrcAppendInference(
    uint   capacity,
    float3 x, float3 o, float3 n, float r,
    float3 alpha, float3 beta, bool backface)
{
    uint slot;
    g_NrcCounters.InterlockedAdd(NRC_C_OFF_INFERENCE_COUNT, 1u, slot);
    if (slot >= capacity) return NRC_INVALID_SLOT;

    const uint base = slot * NRC_INFERENCE_IN_STRIDE;

    const float3 xN  = saturate(NrcCleanFinite3(NrcNormalizePosition(x)));
    const float3 shO = saturate(NrcCleanFinite3(NrcEncodeDir(o)));
    const float3 shN = saturate(NrcCleanFinite3(NrcEncodeDir(n)));
    const float  rE  = saturate(NrcCleanFinite(NrcEncodeRoughness(r)));
    const float3 aC  = clamp(NrcCleanFinite3(alpha), 0.0f, 1e3f);
    const float3 bC  = clamp(NrcCleanFinite3(beta),  0.0f, 1e3f);
    const float  sd  = backface ? -1.0f : 1.0f;

    g_NrcInferenceIn.Store3(base +  0u, asuint(xN));
    g_NrcInferenceIn.Store3(base + 12u, asuint(shO));
    g_NrcInferenceIn.Store3(base + 24u, asuint(shN));
    g_NrcInferenceIn.Store (base + 36u, asuint(rE));
    g_NrcInferenceIn.Store3(base + 40u, asuint(aC));
    g_NrcInferenceIn.Store3(base + 52u, asuint(bC));
    g_NrcInferenceIn.Store (base + 64u, asuint(sd));
    return slot;
}

inline float3 NrcLoadInferenceOutput(uint slot)
{
    if (slot == NRC_INVALID_SLOT) return float3(0, 0, 0);
    const uint base = slot * NRC_INFERENCE_OUT_STRIDE;
    const uint3 raw = g_NrcInferenceOut.Load3(base);
    //MLP predicts L/reflSum directly (linear output, RelativeL2 loss).
    //NrcCleanRadiance sanitizes NaN/Inf and clamps negatives from the
    //final linear layer (upstream ReLU stays non-negative).
    return NrcCleanRadiance(asfloat(raw));
}

//====================================
//PENDING GI RECORD
//====================================
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

//====================================
//CACHE TERMINATION RECORD
//====================================
//returns inference slot or NRC_INVALID_SLOT
//hoists reflectance onto throughput/tpost so MLP output is treated as radiance
inline uint NrcWriteTerminationRecord(
    uint   pixelIdx,
    uint   inferenceCapacity,
    float3 hitPos,
    float3 viewDir,
    float3 hitNormal,
    float  roughness,
    float3 kd,
    float  metallic,
    bool   backface,
    float3 throughput,
    float3 tpost,
    float  pdfProduct)
{
    const float3 alpha   = kd * (1.0f - metallic);
    const float3 betaC   = lerp(float3(0.04f, 0.04f, 0.04f), kd, metallic);
    const float3 reflSum = alpha + betaC;

    const uint slot = NrcAppendInference(
        inferenceCapacity, hitPos, viewDir, hitNormal,
        roughness, alpha, betaC, backface);
    if (slot == NRC_INVALID_SLOT) return NRC_INVALID_SLOT;

    NrcStorePendingGI(
        pixelIdx,
        slot,
        PackRGB9E5(NrcCleanRadiance(throughput * reflSum)),
        PackRGB9E5(NrcCleanRadiance(tpost      * reflSum)),
        NrcCleanFinite(pdfProduct));
    return slot;
}

//====================================
//TRAINING PATH META
//====================================
//returns NRC_INVALID_PATH when per-frame quota exhausted
inline uint NrcAllocateTrainingPath()
{
    uint pathId;
    g_NrcCounters.InterlockedAdd(NRC_C_OFF_TRAINING_PATH, 1u, pathId);
    if (pathId >= NRC_MAX_TRAINING_PATHS) return NRC_INVALID_PATH;
    return pathId;
}

//emitMask holds one bit per vertex slot; bit i set = vertex i is eligible to
//be picked as the per-path emission depth in the CUDA fill kernel. Packed
//into the upper byte of the numVertices word (numVertices itself is bounded
//by NRC_MAX_VERTICES_PER_PATH=8 so bits 0..3 are sufficient for the count).
//Layout: bits 0..7 = numVertices, bits 8..15 = emitMask, bits 16..31 reserved.
//
//Split between head (numVertices+emitMask, written at finalize) and tail
//(kind+slot+rad, written at each termination point) so the live state across
//the bounce-loop trace does not have to carry nrcTailRadPk + nrcTailInfSlot
//in registers from termination through to finalize.
//
//Validity gate is the head word: fill_training_batch_kernel skips records
//with numVertices==0. As long as the head is written last, a torn read is
//impossible (the head is the publish barrier).
inline void NrcStorePathHead(uint pathId, uint numVertices, uint emitMask)
{
    if (pathId >= NRC_MAX_TRAINING_PATHS) return;
    const uint base   = pathId * NRC_TPM_STRIDE;
    const uint packed = (numVertices & 0xFFu) | ((emitMask & 0xFFu) << 8u);
    g_NrcTrainRecords.Store(base + NRC_TPM_OFF_NUMVERTS, packed);
}

inline void NrcStorePathTail(uint pathId, uint tailKind, uint inferenceSlot, uint tailRadiancePk)
{
    if (pathId >= NRC_MAX_TRAINING_PATHS) return;
    const uint base = pathId * NRC_TPM_STRIDE;
    g_NrcTrainRecords.Store3(base + NRC_TPM_OFF_TAILKIND,
                             uint3(tailKind, inferenceSlot, tailRadiancePk));
}

//RR-default seed at allocation, overwritten if the path terminates explicitly
inline void NrcInitPathTail(uint pathId)
{
    NrcStorePathTail(pathId, NRC_TAIL_RR, NRC_INVALID_SLOT, 0u);
}

//====================================
//TRAINING VERTEX
//====================================
//raw-input variant, see NrcAppendInference for the live-state rationale
inline void NrcStoreTrainingVertex(
    uint   pathId, uint vIdx,
    float3 x, float3 o, float3 n, float r,
    float3 alpha, float3 beta, bool backface,
    uint   L_neePk, uint betaLocalPk)
{
    if (pathId >= NRC_MAX_TRAINING_PATHS || vIdx >= NRC_MAX_VERTICES_PER_PATH) return;
    const uint base = NRC_PATH_META_TOTAL +
                      (pathId * NRC_MAX_VERTICES_PER_PATH + vIdx) * NRC_TV_STRIDE;

    const float3 xN  = saturate(NrcCleanFinite3(NrcNormalizePosition(x)));
    const float3 shO = saturate(NrcCleanFinite3(NrcEncodeDir(o)));
    const float3 shN = saturate(NrcCleanFinite3(NrcEncodeDir(n)));
    const float  rE  = saturate(NrcCleanFinite(NrcEncodeRoughness(r)));
    const float3 aC  = clamp(NrcCleanFinite3(alpha), 0.0f, 1e3f);
    const float3 bC  = clamp(NrcCleanFinite3(beta),  0.0f, 1e3f);
    const float  sd  = backface ? -1.0f : 1.0f;

    g_NrcTrainRecords.Store3(base + NRC_TV_OFF_RAW +  0u, asuint(xN));
    g_NrcTrainRecords.Store3(base + NRC_TV_OFF_RAW + 12u, asuint(shO));
    g_NrcTrainRecords.Store3(base + NRC_TV_OFF_RAW + 24u, asuint(shN));
    g_NrcTrainRecords.Store (base + NRC_TV_OFF_RAW + 36u, asuint(rE));
    g_NrcTrainRecords.Store3(base + NRC_TV_OFF_RAW + 40u, asuint(aC));
    g_NrcTrainRecords.Store3(base + NRC_TV_OFF_RAW + 52u, asuint(bC));
    g_NrcTrainRecords.Store (base + NRC_TV_OFF_RAW + 64u, asuint(sd));
    g_NrcTrainRecords.Store (base + NRC_TV_OFF_LNEE, L_neePk);
    g_NrcTrainRecords.Store (base + NRC_TV_OFF_BETA, betaLocalPk);
}

//beta known only after raygen samples the next BSDF, second pass update
inline void NrcUpdateTrainingVertexBeta(uint pathId, uint vIdx, uint betaLocalPk)
{
    if (pathId >= NRC_MAX_TRAINING_PATHS || vIdx >= NRC_MAX_VERTICES_PER_PATH) return;
    const uint base = NRC_PATH_META_TOTAL +
                      (pathId * NRC_MAX_VERTICES_PER_PATH + vIdx) * NRC_TV_STRIDE;
    g_NrcTrainRecords.Store(base + NRC_TV_OFF_BETA, betaLocalPk);
}

//====================================
//PATH CLASS CLASSIFICATION
//====================================
inline uint NrcClassifyPixel(
    uint2 pixel,
    uint  frameIndex,
    uint  tileSide,
    uint2 perFrameTileOffset,
    out bool isTraining,
    out bool isUnbiased)
{
    const uint2 tileOrigin = (pixel / tileSide) * tileSide;
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

#endif
