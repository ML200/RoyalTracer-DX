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
static const uint NRC_RAW_INPUT_DIM         = 16u;
static const uint NRC_OUTPUT_DIM            = 3u;
static const uint NRC_INFERENCE_IN_STRIDE   = NRC_RAW_INPUT_DIM * 4u;  // 64
static const uint NRC_INFERENCE_OUT_STRIDE  = NRC_OUTPUT_DIM   * 4u;   // 12

// Default tile side. Used as a fallback when the renderer hasn't packed
// a dynamic value into nrc_flags yet (e.g. first frame). The runtime
// value is read via NrcTrainingTileSide() below.
static const uint NRC_TRAINING_TILE_SIDE    = 8u;
static const uint NRC_TRAINING_TILE_MIN     = 4u;
static const uint NRC_TRAINING_TILE_MAX     = 32u;
static const uint NRC_UNBIASED_DENOM        = 8u;

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

// TrainingVertex: 64 B features + 4 B L_neePk + 4 B betaLocalPk = 72 B.
static const uint NRC_TV_STRIDE        = 72u;
static const uint NRC_TV_OFF_RAW       = 0u;
static const uint NRC_TV_OFF_LNEE      = 64u;
static const uint NRC_TV_OFF_BETA      = 68u;

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
//   bits 0..2  : behavior toggles (enabled / train / debug view)
//   bits 8..15 : training tile side (0 = use NRC_TRAINING_TILE_SIDE
//                fallback). Adaptive sizing per paper §3.5: renderer
//                scales this each frame from the previous frame's
//                vertex count to keep the trainer saturated.
static const uint NRC_FLAG_ENABLED          = 1u;
static const uint NRC_FLAG_TRAIN            = 2u;
static const uint NRC_FLAG_DEBUG_VIEW       = 4u;
static const uint NRC_FLAG_TILE_SHIFT       = 8u;
static const uint NRC_FLAG_TILE_MASK        = 0xFFu;

inline bool NrcIsEnabled()           { return (nrc_flags & NRC_FLAG_ENABLED)          != 0u; }
inline bool NrcIsTrainOn()           { return (nrc_flags & NRC_FLAG_TRAIN)            != 0u; }
inline bool NrcIsDebugView()         { return (nrc_flags & NRC_FLAG_DEBUG_VIEW)       != 0u; }

// Effective training tile side this frame. Falls back to the static
// default if the renderer didn't pack a value (which is what happens
// before the first TrainFrame produces a vertex count to feed the
// adaptive loop).
inline uint NrcTrainingTileSide()
{
    const uint packed = (nrc_flags >> NRC_FLAG_TILE_SHIFT) & NRC_FLAG_TILE_MASK;
    return (packed == 0u) ? NRC_TRAINING_TILE_SIDE : packed;
}

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
// DIRECTION ENCODING — raw unit 3-vec, remapped to [0,1]³ for SH.
//
// The tcnn SphericalHarmonics encoder takes inputs in [0, 1]³ and
// internally remaps them to [-1, 1]³ on the unit sphere before
// evaluating the degree-4 SH basis (16 coefficients). That gives a
// smooth, analytic, rotationally-equivariant direction encoding —
// no octahedral seam on the lower hemisphere, no OneBlob wraparound
// discontinuity at bin boundaries. Both the view direction and the
// surface normal ride this encoder.
//
// Inputs must be unit vectors; the callers already normalize, so this
// is a pure affine remap.
//====================================================================
inline float3 NrcEncodeDir(float3 d)
{
    return d * 0.5f + 0.5f;
}

inline float NrcEncodeRoughness(float r)
{
    return saturate(1.0f - exp(-max(r, 0.0f)));
}

//====================================================================
// NaN / Inf SANITIZATION
//====================================================================
// A single non-finite value reaching the trainer turns every Adam
// moment into NaN in one update step, after which the entire network
// is permanently dead — no amount of later clean data can resurrect
// it. The upstream path tracer deals with edge cases (degenerate
// BSDFs, denormal pdfs, zero-area triangles, texture fetches outside
// the mip pyramid, etc.) that CAN produce NaN/Inf, so we scrub every
// value that crosses into the NRC boundary.
//
// Ranges are chosen wide enough that legitimate values never hit the
// clamp: [-8, 8] for the normalized position (scene wraps at ±0.5
// already, so 8 periods of headroom), [0, 1] for angle/roughness
// encodings, [0, 1e3] for reflectances (physical values ≤1 but some
// authored materials push past that).
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
    // Non-negative + finite. Caller packs via RGB9E5 (clamps to ~65k)
    // so we don't need an explicit upper bound here.
    return max(NrcCleanFinite3(v), float3(0, 0, 0));
}

//====================================================================
// FEATURE PACK
//====================================================================
// Position normalization — maps the scene AABB to exactly [0, 1]³.
// The renderer sets nrc_scene_scale_inv = 0.5 / halfExtent so a
// vertex at +halfExtent lands at 1.0 and at -halfExtent at 0.0.
// HashGrid (the position encoder used by tcnn) is undefined outside
// [0, 1] — its hash function wraps and produces garbage features —
// so we saturate downstream in NrcBuildFeatures as a safety net for
// any vertex slightly outside the AABB (sub-mm offsets from
// transform jitter, etc.).
inline float3 NrcNormalizePosition(float3 x)
{
    return (x - nrc_scene_center) * nrc_scene_scale_inv + 0.5f;
}

inline void NrcBuildFeatures(
    float3 x, float3 o, float3 n,
    float  r,
    float3 alpha, float3 beta,
    out float features[16])
{
    const float3 xN   = NrcNormalizePosition(x);
    const float3 shO  = NrcEncodeDir(o);
    const float3 shN  = NrcEncodeDir(n);

    // Sanitize + bounds-clamp every feature. This is the one checkpoint
    // between the path tracer and tcnn — if garbage gets past here,
    // it pollutes Adam's state forever.
    // Saturate position to [0, 1] — HashGrid's hash function wraps for
    // inputs outside this range and produces garbage features.
    // Saturate directions to [0, 1] — SH's internal remap expects
    // inputs there; a caller that forgets to normalize would otherwise
    // feed SH a point off the unit sphere and get spurious coefficients.
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
inline uint NrcAppendInference(uint capacity, float features[16])
{
    uint slot;
    g_NrcCounters.InterlockedAdd(NRC_C_OFF_INFERENCE_COUNT, 1u, slot);
    if (slot >= capacity) return NRC_INVALID_SLOT;

    const uint base = slot * NRC_INFERENCE_IN_STRIDE;
    [unroll]
    for (uint i = 0; i < NRC_RAW_INPUT_DIM; ++i) {
        g_NrcInferenceIn.Store(base + i * 4u, asuint(features[i]));
    }
    return slot;
}

inline float3 NrcLoadInferenceOutput(uint slot)
{
    if (slot == NRC_INVALID_SLOT) return float3(0, 0, 0);
    const uint base = slot * NRC_INFERENCE_OUT_STRIDE;
    const uint3 raw = g_NrcInferenceOut.Load3(base);
    // NaN/Inf-safe non-negativity clamp. If the network somehow
    // output a non-finite value (shouldn't happen once features
    // are sanitized in NrcBuildFeatures, but belt-and-suspenders),
    // return 0 rather than letting NaN propagate into the reservoir,
    // the class-1 tail seed, or the debug view. `max(NaN, 0)` is
    // not NaN-safe on GPU (implementation-defined), so we handle
    // the NaN case explicitly.
    const float3 f = NrcCleanFinite3(asfloat(raw));
    return max(f, float3(0, 0, 0));
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

    float features[16];
    NrcBuildFeatures(hitPos, viewDir, hitNormal, roughness, alpha, betaC, features);

    const uint slot = NrcAppendInference(inferenceCapacity, features);
    if (slot == NRC_INVALID_SLOT) return NRC_INVALID_SLOT;

    NrcStorePendingGI(
        pixelIdx,
        slot,
        PackRGB9E5(NrcCleanRadiance(throughput * reflSum)),
        PackRGB9E5(NrcCleanRadiance(tpost      * reflSum)),
        NrcCleanFinite(pdfProduct));
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
    float features[16],
    uint L_neePk,
    uint betaLocalPk)
{
    if (pathId >= NRC_MAX_TRAINING_PATHS || vIdx >= NRC_MAX_VERTICES_PER_PATH) return;
    const uint base = NRC_PATH_META_TOTAL +
                      (pathId * NRC_MAX_VERTICES_PER_PATH + vIdx) * NRC_TV_STRIDE;
    [unroll]
    for (uint i = 0; i < NRC_RAW_INPUT_DIM; ++i) {
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

#endif // NRC_V8_HLSLI
