#pragma once
//====================================
//NRC BUFFER LAYOUT CONTRACTS
//====================================
//shared across C++, CUDA, HLSL, HLSL side mirrors offsets in Nrc_v8.hlsli

#include <cstdint>

namespace nrc {

struct Float3 { float x = 0.0f, y = 0.0f, z = 0.0f; };

//====================================
//RUNTIME SETTINGS
//====================================
//mirrored into push constants every frame
struct Settings {
    bool  enabled            = true;
    bool  trainingEnabled    = true;
    bool  debugView          = false;
    //x1 sharp-reflection split: NRC tap on the perfect-mirror reflection ray
    //plus BSDF specialisation at the primary hit. Off = legacy DLSS-RR-only path.
    bool  sharpReflections   = true;
    float areaSpreadC        = 0.01f;
    float learningRateScale  = 1.0f;
    //scene AABB normalization, x_norm = (x-center)/extent + 0.5
    Float3 sceneCenter   = {};
    float  sceneExtent   = 50.0f;
    //transient UI request, Renderer consumes and clears each frame
    bool  requestReinit      = false;
};

//push-constant flag bits, bits 0..3 toggles, bits 8..15 tile side
namespace flags {
    constexpr uint32_t kEnabled         = 1u << 0;
    constexpr uint32_t kTrain           = 1u << 1;
    constexpr uint32_t kDebugView       = 1u << 2;
    constexpr uint32_t kSharpReflections = 1u << 3;
    constexpr uint32_t kTileShift       = 8u;
    constexpr uint32_t kTileMask        = 0xFFu;
}

//====================================
//NETWORK DIMENSIONS
//====================================
//raw feature vector, tcnn composite expands to 75 dims internally
//0..2 position (HashGrid)
//3..5 scattered dir, unit 3-vec *0.5+0.5 (SH deg 4)
//6..8 normal, unit 3-vec *0.5+0.5 (SH deg 4)
//9    roughness 1-exp(-r) (OneBlob 4 bins)
//10..12 diffuse refl (Identity)
//13..15 specular refl (Identity)
//16     side bit, +1 front / -1 back (Identity), folded into the
//       same Identity slab as 10..15 so the composite stays 5 entries
constexpr uint32_t kRawInputDim  = 17;
constexpr uint32_t kOutputDim    = 3;

//per-path bucket cap, deeper paths drop tail vertices
constexpr uint32_t kMaxVerticesPerPath = 8u;

//5 hidden x 64 ReLU. Deeper net to recover specular/glossy tail detail
//that the 3-layer config softened; combined with the bumped HashGrid
//capacity (log2_hashmap_size=21) and the doubled training batches/frame.
constexpr uint32_t kHiddenWidth  = 64;
constexpr uint32_t kHiddenLayers = 4;

//tcnn batch granularity
constexpr uint32_t kBatchGranularity = 256;

//per-frame training schedule. kTrainingBatchSize is the legacy floor for
//the original fixed-target design; the actual SGD per-batch row count is
//computed at runtime from validVertices/kTrainingBatchesPerFrame so the
//trainer naturally scales with resolution. kTrainingRecordsPerFrame is now
//the BUFFER SIZE ceiling (sized for ~4K worst case), the runtime target
//passed each frame from the renderer caps the fill kernel and the adaptive
//tile feedback below this number.
constexpr uint32_t kTrainingBatchSize       = 8192;
constexpr uint32_t kTrainingBatchesPerFrame = 4;
constexpr uint32_t kTrainingRecordsPerFrame = 131072;

//target one training sample per N screen pixels regardless of resolution
//so per-cell sample density stays consistent. At 1080p, W*H/64 = 32400,
//matches the old fixed target. At 1440p, ~57K. At 4K, ~130K (clamps to
//the buffer cap above).
constexpr uint32_t kPixelsPerTrainingSample = 64u;

//training tile side adapts per frame to saturate trainer
constexpr uint32_t kInitialTrainingTileSide = 8u;
constexpr uint32_t kMinTrainingTileSide     = 4u;
constexpr uint32_t kMaxTrainingTileSide     = 32u;

//100% unbiased training (denom=1). All training paths take the long
//RR-terminated route, no cache-as-tail. Self-bias loop eliminated: stale
//cache values can't reinforce themselves because no training target
//depends on the cache's own prediction. Trade-off: deep bounces are
//noisier (RR truncation, no cache short-circuit) so GI convergence in
//dim multi-bounce regions takes more frames. Paper used 16, raise if
//multi-bounce quality regresses past acceptable.
constexpr uint32_t kUnbiasedDenom = 1u;

//EXPERIMENT: emit training rows ONLY at this exact path depth.
//Backward fill still walks every stored vertex so the emitted row at
//depth K carries the full multi-bounce MC chain from K to terminal,
//but rows for any vertex with depth != K are never written into the
//training batch. Multi-depth unions produce systematic darkening /
//chromatic bias in indirect regions regardless of loss, transform,
//shuffle, or HashGrid capacity -- single-depth training is the only
//configuration that gives clean predictions for indirect illumination.
//  0 -> only primary hit (= x1 in paper's 1-indexed nomenclature)
//  1 -> only first bounce (= x2)           <- currently best observed
//  2 -> only second bounce (= x3)
//  ...
//Used when kTrainingDecorrelatePaths is false.
constexpr uint32_t kTrainingEmitDepthOnly = 1u;

//EXPERIMENT [H1]: per-path random depth selection.
//Each path picks ONE depth from kTrainingDepthSet uniformly at random
//(hashed from pathId) and emits exactly one training row at that depth.
//Backward fill still walks every stored vertex so the emitted row
//carries the full multi-bounce MC chain from its depth to terminal.
//
//This preserves the multi-depth feature distribution but eliminates
//within-path target correlation: the previous "emit at every v in
//{0..K}" mode produced positively correlated gradients on shared MLP
//parameters because target_v and target_{v+1} are both derived from
//the same MC realization of L_s[v+1]. Adam's 2nd-moment EMA accumulates
//that correlated variance and effective LR collapses on the affected
//parameters, locking predictions near the EMA-init mean (~0) -- the
//"indirect underestimated, sky color absorbed into shadow" signature.
//
//Test protocol:
//  (a) kTrainingDecorrelatePaths=false, kTrainingEmitDepthOnly=1
//      -> single-depth baseline, expected to be clean.
//  (b) kTrainingDecorrelatePaths=false with the kernel emit predicate
//      modified to v <= 1 (or any multi-depth union) -> reproduces bias.
//  (c) kTrainingDecorrelatePaths=true, kTrainingDepthSet={0,1}
//      -> H1 test. Same multi-depth feature distribution as (b) but
//         at most one row per path. If indirect regions recover, H1 is
//         confirmed: intra-path correlation drives the bias and the
//         fix is per-sample gradient clipping or AMSGrad. If still
//         biased, H1 is ruled out and the next test is RelativeL2 eps.
//
//When kTrainingDecorrelatePaths is true, kTrainingEmitDepthOnly is ignored.
//
//The pick set is a 32-bit bitmask -- bit i set means depth i is eligible.
//The kernel uses __popc + __ffs to count bits and resolve the n-th set bit
//for the per-path random pick (constexpr arrays are not reachable from
//device code, but a uint32_t constant is).
//  0b00000011 = {0,1}    (cleanest H1 test, same row count as single-depth)
//  0b00000110 = {1,2}    (matches the user's reported "K=1,2" failure case)
//  0b00000111 = {0,1,2}
//  0b11111111 = full {0..7}
constexpr bool     kTrainingDecorrelatePaths = true;
constexpr uint32_t kTrainingDepthMask        = 0b11111111u;

//DIAGNOSTIC: pipeline integrity test.
constexpr bool kDebugConstantTraining = false;

//max training paths per frame. Sized for 4K at the smallest adaptive tile
//(kPixelsPerTrainingSample density), one row per path with decorrelation.
constexpr uint32_t kMaxTrainingPaths   = 131072u;

constexpr uint32_t kInvalidInferenceSlot = 0xFFFFFFFFu;
constexpr uint32_t kInvalidTrainPath     = 0xFFFFFFFFu;

//tail kind, sets base for backward recursion, target[v] = L_nee[v] + beta*target[v+1]
enum TailKind : uint32_t {
    kTailInvalid = 0u,
    kTailRR      = 1u,
    kTailEmitter = 2u,
    kTailMiss    = 3u,
    kTailCache   = 4u,
};

//====================================
//PER-RECORD STRIDES
//====================================
//tcnn column-major, fp32
constexpr uint32_t kInferenceInputStride  = kRawInputDim * sizeof(float);
constexpr uint32_t kInferenceOutputStride = kOutputDim   * sizeof(float);

//====================================
//PENDING GI
//====================================
//raygen writes on cache-term, resolve consumes after inference
struct PendingGI {
    uint32_t inferenceSlot;
    uint32_t throughputPk;
    uint32_t tpostPk;
    float    pdfProduct;
};
static_assert(sizeof(PendingGI) == 16, "PendingGI must match Nrc_v8.hlsli");

//====================================
//TRAINING PATH META
//====================================
//one per training path, at head of training buffer
//tailRadiancePk semantics, emitter/miss hold tail radiance
//cache holds (alpha+beta) at cache-term vertex for radiance recovery
//RR unused, tail=0
struct TrainingPathMeta {
    uint32_t numVertices;
    uint32_t tailKind;
    uint32_t inferenceSlot;
    uint32_t tailRadiancePk;
};
static_assert(sizeof(TrainingPathMeta) == 16, "TrainingPathMeta must match Nrc_v8.hlsli");

//====================================
//TRAINING VERTEX
//====================================
//indexed by [pathId * kMaxVerticesPerPath + vIdx] after kPathMetaTotalBytes
//L_neePk, NEE estimator local to this vertex, no upstream chain
//betaLocalPk, BSDF*cos*T/pdf at this vertex, transports radiance v+1 -> v
struct TrainingVertex {
    float    raw[kRawInputDim];
    uint32_t L_neePk;
    uint32_t betaLocalPk;
};
static_assert(sizeof(TrainingVertex) == 76, "TrainingVertex must match Nrc_v8.hlsli");

constexpr uint32_t kPathMetaStride    = sizeof(TrainingPathMeta);
constexpr uint32_t kTrainVertexStride = sizeof(TrainingVertex);
constexpr uint32_t kPathMetaTotalBytes = kMaxTrainingPaths * kPathMetaStride;

inline constexpr uint32_t TrainingMetaOffset(uint32_t pathId) {
    return pathId * kPathMetaStride;
}
inline constexpr uint32_t TrainingVertexOffset(uint32_t pathId, uint32_t vIdx) {
    return kPathMetaTotalBytes + (pathId * kMaxVerticesPerPath + vIdx) * kTrainVertexStride;
}

//====================================
//COUNTERS
//====================================
struct Counters {
    uint32_t inferenceCount;
    uint32_t trainingPathCount;
    uint32_t pad0;
    uint32_t pad1;
};
static_assert(sizeof(Counters) == 16, "Counters must match Nrc_v8.hlsli");

//====================================
//BUFFER BYTE TOTALS
//====================================
inline uint64_t InferenceInputBytes (uint32_t maxInferenceRecords) {
    return uint64_t(maxInferenceRecords) * kInferenceInputStride;
}
inline uint64_t InferenceOutputBytes(uint32_t maxInferenceRecords) {
    return uint64_t(maxInferenceRecords) * kInferenceOutputStride;
}
inline uint64_t PendingGIBytes      (uint32_t pixelCount) {
    return uint64_t(pixelCount) * sizeof(PendingGI);
}
//meta section + per-vertex bucket section, contiguous
inline uint64_t TrainingBytes       () {
    return uint64_t(kPathMetaTotalBytes) +
           uint64_t(kMaxTrainingPaths) * kMaxVerticesPerPath * kTrainVertexStride;
}
inline uint64_t CountersBytes       () {
    return sizeof(Counters);
}

//round batch up to tcnn granularity
constexpr uint32_t AlignBatch(uint32_t n) {
    return (n + kBatchGranularity - 1u) / kBatchGranularity * kBatchGranularity;
}

}
