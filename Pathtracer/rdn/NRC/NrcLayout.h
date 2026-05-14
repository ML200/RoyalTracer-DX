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

//4 hidden x 128 ReLU, FullyFusedMLP tensor-core path. Width 128 is the top
//of the fused path's supported set {16,32,64,128} -- going wider forces the
//~2x slower CutlassMLP. Bumped from 4x64 now that inference is L2-encoder-
//bound rather than MLP-bound: the extra width is nearly free against the
//encoder cost so we spend it on capacity. Depth kept at 4 (not 5+) to
//limit dying-ReLU compounding in dark scenes; the collapse detector in
//Renderer.cpp plus l2_reg=1e-5 and Adam epsilon=1e-4 guard the worst case
//if a unit cascade still fires. EMA buffer + tcnn params auto-resize from
//network->n_params() in Network::Init / ReinitWeights, no other code keys
//off these two constants.
constexpr uint32_t kHiddenWidth  = 128;
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

//Anchored 1080p density used to derive the fixed training records target
//below: one traced training path per N screen pixels.
//
//History: started at 64. Dropped to 21 (~3x denser) to cut SGD gradient
//variance -- that fixed dark-scene instability and sped adaptation a lot,
//but tripled per-step SGD compute and made training wall-time significant.
//Settled at 42 (~1.5x the original baseline): keeps most of the variance
//reduction (perBatch ~7936 -> ~12288) while halving the cost of the 3x
//peak. Why variance matters: with small batches a dark frame's sparse
//bright NEE hits land unevenly across the 4 SGD batches, inflating Adam's
//2nd moment, which throttles the effective step (slow adaptation) unevenly
//across weights (instability). Bigger batches give each step a
//representative dark/bright mix. Paired with LR raised to 2e-3 to keep
//adaptation snappy at the reduced data volume, and kUnbiasedDenom=16 which
//short-circuits most training paths into the cache so raygen tracing cost
//drops too.
constexpr uint32_t kPixelsPerTrainingSample = 42u;

//FIXED training records target across all resolutions. NRC stability tuning
//was done at 1080p; scaling the sample count with screen area meant UWQHD/4K
//were getting more samples per frame than the optimizer was tuned for, which
//manifested as over-fast adaptation and instability. Anchoring at the 1080p
//value (1920*1080/kPixelsPerTrainingSample = ~98742) keeps behavior
//consistent and lets the adaptive tile side grow with resolution to hit this
//constant target. Stays under the kMaxTrainingPaths / kTrainingRecordsPerFrame
//ceiling of 131072, so the renderer's clamp never bites at common resolutions.
constexpr uint32_t kFixedTrainingRecords = (1920u * 1080u) / kPixelsPerTrainingSample;

//training tile side adapts per frame to saturate trainer
constexpr uint32_t kInitialTrainingTileSide = 8u;
constexpr uint32_t kMinTrainingTileSide     = 4u;
constexpr uint32_t kMaxTrainingTileSide     = 32u;

//Cache-as-tail: with probability 1/denom a training path stays fully
//unbiased (long RR-terminated chain); with probability (denom-1)/denom
//it short-circuits deep bounces via the cache prediction. Higher denom
//means MORE cache reuse (paper used 16 = 15/16 cache-tail).
//
//Set to 16 (paper's value). An earlier session found denom>1 caused
//brightness creep in indirect regions -- biased paths seed their training
//target with the cache's own prediction, a self-reinforcement loop. That
//finding was on the old config (4x64 MLP, log2=21, 1x training data, no
//roughness emit-gate). Re-enabled at 16 because the current config (4x128
//MLP, log2=19 denser cells, 1.5x training data, training-eligibility
//roughness gate) should make the cache accurate enough that the loop
//stays stable -- AND because 15/16 paths short-circuiting into the cache
//is a large raygen cost saving vs tracing 8 full bounces. kTargetMax=10
//is the backstop. WATCH: slow overbrightening of GI/indirect regions over
//tens of seconds; if it returns, drop back toward 1.
//Mirror: NRC_UNBIASED_DENOM in shaders/Nrc_v8.hlsli must match.
constexpr uint32_t kUnbiasedDenom = 16u;

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
//
//numVertices field is packed:
//  bits  0..7  numVertices count (max kMaxVerticesPerPath = 8 fits in 4 bits)
//  bits  8..15 emitMask: bit i set = vertex i is training-emit-eligible
//              (effective roughness >= NRC_TRAIN_ROUGHNESS_MIN at storage time)
//  bits 16..31 reserved
//The fill kernel intersects emitMask with kTrainingDepthMask so that only
//diffuse-dominated vertices can be picked as the per-path emission depth;
//chain accumulation still walks every stored vertex so tails through
//ineligible bounces resolve correctly.
struct TrainingPathMeta {
    uint32_t numVerticesPacked;  // count in low byte, emitMask in next byte
    uint32_t tailKind;
    uint32_t inferenceSlot;
    uint32_t tailRadiancePk;
};
static_assert(sizeof(TrainingPathMeta) == 16, "TrainingPathMeta must match Nrc_v8.hlsli");

//helpers to unpack numVerticesPacked, used by the CUDA fill kernel
inline constexpr uint32_t UnpackNumVertices(uint32_t packed) { return packed & 0xFFu; }
inline constexpr uint32_t UnpackEmitMask   (uint32_t packed) { return (packed >> 8) & 0xFFu; }

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
