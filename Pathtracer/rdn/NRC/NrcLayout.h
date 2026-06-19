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

//4 hidden x 64 ReLU, FullyFusedMLP tensor-core path. Width is from the fused
//path's supported set {16,32,64,128}; wider forces the ~2x slower CutlassMLP.
//Was briefly 128 (widened when inference looked encoder-bound, so the extra
//MLP FLOPs were ~free at query time) -- narrowed back to 64 (the Müller 2021
//width) because TRAINING backprops through the MLP and does pay for width,
//and training wall-time was the concern. Depth kept at 4 to limit dying-ReLU
//compounding in dark scenes; the collapse detector in Renderer.cpp is the
//backstop if a unit cascade still fires. EMA buffer + tcnn params auto-resize
//from network->n_params() in Network::Init / ReinitWeights, no other code
//keys off these two constants.
constexpr uint32_t kHiddenWidth  = 64;
constexpr uint32_t kHiddenLayers = 4;

//tcnn batch granularity
constexpr uint32_t kBatchGranularity = 256;

//Base Adam learning rate (the tuned "Müller-matched" 1e-2; the network trains
//fine at this rate). Effective rate = kBaseLearningRate * Settings.learningRateScale,
//applied live each frame via Network::SetLearningRate. Previously the slider was
//inert — only pushed to a shader constant, never reaching the CUDA optimizer —
//so this just makes it a working runtime knob (default scale 1.0 = unchanged).
constexpr float kBaseLearningRate = 1.0e-2f;

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

//Anchored 1080p density: one traced training PATH per N screen pixels. The
//adaptive tile side grows with resolution to keep this path density constant
//(NRC stability was tuned at 1080p; scaling with screen area destabilized it).
//History: 64 -> 21 (~3x, cut SGD gradient variance, fixed dark-scene
//instability, but tripled SGD wall-time) -> settled 42 (~1.5x baseline).
constexpr uint32_t kPixelsPerTrainingSample = 42u;

//Upper bound on training ROWS one path emits. With multi-row emission (one
//row per eligible vertex -- see kTrainingDecorrelatePaths) and
//kTrainingDepthMask=0b011 a path emits up to 2 rows. The per-frame row budget
//below is paths/frame * this, so the extra-vertex rows are ADDITIVE: they do
//not displace path diversity. If kTrainingDecorrelatePaths is set false (emit
//at ALL stored vertices) the true bound is kMaxVerticesPerPath; the budget
//stays sized for the decorrelated case and the fill kernel just caps
//over-emission, which is safe (dropped, not OOB).
constexpr uint32_t kMaxEmitRowsPerPath = 2u;

//FIXED per-frame training ROW target across all resolutions = paths/frame
//(1080p-anchored) * kMaxEmitRowsPerPath. The adaptive tile side grows with
//resolution to hit this constant. Stays under the kMaxTrainingPaths /
//kTrainingRecordsPerFrame ceiling of 131072 so the renderer's clamp does not
//bite at common resolutions.
constexpr uint32_t kFixedTrainingRecords =
    ((1920u * 1080u) / kPixelsPerTrainingSample) * kMaxEmitRowsPerPath;

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

//Multi-row training emission. The CUDA fill kernel emits a (features,target)
//row for EVERY vertex permitted by the masks below -- a path contributes one
//row per eligible vertex, not one row total.
//  kTrainingDecorrelatePaths = true  -> emit at every vertex whose depth bit
//      is set in kTrainingDepthMask AND whose roughness emit gate (emitMask)
//      passed. e.g. 0b00000011 -> depths 0 and 1 (primary hit + first bounce).
//  kTrainingDecorrelatePaths = false -> emit at every stored vertex that
//      passed the roughness emit gate (kTrainingDepthMask ignored).
//The roughness emit gate (NRC_TRAIN_ROUGHNESS_MIN) always applies in both
//modes -- it is a separate concern from depth selection and is load-bearing.
//
//HISTORY / WARNING: this previously emitted exactly ONE row per path (a
//single random depth) specifically to kill intra-path target correlation --
//target_v and target_{v+1} both derive from the same MC realization of
//L_s[v+1], so multi-row emission produces positively correlated gradients
//that can lock Adam's 2nd moment ("indirect underestimated, sky color
//absorbed into shadow / chromatic darkening"). Multi-row was re-enabled
//2026-05-14 to test whether the current config tolerates it; the row budget
//(kFixedTrainingRecords, via kMaxEmitRowsPerPath) was raised so the extra
//rows are additive, not displacing path diversity. WATCH indirect regions
//for chromatic darkening -- if it returns, single-row (one random depth per
//path) is the proven-safe fallback.
//
//kTrainingDepthMask bit i set = depth i eligible (used when decorrelate=true):
//  0b00000011 = {0,1}   0b00000111 = {0,1,2}   0b11111111 = {0..7}
constexpr bool     kTrainingDecorrelatePaths = true;
constexpr uint32_t kTrainingDepthMask        = 0b00000011u;

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
//              (raw GGX roughness >= NRC_TRAIN_ROUGHNESS_MIN at storage time)
//  bits 16..31 reserved
//The fill kernel intersects emitMask with kTrainingDepthMask so that only
//diffuse-dominated vertices emit training rows; chain accumulation still walks
//every stored vertex so tails through ineligible bounces resolve correctly.
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
