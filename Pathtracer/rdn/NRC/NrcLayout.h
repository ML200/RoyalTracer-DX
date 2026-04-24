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
    float areaSpreadC        = 0.01f;
    float learningRateScale  = 1.0f;
    //scene AABB normalization, x_norm = (x-center)/extent + 0.5
    Float3 sceneCenter   = {};
    float  sceneExtent   = 50.0f;
};

//push-constant flag bits, bits 0..2 toggles, bits 8..15 tile side
namespace flags {
    constexpr uint32_t kEnabled         = 1u << 0;
    constexpr uint32_t kTrain           = 1u << 1;
    constexpr uint32_t kDebugView       = 1u << 2;
    constexpr uint32_t kTileShift       = 8u;
    constexpr uint32_t kTileMask        = 0xFFu;
}

//====================================
//NETWORK DIMENSIONS
//====================================
//raw feature vector, tcnn composite expands to 74 dims internally
//0..2 position (HashGrid)
//3..5 scattered dir, unit 3-vec *0.5+0.5 (SH deg 4)
//6..8 normal, unit 3-vec *0.5+0.5 (SH deg 4)
//9    roughness 1-exp(-r) (OneBlob 4 bins)
//10..12 diffuse refl (Identity)
//13..15 specular refl (Identity)
constexpr uint32_t kRawInputDim  = 16;
constexpr uint32_t kOutputDim    = 3;

//2 hidden x 64 per Instant-NGP Table 6, HashGrid carries the representation
constexpr uint32_t kHiddenWidth  = 64;
constexpr uint32_t kHiddenLayers = 2;

//tcnn batch granularity
constexpr uint32_t kBatchGranularity = 256;

//per-frame training schedule, halved from paper, HashGrid converges fast
constexpr uint32_t kTrainingBatchSize       = 8192;
constexpr uint32_t kTrainingBatchesPerFrame = 4;
constexpr uint32_t kTrainingRecordsPerFrame = kTrainingBatchSize * kTrainingBatchesPerFrame;

//training tile side adapts per frame to saturate trainer
constexpr uint32_t kInitialTrainingTileSide = 8u;
constexpr uint32_t kMinTrainingTileSide     = 4u;
constexpr uint32_t kMaxTrainingTileSide     = 32u;

//1 in N training pixels takes long RR-terminated path for ground truth
constexpr uint32_t kUnbiasedDenom = 8u;

//per-path bucket cap, deeper paths drop tail vertices
constexpr uint32_t kMaxVerticesPerPath = 8u;

//max training paths per frame, headroom above 1080p 8x8 tiles
constexpr uint32_t kMaxTrainingPaths   = 65536u;

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
static_assert(sizeof(TrainingVertex) == 72, "TrainingVertex must match Nrc_v8.hlsli");

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
