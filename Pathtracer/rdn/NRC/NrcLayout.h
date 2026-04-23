#pragma once
// ═══════════════════════════════════════════════════════════════════
// NRC/NrcLayout.h — buffer layout contracts shared across C++, CUDA,
// and HLSL. Any struct or constant that crosses the language boundary
// lives here; the HLSL side mirrors the same offsets in Nrc_v8.hlsli.
// ═══════════════════════════════════════════════════════════════════

#include <cstdint>

namespace nrc {

// ── Runtime-tunable settings (Editor) ───────────────────────────────
// Mirrored into push constants every frame (see Renderer.cpp).
struct Settings {
    bool  enabled          = true;   // master toggle for cache termination + resolve
    bool  trainingEnabled  = true;   // freeze weights when off
    bool  debugView        = false;  // render L̂_s at primary vertex to gOutput slice 3
    float areaSpreadC      = 0.01f;  // paper's `c` — smaller = terminate earlier
    float learningRateScale = 1.0f;  // reserved for a tcnn-side LR override
};

// Push-constant flag bits — keep in sync with Nrc_v8.hlsli.
namespace flags {
    constexpr uint32_t kEnabled     = 1u << 0;
    constexpr uint32_t kTrain       = 1u << 1;
    constexpr uint32_t kDebugView   = 1u << 2;
}

// ── Network dimensions ──────────────────────────────────────────────
// Raw feature vector before tcnn's composite encoding. tcnn expands
// this to 62 dims internally (frequency on pos, one-blob on ω/n/r,
// identity on α/β). Layout — written by raygen, read by Inference:
//   0..2    position
//   3..4    scattered direction, sph(u,v) in [0,1]^2
//   5..6    surface normal,      sph(u,v) in [0,1]^2
//   7       roughness, mapped 1 - exp(-r)
//   8..10   diffuse reflectance  (rgb)
//   11..13  specular reflectance (rgb)
constexpr uint32_t kRawInputDim  = 14;
constexpr uint32_t kOutputDim    = 3;    // scattered radiance rgb

constexpr uint32_t kHiddenWidth  = 64;
constexpr uint32_t kHiddenLayers = 5;

// tcnn requires every inference / training batch to be a multiple of
// this. Pad up in shader / host as needed.
constexpr uint32_t kBatchGranularity = 256;

// Training schedule per frame (paper §3.5).
constexpr uint32_t kTrainingBatchSize       = 16384;
constexpr uint32_t kTrainingBatchesPerFrame = 4;
constexpr uint32_t kTrainingRecordsPerFrame = kTrainingBatchSize * kTrainingBatchesPerFrame; // 65536

// NRC training tile: one training pixel per tile. 8×8 = 64 pixels per
// tile → ~32k training pixels at 1080p → ~32–100k records depending on
// suffix length. Kept simple for now; adaptive sizing is a later pass.
constexpr uint32_t kTrainingTileSide = 8u;

// First-pass training uses ONLY fully unbiased (Russian-roulette-terminated
// or emitter-hit) paths so we can skip the class-1 suffix machinery.
// kUnbiasedDenom still determines the bias/unbiased mix once class 1 lands.
constexpr uint32_t kUnbiasedDenom = 16u;

// Per-path bucket cap. Paths deeper than this get their tail vertices
// dropped — training still captures the prefix correctly. Paper averages
// 1–3 vertices / training path, so 8 is comfortable.
constexpr uint32_t kMaxVerticesPerPath = 8u;

// Maximum number of training paths per frame. Sized with headroom above
// 1920×1080 with 8×8 tiles (~32k training pixels) so higher resolutions
// don't clip and the atomic allocator has slack. Buffer cost scales
// roughly as kMaxTrainingPaths * kMaxVerticesPerPath * 64 B.
constexpr uint32_t kMaxTrainingPaths   = 65536u;

// Sentinels.
constexpr uint32_t kInvalidInferenceSlot = 0xFFFFFFFFu;
constexpr uint32_t kInvalidTrainPath     = 0xFFFFFFFFu;

// Tail kinds — how a training path's terminal vertex ended. Determines
// the base value for the backward-fill recursion:
//   target[v] = L_nee[v] + β[v] · target[v+1], target[last+1] = tail.
enum TailKind : uint32_t {
    kTailInvalid = 0u,   // path never terminated correctly (skip)
    kTailRR      = 1u,   // Russian-roulette killed — tail = 0
    kTailEmitter = 2u,   // hit emitter — tail = emission
    kTailMiss    = 3u,   // missed geometry — tail = env radiance
    kTailCache   = 4u,   // (class-1 only, later) tail = L̂_s[slot]
};

// ── Per-record strides (bytes) ──────────────────────────────────────
// tcnn expects column-major, so sample `i` occupies contiguous bytes
// [i * stride, (i+1) * stride). Both input and output are fp32 —
// tcnn's inference() overload we call takes GPUMatrix<float>.
constexpr uint32_t kInferenceInputStride  = kRawInputDim * sizeof(float);   // 56
constexpr uint32_t kInferenceOutputStride = kOutputDim   * sizeof(float);   // 12

// ── PendingGI ───────────────────────────────────────────────────────
// Per-pixel state written by raygen on cache termination, consumed by
// the NRC resolve compute shader once inference has filled in L̂_s.
struct PendingGI {
    uint32_t inferenceSlot;   // kInvalidInferenceSlot = no pending record
    uint32_t throughputPk;    // RGB9E5, β at terminal vertex excluding L̂_s
    uint32_t tpostPk;         // RGB9E5, Π(BSDF·cos·T) from x2 to terminal
    float    pdfProduct;      // full-path sampling pdf so far
};
static_assert(sizeof(PendingGI) == 16, "PendingGI must match Nrc_v8.hlsli");

// ── TrainingPathMeta ────────────────────────────────────────────────
// One per training path. Lives in the first kPathMetaTotalBytes of the
// training buffer so no separate UAV is needed.
//   numVertices == 0  → path slot unused (ignored by training kernel)
//   tailKind          → picks the base value for the backward recursion
//   inferenceSlot     → valid iff tailKind == kTailCache
//   tailRadiancePk    → packed RGB9E5; valid iff tailKind in {Emitter,Miss}
struct TrainingPathMeta {
    uint32_t numVertices;
    uint32_t tailKind;
    uint32_t inferenceSlot;
    uint32_t tailRadiancePk;
};
static_assert(sizeof(TrainingPathMeta) == 16, "TrainingPathMeta must match Nrc_v8.hlsli");

// ── TrainingVertex ──────────────────────────────────────────────────
// Per-vertex record, indexed by [pathId * kMaxVerticesPerPath + vIdx],
// offset into the training buffer by kPathMetaTotalBytes.
//   raw           — network input features at this vertex
//   L_neePk       — ΣNEE estimator contribution local to this vertex
//                   (already free of upstream throughput / pdf chain)
//   betaLocalPk   — BSDF·cos·T / pdf at this vertex; transports radiance
//                   FROM v+1 TO v during backward fill.
struct TrainingVertex {
    float    raw[kRawInputDim];   // 56 B
    uint32_t L_neePk;             //  4 B
    uint32_t betaLocalPk;         //  4 B
};
static_assert(sizeof(TrainingVertex) == 64, "TrainingVertex must match Nrc_v8.hlsli");

constexpr uint32_t kPathMetaStride    = sizeof(TrainingPathMeta);   // 16
constexpr uint32_t kTrainVertexStride = sizeof(TrainingVertex);     // 64
constexpr uint32_t kPathMetaTotalBytes = kMaxTrainingPaths * kPathMetaStride;  // 524288

inline constexpr uint32_t TrainingMetaOffset(uint32_t pathId) {
    return pathId * kPathMetaStride;
}
inline constexpr uint32_t TrainingVertexOffset(uint32_t pathId, uint32_t vIdx) {
    return kPathMetaTotalBytes + (pathId * kMaxVerticesPerPath + vIdx) * kTrainVertexStride;
}

// ── Counters ────────────────────────────────────────────────────────
struct Counters {
    uint32_t inferenceCount;    // next inference slot to allocate
    uint32_t trainingPathCount; // next training path id to allocate
    uint32_t pad0;
    uint32_t pad1;
};
static_assert(sizeof(Counters) == 16, "Counters must match Nrc_v8.hlsli");

// ── Buffer byte totals ──────────────────────────────────────────────
inline uint64_t InferenceInputBytes (uint32_t maxInferenceRecords) {
    return uint64_t(maxInferenceRecords) * kInferenceInputStride;
}
inline uint64_t InferenceOutputBytes(uint32_t maxInferenceRecords) {
    return uint64_t(maxInferenceRecords) * kInferenceOutputStride;
}
inline uint64_t PendingGIBytes      (uint32_t pixelCount) {
    return uint64_t(pixelCount) * sizeof(PendingGI);
}
// Per-path meta section + per-vertex bucket section, contiguous.
inline uint64_t TrainingBytes       () {
    return uint64_t(kPathMetaTotalBytes) +
           uint64_t(kMaxTrainingPaths) * kMaxVerticesPerPath * kTrainVertexStride;
}
inline uint64_t CountersBytes       () {
    return sizeof(Counters);
}

// Round a batch count up to the tcnn granularity.
constexpr uint32_t AlignBatch(uint32_t n) {
    return (n + kBatchGranularity - 1u) / kBatchGranularity * kBatchGranularity;
}

} // namespace nrc
