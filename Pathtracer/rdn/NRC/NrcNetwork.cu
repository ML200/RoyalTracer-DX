// ═══════════════════════════════════════════════════════════════════
// NRC/NrcNetwork.cu — tcnn wrappers for the NRC radiance cache.
//
// This is the only TU that includes tcnn / CUDA headers. The public
// class lives in NrcNetwork.h and uses only void* / float*.
// ═══════════════════════════════════════════════════════════════════

#include "NrcNetwork.h"

#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/gpu_matrix.h>

#include <cuda_runtime.h>
#include <cstdio>
#include <exception>

namespace nrc {

// ── json builder for the NRC network ──────────────────────────────
static tcnn::json BuildNetworkConfig() {
    return tcnn::json{
        // Relative L2 admits unbiased gradient estimates under a noisy
        // training signal (paper §5, Lehtinen et al. 2018).
        {"loss", {{"otype", "RelativeL2"}}},

        // Paper §3.5 uses Adam with an aggressive LR; we start with the
        // same and let EMA smooth the effective inference weights.
        {"optimizer", {
            {"otype",         "Adam"},
            {"learning_rate", 1e-2f},
            {"beta1",         0.9f},
            {"beta2",         0.99f},
            {"epsilon",       1e-8f},
            {"l2_reg",        1e-6f},
        }},

        // Composite encoding: 14 raw dims → 62 encoded.
        //   3 pos  → Frequency(6 freqs, sin+cos) = 36
        //   2 ω_sph → OneBlob(4 bins)            =  8
        //   2 n_sph → OneBlob(4 bins)            =  8
        //   1 rough → OneBlob(4 bins)            =  4
        //   6 α,β  → Identity                    =  6
        //                                        = 62 → padded to 64 by tcnn
        {"encoding", {
            {"otype",  "Composite"},
            {"nested", tcnn::json::array({
                tcnn::json{
                    {"n_dims_to_encode", 3u},
                    {"otype",            "Frequency"},
                    {"n_frequencies",    6u},
                },
                tcnn::json{
                    {"n_dims_to_encode", 2u},
                    {"otype",            "OneBlob"},
                    {"n_bins",           4u},
                },
                tcnn::json{
                    {"n_dims_to_encode", 2u},
                    {"otype",            "OneBlob"},
                    {"n_bins",           4u},
                },
                tcnn::json{
                    {"n_dims_to_encode", 1u},
                    {"otype",            "OneBlob"},
                    {"n_bins",           4u},
                },
                tcnn::json{
                    {"n_dims_to_encode", 6u},
                    {"otype",            "Identity"},
                },
            })},
        }},

        // Fully-fused MLP, 5 hidden × 64 ReLU, linear out.
        {"network", {
            {"otype",             "FullyFusedMLP"},
            {"activation",        "ReLU"},
            {"output_activation", "None"},
            {"n_neurons",         kHiddenWidth},
            {"n_hidden_layers",   kHiddenLayers},
        }},
    };
}

// ── RGB9E5 unpack (must match shaders/Compression_v8.hlsli) ───────
__device__ __forceinline__ float3 unpack_rgb9e5(uint32_t p) {
    const uint32_t rm = (p >>  0) & 0x1FFu;
    const uint32_t gm = (p >>  9) & 0x1FFu;
    const uint32_t bm = (p >> 18) & 0x1FFu;
    const int      e  = int((p >> 27) & 0x1Fu);
    const float scale = exp2f(float(e - 15 - 9));      // 2^(e - bias - mantissa_bits)
    return make_float3(float(rm) * scale, float(gm) * scale, float(bm) * scale);
}

// ── Backward-fill kernel ──────────────────────────────────────────
// One thread per path id. Reads the TrainingPathMeta, seeds the tail
// from tailKind, walks the per-path vertex bucket in reverse, emits
// (features, target) rows into the tcnn-ready flat buffers. Output
// slot is an atomic-add counter; atomic ordering naturally shuffles
// records across paths within each warp.
//
// Training buffer layout (kept in sync with NrcLayout.h):
//   [0, kPathMetaTotalBytes)                                   : TrainingPathMeta[kMaxTrainingPaths]
//   [kPathMetaTotalBytes, ...)                                  : TrainingVertex[kMaxTrainingPaths][kMaxVerticesPerPath]
// Each TrainingPathMeta is 16 bytes (4 u32).
// Each TrainingVertex is 64 bytes (14 floats + 2 u32).
__global__ void fill_training_batch_kernel(
    const uint8_t* __restrict__ trainBuf,
    const float*   __restrict__ inferenceOut,
    uint32_t                    trainingPathCount,   // live paths this frame (from counter)
    float*         __restrict__ outFeatures,         // [kTrainingRecordsPerFrame × kRawInputDim], col-major
    float*         __restrict__ outTargets,          // [kTrainingRecordsPerFrame × kOutputDim], col-major
    uint32_t*      __restrict__ outCounter)          // atomic write head
{
    const uint32_t pathId = blockIdx.x * blockDim.x + threadIdx.x;
    if (pathId >= trainingPathCount || pathId >= kMaxTrainingPaths) return;

    // Meta at offset pathId * 16.
    const uint32_t* meta = reinterpret_cast<const uint32_t*>(trainBuf + pathId * kPathMetaStride);
    const uint32_t numVertices  = meta[0];
    const uint32_t tailKind     = meta[1];
    const uint32_t inferenceSlot = meta[2];
    const uint32_t tailRadPk    = meta[3];
    if (numVertices == 0 || tailKind == kTailInvalid) return;

    // Seed the backward recursion.
    float tail_r = 0.0f, tail_g = 0.0f, tail_b = 0.0f;
    if (tailKind == kTailEmitter || tailKind == kTailMiss) {
        float3 t = unpack_rgb9e5(tailRadPk);
        tail_r = t.x; tail_g = t.y; tail_b = t.z;
    } else if (tailKind == kTailCache && inferenceSlot != kInvalidInferenceSlot) {
        const float* p = inferenceOut + inferenceSlot * kOutputDim;
        tail_r = p[0]; tail_g = p[1]; tail_b = p[2];
    }
    // kTailRR keeps tail = 0.

    // Per-path vertex region.
    const uint8_t* vertBase = trainBuf + kPathMetaTotalBytes +
                              pathId * kMaxVerticesPerPath * kTrainVertexStride;

    // Backward walk: last vertex first.
    const uint32_t lastV = (numVertices <= kMaxVerticesPerPath) ? (numVertices - 1u)
                                                                : (kMaxVerticesPerPath - 1u);
    for (int32_t v = int32_t(lastV); v >= 0; --v) {
        const uint8_t* vb = vertBase + uint32_t(v) * kTrainVertexStride;
        // Raw features sit as 14 fp32 at offset 0.
        const float* raw = reinterpret_cast<const float*>(vb);
        const uint32_t L_neePk     = *reinterpret_cast<const uint32_t*>(vb + 56);
        const uint32_t betaLocalPk = *reinterpret_cast<const uint32_t*>(vb + 60);

        float3 lnee = unpack_rgb9e5(L_neePk);
        float3 beta = unpack_rgb9e5(betaLocalPk);

        // target = L_nee + β · tail (componentwise)
        const float target_r = lnee.x + beta.x * tail_r;
        const float target_g = lnee.y + beta.y * tail_g;
        const float target_b = lnee.z + beta.z * tail_b;

        // Emit (features, target). Drop if we're past the batch cap.
        const uint32_t outIdx = atomicAdd(outCounter, 1u);
        if (outIdx < kTrainingRecordsPerFrame) {
            // Column-major: sample `outIdx` occupies contiguous 14 floats.
            float* f = outFeatures + outIdx * kRawInputDim;
            #pragma unroll
            for (uint32_t i = 0; i < kRawInputDim; ++i) f[i] = raw[i];
            float* t = outTargets + outIdx * kOutputDim;
            t[0] = target_r; t[1] = target_g; t[2] = target_b;
        }

        // Roll the recursion forward (backward in path order).
        tail_r = target_r; tail_g = target_g; tail_b = target_b;
    }
}

// ── Impl ──────────────────────────────────────────────────────────
struct Network::Impl {
    tcnn::TrainableModel model;

    // Per-frame training buffers on the device. tcnn wraps these via
    // GPUMatrix<float> at TrainingStep time.
    float*    trainFeatures = nullptr;   // kTrainingRecordsPerFrame × kRawInputDim
    float*    trainTargets  = nullptr;   // kTrainingRecordsPerFrame × kOutputDim
    uint32_t* trainCounter  = nullptr;   // single u32

    bool ready = false;
};

Network::Network() : m_impl(std::make_unique<Impl>()) {}
Network::~Network() = default;

bool Network::Init() {
    if (m_impl->ready) return true;
    try {
        m_impl->model = tcnn::create_from_config(kRawInputDim, kOutputDim, BuildNetworkConfig());
        if (!m_impl->model.network || !m_impl->model.trainer) return false;

        // Persistent training scratch.
        const size_t fBytes = size_t(kTrainingRecordsPerFrame) * kRawInputDim * sizeof(float);
        const size_t tBytes = size_t(kTrainingRecordsPerFrame) * kOutputDim   * sizeof(float);
        if (cudaMalloc(&m_impl->trainFeatures, fBytes) != cudaSuccess) return false;
        if (cudaMalloc(&m_impl->trainTargets,  tBytes) != cudaSuccess) return false;
        if (cudaMalloc(&m_impl->trainCounter,  sizeof(uint32_t)) != cudaSuccess) return false;

        m_impl->ready = true;
        return true;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "[NRC] Network::Init threw: %s\n", ex.what());
        return false;
    }
}

void Network::Shutdown() {
    m_impl->ready = false;
    if (m_impl->trainFeatures) { cudaFree(m_impl->trainFeatures); m_impl->trainFeatures = nullptr; }
    if (m_impl->trainTargets)  { cudaFree(m_impl->trainTargets);  m_impl->trainTargets  = nullptr; }
    if (m_impl->trainCounter)  { cudaFree(m_impl->trainCounter);  m_impl->trainCounter  = nullptr; }
    m_impl->model = tcnn::TrainableModel{};
}

bool Network::IsReady() const {
    return m_impl->ready;
}

void Network::Inference(
    void*        streamPtr,
    const float* inputDevPtr,
    float*       outputDevPtr,
    uint32_t     count)
{
    if (!m_impl->ready || count == 0) return;
    if (count % kBatchGranularity != 0) {
        std::fprintf(stderr,
            "[NRC] Inference count %u not a multiple of %u — caller must pad.\n",
            count, kBatchGranularity);
        return;
    }
    cudaStream_t stream = static_cast<cudaStream_t>(streamPtr);

    tcnn::GPUMatrix<float> input (const_cast<float*>(inputDevPtr),  kRawInputDim, count);
    tcnn::GPUMatrix<float> output(                  outputDevPtr,   kOutputDim,   count);
    m_impl->model.network->inference(stream, input, output);
}

float Network::TrainingStep(
    void*        streamPtr,
    const float* inputDevPtr,
    const float* targetDevPtr)
{
    if (!m_impl->ready) return -1.0f;
    cudaStream_t stream = static_cast<cudaStream_t>(streamPtr);

    tcnn::GPUMatrix<float> input (const_cast<float*>(inputDevPtr),  kRawInputDim,  kTrainingBatchSize);
    tcnn::GPUMatrix<float> target(const_cast<float*>(targetDevPtr), kOutputDim,    kTrainingBatchSize);
    auto ctx = m_impl->model.trainer->training_step(stream, input, target);
    (void)ctx;
    return -1.0f;
}

void Network::TrainFrame(
    void*       streamPtr,
    const void* trainRecordsDevPtr,
    const void* inferenceOutDevPtr)
{
    if (!m_impl->ready) return;
    if (!trainRecordsDevPtr) return;
    cudaStream_t stream = static_cast<cudaStream_t>(streamPtr);

    // Zero the training scratch — any vertices we fail to fill stay at
    // (features=0, target=0) which is a harmless no-op row for Adam.
    const size_t fBytes = size_t(kTrainingRecordsPerFrame) * kRawInputDim * sizeof(float);
    const size_t tBytes = size_t(kTrainingRecordsPerFrame) * kOutputDim   * sizeof(float);
    cudaMemsetAsync(m_impl->trainFeatures, 0, fBytes, stream);
    cudaMemsetAsync(m_impl->trainTargets,  0, tBytes, stream);
    cudaMemsetAsync(m_impl->trainCounter,  0, sizeof(uint32_t), stream);

    // Launch the fill kernel over every possible path slot. Threads
    // early-out on numVertices==0, so the cost is dominated by the live
    // paths.
    constexpr uint32_t kThreads = 256;
    const uint32_t blocks = (kMaxTrainingPaths + kThreads - 1u) / kThreads;
    fill_training_batch_kernel<<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(trainRecordsDevPtr),
        reinterpret_cast<const float*>  (inferenceOutDevPtr),
        kMaxTrainingPaths,
        m_impl->trainFeatures,
        m_impl->trainTargets,
        m_impl->trainCounter);

    // Four SGD steps per frame on disjoint 16384-record slices.
    for (uint32_t b = 0; b < kTrainingBatchesPerFrame; ++b) {
        const float* fPtr = m_impl->trainFeatures + b * kTrainingBatchSize * kRawInputDim;
        const float* tPtr = m_impl->trainTargets  + b * kTrainingBatchSize * kOutputDim;
        TrainingStep(streamPtr, fPtr, tPtr);
    }
}

// ── CUDA helpers ──────────────────────────────────────────────────
void Memzero(void* streamPtr, void* devPtr, size_t bytes) {
    if (!devPtr || bytes == 0) return;
    cudaMemsetAsync(devPtr, 0, bytes, static_cast<cudaStream_t>(streamPtr));
}

void Memfill(void* streamPtr, void* devPtr, int value, size_t bytes) {
    if (!devPtr || bytes == 0) return;
    cudaMemsetAsync(devPtr, value, bytes, static_cast<cudaStream_t>(streamPtr));
}

uint32_t ReadU32(void* streamPtr, const void* devPtr) {
    if (!devPtr) return 0u;
    cudaStream_t stream = static_cast<cudaStream_t>(streamPtr);
    uint32_t host = 0u;
    cudaError_t e = cudaMemcpyAsync(&host, devPtr, sizeof(uint32_t),
                                    cudaMemcpyDeviceToHost, stream);
    if (e != cudaSuccess) return 0u;
    cudaStreamSynchronize(stream);
    return host;
}

// ── Smoke test (kept from prior revision) ─────────────────────────
bool SmokeTest(void* streamPtr) {
    cudaStream_t stream = static_cast<cudaStream_t>(streamPtr);

    try {
        tcnn::json cfg = {
            {"loss",      {{"otype", "L2"}}},
            {"optimizer", {{"otype", "Adam"}, {"learning_rate", 1e-3}}},
            {"encoding",  {{"otype", "Identity"}}},
            {"network",   {
                {"otype",             "FullyFusedMLP"},
                {"activation",        "ReLU"},
                {"output_activation", "None"},
                {"n_neurons",         64},
                {"n_hidden_layers",   2},
            }},
        };

        constexpr uint32_t kIn = 32, kOut = 3, kBatch = 256;

        tcnn::TrainableModel model = tcnn::create_from_config(kIn, kOut, cfg);
        if (!model.network || !model.trainer) return false;

        tcnn::GPUMatrix<float> input (kIn,  kBatch);
        tcnn::GPUMatrix<float> target(kOut, kBatch);
        tcnn::GPUMatrix<float> output(kOut, kBatch);
        input .memset_async(stream, 0);
        target.memset_async(stream, 0);

        model.network->inference(stream, input, output);
        model.trainer->training_step(stream, input, target);

        cudaStreamSynchronize(stream);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            std::fprintf(stderr, "[NRC] SmokeTest CUDA error: %s\n",
                cudaGetErrorString(e));
            return false;
        }
        return true;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "[NRC] SmokeTest threw: %s\n", ex.what());
        return false;
    }
}

} // namespace nrc
