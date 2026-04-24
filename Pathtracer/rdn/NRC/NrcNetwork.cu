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
#include <cmath>
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

        // Composite encoding: 14 raw dims → 62 encoded, padded to 64.
        //   3 pos  → TriangleWave(12 freqs)     = 36
        //   2 ω_sph → OneBlob(4 bins)            =  8
        //   2 n_sph → OneBlob(4 bins)            =  8
        //   1 rough → OneBlob(4 bins)            =  4
        //   6 α,β  → Identity                    =  6
        //                                        = 62 → padded to 64 by tcnn
        //
        // Matches paper §3.6 + §5 exactly. TriangleWave is tcnn's own
        // implementation of the NRC paper's per-frequency triangle-wave
        // encoding (sin-only equivalent + quartic approximation of sinf,
        // §5's perf trick). The previous Frequency encoder emitted
        // sin+cos per frequency (72 dims), which meant the first layer's
        // GEMM was 112×64 instead of 64×64 — ~1.75× more FLOPs with no
        // quality benefit.
        //
        // Positions must be normalized to roughly [0, 1] before this
        // encoder — NrcNormalizePosition in Nrc_v8.hlsli does that using
        // the live scene AABB. Without it, the encoder's period-2 lowest
        // frequency wraps many times across any real-size scene and the
        // network degenerates to a grid pattern.
        {"encoding", {
            {"otype",  "Composite"},
            {"nested", tcnn::json::array({
                tcnn::json{
                    {"n_dims_to_encode", 3u},
                    {"otype",            "TriangleWave"},
                    {"n_frequencies",    12u},
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

// Sanitize a target/feature entering the trainer. A single NaN or
// Inf poisons Adam's moment estimates (m/v) irrecoverably — the
// next update turns every weight into NaN, inference returns NaN,
// and the cache is dead for the lifetime of the process. We also
// cap to a large-but-finite upper bound so a legitimate-but-
// extreme value (a bright emitter at grazing angle, say) can't
// produce a gradient spike that knocks Adam's state off balance.
constexpr float kTargetMax = 1.0e4f;
__device__ __forceinline__ float safe_target(float v) {
    if (!isfinite(v)) return 0.0f;
    return fminf(fmaxf(v, 0.0f), kTargetMax);
}

// (Removed the golden-ratio shuffle — it scattered records across the
// full 65536-slot range, which broke contiguous per-batch slicing once
// we started training only on the actually-produced vertex count. The
// atomic counter already interleaves vertices from different paths
// across warps, so consecutive output slots come from different threads
// in practice — adequate decorrelation for SGD.)

// EMA update with bias correction (paper §3.3, eq. 2):
//
//   W_hat_t = (1-α)/η_t · W_t  +  α · η_{t-1}/η_t · W_hat_{t-1}
//   where η_t = 1 - α^t
//
// Equivalently: W_hat_t = coef_new · W_t + coef_old · W_hat_{t-1}
// with (coef_new + coef_old) = 1 and coefs → (1-α, α) as t→∞.
//
// Host passes the precomputed coefs per step. With W_hat_0 = 0 this
// makes W_hat_1 = W_1 (not "0.99·random + 0.01·W_1"), so the cache is
// usable within a few training steps instead of needing ~300 steps
// for the initial random weights to decay away.
__global__ void ema_update_kernel(
    __half*       __restrict__ ema,
    const __half* __restrict__ src,
    float                      coef_new,
    float                      coef_old,
    size_t                     n)
{
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float e = __half2float(ema[i]);
    const float s = __half2float(src[i]);
    ema[i] = __float2half(coef_old * e + coef_new * s);
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

    // Per-path vertex region.
    const uint8_t* vertBase = trainBuf + kPathMetaTotalBytes +
                              pathId * kMaxVerticesPerPath * kTrainVertexStride;
    const uint32_t lastV = (numVertices <= kMaxVerticesPerPath) ? (numVertices - 1u)
                                                                : (kMaxVerticesPerPath - 1u);

    // Seed the backward recursion with ACTUAL radiance at the "beyond last"
    // vertex. The reflectance factorization lives on the training target
    // (target_net = L_s / (α+β)) so:
    //   kTailEmitter / kTailMiss — tailRadPk already stores radiance
    //                              (emission / env radiance), use as-is.
    //   kTailCache                — tailRadPk stores (α+β) at the cache-
    //                              term vertex (NOT at the last training
    //                              vertex). MLP output is irradiance at
    //                              that vertex, so radiance = p · (α+β).
    //                              Using the wrong vertex's reflectance
    //                              biases the tail by the ratio of the
    //                              two reflectances — very visible on
    //                              specular/dark material transitions
    //                              and shows up as view-dependent
    //                              instability of the cache.
    //   kTailRR                   — tail = 0 (path died under RR).
    float tail_r = 0.0f, tail_g = 0.0f, tail_b = 0.0f;
    if (tailKind == kTailEmitter || tailKind == kTailMiss) {
        float3 t = unpack_rgb9e5(tailRadPk);
        tail_r = t.x; tail_g = t.y; tail_b = t.z;
    } else if (tailKind == kTailCache && inferenceSlot != kInvalidInferenceSlot) {
        const float3 reflCT = unpack_rgb9e5(tailRadPk);
        const float* p = inferenceOut + inferenceSlot * kOutputDim;
        // safe_target applies BOTH guards simultaneously:
        //   - NaN/Inf → 0 (prevents dead-Adam propagation if weights
        //     somehow went bad before we caught them)
        //   - negatives → 0 (prevents negative-attractor collapse)
        //   - large-magnitude cap (prevents Adam gradient spikes)
        tail_r = safe_target(p[0] * reflCT.x);
        tail_g = safe_target(p[1] * reflCT.y);
        tail_b = safe_target(p[2] * reflCT.z);
    }

    // Backward walk: last vertex first.
    for (int32_t v = int32_t(lastV); v >= 0; --v) {
        const uint8_t* vb = vertBase + uint32_t(v) * kTrainVertexStride;
        const float*   raw = reinterpret_cast<const float*>(vb);
        const uint32_t L_neePk     = *reinterpret_cast<const uint32_t*>(vb + 56);
        const uint32_t betaLocalPk = *reinterpret_cast<const uint32_t*>(vb + 60);

        float3 lnee = unpack_rgb9e5(L_neePk);
        float3 beta = unpack_rgb9e5(betaLocalPk);

        // MC estimator for actual radiance L_s[v] = L_nee + β · tail.
        const float ls_r = lnee.x + beta.x * tail_r;
        const float ls_g = lnee.y + beta.y * tail_g;
        const float ls_b = lnee.z + beta.z * tail_b;

        // Training target = L_s / (α+β) — irradiance. 1e-5 floor on
        // the denominator avoids blowing up on black materials.
        // `safe_target` collapses any NaN/Inf to 0 (preventing the
        // one-bad-value-kills-the-entire-network failure) and caps
        // the magnitude so a legitimate extreme sample doesn't spike
        // Adam's moments.
        const float rs_r = fmaxf(raw[8]  + raw[11], 1e-5f);
        const float rs_g = fmaxf(raw[9]  + raw[12], 1e-5f);
        const float rs_b = fmaxf(raw[10] + raw[13], 1e-5f);
        const float tgt_r = safe_target(ls_r / rs_r);
        const float tgt_g = safe_target(ls_g / rs_g);
        const float tgt_b = safe_target(ls_b / rs_b);

        // Pack contiguously by atomic counter — slot N gets the N-th
        // record produced (in atomic-order). Critical for the
        // adaptive-batch sizing in TrainFrame: it slices the buffer
        // into kTrainingBatchesPerFrame contiguous chunks of perBatch
        // records, and that's only well-defined if every slot below
        // validVertices is filled.
        const uint32_t outIdx = atomicAdd(outCounter, 1u);
        if (outIdx < kTrainingRecordsPerFrame) {
            float* f = outFeatures + outIdx * kRawInputDim;
            #pragma unroll
            for (uint32_t i = 0; i < kRawInputDim; ++i) f[i] = raw[i];
            float* t = outTargets + outIdx * kOutputDim;
            t[0] = tgt_r; t[1] = tgt_g; t[2] = tgt_b;
        }

        // Recursion passes ACTUAL radiance forward (backward in path order).
        tail_r = ls_r; tail_g = ls_g; tail_b = ls_b;
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

    // EMA of network weights (paper §3.3). tcnn's `inference_params`
    // pointer is rerouted to this buffer after init so inference sees
    // smoothed weights while the optimizer keeps mutating the raw ones.
    tcnn::network_precision_t* emaParams = nullptr;
    size_t                     nParams   = 0;
    float                      emaAlpha  = 0.99f;
    uint64_t                   emaStep   = 0;   // t in η_t = 1 - α^t

    // Last frame's valid vertex count (post-cap). Read by the renderer
    // to drive the adaptive-tile feedback loop. 0 before first frame.
    uint32_t                   lastValidVertices = 0;

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

        // ── EMA inference weights (paper §3.3) ────────────────────
        // Allocate a second fp16 weight buffer and reroute the Network's
        // inference pointer to it. Training keeps updating m_params;
        // inference reads m_emaParams. The EMA is bias-corrected after
        // each training_step in TrainFrame.
        //
        // IMPORTANT: ema starts at zero (W_hat_0 = 0), NOT at the
        // trainer's initial random weights. The bias-correction formula
        // η_t = 1 - α^t only holds under that initialization — seeding
        // from random weights means the first ~300 training steps are
        // dominated by "0.99^t · random_init", producing near-zero
        // inference outputs (i.e. a visibly dark cache) until the
        // random-weight contribution decays.
        const size_t nParams = m_impl->model.network->n_params();
        m_impl->nParams = nParams;
        if (cudaMalloc(&m_impl->emaParams,
                       nParams * sizeof(tcnn::network_precision_t)) != cudaSuccess) return false;
        cudaMemset(m_impl->emaParams, 0,
                   nParams * sizeof(tcnn::network_precision_t));
        m_impl->emaStep = 0;
        m_impl->model.network->set_params(
            m_impl->model.trainer->params(),          // training reads/writes this
            m_impl->emaParams,                        // inference reads this
            m_impl->model.trainer->param_gradients());

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
    if (m_impl->emaParams)     { cudaFree(m_impl->emaParams);     m_impl->emaParams     = nullptr; }
    m_impl->nParams = 0;
    m_impl->emaStep = 0;
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
    const float* targetDevPtr,
    uint32_t     count)
{
    if (!m_impl->ready || count == 0) return -1.0f;
    cudaStream_t stream = static_cast<cudaStream_t>(streamPtr);

    tcnn::GPUMatrix<float> input (const_cast<float*>(inputDevPtr),  kRawInputDim, count);
    tcnn::GPUMatrix<float> target(const_cast<float*>(targetDevPtr), kOutputDim,   count);
    auto ctx = m_impl->model.trainer->training_step(stream, input, target);
    (void)ctx;
    return -1.0f;
}

uint32_t Network::LastValidVertexCount() const {
    return m_impl ? m_impl->lastValidVertices : 0u;
}

void Network::TrainFrame(
    void*       streamPtr,
    const void* trainRecordsDevPtr,
    const void* inferenceOutDevPtr)
{
    if (!m_impl->ready) return;
    if (!trainRecordsDevPtr) return;
    cudaStream_t stream = static_cast<cudaStream_t>(streamPtr);

    // We size the per-batch row count from the actually-produced vertex
    // count below. The `cudaMemset` of features/targets is therefore
    // optional now (Adam never reads past `perBatch`), but we keep it
    // so a partial-fill scenario can't leak last frame's bytes into
    // the unused tail in case downstream code ever inspects the buffer.
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

    // Read back the actual vertex count so we can size the four SGD
    // batches to ONLY the rows we filled. ReadU32 stalls the stream,
    // which is the price for not training Adam on the zero-padded tail
    // (a `(features=0, target=0)` row is a real gradient step pulling
    // the network toward 0 at the all-zeros encoded input — see paper
    // §3.5 on adaptive tiles, which keeps the trainer saturated).
    uint32_t validVertices = ReadU32(streamPtr, m_impl->trainCounter);
    if (validVertices > kTrainingRecordsPerFrame)
        validVertices = kTrainingRecordsPerFrame;
    m_impl->lastValidVertices = validVertices;

    // Split across kTrainingBatchesPerFrame disjoint slices, each
    // rounded DOWN to kBatchGranularity. Round-down is required because
    // the trainer's matrix dim must equal the granularity multiple; if
    // we get e.g. 51000 vertices, that's 12750/batch → 12544 after
    // round-down (the dropped tail is ≤256 rows).
    const uint32_t perBatchRaw = validVertices / kTrainingBatchesPerFrame;
    const uint32_t perBatch    = (perBatchRaw / kBatchGranularity) * kBatchGranularity;
    if (perBatch == 0) return;   // nothing usable this frame

    // SGD + bias-corrected EMA per paper §3.3. tcnn's training_step
    // mutates m_impl->model.trainer->params() in place; we EMA those
    // raw weights into m_impl->emaParams which inference reads from.
    const uint32_t kEmaThreads = 256u;
    const uint32_t kEmaBlocks  = uint32_t((m_impl->nParams + kEmaThreads - 1u) / kEmaThreads);
    const cudaStream_t cudaStream = static_cast<cudaStream_t>(streamPtr);
    const float alpha = m_impl->emaAlpha;
    for (uint32_t b = 0; b < kTrainingBatchesPerFrame; ++b) {
        const float* fPtr = m_impl->trainFeatures + b * perBatch * kRawInputDim;
        const float* tPtr = m_impl->trainTargets  + b * perBatch * kOutputDim;
        TrainingStep(streamPtr, fPtr, tPtr, perBatch);

        // Bias-correction per paper §3.3:
        //   η_t = 1 - α^t
        //   W_hat_t = (1-α)/η_t · W_t + α · η_{t-1}/η_t · W_hat_{t-1}
        // At t=1 this becomes W_hat_1 = W_1 (ema gets the raw weights
        // directly, bypassing the "0.99·zero" attenuation). As t grows
        // the coefs settle to (1-α, α), i.e. the classical EMA.
        ++m_impl->emaStep;
        const double t    = (double)m_impl->emaStep;
        const double da   = (double)alpha;
        const double etaT = 1.0 - pow(da, t);                       // η_t
        const double etaP = (t > 1.0) ? 1.0 - pow(da, t - 1.0) : 0.0; // η_{t-1}
        const float  coefNew = (float)((1.0 - da) / etaT);
        const float  coefOld = (float)(da * etaP / etaT);

        ema_update_kernel<<<kEmaBlocks, kEmaThreads, 0, cudaStream>>>(
            m_impl->emaParams,
            m_impl->model.trainer->params(),
            coefNew,
            coefOld,
            m_impl->nParams);
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
