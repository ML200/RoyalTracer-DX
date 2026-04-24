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

        // Composite encoding: 16 raw dims → 74 encoded, padded to 80.
        //   3 pos  → HashGrid (16 levels × 2 features)  = 32
        //   3 ω    → SphericalHarmonics (degree 4)      = 16
        //   3 n    → SphericalHarmonics (degree 4)      = 16
        //   1 rough → OneBlob(4 bins)                    =  4
        //   6 α,β  → Identity                            =  6
        //                                                = 74 → padded by tcnn
        //
        // Position: HashGrid (Müller et al. "Instant-NGP", 2022). Tuning:
        //   - per_level_scale=1.38 (paper default for bounded scenes) so
        //     N_max = 16·1.38^15 ≈ 1500. An aggressive 1.5 gives sub-mm
        //     cells in a 50m scene — beyond NRC's ~cm radiance-smoothness
        //     regime, which mostly produces hash collisions without useful
        //     detail. 1.38 keeps the finest levels inside the regime where
        //     colliding gradients reinforce rather than fight each other.
        //   - Smoothstep interpolation (C1 continuous) removes the
        //     derivative discontinuities at cell boundaries that otherwise
        //     show up as low-frequency grid artifacts in the cache.
        //
        // Direction / normal: SphericalHarmonics degree 4 replaces the
        // octahedral → OneBlob chain. Benefits:
        //   - Analytic, smooth, rotationally equivariant — no octahedral
        //     seam on the lower hemisphere.
        //   - 16 coeffs per direction vs OneBlob's 8 bins — more
        //     expressive basis for the view-dependent term.
        //   - The HLSL side now emits the raw unit 3-vec remapped
        //     *0.5+0.5 into [0,1]³ (tcnn's SH impl remaps back to the
        //     sphere internally).
        //
        // ~2 MB of trainable hash params (2^19 entries × 2 features × fp16),
        // updated by Adam alongside the MLP weights.
        //
        // IMPORTANT: HashGrid expects positions in [0, 1]³ — the renderer
        // configures nrc_scene_scale_inv accordingly so x_norm lands in
        // exactly [0, 1] across the scene AABB.
        {"encoding", {
            {"otype",  "Composite"},
            {"nested", tcnn::json::array({
                tcnn::json{
                    {"n_dims_to_encode",       3u},
                    {"otype",                  "Grid"},
                    {"type",                   "Hash"},
                    {"n_levels",               16u},
                    {"n_features_per_level",   2u},
                    {"log2_hashmap_size",      19u},
                    {"base_resolution",        16u},
                    {"per_level_scale",        1.38f},
                    {"interpolation",          "Smoothstep"},
                },
                tcnn::json{
                    {"n_dims_to_encode", 3u},
                    {"otype",            "SphericalHarmonics"},
                    {"degree",           4u},
                },
                tcnn::json{
                    {"n_dims_to_encode", 3u},
                    {"otype",            "SphericalHarmonics"},
                    {"degree",           4u},
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

        // Fully-fused MLP, 2 hidden × 64 ReLU, linear out.
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
// Each TrainingVertex is 72 bytes (16 floats + 2 u32).
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
        const uint32_t L_neePk     = *reinterpret_cast<const uint32_t*>(vb + 64);
        const uint32_t betaLocalPk = *reinterpret_cast<const uint32_t*>(vb + 68);

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
        // Reflectance indices per new layout: α @ raw[10..12], β @ raw[13..15].
        const float rs_r = fmaxf(raw[10] + raw[13], 1e-5f);
        const float rs_g = fmaxf(raw[11] + raw[14], 1e-5f);
        const float rs_b = fmaxf(raw[12] + raw[15], 1e-5f);
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

    // Async readback for trainCounter. Pinned host memory + a CUDA event
    // let us overlap the copy with the rest of the frame instead of
    // stalling the CPU on cudaStreamSynchronize. The pending readback is
    // drained at the START of the next TrainFrame — by that point the
    // 4-byte DMA has long finished, so cudaEventSynchronize is a no-op.
    // Trade-off: batch sizing is sourced from the previous frame's
    // count (1-frame lag), which only matters during sharp transitions.
    uint32_t*                  hostCounterReadback   = nullptr;   // pinned
    cudaEvent_t                counterReadbackEvent  = nullptr;
    bool                       counterReadbackPending = false;

    // Pipelined training: a private CUDA stream that runs the entire
    // training pass in parallel with the next frame's raygen.
    //   inferenceDoneEvent — recorded on the main (interop) stream after
    //                        the inference launch; auxStream waits on it
    //                        before reading inferenceOut for class-1 tail
    //                        seeds.
    //   trainDoneEvent     — recorded on auxStream after the EMA update
    //                        for the last batch; the main stream waits
    //                        on it at the START of the next inference so
    //                        emaParams reads see the freshly-trained
    //                        weights instead of mid-write torn data.
    // Net effect: training (~1ms) overlaps with frame N+1's raygen
    // (~3.5ms) and disappears from the wall-clock budget.
    cudaStream_t               auxStream             = nullptr;
    cudaEvent_t                inferenceDoneEvent    = nullptr;
    cudaEvent_t                trainDoneEvent        = nullptr;

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

        // Pinned host buffer + event for async trainCounter readback.
        // cudaEventDisableTiming skips the timing slot (we only use it
        // for ordering, not measurement).
        if (cudaMallocHost(&m_impl->hostCounterReadback, sizeof(uint32_t)) != cudaSuccess) return false;
        *m_impl->hostCounterReadback = 0u;
        if (cudaEventCreateWithFlags(&m_impl->counterReadbackEvent,
                                     cudaEventDisableTiming) != cudaSuccess) return false;
        m_impl->counterReadbackPending = false;

        // Auxiliary stream + sync events for pipelined training.
        // cudaStreamNonBlocking lets auxStream overlap with the default
        // stream and the interop stream without implicit synchronization.
        if (cudaStreamCreateWithFlags(&m_impl->auxStream,
                                      cudaStreamNonBlocking) != cudaSuccess) return false;
        if (cudaEventCreateWithFlags(&m_impl->inferenceDoneEvent,
                                     cudaEventDisableTiming) != cudaSuccess) return false;
        if (cudaEventCreateWithFlags(&m_impl->trainDoneEvent,
                                     cudaEventDisableTiming) != cudaSuccess) return false;

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

        // JIT fusion (NVRTC) — compiles the entire encoder + MLP pipeline
        // into a SINGLE CUDA kernel at first invocation, instead of one
        // launch per encoder + one for the MLP. For our Composite encoder
        // with 5 nested sub-encoders this collapses 6 kernel launches per
        // inference into 1, which is the bulk of the inference cost on
        // modern GPUs (compute is fast, launch overhead × 6 is not).
        // Requires TCNN_RTC at build time (already enabled). The first
        // inference call pays a one-time NVRTC compile (~tens of ms);
        // every subsequent call uses the cached fused kernel. Falls back
        // gracefully to the non-fused path if the device or build doesn't
        // support it (set_jit_fusion clears m_jit_fusion internally on
        // any failure during kernel materialization).
        m_impl->model.network->set_jit_fusion(true);

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
    if (m_impl->hostCounterReadback) {
        cudaFreeHost(m_impl->hostCounterReadback);
        m_impl->hostCounterReadback = nullptr;
    }
    if (m_impl->counterReadbackEvent) {
        cudaEventDestroy(m_impl->counterReadbackEvent);
        m_impl->counterReadbackEvent = nullptr;
    }
    m_impl->counterReadbackPending = false;
    // Drain any in-flight training before tearing down the aux stream
    // so we don't destroy events that the GPU might still be waiting on.
    if (m_impl->auxStream) {
        cudaStreamSynchronize(m_impl->auxStream);
        cudaStreamDestroy(m_impl->auxStream);
        m_impl->auxStream = nullptr;
    }
    if (m_impl->inferenceDoneEvent) {
        cudaEventDestroy(m_impl->inferenceDoneEvent);
        m_impl->inferenceDoneEvent = nullptr;
    }
    if (m_impl->trainDoneEvent) {
        cudaEventDestroy(m_impl->trainDoneEvent);
        m_impl->trainDoneEvent = nullptr;
    }
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

    // Pipelined training: wait for the previous frame's training pass to
    // finish updating emaParams before reading them here. The wait is
    // recorded on the inference stream and is a no-op once the aux
    // stream's event has fired (which is virtually always — training
    // takes ~1 ms and runs entirely behind the next frame's raygen).
    // First call does nothing because the event has never been recorded.
    cudaStreamWaitEvent(stream, m_impl->trainDoneEvent, 0);

    tcnn::GPUMatrix<float> input (const_cast<float*>(inputDevPtr),  kRawInputDim, count);
    tcnn::GPUMatrix<float> output(                  outputDevPtr,   kOutputDim,   count);
    m_impl->model.network->inference(stream, input, output);

    // Signal that inferenceOut is now valid for the training pass to
    // consume (class-1 paths use this frame's predictions as their
    // backward-fill tail). TrainFrame waits on this event on auxStream.
    cudaEventRecord(m_impl->inferenceDoneEvent, stream);
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

void Network::WaitIdle() {
    if (!m_impl || !m_impl->auxStream) return;
    cudaStreamSynchronize(m_impl->auxStream);
    // Drain the pending counter readback too — once we return, the host
    // can safely reallocate buffers without worrying about a kernel still
    // referencing them.
    if (m_impl->counterReadbackPending) {
        cudaEventSynchronize(m_impl->counterReadbackEvent);
        uint32_t v = *m_impl->hostCounterReadback;
        if (v > kTrainingRecordsPerFrame) v = kTrainingRecordsPerFrame;
        m_impl->lastValidVertices = v;
        m_impl->counterReadbackPending = false;
    }
}

void Network::TrainFrame(
    void*       streamPtr,
    const void* trainRecordsDevPtr,
    const void* inferenceOutDevPtr)
{
    if (!m_impl->ready) return;
    if (!trainRecordsDevPtr) return;
    // streamPtr is the renderer's main interop stream; we ignore it for
    // training and run everything on auxStream so this whole pass can
    // overlap with the next frame's raygen / inference. The cross-stream
    // sync below makes the dependency on inferenceOut explicit.
    (void)streamPtr;
    cudaStream_t stream = m_impl->auxStream;

    // Wait for THIS frame's inference to finish writing inferenceOut —
    // class-1 (TRAIN_BIASED) paths' backward fill seeds their tail from
    // the network's own prediction at the cache-term vertex, so the
    // fill kernel below reads those float3 outputs and would race with
    // the inference write without this barrier.
    cudaStreamWaitEvent(stream, m_impl->inferenceDoneEvent, 0);

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

    // Drain the previous frame's async readback BEFORE issuing a new one.
    // The 4-byte D2H DMA finishes in microseconds; with a frame interval
    // of many ms, the event is virtually always already signaled here, so
    // cudaEventSynchronize is a no-op. Worst case (first call after a
    // long stall) this waits a few μs — far better than the per-frame
    // cudaStreamSynchronize the synchronous ReadU32 used to do.
    if (m_impl->counterReadbackPending) {
        cudaEventSynchronize(m_impl->counterReadbackEvent);
        uint32_t v = *m_impl->hostCounterReadback;
        if (v > kTrainingRecordsPerFrame) v = kTrainingRecordsPerFrame;
        m_impl->lastValidVertices = v;
        m_impl->counterReadbackPending = false;
    }

    // Queue the readback for THIS frame's count — drained by next frame's
    // call. Recording an event AFTER the memcpy on the same stream lets
    // us synchronize on it without serializing the rest of the pipeline.
    cudaMemcpyAsync(m_impl->hostCounterReadback, m_impl->trainCounter,
                    sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    cudaEventRecord(m_impl->counterReadbackEvent, stream);
    m_impl->counterReadbackPending = true;

    // Size THIS frame's training batches from the PREVIOUS frame's count.
    // For stable scenes this is essentially the same as the current count;
    // for fast-changing ones, batch size lags by one frame which is
    // acceptable (a few hundred vertices over- or under-trained out of
    // tens of thousands). The trade-off buys back the per-frame stall.
    const uint32_t validVertices = m_impl->lastValidVertices;

    // Split across kTrainingBatchesPerFrame disjoint slices, each
    // rounded DOWN to kBatchGranularity. Round-down is required because
    // the trainer's matrix dim must equal the granularity multiple; if
    // we get e.g. 51000 vertices, that's 12750/batch → 12544 after
    // round-down (the dropped tail is ≤256 rows).
    const uint32_t perBatchRaw = validVertices / kTrainingBatchesPerFrame;
    const uint32_t perBatch    = (perBatchRaw / kBatchGranularity) * kBatchGranularity;
    if (perBatch == 0) {
        // Even when there's nothing to train, signal trainDoneEvent so
        // the next inference doesn't wait on a stale event from a prior
        // training pass that may have referenced different emaParams.
        cudaEventRecord(m_impl->trainDoneEvent, stream);
        return;
    }

    // SGD + bias-corrected EMA per paper §3.3. tcnn's training_step
    // mutates m_impl->model.trainer->params() in place; we EMA those
    // raw weights into m_impl->emaParams which inference reads from.
    const uint32_t kEmaThreads = 256u;
    const uint32_t kEmaBlocks  = uint32_t((m_impl->nParams + kEmaThreads - 1u) / kEmaThreads);
    const float alpha = m_impl->emaAlpha;
    for (uint32_t b = 0; b < kTrainingBatchesPerFrame; ++b) {
        const float* fPtr = m_impl->trainFeatures + b * perBatch * kRawInputDim;
        const float* tPtr = m_impl->trainTargets  + b * perBatch * kOutputDim;
        TrainingStep((void*)stream, fPtr, tPtr, perBatch);

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

        ema_update_kernel<<<kEmaBlocks, kEmaThreads, 0, stream>>>(
            m_impl->emaParams,
            m_impl->model.trainer->params(),
            coefNew,
            coefOld,
            m_impl->nParams);
    }

    // Signal that emaParams is fully updated for this frame — the next
    // frame's inference will wait on this event before reading.
    cudaEventRecord(m_impl->trainDoneEvent, stream);
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
