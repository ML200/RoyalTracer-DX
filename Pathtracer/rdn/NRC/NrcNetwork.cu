//====================================
//NRC TCNN WRAPPER
//====================================
//only TU that includes tcnn/CUDA headers, public class in NrcNetwork.h

#include "NrcNetwork.h"

#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/gpu_matrix.h>

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <exception>

namespace nrc {

//====================================
//NETWORK CONFIG
//====================================
static tcnn::json BuildNetworkConfig() {
    return tcnn::json{
        //RelativeL2 per Müller et al. 2021 §5: (pred - tgt)^2 / (sg(pred)^2 + eps).
        //Plain L2 gradient scales with target magnitude, so across a shuffled
        //batch mixing direct-illumination targets (O(1000)) with indirect
        //targets (O(0.01-1)), the bright samples dominate Adam's updates and
        //the MLP underfits dark regions. RelativeL2's per-sample normalization
        //by prediction magnitude makes gradient contributions scale-invariant,
        //giving dark samples equal pull. Linear target (no sqrt/log) keeps
        //the optimum unbiased at E[L/r]; safe_target's kTargetMax=1e4 cap
        //keeps sg(pred)^2 in fp16-safe range (max 6.5e4).
        {"loss", {{"otype", "RelativeL2"}}},

        //Conservative LR matching the prior stable baseline.
        {"optimizer", {
            {"otype",         "Adam"},
            {"learning_rate", 1e-3f},
            {"beta1",         0.9f},
            {"beta2",         0.99f},
            {"epsilon",       1e-8f},
            {"l2_reg",        1e-6f},
        }},

        //composite encoding, 16 raw -> 74 encoded
        //3 pos HashGrid 16 levels x 2 features = 32
        //3 dir SH deg 4 = 16, 3 normal SH deg 4 = 16
        //1 rough OneBlob 4 bins = 4, 6 alpha/beta Identity = 6
        //HashGrid per_level_scale 1.38 bounds N_max to ~cm regime, Smoothstep C1 for clean gradients
        //SH replaces octahedral+OneBlob, analytic smooth rotation-equivariant
        //HashGrid requires positions in [0,1]^3, renderer normalizes via nrc_scene_scale_inv
        {"encoding", {
            {"otype",  "Composite"},
            {"nested", tcnn::json::array({
                tcnn::json{
                    {"n_dims_to_encode",       3u},
                    {"otype",                  "Grid"},
                    {"type",                   "Hash"},
                    {"n_levels",               16u},
                    {"n_features_per_level",   2u},
                    //log2=19 -- the bump to 21 was added to give the encoder
                    //extra capacity for reconciling v=0 vs v>0 distributions
                    //before per-path decorrelation existed. With one-row-per-path
                    //the joint-distribution conflict is gone, so 19 (4x less
                    //memory, proportionally less gradient scatter) suffices.
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

        //fully-fused MLP, 4 hidden x 64 ReLU, LINEAR output.
        //Target is L/reflSum directly (no transform). RelativeL2 + linear
        //gives an unbiased optimum at E[L/r]; the prior sqrt-target was
        //Jensen-biased low (converged to (E[sqrt(L/r)])^2), and log1p was
        //worse still. fp16 safety holds because safe_target caps L/r at
        //kTargetMax=1e4, well under fp16 max 6.5e4.
        {"network", {
            {"otype",             "FullyFusedMLP"},
            {"activation",        "ReLU"},
            {"output_activation", "None"},
            {"n_neurons",         kHiddenWidth},
            {"n_hidden_layers",   kHiddenLayers},
        }},
    };
}

//====================================
//RGB9E5 UNPACK
//====================================
//must match shaders/Compression_v8.hlsli
__device__ __forceinline__ float3 unpack_rgb9e5(uint32_t p) {
    const uint32_t rm = (p >>  0) & 0x1FFu;
    const uint32_t gm = (p >>  9) & 0x1FFu;
    const uint32_t bm = (p >> 18) & 0x1FFu;
    const int      e  = int((p >> 27) & 0x1Fu);
    const float scale = exp2f(float(e - 15 - 9));
    return make_float3(float(rm) * scale, float(gm) * scale, float(bm) * scale);
}

//NaN/Inf -> 0, negatives -> 0, caps magnitude to prevent Adam moment spikes
//With sqrt target transform, sqrt(kTargetMax) must stay within fp16-safe range
//for the L2 loss: (pred - sqrt_target)^2 is computed on-device. sqrt(1e4)=100,
//worst-case diff^2 = 1e4, comfortably under fp16 max (6.5e4).
constexpr float kTargetMax = 1.0e4f;
__device__ __forceinline__ float safe_target(float v) {
    if (!isfinite(v)) return 0.0f;
    return fminf(fmaxf(v, 0.0f), kTargetMax);
}

//====================================
//EMA UPDATE KERNEL
//====================================
//bias-corrected, W_hat_t = coef_new*W_t + coef_old*W_hat_{t-1}
//W_hat_0 = 0 so W_hat_1 = W_1 directly, cache usable within a few steps
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

//====================================
//BACKWARD-FILL KERNEL
//====================================
//one thread per path, walks vertex bucket in reverse, emits (features, target) rows
//output slot is atomic counter, naturally shuffles records across paths per warp
__global__ void fill_training_batch_kernel(
    const uint8_t* __restrict__ trainBuf,
    const float*   __restrict__ inferenceOut,
    uint32_t                    trainingPathCount,
    float*         __restrict__ outFeatures,
    float*         __restrict__ outTargets,
    uint32_t*      __restrict__ outCounter)
{
    const uint32_t pathId = blockIdx.x * blockDim.x + threadIdx.x;
    if (pathId >= trainingPathCount || pathId >= kMaxTrainingPaths) return;

    const uint32_t* meta = reinterpret_cast<const uint32_t*>(trainBuf + pathId * kPathMetaStride);
    const uint32_t numVertices  = meta[0];
    const uint32_t tailKind     = meta[1];
    const uint32_t inferenceSlot = meta[2];
    const uint32_t tailRadPk    = meta[3];
    if (numVertices == 0 || tailKind == kTailInvalid) return;

    const uint8_t* vertBase = trainBuf + kPathMetaTotalBytes +
                              pathId * kMaxVerticesPerPath * kTrainVertexStride;
    const uint32_t lastV = (numVertices <= kMaxVerticesPerPath) ? (numVertices - 1u)
                                                                : (kMaxVerticesPerPath - 1u);

    //resolve emit depth: single-depth uses kTrainingEmitDepthOnly directly,
    //decorrelated multi-depth picks one depth from kTrainingDepthMask via
    //pathId hash so each path emits at most one row -- killing intra-path
    //target correlation while preserving the multi-depth feature distribution.
    //sentinel 0xFFFFFFFFu means "no valid depth", path emits nothing.
    uint32_t chosenDepth = kTrainingEmitDepthOnly;
    if (kTrainingDecorrelatePaths) {
        const uint32_t mask  = kTrainingDepthMask;
        const uint32_t count = static_cast<uint32_t>(__popc(mask));
        if (count == 0u) {
            chosenDepth = 0xFFFFFFFFu;
        } else {
            uint32_t h = pathId * 0x9E3779B9u;
            h ^= h >> 16; h *= 0xC2B2AE35u;
            h ^= h >> 13; h *= 0x27D4EB2Fu;
            h ^= h >> 16;
            const uint32_t target = h % count;
            uint32_t cursor = mask;
            //clear the `target` lowest set bits, leaving the picked one as the new lowest
            for (uint32_t i = 0u; i < target; ++i) cursor &= cursor - 1u;
            //__ffs returns 1-indexed position of lowest set bit, cursor != 0 here
            chosenDepth = static_cast<uint32_t>(__ffs(static_cast<int>(cursor)) - 1);
        }
    }

    //seed backward recursion, tailRadPk semantics depend on tailKind
    //emitter/miss, direct radiance, cache, (alpha+beta) at cache-term vertex, RR tail=0
    //using wrong vertex reflectance biases by the ratio, visible as view-dependent instability
    float tail_r = 0.0f, tail_g = 0.0f, tail_b = 0.0f;
    if (tailKind == kTailEmitter || tailKind == kTailMiss) {
        float3 t = unpack_rgb9e5(tailRadPk);
        tail_r = t.x; tail_g = t.y; tail_b = t.z;
    } else if (tailKind == kTailCache && inferenceSlot != kInvalidInferenceSlot) {
        const float3 reflCT = unpack_rgb9e5(tailRadPk);
        const float* p = inferenceOut + inferenceSlot * kOutputDim;
        //MLP output is L/reflSum directly. Clamp negatives from the linear
        //output layer (upstream ReLU is non-negative; only the final linear
        //layer can emit small negative values).
        tail_r = safe_target(fmaxf(p[0], 0.0f) * reflCT.x);
        tail_g = safe_target(fmaxf(p[1], 0.0f) * reflCT.y);
        tail_b = safe_target(fmaxf(p[2], 0.0f) * reflCT.z);
    }

    //backward walk
    for (int32_t v = int32_t(lastV); v >= 0; --v) {
        const uint8_t* vb = vertBase + uint32_t(v) * kTrainVertexStride;
        const float*   raw = reinterpret_cast<const float*>(vb);
        const uint32_t L_neePk     = *reinterpret_cast<const uint32_t*>(vb + 64);
        const uint32_t betaLocalPk = *reinterpret_cast<const uint32_t*>(vb + 68);

        float3 lnee = unpack_rgb9e5(L_neePk);
        float3 beta = unpack_rgb9e5(betaLocalPk);

        //MC estimator, L_s[v] = L_nee + beta*tail
        const float ls_r = lnee.x + beta.x * tail_r;
        const float ls_g = lnee.y + beta.y * tail_g;
        const float ls_b = lnee.z + beta.z * tail_b;

        //target = L_s/(alpha+beta), LINEAR (no transform). RelativeL2 + linear
        //gives an unbiased optimum at E[L/r]; the prior sqrt transform converged
        //to (E[sqrt(L/r)])^2 which is strictly <= E[L/r] by Jensen, producing a
        //uniform darkening proportional to per-cell MC variance. Output is
        //fp16-safe because safe_target caps at kTargetMax=1e4 < fp16 max 6.5e4.
        //alpha @ raw[10..12], beta @ raw[13..15]. 1e-5 floor avoids div-by-zero
        //on black materials.
        const float rs_r = fmaxf(raw[10] + raw[13], 1e-5f);
        const float rs_g = fmaxf(raw[11] + raw[14], 1e-5f);
        const float rs_b = fmaxf(raw[12] + raw[15], 1e-5f);
        const float tgt_r = safe_target(ls_r / rs_r);
        const float tgt_g = safe_target(ls_g / rs_g);
        const float tgt_b = safe_target(ls_b / rs_b);

        //Per-path single-row emission: row at chosenDepth (resolved above)
        //enters the batch, all other vertices contribute only via backward
        //fill. Decorrelated multi-depth gives chosenDepth a uniform pick from
        //kTrainingDepthSet per pathId; otherwise chosenDepth = kTrainingEmitDepthOnly.
        const bool shouldEmit = kDebugConstantTraining
            ? true
            : (static_cast<uint32_t>(v) == chosenDepth);
        if (shouldEmit) {
            const uint32_t localCount = atomicAdd(outCounter, 1u);
            if (localCount < kTrainingRecordsPerFrame) {
                static_assert((kTrainingRecordsPerFrame & (kTrainingRecordsPerFrame - 1u)) == 0u,
                              "kTrainingRecordsPerFrame must be a power of two for the bijective scatter");
                const uint32_t outIdx =
                    (localCount * 2654435761u + 0x9E3779B1u) & (kTrainingRecordsPerFrame - 1u);
                float* f = outFeatures + outIdx * kRawInputDim;
                #pragma unroll
                for (uint32_t i = 0; i < kRawInputDim; ++i) f[i] = raw[i];
                float* t = outTargets + outIdx * kOutputDim;
                if (kDebugConstantTraining) {
                    t[0] = 1.0f;
                    t[1] = 0.5f;
                    t[2] = 0.25f;
                } else {
                    t[0] = tgt_r; t[1] = tgt_g; t[2] = tgt_b;
                }
            }
        }

        //pass actual radiance forward in path order
        tail_r = ls_r; tail_g = ls_g; tail_b = ls_b;
    }
}

//====================================
//IMPL
//====================================
struct Network::Impl {
    tcnn::TrainableModel model;

    //per-frame training scratch, tcnn wraps via GPUMatrix at step time
    float*    trainFeatures = nullptr;
    float*    trainTargets  = nullptr;
    uint32_t* trainCounter  = nullptr;

    //EMA weights, inference reads emaParams, training mutates raw params
    tcnn::network_precision_t* emaParams = nullptr;
    size_t                     nParams   = 0;
    float                      emaAlpha  = 0.99f;
    uint64_t                   emaStep   = 0;

    //last frame's valid vertex count, drives adaptive-tile feedback
    uint32_t                   lastValidVertices = 0;

    //async readback overlaps counter copy with frame work, 1-frame lag on batch sizing
    uint32_t*                  hostCounterReadback   = nullptr;
    cudaEvent_t                counterReadbackEvent  = nullptr;
    bool                       counterReadbackPending = false;

    //private stream, training overlaps next frame's raygen
    //inferenceDoneEvent, main stream -> auxStream before reading inferenceOut
    //trainDoneEvent, auxStream -> main stream before next inference reads emaParams
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

        const size_t fBytes = size_t(kTrainingRecordsPerFrame) * kRawInputDim * sizeof(float);
        const size_t tBytes = size_t(kTrainingRecordsPerFrame) * kOutputDim   * sizeof(float);
        if (cudaMalloc(&m_impl->trainFeatures, fBytes) != cudaSuccess) return false;
        if (cudaMalloc(&m_impl->trainTargets,  tBytes) != cudaSuccess) return false;
        if (cudaMalloc(&m_impl->trainCounter,  sizeof(uint32_t)) != cudaSuccess) return false;

        if (cudaMallocHost(&m_impl->hostCounterReadback, sizeof(uint32_t)) != cudaSuccess) return false;
        *m_impl->hostCounterReadback = 0u;
        if (cudaEventCreateWithFlags(&m_impl->counterReadbackEvent,
                                     cudaEventDisableTiming) != cudaSuccess) return false;
        m_impl->counterReadbackPending = false;

        //cudaStreamNonBlocking lets auxStream overlap without implicit sync
        if (cudaStreamCreateWithFlags(&m_impl->auxStream,
                                      cudaStreamNonBlocking) != cudaSuccess) return false;
        if (cudaEventCreateWithFlags(&m_impl->inferenceDoneEvent,
                                     cudaEventDisableTiming) != cudaSuccess) return false;
        if (cudaEventCreateWithFlags(&m_impl->trainDoneEvent,
                                     cudaEventDisableTiming) != cudaSuccess) return false;

        //EMA starts at zero, bias correction formula requires it
        //seeding from random weights dominates first ~300 steps with decaying init noise
        const size_t nParams = m_impl->model.network->n_params();
        m_impl->nParams = nParams;
        if (cudaMalloc(&m_impl->emaParams,
                       nParams * sizeof(tcnn::network_precision_t)) != cudaSuccess) return false;
        cudaMemset(m_impl->emaParams, 0,
                   nParams * sizeof(tcnn::network_precision_t));
        m_impl->emaStep = 0;
        m_impl->model.network->set_params(
            m_impl->model.trainer->params(),
            m_impl->emaParams,
            m_impl->model.trainer->param_gradients());

        //JIT fusion collapses encoder+MLP launches into 1, first call pays NVRTC compile
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
    //drain in-flight training before destroying events
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

bool Network::ReinitWeights() {
    if (!m_impl->ready) return false;
    try {
        //drain any in-flight training/inference on auxStream before rebuild
        WaitIdle();
        if (m_impl->counterReadbackPending) {
            cudaEventSynchronize(m_impl->counterReadbackEvent);
            m_impl->counterReadbackPending = false;
        }

        //tcnn::create_from_config reseeds internally, no manual seed plumbing required
        //keep CUDA buffers/events/stream untouched, only model + EMA get reset
        m_impl->model = tcnn::create_from_config(kRawInputDim, kOutputDim, BuildNetworkConfig());
        if (!m_impl->model.network || !m_impl->model.trainer) {
            m_impl->ready = false;
            return false;
        }

        const size_t nParams = m_impl->model.network->n_params();
        if (nParams != m_impl->nParams) {
            //config changed params count -- reallocate EMA buffer
            if (m_impl->emaParams) cudaFree(m_impl->emaParams);
            if (cudaMalloc(&m_impl->emaParams,
                           nParams * sizeof(tcnn::network_precision_t)) != cudaSuccess) {
                m_impl->emaParams = nullptr;
                m_impl->nParams   = 0;
                m_impl->ready     = false;
                return false;
            }
            m_impl->nParams = nParams;
        }
        cudaMemset(m_impl->emaParams, 0,
                   m_impl->nParams * sizeof(tcnn::network_precision_t));
        m_impl->emaStep = 0;
        m_impl->lastValidVertices = 0;

        m_impl->model.network->set_params(
            m_impl->model.trainer->params(),
            m_impl->emaParams,
            m_impl->model.trainer->param_gradients());
        m_impl->model.network->set_jit_fusion(true);
        return true;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "[NRC] ReinitWeights threw: %s\n", ex.what());
        m_impl->ready = false;
        return false;
    }
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
            "[NRC] Inference count %u not a multiple of %u, caller must pad.\n",
            count, kBatchGranularity);
        return;
    }
    cudaStream_t stream = static_cast<cudaStream_t>(streamPtr);

    //wait for prev frame's training to finish writing emaParams, first call no-op
    cudaStreamWaitEvent(stream, m_impl->trainDoneEvent, 0);

    tcnn::GPUMatrix<float> input (const_cast<float*>(inputDevPtr),  kRawInputDim, count);
    tcnn::GPUMatrix<float> output(                  outputDevPtr,   kOutputDim,   count);
    m_impl->model.network->inference(stream, input, output);

    //signal inferenceOut valid, class-1 backward-fill reads this
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
    //drain pending readback, host can safely reallocate after return
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
    //ignore caller's main stream, run on auxStream to overlap next frame's work
    (void)streamPtr;
    cudaStream_t stream = m_impl->auxStream;

    //wait for this frame's inference to finish writing inferenceOut
    //class-1 backward fill reads those outputs, would race without this barrier
    cudaStreamWaitEvent(stream, m_impl->inferenceDoneEvent, 0);

    //memset is optional since Adam reads only perBatch rows, kept as defensive
    const size_t fBytes = size_t(kTrainingRecordsPerFrame) * kRawInputDim * sizeof(float);
    const size_t tBytes = size_t(kTrainingRecordsPerFrame) * kOutputDim   * sizeof(float);
    cudaMemsetAsync(m_impl->trainFeatures, 0, fBytes, stream);
    cudaMemsetAsync(m_impl->trainTargets,  0, tBytes, stream);
    cudaMemsetAsync(m_impl->trainCounter,  0, sizeof(uint32_t), stream);

    //launch over all path slots, threads early-out on numVertices==0
    constexpr uint32_t kThreads = 256;
    const uint32_t blocks = (kMaxTrainingPaths + kThreads - 1u) / kThreads;
    fill_training_batch_kernel<<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(trainRecordsDevPtr),
        reinterpret_cast<const float*>  (inferenceOutDevPtr),
        kMaxTrainingPaths,
        m_impl->trainFeatures,
        m_impl->trainTargets,
        m_impl->trainCounter);

    //non-blocking consume of prev readback, lastValidVertices only feeds adaptive
    //tile feedback + batch sizing -- both already tolerate multi-frame lag, so a
    //pending event just means we reuse the previous cached value this frame.
    //Blocking here (old cudaEventSynchronize) made auxStream spikes turn into
    //main-thread stalls that showed up as random long frame drops.
    if (m_impl->counterReadbackPending) {
        if (cudaEventQuery(m_impl->counterReadbackEvent) == cudaSuccess) {
            uint32_t v = *m_impl->hostCounterReadback;
            if (v > kTrainingRecordsPerFrame) v = kTrainingRecordsPerFrame;
            m_impl->lastValidVertices = v;
            m_impl->counterReadbackPending = false;
        }
    }

    //only issue a new readback when the previous one has landed -- otherwise the
    //event would be re-recorded while still referenced and the prior value lost
    if (!m_impl->counterReadbackPending) {
        cudaMemcpyAsync(m_impl->hostCounterReadback, m_impl->trainCounter,
                        sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
        cudaEventRecord(m_impl->counterReadbackEvent, stream);
        m_impl->counterReadbackPending = true;
    }

    //size this frame's batches from prev frame's count, 1-frame lag acceptable
    const uint32_t validVertices = m_impl->lastValidVertices;

    //round down to tcnn granularity, drops <=255 rows
    const uint32_t perBatchRaw = validVertices / kTrainingBatchesPerFrame;
    const uint32_t perBatch    = (perBatchRaw / kBatchGranularity) * kBatchGranularity;
    if (perBatch == 0) {
        //still signal done so next inference doesn't wait on stale event
        cudaEventRecord(m_impl->trainDoneEvent, stream);
        return;
    }

    //SGD + bias-corrected EMA
    const uint32_t kEmaThreads = 256u;
    const uint32_t kEmaBlocks  = uint32_t((m_impl->nParams + kEmaThreads - 1u) / kEmaThreads);
    const float alpha = m_impl->emaAlpha;
    for (uint32_t b = 0; b < kTrainingBatchesPerFrame; ++b) {
        const float* fPtr = m_impl->trainFeatures + b * perBatch * kRawInputDim;
        const float* tPtr = m_impl->trainTargets  + b * perBatch * kOutputDim;
        TrainingStep((void*)stream, fPtr, tPtr, perBatch);

        //eta_t = 1-alpha^t, at t=1 W_hat_1 = W_1 directly
        ++m_impl->emaStep;
        const double t    = (double)m_impl->emaStep;
        const double da   = (double)alpha;
        const double etaT = 1.0 - pow(da, t);
        const double etaP = (t > 1.0) ? 1.0 - pow(da, t - 1.0) : 0.0;
        const float  coefNew = (float)((1.0 - da) / etaT);
        const float  coefOld = (float)(da * etaP / etaT);

        ema_update_kernel<<<kEmaBlocks, kEmaThreads, 0, stream>>>(
            m_impl->emaParams,
            m_impl->model.trainer->params(),
            coefNew,
            coefOld,
            m_impl->nParams);
    }

    //signal emaParams fully updated, next inference waits on this
    cudaEventRecord(m_impl->trainDoneEvent, stream);
}

//====================================
//CUDA HELPERS
//====================================
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

//====================================
//SMOKE TEST
//====================================
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

}
