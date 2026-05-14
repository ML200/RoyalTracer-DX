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
        //RelativeL2Luminance: the loss Müller et al. 2021 used for NRC (tcnn's
        //docs cite it as "Used in Neural Radiance Caching"). Scale-invariant --
        //error is normalized by predicted luminance^2 -- which is what lets the
        //LR run at 1e-2; a plain-L2 setup could not. HISTORY: an earlier
        //session rejected plain per-channel RelativeL2 for asymmetric-gradient
        //dark-scene overshoot. RelativeL2Luminance is the shared-luminance-
        //denominator variant (one scalar denom, no per-channel collapse) and
        //is being re-evaluated on the current config (4x128, log2=19, 1.5x
        //data, roughness-gated emit) -- WATCH dark/indirect regions for slow
        //upward brightness drift, that is the failure signature.
        {"loss", {{"otype", "RelativeL2Luminance"}}},
        {"optimizer", {
            {"otype",         "Adam"},
            //Müller-matched Adam: lr 1e-2 + standard betas + standard tcnn
            //epsilon/l2_reg. Replaces the prior hand-tuned set (lr 2e-3,
            //beta2 0.995, epsilon 1e-4, l2_reg 1e-5) which was tuned around
            //the plain-L2 loss; this set tracks the paper's recipe and pairs
            //with the scale-invariant RelativeL2Luminance loss above.
            {"learning_rate", 1.0e-2f},
            {"beta1",         0.9f},
            {"beta2",         0.999f},
            {"epsilon",       1e-8f},
            {"l2_reg",        1e-8f},
        }},
        {"encoding", {
            {"otype",  "Composite"},
            {"nested", tcnn::json::array({
                //Position encoding: TriangleWave (Müller 2021's original
                //choice), swapped back 2026-05-14 from a 16-level HashGrid.
                //The hash grid gave finer spatial detail, but: (1) its 16
                //scattered level lookups per query were L2-bandwidth-bound
                //(SMs stalling on memory -> "GPU idle"); (2) its ~16.7M learned
                //params were gradient-scattered + Adam-updated every training
                //step -- the dominant training cost; (3) the level-indexing
                //logic was register-heavy inside the JIT-fused kernel, capping
                //occupancy. TriangleWave is a fixed (non-learned) basis: pure
                //ALU, no lookups, no encoder backward, no encoder params in
                //the optimizer, tiny register footprint. Trade is lower
                //spatial resolution in the cache -- acceptable here because
                //the cache only fires at x3+ (deep bounces) where the radiance
                //it represents is low-frequency indirect GI. n_frequencies=12
                //matches tcnn's documented NRC-replication config. To revert,
                //restore the Grid/Hash block (log2_hashmap_size 19,
                //base_resolution 16, per_level_scale 1.38, 16 levels x 2).
                tcnn::json{
                    {"n_dims_to_encode", 3u},
                    {"otype",            "TriangleWave"},
                    {"n_frequencies",    12u},
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
                    {"n_dims_to_encode", 7u},
                    {"otype",            "Identity"},
                },
            })},
        }},

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

//NaN/Inf -> 0, negatives -> 0, caps magnitude to prevent Adam moment spikes.
//Under L2 the gradient is 2(pred-target), so worst-case per-sample gradient
//magnitude is exactly 2 * kTargetMax = 20. Aggressive cap chosen alongside
//log2_hashmap_size=21 (more spatial entries, fewer samples per cell) -- the
//tighter cap compensates for the lower per-cell density by guaranteeing no
//single rare bright NEE sample can drift the cell prediction far. Will dim
//physically bright surfaces (strong direct sun, specular peaks); accept
//that trade for full stability in dark scenes.
constexpr float kTargetMax = 1.0e1f;
__device__ __forceinline__ float safe_target(float v) {
    if (!isfinite(v)) return 0.0f;
    return fminf(fmaxf(v, 0.0f), kTargetMax);
}

//====================================
//SPATIAL SORT KERNELS FOR INFERENCE
//====================================
//bucket grid is 32^3 = 32768 cells over the normalized [0,1]^3 scene
//domain. Granularity tuned so the prefix scan fits in a single block
//(1024 threads x 32 buckets per thread = 32768) and the per bucket
//queue stays small enough to keep cache lines hot inside the encoder.
//A finer grid (64^3) buys minor extra coherence at fine HashGrid levels
//but quadruples the scan cost and the bucket scratch.
//
//Master toggle. The sort intends to cluster queries spatially so HashGrid
//cache lines stay hot, but for very large inference batches the count and
//scatter kernels are dominated by global atomic contention into the 32k
//bucket counters and the cost can outweigh the encoder locality win,
//particularly with log2_hashmap_size=19 where the encoder fits in a few MB
//of L2 anyway. Set false to skip the entire sort phase and dispatch
//inference directly on the renderer's input buffer.
constexpr bool kEnableSpatialSort = false;

constexpr uint32_t kSortBucketsPerAxis = 32u;
constexpr uint32_t kSortBuckets        = kSortBucketsPerAxis * kSortBucketsPerAxis * kSortBucketsPerAxis;

__device__ __forceinline__ uint32_t nrc_bucket_from_features(const float* feats)
{
    //features[0..2] are NrcNormalizePosition output saturated to [0,1].
    //Quantise per axis, clamp the 1.0 endpoint into the last cell.
    const float fx = fminf(fmaxf(feats[0], 0.0f), 1.0f);
    const float fy = fminf(fmaxf(feats[1], 0.0f), 1.0f);
    const float fz = fminf(fmaxf(feats[2], 0.0f), 1.0f);
    uint32_t bx = (uint32_t)(fx * (float)kSortBucketsPerAxis);
    uint32_t by = (uint32_t)(fy * (float)kSortBucketsPerAxis);
    uint32_t bz = (uint32_t)(fz * (float)kSortBucketsPerAxis);
    if (bx >= kSortBucketsPerAxis) bx = kSortBucketsPerAxis - 1u;
    if (by >= kSortBucketsPerAxis) by = kSortBucketsPerAxis - 1u;
    if (bz >= kSortBucketsPerAxis) bz = kSortBucketsPerAxis - 1u;
    return bx + by * kSortBucketsPerAxis + bz * kSortBucketsPerAxis * kSortBucketsPerAxis;
}

__global__ void nrc_sort_count_kernel(
    const float* __restrict__ in,
    uint32_t                  count,
    uint32_t*    __restrict__ bucketCounts)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    const uint32_t bucket = nrc_bucket_from_features(in + tid * kRawInputDim);
    atomicAdd(&bucketCounts[bucket], 1u);
}

//in block exclusive prefix scan over kSortBuckets entries. One block,
//1024 threads, 32 entries per thread. Hierarchy is per thread sequential
//scan, then warp shfl scan of thread sums, then warp 0 scans the warp
//sums via a second shfl pass.
__global__ void nrc_sort_scan_kernel(uint32_t* __restrict__ values)
{
    constexpr uint32_t kPerThread = 32u;
    constexpr uint32_t kThreads   = 1024u;
    static_assert(kPerThread * kThreads == kSortBuckets,
                  "scan kernel sized for a single block over kSortBuckets");

    __shared__ uint32_t warpSums[32];

    const uint32_t tid    = threadIdx.x;
    const uint32_t lane   = tid & 31u;
    const uint32_t warpId = tid >> 5;

    uint32_t local[kPerThread];
    uint32_t threadSum = 0u;
    #pragma unroll
    for (uint32_t i = 0; i < kPerThread; ++i) {
        const uint32_t v = values[tid * kPerThread + i];
        local[i] = threadSum;
        threadSum += v;
    }

    //inclusive in warp scan of threadSum
    uint32_t inc = threadSum;
    for (uint32_t d = 1u; d < 32u; d <<= 1) {
        const uint32_t up = __shfl_up_sync(0xFFFFFFFFu, inc, d);
        if (lane >= d) inc += up;
    }
    const uint32_t exclusiveLane = inc - threadSum;

    if (lane == 31u) warpSums[warpId] = inc;
    __syncthreads();

    //warp 0 scans the 32 warp totals
    if (warpId == 0u) {
        uint32_t ws  = warpSums[lane];
        uint32_t inc2 = ws;
        for (uint32_t d = 1u; d < 32u; d <<= 1) {
            const uint32_t up = __shfl_up_sync(0xFFFFFFFFu, inc2, d);
            if (lane >= d) inc2 += up;
        }
        warpSums[lane] = inc2 - ws;
    }
    __syncthreads();

    const uint32_t threadStart = warpSums[warpId] + exclusiveLane;
    #pragma unroll
    for (uint32_t i = 0; i < kPerThread; ++i) {
        values[tid * kPerThread + i] = threadStart + local[i];
    }
}

__global__ void nrc_sort_scatter_kernel(
    const float* __restrict__ in,
    uint32_t                  count,
    uint32_t*    __restrict__ bucketCursors,
    float*       __restrict__ out,
    uint32_t*    __restrict__ perm)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    const float* feats = in + tid * kRawInputDim;
    const uint32_t bucket = nrc_bucket_from_features(feats);
    const uint32_t newSlot = atomicAdd(&bucketCursors[bucket], 1u);
    //defensive: a correct scan keeps newSlot < count, but if the prefix
    //sum is ever off by a bucket the scatter would OOB into adjacent
    //allocations and trigger an MMU fault during the kernel which TDRs.
    //Drop the write instead.
    if (newSlot >= count) return;

    float* dst = out + newSlot * kRawInputDim;
    #pragma unroll
    for (uint32_t i = 0; i < kRawInputDim; ++i) dst[i] = feats[i];
    perm[newSlot] = tid;
}

__global__ void nrc_unsort_kernel(
    const float*    __restrict__ sortedOut,
    const uint32_t* __restrict__ perm,
    uint32_t                     count,
    float*          __restrict__ out)
{
    const uint32_t newSlot = blockIdx.x * blockDim.x + threadIdx.x;
    if (newSlot >= count) return;
    const uint32_t oldSlot = perm[newSlot];
    //defensive: scatter only writes perm[newSlot] = tid where tid < count,
    //so oldSlot < count by construction. Guard against an unwritten perm
    //entry (rare scatter early out above) so the back scatter never
    //touches memory past the renderer's outputDevPtr allocation.
    if (oldSlot >= count) return;
    const float*   src     = sortedOut + newSlot * kOutputDim;
    float*         dst     = out       + oldSlot * kOutputDim;
    #pragma unroll
    for (uint32_t i = 0; i < kOutputDim; ++i) dst[i] = src[i];
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
//COLLAPSE DETECTOR REDUCTION
//====================================
//sum |out[r]|+|out[g]|+|out[b]| across the inference batch into a single
//float. The host divides by (count*3) for a mean magnitude per channel
//that the renderer compares against a collapse threshold.
//
//Hierarchical reduction: warp-shuffle inside each warp, shared mem across
//warps inside a block, then ONE atomicAdd per block into the global sum.
//With 256 threads/block on a multi-million query batch this drops global
//atomic ops from ~N (one per thread) to ~N/256 (one per block) and
//eliminates the L2 atomic contention that was serialising the whole
//kernel. The previous version did `atomicAdd(sumOut, ...)` per thread,
//which on a ~5M batch took multiple ms of essentially sequential atomic
//queue traffic and starved overlapping streams of GPU resources -- the
//visible "both inference and training streams sit idle" gap in the Nsight
//timeline came directly from this kernel saturating the atomic units.
//Block size MUST be 256 to match the host launch and the warpSums size.
__global__ void nrc_infout_sum_kernel(
    const float* __restrict__ out,
    uint32_t                  count,
    float*       __restrict__ sumOut)
{
    __shared__ float warpSums[8]; //256 threads / 32 lanes per warp = 8 warps
    const uint32_t tid  = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5;

    float local = 0.0f;
    if (tid < count) {
        const float r = out[tid * kOutputDim + 0];
        const float g = out[tid * kOutputDim + 1];
        const float b = out[tid * kOutputDim + 2];
        local = fabsf(r) + fabsf(g) + fabsf(b);
    }

    //warp reduction via shuffle, no shared mem traffic for the inner step
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1) {
        local += __shfl_down_sync(0xFFFFFFFFu, local, offset);
    }

    //one shared mem write per warp, lane 0 of each warp publishes the warp sum
    if (lane == 0u) warpSums[warp] = local;
    __syncthreads();

    //first warp reduces the 8 warp sums and emits a single atomic per block
    if (warp == 0u) {
        local = (lane < 8u) ? warpSums[lane] : 0.0f;
        for (uint32_t offset = 4u; offset > 0u; offset >>= 1) {
            local += __shfl_down_sync(0xFFFFFFFFu, local, offset);
        }
        if (lane == 0u) atomicAdd(sumOut, local);
    }
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
    uint32_t                    inferenceOutCapacity,
    uint32_t                    targetRecords,
    float*         __restrict__ outFeatures,
    float*         __restrict__ outTargets,
    uint32_t*      __restrict__ outCounter)
{
    const uint32_t pathId = blockIdx.x * blockDim.x + threadIdx.x;
    if (pathId >= trainingPathCount || pathId >= kMaxTrainingPaths) return;

    const uint32_t* meta = reinterpret_cast<const uint32_t*>(trainBuf + pathId * kPathMetaStride);
    const uint32_t metaWord0    = meta[0];
    const uint32_t numVertices  = UnpackNumVertices(metaWord0);
    const uint32_t emitMask     = UnpackEmitMask(metaWord0);
    const uint32_t tailKind     = meta[1];
    const uint32_t inferenceSlot = meta[2];
    const uint32_t tailRadPk    = meta[3];
    if (numVertices == 0 || tailKind == kTailInvalid) return;

    const uint8_t* vertBase = trainBuf + kPathMetaTotalBytes +
                              pathId * kMaxVerticesPerPath * kTrainVertexStride;
    const uint32_t lastV = (numVertices <= kMaxVerticesPerPath) ? (numVertices - 1u)
                                                                : (kMaxVerticesPerPath - 1u);

    //resolve which vertices emit a training row. MULTI-ROW: every permitted
    //vertex emits its own (features, target) row -- a path is no longer
    //collapsed to a single random depth.
    //  kTrainingDecorrelatePaths = true  -> emit at every vertex in
    //      (kTrainingDepthMask & emitMask): the depth mask AND the roughness
    //      emit gate both have to permit it.
    //  kTrainingDecorrelatePaths = false -> emit at every stored vertex that
    //      passed the roughness emit gate (depth mask ignored).
    //emitMask (the NRC_TRAIN_ROUGHNESS_MIN gate) always applies in both modes
    //-- it is a separate concern from depth selection and is load-bearing
    //(keeps glossy/specular surfaces out of the cache).
    //
    //WARNING: multi-row reintroduces intra-path target correlation -- target_v
    //and target_{v+1} share the L_s[v+1] term. The row budget
    //(kFixedTrainingRecords) was raised so the extra rows are additive rather
    //than displacing path diversity; watch indirect regions for chromatic
    //darkening, the historical failure mode of multi-row emission.
    const uint32_t storedMask = (numVertices >= 32u) ? 0xFFFFFFFFu
                                                     : ((1u << numVertices) - 1u);
    const uint32_t emitDepthMask = kTrainingDecorrelatePaths
        ? (kTrainingDepthMask & emitMask & storedMask)
        : (emitMask & storedMask);

    //early-out when no vertex emits (debug-mode emits at every vertex
    //regardless, so keep walking the chain in that case).
    if (!kDebugConstantTraining && emitDepthMask == 0u) return;

    //seed backward recursion, tailRadPk semantics depend on tailKind
    //emitter/miss, direct radiance, cache, (alpha+beta) at cache-term vertex, RR tail=0
    //using wrong vertex reflectance biases by the ratio, visible as view-dependent instability
    float tail_r = 0.0f, tail_g = 0.0f, tail_b = 0.0f;
    if (tailKind == kTailEmitter || tailKind == kTailMiss) {
        float3 t = unpack_rgb9e5(tailRadPk);
        tail_r = t.x; tail_g = t.y; tail_b = t.z;
    } else if (tailKind == kTailCache &&
               inferenceSlot != kInvalidInferenceSlot &&
               inferenceSlot < inferenceOutCapacity) {
        //defensive: a torn meta read from a concurrent next frame frame_begin
        //memset (race window if frame_begin fires before this auxStream fill
        //kernel drains) can land tailKind==kTailCache alongside a garbage
        //inferenceSlot. Without the capacity clamp, inferenceOut+slot*kOutputDim
        //walks past the renderer's NRC_InferenceOut allocation and the deref
        //below MMU faults inside the kernel which TDRs. Treat an out of range
        //slot as if the cache contribution were zero, the path still emits
        //its NEE rows for the depths up to the cache term vertex.
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
        //offsets shift with kRawInputDim: raw is kRawInputDim*4 bytes,
        //then L_neePk (4) then betaLocalPk (4)
        constexpr uint32_t kLNeeOffset  = kRawInputDim * sizeof(float);
        constexpr uint32_t kBetaOffset  = kLNeeOffset + sizeof(uint32_t);
        const uint32_t L_neePk     = *reinterpret_cast<const uint32_t*>(vb + kLNeeOffset);
        const uint32_t betaLocalPk = *reinterpret_cast<const uint32_t*>(vb + kBetaOffset);

        float3 lnee = unpack_rgb9e5(L_neePk);
        float3 beta = unpack_rgb9e5(betaLocalPk);

        //MC estimator, L_s[v] = L_nee + beta*tail -- EXCEPT for sub-threshold
        //(glossy/specular) vertices, whose emitMask bit is clear. Those
        //contribute 0 to the chain: their outgoing radiance is a view-
        //dependent reflection the position-dominant cache cannot represent,
        //and beta~=1 / L_nee~=0 on a glossy surface means they otherwise pass
        //the reflected tail straight through into the target of the eligible
        //diffuse vertex upstream -- that is the "cache still learns
        //reflections" artifact (the emit gate only stops the glossy vertex
        //being a training ROW, not it leaking through the transport chain).
        //Truncating here costs the GI bounce *off* glossy surfaces (slight
        //darkening near shiny objects) but 0 is bounded error where the leak
        //was not. Emitted vertices are always emitMask-eligible (emitDepthMask
        //is a subset of emitMask) so an emitted target is never zeroed by this
        //-- only transport-only vertices are.
        const bool vEligible = (emitMask & (1u << static_cast<uint32_t>(v))) != 0u;
        const float ls_r = vEligible ? (lnee.x + beta.x * tail_r) : 0.0f;
        const float ls_g = vEligible ? (lnee.y + beta.y * tail_g) : 0.0f;
        const float ls_b = vEligible ? (lnee.z + beta.z * tail_b) : 0.0f;

        //target = L_s/(alpha+beta), LINEAR (no transform). L2 + linear gives an
        //unbiased optimum at E[L/r]; prior sqrt transform was Jensen-biased
        //(E[sqrt(L/r)])^2 <= E[L/r]. RelativeL2 was tested -- mathematically
        //unbiased but tcnn's grad = 2(pred-target)/(pred^2+0.01) is asymmetric:
        //once pred drifts up, the down-pull from dark samples collapses while
        //the up-pull from rare bright NEE samples stays strong, producing a
        //stable overshoot above E[L/r] in dark scenes with sparse bright
        //sources. RelativeL1 was also tested -- median-biased, crushed night
        //scenes to zero. L2 has symmetric grad = 2(pred-target), gradient
        //spikes bounded by kTargetMax and absorbed by beta2=0.999.
        //alpha @ raw[10..12], beta @ raw[13..15]. 1e-5 floor avoids div-by-zero
        //on black materials.
        const float rs_r = fmaxf(raw[10] + raw[13], 1e-5f);
        const float rs_g = fmaxf(raw[11] + raw[14], 1e-5f);
        const float rs_b = fmaxf(raw[12] + raw[15], 1e-5f);
        const float tgt_r = safe_target(ls_r / rs_r);
        const float tgt_g = safe_target(ls_g / rs_g);
        const float tgt_b = safe_target(ls_b / rs_b);

        //Multi-row emission: a vertex emits iff its bit is set in
        //emitDepthMask (resolved above from the depth mask + roughness gate).
        //Every vertex still participates in the backward chain regardless of
        //whether it emits a row.
        const bool shouldEmit = kDebugConstantTraining
            ? true
            : (((emitDepthMask >> static_cast<uint32_t>(v)) & 1u) != 0u);
        if (shouldEmit) {
            const uint32_t localCount = atomicAdd(outCounter, 1u);
            //cap at runtime target (renderer scales with resolution) AND at the
            //compile-time buffer ceiling so we never OOB outFeatures/outTargets
            const uint32_t cap = (targetRecords < kTrainingRecordsPerFrame)
                ? targetRecords : kTrainingRecordsPerFrame;
            if (localCount < cap) {
                //sequential output index. The prior bijective scatter
                //(localCount * 2654435761u & mask) was a hash shuffle to
                //decorrelate paths inside a batch, but kTrainingDecorrelatePaths
                //already enforces one row per path and the four SGD steps
                //slice the buffer by index so adjacent localCounts land in
                //adjacent batches anyway. Sequential writes let warp lanes
                //hit consecutive 68 byte rows, recovering the 32x write
                //transaction overhead the scatter was costing in VRAM bound
                //training. tcnn shuffles internally per step.
                const uint32_t outIdx = localCount;
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

    //EMA weights, inference reads emaParams, training mutates raw params.
    //emaAlpha is a fixed smoothing factor. An adaptive brightness-keyed
    //emaAlpha was tried and reverted -- a constant 0.97 was preferred.
    tcnn::network_precision_t* emaParams = nullptr;
    size_t                     nParams   = 0;
    float                      emaAlpha  = 0.97f;
    uint64_t                   emaStep   = 0;

    //last frame's valid vertex count, drives adaptive-tile feedback
    uint32_t                   lastValidVertices = 0;

    //inference query spatial sort scratch. The HashGrid encoder is L2 bound
    //because each query does 16 levels x 8 corner reads scattered across
    //the table. Sorting queries by 3D position before inference clusters
    //spatially close queries into adjacent batch slots, so warps hit
    //overlapping cache lines instead of random ones, lifting the encoder
    //out of L2 bandwidth. sortedIn/sortedOut hold the reordered batch,
    //perm[new_slot] = old_slot lets us unsort outputs back to the slot
    //numbers stored in PendingGI / gScratchPing so consumers don't change.
    //bucketCounts doubles as start offsets after the in block prefix scan,
    //bucketCursors is a per launch copy of the starts that the scatter
    //kernel atomicAdds into.
    float*                     sortedIn          = nullptr;
    float*                     sortedOut         = nullptr;
    uint32_t*                  sortPerm          = nullptr;
    uint32_t*                  sortBucketCounts  = nullptr;
    uint32_t*                  sortBucketCursors = nullptr;
    uint32_t                   sortCapacity      = 0;

    //defined out of line below, after the kSortBuckets constant is in scope
    bool EnsureSortScratch(uint32_t capacity);

    //async readback overlaps counter copy with frame work, 1-frame lag on batch sizing
    uint32_t*                  hostCounterReadback   = nullptr;
    cudaEvent_t                counterReadbackEvent  = nullptr;
    bool                       counterReadbackPending = false;

    //inference counter readback, sized in renderer to scope raygen's
    //NrcAppendInference cap and the inference dispatch count for the next
    //frame. Same async pattern as the training counter so the host never
    //blocks on the GPU. lastInferenceCount stays 0 until the first readback
    //lands, the renderer falls back to the static 2*W*H buffer cap meanwhile.
    uint32_t*                  hostInferenceCount        = nullptr;
    cudaEvent_t                inferenceCountEvent       = nullptr;
    bool                       inferenceCountPending     = false;
    uint32_t                   lastInferenceCount        = 0;

    //collapse detector. ScheduleInferenceOutSumReadback launches the reduction
    //kernel on the same stream as inference and copies the resulting sum back
    //to host. infOutSumCountInFlight remembers the count used by the pending
    //readback so LastInferenceOutMagnitudeMean can normalize correctly when
    //the readback lands. Renderer polls the mean and triggers ReinitWeights
    //when it stays near zero across many frames.
    float*                     devInfOutSum              = nullptr;
    float*                     hostInfOutSum             = nullptr;
    cudaEvent_t                infOutSumEvent            = nullptr;
    bool                       infOutSumPending          = false;
    uint32_t                   infOutSumCountInFlight    = 0;
    float                      lastInferenceOutSum       = 0.0f;
    uint32_t                   lastInferenceOutCount     = 0;

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

        if (cudaMallocHost(&m_impl->hostInferenceCount, sizeof(uint32_t)) != cudaSuccess) return false;
        *m_impl->hostInferenceCount = 0u;
        if (cudaEventCreateWithFlags(&m_impl->inferenceCountEvent,
                                     cudaEventDisableTiming) != cudaSuccess) return false;
        m_impl->inferenceCountPending = false;
        m_impl->lastInferenceCount    = 0u;

        if (cudaMalloc(&m_impl->devInfOutSum, sizeof(float)) != cudaSuccess) return false;
        cudaMemset(m_impl->devInfOutSum, 0, sizeof(float));
        if (cudaMallocHost(&m_impl->hostInfOutSum, sizeof(float)) != cudaSuccess) return false;
        *m_impl->hostInfOutSum = 0.0f;
        if (cudaEventCreateWithFlags(&m_impl->infOutSumEvent,
                                     cudaEventDisableTiming) != cudaSuccess) return false;
        m_impl->infOutSumPending       = false;
        m_impl->infOutSumCountInFlight = 0u;
        m_impl->lastInferenceOutSum    = 0.0f;
        m_impl->lastInferenceOutCount  = 0u;

        //cudaStreamNonBlocking lets auxStream overlap without implicit sync
        //against the legacy/main stream. Default priority -- the high
        //priority variant (cudaStreamCreateWithPriority + greatestPrio)
        //was tried as an overlap hint for the WDDM/HWS scheduler but is
        //correlated with rare multi second TDR style freezes on some
        //driver versions where compute preemption against a non
        //preemptible FullyFusedMLP encoder enters a retry loop.
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
    if (m_impl->sortedIn)          { cudaFree(m_impl->sortedIn);          m_impl->sortedIn          = nullptr; }
    if (m_impl->sortedOut)         { cudaFree(m_impl->sortedOut);         m_impl->sortedOut         = nullptr; }
    if (m_impl->sortPerm)          { cudaFree(m_impl->sortPerm);          m_impl->sortPerm          = nullptr; }
    if (m_impl->sortBucketCounts)  { cudaFree(m_impl->sortBucketCounts);  m_impl->sortBucketCounts  = nullptr; }
    if (m_impl->sortBucketCursors) { cudaFree(m_impl->sortBucketCursors); m_impl->sortBucketCursors = nullptr; }
    m_impl->sortCapacity = 0;
    if (m_impl->hostCounterReadback) {
        cudaFreeHost(m_impl->hostCounterReadback);
        m_impl->hostCounterReadback = nullptr;
    }
    if (m_impl->counterReadbackEvent) {
        cudaEventDestroy(m_impl->counterReadbackEvent);
        m_impl->counterReadbackEvent = nullptr;
    }
    m_impl->counterReadbackPending = false;
    if (m_impl->hostInferenceCount) {
        cudaFreeHost(m_impl->hostInferenceCount);
        m_impl->hostInferenceCount = nullptr;
    }
    if (m_impl->inferenceCountEvent) {
        cudaEventDestroy(m_impl->inferenceCountEvent);
        m_impl->inferenceCountEvent = nullptr;
    }
    m_impl->inferenceCountPending = false;
    m_impl->lastInferenceCount    = 0u;
    if (m_impl->devInfOutSum) {
        cudaFree(m_impl->devInfOutSum);
        m_impl->devInfOutSum = nullptr;
    }
    if (m_impl->hostInfOutSum) {
        cudaFreeHost(m_impl->hostInfOutSum);
        m_impl->hostInfOutSum = nullptr;
    }
    if (m_impl->infOutSumEvent) {
        cudaEventDestroy(m_impl->infOutSumEvent);
        m_impl->infOutSumEvent = nullptr;
    }
    m_impl->infOutSumPending       = false;
    m_impl->infOutSumCountInFlight = 0u;
    m_impl->lastInferenceOutSum    = 0.0f;
    m_impl->lastInferenceOutCount  = 0u;
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
        if (m_impl->inferenceCountPending) {
            cudaEventSynchronize(m_impl->inferenceCountEvent);
            m_impl->inferenceCountPending = false;
        }
        m_impl->lastInferenceCount = 0u;
        if (m_impl->infOutSumPending) {
            cudaEventSynchronize(m_impl->infOutSumEvent);
            m_impl->infOutSumPending = false;
        }
        m_impl->infOutSumCountInFlight = 0u;
        m_impl->lastInferenceOutSum    = 0.0f;
        m_impl->lastInferenceOutCount  = 0u;
        if (m_impl->devInfOutSum) cudaMemset(m_impl->devInfOutSum, 0, sizeof(float));

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

//Lazy allocate the spatial sort scratch when the first inference call lands
//or when the renderer grows the inference capacity (window resize). Sized
//to the requested capacity, so subsequent calls within the same resolution
//hit the early out. Returns false on cudaMalloc failure -- the caller can
//then fall back to the unsorted path.
bool Network::Impl::EnsureSortScratch(uint32_t capacity)
{
    if (capacity <= sortCapacity && sortedIn) return true;

    if (sortedIn)          { cudaFree(sortedIn);          sortedIn = nullptr; }
    if (sortedOut)         { cudaFree(sortedOut);         sortedOut = nullptr; }
    if (sortPerm)          { cudaFree(sortPerm);          sortPerm = nullptr; }
    if (sortBucketCounts)  { cudaFree(sortBucketCounts);  sortBucketCounts = nullptr; }
    if (sortBucketCursors) { cudaFree(sortBucketCursors); sortBucketCursors = nullptr; }
    sortCapacity = 0;

    const size_t inBytes     = size_t(capacity) * kRawInputDim * sizeof(float);
    const size_t outBytes    = size_t(capacity) * kOutputDim   * sizeof(float);
    const size_t permBytes   = size_t(capacity) * sizeof(uint32_t);
    const size_t bucketBytes = size_t(kSortBuckets) * sizeof(uint32_t);

    if (cudaMalloc(&sortedIn,          inBytes)     != cudaSuccess) return false;
    if (cudaMalloc(&sortedOut,         outBytes)    != cudaSuccess) return false;
    if (cudaMalloc(&sortPerm,          permBytes)   != cudaSuccess) return false;
    if (cudaMalloc(&sortBucketCounts,  bucketBytes) != cudaSuccess) return false;
    if (cudaMalloc(&sortBucketCursors, bucketBytes) != cudaSuccess) return false;
    sortCapacity = capacity;
    return true;
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

    //Master toggle bypass: dispatch tcnn directly on the renderer's input
    //buffer. The 4 sort/unsort helper kernels and their scratch buffers are
    //skipped entirely. Used to A/B test whether the sort's HashGrid locality
    //win exceeds its global atomic contention cost on this scene size.
    if (!kEnableSpatialSort) {
        tcnn::GPUMatrix<float> input (const_cast<float*>(inputDevPtr),  kRawInputDim, count);
        tcnn::GPUMatrix<float> output(                  outputDevPtr,   kOutputDim,   count);
        m_impl->model.network->inference(stream, input, output);
        cudaEventRecord(m_impl->inferenceDoneEvent, stream);
        return;
    }

    const bool sortOK = m_impl->EnsureSortScratch(count);
    if (!sortOK) {
        //fallback: original unsorted path. Keeps the renderer alive if the
        //sort scratch alloc ever fails (low memory after a resize).
        tcnn::GPUMatrix<float> input (const_cast<float*>(inputDevPtr),  kRawInputDim, count);
        tcnn::GPUMatrix<float> output(                  outputDevPtr,   kOutputDim,   count);
        m_impl->model.network->inference(stream, input, output);
        cudaEventRecord(m_impl->inferenceDoneEvent, stream);
        return;
    }

    //----------------------------------------
    //Spatial sort phase. Cluster queries by 32^3 position bucket so warps
    //hit overlapping HashGrid cache lines instead of random ones across
    //the 16 MB encoder table. Three steps: count per bucket, in block
    //exclusive scan to derive starts, scatter into the sorted batch.
    //----------------------------------------
    cudaMemsetAsync(m_impl->sortBucketCounts, 0,
                    size_t(kSortBuckets) * sizeof(uint32_t), stream);

    constexpr uint32_t kSortThreads = 256u;
    const uint32_t kSortBlocks = (count + kSortThreads - 1u) / kSortThreads;
    nrc_sort_count_kernel<<<kSortBlocks, kSortThreads, 0, stream>>>(
        inputDevPtr, count, m_impl->sortBucketCounts);

    //single block scan over kSortBuckets entries, in place exclusive sum
    nrc_sort_scan_kernel<<<1, 1024, 0, stream>>>(m_impl->sortBucketCounts);

    //cursors start at the bucket starts, scatter atomicAdds into them
    cudaMemcpyAsync(m_impl->sortBucketCursors, m_impl->sortBucketCounts,
                    size_t(kSortBuckets) * sizeof(uint32_t),
                    cudaMemcpyDeviceToDevice, stream);

    nrc_sort_scatter_kernel<<<kSortBlocks, kSortThreads, 0, stream>>>(
        inputDevPtr, count, m_impl->sortBucketCursors,
        m_impl->sortedIn, m_impl->sortPerm);

    //----------------------------------------
    //Inference on the reordered batch. tcnn writes into sortedOut, the
    //unsort kernel scatters back into outputDevPtr at the original slot
    //numbers so PendingGI / gScratchPing references stay valid.
    //----------------------------------------
    tcnn::GPUMatrix<float> input (m_impl->sortedIn,  kRawInputDim, count);
    tcnn::GPUMatrix<float> output(m_impl->sortedOut, kOutputDim,   count);
    m_impl->model.network->inference(stream, input, output);

    constexpr uint32_t kUnsortThreads = 256u;
    const uint32_t kUnsortBlocks = (count + kUnsortThreads - 1u) / kUnsortThreads;
    nrc_unsort_kernel<<<kUnsortBlocks, kUnsortThreads, 0, stream>>>(
        m_impl->sortedOut, m_impl->sortPerm, count, outputDevPtr);

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

uint32_t Network::LastInferenceCount() const {
    return m_impl ? m_impl->lastInferenceCount : 0u;
}

void Network::ScheduleInferenceCounterReadback(void* streamPtr, const void* devCounterPtr) {
    if (!m_impl || !m_impl->ready || !devCounterPtr) return;
    cudaStream_t stream = static_cast<cudaStream_t>(streamPtr);

    //consume the previous readback if it has landed. Non blocking, leaves the
    //stale value in lastInferenceCount when still pending so the renderer just
    //reuses the prior cached cap for one more frame.
    if (m_impl->inferenceCountPending) {
        if (cudaEventQuery(m_impl->inferenceCountEvent) == cudaSuccess) {
            m_impl->lastInferenceCount    = *m_impl->hostInferenceCount;
            m_impl->inferenceCountPending = false;
        }
    }

    //only re-record while the prior event has been retired, otherwise the
    //recorded event would alias and the prior value would be lost
    if (!m_impl->inferenceCountPending) {
        cudaMemcpyAsync(m_impl->hostInferenceCount, devCounterPtr,
                        sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
        cudaEventRecord(m_impl->inferenceCountEvent, stream);
        m_impl->inferenceCountPending = true;
    }
}

void Network::ScheduleInferenceOutSumReadback(
    void*        streamPtr,
    const float* outputDevPtr,
    uint32_t     count)
{
    if (!m_impl || !m_impl->ready || !outputDevPtr || count == 0u) return;
    cudaStream_t stream = static_cast<cudaStream_t>(streamPtr);

    //consume the previous readback if it has landed. Same non blocking
    //pattern as ScheduleInferenceCounterReadback. lastInferenceOutSum stays
    //at the prior value until the new event retires, so the renderer's
    //collapse window keeps observing fresh data each frame.
    if (m_impl->infOutSumPending) {
        if (cudaEventQuery(m_impl->infOutSumEvent) == cudaSuccess) {
            m_impl->lastInferenceOutSum   = *m_impl->hostInfOutSum;
            m_impl->lastInferenceOutCount = m_impl->infOutSumCountInFlight;
            m_impl->infOutSumPending      = false;
        }
    }

    if (m_impl->infOutSumPending) return;

    //zero device accumulator before each launch; atomicAdd reduces into it
    cudaMemsetAsync(m_impl->devInfOutSum, 0, sizeof(float), stream);

    constexpr uint32_t kThreads = 256u;
    const uint32_t kBlocks = (count + kThreads - 1u) / kThreads;
    nrc_infout_sum_kernel<<<kBlocks, kThreads, 0, stream>>>(
        outputDevPtr, count, m_impl->devInfOutSum);

    cudaMemcpyAsync(m_impl->hostInfOutSum, m_impl->devInfOutSum,
                    sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaEventRecord(m_impl->infOutSumEvent, stream);
    m_impl->infOutSumPending       = true;
    m_impl->infOutSumCountInFlight = count;
}

float Network::LastInferenceOutMagnitudeMean() const {
    //sentinel -1: no readback has landed yet. Caller treats this as "no data"
    //rather than as zero so the collapse detector doesn't fire during warmup.
    if (!m_impl || m_impl->lastInferenceOutCount == 0u) return -1.0f;
    const float denom = float(m_impl->lastInferenceOutCount) * 3.0f;
    return m_impl->lastInferenceOutSum / denom;
}

void Network::WaitTrainDoneOnStream(void* streamPtr) {
    //inject a one way wait on trainDoneEvent into the caller's stream so any
    //subsequent main stream work (notably nrc_frame_begin's cudaMemsetAsync on
    //m_nrcTrainRecords) is ordered after the auxStream training and the fill
    //kernel that reads the same allocation. CUDA treats an event that has not
    //yet been recorded as already completed, so the very first call (before
    //the first TrainFrame finishes) returns immediately and the renderer does
    //not need a separate first frame guard.
    if (!m_impl || !m_impl->ready || !m_impl->trainDoneEvent || !streamPtr) return;
    cudaStream_t stream = static_cast<cudaStream_t>(streamPtr);
    cudaStreamWaitEvent(stream, m_impl->trainDoneEvent, 0);
}

void Network::WaitIdle() {
    if (!m_impl || !m_impl->auxStream) return;
    cudaStreamSynchronize(m_impl->auxStream);
    //drain pending readbacks, host can safely reallocate after return
    if (m_impl->counterReadbackPending) {
        cudaEventSynchronize(m_impl->counterReadbackEvent);
        uint32_t v = *m_impl->hostCounterReadback;
        if (v > kTrainingRecordsPerFrame) v = kTrainingRecordsPerFrame;
        m_impl->lastValidVertices = v;
        m_impl->counterReadbackPending = false;
    }
    if (m_impl->inferenceCountPending) {
        cudaEventSynchronize(m_impl->inferenceCountEvent);
        m_impl->lastInferenceCount     = *m_impl->hostInferenceCount;
        m_impl->inferenceCountPending  = false;
    }
}

void Network::TrainFrame(
    void*       streamPtr,
    const void* trainRecordsDevPtr,
    const void* inferenceOutDevPtr,
    uint32_t    inferenceOutCapacity,
    uint32_t    targetRecords)
{
    if (!m_impl->ready) return;
    if (!trainRecordsDevPtr) return;
    //ignore caller's main stream, run on auxStream to overlap next frame's work
    (void)streamPtr;
    cudaStream_t stream = m_impl->auxStream;

    //wait for this frame's inference to finish writing inferenceOut
    //class-1 backward fill reads those outputs, would race without this barrier
    cudaStreamWaitEvent(stream, m_impl->inferenceDoneEvent, 0);

    //Only the counter needs a reset, the fill kernel atomicAdds into it.
    //Feature/target scratch was previously zeroed defensively, but tcnn's
    //training_step only reads `perBatch` rows starting at b*perBatch, all of
    //which the kernel populates before the SGD loop runs. Padding past the
    //emitted count is unread, so the memsets were pure stream serialisation.
    cudaMemsetAsync(m_impl->trainCounter,  0, sizeof(uint32_t), stream);

    //launch over all path slots, threads early-out on numVertices==0
    constexpr uint32_t kThreads = 256;
    const uint32_t blocks = (kMaxTrainingPaths + kThreads - 1u) / kThreads;
    fill_training_batch_kernel<<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(trainRecordsDevPtr),
        reinterpret_cast<const float*>  (inferenceOutDevPtr),
        kMaxTrainingPaths,
        inferenceOutCapacity,
        targetRecords,
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

    //clamp to the fill kernel's actual write cap before sizing the batches.
    //outCounter counts emit ATTEMPTS; with multi-row emission it can overshoot
    //the cap, but only min(cap, outCounter) rows were actually WRITTEN. Sizing
    //perBatch off the raw counter would make the SGD read past the written
    //region into stale / uninitialized rows. lastValidVertices stays the raw
    //(buffer-ceiling-clamped) value so the adaptive-tile feedback still sees
    //true demand; only the SGD sizing uses the tighter write cap.
    const uint32_t writeCap = (targetRecords < kTrainingRecordsPerFrame)
                              ? targetRecords : kTrainingRecordsPerFrame;
    const uint32_t writtenRows = (validVertices < writeCap) ? validVertices : writeCap;

    //round down to tcnn granularity, drops <=255 rows
    const uint32_t perBatchRaw = writtenRows / kTrainingBatchesPerFrame;
    const uint32_t perBatch    = (perBatchRaw / kBatchGranularity) * kBatchGranularity;
    if (perBatch == 0) {
        //still signal done so next inference doesn't wait on stale event
        cudaEventRecord(m_impl->trainDoneEvent, stream);
        return;
    }

    //SGD steps run back to back, EMA is applied once at the end. Folding the
    //per step EMA into a single launch saves three ema_update_kernel
    //dispatches per frame, which is pure scheduler overhead between SGD
    //steps where ReSTIR could otherwise interleave on the D3D12 queue.
    //Quality cost: the exact recursion would weight the four intermediate
    //states W_1..W_3 with (1-a)*a^k coefficients, but tcnn does not expose
    //them, so the bracket is collapsed onto W_final. Total weight on the
    //SGD block is still (1 - a^N) which matches the exact form, only the
    //within bracket distribution is approximated.
    const float alpha = m_impl->emaAlpha;
    for (uint32_t b = 0; b < kTrainingBatchesPerFrame; ++b) {
        const float* fPtr = m_impl->trainFeatures + b * perBatch * kRawInputDim;
        const float* tPtr = m_impl->trainTargets  + b * perBatch * kOutputDim;
        TrainingStep((void*)stream, fPtr, tPtr, perBatch);
    }

    const uint64_t emaStepBefore = m_impl->emaStep;
    m_impl->emaStep += kTrainingBatchesPerFrame;

    //cumulative coefficients across N = kTrainingBatchesPerFrame steps:
    //  W_hat_t1 = (1 - a^N)/eta_t1 * W_final + a^N * eta_t0/eta_t1 * W_hat_t0
    //  eta_t = 1 - a^t with eta_0 = 0, so first-frame collapses to W_final.
    const double da    = (double)alpha;
    const double daN   = pow(da, (double)kTrainingBatchesPerFrame);
    const double t1    = (double)m_impl->emaStep;
    const double t0    = (double)emaStepBefore;
    const double etaT1 = 1.0 - pow(da, t1);
    const double etaT0 = (t0 > 0.0) ? 1.0 - pow(da, t0) : 0.0;
    const float  coefNew = (float)((1.0 - daN) / etaT1);
    const float  coefOld = (float)(daN * etaT0 / etaT1);

    const uint32_t kEmaThreads = 256u;
    const uint32_t kEmaBlocks  = uint32_t((m_impl->nParams + kEmaThreads - 1u) / kEmaThreads);
    ema_update_kernel<<<kEmaBlocks, kEmaThreads, 0, stream>>>(
        m_impl->emaParams,
        m_impl->model.trainer->params(),
        coefNew,
        coefOld,
        m_impl->nParams);

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
