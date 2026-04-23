#pragma once
// ═══════════════════════════════════════════════════════════════════
// NRC/NrcNetwork.h — C++ entry points for the NRC network.
//
// Implementation lives in NrcNetwork.cu. Callers see no CUDA / tcnn
// headers; streams and device pointers are passed as void*/float*.
// Buffer layout contracts live in NrcLayout.h.
// ═══════════════════════════════════════════════════════════════════

#include "NrcLayout.h"

#include <cstdint>
#include <memory>

namespace nrc {

// Legacy smoke test — constructs a toy model and runs one step. Kept
// for tcnn/CUDA toolchain validation at boot.
bool SmokeTest(void* stream);

// ── CUDA helpers (thin wrappers so callers don't need cuda_runtime) ──
// Asynchronously zero-fill `bytes` at `devPtr` on `stream`.
void Memzero(void* stream, void* devPtr, size_t bytes);

// Asynchronously fill `bytes` at `devPtr` with `value` byte on `stream`.
void Memfill(void* stream, void* devPtr, int value, size_t bytes);

// Read one u32 from `devPtr`, stalling `stream` to the host. Returns 0
// on error. Intended for small per-frame dispatch gates like "how many
// inference records did raygen produce?".
uint32_t ReadU32(void* stream, const void* devPtr);

// Full network: 14-input (14 → composite encoding → 62) → 5 hidden
// layers × 64 ReLU → 3 out. Reflectance factorisation, EMA inference
// weights, etc. live on the caller side for now; this class only owns
// the raw training / inference machinery.
class Network {
public:
    Network();
    ~Network();
    Network(const Network&)            = delete;
    Network& operator=(const Network&) = delete;

    // Initialize tcnn with the NRC config. Safe to call once.
    bool Init();
    void Shutdown();
    bool IsReady() const;

    // Run inference on `count` packed rows of kRawInputDim fp32. Output
    // is `count` rows of kOutputDim fp32. `count` MUST be a multiple of
    // kBatchGranularity (use AlignBatch() to pad). Inputs beyond the
    // real record count contain garbage but are cheap.
    //
    //   stream       : cudaStream_t (use CudaInterop::Stream())
    //   inputDevPtr  : device ptr into g_NrcInferenceIn  (same buffer
    //                  the HLSL side appends to)
    //   outputDevPtr : device ptr into g_NrcInferenceOut
    void Inference(
        void*        stream,
        const float* inputDevPtr,
        float*       outputDevPtr,
        uint32_t     count);

    // One training step on exactly kTrainingBatchSize rows. Caller is
    // responsible for shuffling and selecting the subset. Returns the
    // reported loss (or -1 if the loss wasn't scalar).
    float TrainingStep(
        void*        stream,
        const float* inputDevPtr,
        const float* targetDevPtr);

    // Per-frame end-to-end training pipeline, given the shared D3D/CUDA
    // training buffer (meta + vertex sections) and the frame's inference
    // output buffer (unused until class-1 suffix training lands).
    //
    //   trainRecordsDevPtr : device ptr to g_NrcTrainRecords backing memory
    //   inferenceOutDevPtr : device ptr to g_NrcInferenceOut (for class-1)
    //
    // Runs: clear output counters → backward-fill kernel (one thread per
    // path, walks buckets in reverse, emits (features, target) rows into
    // the internal training buffer) → kTrainingBatchesPerFrame calls to
    // TrainingStep. Silently no-ops when no paths were produced.
    void TrainFrame(
        void*       stream,
        const void* trainRecordsDevPtr,
        const void* inferenceOutDevPtr);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace nrc
