#pragma once
//====================================
//NRC NETWORK ENTRY POINTS
//====================================
//callers see only void* and float*, impl in NrcNetwork.cu, contracts in NrcLayout.h

#include "NrcLayout.h"

#include <cstdint>
#include <memory>

namespace nrc {

//tcnn toolchain validation at boot
bool SmokeTest(void* stream);

//====================================
//CUDA HELPERS
//====================================
void Memzero(void* stream, void* devPtr, size_t bytes);
void Memfill(void* stream, void* devPtr, int value, size_t bytes);

//stalls stream to host, returns 0 on error
uint32_t ReadU32(void* stream, const void* devPtr);

//====================================
//NETWORK
//====================================
//16-input, composite encoding to 74 dims, 2 hidden x 64 ReLU, 3 out
class Network {
public:
    Network();
    ~Network();
    Network(const Network&)            = delete;
    Network& operator=(const Network&) = delete;

    bool Init();
    void Shutdown();
    bool IsReady() const;

    //reseed weights, zero EMA, keep buffers/stream/events alive
    //caller must ensure no in-flight NRC ops on the main stream
    bool ReinitWeights();

    //count must be multiple of kBatchGranularity, caller pads
    void Inference(
        void*        stream,
        const float* inputDevPtr,
        float*       outputDevPtr,
        uint32_t     count);

    //one SGD step, caller shuffles and sizes, returns loss or -1
    float TrainingStep(
        void*        stream,
        const float* inputDevPtr,
        const float* targetDevPtr,
        uint32_t     count);

    //end-to-end per-frame training, clear then backward-fill then kTrainingBatchesPerFrame steps
    //sized to actual vertex count to avoid training on zero-padded tail
    //inferenceOutCapacity bounds the kTailCache slot deref in the fill kernel,
    //defensive against torn meta reads if the caller forgets to gate the next
    //frame's frame_begin memset on trainDoneEvent
    void TrainFrame(
        void*       stream,
        const void* trainRecordsDevPtr,
        const void* inferenceOutDevPtr,
        uint32_t    inferenceOutCapacity);

    //valid vertex count after last TrainFrame, drives adaptive tile feedback
    uint32_t LastValidVertexCount() const;

    //schedule async D2H copy of the inference counter on `stream`, harvested
    //next frame by LastInferenceCount. devCounterPtr points at counter[0]
    //(NRC_C_OFF_INFERENCE_COUNT). Non blocking, no host sync.
    void ScheduleInferenceCounterReadback(void* stream, const void* devCounterPtr);

    //last harvested inference counter, 0 until first readback lands. Used by
    //the renderer to size the next frame's nrc_inference_capacity (raygen cap
    //plus inference dispatch count).
    uint32_t LastInferenceCount() const;

    //block on auxStream, required before reallocating shared buffers
    void WaitIdle();

    //inject a wait on the previous frame's trainDoneEvent into `stream` so
    //subsequent work on `stream` is ordered after the auxStream training and
    //fill kernel finish. Required before any main stream write that aliases
    //resources read by fill_training_batch_kernel (the trainRecords meta
    //section in particular). No-op until trainDoneEvent has been recorded
    //at least once, CUDA treats unrecorded events as already completed.
    void WaitTrainDoneOnStream(void* stream);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

}
