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
    void TrainFrame(
        void*       stream,
        const void* trainRecordsDevPtr,
        const void* inferenceOutDevPtr);

    //valid vertex count after last TrainFrame, drives adaptive tile feedback
    uint32_t LastValidVertexCount() const;

    //block on auxStream, required before reallocating shared buffers
    void WaitIdle();

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

}
