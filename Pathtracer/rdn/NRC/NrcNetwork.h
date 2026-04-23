#pragma once
// ═══════════════════════════════════════════════════════════════════
// NRC/NrcNetwork.h — C++ entry point for the CUDA-side NRC network.
//
// Implementation lives in NrcNetwork.cu and pulls in tiny-cuda-nn;
// callers see only this header and never include any tcnn or CUDA
// headers. 'stream' is a cudaStream_t passed as void* (from
// CudaInterop::Stream()).
// ═══════════════════════════════════════════════════════════════════

namespace nrc {

// Smoke test: constructs a small fused MLP, runs one inference pass
// and one training step on zero-filled tensors, reports success.
// Used only to validate the tcnn + CUDA toolchain end-to-end.
bool SmokeTest(void* stream);

} // namespace nrc
