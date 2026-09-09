#pragma once
#include <d3d12.h>
#include <cstdint>

#ifdef DLSSNR_BRIDGE_BUILD
#define DLSSNR_API __declspec(dllexport)
#else
#define DLSSNR_API __declspec(dllimport)
#endif

// Engine-owned ABI. No private NVIDIA structures cross this boundary.
namespace dlssnr {
struct Context;
inline constexpr int kModelStyleCount = 3;
inline constexpr int kRenderPresetCount = 4;
struct Tuning {
    float intensity = 1.0f;
    float localTone = 1.0f;
    float localStructure = 1.0f;
    float skinStructure = -1.0f;
    uint32_t autoMask = 0;
    uint32_t modelStyle = 0;   // DLSSNR.Style: Default, Natural, Cinematic
    uint32_t renderPreset = 0; // DLSSNR.Hint.Render.Preset: Default, 1, 2, 3
};
struct Frame {
    ID3D12Resource* color = nullptr;
    ID3D12Resource* output = nullptr;
    ID3D12Resource* depth = nullptr;
    ID3D12Resource* motion = nullptr;
    uint32_t width = 0, height = 0;
    uint32_t guideWidth = 0, guideHeight = 0;
    // Converts stored vectors to pixels of the motion-vector subrect.
    float motionScaleX = 1.0f, motionScaleY = 1.0f;
    uint32_t reset = 1;
};
// All calls are on the render thread. Release/Destroy require an idle GPU.
extern "C" {
DLSSNR_API uint32_t RoyalNRCreateContext(const wchar_t* runtime, ID3D12Device* nativeDevice, Context** context);
DLSSNR_API uint32_t RoyalNRCreateFeature(Context*, ID3D12GraphicsCommandList*, uint32_t width, uint32_t height, const Tuning*);
DLSSNR_API uint32_t RoyalNREvaluate(Context*, ID3D12GraphicsCommandList*, const Frame*);
DLSSNR_API uint32_t RoyalNRReleaseFeature(Context*);
DLSSNR_API void RoyalNRDestroyContext(Context*);
}
}
