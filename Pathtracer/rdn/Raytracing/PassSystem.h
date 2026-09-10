#pragma once
//====================================
//PASS SYSTEM
//====================================
//data-driven pass pipeline, parses tokens into stages, dispatch in Renderer

#include "../Common.h"
#include <dxcapi.h>

enum class Stage {
    RayGen, Compute, FixedCompute, Wavefront, Barrier,
    LoopStart, LoopEnd, PingSwap, ClearSort, Callable, DLSS,
    //CudaOp runs callback registered on Renderer, token L"cuda:<name>"
    CudaOp
};

namespace pass_feature {
    constexpr uint32_t PathTracer = 1u << 0;
    constexpr uint32_t LegacyReSTIR = 1u << 1;
    constexpr uint32_t Sharc = 1u << 2;
    constexpr uint32_t DiffuseReuse = 1u << 3;
    constexpr uint32_t SpatialReuse = 1u << 4;
    constexpr uint32_t Clouds = 1u << 5;
    constexpr uint32_t CloudNoise = 1u << 6;
    constexpr uint32_t CloudDensity = 1u << 7;
    constexpr uint32_t CloudAmbient = 1u << 8;
    constexpr uint32_t MeshLights = 1u << 9;
    constexpr uint32_t LightLearning = 1u << 10;
}

struct PassDesc {
    std::wstring  file;
    Stage         stage      = Stage::RayGen;
    uint32_t      groupX     = 0;
    uint32_t      groupY     = 0;
    uint32_t      psoIdx     = UINT32_MAX;
    bool          isWorkGraph = false;
    uint32_t      wgIdx      = UINT32_MAX;
    uint32_t      loopCount  = 0;
    int32_t       targetIdx  = -1;
    //optional "rg:<tag>" annotation on a RayGen entry (ParseToken) — lets two
    //pass-list entries sharing the same file (same SBT record) pick a
    //different per-dispatch Depth in Stage::RayGen (e.g. shift's temporal vs
    //spatial role count). Empty for every ordinary RayGen entry.
    std::wstring  dispatchTag;
    uint32_t      requiredFeatures = 0;
    bool          executedLastFrame = false;

    bool IsEnabled(uint32_t features) const {
        return (features & requiredFeatures) == requiredFeatures;
    }
};

class PassSystem {
public:
    //parse tokens like L"raygen.hlsl|rg", L"barrier", L"cs.hlsl|cs:16x16"
    void Build(const std::vector<std::wstring>& tokens);

    //accessors
    const std::vector<PassDesc>&    Passes()    const { return m_passes; }
    std::vector<PassDesc>&          Passes()          { return m_passes; }
    const std::vector<std::wstring>& Tokens()   const { return m_tokens; }

    uint32_t PassIndexByFile(const std::wstring& file) const {
        auto it = m_passIndex.find(file);
        return (it != m_passIndex.end()) ? it->second : UINT32_MAX;
    }

    void RegisterPassIndex(const std::wstring& file, uint32_t index) {
        m_passIndex[file] = index;
    }

    //editor rebuild
    void Rebuild(const std::vector<std::wstring>& newTokens) { Build(newTokens); }

private:
    static uint32_t RequiredFeatures(const std::wstring& file);
    static PassDesc ParseToken(const std::wstring& token);
    void LinkLoops();

    std::vector<std::wstring>                   m_tokens;
    std::vector<PassDesc>                       m_passes;
    std::unordered_map<std::wstring, uint32_t>  m_passIndex;
};
