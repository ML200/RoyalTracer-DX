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
    static PassDesc ParseToken(const std::wstring& token);
    void LinkLoops();

    std::vector<std::wstring>                   m_tokens;
    std::vector<PassDesc>                       m_passes;
    std::unordered_map<std::wstring, uint32_t>  m_passIndex;
};
