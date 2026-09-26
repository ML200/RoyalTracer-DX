#pragma once

#include "../Common.h"
#include <dxcapi.h>

// A pass token expands into one dispatch stage.
enum class Stage {
    RayGen,
    Compute,
    FixedCompute,
    Barrier,
    LoopStart,
    LoopEnd,
    DLSS
};

namespace pass_feature {
constexpr uint32_t Sharc = 1u << 0;         // radiance cache maintenance and training
constexpr uint32_t SharcDebug = 1u << 1;    // cache inspection view
constexpr uint32_t DiffuseReuse = 1u << 2;
constexpr uint32_t SpatialReuse = 1u << 3;
constexpr uint32_t MeshLights = 1u << 4;
constexpr uint32_t LightLearning = 1u << 5;
}

struct PassDesc {
    std::wstring file;
    Stage stage = Stage::RayGen;
    uint32_t groupX = 0;
    uint32_t groupY = 0;
    uint32_t psoIdx = UINT32_MAX;
    uint32_t loopCount = 0;
    std::wstring loopTag; // runtime-resolved trip count (loop:pt_samples)
    int32_t targetIdx = -1;

    uint32_t requiredFeatures = 0;
    bool executedLastFrame = false;

    bool IsEnabled(uint32_t features) const { return (features & requiredFeatures) == requiredFeatures; }
};

class PassSystem {
  public:
    void Build(const std::vector<std::wstring>& tokens);

    const std::vector<PassDesc>& Passes() const { return m_passes; }
    std::vector<PassDesc>& Passes() { return m_passes; }
    const std::vector<std::wstring>& Tokens() const { return m_tokens; }

    uint32_t PassIndexByFile(const std::wstring& file) const {
        auto it = m_passIndex.find(file);
        return (it != m_passIndex.end()) ? it->second : UINT32_MAX;
    }

    void RegisterPassIndex(const std::wstring& file, uint32_t index) { m_passIndex[file] = index; }

  private:
    static uint32_t RequiredFeatures(const std::wstring& file);
    static PassDesc ParseToken(const std::wstring& token);
    void LinkLoops();

    std::vector<std::wstring> m_tokens;
    std::vector<PassDesc> m_passes;
    std::unordered_map<std::wstring, uint32_t> m_passIndex;
};
