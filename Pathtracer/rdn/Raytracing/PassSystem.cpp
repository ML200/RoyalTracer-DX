#include "PassSystem.h"

void PassSystem::Build(const std::vector<std::wstring>& tokens) {
    m_tokens = tokens;
    m_passes.clear();
    m_passIndex.clear();

    // Resolve feature requirements before linking loop control tokens.
    for (auto& t : tokens) {
        auto pass = ParseToken(t);
        pass.requiredFeatures = RequiredFeatures(pass.file);
        m_passes.push_back(std::move(pass));
    }

    LinkLoops();
}

uint32_t PassSystem::RequiredFeatures(const std::wstring& file) {
    using namespace pass_feature;
    if (file == L"Pass_light_learning_v8.hlsl")
        return MeshLights | LightLearning;
    if (file == L"Pass_sharc_debug_v8.hlsl")
        return Sharc | SharcDebug;
    if (file.rfind(L"Pass_sharc_", 0) == 0)
        return Sharc;
    if (file.rfind(L"Pass_lite_", 0) == 0)
        return DiffuseReuse | (file == L"Pass_lite_shift_v8.hlsl" ? SpatialReuse : 0u);
    return 0;
}

PassDesc PassSystem::ParseToken(const std::wstring& token) {
    PassDesc p{};

    if (token == L"barrier") {
        p.stage = Stage::Barrier;
        return p;
    }
    if (token == L"endloop") {
        p.stage = Stage::LoopEnd;
        return p;
    }
    if (token == L"dlss") {
        p.stage = Stage::DLSS;
        return p;
    }

    if (token.rfind(L"loop:", 0) == 0) {
        p.stage = Stage::LoopStart;
        const std::wstring count = token.substr(5);
        if (count.empty())
            throw std::runtime_error("Invalid loop count");
        if (iswdigit(count[0])) {
            if (swscanf_s(count.c_str(), L"%u", &p.loopCount) != 1)
                throw std::runtime_error("Invalid loop count");
        } else {
            p.loopTag = count; // resolved by the renderer each frame
        }
        return p;
    }

    const size_t bar = token.find(L'|');
    p.file = token.substr(0, bar);
    if (bar == std::wstring::npos)
        return p;

    const std::wstring tail = token.substr(bar + 1);

    if (tail == L"rg")
        return p;
    if (tail.rfind(L"cs:", 0) == 0) {
        p.stage = Stage::Compute;
        if (swscanf_s(tail.c_str() + 3, L"%ux%u", &p.groupX, &p.groupY) != 2)
            throw std::runtime_error("Invalid cs size");
        return p;
    }
    if (tail.rfind(L"fx:", 0) == 0) {
        p.stage = Stage::FixedCompute;
        if (swscanf_s(tail.c_str() + 3, L"%u", &p.groupX) != 1)
            throw std::runtime_error("Invalid fx size");
        p.groupY = 1;
        return p;
    }

    throw std::runtime_error("Unknown stage spec in pass string");
}

void PassSystem::LinkLoops() {
    std::vector<size_t> stack;
    for (size_t i = 0; i < m_passes.size(); ++i) {
        if (m_passes[i].stage == Stage::LoopStart)
            stack.push_back(i);
        else if (m_passes[i].stage == Stage::LoopEnd) {
            if (stack.empty())
                throw std::runtime_error("Found 'endloop' without matching 'loop:'");
            m_passes[i].targetIdx = static_cast<int32_t>(stack.back());
            stack.pop_back();
        }
    }
    if (!stack.empty())
        throw std::runtime_error("Found 'loop:' without matching 'endloop'");
}
