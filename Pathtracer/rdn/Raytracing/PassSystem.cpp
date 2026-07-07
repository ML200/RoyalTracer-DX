//====================================
//PASS SYSTEM
//====================================

#include "PassSystem.h"

void PassSystem::Build(const std::vector<std::wstring>& tokens) {
    m_tokens = tokens;
    m_passes.clear();
    m_passIndex.clear();

    for (auto& t : tokens)
        m_passes.push_back(ParseToken(t));

    LinkLoops();
}

//====================================
//PARSE TOKEN
//====================================
PassDesc PassSystem::ParseToken(const std::wstring& token) {
    PassDesc p{};

    //keywords
    if (token == L"barrier")   { p.stage = Stage::Barrier;   return p; }
    if (token == L"pingswap")  { p.stage = Stage::PingSwap;  return p; }
    if (token == L"endloop")   { p.stage = Stage::LoopEnd;   return p; }
    if (token == L"clearsort") { p.stage = Stage::ClearSort;  return p; }
    if (token == L"dlss")      { p.stage = Stage::DLSS;       return p; }

    //cuda:<name>, name stored in 'file' for PassIndexByFile lookup
    if (token.rfind(L"cuda:", 0) == 0) {
        p.stage = Stage::CudaOp;
        p.file  = token.substr(5);
        return p;
    }

    //loop:N — redispatch the passes up to the matching 'endloop' N times.
    if (token.rfind(L"loop:", 0) == 0) {
        p.stage = Stage::LoopStart;
        if (swscanf_s(token.c_str() + 5, L"%u", &p.loopCount) != 1)
            throw std::runtime_error("Invalid loop count");
        return p;
    }

    //file|stage_spec
    const size_t bar = token.find(L'|');
    p.file = token.substr(0, bar);
    if (bar == std::wstring::npos) return p;

    const std::wstring tail = token.substr(bar + 1);

    if (tail == L"rg" || tail == L"raygen") return p;
    //rg:<tag> — a RayGen entry with a dispatch-shape annotation (see PassDesc::
    //dispatchTag): lets two entries dispatching the SAME file pick a different
    //Depth in Stage::RayGen, without needing two different compiled shaders.
    if (tail.rfind(L"rg:", 0) == 0) { p.dispatchTag = tail.substr(3); return p; }
    if (tail == L"call") { p.stage = Stage::Callable; return p; }

    if (tail.rfind(L"wg:", 0) == 0) {
        p.stage = Stage::Compute; p.isWorkGraph = true;
        swscanf_s(tail.c_str() + 3, L"%ux%u", &p.groupX, &p.groupY);
        return p;
    }
    if (tail.rfind(L"wf:", 0) == 0) {
        p.stage = Stage::Wavefront;
        if (swscanf_s(tail.c_str() + 3, L"%u", &p.groupX) != 1)
            throw std::runtime_error("Invalid wf size");
        p.groupY = 1; return p;
    }
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
        p.groupY = 1; return p;
    }

    throw std::runtime_error("Unknown stage spec in pass string");
}

//====================================
//LINK LOOPS
//====================================
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
