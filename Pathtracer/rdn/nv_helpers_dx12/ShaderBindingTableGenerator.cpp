/*-----------------------------------------------------------------------
Copyright (c) 2014-2018, NVIDIA. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
* Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
* Neither the name of its contributors may be used to endorse
or promote products derived from this software without specific
prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-----------------------------------------------------------------------*/

#include <stdexcept>
#include <algorithm>
#include "ShaderBindingTableGenerator.h"

#ifndef D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT
#define D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT 64
#endif

#define ALIGN_64(v)                                                                                                    \
    (((v) + (D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT) - 1) & ~((D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT) - 1))

#ifndef ROUND_UP
#define ROUND_UP(v, powerOf2Alignment) (((v) + (powerOf2Alignment) - 1) & ~((powerOf2Alignment) - 1))
#endif

namespace nv_helpers_dx12 {

void ShaderBindingTableGenerator::AddRayGenerationProgram(const std::wstring& entryPoint,
                                                          const std::vector<void*>& inputData) {
    m_rayGen.emplace_back(SBTEntry(entryPoint, inputData));
}

void ShaderBindingTableGenerator::AddMissProgram(const std::wstring& entryPoint, const std::vector<void*>& inputData) {
    m_miss.emplace_back(SBTEntry(entryPoint, inputData));
}

void ShaderBindingTableGenerator::AddHitGroup(const std::wstring& entryPoint, const std::vector<void*>& inputData) {
    m_hitGroup.emplace_back(SBTEntry(entryPoint, inputData));
}

void ShaderBindingTableGenerator::AddCallableProgram(const std::wstring& entryPoint,
                                                     const std::vector<void*>& inputData) {
    m_callable.emplace_back(SBTEntry(entryPoint, inputData));
}

// Align records within each section, then align the section boundaries.
uint32_t ShaderBindingTableGenerator::ComputeSBTSize() {
    m_progIdSize = D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT;

    m_rayGenEntrySize = GetEntrySize(m_rayGen);
    m_missEntrySize = GetEntrySize(m_miss);
    m_hitGroupEntrySize = GetEntrySize(m_hitGroup);
    m_callableEntrySize = GetEntrySize(m_callable);

    uint32_t sbtSize = ALIGN_64(m_rayGenEntrySize * static_cast<UINT>(m_rayGen.size())) +
                       ALIGN_64(m_missEntrySize * static_cast<UINT>(m_miss.size())) +
                       ALIGN_64(m_hitGroupEntrySize * static_cast<UINT>(m_hitGroup.size())) +
                       ALIGN_64(m_callableEntrySize * static_cast<UINT>(m_callable.size()));

    return ROUND_UP(sbtSize, 256);
}

void ShaderBindingTableGenerator::Generate(ID3D12Resource* sbtBuffer, ID3D12StateObjectProperties* raytracingPipeline) {
    uint8_t* pData;
    HRESULT hr = sbtBuffer->Map(0, nullptr, reinterpret_cast<void**>(&pData));
    if (FAILED(hr))
        throw std::logic_error("Could not map the shader binding table");

    uint8_t* pStart = pData;

    uint32_t rgSize = m_rayGen.size() * m_rayGenEntrySize;
    CopyShaderData(raytracingPipeline, pData, m_rayGen, m_rayGenEntrySize);
    pData += ALIGN_64(rgSize);

    uint32_t missSize = m_miss.size() * m_missEntrySize;
    CopyShaderData(raytracingPipeline, pData, m_miss, m_missEntrySize);
    pData += ALIGN_64(missSize);

    uint32_t hitSize = m_hitGroup.size() * m_hitGroupEntrySize;
    CopyShaderData(raytracingPipeline, pData, m_hitGroup, m_hitGroupEntrySize);
    pData += ALIGN_64(hitSize);

    uint32_t callSize = m_callable.size() * m_callableEntrySize;
    CopyShaderData(raytracingPipeline, pData, m_callable, m_callableEntrySize);
    pData += ALIGN_64(callSize);

    sbtBuffer->Unmap(0, nullptr);
}

void ShaderBindingTableGenerator::Reset() {
    m_rayGen.clear();
    m_miss.clear();
    m_hitGroup.clear();
    m_callable.clear();

    m_rayGenEntrySize = 0;
    m_missEntrySize = 0;
    m_hitGroupEntrySize = 0;
    m_callableEntrySize = 0;
    m_progIdSize = 0;
}

UINT ShaderBindingTableGenerator::GetRayGenSectionSize() const {
    return ALIGN_64(m_rayGenEntrySize * static_cast<UINT>(m_rayGen.size()));
}

UINT ShaderBindingTableGenerator::GetRayGenEntrySize() const {
    return m_rayGenEntrySize;
}

UINT ShaderBindingTableGenerator::GetMissSectionSize() const {
    return ALIGN_64(m_missEntrySize * static_cast<UINT>(m_miss.size()));
}

UINT ShaderBindingTableGenerator::GetMissEntrySize() {
    return m_missEntrySize;
}

UINT ShaderBindingTableGenerator::GetHitGroupSectionSize() const {
    return ALIGN_64(m_hitGroupEntrySize * static_cast<UINT>(m_hitGroup.size()));
}

UINT ShaderBindingTableGenerator::GetHitGroupEntrySize() const {
    return m_hitGroupEntrySize;
}

UINT ShaderBindingTableGenerator::GetCallableSectionSize() const {
    return ALIGN_64(m_callableEntrySize * static_cast<UINT>(m_callable.size()));
}

UINT ShaderBindingTableGenerator::GetCallableEntrySize() const {
    return m_callableEntrySize;
}

// Each record stores a shader identifier followed by local root arguments.
uint32_t ShaderBindingTableGenerator::CopyShaderData(ID3D12StateObjectProperties* raytracingPipeline,
                                                     uint8_t* outputData, const std::vector<SBTEntry>& shaders,
                                                     uint32_t entrySize) {
    uint8_t* pData = outputData;
    for (const auto& shader : shaders) {

        void* id = raytracingPipeline->GetShaderIdentifier(shader.m_entryPoint.c_str());
        if (!id) {
            std::wstring errMsg(std::wstring(L"Unknown shader identifier used in the SBT: ") + shader.m_entryPoint);
            throw std::logic_error(std::string(errMsg.begin(), errMsg.end()));
        }

        memcpy(pData, id, m_progIdSize);

        memcpy(pData + m_progIdSize, shader.m_inputData.data(), shader.m_inputData.size() * 8);

        pData += entrySize;
    }

    return static_cast<uint32_t>(shaders.size()) * entrySize;
}

uint32_t ShaderBindingTableGenerator::GetEntrySize(const std::vector<SBTEntry>& entries) {

    size_t maxArgs = 0;
    for (const auto& shader : entries) {
        maxArgs = std::max(maxArgs, shader.m_inputData.size());
    }

    uint32_t entrySize = m_progIdSize + 8 * static_cast<uint32_t>(maxArgs);

    entrySize = ROUND_UP(entrySize, D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT);

    return entrySize;
}

ShaderBindingTableGenerator::SBTEntry::SBTEntry(std::wstring entryPoint, std::vector<void*> inputData)
    : m_entryPoint(std::move(entryPoint)), m_inputData(std::move(inputData)) {}
} // namespace nv_helpers_dx12
