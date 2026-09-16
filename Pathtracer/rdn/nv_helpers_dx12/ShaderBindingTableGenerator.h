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

#pragma once

#include "d3d12.h"

#include <vector>
#include <string>

namespace nv_helpers_dx12 {

class ShaderBindingTableGenerator {
  public:
    void AddRayGenerationProgram(const std::wstring& entryPoint, const std::vector<void*>& inputData);

    void AddMissProgram(const std::wstring& entryPoint, const std::vector<void*>& inputData);

    void AddHitGroup(const std::wstring& entryPoint, const std::vector<void*>& inputData);

    void AddCallableProgram(const std::wstring& entryPoint, const std::vector<void*>& inputData);

    uint32_t ComputeSBTSize();

    void Generate(ID3D12Resource* sbtBuffer, ID3D12StateObjectProperties* raytracingPipeline);

    void Reset();

    UINT GetRayGenSectionSize() const;

    UINT GetRayGenEntrySize() const;

    UINT GetMissSectionSize() const;

    UINT GetMissEntrySize();

    UINT GetHitGroupSectionSize() const;

    UINT GetHitGroupEntrySize() const;

    UINT GetCallableSectionSize() const;
    UINT GetCallableEntrySize() const;

  private:
    struct SBTEntry {
        SBTEntry(std::wstring entryPoint, std::vector<void*> inputData);

        const std::wstring m_entryPoint;
        const std::vector<void*> m_inputData;
    };

    uint32_t CopyShaderData(ID3D12StateObjectProperties* raytracingPipeline, uint8_t* outputData,
                            const std::vector<SBTEntry>& shaders, uint32_t entrySize);

    uint32_t GetEntrySize(const std::vector<SBTEntry>& entries);

    std::vector<SBTEntry> m_rayGen;
    std::vector<SBTEntry> m_miss;
    std::vector<SBTEntry> m_hitGroup;
    std::vector<SBTEntry> m_callable;

    uint32_t m_rayGenEntrySize;
    uint32_t m_missEntrySize;
    uint32_t m_hitGroupEntrySize;
    uint32_t m_callableEntrySize;

    UINT m_progIdSize;
};
} // namespace nv_helpers_dx12
