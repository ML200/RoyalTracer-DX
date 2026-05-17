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

/*
Contacts for feedback:
- pgautron@nvidia.com (Pascal Gautron)
- mlefrancois@nvidia.com (Martin-Karl Lefrancois)

The ShaderBindingTable is a helper to construct the SBT. It helps to maintain the
proper offsets of each element, required when constructing the SBT, but also when filling the
dispatch rays description.

*/

#include <stdexcept>
#include <algorithm>
#include "ShaderBindingTableGenerator.h"

// Ensure we have the alignment constant (64 bytes)
#ifndef D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT
#define D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT 64
#endif

// Helper macro to align values to 64 bytes
#define ALIGN_64(v) (((v) + (D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT)-1) & ~((D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT)-1))


// Helper to compute aligned buffer sizes
#ifndef ROUND_UP
#define ROUND_UP(v, powerOf2Alignment) (((v) + (powerOf2Alignment)-1) & ~((powerOf2Alignment)-1))
#endif

namespace nv_helpers_dx12
{

//--------------------------------------------------------------------------------------------------
//
// Add a ray generation program by name, with its list of data pointers or values according to
// the layout of its root signature
void ShaderBindingTableGenerator::AddRayGenerationProgram(const std::wstring& entryPoint,
                                                          const std::vector<void*>& inputData)
{
  m_rayGen.emplace_back(SBTEntry(entryPoint, inputData));
}

//--------------------------------------------------------------------------------------------------
//
// Add a miss program by name, with its list of data pointers or values according to
// the layout of its root signature
void ShaderBindingTableGenerator::AddMissProgram(const std::wstring& entryPoint,
                                                 const std::vector<void*>& inputData)
{
  m_miss.emplace_back(SBTEntry(entryPoint, inputData));
}

//--------------------------------------------------------------------------------------------------
//
// Add a hit group by name, with its list of data pointers or values according to
// the layout of its root signature
void ShaderBindingTableGenerator::AddHitGroup(const std::wstring& entryPoint,
                                              const std::vector<void*>& inputData)
{
  m_hitGroup.emplace_back(SBTEntry(entryPoint, inputData));
}

void ShaderBindingTableGenerator::AddCallableProgram(const std::wstring& entryPoint,
                                                     const std::vector<void*>& inputData)
{
  m_callable.emplace_back(SBTEntry(entryPoint, inputData));
}

//--------------------------------------------------------------------------------------------------
//
// Compute the size of the SBT based on the set of programs and hit groups it contains
uint32_t ShaderBindingTableGenerator::ComputeSBTSize()
{
  m_progIdSize = D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT;

  m_rayGenEntrySize   = GetEntrySize(m_rayGen);
  m_missEntrySize     = GetEntrySize(m_miss);
  m_hitGroupEntrySize = GetEntrySize(m_hitGroup);
  m_callableEntrySize = GetEntrySize(m_callable);

  // Sum up aligned section sizes to ensure pData pointers in Generate() land on 64-byte boundaries
  uint32_t sbtSize =
      ALIGN_64(m_rayGenEntrySize   * static_cast<UINT>(m_rayGen.size())) +
      ALIGN_64(m_missEntrySize     * static_cast<UINT>(m_miss.size())) +
      ALIGN_64(m_hitGroupEntrySize * static_cast<UINT>(m_hitGroup.size())) +
      ALIGN_64(m_callableEntrySize * static_cast<UINT>(m_callable.size()));

  return ROUND_UP(sbtSize, 256);
}

//--------------------------------------------------------------------------------------------------
//
// Build the SBT and store it into sbtBuffer, which has to be pre-allocated on the upload heap.
// Access to the raytracing pipeline object is required to fetch program identifiers using their
// names
void ShaderBindingTableGenerator::Generate(ID3D12Resource* sbtBuffer,
                                             ID3D12StateObjectProperties* raytracingPipeline)
{
  uint8_t* pData;
  HRESULT hr = sbtBuffer->Map(0, nullptr, reinterpret_cast<void**>(&pData));
  if (FAILED(hr)) throw std::logic_error("Could not map the shader binding table");

  // Track the start of the buffer
  uint8_t* pStart = pData;

  // 1. RayGen
  uint32_t rgSize = m_rayGen.size() * m_rayGenEntrySize;
  CopyShaderData(raytracingPipeline, pData, m_rayGen, m_rayGenEntrySize);
  pData += ALIGN_64(rgSize); // Advance pointer by aligned size

  // 2. Miss
  uint32_t missSize = m_miss.size() * m_missEntrySize;
  CopyShaderData(raytracingPipeline, pData, m_miss, m_missEntrySize);
  pData += ALIGN_64(missSize);

  // 3. HitGroup
  uint32_t hitSize = m_hitGroup.size() * m_hitGroupEntrySize;
  CopyShaderData(raytracingPipeline, pData, m_hitGroup, m_hitGroupEntrySize);
  pData += ALIGN_64(hitSize);

  // 4. Callable
  uint32_t callSize = m_callable.size() * m_callableEntrySize;
  CopyShaderData(raytracingPipeline, pData, m_callable, m_callableEntrySize);
  pData += ALIGN_64(callSize);

  sbtBuffer->Unmap(0, nullptr);
}

//--------------------------------------------------------------------------------------------------
//
// Reset the sets of programs and hit groups
  void ShaderBindingTableGenerator::Reset()
{
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

//--------------------------------------------------------------------------------------------------
// The following getters are used to simplify the call to DispatchRays where the offsets of the
// shader programs must be exactly following the SBT layout

//--------------------------------------------------------------------------------------------------
//
// Get the size in bytes of the SBT section dedicated to ray generation programs
UINT ShaderBindingTableGenerator::GetRayGenSectionSize() const
{
  return ALIGN_64(m_rayGenEntrySize * static_cast<UINT>(m_rayGen.size()));
}

//--------------------------------------------------------------------------------------------------
//
// Get the size in bytes of one ray generation program entry in the SBT
UINT ShaderBindingTableGenerator::GetRayGenEntrySize() const
{
  return m_rayGenEntrySize;
}

//--------------------------------------------------------------------------------------------------
//
// Get the size in bytes of the SBT section dedicated to miss programs
UINT ShaderBindingTableGenerator::GetMissSectionSize() const
{
  return ALIGN_64(m_missEntrySize * static_cast<UINT>(m_miss.size()));
}

//--------------------------------------------------------------------------------------------------
//
// Get the size in bytes of one miss program entry in the SBT
UINT ShaderBindingTableGenerator::GetMissEntrySize()
{
  return m_missEntrySize;
}

//--------------------------------------------------------------------------------------------------
//
// Get the size in bytes of the SBT section dedicated to hit groups
UINT ShaderBindingTableGenerator::GetHitGroupSectionSize() const
{
  return ALIGN_64(m_hitGroupEntrySize * static_cast<UINT>(m_hitGroup.size()));
}

//--------------------------------------------------------------------------------------------------
//
// Get the size in bytes of one hit group entry in the SBT
UINT ShaderBindingTableGenerator::GetHitGroupEntrySize() const
{
  return m_hitGroupEntrySize;
}

UINT ShaderBindingTableGenerator::GetCallableSectionSize() const
{
  return ALIGN_64(m_callableEntrySize * static_cast<UINT>(m_callable.size()));
}

UINT ShaderBindingTableGenerator::GetCallableEntrySize() const
{
  return m_callableEntrySize;
}

//--------------------------------------------------------------------------------------------------
//
// For each entry, copy the shader identifier followed by its resource pointers and/or root
// constants in outputData, with a stride in bytes of entrySize, and returns the size in bytes
// actually written to outputData.
uint32_t ShaderBindingTableGenerator::CopyShaderData(
    ID3D12StateObjectProperties* raytracingPipeline, uint8_t* outputData,
    const std::vector<SBTEntry>& shaders, uint32_t entrySize)
{
  uint8_t* pData = outputData;
  for (const auto& shader : shaders)
  {
    // Get the shader identifier, and check whether that identifier is known
    void* id = raytracingPipeline->GetShaderIdentifier(shader.m_entryPoint.c_str());
    if (!id)
    {
      std::wstring errMsg(std::wstring(L"Unknown shader identifier used in the SBT: ") +
                          shader.m_entryPoint);
      throw std::logic_error(std::string(errMsg.begin(), errMsg.end()));
    }
    // Copy the shader identifier
    memcpy(pData, id, m_progIdSize);
    // Copy all its resources pointers or values in bulk
    memcpy(pData + m_progIdSize, shader.m_inputData.data(), shader.m_inputData.size() * 8);

    pData += entrySize;
  }
  // Return the number of bytes actually written to the output buffer
  return static_cast<uint32_t>(shaders.size()) * entrySize;
}

//--------------------------------------------------------------------------------------------------
//
// Compute the size of the SBT entries for a set of entries, which is determined by the maximum
// number of parameters of their root signature
uint32_t ShaderBindingTableGenerator::GetEntrySize(const std::vector<SBTEntry>& entries)
{
  // Find the maximum number of parameters used by a single entry
  size_t maxArgs = 0;
  for (const auto& shader : entries)
  {
    maxArgs = std::max(maxArgs, shader.m_inputData.size());
  }
  // A SBT entry is made of a program ID and a set of parameters, taking 8 bytes each. Those
  // parameters can either be 8-bytes pointers, or 4-bytes constants
  uint32_t entrySize = m_progIdSize + 8 * static_cast<uint32_t>(maxArgs);

  // The entries of the shader binding table must be 16-bytes-aligned
  entrySize = ROUND_UP(entrySize, D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT);

  return entrySize;
}

//--------------------------------------------------------------------------------------------------
//
//
ShaderBindingTableGenerator::SBTEntry::SBTEntry(std::wstring entryPoint,
                                                std::vector<void*> inputData)
    : m_entryPoint(std::move(entryPoint)), m_inputData(std::move(inputData))
{
}
} // namespace nv_helpers_dx12
