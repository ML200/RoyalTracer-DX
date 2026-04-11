/******************************************************************************
 * SlangHelper.h — Runtime Slang-to-DXIL compilation wrapper.
 *
 * Drop-in replacement for the DXC compilation functions in DXRHelper.h.
 * Uses the Slang C++ API (slang.h) to compile .slang files to DXIL blobs
 * that D3D12 consumes identically to DXC output.
 *
 * NOTE: This header is #included from inside namespace nv_helpers_dx12 {}
 *       in DXRHelper.h, so all Slang types must be globally qualified
 *       with the leading :: prefix.
 *****************************************************************************/

#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <set>
#include <stdexcept>

#include <d3d12.h>
#include <dxcapi.h>
#include <wrl/client.h>

#include "DXSampleHelper.h"

//--------------------------------------------------------------------------------------------------
// Minimal IDxcBlob wrapper around a Slang-produced DXIL byte buffer.
// D3D12 only needs GetBufferPointer() and GetBufferSize().
// Defined outside nv_helpers_dx12 to avoid namespace collisions with slang::.
//--------------------------------------------------------------------------------------------------
class SlangDxilBlob : public IDxcBlob
{
public:
    SlangDxilBlob(const void* data, size_t size)
    {
        m_data.resize(size);
        memcpy(m_data.data(), data, size);
    }

    // IUnknown
    ULONG STDMETHODCALLTYPE AddRef()  override { return ++m_ref; }
    ULONG STDMETHODCALLTYPE Release() override
    {
        ULONG r = --m_ref;
        if (r == 0) delete this;
        return r;
    }
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppv) override
    {
        if (riid == __uuidof(IUnknown) || riid == __uuidof(IDxcBlob))
        {
            *ppv = static_cast<IDxcBlob*>(this);
            AddRef();
            return S_OK;
        }
        *ppv = nullptr;
        return E_NOINTERFACE;
    }

    // IDxcBlob
    LPVOID STDMETHODCALLTYPE GetBufferPointer() override { return m_data.data(); }
    SIZE_T STDMETHODCALLTYPE GetBufferSize()    override { return m_data.size(); }

private:
    std::vector<uint8_t> m_data;
    ULONG m_ref = 1;
};

//--------------------------------------------------------------------------------------------------
// Read a file from disk into a std::string.
//--------------------------------------------------------------------------------------------------
inline std::string ReadFileToString(const std::string& path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open())
        return {};
    auto sz = f.tellg();
    f.seekg(0, std::ios::beg);
    std::string content(static_cast<size_t>(sz), '\0');
    f.read(content.data(), sz);
    f.close();
    return content;
}

//--------------------------------------------------------------------------------------------------
// Recursively resolve #include "..." directives so the entire translation unit
// is a single source string.  This is necessary because Slang determines the
// parsing language from the file extension: .slang files on disk would be parsed
// in Slang mode even though the top-level module uses a virtual .hlsl path.
// By resolving includes ourselves, the entire source is parsed in HLSL mode.
//
// - Only handles #include "..." (quoted), not <...> (angle brackets).
// - Tracks already-included filenames to emulate #pragma once / include guards.
//--------------------------------------------------------------------------------------------------
inline std::string ResolveIncludes(
    const std::string& source,
    const std::string& baseDir,
    std::set<std::string>& included)
{
    std::string result;
    result.reserve(source.size() * 2);

    std::istringstream iss(source);
    std::string line;

    while (std::getline(iss, line))
    {
        // Find first non-whitespace
        size_t firstCh = line.find_first_not_of(" \t");
        if (firstCh == std::string::npos)
        {
            result += line;
            result += '\n';
            continue;
        }

        // Check if line is an #include directive
        if (line[firstCh] == '#')
        {
            // Skip whitespace after '#'
            size_t afterHash = line.find_first_not_of(" \t", firstCh + 1);
            if (afterHash != std::string::npos &&
                line.compare(afterHash, 7, "include") == 0)
            {
                // Make sure this isn't inside a // comment
                size_t commentPos = line.find("//");
                if (commentPos != std::string::npos && commentPos < firstCh)
                {
                    result += line;
                    result += '\n';
                    continue;
                }

                // Extract quoted filename
                size_t q1 = line.find('"', afterHash + 7);
                size_t q2 = (q1 != std::string::npos) ? line.find('"', q1 + 1) : std::string::npos;
                if (q1 != std::string::npos && q2 != std::string::npos)
                {
                    std::string incFile = line.substr(q1 + 1, q2 - q1 - 1);

                    // Use filename as key (all shaders in same directory)
                    if (included.find(incFile) == included.end())
                    {
                        included.insert(incFile);

                        std::string fullPath = baseDir.empty()
                            ? incFile
                            : (baseDir + "/" + incFile);

                        std::string content = ReadFileToString(fullPath);
                        if (!content.empty())
                        {
                            // Recursively resolve nested includes
                            result += ResolveIncludes(content, baseDir, included);
                            result += '\n';
                        }
                        else
                        {
                            // File not found — leave the #include for Slang to report
                            result += line;
                            result += '\n';
                        }
                    }
                    // else: already included, skip this line (acts like #pragma once)
                    continue;
                }
            }
        }

        result += line;
        result += '\n';
    }

    return result;
}

//--------------------------------------------------------------------------------------------------
// Core compilation: .slang -> slangc.exe -> HLSL -> patch -> DXC API -> DXIL
//
// Uses slangc CLI to translate .slang (HLSL + Slang extensions like HitObject,
// DescriptorHandle, [mutating]) into standard HLSL, applies targeted patches for
// known Slang codegen bugs, then compiles to DXIL via the DXC API.
//--------------------------------------------------------------------------------------------------

// Helper: run a command and capture stdout + stderr
inline std::pair<std::string, int> RunProcess(const std::string& cmd)
{
    std::string result;
    HANDLE hReadPipe, hWritePipe;
    SECURITY_ATTRIBUTES sa = { sizeof(sa), nullptr, TRUE };
    if (!CreatePipe(&hReadPipe, &hWritePipe, &sa, 0))
        throw std::runtime_error("CreatePipe failed");
    SetHandleInformation(hReadPipe, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOA si = { sizeof(si) };
    si.dwFlags    = STARTF_USESTDHANDLES;
    si.hStdOutput = hWritePipe;
    si.hStdError  = hWritePipe;
    PROCESS_INFORMATION pi = {};

    std::vector<char> cmdBuf(cmd.begin(), cmd.end());
    cmdBuf.push_back('\0');

    if (!CreateProcessA(nullptr, cmdBuf.data(), nullptr, nullptr, TRUE,
                        CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi))
    {
        CloseHandle(hReadPipe);
        CloseHandle(hWritePipe);
        throw std::runtime_error("CreateProcess failed for: " + cmd);
    }
    CloseHandle(hWritePipe);

    char buf[4096];
    DWORD bytesRead;
    while (ReadFile(hReadPipe, buf, sizeof(buf), &bytesRead, nullptr) && bytesRead > 0)
        result.append(buf, bytesRead);
    CloseHandle(hReadPipe);

    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD exitCode = 0;
    GetExitCodeProcess(pi.hProcess, &exitCode);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return { result, (int)exitCode };
}

// Scan a .slang source file for [shader("stage")] entry points and build
// the corresponding "-entry Name -stage stage" arguments for slangc.
inline std::string BuildEntryPointArgs(const std::string& source)
{
    std::string args;
    const std::string searchStr = "[shader(\"";
    size_t pos = 0;
    while ((pos = source.find(searchStr, pos)) != std::string::npos)
    {
        size_t stageStart = pos + searchStr.size();
        size_t stageEnd = source.find("\"", stageStart);
        if (stageEnd == std::string::npos) break;
        std::string stage = source.substr(stageStart, stageEnd - stageStart);

        size_t attrEnd = source.find(")]", pos);
        if (attrEnd == std::string::npos) break;
        attrEnd += 2;

        size_t fnStart = source.find_first_not_of(" \t\r\n", attrEnd);
        if (fnStart == std::string::npos) break;
        size_t spaceAfterType = source.find(' ', fnStart);
        if (spaceAfterType == std::string::npos) break;
        size_t nameStart = spaceAfterType + 1;
        size_t nameEnd = source.find_first_of("( \t", nameStart);
        if (nameEnd == std::string::npos) break;
        std::string epName = source.substr(nameStart, nameEnd - nameStart);

        if (!epName.empty())
            args += " -entry " + epName + " -stage " + stage;

        pos = attrEnd;
    }
    return args;
}

// Patch known Slang HLSL codegen bugs in the emitted source.
inline void PatchSlangHLSL(std::string& hlsl)
{
    // Slang bug: emits "Type var = obj.GetAttributes();" (no template, return-value form).
    // DXC requires the out-parameter form: "Type var; obj.GetAttributes(var);".
    const std::string pattern = "BuiltInTriangleIntersectionAttributes ";
    size_t pos = 0;
    while ((pos = hlsl.find(pattern, pos)) != std::string::npos)
    {
        size_t varStart = pos + pattern.size();
        size_t varEnd = hlsl.find_first_of(" =;", varStart);
        if (varEnd == std::string::npos) { pos = varStart; continue; }
        std::string varName = hlsl.substr(varStart, varEnd - varStart);

        std::string assignPat = varName + " = ";
        size_t assignPos = hlsl.find(assignPat, varStart);
        if (assignPos == std::string::npos || assignPos > varStart + varName.size() + 5)
            { pos = varStart; continue; }

        size_t dotPos = assignPos + assignPat.size();
        size_t getAttrPos = hlsl.find(".GetAttributes()", dotPos);
        if (getAttrPos == std::string::npos || getAttrPos > dotPos + 128)
            { pos = varStart; continue; }

        std::string objName = hlsl.substr(dotPos, getAttrPos - dotPos);
        size_t semiPos = hlsl.find(';', getAttrPos);
        if (semiPos == std::string::npos) { pos = varStart; continue; }

        std::string replacement = pattern + varName + "; " + objName + ".GetAttributes(" + varName + ");";
        hlsl.replace(pos, semiPos + 1 - pos, replacement);
        pos += replacement.size();
    }
}

inline IDxcBlob* CompileSlangShader(
    LPCWSTR fileName,
    LPCWSTR entryPoint,
    const char* slangProfile,
    LPCWSTR dxcProfile = L"lib_6_9")
{
    // --- Convert wide strings to narrow (UTF-8) ------------------------------
    auto WideToNarrow = [](const std::wstring& w) -> std::string {
        if (w.empty()) return {};
        int sz = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, nullptr, 0, nullptr, nullptr);
        std::string s(sz, 0);
        WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, s.data(), sz, nullptr, nullptr);
        while (!s.empty() && s.back() == '\0') s.pop_back();
        return s;
    };

    std::string cleanFileName = WideToNarrow(fileName);
    std::string sEntry = WideToNarrow(entryPoint ? entryPoint : L"");

    std::string moduleName = cleanFileName;
    for (const auto& ext : { ".slang", ".hlsl" })
    {
        std::string e(ext);
        if (moduleName.size() > e.size() &&
            moduleName.compare(moduleName.size() - e.size(), e.size(), e) == 0)
        { moduleName.erase(moduleName.size() - e.size()); break; }
    }

    // --- Step 1: slangc -> HLSL ----------------------------------------------
    // slangc requires each -entry to be followed by its own -o.
    // For library targets (sEntry empty) we scan [shader("...")] attributes
    // and compile each entry to a separate HLSL file, then concatenate them.
    std::string hlslPath = moduleName + "_slang_hlsl.hlsl";

    struct EntryInfo { std::string name; std::string stage; };
    std::vector<EntryInfo> entries;

    if (!sEntry.empty())
    {
        // Single entry point specified by caller (CS, work-graph, etc.)
        entries.push_back({ sEntry, "" });
    }
    else
    {
        // Library target — scan source for [shader("...")] annotated entry points
        std::string source = ReadFileToString(cleanFileName);
        const std::string searchStr = "[shader(\"";
        size_t pos = 0;
        while ((pos = source.find(searchStr, pos)) != std::string::npos)
        {
            size_t stageStart = pos + searchStr.size();
            size_t stageEnd = source.find("\"", stageStart);
            if (stageEnd == std::string::npos) break;
            std::string stage = source.substr(stageStart, stageEnd - stageStart);

            size_t attrEnd = source.find(")]", pos);
            if (attrEnd == std::string::npos) break;
            attrEnd += 2;

            size_t fnStart = source.find_first_not_of(" \t\r\n", attrEnd);
            if (fnStart == std::string::npos) break;
            size_t spaceAfterType = source.find(' ', fnStart);
            if (spaceAfterType == std::string::npos) break;
            size_t nameStart = spaceAfterType + 1;
            size_t nameEnd = source.find_first_of("( \t", nameStart);
            if (nameEnd == std::string::npos) break;
            std::string epName = source.substr(nameStart, nameEnd - nameStart);

            if (!epName.empty())
                entries.push_back({ epName, stage });
            pos = attrEnd;
        }
        if (entries.empty())
            throw std::logic_error("No entry points found in " + cleanFileName);
    }

    std::string hlslSource;

    if (entries.size() == 1)
    {
        // Single entry point — one slangc invocation, one -o
        std::string cmd = "slangc.exe " + cleanFileName +
            " -target hlsl -profile " + std::string(slangProfile) + " -DMAX_REGS=96" +
            " -entry " + entries[0].name;
        if (!entries[0].stage.empty())
            cmd += " -stage " + entries[0].stage;
        cmd += " -o \"" + hlslPath + "\"";

        OutputDebugStringA(("slangc: " + cmd + "\n").c_str());
        auto [slangOut, slangExit] = RunProcess(cmd);
        if (!slangOut.empty())
            OutputDebugStringA(("slangc output:\n" + slangOut).c_str());
        if (slangExit != 0)
        {
            std::string errMsg = "slangc failed for " + cleanFileName + " (exit " + std::to_string(slangExit) + ")\n" + slangOut;
            MessageBoxA(nullptr, errMsg.c_str(), "Slang Compilation Failed", MB_OK | MB_ICONERROR);
            throw std::logic_error(errMsg);
        }
        hlslSource = ReadFileToString(hlslPath);
    }
    else
    {
        // Multiple entry points — slangc can't handle multiple -entry with HLSL target
        // in a single invocation. Run slangc separately for each entry point, then
        // concatenate: use the first file as-is (it has all shared declarations),
        // and extract just the entry-point function from subsequent files.
        std::vector<std::string> perEntryPaths;
        for (size_t i = 0; i < entries.size(); ++i)
        {
            std::string epHlsl = moduleName + "_ep" + std::to_string(i) + "_slang_hlsl.hlsl";
            perEntryPaths.push_back(epHlsl);

            std::string cmd = "slangc.exe " + cleanFileName +
                " -target hlsl -profile " + std::string(slangProfile) + " -DMAX_REGS=96" +
                " -entry " + entries[i].name;
            if (!entries[i].stage.empty())
                cmd += " -stage " + entries[i].stage;
            cmd += " -o \"" + epHlsl + "\"";

            OutputDebugStringA(("slangc: " + cmd + "\n").c_str());
            auto [slangOut, slangExit] = RunProcess(cmd);
            if (!slangOut.empty())
                OutputDebugStringA(("slangc output:\n" + slangOut).c_str());
            if (slangExit != 0)
            {
                std::string errMsg = "slangc failed for " + cleanFileName + " entry " + entries[i].name +
                    " (exit " + std::to_string(slangExit) + ")\n" + slangOut;
                MessageBoxA(nullptr, errMsg.c_str(), "Slang Compilation Failed", MB_OK | MB_ICONERROR);
                throw std::logic_error(errMsg);
            }
        }

        // Concatenate: first file has all shared declarations + first entry point.
        // From subsequent files, extract only the entry-point function (at the end).
        std::string firstSrc = ReadFileToString(perEntryPaths[0]);
        hlslSource = firstSrc;
        for (size_t i = 1; i < perEntryPaths.size(); ++i)
        {
            std::string epSrc = ReadFileToString(perEntryPaths[i]);
            // Find the entry-point function by looking for its [shader(...)] attribute
            // in the emitted HLSL. Slang emits it near the end of the file.
            std::string marker = "[shader(\"" + entries[i].stage + "\")]";
            size_t markerPos = epSrc.rfind(marker);
            if (markerPos != std::string::npos)
            {
                // Include from the start of the line containing the attribute
                size_t lineStart = epSrc.rfind('\n', markerPos);
                lineStart = (lineStart == std::string::npos) ? markerPos : lineStart;
                hlslSource += "\n" + epSrc.substr(lineStart);
            }
            else
            {
                // Fallback: just append the whole file
                hlslSource += "\n// --- entry point " + entries[i].name + " ---\n" + epSrc;
            }
        }

        // Write combined HLSL for debugging
        { std::ofstream ofs(hlslPath, std::ios::binary); if (ofs) ofs.write(hlslSource.data(), hlslSource.size()); }
    }

    if (hlslSource.empty())
        throw std::logic_error("slangc produced empty HLSL for " + cleanFileName);

    PatchSlangHLSL(hlslSource);

    // Write patched HLSL back for debugging
    { std::ofstream ofs(hlslPath, std::ios::binary); if (ofs) ofs.write(hlslSource.data(), hlslSource.size()); }

    OutputDebugStringA(("slangc: " + std::to_string(hlslSource.size()) + " bytes HLSL for " + cleanFileName + "\n").c_str());

    // --- Step 3: DXC API -> DXIL ---------------------------------------------
    Microsoft::WRL::ComPtr<IDxcUtils> dxcUtils;
    Microsoft::WRL::ComPtr<IDxcCompiler3> dxcCompiler;
    ThrowIfFailed(DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&dxcUtils)));
    ThrowIfFailed(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&dxcCompiler)));

    DxcBuffer sourceBuffer = { hlslSource.data(), hlslSource.size(), DXC_CP_UTF8 };

    std::vector<LPCWSTR> dxcArgs;
    dxcArgs.push_back(L"-T");
    dxcArgs.push_back(dxcProfile);
    if (!sEntry.empty())
    {
        dxcArgs.push_back(L"-E");
        dxcArgs.push_back(entryPoint);
    }
    dxcArgs.push_back(L"-enable-16bit-types");
    dxcArgs.push_back(L"-O3");
    dxcArgs.push_back(L"-HV");
    dxcArgs.push_back(L"2021");
    dxcArgs.push_back(L"-Wno-unknown-attributes");

    Microsoft::WRL::ComPtr<IDxcResult> dxcResult;
    HRESULT hr = dxcCompiler->Compile(&sourceBuffer, dxcArgs.data(), (UINT32)dxcArgs.size(),
                                       nullptr, IID_PPV_ARGS(&dxcResult));

    Microsoft::WRL::ComPtr<IDxcBlobUtf8> dxcErrors;
    dxcResult->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&dxcErrors), nullptr);
    if (dxcErrors && dxcErrors->GetStringLength() > 0)
        OutputDebugStringA(("DXC output for " + cleanFileName + ":\n" +
            std::string(dxcErrors->GetStringPointer(), dxcErrors->GetStringLength())).c_str());

    HRESULT statusHR;
    dxcResult->GetStatus(&statusHR);
    if (FAILED(hr) || FAILED(statusHR))
    {
        std::string errMsg = "DXC failed for " + cleanFileName;
        if (dxcErrors && dxcErrors->GetStringLength() > 0)
            errMsg += "\n" + std::string(dxcErrors->GetStringPointer(), dxcErrors->GetStringLength());
        MessageBoxA(nullptr, errMsg.c_str(), "DXC Compilation Failed", MB_OK | MB_ICONERROR);
        throw std::logic_error(errMsg);
    }

    Microsoft::WRL::ComPtr<IDxcBlob> dxilBlob;
    dxcResult->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&dxilBlob), nullptr);
    if (!dxilBlob || dxilBlob->GetBufferSize() == 0)
        throw std::logic_error("DXC produced empty DXIL for " + cleanFileName);

    OutputDebugStringA(("Compiled " + cleanFileName + " -> " +
        std::to_string(dxilBlob->GetBufferSize()) + " bytes DXIL\n").c_str());

    return new SlangDxilBlob(dxilBlob->GetBufferPointer(), dxilBlob->GetBufferSize());
}

//--------------------------------------------------------------------------------------------------
// Direct DXIL compilation: .slang -> slangc (-target dxil) -> DXIL blob
//
// Uses Slang's own LLVM backend (slang-llvm.dll) to emit DXIL directly,
// bypassing the HLSL intermediate step. Required for neural shader features
// (CoopVec, neural inference, NeuralNetworkTexture, etc.) that have no
// standard HLSL representation.
//--------------------------------------------------------------------------------------------------
inline IDxcBlob* CompileSlangDirectDXIL(
    LPCWSTR fileName,
    LPCWSTR entryPoint,
    const char* slangProfile)
{
    auto WideToNarrow = [](const std::wstring& w) -> std::string {
        if (w.empty()) return {};
        int sz = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, nullptr, 0, nullptr, nullptr);
        std::string s(sz, 0);
        WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, s.data(), sz, nullptr, nullptr);
        while (!s.empty() && s.back() == '\0') s.pop_back();
        return s;
    };

    std::string cleanFileName = WideToNarrow(fileName);
    std::string sEntry = WideToNarrow(entryPoint ? entryPoint : L"");

    std::string moduleName = cleanFileName;
    for (const auto& ext : { ".slang", ".hlsl" })
    {
        std::string e(ext);
        if (moduleName.size() > e.size() &&
            moduleName.compare(moduleName.size() - e.size(), e.size(), e) == 0)
        { moduleName.erase(moduleName.size() - e.size()); break; }
    }

    std::string dxilPath = moduleName + "_slang.dxil";

    // Build slangc command for direct DXIL emission
    std::string cmd = "slangc.exe " + cleanFileName +
        " -target dxil -profile " + std::string(slangProfile) + " -DMAX_REGS=96";

    // For library targets (sEntry empty), scan for entry points
    if (!sEntry.empty())
    {
        cmd += " -entry " + sEntry;
    }
    else
    {
        // Library target — scan for [shader("...")] entry points
        std::string source = ReadFileToString(cleanFileName);
        const std::string searchStr = "[shader(\"";
        size_t pos = 0;
        bool foundAny = false;
        while ((pos = source.find(searchStr, pos)) != std::string::npos)
        {
            size_t stageStart = pos + searchStr.size();
            size_t stageEnd = source.find("\"", stageStart);
            if (stageEnd == std::string::npos) break;
            std::string stage = source.substr(stageStart, stageEnd - stageStart);

            size_t attrEnd = source.find(")]", pos);
            if (attrEnd == std::string::npos) break;
            attrEnd += 2;

            size_t fnStart = source.find_first_not_of(" \t\r\n", attrEnd);
            if (fnStart == std::string::npos) break;
            size_t spaceAfterType = source.find(' ', fnStart);
            if (spaceAfterType == std::string::npos) break;
            size_t nameStart = spaceAfterType + 1;
            size_t nameEnd = source.find_first_of("( \t", nameStart);
            if (nameEnd == std::string::npos) break;
            std::string epName = source.substr(nameStart, nameEnd - nameStart);

            if (!epName.empty())
            {
                cmd += " -entry " + epName + " -stage " + stage;
                foundAny = true;
            }
            pos = attrEnd;
        }
        if (!foundAny)
            throw std::logic_error("No entry points found in " + cleanFileName);
    }

    cmd += " -o \"" + dxilPath + "\"";

    OutputDebugStringA(("slangc (direct DXIL): " + cmd + "\n").c_str());
    auto [slangOut, slangExit] = RunProcess(cmd);
    if (!slangOut.empty())
        OutputDebugStringA(("slangc output:\n" + slangOut).c_str());
    if (slangExit != 0)
    {
        std::string errMsg = "slangc direct DXIL failed for " + cleanFileName +
            " (exit " + std::to_string(slangExit) + ")\n" + slangOut;
        MessageBoxA(nullptr, errMsg.c_str(), "Slang Neural Compilation Failed", MB_OK | MB_ICONERROR);
        throw std::logic_error(errMsg);
    }

    // Read the DXIL binary blob
    std::ifstream dxilFile(dxilPath, std::ios::binary | std::ios::ate);
    if (!dxilFile.is_open())
        throw std::logic_error("slangc produced no DXIL file for " + cleanFileName);

    size_t dxilSize = static_cast<size_t>(dxilFile.tellg());
    dxilFile.seekg(0, std::ios::beg);
    std::vector<uint8_t> dxilData(dxilSize);
    dxilFile.read(reinterpret_cast<char*>(dxilData.data()), dxilSize);
    dxilFile.close();

    if (dxilData.empty())
        throw std::logic_error("slangc produced empty DXIL for " + cleanFileName);

    OutputDebugStringA(("Compiled (neural) " + cleanFileName + " -> " +
        std::to_string(dxilData.size()) + " bytes DXIL\n").c_str());

    return new SlangDxilBlob(dxilData.data(), dxilData.size());
}

//--------------------------------------------------------------------------------------------------
// Public API — mirrors the DXC wrappers in DXRHelper.h
//--------------------------------------------------------------------------------------------------

// ── Standard path: .slang → HLSL → DXC → DXIL (for existing shaders) ──

inline IDxcBlob* CompileSlangLibrary(LPCWSTR fileName)
{
    return CompileSlangShader(fileName, L"", "lib_6_9+ser_hlsl_native", L"lib_6_9");
}

inline Microsoft::WRL::ComPtr<IDxcBlob> CompileSlangCS(LPCWSTR fileName, LPCWSTR entryPoint = L"main")
{
    return CompileSlangShader(fileName, entryPoint, "cs_6_9", L"cs_6_9");
}

inline Microsoft::WRL::ComPtr<IDxcBlob> CompileSlangWG(LPCWSTR fileName, LPCWSTR entryPoint = L"main")
{
    return CompileSlangShader(fileName, entryPoint, "lib_6_9", L"lib_6_9");
}

// ── Neural path: .slang → slangc direct DXIL (for neural shader features) ──
// Requires slang-llvm.dll at runtime. Use these for shaders that use CoopVec,
// neural inference, NeuralNetworkTexture, or other Slang-specific neural constructs.

inline IDxcBlob* CompileSlangNeuralLibrary(LPCWSTR fileName)
{
    return CompileSlangDirectDXIL(fileName, L"", "lib_6_9");
}

inline Microsoft::WRL::ComPtr<IDxcBlob> CompileSlangNeuralCS(LPCWSTR fileName, LPCWSTR entryPoint = L"main")
{
    return CompileSlangDirectDXIL(fileName, entryPoint, "sm_6_9");
}
