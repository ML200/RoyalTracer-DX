#pragma once
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <string>
#include <vector>
#include <algorithm>
#include <memory>
#include <sstream>
#include <ctime>
#include <filesystem>
#include <Windows.h>

// Every TU must agree on this, or the linker may keep the stubs.
#ifndef ENABLE_D3D12_DIAGNOSTICS
#define ENABLE_D3D12_DIAGNOSTICS 1
#endif

#if ENABLE_D3D12_DIAGNOSTICS
#pragma comment(lib, "dxguid.lib")
#endif

namespace dxdiag {
#if ENABLE_D3D12_DIAGNOSTICS
using Microsoft::WRL::ComPtr;

inline std::wstring CrashLogPath() {
    wchar_t exePath[MAX_PATH] = {0};
    GetModuleFileNameW(nullptr, exePath, MAX_PATH);
    std::wstring p(exePath);
    const size_t slash = p.find_last_of(L"\\/");
    if (slash != std::wstring::npos)
        p.resize(slash + 1);
    else
        p.clear();
    p += L"crash_dxdiag.log";
    return p;
}
inline std::wofstream& CrashLogFile() {
    static std::wofstream f([] {
        const std::wstring path = CrashLogPath();
        std::wofstream tmp(path.c_str(), std::ios::out | std::ios::trunc);
        if (tmp.is_open()) {
            SYSTEMTIME st;
            GetLocalTime(&st);
            tmp << L"=== dxdiag startup banner " << st.wYear << L"-" << st.wMonth << L"-" << st.wDay << L" " << st.wHour
                << L":" << st.wMinute << L":" << st.wSecond << L" log path: " << path << L" ===\n";
            tmp.flush();
        }
        return tmp;
    }());
    return f;
}
inline void CrashLog(const std::wstring& s) {
    auto& f = CrashLogFile();
    if (f.is_open()) {
        f << s;
        f.flush();
    }
    std::wcerr << s;
    std::wcerr.flush();
    OutputDebugStringW(s.c_str());
}

template <typename... Args> inline void CrashLogF(const wchar_t* fmt, Args... args) {
    wchar_t buf[1024];
    _snwprintf_s(buf, _TRUNCATE, fmt, args...);
    CrashLog(std::wstring(buf));
}

inline ComPtr<ID3D12InfoQueue> g_infoQ;
inline ComPtr<ID3D12DeviceRemovedExtendedData> g_dred;

LONG WINAPI CrashExceptionFilter(EXCEPTION_POINTERS* ep);
// Logs the caller's symbolized stack (CrashHandler.cpp).
void LogCallStack(const wchar_t* title);

// Runs inside the failing D3D12 call, on its thread.
inline void CALLBACK InfoQueueCallback(D3D12_MESSAGE_CATEGORY, D3D12_MESSAGE_SEVERITY severity, D3D12_MESSAGE_ID id,
                                       LPCSTR description, void*) {
    if (severity > D3D12_MESSAGE_SEVERITY_ERROR)
        return;
    static std::mutex guard;
    static std::unordered_set<int> traced;
    std::lock_guard<std::mutex> lock(guard);
    if (!traced.insert((int)id).second)
        return;
    CrashLogF(L"[DX] call stack of the first message %d (%.80hs...):\n", (int)id, description ? description : "");
    LogCallStack(L"");
}
inline void InstallCrashHandler() {
    SetUnhandledExceptionFilter(&CrashExceptionFilter);
}

// Nsight Aftermath GPU crash dumps, loaded at run time. Decode with:
//   nv-aftermath-format -D aftermath -B aftermath gpu_crash_<pid>.nv-gpudmp
namespace aftermath {
// Mirrors GFSDK_Aftermath.h / GFSDK_Aftermath_GpuCrashDump.h (API 2.27).
constexpr uint32_t kVersionApi = 0x21B;
constexpr uint32_t kWatchDx = 0x1;
constexpr uint32_t kDeferDebugInfoCallbacks = 0x1;
constexpr uint32_t kResourceTracking = 0x2, kShaderDebugInfo = 0x8, kShaderErrorReporting = 0x10;
constexpr uint32_t kStatusCollectingFailed = 2, kStatusFinished = 4, kStatusUnknown = 5;
using Result = uint32_t;
inline bool Ok(Result r) { return (r & 0xFFF00000u) != 0xBAD00000u; }
struct DebugInfoId { uint64_t id[2]; };
using DataCb = void(__cdecl*)(const void*, uint32_t, void*);
using AddDescriptionFn = void(__cdecl*)(uint32_t, const char*);
using DescriptionCb = void(__cdecl*)(AddDescriptionFn, void*);
using EnableFn = Result(__cdecl*)(uint32_t, uint32_t, uint32_t, DataCb, DataCb, DescriptionCb, void*, void*);
using InitDx12Fn = Result(__cdecl*)(uint32_t, uint32_t, ID3D12Device*);
using StatusFn = Result(__cdecl*)(uint32_t*);
using DebugInfoIdFn = Result(__cdecl*)(uint32_t, const void*, uint32_t, DebugInfoId*);

struct State {
    InitDx12Fn initDx12 = nullptr;
    StatusFn status = nullptr;
    DebugInfoIdFn debugInfoId = nullptr;
    bool enabled = false;
    std::wstring outDir;
    std::mutex files;
};
inline State& Get() {
    static State s;
    return s;
}

inline void WriteFile(const std::wstring& path, const void* data, uint32_t size) {
    std::ofstream f(path.c_str(), std::ios::binary | std::ios::trunc);
    f.write(static_cast<const char*>(data), size);
}

inline void __cdecl OnCrashDump(const void* data, uint32_t size, void*) {
    std::lock_guard<std::mutex> lock(Get().files);
    std::wstring path = CrashLogPath();
    path.resize(path.find_last_of(L"\\/") + 1);
    path += L"gpu_crash_" + std::to_wstring(GetCurrentProcessId()) + L".nv-gpudmp";
    WriteFile(path, data, size);
    CrashLogF(L"[aftermath] GPU crash dump written: %ls (%u bytes)\n", path.c_str(), size);
}

inline void __cdecl OnShaderDebugInfo(const void* data, uint32_t size, void*) {
    State& s = Get();
    DebugInfoId id{};
    if (!s.debugInfoId || !Ok(s.debugInfoId(kVersionApi, data, size, &id)))
        return;
    wchar_t name[64];
    swprintf_s(name, L"shader-%016llX%016llX.nvdbg", (unsigned long long)id.id[0], (unsigned long long)id.id[1]);
    std::lock_guard<std::mutex> lock(s.files);
    WriteFile(s.outDir + name, data, size);
}

inline void __cdecl OnDescription(AddDescriptionFn add, void*) {
    add(0x1u, "RoyalTracer Pathtracer");
}

// Newest Nsight Graphics install's Aftermath DLL.
inline std::wstring FindLibrary() {
    wchar_t overridePath[MAX_PATH] = {};
    if (GetEnvironmentVariableW(L"RT_AFTERMATH_DLL", overridePath, MAX_PATH) > 0)
        return overridePath;
    namespace fs = std::filesystem;
    std::error_code ec;
    std::wstring best;
    const fs::path root = L"C:\\Program Files\\NVIDIA Corporation";
    for (const auto& install : fs::directory_iterator(root, ec)) {
        if (install.path().filename().wstring().rfind(L"Nsight Graphics", 0) != 0)
            continue;
        for (const auto& sdk : fs::directory_iterator(install.path() / L"SDKs" / L"NsightAftermathSDK", ec)) {
            const fs::path dll = sdk.path() / L"lib" / L"x64" / L"GFSDK_Aftermath_Lib.x64.dll";
            if (fs::exists(dll, ec) && dll.wstring() > best)
                best = dll.wstring();
        }
    }
    return best;
}

// Before the device is created.
inline void Enable() {
    if (GetEnvironmentVariableA("RT_NO_AFTERMATH", nullptr, 0) > 0) {
        CrashLog(L"[aftermath] skipped (RT_NO_AFTERMATH set)\n");
        return;
    }
    const std::wstring dll = FindLibrary();
    HMODULE lib = dll.empty() ? nullptr : LoadLibraryW(dll.c_str());
    if (!lib) {
        CrashLog(L"[aftermath] library not found: install Nsight Graphics or set RT_AFTERMATH_DLL\n");
        return;
    }
    State& s = Get();
    const auto enable = reinterpret_cast<EnableFn>(GetProcAddress(lib, "GFSDK_Aftermath_EnableGpuCrashDumps"));
    s.initDx12 = reinterpret_cast<InitDx12Fn>(GetProcAddress(lib, "GFSDK_Aftermath_DX12_Initialize"));
    s.status = reinterpret_cast<StatusFn>(GetProcAddress(lib, "GFSDK_Aftermath_GetCrashDumpStatus"));
    s.debugInfoId =
        reinterpret_cast<DebugInfoIdFn>(GetProcAddress(lib, "GFSDK_Aftermath_GetShaderDebugInfoIdentifier"));
    if (!enable || !s.initDx12 || !s.status) {
        CrashLogF(L"[aftermath] %ls lacks the crash dump entry points\n", dll.c_str());
        return;
    }
    s.outDir = CrashLogPath();
    s.outDir.resize(s.outDir.find_last_of(L"\\/") + 1);
    s.outDir += L"aftermath\\";
    std::error_code ec;
    std::filesystem::create_directories(s.outDir, ec);
    const Result r = enable(kVersionApi, kWatchDx, kDeferDebugInfoCallbacks, &OnCrashDump, &OnShaderDebugInfo,
                            &OnDescription, nullptr, nullptr);
    s.enabled = Ok(r);
    if (s.enabled)
        SetEnvironmentVariableW(L"RT_AFTERMATH_SHADER_DIR", s.outDir.c_str());
    CrashLogF(L"[aftermath] %ls: %ls (0x%08X)\n", s.enabled ? L"GPU crash dumps on" : L"enable failed", dll.c_str(),
              (unsigned)r);
}

// After device creation, on the native device.
inline void InitDevice(ID3D12Device* device) {
    State& s = Get();
    if (!s.enabled)
        return;
    const Result r = s.initDx12(kVersionApi, kResourceTracking | kShaderDebugInfo | kShaderErrorReporting, device);
    CrashLogF(L"[aftermath] DX12 initialize: 0x%08X\n", (unsigned)r);
}

// After device removal; the dump is written on a driver thread.
inline void WaitForDump() {
    State& s = Get();
    if (!s.enabled)
        return;
    uint32_t status = 0;
    for (int i = 0; i < 1500; ++i) {
        if (!Ok(s.status(&status)) || status == kStatusFinished || status == kStatusCollectingFailed ||
            status == kStatusUnknown)
            break;
        Sleep(10);
    }
    CrashLogF(L"[aftermath] crash dump status %u (4 = written, 2 = collection failed)\n", status);
}
} // namespace aftermath

// Off by default: validation slowdown can trigger false TDRs.
#ifndef DXDIAG_ENABLE_DEBUG_LAYER
#define DXDIAG_ENABLE_DEBUG_LAYER 0
#endif

// Before device creation.
inline void EnableDebugLayerAndDred() {
    CrashLog(L"[dxdiag] EnableDebugLayerAndDred reached, diagnostic plumbing live\n");
    aftermath::Enable();

#if DXDIAG_ENABLE_DEBUG_LAYER
    // Opt-out: slow, and preview layers crash in CreateStateObject.
    char noDebugLayer[8] = {};
    if (GetEnvironmentVariableA("RT_NO_DEBUG_LAYER", noDebugLayer, sizeof(noDebugLayer)) > 0 && noDebugLayer[0] != '0') {
        CrashLog(L"[DX]  Debug-layer SKIPPED (RT_NO_DEBUG_LAYER set)\n");
    } else {
        ComPtr<ID3D12Debug> dbg;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dbg)))) {
            dbg->EnableDebugLayer();
            CrashLog(L"[DX]  Debug-layer enabled (heavy validation, slow)\n");
        }
    }
#else
    CrashLog(L"[DX]  Debug-layer SKIPPED (DXDIAG_ENABLE_DEBUG_LAYER=0)\n");
#endif

    ComPtr<ID3D12DeviceRemovedExtendedDataSettings> dredSet;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dredSet)))) {
        dredSet->SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
        dredSet->SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
        // Contexts carry the profiler's per-pass markers.
        ComPtr<ID3D12DeviceRemovedExtendedDataSettings1> dredSet1;
        if (SUCCEEDED(dredSet.As(&dredSet1)))
            dredSet1->SetBreadcrumbContextEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
        CrashLog(L"[DX]  DRED enabled (cheap, used for TDR breadcrumbs)\n");
    }
}

inline void HookDevice(ID3D12Device* device) {
    device->QueryInterface(IID_PPV_ARGS(&g_infoQ));
    device->QueryInterface(IID_PPV_ARGS(&g_dred));
    aftermath::InitDevice(device);

#if DXDIAG_ENABLE_DEBUG_LAYER
    if (g_infoQ) {
        if (IsDebuggerPresent()) {
            g_infoQ->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, true);
            g_infoQ->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, true);
        }
        g_infoQ->SetMessageCountLimit(4096);
        ComPtr<ID3D12InfoQueue1> infoQ1;
        if (SUCCEEDED(g_infoQ.As(&infoQ1))) {
            DWORD cookie = 0;
            infoQ1->RegisterMessageCallback(&InfoQueueCallback, D3D12_MESSAGE_CALLBACK_FLAG_NONE, nullptr, &cookie);
        }
    }
#endif
}

// Logs each message once; frequent IDs are denied at source.
inline void DumpNewMessages() {
    if (!g_infoQ)
        return;

    static std::unordered_map<std::string, uint64_t> seen;
    static std::vector<D3D12_MESSAGE_ID> denied;
    static uint64_t suppressed = 0, drains = 0;
    constexpr uint64_t kRepeatsBeforeDeny = 8;
    constexpr uint64_t kSummaryEveryDrains = 600;

    const UINT64 nMsg = g_infoQ->GetNumStoredMessagesAllowedByRetrievalFilter();
    bool denyListGrew = false;
    for (UINT64 i = 0; i < nMsg; ++i) {
        SIZE_T sz = 0;
        g_infoQ->GetMessage(i, nullptr, &sz);

        std::unique_ptr<uint8_t[]> blob(new uint8_t[sz]);
        D3D12_MESSAGE* msg = reinterpret_cast<D3D12_MESSAGE*>(blob.get());

        g_infoQ->GetMessage(i, msg, &sz);
        if (msg->Severity > D3D12_MESSAGE_SEVERITY_WARNING)
            continue;

        const std::string key = std::to_string((int)msg->ID) + ":" + (msg->pDescription ? msg->pDescription : "");
        const uint64_t count = ++seen[key];
        if (count == 1) {
            CrashLogF(L"[DX] %hs\n", msg->pDescription ? msg->pDescription : "<no description>");
        } else {
            ++suppressed;
            if (count == kRepeatsBeforeDeny && std::find(denied.begin(), denied.end(), msg->ID) == denied.end()) {
                denied.push_back(msg->ID);
                denyListGrew = true;
                CrashLogF(L"[DX] message %d repeated %llu times; further copies are not stored\n", (int)msg->ID,
                          (unsigned long long)kRepeatsBeforeDeny);
            }
        }
    }
    g_infoQ->ClearStoredMessages();

    if (denyListGrew) {
        D3D12_INFO_QUEUE_FILTER filter = {};
        filter.DenyList.NumIDs = (UINT)denied.size();
        filter.DenyList.pIDList = denied.data();
        g_infoQ->ClearStorageFilter();
        g_infoQ->AddStorageFilterEntries(&filter);
    }
    if (++drains % kSummaryEveryDrains == 0 && suppressed)
        CrashLogF(L"[DX] %llu repeated debug-layer messages suppressed so far\n", (unsigned long long)suppressed);
}

inline const char* BreadcrumbOpName(D3D12_AUTO_BREADCRUMB_OP op) {
    switch (op) {
    case D3D12_AUTO_BREADCRUMB_OP_SETMARKER:
        return "SetMarker";
    case D3D12_AUTO_BREADCRUMB_OP_BEGINEVENT:
        return "BeginEvent";
    case D3D12_AUTO_BREADCRUMB_OP_ENDEVENT:
        return "EndEvent";
    case D3D12_AUTO_BREADCRUMB_OP_DRAWINSTANCED:
        return "DrawInstanced";
    case D3D12_AUTO_BREADCRUMB_OP_DRAWINDEXEDINSTANCED:
        return "DrawIndexedInstanced";
    case D3D12_AUTO_BREADCRUMB_OP_EXECUTEINDIRECT:
        return "ExecuteIndirect";
    case D3D12_AUTO_BREADCRUMB_OP_DISPATCH:
        return "Dispatch";
    case D3D12_AUTO_BREADCRUMB_OP_COPYBUFFERREGION:
        return "CopyBufferRegion";
    case D3D12_AUTO_BREADCRUMB_OP_COPYTEXTUREREGION:
        return "CopyTextureRegion";
    case D3D12_AUTO_BREADCRUMB_OP_COPYRESOURCE:
        return "CopyResource";
    case D3D12_AUTO_BREADCRUMB_OP_COPYTILES:
        return "CopyTiles";
    case D3D12_AUTO_BREADCRUMB_OP_RESOLVESUBRESOURCE:
        return "ResolveSubresource";
    case D3D12_AUTO_BREADCRUMB_OP_CLEARRENDERTARGETVIEW:
        return "ClearRenderTargetView";
    case D3D12_AUTO_BREADCRUMB_OP_CLEARUNORDEREDACCESSVIEW:
        return "ClearUAV";
    case D3D12_AUTO_BREADCRUMB_OP_CLEARDEPTHSTENCILVIEW:
        return "ClearDepthStencilView";
    case D3D12_AUTO_BREADCRUMB_OP_RESOURCEBARRIER:
        return "ResourceBarrier";
    case D3D12_AUTO_BREADCRUMB_OP_EXECUTEBUNDLE:
        return "ExecuteBundle";
    case D3D12_AUTO_BREADCRUMB_OP_PRESENT:
        return "Present";
    case D3D12_AUTO_BREADCRUMB_OP_RESOLVEQUERYDATA:
        return "ResolveQueryData";
    case D3D12_AUTO_BREADCRUMB_OP_BEGINSUBMISSION:
        return "BeginSubmission";
    case D3D12_AUTO_BREADCRUMB_OP_ENDSUBMISSION:
        return "EndSubmission";
    case D3D12_AUTO_BREADCRUMB_OP_DECODEFRAME:
        return "DecodeFrame";
    case D3D12_AUTO_BREADCRUMB_OP_PROCESSFRAMES:
        return "ProcessFrames";
    case D3D12_AUTO_BREADCRUMB_OP_ATOMICCOPYBUFFERUINT:
        return "AtomicCopyBufferUint";
    case D3D12_AUTO_BREADCRUMB_OP_ATOMICCOPYBUFFERUINT64:
        return "AtomicCopyBufferUint64";
    case D3D12_AUTO_BREADCRUMB_OP_RESOLVESUBRESOURCEREGION:
        return "ResolveSubresourceRegion";
    case D3D12_AUTO_BREADCRUMB_OP_WRITEBUFFERIMMEDIATE:
        return "WriteBufferImmediate";
    case D3D12_AUTO_BREADCRUMB_OP_DECODEFRAME1:
        return "DecodeFrame1";
    case D3D12_AUTO_BREADCRUMB_OP_SETPROTECTEDRESOURCESESSION:
        return "SetProtectedResourceSession";
    case D3D12_AUTO_BREADCRUMB_OP_DECODEFRAME2:
        return "DecodeFrame2";
    case D3D12_AUTO_BREADCRUMB_OP_PROCESSFRAMES1:
        return "ProcessFrames1";
    case D3D12_AUTO_BREADCRUMB_OP_BUILDRAYTRACINGACCELERATIONSTRUCTURE:
        return "BuildAccelStruct";
    case D3D12_AUTO_BREADCRUMB_OP_EMITRAYTRACINGACCELERATIONSTRUCTUREPOSTBUILDINFO:
        return "EmitAccelPostbuild";
    case D3D12_AUTO_BREADCRUMB_OP_COPYRAYTRACINGACCELERATIONSTRUCTURE:
        return "CopyAccelStruct";
    case D3D12_AUTO_BREADCRUMB_OP_DISPATCHRAYS:
        return "DispatchRays";
    case D3D12_AUTO_BREADCRUMB_OP_INITIALIZEMETACOMMAND:
        return "InitializeMetaCommand";
    case D3D12_AUTO_BREADCRUMB_OP_EXECUTEMETACOMMAND:
        return "ExecuteMetaCommand";
    case D3D12_AUTO_BREADCRUMB_OP_ESTIMATEMOTION:
        return "EstimateMotion";
    case D3D12_AUTO_BREADCRUMB_OP_RESOLVEMOTIONVECTORHEAP:
        return "ResolveMotionVectorHeap";
    case D3D12_AUTO_BREADCRUMB_OP_SETPIPELINESTATE1:
        return "SetPipelineState1";
    case D3D12_AUTO_BREADCRUMB_OP_INITIALIZEEXTENSIONCOMMAND:
        return "InitializeExtensionCommand";
    case D3D12_AUTO_BREADCRUMB_OP_EXECUTEEXTENSIONCOMMAND:
        return "ExecuteExtensionCommand";
    case D3D12_AUTO_BREADCRUMB_OP_DISPATCHMESH:
        return "DispatchMesh";
    default:
        return "Unknown";
    }
}

inline void CheckDeviceRemoved(ID3D12Device* device, int pollMs = 0) {
    HRESULT hr = device->GetDeviceRemovedReason();
    if (pollMs > 0 && SUCCEEDED(hr)) {
        const int steps = pollMs / 10;
        for (int i = 0; i < steps && SUCCEEDED(hr); ++i) {
            Sleep(10);
            hr = device->GetDeviceRemovedReason();
        }
    }
    if (SUCCEEDED(hr)) {
        return;
    }
    CrashLogF(L"\n*** dxdiag::CheckDeviceRemoved fired, GetDeviceRemovedReason = 0x%08X ***\n", (unsigned)hr);

    const wchar_t* reasonName = L"unknown";
    switch ((unsigned)hr) {
    case 0x887A0005:
        reasonName = L"DXGI_ERROR_DEVICE_REMOVED";
        break;
    case 0x887A0006:
        reasonName = L"DXGI_ERROR_DEVICE_HUNG (TDR)";
        break;
    case 0x887A0007:
        reasonName = L"DXGI_ERROR_DEVICE_RESET";
        break;
    case 0x887A0001:
        reasonName = L"DXGI_ERROR_INVALID_CALL";
        break;
    case 0x887A0020:
        reasonName = L"DXGI_ERROR_DRIVER_INTERNAL_ERROR";
        break;
    }
    CrashLogF(L"    reason: %ls\n", reasonName);

    if (!g_dred) {
        CrashLog(L"    (DRED interface not available, no breadcrumbs)\n");
    } else {
        // DRED 1.2 contexts name the failing pass.
        ComPtr<ID3D12DeviceRemovedExtendedData1> dred1;
        g_dred.As(&dred1);
        D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT1 bc1 = {};
        D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT bc = {};
        HRESULT hrBC = dred1 ? dred1->GetAutoBreadcrumbsOutput1(&bc1) : E_NOINTERFACE;
        const bool withContext = SUCCEEDED(hrBC);
        if (!withContext)
            hrBC = g_dred->GetAutoBreadcrumbsOutput(&bc);
        D3D12_DRED_PAGE_FAULT_OUTPUT pf = {};
        HRESULT hrPF = g_dred->GetPageFaultAllocationOutput(&pf);
        CrashLogF(L"    DRED breadcrumbs hr=0x%08X (contexts %ls), page fault hr=0x%08X\n", (unsigned)hrBC,
                  withContext ? L"yes" : L"no", (unsigned)hrPF);

        auto dumpNode = [&](int nodeIdx, const char* listName, UINT32 lastVal, UINT32 count,
                            const D3D12_AUTO_BREADCRUMB_OP* history, const D3D12_DRED_BREADCRUMB_CONTEXT* contexts,
                            UINT contextCount) {
            char nameBuf[256] = "<unnamed>";
            if (listName)
                strncpy_s(nameBuf, listName, _TRUNCATE);
            wchar_t wname[256];
            size_t conv = 0;
            mbstowcs_s(&conv, wname, nameBuf, _TRUNCATE);
            CrashLogF(L"  Node %d: list='%ls' completed %u/%u ops\n", nodeIdx, wname, lastVal, count);

            const wchar_t* pass = L"<no marker>";
            for (UINT c = 0; c < contextCount; ++c)
                if (contexts[c].BreadcrumbIndex <= lastVal && contexts[c].pContextString)
                    pass = contexts[c].pContextString;
            if (contextCount)
                CrashLogF(L"    last pass marker at or before the failing op: '%ls'\n", pass);

            const UINT32 first = (lastVal > 12u) ? (lastVal - 12u) : 0u;
            const UINT32 last = std::min<UINT32>(lastVal + 1u, count);
            for (UINT32 i = first; i < last; ++i) {
                const wchar_t* tag = (i == lastVal) ? L"  >>>" : L"     ";
                char opNameA[64] = {0};
                strncpy_s(opNameA, BreadcrumbOpName(history[i]), _TRUNCATE);
                wchar_t opNameW[64];
                mbstowcs_s(&conv, opNameW, opNameA, _TRUNCATE);
                const wchar_t* ctxStr = L"";
                for (UINT c = 0; c < contextCount; ++c)
                    if (contexts[c].BreadcrumbIndex == i && contexts[c].pContextString)
                        ctxStr = contexts[c].pContextString;
                CrashLogF(L"%ls op %u: %ls %ls\n", tag, i, opNameW, ctxStr);
            }
        };

        int nodeIdx = 0;
        if (withContext) {
            for (auto node = bc1.pHeadAutoBreadcrumbNode; node; node = node->pNext, ++nodeIdx)
                dumpNode(nodeIdx, node->pCommandListDebugNameA,
                         node->pLastBreadcrumbValue ? *node->pLastBreadcrumbValue : 0u, node->BreadcrumbCount,
                         node->pCommandHistory, node->pBreadcrumbContexts, node->BreadcrumbContextsCount);
        } else {
            for (auto node = bc.pHeadAutoBreadcrumbNode; node; node = node->pNext, ++nodeIdx)
                dumpNode(nodeIdx, node->pCommandListDebugNameA,
                         node->pLastBreadcrumbValue ? *node->pLastBreadcrumbValue : 0u, node->BreadcrumbCount,
                         node->pCommandHistory, nullptr, 0u);
        }
        if (nodeIdx == 0) {
            CrashLog(L"    (no breadcrumb nodes; the hang may have been outside the tracked queue)\n");
        }

        if (pf.PageFaultVA) {
            CrashLogF(L"  Page fault at GPU VA: 0x%llX\n", (unsigned long long)pf.PageFaultVA);
            int allocIdx = 0;
            for (auto a = pf.pHeadExistingAllocationNode; a && allocIdx < 8; a = a->pNext, ++allocIdx) {
                const wchar_t* objName = a->ObjectNameW ? a->ObjectNameW : L"<unnamed>";
                CrashLogF(L"    existing alloc: '%ls' type=%d\n", objName, (int)a->AllocationType);
            }
            allocIdx = 0;
            for (auto a = pf.pHeadRecentFreedAllocationNode; a && allocIdx < 8; a = a->pNext, ++allocIdx) {
                const wchar_t* objName = a->ObjectNameW ? a->ObjectNameW : L"<unnamed>";
                CrashLogF(L"    recent freed:   '%ls' type=%d\n", objName, (int)a->AllocationType);
            }
        }
    }
    CrashLog(L"*** end of crash dump ***\n");
    aftermath::WaitForDump();
    if (CrashLogFile().is_open())
        CrashLogFile().flush();
    std::wcerr.flush();
    std::terminate();
}
#else
inline void CrashLog(const std::wstring&) {}
template <typename... Args> inline void CrashLogF(const wchar_t*, Args...) {}
inline void EnableDebugLayerAndDred() {}
inline void HookDevice(ID3D12Device*) {}
inline void DumpNewMessages() {}
inline void CheckDeviceRemoved(ID3D12Device*, int = 0) {}
inline void InstallCrashHandler() {}
#endif
}
