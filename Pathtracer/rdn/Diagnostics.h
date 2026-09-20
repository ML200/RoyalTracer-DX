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
#include <Windows.h>

// The functions below are inline: every translation unit that includes this header must see the
// same definitions, or the linker picks one of the stub bodies for the whole program and the
// device-removal dump silently never runs. The switch therefore defaults to on here.
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
// Symbolized call stack of the caller, written to the crash log (CrashHandler.cpp).
void LogCallStack(const wchar_t* title);

// The first occurrence of every debug-layer error gets the call stack of the API call that
// raised it: the message queue says what went wrong, the stack says where. Runs inside the
// D3D12 call on whichever thread made it.
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

#ifndef DXDIAG_ENABLE_DEBUG_LAYER
#define DXDIAG_ENABLE_DEBUG_LAYER 1
#endif

// Enable device-removal breadcrumbs before creating the D3D12 device.
inline void EnableDebugLayerAndDred() {
    CrashLog(L"[dxdiag] EnableDebugLayerAndDred reached, diagnostic plumbing live\n");

#if DXDIAG_ENABLE_DEBUG_LAYER
    // RT_NO_DEBUG_LAYER=1 in the environment skips the validation layer for one run: it is slow,
    // and the preview SDK layers crash inside CreateStateObject on some valid pipelines.
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
        // Breadcrumb contexts record the marker the profiler sets per render pass, so the dump
        // names the pass the GPU was in.
        ComPtr<ID3D12DeviceRemovedExtendedDataSettings1> dredSet1;
        if (SUCCEEDED(dredSet.As(&dredSet1)))
            dredSet1->SetBreadcrumbContextEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
        CrashLog(L"[DX]  DRED enabled (cheap, used for TDR breadcrumbs)\n");
    }
}

inline void HookDevice(ID3D12Device* device) {
    device->QueryInterface(IID_PPV_ARGS(&g_infoQ));
    device->QueryInterface(IID_PPV_ARGS(&g_dred));

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

// Drain the debug layer's message queue. Each distinct message is logged once with its count of
// repeats reported later: a validation error that fires per frame would otherwise cost more in
// console, file and debugger output than the frame itself. After a few repeats the message ID is
// also denied at the source, so the runtime stops formatting it; validation itself stays on.
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
            continue; // info and plain messages carry nothing worth the output cost

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
        // DRED 1.2 adds the marker strings of each list (breadcrumb contexts); the profiler sets
        // one per render pass, so the last context before the failing op names the pass.
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
