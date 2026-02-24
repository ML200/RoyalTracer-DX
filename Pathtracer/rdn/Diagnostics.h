#pragma once
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl.h>
#include <iostream>
#include <iomanip>

#if ENABLE_D3D12_DIAGNOSTICS
#   pragma comment(lib,"dxguid.lib")
#endif

namespace dxdiag
{
#if ENABLE_D3D12_DIAGNOSTICS
    using Microsoft::WRL::ComPtr;

    inline ComPtr<ID3D12InfoQueue>                g_infoQ;
    inline ComPtr<ID3D12DeviceRemovedExtendedData> g_dred;

    inline void EnableDebugLayerAndDred()
    {
        ComPtr<ID3D12Debug> dbg;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dbg))))
        {
            dbg->EnableDebugLayer();
            std::wcout << L"[DX]  Debug-layer enabled\n";
        }

        ComPtr<ID3D12DeviceRemovedExtendedDataSettings> dredSet;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dredSet))))
        {
            dredSet->SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            dredSet->SetPageFaultEnablement      (D3D12_DRED_ENABLEMENT_FORCED_ON);
            std::wcout << L"[DX]  DRED enabled\n";
        }
    }

    inline void HookDevice( ID3D12Device* device )
    {
        device->QueryInterface(IID_PPV_ARGS(&g_infoQ));
        device->QueryInterface(IID_PPV_ARGS(&g_dred));

        if (g_infoQ)
        {
            // don't spam – only ERRORS and CORRUPTION
            g_infoQ->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, true);
            g_infoQ->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR,      true);
            g_infoQ->SetMessageCountLimit(4096);
        }
    }

    inline void DumpNewMessages()
    {
        if (!g_infoQ) return;

        const UINT64 nMsg = g_infoQ->GetNumStoredMessagesAllowedByRetrievalFilter();
        for (UINT64 i = 0; i < nMsg; ++i)
        {
            SIZE_T sz = 0;
            // 1. Get the required size
            g_infoQ->GetMessage(i, nullptr, &sz);

            std::unique_ptr<uint8_t[]> blob(new uint8_t[sz]);
            D3D12_MESSAGE* msg = reinterpret_cast<D3D12_MESSAGE*>(blob.get());

            // 2. ACTUALLY FETCH THE DATA FIRST
            g_infoQ->GetMessage(i, msg, &sz);

            // 3. Now it is safe to read msg->pDescription
            /*if (msg->pDescription != nullptr &&
                strstr(msg->pDescription, "sl.dlss_d.mvec") != nullptr &&
                strstr(msg->pDescription, "RESOURCE_BARRIER_BEFORE_AFTER_MISMATCH") != nullptr)
            {
                continue; // Trash it silently
            }*/

            std::wcout << L"[DX] " << msg->pDescription << std::endl;
        }
        g_infoQ->ClearStoredMessages();
    }

    inline void CheckDeviceRemoved( ID3D12Device* device )
    {
        HRESULT hr = device->GetDeviceRemovedReason();
        if (SUCCEEDED(hr)) return;

        std::wcerr << L"\n*** DEVICE LOST: 0x"
                   << std::hex << hr << std::dec << L" ***\n";

        if (g_dred)
        {
            D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT bc = {};
            D3D12_DRED_PAGE_FAULT_OUTPUT       pf = {};
            g_dred->GetAutoBreadcrumbsOutput(&bc);
            g_dred->GetPageFaultAllocationOutput(&pf);

            // very compact breadcrumb dump – customise as you like
            for (auto node = bc.pHeadAutoBreadcrumbNode;
                 node; node = node->pNext)
            {
                std::wcerr << L"  ► Last GPU command list 0x"
                           << node->pCommandListDebugNameA << L"\n";
            }

            if (pf.PageFaultVA)
            {
                std::wcerr << L"  ► Page-fault at GPU VA : 0x"
                           << std::hex << pf.PageFaultVA << std::dec << L"\n";
            }
        }
        std::terminate();
    }
#else
    inline void EnableDebugLayerAndDred()  {}
    inline void HookDevice(ID3D12Device*)  {}
    inline void DumpNewMessages()          {}
    inline void CheckDeviceRemoved(ID3D12Device*) {}
#endif
} // namespace dxdiag
