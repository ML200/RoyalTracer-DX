#pragma once
#include <d3d12.h>
#include <wrl.h>
#include <unordered_map>
#include <vector>
#include <cassert>

class ResourceStateTracker
{
public:
    static constexpr UINT ALL = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    void Reset()
    {
        m_states.clear();
    }

    //call once on creation, including swapchain buffers
    void SetInitialState(ID3D12Resource* res, D3D12_RESOURCE_STATES state, UINT subresource = ALL)
    {
        if (!res) return;
        auto& entry = m_states[res];
        entry.subresourceStates.clear();
        entry.isPerSubresource = false;
        entry.globalState = state;

        if (subresource != ALL)
        {
            EnsurePerSubresource(res, entry);
            entry.subresourceStates[subresource] = state;
        }
    }

    D3D12_RESOURCE_STATES GetState(ID3D12Resource* res, UINT subresource = ALL) const
    {
        auto it = m_states.find(res);
        if (it == m_states.end())
        {
            //missing SetInitialState
            assert(false && "ResourceStateTracker: missing initial state");
            return D3D12_RESOURCE_STATE_COMMON;
        }
        const Entry& e = it->second;

        if (!e.isPerSubresource || subresource == ALL)
            return e.globalState;

        auto jt = e.subresourceStates.find(subresource);
        if (jt != e.subresourceStates.end())
            return jt->second;

        return e.globalState;
    }

    //transition barrier using tracked before state, updates tracking
    void Transition(ID3D12GraphicsCommandList* cl,
                    ID3D12Resource* res,
                    D3D12_RESOURCE_STATES to,
                    UINT subresource = ALL)
    {
        if (!cl || !res) return;

        auto& entry = m_states[res];
        if (!entry.initialized)
        {
            assert(false && "ResourceStateTracker: Transition called without SetInitialState");
            entry.initialized = true;
            entry.globalState = D3D12_RESOURCE_STATE_COMMON;
        }

        if (subresource != ALL)
            EnsurePerSubresource(res, entry);

        D3D12_RESOURCE_STATES before = GetState(res, subresource);
        if (before == to) return;

        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        b.Transition.pResource = res;
        b.Transition.Subresource = subresource;
        b.Transition.StateBefore = before;
        b.Transition.StateAfter  = to;

        cl->ResourceBarrier(1, &b);

        SetStateInternal(entry, subresource, to);
    }

    //UAV barrier does not change tracked state, ordering only
    void UAV(ID3D12GraphicsCommandList* cl, ID3D12Resource* res)
    {
        if (!cl || !res) return;
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        b.UAV.pResource = res;
        cl->ResourceBarrier(1, &b);
    }

    //explicit override when external system (Streamline) changes state behind our back
    void ForceState(ID3D12Resource* res, D3D12_RESOURCE_STATES state, UINT subresource = ALL)
    {
        if (!res) return;
        auto& entry = m_states[res];
        entry.initialized = true;

        if (subresource == ALL)
        {
            entry.isPerSubresource = false;
            entry.subresourceStates.clear();
            entry.globalState = state;
        }
        else
        {
            EnsurePerSubresource(res, entry);
            entry.subresourceStates[subresource] = state;
        }
    }

private:
    struct Entry
    {
        bool initialized = true;
        bool isPerSubresource = false;
        D3D12_RESOURCE_STATES globalState = D3D12_RESOURCE_STATE_COMMON;
        std::unordered_map<UINT, D3D12_RESOURCE_STATES> subresourceStates;
    };

    void EnsurePerSubresource(ID3D12Resource* res, Entry& e)
    {
        if (e.isPerSubresource) return;
        e.isPerSubresource = true;
    }

    void SetStateInternal(Entry& e, UINT subresource, D3D12_RESOURCE_STATES st)
    {
        if (!e.isPerSubresource || subresource == ALL)
        {
            e.globalState = st;
            if (subresource == ALL)
            {
                e.subresourceStates.clear();
                e.isPerSubresource = false;
            }
        }
        else
        {
            e.subresourceStates[subresource] = st;
        }
    }

    std::unordered_map<ID3D12Resource*, Entry> m_states;
};
