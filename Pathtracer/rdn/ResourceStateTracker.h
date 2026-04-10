#pragma once
#include <d3d12.h>
#include <wrl.h>
#include <unordered_map>
#include <vector>
#include <cassert>

class ResourceStateTracker
{
public:
    // Use D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES for whole-resource tracking.
    static constexpr UINT ALL = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    void Reset()
    {
        m_states.clear();
    }

    // Call once when you create/receive a resource (including swapchain buffers).
    void SetInitialState(ID3D12Resource* res, D3D12_RESOURCE_STATES state, UINT subresource = ALL)
    {
        if (!res) return;
        auto& entry = m_states[res];
        entry.subresourceStates.clear();
        entry.isPerSubresource = false;
        entry.globalState = state;

        if (subresource != ALL)
        {
            // Convert to per-subresource tracking
            EnsurePerSubresource(res, entry);
            entry.subresourceStates[subresource] = state;
        }
    }

    // Query what we believe the state currently is.
    D3D12_RESOURCE_STATES GetState(ID3D12Resource* res, UINT subresource = ALL) const
    {
        auto it = m_states.find(res);
        if (it == m_states.end())
        {
            // If you hit this, you forgot SetInitialState for that resource.
            // Prefer asserting in debug.
            assert(false && "ResourceStateTracker: missing initial state");
            return D3D12_RESOURCE_STATE_COMMON;
        }
        const Entry& e = it->second;

        if (!e.isPerSubresource || subresource == ALL)
            return e.globalState;

        auto jt = e.subresourceStates.find(subresource);
        if (jt != e.subresourceStates.end())
            return jt->second;

        // If not explicitly set, assume the globalState is correct baseline.
        return e.globalState;
    }

    // Record a transition barrier using tracked "before".
    // Updates tracking immediately.
    void Transition(ID3D12GraphicsCommandList* cl,
                    ID3D12Resource* res,
                    D3D12_RESOURCE_STATES to,
                    UINT subresource = ALL)
    {
        if (!cl || !res) return;

        auto& entry = m_states[res];
        if (!entry.initialized)
        {
            // Strongly recommend setting explicit initial state instead of defaulting
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

        // Update tracking
        SetStateInternal(entry, subresource, to);
    }

    // UAV barrier does NOT change tracked state, just ordering.
    void UAV(ID3D12GraphicsCommandList* cl, ID3D12Resource* res)
    {
        if (!cl || !res) return;
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        b.UAV.pResource = res;
        cl->ResourceBarrier(1, &b);
    }

    // Use this if an external system (Streamline) changes resource state behind your back.
    // You are explicitly overriding what the tracker believes.
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
        bool initialized = true; // if in map, assume initialized
        bool isPerSubresource = false;
        D3D12_RESOURCE_STATES globalState = D3D12_RESOURCE_STATE_COMMON;
        std::unordered_map<UINT, D3D12_RESOURCE_STATES> subresourceStates;
    };

    void EnsurePerSubresource(ID3D12Resource* res, Entry& e)
    {
        if (e.isPerSubresource) return;
        e.isPerSubresource = true;

        // Seed per-subresource entries lazily; we keep globalState as baseline
        // and only store overrides for subresources we transition.
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