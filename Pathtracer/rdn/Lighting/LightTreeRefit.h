#pragma once
// ═══════════════════════════════════════════════════════════════════
// Lighting/LightTreeRefit.h — Async light-tree TLAS rebuilder.
//
// When instances move, only the TLAS needs rebuilding (BLASes are
// per-instance in object space; their internal nodes are approximate
// for importance sampling). This runs on a background thread and
// produces new TLAS nodes for GPU upload.
//
// REQUIRES: Add this one-line accessor to LightTree.h's public section:
//   ID3D12Resource* GetTLASGpuBuffer() const { return m_gpu.TLASNodes.Get(); }
// ═══════════════════════════════════════════════════════════════════

#include "../Common.h"
#include "../LightTree.h"
#include <future>
#include <mutex>
#include <atomic>

namespace lt {

// ── Data computed per BLAS for fast TLAS refit ───────────────────
struct BLASRootLocal {
    Aabb   localAabb;        // object-space AABB of all tris in this BLAS
    float  power    = 0.f;
    Cone   localCone;        // object-space orientation cone
    uint32_t primCount  = 0;
    float  sumPower   = 0.f;
    float  sumPowerSq = 0.f;
    UINT   instanceID = 0;
};

// ── Result of an async TLAS rebuild ──────────────────────────────
struct TLASRefitResult {
    std::vector<LightTLASNodeGpu> nodes;
    std::vector<uint32_t>         blasToItem;
};

// ═════════════════════════════════════════════════════════════════
// Compute local-space BLAS root info from scene emissive triangles.
// Call once at init, and again if emission values change.
// ═════════════════════════════════════════════════════════════════
inline std::vector<BLASRootLocal> ComputeBLASLocalRoots(
    const std::vector<LightTriangle>& tris)
{
    // Group by instanceID (same as LightTreeBuilder::buildBLASes_SAOH)
    std::unordered_map<UINT, std::vector<uint32_t>> groups;
    for (uint32_t i = 0; i < (uint32_t)tris.size(); ++i)
        groups[tris[i].instanceID].push_back(i);

    std::vector<BLASRootLocal> roots;
    roots.reserve(groups.size());

    for (auto& [instID, idxs] : groups) {
        BLASRootLocal root{};
        root.instanceID = instID;

        bool first = true;
        for (uint32_t ti : idxs) {
            const auto& t = tris[ti];

            // Object-space AABB
            Aabb ta;
            ta.mn = min3(t.x, min3(t.y, t.z));
            ta.mx = max3(t.x, max3(t.y, t.z));
            if (first) { root.localAabb = ta; first = false; }
            else root.localAabb = unionAabb(root.localAabb, ta);

            root.power   += t.weight;
            root.sumPower += t.weight;
            root.sumPowerSq += t.weight * t.weight;
            root.primCount++;

            // Compute local normal for cone
            XMFLOAT3 e1 = sub3(t.y, t.x);
            XMFLOAT3 e2 = sub3(t.z, t.x);
            XMFLOAT3 n  = cross3(e1, e2);
            float len = length3(n);

            Cone tc;
            if (len < 1e-12f) {
                tc.axis = {0, 0, 1}; tc.theta_o = LT_PI; tc.theta_e = LT_HALF_PI;
            } else {
                tc.axis = normalize3(n); tc.theta_o = 0.f; tc.theta_e = LT_HALF_PI;
            }
            root.localCone = coneUnion(root.localCone, tc);
        }

        roots.push_back(root);
    }
    return roots;
}

// ═════════════════════════════════════════════════════════════════
// Transform a local AABB to world space (8-corner method)
// ═════════════════════════════════════════════════════════════════
inline Aabb TransformAabb(const Aabb& local, const XMFLOAT4X4& world) {
    XMFLOAT3 corners[8] = {
        { local.mn.x, local.mn.y, local.mn.z },
        { local.mx.x, local.mn.y, local.mn.z },
        { local.mn.x, local.mx.y, local.mn.z },
        { local.mx.x, local.mx.y, local.mn.z },
        { local.mn.x, local.mn.y, local.mx.z },
        { local.mx.x, local.mn.y, local.mx.z },
        { local.mn.x, local.mx.y, local.mx.z },
        { local.mx.x, local.mx.y, local.mx.z },
    };

    XMFLOAT3 wc = transformPointW(corners[0], world);
    Aabb result = { wc, wc };
    for (int i = 1; i < 8; ++i) {
        wc = transformPointW(corners[i], world);
        result.mn = min3(result.mn, wc);
        result.mx = max3(result.mx, wc);
    }
    return result;
}

// ═════════════════════════════════════════════════════════════════
// Standalone TLAS builder (same SAOH algorithm as LightTreeBuilder
// but operates on pre-computed BLAS root info + world transforms)
// ═════════════════════════════════════════════════════════════════
class TLASRebuilder {
public:
    TLASRefitResult Build(
        const std::vector<BLASRootLocal>& blasRoots,
        const std::vector<InstanceXformCPU>& xforms,
        uint32_t buildBins = 64)
    {
        m_bins = buildBins;

        // Transform BLAS roots to world space
        std::vector<TItem> items;
        items.reserve(blasRoots.size());

        for (uint32_t i = 0; i < (uint32_t)blasRoots.size(); ++i) {
            const auto& root = blasRoots[i];
            XMFLOAT4X4 world = (root.instanceID < xforms.size())
                ? xforms[root.instanceID].objectToWorld
                : LT_IDENTITY_4X4;

            Aabb worldAabb = TransformAabb(root.localAabb, world);

            // Rotate cone axis to world space
            XMFLOAT3X3 norm33;
            computeNormal33FromWorld(world, norm33);
            XMFLOAT3 worldAxis = transformNormalW(root.localCone.axis, norm33);

            Cone worldCone;
            worldCone.axis    = worldAxis;
            worldCone.theta_o = root.localCone.theta_o;
            worldCone.theta_e = root.localCone.theta_e;

            TItem it;
            it.idx       = i;
            it.a         = worldAabb;
            it.c         = aabbCenter(worldAabb);
            it.p         = root.power;
            it.cone      = worldCone;
            it.primCount = root.primCount;
            it.sumP      = root.sumPower;
            it.sumP2     = root.sumPowerSq;
            items.push_back(it);
        }

        m_tlas.clear();
        m_tlas.reserve(items.size() * 4 + 32);

        if (!items.empty())
            buildRecursive(items, 0, (uint32_t)items.size());

        // Build blasToItem mapping
        std::vector<uint32_t> blasToItem(blasRoots.size(), 0);
        for (uint32_t i = 0; i < (uint32_t)items.size(); ++i)
            blasToItem[items[i].idx] = i;

        return { std::move(m_tlas), std::move(blasToItem) };
    }

private:
    struct TItem {
        uint32_t idx; Aabb a; XMFLOAT3 c; float p;
        Cone cone; uint32_t primCount; float sumP, sumP2;
    };

    struct AggT {
        bool valid = false; Aabb a; float E = 0;
        Cone cone{}; uint32_t N = 0; float sumP = 0, sumP2 = 0;
    };

    static void aggAdd(AggT& A, const TItem& t) {
        if (!A.valid) { A.valid = true; A.a = t.a; A.E = t.p; A.cone = t.cone;
                        A.N = t.primCount; A.sumP = t.sumP; A.sumP2 = t.sumP2; return; }
        A.a = unionAabb(A.a, t.a); A.E += t.p; A.cone = coneUnion(A.cone, t.cone);
        A.N += t.primCount; A.sumP += t.sumP; A.sumP2 += t.sumP2;
    }

    static void aggMerge(AggT& A, const AggT& B) {
        if (!B.valid) return; if (!A.valid) { A = B; return; }
        A.a = unionAabb(A.a, B.a); A.E += B.E; A.cone = coneUnion(A.cone, B.cone);
        A.N += B.N; A.sumP += B.sumP; A.sumP2 += B.sumP2;
    }

    std::vector<LightTLASNodeGpu> m_tlas;
    uint32_t m_bins = 64;

    uint32_t buildRecursive(std::vector<TItem>& it, uint32_t begin, uint32_t end) {
        const uint32_t nodeIdx = (uint32_t)m_tlas.size();
        m_tlas.push_back({});

        AggT parent{};
        for (uint32_t i = begin; i < end; ++i) aggAdd(parent, it[i]);

        auto& N0 = m_tlas[nodeIdx];
        N0.bmin = parent.a.mn; N0.bmax = parent.a.mx; N0.power = parent.E;
        N0.axis = parent.cone.axis;
        N0.cosTheta_o = std::cos(lt::clampf(parent.cone.theta_o, 0.f, lt::LT_PI));
        N0.cosTheta_e = std::cos(lt::clampf(parent.cone.theta_e, 0.f, lt::LT_PI));
        N0.primCount = parent.N; N0.sumPower = parent.sumP; N0.sumPowerSq = parent.sumP2;
        N0.itemFirst = begin; N0.itemCount = end - begin;
        N0.firstChild = 0xFFFFFFFF; N0.childCount = 0;
        N0.blasIndex = UINT32_MAX;

        const uint32_t count = end - begin;
        if (count == 1) { N0.blasIndex = it[begin].idx; return nodeIdx; }

        // SAOH binary split
        int bestAxis = -1; float bestCost = std::numeric_limits<float>::infinity(), bestPos = 0;
        const float parentMA = (std::fmax)(1e-12f, aabbSurfaceArea(parent.a));
        const float parentMO = (std::fmax)(1e-12f, orientationMeasure(parent.cone));
        XMFLOAT3 ext = aabbExtent(parent.a);
        float lenMax = (std::fmax)(ext.x, (std::fmax)(ext.y, ext.z));

        for (int axis = 0; axis < 3; ++axis) {
            float mn = (&it[begin].c.x)[axis], mx = mn;
            for (uint32_t i = begin; i < end; ++i) {
                float v = (&it[i].c.x)[axis]; mn = (std::fmin)(mn, v); mx = (std::fmax)(mx, v);
            }
            float span = mx - mn; if (span <= 1e-20f) continue;

            const uint32_t B = (std::fmin)(64u, (std::fmax)(4u, m_bins));
            std::vector<AggT> bins(B);
            float invSpan = 1.f / span;
            for (uint32_t i = begin; i < end; ++i) {
                float v = (&it[i].c.x)[axis];
                uint32_t bi = (std::fmin)(B - 1u, (uint32_t)std::floor((v - mn) * invSpan * B));
                aggAdd(bins[bi], it[i]);
            }

            std::vector<AggT> pref(B), suff(B);
            for (uint32_t i = 0; i < B; ++i) { pref[i] = (i == 0) ? bins[i] : pref[i-1]; if (i > 0) aggMerge(pref[i], bins[i]); }
            for (int i = (int)B - 1; i >= 0; --i) { suff[i] = ((uint32_t)i == B-1) ? bins[i] : suff[i+1]; if ((uint32_t)i < B-1) aggMerge(suff[i], bins[i]); }

            float length_i = (&ext.x)[axis];
            float Kr = (length_i > 1e-20f) ? (lenMax / length_i) : 1e6f;

            for (uint32_t s = 1; s < B; ++s) {
                const AggT& L = pref[s-1]; const AggT& R = suff[s];
                if (!L.valid || !R.valid) continue;
                float cost = Kr * (L.E * aabbSurfaceArea(L.a) * orientationMeasure(L.cone)
                                 + R.E * aabbSurfaceArea(R.a) * orientationMeasure(R.cone))
                           / (parentMA * parentMO);
                if (cost < bestCost) { bestCost = cost; bestAxis = axis; bestPos = mn + span * s / B; }
            }
        }

        uint32_t mid;
        if (bestAxis >= 0) {
            auto midIt = std::partition(it.begin() + begin, it.begin() + end,
                [&](const TItem& t) { return (&t.c.x)[bestAxis] < bestPos; });
            mid = (uint32_t)(midIt - it.begin());
            if (mid == begin || mid == end) mid = (begin + end) / 2;
        } else {
            mid = (begin + end) / 2;
        }

        // Build 2 children
        m_tlas[nodeIdx].firstChild = (uint32_t)m_tlas.size();
        m_tlas[nodeIdx].childCount = 2;
        m_tlas.push_back({}); m_tlas.push_back({});

        uint32_t builtL = buildRecursive(it, begin, mid);
        uint32_t desiredL = m_tlas[nodeIdx].firstChild;
        if (builtL != desiredL) std::swap(m_tlas[builtL], m_tlas[desiredL]);

        uint32_t builtR = buildRecursive(it, mid, end);
        uint32_t desiredR = m_tlas[nodeIdx].firstChild + 1;
        if (builtR != desiredR) std::swap(m_tlas[builtR], m_tlas[desiredR]);

        return nodeIdx;
    }
};

// ═════════════════════════════════════════════════════════════════
// Async refit manager — kicks off CPU rebuild, polls completion
// ═════════════════════════════════════════════════════════════════
class LightTreeRefitManager {
public:
    // Kick off an async TLAS rebuild (non-blocking)
    void RequestRefit(
        std::vector<BLASRootLocal> blasRoots,
        std::vector<InstanceXformCPU> xforms)
    {
        if (m_pending.load()) return; // already in flight
        m_pending.store(true);

        m_future = std::async(std::launch::async,
            [roots = std::move(blasRoots), xf = std::move(xforms)]() {
                TLASRebuilder builder;
                return builder.Build(roots, xf);
            });
    }

    // Check if a result is ready (call each frame, non-blocking)
    bool PollResult(TLASRefitResult& outResult) {
        if (!m_pending.load()) return false;
        if (m_future.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready)
            return false;

        outResult = m_future.get();
        m_pending.store(false);
        return true;
    }

    bool IsPending() const { return m_pending.load(); }

private:
    std::future<TLASRefitResult> m_future;
    std::atomic<bool> m_pending{false};
};

} // namespace lt
