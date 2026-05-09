#pragma once
//====================================
//ASYNC LIGHT-TREE TLAS REBUILDER
//====================================
//moved instances rebuild TLAS only, BLASes per-instance in object space
//runs on background thread, produces TLAS nodes for GPU upload
//requires LightTree.h to expose GetTLASGpuBuffer()

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
    std::vector<uint32_t>         blasBitTrails;    // 2-bits-per-level TLAS descent path per BLAS
    std::vector<XMFLOAT4X4>      blasWorldToLocal;  // updated inverse transforms
};

//====================================
//BLAS LOCAL ROOTS
//====================================
//local-space BLAS root info from scene emissive triangles
//call once at init, again if emission changes
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

//====================================
//TRANSFORM AABB
//====================================
//local AABB to world, 8-corner method
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

//====================================
//STANDALONE TLAS BUILDER
//====================================
//same SAOH algo as LightTreeBuilder
//operates on pre-computed BLAS root info + world transforms
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

        // populated at TLAS leaves during recursion (one entry per BLAS)
        m_blasBitTrails.assign(blasRoots.size(), 0u);
        if (!items.empty())
            buildRecursive(items, 0, (uint32_t)items.size(), 0u, 0u);

        // Compute inverse world transforms (worldToLocal) for each BLAS
        std::vector<XMFLOAT4X4> blasWorldToLocal(blasRoots.size());
        for (uint32_t i = 0; i < (uint32_t)blasRoots.size(); ++i) {
            XMFLOAT4X4 world = (blasRoots[i].instanceID < xforms.size())
                ? xforms[blasRoots[i].instanceID].objectToWorld
                : LT_IDENTITY_4X4;
            XMVECTOR det;
            XMMATRIX Winv = XMMatrixInverse(&det, XMLoadFloat4x4(&world));
            XMStoreFloat4x4(&blasWorldToLocal[i], Winv);
        }

        return { std::move(m_tlas), std::move(m_blasBitTrails), std::move(blasWorldToLocal) };
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
    std::vector<uint32_t>         m_blasBitTrails;  // populated as TLAS leaves are reached
    uint32_t m_bins = 64;
    static constexpr uint32_t LT_TRAIL_MAX_DEPTH = 16; // 32 bits / 2 bits per level

    uint32_t buildRecursive(std::vector<TItem>& it, uint32_t begin, uint32_t end,
                            uint32_t bitTrail, uint32_t depth) {
        const uint32_t nodeIdx = (uint32_t)m_tlas.size();
        m_tlas.push_back({});

        AggT parent{};
        for (uint32_t i = begin; i < end; ++i) aggAdd(parent, it[i]);

        auto& N0 = m_tlas[nodeIdx];
        N0.bmin = parent.a.mn; N0.bmax = parent.a.mx; N0.power = parent.E;
        N0.axis = parent.cone.axis;
        N0.cosTheta_o = std::cos(lt::clampf(parent.cone.theta_o, 0.f, lt::LT_PI));
        N0.sinTheta_o = std::sqrt((std::fmax)(0.f, 1.f - N0.cosTheta_o * N0.cosTheta_o));
        N0.primCount = parent.N; N0.sumPower = parent.sumP; N0.sumPowerSq = parent.sumP2;
        N0.itemFirst = begin; N0.itemCount = end - begin;
        N0.firstChild = 0xFFFFFFFF; N0.childCount = 0;
        N0.blasIndex = UINT32_MAX;

        const uint32_t count = end - begin;
        if (count == 1) {
            N0.blasIndex = it[begin].idx;
            m_blasBitTrails[it[begin].idx] = bitTrail;
            return nodeIdx;
        }

        // SAOH binary split helper, mirrors LightTreeBuilder::buildTLASRecursive_SAOH
        auto findBinarySplit = [&](uint32_t b0, uint32_t e0,
                                   int& axisOut, float& posOut, uint32_t& midOut) -> bool
        {
            AggT parentL{}; for (uint32_t i = b0; i < e0; ++i) aggAdd(parentL, it[i]);
            const Aabb     aabb     = parentL.a;
            const XMFLOAT3 ext      = aabbExtent(aabb);
            const float    lenX     = ext.x, lenY = ext.y, lenZ = ext.z;
            const float    lenMax   = (std::fmax)(lenX, (std::fmax)(lenY, lenZ));
            const float    parentMA = (std::fmax)(1e-12f, aabbSurfaceArea(aabb));
            const float    parentMO = (std::fmax)(1e-12f, orientationMeasure(parentL.cone));

            int   bestAxis = -1;
            float bestCost = std::numeric_limits<float>::infinity();
            float bestPos  = 0.f;
            const uint32_t B = (std::fmin)(64u, (std::fmax)(4u, m_bins));

            for (int axis = 0; axis < 3; ++axis) {
                float mn = (&it[b0].c.x)[axis], mx = mn;
                for (uint32_t i = b0; i < e0; ++i) {
                    float v = (&it[i].c.x)[axis]; mn = (std::fmin)(mn, v); mx = (std::fmax)(mx, v);
                }
                float span = mx - mn; if (span <= 1e-20f) continue;

                std::vector<AggT> bins(B);
                float invSpan = 1.f / span;
                for (uint32_t i = b0; i < e0; ++i) {
                    float v = (&it[i].c.x)[axis];
                    uint32_t bi = (std::fmin)(B - 1u, (uint32_t)std::floor((v - mn) * invSpan * B));
                    aggAdd(bins[bi], it[i]);
                }

                std::vector<AggT> pref(B), suff(B);
                for (uint32_t i = 0; i < B; ++i) { pref[i] = (i == 0) ? bins[i] : pref[i-1]; if (i > 0) aggMerge(pref[i], bins[i]); }
                for (int i = (int)B - 1; i >= 0; --i) { suff[i] = ((uint32_t)i == B-1) ? bins[i] : suff[i+1]; if ((uint32_t)i < B-1) aggMerge(suff[i], bins[i]); }

                float length_i = (axis == 0 ? lenX : (axis == 1 ? lenY : lenZ));
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

            if (bestAxis < 0 || !std::isfinite(bestCost)) return false;

            auto midIt = std::partition(it.begin() + b0, it.begin() + e0,
                [&](const TItem& t) { return (&t.c.x)[bestAxis] < bestPos; });
            uint32_t mid = (uint32_t)(midIt - (it.begin() + b0)) + b0;
            if (mid == b0 || mid == e0) {
                mid = (b0 + e0) / 2;
                std::nth_element(it.begin() + b0, it.begin() + mid, it.begin() + e0,
                    [&](const TItem& A, const TItem& B){ return (&A.c.x)[bestAxis] < (&B.c.x)[bestAxis]; });
            }

            axisOut = bestAxis; posOut = bestPos; midOut = mid; return true;
        };

        // Top-level binary split, fall back to median split on the widest axis if SAOH fails
        int ax; float pos; uint32_t mid;
        bool ok = findBinarySplit(begin, end, ax, pos, mid);
        if (!ok) {
            int widest = 0;
            XMFLOAT3 e = aabbExtent(parent.a);
            if (e.y > e.x && e.y >= e.z) widest = 1;
            else if (e.z > e.x && e.z >= e.y) widest = 2;
            mid = (begin + end) / 2;
            std::nth_element(it.begin() + begin, it.begin() + mid, it.begin() + end,
                [&](const TItem& A, const TItem& B){ return (&A.c.x)[widest] < (&B.c.x)[widest]; });
        }

        // Sub-split each half once more to grow up to 4 buckets, matching LightTreeBuilder
        struct Range { uint32_t b, e; };
        Range buckets[4]; uint32_t bucketCount = 0;

        auto pushOrSplitOnce = [&](uint32_t b, uint32_t e) {
            if (e <= b) return;
            const uint32_t c = e - b;
            if (c == 1) { buckets[bucketCount++] = {b, e}; return; }
            int ax2; float pos2; uint32_t mid2;
            if (findBinarySplit(b, e, ax2, pos2, mid2) && mid2 > b && mid2 < e) {
                buckets[bucketCount++] = {b, mid2};
                buckets[bucketCount++] = {mid2, e};
            } else {
                buckets[bucketCount++] = {b, e};
            }
        };

        pushOrSplitOnce(begin, mid);
        pushOrSplitOnce(mid,   end);

        // Allocate child placeholders contiguously, build into them via swap
        m_tlas[nodeIdx].firstChild = (uint32_t)m_tlas.size();
        m_tlas[nodeIdx].childCount = bucketCount;
        for (uint32_t i = 0; i < bucketCount; ++i) m_tlas.push_back({});

        const bool   depthOverflow = (depth >= LT_TRAIL_MAX_DEPTH);
        const uint32_t shift       = 2u * depth;
        for (uint32_t c = 0; c < bucketCount; ++c) {
            const uint32_t childTrail = depthOverflow ? bitTrail : (bitTrail | (c << shift));
            uint32_t built   = buildRecursive(it, buckets[c].b, buckets[c].e, childTrail, depth + 1u);
            uint32_t desired = m_tlas[nodeIdx].firstChild + c;
            if (built != desired) std::swap(m_tlas[built], m_tlas[desired]);
        }

        return nodeIdx;
    }
};

//====================================
//ASYNC REFIT MANAGER
//====================================
//kicks off CPU rebuild, polls completion
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
