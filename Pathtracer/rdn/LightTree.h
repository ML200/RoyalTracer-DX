#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <d3d12.h>
#include "d3dx12.h"
#include <wrl.h>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <DirectXMath.h>
#include <unordered_map>
#include <map>
#include <iostream>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include "../shaders/LightTreeTrail.h"
#include "Lighting/LightTreePacking.h"

using Microsoft::WRL::ComPtr;
using namespace DirectX;

#ifndef LT_ENABLE_LOGS
#define LT_ENABLE_LOGS 1
#endif
#ifndef LT_LOG_BUILD_SPAM
#define LT_LOG_BUILD_SPAM 0
#endif
#ifndef LT_ENABLE_TIMING
#define LT_ENABLE_TIMING 1
#endif

#if LT_ENABLE_LOGS
#define LT_LOG(expr)                                                                                                   \
    do {                                                                                                               \
        std::wcout << L"[LightTree] " << expr << std::endl;                                                            \
    } while (0)
#define LT_WARN(expr)                                                                                                  \
    do {                                                                                                               \
        std::wcout << L"[LightTree][WARN] " << expr << std::endl;                                                      \
    } while (0)
#else
#define LT_LOG(expr)                                                                                                   \
    do {                                                                                                               \
    } while (0)
#define LT_WARN(expr)                                                                                                  \
    do {                                                                                                               \
    } while (0)
#endif

#define LT_CONCAT_INNER(a, b) a##b
#define LT_CONCAT(a, b) LT_CONCAT_INNER(a, b)

namespace lt {
namespace detail {
#if LT_ENABLE_TIMING
struct ScopedTimer {
    using clock = std::chrono::high_resolution_clock;
    const wchar_t* label;
    clock::time_point t0;
    explicit ScopedTimer(const wchar_t* l) : label(l), t0(clock::now()) {}
    ~ScopedTimer() {
        const double ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
        std::wcout << L"[LightTree][time] " << label << L": " << ms << L" ms" << std::endl;
    }
};
#endif
} // namespace detail
} // namespace lt

#if LT_ENABLE_TIMING
#define LT_TIME_SCOPE(LABEL_WIDE) ::lt::detail::ScopedTimer LT_CONCAT(_lt_scope_timer_, __LINE__)(LABEL_WIDE)
#else
#define LT_TIME_SCOPE(LABEL_WIDE)                                                                                      \
    do {                                                                                                               \
    } while (0)
#endif

struct LightTriangle {
    XMFLOAT3 x;
    float cdf;
    XMFLOAT3 y;
    UINT meshID;
    XMFLOAT3 z;
    float weight;
    XMFLOAT3 emission;
    UINT triCount;
    float totalWeight;
    XMFLOAT3 pad0;
};

struct InstanceXformCPU {
    XMFLOAT4X4 objectToWorld;
};

namespace lt {
struct LightInstanceRef {
    UINT instanceID;
    UINT meshID;
};
} // namespace lt

namespace lt {
#pragma pack(push, 1)

// Full-precision CPU/decoded records; GPU resources use Light*NodePacked.
struct LightTLASNodeGpu {
    XMFLOAT3 bmin;
    float power;
    XMFLOAT3 bmax;
    float cosTheta_o;
    XMFLOAT3 axis;
    float sinTheta_o; // conservative orientation cone

    uint32_t firstChild;
    uint32_t childCount;
    uint32_t slot;
    uint32_t _pad; // keeps struct stride 16B aligned
};

struct LightBLASNodeGpu {
    XMFLOAT3 bmin;
    float power;
    XMFLOAT3 bmax;
    float cosTheta_o;
    XMFLOAT3 axis;
    float sinTheta_o; // conservative orientation cone

    uint32_t firstChild;
    uint32_t childCount;
    uint32_t triFirst;
    uint32_t triCount;
};

struct BlasRangeGpu {
    uint32_t nodeOffset;
    uint32_t nodeCount;
    uint32_t triIndexOffset;
    uint32_t triIndexCount;
};

// One instance of a shared mesh light tree.
struct LightSlotGpu {
    float worldToLocal[12];
    uint32_t instanceID;
    uint32_t nodeOffset;
    float powerScale;
    uint32_t _pad;
};
#pragma pack(pop)

static_assert(sizeof(LightTLASNodeGpu) == 64 && sizeof(LightBLASNodeGpu) == 64);
static_assert(sizeof(BlasRangeGpu) == 16 && sizeof(LightSlotGpu) == 64);

static constexpr float LT_PI = 3.14159265358979323846f;
static constexpr float LT_HALF_PI = 1.57079632679489661923f;

struct Aabb {
    XMFLOAT3 mn, mx;
};

static XMFLOAT3 add3(const XMFLOAT3& a, const XMFLOAT3& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
static XMFLOAT3 sub3(const XMFLOAT3& a, const XMFLOAT3& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
static XMFLOAT3 mul3(const XMFLOAT3& a, float s) {
    return {a.x * s, a.y * s, a.z * s};
}
static float dot3(const XMFLOAT3& a, const XMFLOAT3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
static XMFLOAT3 cross3(const XMFLOAT3& a, const XMFLOAT3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
static float length3(const XMFLOAT3& a) {
    return std::sqrt((std::fmax)(0.f, dot3(a, a)));
}
static XMFLOAT3 normalize3(const XMFLOAT3& a) {
    float len = length3(a);
    if (len < 1e-20f)
        return {0, 0, 1};
    float inv = 1.0f / len;
    return {a.x * inv, a.y * inv, a.z * inv};
}
static XMFLOAT3 min3(const XMFLOAT3& a, const XMFLOAT3& b) {
    return {(std::fmin)(a.x, b.x), (std::fmin)(a.y, b.y), (std::fmin)(a.z, b.z)};
}
static XMFLOAT3 max3(const XMFLOAT3& a, const XMFLOAT3& b) {
    return {(std::fmax)(a.x, b.x), (std::fmax)(a.y, b.y), (std::fmax)(a.z, b.z)};
}

static Aabb unionAabb(const Aabb& a, const Aabb& b) {
    return {min3(a.mn, b.mn), max3(a.mx, b.mx)};
}
static float aabbSurfaceArea(const Aabb& a) {
    XMFLOAT3 e = sub3(a.mx, a.mn);
    return 2.0f * (e.x * e.y + e.x * e.z + e.y * e.z);
}
static XMFLOAT3 aabbCenter(const Aabb& a) {
    return mul3(add3(a.mn, a.mx), 0.5f);
}
static XMFLOAT3 aabbExtent(const Aabb& a) {
    return sub3(a.mx, a.mn);
}

static float clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}
static float safe_acosf(float x) {
    return std::acos(clampf(x, -1.f, 1.f));
}

static XMFLOAT3 slerpUnit(const XMFLOAT3& a, const XMFLOAT3& b, float t) {
    float cosT = clampf(dot3(a, b), -1.f, 1.f);
    float theta = std::acos(cosT);
    if (theta < 1e-6f)
        return a;
    float s = std::sin(theta);
    float w0 = std::sin((1.f - t) * theta) / s;
    float w1 = std::sin(t * theta) / s;
    return normalize3(add3(mul3(a, w0), mul3(b, w1)));
}

struct Cone {
    XMFLOAT3 axis{0, 0, 1};
    float theta_o = LT_PI;
    float theta_e = LT_HALF_PI;
};

// Conservative cone union (Conty & Kulla 2018).
static Cone coneUnion(const Cone& A, const Cone& B) {
    Cone a = A, b = B;
    if (b.theta_o > a.theta_o)
        std::swap(a, b);

    const float d = clampf(dot3(a.axis, b.axis), -1.f, 1.f);
    const float theta_d = safe_acosf(d);

    Cone out;
    out.theta_e = (std::fmax)(a.theta_e, b.theta_e);

    if ((std::fmin)(theta_d + b.theta_o, LT_PI) <= a.theta_o) {
        out.axis = a.axis;
        out.theta_o = a.theta_o;
        return out;
    }

    float theta_o = (a.theta_o + theta_d + b.theta_o) * 0.5f;
    theta_o = (std::fmin)(theta_o, LT_PI);

    if (theta_d < 1e-7f) {
        out.axis = a.axis;
    } else if (LT_PI - theta_d < 1e-7f) {
        // Opposite axes: slerp is undefined.
        XMFLOAT3 t = (std::fabs(a.axis.x) < 0.9f) ? XMFLOAT3{1, 0, 0} : XMFLOAT3{0, 1, 0};
        const auto perpendicular = normalize3(cross3(a.axis, t));
        const float angle = theta_o - a.theta_o;
        out.axis = normalize3(add3(mul3(a.axis, std::cos(angle)), mul3(perpendicular, std::sin(angle))));
    } else {
        float t = clampf((theta_o - a.theta_o) / theta_d, 0.f, 1.f);
        out.axis = slerpUnit(a.axis, b.axis, t);
    }

    out.theta_o = theta_o;
    return out;
}

// SAOH orientation measure (Conty & Kulla 2018).
static float orientationMeasure(const Cone& c) {
    const float theta_o = clampf(c.theta_o, 0.f, LT_PI);
    const float theta_w = (std::fmin)(theta_o + clampf(c.theta_e, 0.f, LT_PI), LT_PI);
    const double to = theta_o, tw = theta_w;
    const double term0 = 2.0 * LT_PI * (1.0 - std::cos(to));
    const double extra =
        (LT_HALF_PI) * (2.0 * tw * std::sin(to) - std::cos(to - 2.0 * tw) - 2.0 * to * std::sin(to) + std::cos(to));
    return static_cast<float>(term0 + extra);
}

static const XMFLOAT4X4 LT_IDENTITY_4X4 = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

static const XMFLOAT3X3 LT_IDENTITY_3X3 = {1, 0, 0, 0, 1, 0, 0, 0, 1};

static XMFLOAT3 transformPointW(const XMFLOAT3& p, const XMFLOAT4X4& world) {
    XMVECTOR vp = XMLoadFloat3(&p);
    XMMATRIX W = XMLoadFloat4x4(&world);
    XMVECTOR o = XMVector3TransformCoord(vp, W);
    XMFLOAT3 out;
    XMStoreFloat3(&out, o);
    return out;
}

static void computeNormal33FromWorld(const XMFLOAT4X4& world, XMFLOAT3X3& out33) {
    XMMATRIX W = XMLoadFloat4x4(&world);
    XMVECTOR det;
    XMMATRIX InvW = XMMatrixInverse(&det, W);
    const float detx = XMVectorGetX(det);
    if (!std::isfinite(detx) || std::fabs(detx) < 1e-30f) {
        XMStoreFloat3x3(&out33, XMMatrixIdentity());
        return;
    }
    XMMATRIX N = XMMatrixTranspose(InvW);
    XMStoreFloat3x3(&out33, N);
}

static XMFLOAT3 transformNormalW(const XMFLOAT3& n, const XMFLOAT3X3& N33) {
    XMMATRIX N = XMLoadFloat3x3(&N33);
    XMVECTOR vn = XMLoadFloat3(&n);
    XMVECTOR wo = XMVector3TransformNormal(vn, N);
    XMFLOAT3 out;
    XMStoreFloat3(&out, wo);
    return normalize3(out);
}

// Similarity transforms preserve cone angles.
static bool similarityTransform(const XMFLOAT4X4& w) {
    const XMFLOAT3 a{w._11, w._12, w._13}, b{w._21, w._22, w._23}, c{w._31, w._32, w._33};
    const float scale = (std::max)({dot3(a, a), dot3(b, b), dot3(c, c), 1e-20f});
    return std::fabs(dot3(a, a) - dot3(b, b)) < scale * 1e-5f && std::fabs(dot3(a, a) - dot3(c, c)) < scale * 1e-5f &&
           std::fabs(dot3(a, b)) < scale * 1e-5f && std::fabs(dot3(a, c)) < scale * 1e-5f &&
           std::fabs(dot3(b, c)) < scale * 1e-5f;
}
static Aabb transformBounds(const Aabb& a, const XMFLOAT4X4& w) {
    Aabb r{};
    for (int i = 0; i < 8; ++i) {
        const auto p = transformPointW({i & 1 ? a.mx.x : a.mn.x, i & 2 ? a.mx.y : a.mn.y, i & 4 ? a.mx.z : a.mn.z}, w);
        if (i == 0)
            r = {p, p};
        else {
            r.mn = min3(r.mn, p);
            r.mx = max3(r.mx, p);
        }
    }
    return r;
}

// Area scale of the equivalent uniform scaling.
static float areaScale(const XMFLOAT4X4& w) {
    const XMFLOAT3 a{w._11, w._12, w._13}, b{w._21, w._22, w._23}, c{w._31, w._32, w._33};
    const float det = std::fabs(dot3(a, cross3(b, c)));
    if (!std::isfinite(det) || det <= 0.f)
        return 0.f;
    return std::cbrt(det * det);
}

static void setSlotTransform(LightSlotGpu& s, const XMFLOAT4X4& world) {
    XMVECTOR det;
    XMMATRIX inv = XMMatrixInverse(&det, XMLoadFloat4x4(&world));
    const float detx = XMVectorGetX(det);
    if (!std::isfinite(detx) || std::fabs(detx) < 1e-30f)
        inv = XMMatrixIdentity();
    for (int row = 0; row < 4; ++row) {
        XMFLOAT4 f;
        XMStoreFloat4(&f, inv.r[row]);
        s.worldToLocal[row * 3 + 0] = f.x;
        s.worldToLocal[row * 3 + 1] = f.y;
        s.worldToLocal[row * 3 + 2] = f.z;
    }
    s.powerScale = areaScale(world);
}
static LightSlotGpu makeSlotRecord(UINT instanceID, uint32_t nodeOffset, const XMFLOAT4X4& world) {
    LightSlotGpu s{};
    s.instanceID = instanceID;
    s.nodeOffset = nodeOffset;
    s._pad = 0u;
    setSlotTransform(s, world);
    return s;
}

struct BLASNode {
    Aabb aabb{};
    float power = 0.f;
    Cone cone{};
    uint32_t firstChild = 0xFFFFFFFF;
    uint32_t childCount = 0;
    uint32_t primCount = 0;
    float sumPower = 0.f;
    float sumPowerSq = 0.f;
    uint32_t triFirst = 0, triCount = 0;

    bool isLeaf() const { return childCount == 0; }
};

struct BLASBuild {
    UINT meshID = 0;
    std::vector<uint32_t> triIndices;
    std::vector<BLASNode> nodes; // nodes[0] is root after build
    std::vector<uint32_t> leafTriList;
};

class LightTreeBuilder {
  private:
    struct TItem {
        uint32_t idx;
        Aabb a;
        XMFLOAT3 c;
        float p;
        Cone cone;
        uint32_t primCount;
        float sumP, sumP2;
    };
    std::vector<LightTreeTrail> m_triBitTrails;
    std::vector<LightTreeTrail> m_slotBitTrails;

  public:
    struct Settings {
        bool compactGpuNodes = true;
        uint32_t maxLeafTris = 1;
        bool useTwoLevel = true;
        uint32_t buildBins = 64;
        enum class Heuristic { SAOH, SAH };
        Heuristic heuristic = Heuristic::SAOH;
    };

    struct GpuBuffers {
        ComPtr<ID3D12Resource> TLASNodes;
        ComPtr<ID3D12Resource> BLASNodes;
        ComPtr<ID3D12Resource> BLASRanges;
        ComPtr<ID3D12Resource> LeafTriIndex;
        ComPtr<ID3D12Resource> LeafAliasProb;
        ComPtr<ID3D12Resource> LeafAliasIdx;
        ComPtr<ID3D12Resource> TriToBLAS;
        ComPtr<ID3D12Resource> TriBitTrail;
        ComPtr<ID3D12Resource> BLASBitTrail;
        ComPtr<ID3D12Resource> Slots;

        std::vector<ComPtr<ID3D12Resource>> staging;
    };

    const std::vector<LightTLASNodeGpu>& GetCpuTLASNodes() const { return m_tlas; }
    const std::vector<LightInstanceRef>& Slots() const { return m_slots; }
    const std::vector<LightSlotGpu>& SlotRecords() const { return m_slotGpu; }

    uint32_t BLASIndexOfMesh(UINT meshID) const {
        const auto it = m_blasOfMesh.find(meshID);
        return it == m_blasOfMesh.end() ? UINT32_MAX : it->second;
    }

    // Trails address single triangles: one per leaf.
    void enforceLeafInvariant() {
        if (m_cfg.maxLeafTris != 1u) {
            LT_WARN(L"maxLeafTris=" << m_cfg.maxLeafTris << L" ignored; BLAS leaves are fixed at 1 triangle.");
            m_cfg.maxLeafTris = 1u;
        }
    }

    // Per-mesh trees, then one over light instances.
    void Build(const std::vector<::LightTriangle>& records, const std::vector<LightInstanceRef>& slots,
               const std::vector<InstanceXformCPU>& xforms, const Settings& cfg = {}) {
        LT_TIME_SCOPE(L"Build()");
        LT_LOG(L"Build: records=" << records.size() << L", slots=" << slots.size() << L", xforms=" << xforms.size()
                                  << L", maxLeafTris=" << cfg.maxLeafTris << L", twoLevel="
                                  << (cfg.useTwoLevel ? L"true" : L"false") << L", bins=" << cfg.buildBins);
        if (records.empty())
            LT_WARN(L"No emissive records; tree will be empty.");
        m_cfg = cfg;
        m_tris = &records;
        m_xforms = &xforms;
        m_slots = slots;
        enforceLeafInvariant();
        rebuildXformCaches();
        buildBLASes_SAOH();
        buildSlotRecords();
        buildTLAS_SAOH();
        m_xforms = nullptr;
        LT_LOG(L"Build done: BLAS=" << m_blas.size() << L", TLAS nodes=" << m_tlas.size());
    }

    void Build(const std::vector<::LightTriangle>& tris, const Settings& cfg = {}) {
        Build(tris, SlotsFromRecords(tris), std::vector<InstanceXformCPU>{}, cfg);
    }
    void Build(const std::vector<::LightTriangle>& tris, const std::vector<InstanceXformCPU>& xforms,
               const Settings& cfg = {}) {
        Build(tris, SlotsFromRecords(tris), xforms, cfg);
    }
    static std::vector<LightInstanceRef> SlotsFromRecords(const std::vector<::LightTriangle>& tris) {
        std::map<UINT, bool> ids;
        for (const auto& t : tris)
            ids[t.meshID] = true;
        std::vector<LightInstanceRef> s;
        s.reserve(ids.size());
        for (const auto& kv : ids)
            s.push_back({kv.first, kv.first});
        return s;
    }

    struct SingleBLAS {
        std::vector<LightBLASNodeGpu> nodes;
        std::vector<uint32_t> leafTriLocal;
        std::vector<LightTreeTrail> trails;
    };
    // Also builds per-triangle traversal trails.
    static void BuildSingleBLAS(const std::vector<::LightTriangle>& tris, SingleBLAS& out, uint32_t bins = 32u) {
        out.nodes.clear();
        out.leafTriLocal.clear();
        out.trails.clear();
        if (tris.empty())
            return;
        LightTreeBuilder b;
        b.m_cfg.buildBins = bins;
        b.m_cfg.maxLeafTris = 1u;
        b.m_tris = &tris;
        b.m_triBitTrails.assign(tris.size(), 0u);
        BLASBuild blas;
        blas.nodes.reserve(tris.size() * 2u + 8u);
        std::vector<TmpTri> tmp;
        tmp.reserve(tris.size());
        for (uint32_t i = 0; i < (uint32_t)tris.size(); ++i)
            tmp.push_back(makeTmpTri(tris[i], i));
        b.buildBLASRecursive_SAOH(tmp, blas, 0u, (uint32_t)tmp.size(), 0u, 0u);

        // BFS order keeps siblings contiguous.
        std::vector<uint32_t> order;
        order.reserve(blas.nodes.size());
        std::vector<uint32_t> remap(blas.nodes.size(), 0xFFFFFFFFu);
        order.push_back(0u);
        remap[0] = 0u;
        for (size_t q = 0; q < order.size(); ++q) {
            const BLASNode& n = blas.nodes[order[q]];
            for (uint32_t c = 0; c < n.childCount; ++c) {
                remap[n.firstChild + c] = (uint32_t)order.size();
                order.push_back(n.firstChild + c);
            }
        }
        out.nodes.reserve(order.size());
        for (uint32_t old : order) {
            LightBLASNodeGpu g = toGpu(blas.nodes[old]);
            if (g.childCount)
                g.firstChild = remap[blas.nodes[old].firstChild];
            out.nodes.push_back(g);
        }
        out.leafTriLocal = std::move(blas.leafTriList);
        out.trails = std::move(b.m_triBitTrails);
        b.m_tris = nullptr;
    }

    // Flattens mesh trees into shared buffers.
    void UploadAll(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList) {
        LT_TIME_SCOPE(L"UploadAll()");
        const uint32_t nodeWords = LightBLASNodeStride(m_cfg.compactGpuNodes) / sizeof(uint32_t);
        std::vector<uint32_t> gpuBlasNodes;
        gpuBlasNodes.reserve((totalBLASNodeCount() + m_blas.size()) * nodeWords);
        std::vector<uint32_t> gpuLeafTriIndex;
        gpuLeafTriIndex.reserve(totalLeafIndexCount());
        std::vector<BlasRangeGpu> gpuRanges;
        gpuRanges.reserve(m_blas.size());

        for (const auto& b : m_blas) {
            BlasRangeGpu r{};
            r.nodeOffset = static_cast<uint32_t>(gpuBlasNodes.size() / nodeWords);
            r.nodeCount = static_cast<uint32_t>(b.nodes.size());
            r.triIndexOffset = static_cast<uint32_t>(gpuLeafTriIndex.size());
            r.triIndexCount = static_cast<uint32_t>(b.leafTriList.size());

            std::vector<LightBLASNodeGpu> nodes;
            nodes.reserve(b.nodes.size());
            for (const auto& n : b.nodes) nodes.push_back(toGpu(n));
            const auto packed = EncodeLightBLAS(nodes, m_cfg.compactGpuNodes, r.triIndexOffset);
            gpuBlasNodes.insert(gpuBlasNodes.end(), packed.begin(), packed.end());

            gpuLeafTriIndex.insert(gpuLeafTriIndex.end(), b.leafTriList.begin(), b.leafTriList.end());
            gpuRanges.push_back(r);
        }

        const auto gpuTlasNodes = m_cfg.compactGpuNodes ? PackTLAS(m_tlas) : std::vector<LightTLASNodePacked>{};

        LT_LOG(L"UploadAll: BLASNodes=" << gpuBlasNodes.size() / nodeWords << L", LeafTriIndex=" << gpuLeafTriIndex.size()
                                        << L", BLASRanges=" << gpuRanges.size() << L", Slots=" << m_slotGpu.size()
                                        << L", TLASNodes=" << m_tlas.size());
        const auto KiB = [](uint64_t b) { return b / 1024.0; };
        LT_LOG(L"  sizes: TLAS=" << KiB(m_tlas.size() * LightTLASNodeStride(m_cfg.compactGpuNodes)) << L" KiB" << L", BLAS="
                                 << KiB(gpuBlasNodes.size() * sizeof(uint32_t)) << L" KiB" << L", Ranges="
                                 << KiB(gpuRanges.size() * sizeof(BlasRangeGpu)) << L" KiB" << L", Slots="
                                 << KiB(m_slotGpu.size() * sizeof(LightSlotGpu)) << L" KiB" << L", LeafIdx="
                                 << KiB(gpuLeafTriIndex.size() * sizeof(uint32_t)) << L" KiB");

        std::vector<uint32_t> triToBLAS(m_tris ? m_tris->size() : 0, 0xFFFFFFFFu);
        for (uint32_t bIdx = 0; bIdx < m_blas.size(); ++bIdx) {
            const auto& b = m_blas[bIdx];
            for (uint32_t j = 0; j < b.leafTriList.size(); ++j) {
                triToBLAS[b.leafTriList[j]] = bIdx;
            }
        }

        m_gpu = {};
        m_gpu.BLASNodes = uploadVector(device, cmdList, gpuBlasNodes);
        m_gpu.LeafTriIndex = uploadVector(device, cmdList, gpuLeafTriIndex);
        m_gpu.BLASRanges = uploadVector(device, cmdList, gpuRanges);
        m_gpu.TLASNodes = m_cfg.compactGpuNodes ? uploadVector(device, cmdList, gpuTlasNodes) : uploadVector(device, cmdList, m_tlas);
        m_gpu.TriToBLAS = uploadVector(device, cmdList, triToBLAS);
        m_gpu.TriBitTrail = uploadVector(device, cmdList, m_triBitTrails);
        m_gpu.BLASBitTrail = uploadVector(device, cmdList, m_slotBitTrails);
        m_gpu.Slots = uploadVector(device, cmdList, m_slotGpu);
    }

    static void writeBufSrv(ID3D12Device* device, ID3D12Resource* res, UINT numElems, UINT stride, DXGI_FORMAT fmt,
                            D3D12_CPU_DESCRIPTOR_HANDLE h) {
        D3D12_SHADER_RESOURCE_VIEW_DESC d{};
        d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        d.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        d.Buffer.FirstElement = 0;
        d.Buffer.NumElements = res ? numElems : 1u;
        d.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
        if (stride == 0) {
            d.Format = fmt;
            d.Buffer.StructureByteStride = 0;
        } else {
            d.Format = DXGI_FORMAT_UNKNOWN;
            d.Buffer.StructureByteStride = stride;
        }
        device->CreateShaderResourceView(res, &d, h);
    }

    void WriteSrvs(ID3D12Device* device, D3D12_CPU_DESCRIPTOR_HANDLE dst) const {
        LT_TIME_SCOPE(L"WriteSrvs()");
        const UINT inc = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        auto elems = [](ID3D12Resource* r, size_t stride) -> UINT {
            return r ? static_cast<UINT>(r->GetDesc().Width / stride) : 0u;
        };
        if (!m_gpu.TLASNodes)
            LT_WARN(L"WriteSrvs: TLASNodes is null (no light slots), writing a null SRV.");
        writeBufSrv(device, m_gpu.TLASNodes.Get(), elems(m_gpu.TLASNodes.Get(), 16u),
                    16u, DXGI_FORMAT_UNKNOWN, dst);
        dst.ptr += inc;
        writeBufSrv(device, m_gpu.BLASNodes.Get(), elems(m_gpu.BLASNodes.Get(), 16u),
                    16u, DXGI_FORMAT_UNKNOWN, dst);
        dst.ptr += inc;
        writeBufSrv(device, m_gpu.BLASRanges.Get(), elems(m_gpu.BLASRanges.Get(), sizeof(BlasRangeGpu)),
                    sizeof(BlasRangeGpu), DXGI_FORMAT_UNKNOWN, dst);
        dst.ptr += inc;
        writeBufSrv(device, m_gpu.LeafTriIndex.Get(), elems(m_gpu.LeafTriIndex.Get(), 4), 0, DXGI_FORMAT_R32_UINT, dst);
    }

    void WriteSlotSrv(ID3D12Device* device, D3D12_CPU_DESCRIPTOR_HANDLE dst) const {
        const UINT n = m_gpu.Slots ? static_cast<UINT>(m_gpu.Slots->GetDesc().Width / sizeof(LightSlotGpu)) : 0u;
        writeBufSrv(device, m_gpu.Slots.Get(), n, sizeof(LightSlotGpu), DXGI_FORMAT_UNKNOWN, dst);
    }

    void WriteAliasSrvs(ID3D12Device* device, D3D12_CPU_DESCRIPTOR_HANDLE dst) const {
        LT_TIME_SCOPE(L"WriteAliasSrvs()");
        const UINT inc = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        auto makeTyped = [&](ID3D12Resource* res, DXGI_FORMAT fmt, UINT n, D3D12_CPU_DESCRIPTOR_HANDLE h) {
            D3D12_SHADER_RESOURCE_VIEW_DESC d{};
            d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            d.Format = fmt;
            d.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            d.Buffer.FirstElement = 0;
            d.Buffer.NumElements = n;
            d.Buffer.StructureByteStride = 0;
            d.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
            device->CreateShaderResourceView(res, &d, h);
        };
        if (m_gpu.LeafAliasProb) {
            auto rd = m_gpu.LeafAliasProb->GetDesc();
            const UINT n = static_cast<UINT>(rd.Width / 4);
            LT_LOG(L"WriteAliasSrvs: LeafAliasProb count=" << n);
            makeTyped(m_gpu.LeafAliasProb.Get(), DXGI_FORMAT_R32_FLOAT, n, dst);
        } else
            LT_WARN(L"WriteAliasSrvs: LeafAliasProb is null (no leaves?)");
        dst.ptr += inc;
        if (m_gpu.LeafAliasIdx) {
            auto rd = m_gpu.LeafAliasIdx->GetDesc();
            const UINT n = static_cast<UINT>(rd.Width / 4);
            LT_LOG(L"WriteAliasSrvs: LeafAliasIdx count=" << n);
            makeTyped(m_gpu.LeafAliasIdx.Get(), DXGI_FORMAT_R32_UINT, n, dst);
        } else
            LT_WARN(L"WriteAliasSrvs: LeafAliasIdx is null (no leaves?)");
    }

    void WriteLookupSrvs(ID3D12Device* device, D3D12_CPU_DESCRIPTOR_HANDLE dst) const {
        const UINT inc = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        auto elems = [](ID3D12Resource* r, size_t stride) -> UINT {
            return r ? static_cast<UINT>(r->GetDesc().Width / stride) : 0u;
        };

        writeBufSrv(device, m_gpu.TriToBLAS.Get(), elems(m_gpu.TriToBLAS.Get(), 4), 0, DXGI_FORMAT_R32_UINT, dst);
        dst.ptr += inc;

        writeBufSrv(device, m_gpu.TriBitTrail.Get(), elems(m_gpu.TriBitTrail.Get(), sizeof(LightTreeTrail)), 0,
                    DXGI_FORMAT_R32G32_UINT, dst);
        dst.ptr += inc;

        writeBufSrv(device, m_gpu.BLASBitTrail.Get(), elems(m_gpu.BLASBitTrail.Get(), sizeof(LightTreeTrail)), 0,
                    DXGI_FORMAT_R32G32_UINT, dst);
    }

    // Only after the uploads complete on the GPU.
    void ReleaseStaging() {
        LT_TIME_SCOPE(L"ReleaseStaging()");
        LT_LOG(L"ReleaseStaging: " << m_gpu.staging.size() << L" upload buffers freed");
        m_gpu.staging.clear();
    }

    bool CompactGpuNodes() const { return m_cfg.compactGpuNodes; }

    const GpuBuffers& GetGpu() const { return m_gpu; }

    uint32_t BLASCount() const { return static_cast<uint32_t>(m_blas.size()); }
    uint32_t SlotCount() const { return static_cast<uint32_t>(m_slots.size()); }

    void PrintMetrics() const {
        LT_TIME_SCOPE(L"PrintMetrics()");
        struct ND {
            uint32_t i;
            uint32_t d;
        };

        if (m_tlas.empty()) {
            LT_WARN(L"TLAS is empty.");
        } else {
            uint64_t nodeCount = 0, inner = 0, leaf = 0, leafDepthSum = 0, childrenSum = 0;
            uint32_t maxDepth = 0, maxChildren = 0;

            std::vector<ND> st;
            st.reserve(m_tlas.size());
            st.push_back({0, 0});
            while (!st.empty()) {
                ND cur = st.back();
                st.pop_back();
                const auto& n = m_tlas[cur.i];
                nodeCount++;
                maxDepth = (std::max)(maxDepth, cur.d);
                if (n.childCount == 0) {
                    leaf++;
                    leafDepthSum += cur.d;
                } else {
                    inner++;
                    childrenSum += n.childCount;
                    maxChildren = (std::max)(maxChildren, n.childCount);
                    for (uint32_t c = 0; c < n.childCount; ++c)
                        st.push_back({n.firstChild + c, cur.d + 1});
                }
            }

            const double avgLeafDepth = leaf ? double(leafDepthSum) / double(leaf) : 0.0;
            const double avgChildren = inner ? double(childrenSum) / double(inner) : 0.0;

            LT_LOG(L"TLAS: nodes=" << nodeCount << L" (inner=" << inner << L", leaf=" << leaf << L")" << L", maxDepth="
                                   << maxDepth << L", avgLeafDepth=" << avgLeafDepth << L", avgChildren=" << avgChildren
                                   << L", maxChildren=" << maxChildren);
        }

        uint64_t allNodes = 0, allInner = 0, allLeaf = 0, allLeafDepthSum = 0, allLeafTriSum = 0;
        uint32_t allMaxDepth = 0, globalMinLeafTri = UINT32_MAX, globalMaxLeafTri = 0;

        for (uint32_t b = 0; b < m_blas.size(); ++b) {
            const auto& B = m_blas[b];
            if (B.nodes.empty()) {
                LT_WARN(L"BLAS[" << b << L"] is empty.");
                continue;
            }

            uint64_t nodes = 0, inner = 0, leaf = 0, leafDepthSum = 0, leafTriSum = 0;
            uint32_t maxDepth = 0, minLeafTri = UINT32_MAX, maxLeafTri = 0;

            std::vector<ND> st;
            st.reserve(B.nodes.size());
            st.push_back({0, 0});
            while (!st.empty()) {
                ND cur = st.back();
                st.pop_back();
                const auto& n = B.nodes[cur.i];
                nodes++;
                maxDepth = (std::max)(maxDepth, cur.d);
                if (n.childCount == 0) {
                    leaf++;
                    leafDepthSum += cur.d;
                    minLeafTri = (std::min)(minLeafTri, n.triCount);
                    maxLeafTri = (std::max)(maxLeafTri, n.triCount);
                    leafTriSum += n.triCount;
                } else {
                    inner++;
                    for (uint32_t c = 0; c < n.childCount; ++c)
                        st.push_back({n.firstChild + c, cur.d + 1});
                }
            }

            const double avgLeafDepth = leaf ? double(leafDepthSum) / double(leaf) : 0.0;
            const double avgLeafTris = leaf ? double(leafTriSum) / double(leaf) : 0.0;

            LT_LOG(L"BLAS[" << b << L"] mesh " << B.meshID << L": nodes=" << nodes << L" (inner=" << inner << L", leaf="
                            << leaf << L")" << L", maxDepth=" << maxDepth << L", avgLeafDepth=" << avgLeafDepth
                            << L", leafTris(min/avg/max)=" << (minLeafTri == UINT32_MAX ? 0 : minLeafTri) << L"/"
                            << avgLeafTris << L"/" << maxLeafTri);

            allNodes += nodes;
            allInner += inner;
            allLeaf += leaf;
            allLeafDepthSum += leafDepthSum;
            allLeafTriSum += leafTriSum;
            allMaxDepth = (std::max)(allMaxDepth, maxDepth);
            globalMinLeafTri = (std::min)(globalMinLeafTri, minLeafTri);
            globalMaxLeafTri = (std::max)(globalMaxLeafTri, maxLeafTri);
        }

        if (!m_blas.empty()) {
            const double avgLeafDepthAll = allLeaf ? double(allLeafDepthSum) / double(allLeaf) : 0.0;
            const double avgLeafTrisAll = allLeaf ? double(allLeafTriSum) / double(allLeaf) : 0.0;

            LT_LOG(L"BLAS (all): nodes=" << allNodes << L" (inner=" << allInner << L", leaf=" << allLeaf << L")"
                                         << L", maxDepth=" << allMaxDepth << L", avgLeafDepth=" << avgLeafDepthAll
                                         << L", leafTris(min/avg/max)="
                                         << (globalMinLeafTri == UINT32_MAX ? 0 : globalMinLeafTri) << L"/"
                                         << avgLeafTrisAll << L"/" << globalMaxLeafTri << L", slots="
                                         << m_slots.size());
        }
    }

  private:
    const std::vector<::LightTriangle>* m_tris = nullptr;
    Settings m_cfg{};
    const std::vector<InstanceXformCPU>* m_xforms = nullptr;
    std::vector<XMFLOAT4X4> m_worldByInstance;
    std::vector<XMFLOAT3X3> m_normalByInstance;

    std::vector<LightInstanceRef> m_slots;
    std::vector<LightSlotGpu> m_slotGpu;
    std::unordered_map<UINT, uint32_t> m_blasOfMesh;
    std::vector<BLASBuild> m_blas;
    std::vector<LightTLASNodeGpu> m_tlas;
    GpuBuffers m_gpu{};

    struct TmpTri {
        uint32_t triIndex;
        XMFLOAT3 centroid;
        Aabb aabb;
        float power;
        Cone cone;
    };

    static TmpTri makeTmpTri(const ::LightTriangle& t, uint32_t triIndex) {
        const XMFLOAT3 c{(t.x.x + t.y.x + t.z.x) / 3.f, (t.x.y + t.y.y + t.z.y) / 3.f, (t.x.z + t.y.z + t.z.z) / 3.f};
        const Aabb a = {min3(t.x, min3(t.y, t.z)), max3(t.x, max3(t.y, t.z))};
        const XMFLOAT3 nL = cross3(sub3(t.y, t.x), sub3(t.z, t.x));
        Cone lc;
        if (length3(nL) < 1e-12f) {
            lc.axis = {0, 0, 1};
            lc.theta_o = LT_PI;
            lc.theta_e = LT_HALF_PI;
        } else {
            lc.axis = normalize3(nL);
            lc.theta_o = 0.f;
            lc.theta_e = LT_HALF_PI;
        }
        return TmpTri{triIndex, c, a, t.weight, lc};
    }

    void rebuildXformCaches() {
        m_worldByInstance.clear();
        m_normalByInstance.clear();
        if (!m_xforms || m_xforms->empty())
            return;
        const size_t n = m_xforms->size();
        m_worldByInstance.resize(n);
        m_normalByInstance.resize(n);
        for (size_t i = 0; i < n; ++i) {
            m_worldByInstance[i] = (*m_xforms)[i].objectToWorld;
            computeNormal33FromWorld(m_worldByInstance[i], m_normalByInstance[i]);
        }
    }

    const XMFLOAT4X4& worldXformFor(UINT id) const {
        if (id < m_worldByInstance.size())
            return m_worldByInstance[id];
        if (!m_worldByInstance.empty())
            LT_WARN(L"InstanceID " << id << L" out of range for world matrix; using identity.");
        return LT_IDENTITY_4X4;
    }
    const XMFLOAT3X3& normalXformFor(UINT id) const {
        if (id < m_normalByInstance.size())
            return m_normalByInstance[id];
        return LT_IDENTITY_3X3;
    }

    void buildBLASes_SAOH() {
        LT_TIME_SCOPE(L"buildBLASes_SAOH()");
        LT_LOG(L"Grouping " << (m_tris ? m_tris->size() : 0) << L" emissive records by mesh...");
        m_blas.clear();
        m_blasOfMesh.clear();
        if (!m_tris) {
            LT_WARN(L"m_tris == nullptr");
            return;
        }
        std::map<UINT, std::vector<uint32_t>> groups;
        for (uint32_t i = 0; i < m_tris->size(); ++i)
            groups[(*m_tris)[i].meshID].push_back(i);
        LT_LOG(L"buildBLASes: groups=" << groups.size());
        m_blas.reserve(groups.size());

        m_triBitTrails.assign(m_tris->size(), 0u);
        for (auto& kv : groups) {
            const auto& idxs = kv.second;
            LT_LOG(L"  BLAS[" << m_blas.size() << L"] mesh " << kv.first << L" tris=" << idxs.size());
            BLASBuild b;
            b.meshID = kv.first;
            b.triIndices = idxs;

            b.nodes.reserve(static_cast<size_t>(idxs.size()) * 2u + 32u);
            std::vector<TmpTri> tmp;
            tmp.reserve(idxs.size());

            for (uint32_t j = 0; j < idxs.size(); ++j)
                tmp.push_back(makeTmpTri((*m_tris)[idxs[j]], idxs[j]));

            buildBLASRecursive_SAOH(tmp, b, 0, static_cast<uint32_t>(tmp.size()), 0u, 0u);

            uint32_t leafs = 0, inner = 0;
            for (const auto& n : b.nodes)
                (n.isLeaf() ? leafs : inner)++;
            LT_LOG(L"    nodes=" << b.nodes.size() << L" (inner=" << inner << L", leaf=" << leafs << L")"
                                 << L", leafTriIndexCount=" << b.leafTriList.size());
            m_blasOfMesh[kv.first] = static_cast<uint32_t>(m_blas.size());
            m_blas.push_back(std::move(b));
        }
    }

    void buildSlotRecords() {
        std::vector<uint32_t> nodeOff(m_blas.size());
        uint32_t n = 0;
        for (size_t i = 0; i < m_blas.size(); ++i) {
            nodeOff[i] = n;
            n += static_cast<uint32_t>(m_blas[i].nodes.size()) + (m_cfg.compactGpuNodes ? 1u : 0u);
        }
        m_slotGpu.assign(m_slots.size(), LightSlotGpu{});
        for (size_t s = 0; s < m_slots.size(); ++s) {
            const uint32_t b = BLASIndexOfMesh(m_slots[s].meshID);
            if (b == UINT32_MAX) {
                LT_WARN(L"slot " << s << L" names mesh " << m_slots[s].meshID
                                 << L" which has no emissive records; it gets no leaf.");
                m_slotGpu[s] = makeSlotRecord(m_slots[s].instanceID, 0u, LT_IDENTITY_4X4);
                m_slotGpu[s].powerScale = 0.f;
                continue;
            }
            m_slotGpu[s] = makeSlotRecord(m_slots[s].instanceID, nodeOff[b], worldXformFor(m_slots[s].instanceID));
        }
    }

    struct Agg {
        bool valid = false;
        Aabb a;
        float E = 0;
        Cone cone{};
        uint32_t N = 0;
        float sumP = 0.f;
        float sumP2 = 0.f;
    };

    static void aggAdd(Agg& A, const TmpTri& t) {
        if (!A.valid) {
            A.valid = true;
            A.a = t.aabb;
            A.E = t.power;
            A.cone = t.cone;
            A.N = 1;
            A.sumP = t.power;
            A.sumP2 = t.power * t.power;
            return;
        }
        A.a = unionAabb(A.a, t.aabb);
        A.E += t.power;
        A.cone = coneUnion(A.cone, t.cone);
        A.N++;
        A.sumP += t.power;
        A.sumP2 += t.power * t.power;
    }

    static void aggMerge(Agg& A, const Agg& B) {
        if (!B.valid)
            return;
        if (!A.valid) {
            A = B;
            return;
        }
        A.a = unionAabb(A.a, B.a);
        A.E += B.E;
        A.cone = coneUnion(A.cone, B.cone);
        A.N += B.N;
        A.sumP += B.sumP;
        A.sumP2 += B.sumP2;
    }

    // Builds four-way nodes.
    uint32_t buildBLASRecursive_SAOH(std::vector<TmpTri>& tmp, BLASBuild& out, uint32_t begin, uint32_t end,
                                     LightTreeTrail bitTrail, uint32_t depth, uint32_t destination = UINT32_MAX) {
        const uint32_t nodeIdx = destination == UINT32_MAX ? static_cast<uint32_t>(out.nodes.size()) : destination;
        if (destination == UINT32_MAX) out.nodes.emplace_back();

        auto nodeAt = [&](uint32_t i) -> BLASNode& { return out.nodes[i]; };
        BLASNode& N0 = nodeAt(nodeIdx);

        Agg parent{};
        for (uint32_t i = begin; i < end; ++i)
            aggAdd(parent, tmp[i]);

        N0.aabb = parent.a;
        N0.power = parent.E;
        N0.cone = parent.cone;
        N0.primCount = parent.N;
        N0.sumPower = parent.sumP;
        N0.sumPowerSq = parent.sumP2;

        const uint32_t count = end - begin;
        if (count <= m_cfg.maxLeafTris) {
            N0.triFirst = static_cast<uint32_t>(out.leafTriList.size());
            N0.triCount = count;
            for (uint32_t i = begin; i < end; ++i) {
                const uint32_t tri = tmp[i].triIndex;
                out.leafTriList.push_back(tri);
                m_triBitTrails[tri] = bitTrail;
            }
            N0.firstChild = 0xFFFFFFFF;
            N0.childCount = 0;
            return nodeIdx;
        }

        auto findBinarySplit = [&](uint32_t b0, uint32_t e0, int& axisOut, float& splitPosOut,
                                   uint32_t& midOut) -> bool {
            Agg parentL{};
            for (uint32_t i = b0; i < e0; ++i)
                aggAdd(parentL, tmp[i]);
            const Aabb aabb = parentL.a;
            const XMFLOAT3 ext = aabbExtent(aabb);
            // Median split before trail bits run out.
            if (LightTreeNeedsBalancedSplit(e0 - b0, depth)) {
                axisOut = (ext.y > ext.x && ext.y >= ext.z) ? 1 : (ext.z > ext.x ? 2 : 0);
                midOut = b0 + (e0 - b0) / 2u;
                std::nth_element(tmp.begin() + b0, tmp.begin() + midOut, tmp.begin() + e0,
                                 [&](const TmpTri& a, const TmpTri& b) {
                                     return (&a.centroid.x)[axisOut] < (&b.centroid.x)[axisOut];
                                 });
                splitPosOut = (&tmp[midOut].centroid.x)[axisOut];
                return true;
            }
            const float lenX = ext.x, lenY = ext.y, lenZ = ext.z, lenMax = (std::fmax)(lenX, (std::fmax)(lenY, lenZ));
            const float parentMA = (std::fmax)(1e-12f, aabbSurfaceArea(aabb));
            const float parentMO = (std::fmax)(1e-12f, orientationMeasure(parentL.cone));

            struct Best {
                float cost = std::numeric_limits<float>::infinity();
                int axis = -1;
                float splitPos = 0;
            } best;
            const uint32_t B = (std::fmax)(4u, (std::fmin)(64u, m_cfg.buildBins));

            for (int axis = 0; axis < 3; ++axis) {
                float mn = (&tmp[b0].centroid.x)[axis], mx = mn;
                for (uint32_t i = b0; i < e0; ++i) {
                    float v = (&tmp[i].centroid.x)[axis];
                    mn = (std::fmin)(mn, v);
                    mx = (std::fmax)(mx, v);
                }
                float span = mx - mn;
                if (span <= 1e-20f)
                    continue;

                std::vector<Agg> bins(B);
                float invSpan = 1.0f / span;
                for (uint32_t i = b0; i < e0; ++i) {
                    float v = (&tmp[i].centroid.x)[axis];
                    uint32_t bi = (std::fmin)(B - 1u, (uint32_t)std::floor((v - mn) * invSpan * B));
                    aggAdd(bins[bi], tmp[i]);
                }

                std::vector<Agg> pref(B), suff(B);
                for (uint32_t i = 0; i < B; ++i) {
                    if (i == 0)
                        pref[i] = bins[i];
                    else {
                        pref[i] = pref[i - 1];
                        aggMerge(pref[i], bins[i]);
                    }
                }
                for (int i = (int)B - 1; i >= 0; --i) {
                    if ((uint32_t)i == B - 1)
                        suff[i] = bins[i];
                    else {
                        suff[i] = suff[i + 1];
                        aggMerge(suff[i], bins[i]);
                    }
                }

                float length_i = (axis == 0 ? lenX : (axis == 1 ? lenY : lenZ));
                float Kr = (length_i > 1e-20f) ? (lenMax / length_i) : 1e6f;

                for (uint32_t s = 1; s < B; ++s) {
                    const Agg& L = pref[s - 1];
                    const Agg& R = suff[s];
                    if (!L.valid || !R.valid)
                        continue;

                    float ML = aabbSurfaceArea(L.a), MR = aabbSurfaceArea(R.a);
                    float MoL = orientationMeasure(L.cone);
                    float MoR = orientationMeasure(R.cone);

                    float cost;
                    if (m_cfg.heuristic == Settings::Heuristic::SAH) {
                        cost = L.N * ML + R.N * MR;
                    } else {
                        cost = Kr * (L.E * ML * MoL + R.E * MR * MoR) / (parentMA * parentMO);
                    }
                    if (cost < best.cost) {
                        best.cost = cost;
                        best.axis = axis;
                        best.splitPos = mn + (span * (float)s / (float)B);
                    }
                }
            }

            if (best.axis < 0 || !std::isfinite(best.cost))
                return false;

            auto itMid = std::partition(tmp.begin() + b0, tmp.begin() + e0,
                                        [&](const TmpTri& t) { return (&t.centroid.x)[best.axis] < best.splitPos; });

            uint32_t mid = static_cast<uint32_t>(itMid - (tmp.begin() + b0)) + b0;
            if (mid == b0 || mid == e0) {
                mid = (b0 + e0) / 2;
                std::nth_element(tmp.begin() + b0, tmp.begin() + mid, tmp.begin() + e0,
                                 [&](const TmpTri& A, const TmpTri& B) {
                                     return (&A.centroid.x)[best.axis] < (&B.centroid.x)[best.axis];
                                 });
            }

            axisOut = best.axis;
            splitPosOut = best.splitPos;
            midOut = mid;
            return true;
        };

        int ax = 0;
        float pos = 0.f;
        uint32_t mid = 0;
        bool ok = findBinarySplit(begin, end, ax, pos, mid);
        if (!ok) {
            mid = (begin + end) / 2;
        }

        struct Range {
            uint32_t b, e;
        };
        Range buckets[4];
        uint32_t bucketCount = 0;

        auto pushOrSplitOnce = [&](uint32_t b, uint32_t e) {
            if (e <= b)
                return;
            const uint32_t c = e - b;
            if (c <= m_cfg.maxLeafTris) {
                buckets[bucketCount++] = {b, e};
                return;
            }
            int ax2;
            float pos2;
            uint32_t mid2;
            if (findBinarySplit(b, e, ax2, pos2, mid2) && mid2 > b && mid2 < e) {
                buckets[bucketCount++] = {b, mid2};
                buckets[bucketCount++] = {mid2, e};
            } else {
                mid2 = (b + e) / 2;
                buckets[bucketCount++] = {b, mid2};
                buckets[bucketCount++] = {mid2, e};
            }
        };

        pushOrSplitOnce(begin, mid);
        pushOrSplitOnce(mid, end);

        // Contiguous child slots before recursing.
        nodeAt(nodeIdx).firstChild = static_cast<uint32_t>(out.nodes.size());
        nodeAt(nodeIdx).childCount = bucketCount;
        for (uint32_t i = 0; i < bucketCount; i++)
            out.nodes.emplace_back();

        for (uint32_t c = 0; c < bucketCount; ++c) {
            const LightTreeTrail childTrail = AppendLightTreeTrail(bitTrail, c, depth);
            const uint32_t desired = nodeAt(nodeIdx).firstChild + c;
            buildBLASRecursive_SAOH(tmp, out, buckets[c].b, buckets[c].e, childTrail, depth + 1u, desired);
        }

        // Re-fetch: recursion may reallocate nodes.
        BLASNode& N = nodeAt(nodeIdx);
        N.triFirst = (std::numeric_limits<uint32_t>::max)();
        N.triCount = 0;
        for (uint32_t c = 0; c < N.childCount; ++c) {
            const BLASNode& C = out.nodes[N.firstChild + c];
            N.triFirst = (std::min)(N.triFirst, C.triFirst);
            N.triCount += C.triCount;
        }

        return nodeIdx;
    }

    // Leaves bound transformed mesh roots; trees stay shared.
    void buildTLAS_SAOH() {
        LT_TIME_SCOPE(L"buildTLAS_SAOH()");

        LT_LOG(L"buildTLAS: slots=" << m_slots.size() << L" over BLASes=" << m_blas.size());
        std::vector<TItem> items;
        items.reserve(m_slots.size());
        for (uint32_t s = 0; s < m_slots.size(); ++s) {
            const uint32_t bi = BLASIndexOfMesh(m_slots[s].meshID);
            if (bi == UINT32_MAX || m_blas[bi].nodes.empty())
                continue;
            const auto& r = m_blas[bi].nodes[0];
            const UINT inst = m_slots[s].instanceID;
            const auto& world = worldXformFor(inst);
            const float scale = m_slotGpu[s].powerScale;
            TItem it;
            it.idx = s;
            it.a = transformBounds(r.aabb, world);
            it.c = aabbCenter(it.a);
            it.p = r.power * scale;
            it.cone = r.cone;
            it.cone.axis = transformNormalW(r.cone.axis, normalXformFor(inst));
            // Nonuniform scale can widen the normal cone.
            if (!similarityTransform(world))
                it.cone.theta_o = LT_PI;

            it.primCount = r.primCount;
            it.sumP = r.sumPower * scale;
            it.sumP2 = r.sumPowerSq * scale * scale;
            items.push_back(it);
        }
        m_tlas.clear();

        m_slotBitTrails.assign(m_slots.size(), 0u);
        if (items.empty()) {
            LT_WARN(L"buildTLAS: no items");
            return;
        }

        m_tlas.reserve(items.size() * 2u + 32u);

        buildTLASRecursive_SAOH(items, 0, static_cast<uint32_t>(items.size()), 0u, 0u);
        LT_LOG(L"buildTLAS done: TLAS nodes=" << m_tlas.size());
    }

    struct AggT {
        bool valid = false;
        Aabb a;
        float E = 0;
        Cone cone{};
        uint32_t N = 0;
        float sumP = 0, sumP2 = 0;
    };
    static void aggTAdd(AggT& A, const TItem& t) {
        if (!A.valid) {
            A.valid = true;
            A.a = t.a;
            A.E = t.p;
            A.cone = t.cone;
            A.N = t.primCount;
            A.sumP = t.sumP;
            A.sumP2 = t.sumP2;
            return;
        }
        A.a = unionAabb(A.a, t.a);
        A.E += t.p;
        A.cone = coneUnion(A.cone, t.cone);
        A.N += t.primCount;
        A.sumP += t.sumP;
        A.sumP2 += t.sumP2;
    }
    static void aggTMerge(AggT& A, const AggT& B) {
        if (!B.valid)
            return;
        if (!A.valid) {
            A = B;
            return;
        }
        A.a = unionAabb(A.a, B.a);
        A.E += B.E;
        A.cone = coneUnion(A.cone, B.cone);
        A.N += B.N;
        A.sumP += B.sumP;
        A.sumP2 += B.sumP2;
    }

    uint32_t buildTLASRecursive_SAOH(std::vector<TItem>& it, uint32_t begin, uint32_t end, LightTreeTrail bitTrail,
                                     uint32_t depth, uint32_t destination = UINT32_MAX) {
        const uint32_t nodeIdx = destination == UINT32_MAX ? static_cast<uint32_t>(m_tlas.size()) : destination;
        if (destination == UINT32_MAX) m_tlas.push_back({});

        auto nodeAt = [&](uint32_t i) -> LightTLASNodeGpu& { return m_tlas[i]; };
        LightTLASNodeGpu& N0 = nodeAt(nodeIdx);

        AggT parent{};
        for (uint32_t i = begin; i < end; ++i)
            aggTAdd(parent, it[i]);

        N0.bmin = parent.a.mn;
        N0.bmax = parent.a.mx;
        N0.power = parent.E;
        N0.axis = parent.cone.axis;
        N0.cosTheta_o = std::cos(clampf(parent.cone.theta_o, 0.f, LT_PI));
        N0.sinTheta_o = std::sqrt((std::fmax)(0.f, 1.f - N0.cosTheta_o * N0.cosTheta_o));

        N0.firstChild = 0xFFFFFFFF;
        N0.childCount = 0;
        N0.slot = UINT32_MAX;
        N0._pad = 0;

        const uint32_t count = end - begin;
        if (count == 1) {
            N0.slot = it[begin].idx;
            m_slotBitTrails[it[begin].idx] = bitTrail;
            return nodeIdx;
        }

        auto findBinarySplit = [&](uint32_t b0, uint32_t e0, int& axisOut, float& splitPosOut,
                                   uint32_t& midOut) -> bool {
            AggT parentL{};
            for (uint32_t i = b0; i < e0; ++i)
                aggTAdd(parentL, it[i]);

            const Aabb aabb = parentL.a;
            const XMFLOAT3 ext = aabbExtent(aabb);
            if (LightTreeNeedsBalancedSplit(e0 - b0, depth)) {
                axisOut = (ext.y > ext.x && ext.y >= ext.z) ? 1 : (ext.z > ext.x ? 2 : 0);
                midOut = b0 + (e0 - b0) / 2u;
                std::nth_element(it.begin() + b0, it.begin() + midOut, it.begin() + e0,
                                 [&](const TItem& a, const TItem& b) { return (&a.c.x)[axisOut] < (&b.c.x)[axisOut]; });
                splitPosOut = (&it[midOut].c.x)[axisOut];
                return true;
            }
            const float lenX = ext.x, lenY = ext.y, lenZ = ext.z, lenMax = (std::fmax)(lenX, (std::fmax)(lenY, lenZ));
            const float parentMA = (std::fmax)(1e-12f, aabbSurfaceArea(aabb));
            const float parentMO = (std::fmax)(1e-12f, orientationMeasure(parentL.cone));

            struct Best {
                float cost = std::numeric_limits<float>::infinity();
                int axis = -1;
                float pos = 0;
            } best;
            const uint32_t B = (std::fmax)(4u, (std::fmin)(64u, m_cfg.buildBins));

            for (int axis = 0; axis < 3; ++axis) {
                float mn = (&it[b0].c.x)[axis], mx = mn;
                for (uint32_t i = b0; i < e0; ++i) {
                    float v = (&it[i].c.x)[axis];
                    mn = (std::fmin)(mn, v);
                    mx = (std::fmax)(mx, v);
                }
                float span = mx - mn;
                if (span <= 1e-20f)
                    continue;

                std::vector<AggT> bins(B);
                float invSpan = 1.f / span;
                for (uint32_t i = b0; i < e0; ++i) {
                    float v = (&it[i].c.x)[axis];
                    uint32_t bi = (std::fmin)(B - 1u, (uint32_t)std::floor((v - mn) * invSpan * B));
                    aggTAdd(bins[bi], it[i]);
                }

                std::vector<AggT> pref(B), suff(B);
                for (uint32_t i = 0; i < B; ++i) {
                    if (i == 0)
                        pref[i] = bins[i];
                    else {
                        pref[i] = pref[i - 1];
                        aggTMerge(pref[i], bins[i]);
                    }
                }
                for (int i = (int)B - 1; i >= 0; --i) {
                    if ((uint32_t)i == B - 1)
                        suff[i] = bins[i];
                    else {
                        suff[i] = suff[i + 1];
                        aggTMerge(suff[i], bins[i]);
                    }
                }

                float length_i = (axis == 0 ? lenX : (axis == 1 ? lenY : lenZ));
                float Kr = (length_i > 1e-20f) ? (lenMax / length_i) : 1e6f;

                for (uint32_t s = 1; s < B; ++s) {
                    const AggT& L = pref[s - 1];
                    const AggT& R = suff[s];
                    if (!L.valid || !R.valid)
                        continue;
                    float ML = aabbSurfaceArea(L.a), MR = aabbSurfaceArea(R.a);
                    float MoL = orientationMeasure(L.cone);
                    float MoR = orientationMeasure(R.cone);

                    float cost;
                    if (m_cfg.heuristic == Settings::Heuristic::SAH) {
                        cost = L.N * ML + R.N * MR;
                    } else {
                        cost = Kr * (L.E * ML * MoL + R.E * MR * MoR) / (parentMA * parentMO);
                    }
                    if (cost < best.cost) {
                        best.cost = cost;
                        best.axis = axis;
                        best.pos = mn + (span * (float)s / (float)B);
                    }
                }
            }

            if (best.axis < 0 || !std::isfinite(best.cost))
                return false;

            auto midIter = std::partition(it.begin() + b0, it.begin() + e0,
                                          [&](const TItem& t) { return (&t.c.x)[best.axis] < best.pos; });

            uint32_t mid = static_cast<uint32_t>(midIter - (it.begin() + b0)) + b0;
            if (mid == b0 || mid == e0) {
                mid = (b0 + e0) / 2;
                std::nth_element(
                    it.begin() + b0, it.begin() + mid, it.begin() + e0,
                    [&](const TItem& A, const TItem& B) { return (&A.c.x)[best.axis] < (&B.c.x)[best.axis]; });
            }

            axisOut = best.axis;
            splitPosOut = best.pos;
            midOut = mid;
            return true;
        };

        int ax;
        float pos;
        uint32_t mid;
        bool ok = findBinarySplit(begin, end, ax, pos, mid);
        if (!ok) {
            int widest = 0;
            XMFLOAT3 e = aabbExtent(parent.a);
            if (e.y > e.x && e.y >= e.z)
                widest = 1;
            else if (e.z > e.x && e.z >= e.y)
                widest = 2;
            mid = (begin + end) / 2;
            std::nth_element(it.begin() + begin, it.begin() + mid, it.begin() + end,
                             [&](const TItem& A, const TItem& B) { return (&A.c.x)[widest] < (&B.c.x)[widest]; });
        }

        struct Range {
            uint32_t b, e;
        };
        Range buckets[4];
        uint32_t bucketCount = 0;

        auto pushOrSplitOnce = [&](uint32_t b, uint32_t e) {
            if (e <= b)
                return;
            const uint32_t c = e - b;
            if (c == 1) {
                buckets[bucketCount++] = {b, e};
                return;
            }
            int ax2;
            float pos2;
            uint32_t mid2;
            if (findBinarySplit(b, e, ax2, pos2, mid2) && mid2 > b && mid2 < e) {
                buckets[bucketCount++] = {b, mid2};
                buckets[bucketCount++] = {mid2, e};
            } else {
                buckets[bucketCount++] = {b, e};
            }
        };

        pushOrSplitOnce(begin, mid);
        pushOrSplitOnce(mid, end);

        nodeAt(nodeIdx).firstChild = static_cast<uint32_t>(m_tlas.size());
        nodeAt(nodeIdx).childCount = bucketCount;
        for (uint32_t i = 0; i < bucketCount; i++)
            m_tlas.push_back({});

        for (uint32_t c = 0; c < bucketCount; c++) {
            const LightTreeTrail childTrail = AppendLightTreeTrail(bitTrail, c, depth);
            const uint32_t desired = nodeAt(nodeIdx).firstChild + c;
            buildTLASRecursive_SAOH(it, buckets[c].b, buckets[c].e, childTrail, depth + 1u, desired);
        }

        return nodeIdx;
    }

    template <typename T>
    ComPtr<ID3D12Resource> uploadVector(ID3D12Device* device, ID3D12GraphicsCommandList* cmd, const std::vector<T>& v) {
        if (v.empty())
            return {};
        const UINT64 bytes = static_cast<UINT64>(v.size() * sizeof(T));
#if LT_ENABLE_TIMING
        const auto start = std::chrono::high_resolution_clock::now();
#endif
        ComPtr<ID3D12Resource> upload;
        CD3DX12_HEAP_PROPERTIES hpU(D3D12_HEAP_TYPE_UPLOAD);
        const auto desc = CD3DX12_RESOURCE_DESC::Buffer(bytes);
        if (FAILED(device->CreateCommittedResource(&hpU, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ,
                                                   nullptr, IID_PPV_ARGS(&upload))))
            throw std::runtime_error("Light-tree upload allocation failed");
        void* mapped = nullptr;
        const CD3DX12_RANGE readRange(0, 0);
        if (FAILED(upload->Map(0, &readRange, &mapped)))
            throw std::runtime_error("Light-tree upload mapping failed");
        memcpy(mapped, v.data(), bytes);
        upload->Unmap(0, nullptr);

        ComPtr<ID3D12Resource> gpu;
        CD3DX12_HEAP_PROPERTIES hpD(D3D12_HEAP_TYPE_DEFAULT);
        if (FAILED(device->CreateCommittedResource(&hpD, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST,
                                                   nullptr, IID_PPV_ARGS(&gpu))))
            throw std::runtime_error("Light-tree GPU allocation failed");
        cmd->CopyBufferRegion(gpu.Get(), 0, upload.Get(), 0, bytes);
        auto br = CD3DX12_RESOURCE_BARRIER::Transition(gpu.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                                       D3D12_RESOURCE_STATE_GENERIC_READ);
        cmd->ResourceBarrier(1, &br);
        m_gpu.staging.push_back(upload); // Retained until the upload fence completes.
#if LT_ENABLE_TIMING
        const auto elapsed =
            std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count();
        LT_LOG(L"uploadVector: " << (bytes / 1024.0) << L" KiB in " << elapsed << L" ms");
#endif
        return gpu;
    }

    uint32_t totalBLASNodeCount() const {
        uint32_t n = 0;
        for (auto& b : m_blas)
            n += static_cast<uint32_t>(b.nodes.size());
        return n;
    }
    uint32_t totalLeafIndexCount() const {
        uint32_t n = 0;
        for (auto& b : m_blas)
            n += static_cast<uint32_t>(b.leafTriList.size());
        return n;
    }

    static LightBLASNodeGpu toGpu(const BLASNode& n) {
        LightBLASNodeGpu g{};
        g.bmin = n.aabb.mn;
        g.bmax = n.aabb.mx;
        g.power = n.power;
        g.axis = n.cone.axis;
        g.cosTheta_o = std::cos(clampf(n.cone.theta_o, 0.f, LT_PI));
        g.sinTheta_o = std::sqrt((std::fmax)(0.f, 1.f - g.cosTheta_o * g.cosTheta_o));

        g.firstChild = n.firstChild;
        g.childCount = n.childCount;

        g.triFirst = n.triFirst;
        g.triCount = n.triCount;

        return g;
    }
};
} // namespace lt
