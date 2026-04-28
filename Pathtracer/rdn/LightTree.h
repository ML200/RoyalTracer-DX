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
#include <iostream>
#include <chrono>
#include <cmath>

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
  #define LT_LOG(expr)  do { std::wcout << L"[LightTree] "      << expr << std::endl; } while(0)
  #define LT_WARN(expr) do { std::wcout << L"[LightTree][WARN] " << expr << std::endl; } while(0)
#else
  #define LT_LOG(expr)  do {} while(0)
  #define LT_WARN(expr) do {} while(0)
#endif

#define LT_CONCAT_INNER(a,b) a##b
#define LT_CONCAT(a,b) LT_CONCAT_INNER(a,b)

namespace lt { namespace detail {
#if LT_ENABLE_TIMING
struct ScopedTimer {
    using clock = std::chrono::high_resolution_clock;
    const wchar_t* label; clock::time_point t0; explicit ScopedTimer(const wchar_t* l) : label(l), t0(clock::now()) {}
    ~ScopedTimer(){ const double ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
        std::wcout << L"[LightTree][time] " << label << L": " << ms << L" ms" << std::endl; }
};
#endif
}}

#if LT_ENABLE_TIMING
  #define LT_TIME_SCOPE(LABEL_WIDE) ::lt::detail::ScopedTimer LT_CONCAT(_lt_scope_timer_, __LINE__)(LABEL_WIDE)
#else
  #define LT_TIME_SCOPE(LABEL_WIDE) do {} while(0)
#endif

struct LightTriangle {
    XMFLOAT3 x; float cdf;
    XMFLOAT3 y; UINT instanceID;
    XMFLOAT3 z; float weight;
    XMFLOAT3 emission; UINT triCount; float totalWeight; XMFLOAT3 pad0;
};

struct InstanceXformCPU {
    XMFLOAT4X4 objectToWorld;
};

namespace lt
{
#pragma pack(push, 1)
    struct LightTLASNodeGpu {
        XMFLOAT3 bmin; float power;
        XMFLOAT3 bmax; float cosTheta_o;
        XMFLOAT3 axis; float cosTheta_e;

        uint32_t firstChild;
        uint32_t childCount;
        uint32_t blasIndex;

        uint32_t primCount;
        float    sumPower;
        float    sumPowerSq;

        uint32_t itemFirst;
        uint32_t itemCount;
    };

    struct LightBLASNodeGpu {
        XMFLOAT3 bmin; float power;
        XMFLOAT3 bmax; float cosTheta_o;
        XMFLOAT3 axis; float cosTheta_e;

        uint32_t firstChild;
        uint32_t childCount;
        uint32_t triFirst;
        uint32_t triCount;

        uint32_t primCount;    uint32_t _pad0;
        float    sumPower;     float    sumPowerSq;
    };

    struct BlasRangeGpu {
        uint32_t nodeOffset;
        uint32_t nodeCount;
        uint32_t triIndexOffset;
        uint32_t triIndexCount;
        XMFLOAT4X4 worldToLocal;
    };
#pragma pack(pop)


static constexpr float LT_PI = 3.14159265358979323846f;
static constexpr float LT_HALF_PI = 1.57079632679489661923f;

struct Aabb { XMFLOAT3 mn, mx; };

static XMFLOAT3 add3(const XMFLOAT3& a, const XMFLOAT3& b){ return {a.x+b.x, a.y+b.y, a.z+b.z}; }
static XMFLOAT3 sub3(const XMFLOAT3& a, const XMFLOAT3& b){ return {a.x-b.x, a.y-b.y, a.z-b.z}; }
static XMFLOAT3 mul3(const XMFLOAT3& a, float s){ return {a.x*s, a.y*s, a.z*s}; }
static float dot3(const XMFLOAT3& a, const XMFLOAT3& b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
static XMFLOAT3 cross3(const XMFLOAT3& a, const XMFLOAT3& b){ return { a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x }; }
static float length3(const XMFLOAT3& a){ return std::sqrt((std::fmax)(0.f, dot3(a,a))); }
static XMFLOAT3 normalize3(const XMFLOAT3& a){ float len = length3(a); if (len < 1e-20f) return {0,0,1}; float inv = 1.0f/len; return {a.x*inv, a.y*inv, a.z*inv}; }
static XMFLOAT3 min3(const XMFLOAT3& a, const XMFLOAT3& b){ return { (std::fmin)(a.x,b.x), (std::fmin)(a.y,b.y), (std::fmin)(a.z,b.z) }; }
static XMFLOAT3 max3(const XMFLOAT3& a, const XMFLOAT3& b){ return { (std::fmax)(a.x,b.x), (std::fmax)(a.y,b.y), (std::fmax)(a.z,b.z) }; }

static Aabb triAabb(const ::LightTriangle& t){ Aabb a; a.mn = min3(t.x, min3(t.y,t.z)); a.mx = max3(t.x, max3(t.y,t.z)); return a; }
static Aabb unionAabb(const Aabb& a, const Aabb& b){ return { min3(a.mn,b.mn), max3(a.mx,b.mx) }; }
static float aabbSurfaceArea(const Aabb& a){ XMFLOAT3 e = sub3(a.mx, a.mn); return 2.0f * (e.x*e.y + e.x*e.z + e.y*e.z); }
static XMFLOAT3 aabbCenter(const Aabb& a){ return mul3(add3(a.mn, a.mx), 0.5f); }
static XMFLOAT3 aabbExtent(const Aabb& a){ return sub3(a.mx, a.mn); }

static float clampf(float x, float lo, float hi){ return x < lo ? lo : (x > hi ? hi : x); }
static float safe_acosf(float x){ return std::acos(clampf(x, -1.f, 1.f)); }
static float safe_asinf(float x){ return std::asin(clampf(x, -1.f, 1.f)); }

// Spherical linear interpolation for unit vectors
static XMFLOAT3 slerpUnit(const XMFLOAT3& a, const XMFLOAT3& b, float t){
    float cosT = clampf(dot3(a,b), -1.f, 1.f);
    float theta = std::acos(cosT);
    if (theta < 1e-6f) return a;
    float s = std::sin(theta);
    float w0 = std::sin((1.f - t)*theta) / s;
    float w1 = std::sin(t*theta) / s;
    return normalize3(add3(mul3(a,w0), mul3(b,w1)));
}

struct Cone {
    XMFLOAT3 axis{0,0,1};
    float theta_o = LT_PI;
    float theta_e = LT_HALF_PI;
};

// Merge two cones per Algorithm 1 in the paper
static Cone coneUnion(const Cone& A, const Cone& B){
    Cone a=A, b=B; if (b.theta_o > a.theta_o) std::swap(a,b);

    const float d = clampf(dot3(a.axis, b.axis), -1.f, 1.f);
    const float theta_d = safe_acosf(d);

    Cone out; out.theta_e = (std::fmax)(a.theta_e, b.theta_e);

    // b fully inside a?
    if ((std::fmin)(theta_d + b.theta_o, LT_PI) <= a.theta_o) {
        out.axis = a.axis; out.theta_o = a.theta_o; return out;
    }

    // target half-angle (Algorithm 1)
    float theta_o = (a.theta_o + theta_d + b.theta_o) * 0.5f;
    theta_o = (std::fmin)(theta_o, LT_PI);

    if (theta_d < 1e-7f) {
        // nearly parallel
        out.axis = a.axis;
    } else if (LT_PI - theta_d < 1e-7f) {
        // nearly anti-parallel: choose any perpendicular axis
        XMFLOAT3 t = (std::fabs(a.axis.x) < 0.9f) ? XMFLOAT3{1,0,0} : XMFLOAT3{0,1,0};
        out.axis = normalize3(cross3(a.axis, t));
    } else {
        // regular case
        float t = clampf((theta_o - a.theta_o) / theta_d, 0.f, 1.f);
        out.axis = slerpUnit(a.axis, b.axis, t);
    }

    out.theta_o = theta_o;
    return out;
}


// Orientation measure M_omega (Eq. 1): scalar from cone (axis unused)
static float orientationMeasure(const Cone& c){
    const float theta_o = clampf(c.theta_o, 0.f, LT_PI);
    const float theta_w = (std::fmin)(theta_o + clampf(c.theta_e, 0.f, LT_PI), LT_PI);
    const double to = theta_o, tw = theta_w;
    const double term0 = 2.0 * LT_PI * (1.0 - std::cos(to));
    const double extra = (LT_HALF_PI) * ( 2.0*tw*std::sin(to)
        - std::cos(to - 2.0*tw) - 2.0*to*std::sin(to) + std::cos(to) );
    return static_cast<float>(term0 + extra);
}

// Identity constants
static const XMFLOAT4X4 LT_IDENTITY_4X4 =
    { 1,0,0,0,
      0,1,0,0,
      0,0,1,0,
      0,0,0,1 };

static const XMFLOAT3X3 LT_IDENTITY_3X3 =
    { 1,0,0,
      0,1,0,
      0,0,1 };

// Transform a point by a 4x4 world matrix
static XMFLOAT3 transformPointW(const XMFLOAT3& p, const XMFLOAT4X4& world)
{
    XMVECTOR vp = XMLoadFloat3(&p);
    XMMATRIX W = XMLoadFloat4x4(&world);
    XMVECTOR o = XMVector3TransformCoord(vp, W);
    XMFLOAT3 out; XMStoreFloat3(&out, o); return out;
}

// Build a normal (3x3) matrix = transpose(inverse(world)), with singular fallback
static void computeNormal33FromWorld(const XMFLOAT4X4& world, XMFLOAT3X3& out33)
{
    XMMATRIX W = XMLoadFloat4x4(&world);
    XMVECTOR det;
    XMMATRIX InvW = XMMatrixInverse(&det, W);
    const float detx = XMVectorGetX(det);
    if (!std::isfinite(detx) || std::fabs(detx) < 1e-30f){
        // Singular: fall back to identity to avoid NaNs
        XMStoreFloat3x3(&out33, XMMatrixIdentity());
        return;
    }
    XMMATRIX N = XMMatrixTranspose(InvW);
    XMStoreFloat3x3(&out33, N);
}

// Transform and normalize a normal by a 3x3 normal matrix
static XMFLOAT3 transformNormalW(const XMFLOAT3& n, const XMFLOAT3X3& N33)
{
    XMMATRIX N = XMLoadFloat3x3(&N33);
    XMVECTOR vn = XMLoadFloat3(&n);
    XMVECTOR wo = XMVector3TransformNormal(vn, N);
    XMFLOAT3 out; XMStoreFloat3(&out, wo);
    return normalize3(out);
}

// AABB of a triangle in world space
static Aabb triAabbWorld(const ::LightTriangle& t, const XMFLOAT4X4& world)
{
    const XMFLOAT3 X = transformPointW(t.x, world);
    const XMFLOAT3 Y = transformPointW(t.y, world);
    const XMFLOAT3 Z = transformPointW(t.z, world);
    Aabb a;
    a.mn = min3(X, min3(Y, Z));
    a.mx = max3(X, max3(Y, Z));
    return a;
}

//====================================
//CPU NODE
//====================================
    struct BLASNode {
    Aabb aabb{}; float power = 0.f;
    Cone cone{};
    uint32_t firstChild = 0xFFFFFFFF; // index of first child (contiguous)
    uint32_t childCount = 0;          // 0 => leaf
    uint32_t primCount = 0;
    float    sumPower = 0.f;
    float    sumPowerSq = 0.f;
    uint32_t triFirst = 0, triCount = 0;
    bool isLeaf() const { return childCount == 0; }
};

struct BLASBuild {
    std::vector<uint32_t> triIndices;   // indices of global emissive tris (for this BLAS)
    std::vector<BLASNode> nodes;        // nodes[0] is root after build
    std::vector<uint32_t> leafTriList;  // concatenation of per-leaf triangle lists
};

//====================================
//LIGHT TREE BUILDER
//====================================
class LightTreeBuilder {
private:
    struct TItem { // TLAS items over BLAS roots
        uint32_t idx; Aabb a; XMFLOAT3 c; float p; Cone cone; uint32_t primCount; float sumP, sumP2;
    };
    std::vector<uint32_t> m_blasToItem;

public:
    struct Settings {
        uint32_t maxLeafTris = 16;       // triangles per BLAS leaf
        bool     useTwoLevel = true;     // group by instanceID into BLASes
        uint32_t buildBins   = 64;       // spatial bin count for SAOH
        enum class Heuristic { SAOH, SAH }; // What heuristic should we use?
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
        ComPtr<ID3D12Resource> TriToLeafOffset;
        ComPtr<ID3D12Resource> BLASToItem;
        std::vector<ComPtr<ID3D12Resource>> staging;
    };

    // CPU-side copy of BLASRanges for dynamic worldToLocal updates
    std::vector<BlasRangeGpu> m_cpuBlasRanges;

    const std::vector<BlasRangeGpu>& GetCpuBLASRanges() const { return m_cpuBlasRanges; }

    //====================================
    //API
    //====================================

    void Build(const std::vector<::LightTriangle>& tris, const Settings& cfg = {}) {
        LT_TIME_SCOPE(L"Build()");
        LT_LOG(L"Build: tris=" << tris.size() << L", maxLeafTris=" << cfg.maxLeafTris
              << L", twoLevel=" << (cfg.useTwoLevel ? L"true" : L"false")
              << L", bins=" << cfg.buildBins);
        if (tris.empty()) LT_WARN(L"No emissive triangles; tree will be empty.");
        m_cfg = cfg; m_tris = &tris; m_xforms = nullptr;
        rebuildXformCaches();
        buildBLASes_SAOH();
        buildTLAS_SAOH();
        LT_LOG(L"Build done: BLAS=" << m_blas.size() << L", TLAS nodes=" << m_tlas.size());
    }
    // Build fully in world space
    void Build(const std::vector<::LightTriangle>& tris,
               const std::vector<InstanceXformCPU>& xforms,
               const Settings& cfg = {})
    {
        LT_TIME_SCOPE(L"Build(xforms)");
        LT_LOG(L"Build(xforms): tris=" << tris.size()
              << L", xforms=" << xforms.size()
              << L", maxLeafTris=" << cfg.maxLeafTris
              << L", twoLevel=" << (cfg.useTwoLevel ? L"true" : L"false")
              << L", bins=" << cfg.buildBins);
        if (tris.empty()) LT_WARN(L"No emissive triangles; tree will be empty.");
        m_cfg = cfg; m_tris = &tris; m_xforms = &xforms;
        rebuildXformCaches();
        buildBLASes_SAOH();
        buildTLAS_SAOH();
        LT_LOG(L"Build done: BLAS=" << m_blas.size() << L", TLAS nodes=" << m_tlas.size());
    }

    // TODO: refit

    void UploadAll(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList){
        LT_TIME_SCOPE(L"UploadAll()");
        std::vector<LightBLASNodeGpu> gpuBlasNodes;    gpuBlasNodes.reserve(totalBLASNodeCount());
        std::vector<uint32_t>         gpuLeafTriIndex; gpuLeafTriIndex.reserve(totalLeafIndexCount());
        std::vector<BlasRangeGpu>     gpuRanges;       gpuRanges.reserve(m_blas.size());

        // Gather BLAS data (nodes, leaf tri indices, per-BLAS ranges)
        for (const auto& b : m_blas){
            BlasRangeGpu r{};
            r.nodeOffset    = static_cast<uint32_t>(gpuBlasNodes.size());
            r.nodeCount     = static_cast<uint32_t>(b.nodes.size());
            r.triIndexOffset= static_cast<uint32_t>(gpuLeafTriIndex.size());
            r.triIndexCount = static_cast<uint32_t>(b.leafTriList.size());

            // Compute worldToLocal from the instance that owns this BLAS
            XMStoreFloat4x4(&r.worldToLocal, XMMatrixIdentity());
            if (!b.triIndices.empty() && m_tris) {
                UINT instID = (*m_tris)[b.triIndices[0]].instanceID;
                const XMFLOAT4X4& W = worldXformFor(instID);
                XMVECTOR det;
                XMMATRIX Winv = XMMatrixInverse(&det, XMLoadFloat4x4(&W));
                XMStoreFloat4x4(&r.worldToLocal, Winv);
            }

            for (const auto& n : b.nodes)
                gpuBlasNodes.push_back(toGpu(n));

            gpuLeafTriIndex.insert(gpuLeafTriIndex.end(), b.leafTriList.begin(), b.leafTriList.end());
            gpuRanges.push_back(r);
        }

        // TLAS
        std::vector<LightTLASNodeGpu> gpuTlasNodes(m_tlas.size());
        for (size_t i = 0; i < m_tlas.size(); ++i)
            gpuTlasNodes[i] = m_tlas[i];

        // Stats (no alias tables anymore)
        LT_LOG(L"UploadAll: BLASNodes=" << gpuBlasNodes.size()
              << L", LeafTriIndex=" << gpuLeafTriIndex.size()
              << L", BLASRanges=" << gpuRanges.size()
              << L", TLASNodes=" << gpuTlasNodes.size());
        const auto KiB = [](uint64_t b){ return b/1024.0; };
        LT_LOG(L"  sizes: TLAS="   << KiB(gpuTlasNodes.size()*sizeof(LightTLASNodeGpu))  << L" KiB"
              << L", BLAS="        << KiB(gpuBlasNodes.size()*sizeof(LightBLASNodeGpu))  << L" KiB"
              << L", Ranges="      << KiB(gpuRanges.size()*sizeof(BlasRangeGpu))         << L" KiB"
              << L", LeafIdx="     << KiB(gpuLeafTriIndex.size()*sizeof(uint32_t))       << L" KiB");

        // Lookup tables
        std::vector<uint32_t> triToBLAS(m_tris ? m_tris->size() : 0, 0xFFFFFFFFu);
        std::vector<uint32_t> triToLeafOff(m_tris ? m_tris->size() : 0, 0);

        for (uint32_t bIdx = 0; bIdx < m_blas.size(); ++bIdx) {
            const auto& b = m_blas[bIdx];
            for (uint32_t j = 0; j < b.leafTriList.size(); ++j) {
                uint32_t tri = b.leafTriList[j];
                triToBLAS[tri]    = bIdx;
                triToLeafOff[tri] = j;
            }
        }

        // Keep CPU copy of BLASRanges for dynamic worldToLocal updates
        m_cpuBlasRanges = gpuRanges;

        // Upload everything (alias buffers removed)
        m_gpu = {};
        m_gpu.BLASNodes       = uploadVector(device, cmdList, gpuBlasNodes);
        m_gpu.LeafTriIndex    = uploadVector(device, cmdList, gpuLeafTriIndex);
        m_gpu.BLASRanges      = uploadVector(device, cmdList, gpuRanges);
        m_gpu.TLASNodes       = uploadVector(device, cmdList, gpuTlasNodes);
        m_gpu.TriToBLAS       = uploadVector(device, cmdList, triToBLAS);
        m_gpu.TriToLeafOffset = uploadVector(device, cmdList, triToLeafOff);
        m_gpu.BLASToItem      = uploadVector(device, cmdList, m_blasToItem);
    }

    void WriteSrvs(ID3D12Device* device, D3D12_CPU_DESCRIPTOR_HANDLE dst) const {
        LT_TIME_SCOPE(L"WriteSrvs()");
        const UINT inc = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

        auto makeBufSrv = [&](ID3D12Resource* res, UINT numElems, UINT stride, DXGI_FORMAT fmt, D3D12_CPU_DESCRIPTOR_HANDLE h){
            D3D12_SHADER_RESOURCE_VIEW_DESC d{}; d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            if (stride==0){
                d.Format = fmt; d.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
                d.Buffer.FirstElement = 0; d.Buffer.NumElements = numElems; d.Buffer.StructureByteStride = 0; d.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
            } else {
                d.Format = DXGI_FORMAT_UNKNOWN; d.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
                d.Buffer.FirstElement = 0; d.Buffer.NumElements = numElems; d.Buffer.StructureByteStride = stride; d.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
            }
            device->CreateShaderResourceView(res, &d, h);
        };

        // Only write descriptors for resources that exist.
        if (m_gpu.TLASNodes) {
            D3D12_RESOURCE_DESC rd = m_gpu.TLASNodes->GetDesc();
            const UINT n = static_cast<UINT>(rd.Width / sizeof(LightTLASNodeGpu));
            LT_LOG(L"WriteSrvs: TLASNodes count=" << n);
            makeBufSrv(m_gpu.TLASNodes.Get(), n, sizeof(LightTLASNodeGpu), DXGI_FORMAT_UNKNOWN, dst);
            dst.ptr += inc;
        } else {
            LT_WARN(L"WriteSrvs: TLASNodes is null, skipping SRV.");
        }

        if (m_gpu.BLASNodes) {
            D3D12_RESOURCE_DESC rd = m_gpu.BLASNodes->GetDesc();
            const UINT n = static_cast<UINT>(rd.Width / sizeof(LightBLASNodeGpu));
            LT_LOG(L"WriteSrvs: BLASNodes count=" << n);
            makeBufSrv(m_gpu.BLASNodes.Get(), n, sizeof(LightBLASNodeGpu), DXGI_FORMAT_UNKNOWN, dst);
            dst.ptr += inc;
        } else {
            LT_WARN(L"WriteSrvs: BLASNodes is null, skipping SRV.");
        }

        if (m_gpu.BLASRanges) {
            D3D12_RESOURCE_DESC rd = m_gpu.BLASRanges->GetDesc();
            const UINT n = static_cast<UINT>(rd.Width / sizeof(BlasRangeGpu));
            LT_LOG(L"WriteSrvs: BLASRanges count=" << n);
            makeBufSrv(m_gpu.BLASRanges.Get(), n, sizeof(BlasRangeGpu), DXGI_FORMAT_UNKNOWN, dst);
            dst.ptr += inc;
        } else {
            LT_WARN(L"WriteSrvs: BLASRanges is null, skipping SRV.");
        }

        if (m_gpu.LeafTriIndex) {
            D3D12_RESOURCE_DESC rd = m_gpu.LeafTriIndex->GetDesc();
            const UINT n = static_cast<UINT>(rd.Width / 4);
            LT_LOG(L"WriteSrvs: LeafTriIndex count=" << n);
            makeBufSrv(m_gpu.LeafTriIndex.Get(), n, 0, DXGI_FORMAT_R32_UINT, dst);
            dst.ptr += inc;
        } else {
            LT_WARN(L"WriteSrvs: LeafTriIndex is null, skipping SRV.");
        }
    }



    void WriteAliasSrvs(ID3D12Device* device, D3D12_CPU_DESCRIPTOR_HANDLE dst) const {
        LT_TIME_SCOPE(L"WriteAliasSrvs()");
        const UINT inc = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        auto makeTyped = [&](ID3D12Resource* res, DXGI_FORMAT fmt, UINT n, D3D12_CPU_DESCRIPTOR_HANDLE h){
            D3D12_SHADER_RESOURCE_VIEW_DESC d{}; d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            d.Format = fmt; d.ViewDimension = D3D12_SRV_DIMENSION_BUFFER; d.Buffer.FirstElement = 0; d.Buffer.NumElements = n; d.Buffer.StructureByteStride = 0; d.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
            device->CreateShaderResourceView(res, &d, h);
        };
        if (m_gpu.LeafAliasProb){ auto rd = m_gpu.LeafAliasProb->GetDesc(); const UINT n = static_cast<UINT>(rd.Width / 4);
            LT_LOG(L"WriteAliasSrvs: LeafAliasProb count=" << n); makeTyped(m_gpu.LeafAliasProb.Get(), DXGI_FORMAT_R32_FLOAT, n, dst);
        } else LT_WARN(L"WriteAliasSrvs: LeafAliasProb is null (no leaves?)");
        dst.ptr += inc;
        if (m_gpu.LeafAliasIdx){ auto rd = m_gpu.LeafAliasIdx->GetDesc(); const UINT n = static_cast<UINT>(rd.Width / 4);
            LT_LOG(L"WriteAliasSrvs: LeafAliasIdx count=" << n); makeTyped(m_gpu.LeafAliasIdx.Get(), DXGI_FORMAT_R32_UINT, n, dst);
        } else LT_WARN(L"WriteAliasSrvs: LeafAliasIdx is null (no leaves?)");
    }

    void WriteLookupSrvs(ID3D12Device* device, D3D12_CPU_DESCRIPTOR_HANDLE dst) const {
        const UINT inc = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

        auto makeTyped = [&](ID3D12Resource* res, DXGI_FORMAT fmt, UINT count, D3D12_CPU_DESCRIPTOR_HANDLE h){
            if (!res) { LT_WARN(L"WriteLookupSrvs: null resource"); return; }
            D3D12_SHADER_RESOURCE_VIEW_DESC d{};
            d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            d.Format = fmt;
            d.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            d.Buffer.FirstElement = 0;
            d.Buffer.NumElements = count;
            d.Buffer.StructureByteStride = 0;
            d.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
            device->CreateShaderResourceView(res, &d, h);
        };

        // TriToBLAS (R32_UINT), count = #tris
        if (m_gpu.TriToBLAS) {
            UINT n = static_cast<UINT>(m_gpu.TriToBLAS->GetDesc().Width / 4);
            LT_LOG(L"WriteLookupSrvs: TriToBLAS count=" << n);
            makeTyped(m_gpu.TriToBLAS.Get(), DXGI_FORMAT_R32_UINT, n, dst);
        }
        dst.ptr += inc;

        // TriToLeafOffset (R32_UINT)
        if (m_gpu.TriToLeafOffset) {
            UINT n = static_cast<UINT>(m_gpu.TriToLeafOffset->GetDesc().Width / 4);
            LT_LOG(L"WriteLookupSrvs: TriToLeafOffset count=" << n);
            makeTyped(m_gpu.TriToLeafOffset.Get(), DXGI_FORMAT_R32_UINT, n, dst);
        }
        dst.ptr += inc;

        // BLASToItem (R32_UINT), count = #BLASes
        if (m_gpu.BLASToItem) {
            UINT n = static_cast<UINT>(m_gpu.BLASToItem->GetDesc().Width / 4);
            LT_LOG(L"WriteLookupSrvs: BLASToItem count=" << n);
            makeTyped(m_gpu.BLASToItem.Get(), DXGI_FORMAT_R32_UINT, n, dst);
        }
    }


    void ReleaseStaging(){ LT_TIME_SCOPE(L"ReleaseStaging()"); LT_LOG(L"ReleaseStaging: " << m_gpu.staging.size() << L" upload buffers freed"); m_gpu.staging.clear(); }

    ID3D12Resource* GetTLASGpuBuffer() const { return m_gpu.TLASNodes.Get(); }
    ID3D12Resource* GetBLASToItemGpuBuffer() const { return m_gpu.BLASToItem.Get(); }
    uint32_t GetTLASBufferSize() const { return static_cast<uint32_t>(m_tlas.size() * sizeof(LightTLASNodeGpu)); }

    const GpuBuffers& GetGpu() const { return m_gpu; }

    uint32_t TLASNodeCount() const { return static_cast<uint32_t>(m_tlas.size()); }
    uint32_t BLASCount()     const { return static_cast<uint32_t>(m_blas.size()); }

    // metrics
    void PrintMetrics() const
    {
        LT_TIME_SCOPE(L"PrintMetrics()");
        struct ND { uint32_t i; uint32_t d; };

        //TLAS
        if (m_tlas.empty()){
            LT_WARN(L"TLAS is empty.");
        } else {
            uint64_t nodeCount=0, inner=0, leaf=0, leafDepthSum=0, childrenSum=0;
            uint32_t maxDepth=0, maxChildren=0;

            std::vector<ND> st; st.reserve(m_tlas.size());
            st.push_back({0,0});
            while (!st.empty()){
                ND cur = st.back(); st.pop_back();
                const auto& n = m_tlas[cur.i];
                nodeCount++; maxDepth = (std::max)(maxDepth, cur.d);
                if (n.childCount == 0){
                    leaf++; leafDepthSum += cur.d;
                } else {
                    inner++; childrenSum += n.childCount; maxChildren = (std::max)(maxChildren, n.childCount);
                    for (uint32_t c=0; c<n.childCount; ++c) st.push_back({ n.firstChild + c, cur.d + 1 });
                }
            }

            const double avgLeafDepth  = leaf  ? double(leafDepthSum) / double(leaf)  : 0.0;
            const double avgChildren   = inner ? double(childrenSum) / double(inner) : 0.0;

            LT_LOG(L"TLAS: nodes=" << nodeCount
                  << L" (inner=" << inner << L", leaf=" << leaf << L")"
                  << L", maxDepth=" << maxDepth
                  << L", avgLeafDepth=" << avgLeafDepth
                  << L", avgChildren=" << avgChildren
                  << L", maxChildren=" << maxChildren);
        }

        //BLAS per instance + aggregate
        uint64_t allNodes=0, allInner=0, allLeaf=0, allLeafDepthSum=0, allLeafTriSum=0;
        uint32_t allMaxDepth=0, globalMinLeafTri=UINT32_MAX, globalMaxLeafTri=0;

        for (uint32_t b=0; b < m_blas.size(); ++b){
            const auto& B = m_blas[b];
            if (B.nodes.empty()){
                LT_WARN(L"BLAS[" << b << L"] is empty.");
                continue;
            }

            uint64_t nodes=0, inner=0, leaf=0, leafDepthSum=0, leafTriSum=0;
            uint32_t maxDepth=0, minLeafTri=UINT32_MAX, maxLeafTri=0;

            std::vector<ND> st; st.reserve(B.nodes.size());
            st.push_back({0,0});
            while (!st.empty()){
                ND cur = st.back(); st.pop_back();
                const auto& n = B.nodes[cur.i];
                nodes++; maxDepth = (std::max)(maxDepth, cur.d);
                if (n.childCount == 0){
                    leaf++; leafDepthSum += cur.d;
                    minLeafTri = (std::min)(minLeafTri, n.triCount);
                    maxLeafTri = (std::max)(maxLeafTri, n.triCount);
                    leafTriSum += n.triCount;
                } else {
                    inner++;
                    for (uint32_t c=0; c<n.childCount; ++c) st.push_back({ n.firstChild + c, cur.d + 1 });
                }
            }

            const double avgLeafDepth = leaf ? double(leafDepthSum) / double(leaf) : 0.0;
            const double avgLeafTris  = leaf ? double(leafTriSum)  / double(leaf) : 0.0;

            LT_LOG(L"BLAS[" << b << L"]: nodes=" << nodes
                  << L" (inner=" << inner << L", leaf=" << leaf << L")"
                  << L", maxDepth=" << maxDepth
                  << L", avgLeafDepth=" << avgLeafDepth
                  << L", leafTris(min/avg/max)="
                  << (minLeafTri==UINT32_MAX?0:minLeafTri) << L"/" << avgLeafTris << L"/" << maxLeafTri);

            // accumulate globals
            allNodes += nodes; allInner += inner; allLeaf += leaf;
            allLeafDepthSum += leafDepthSum; allLeafTriSum += leafTriSum;
            allMaxDepth = (std::max)(allMaxDepth, maxDepth);
            globalMinLeafTri = (std::min)(globalMinLeafTri, minLeafTri);
            globalMaxLeafTri = (std::max)(globalMaxLeafTri, maxLeafTri);
        }

        if (!m_blas.empty()){
            const double avgLeafDepthAll = allLeaf ? double(allLeafDepthSum) / double(allLeaf) : 0.0;
            const double avgLeafTrisAll  = allLeaf ? double(allLeafTriSum)  / double(allLeaf) : 0.0;

            LT_LOG(L"BLAS (all): nodes=" << allNodes
                  << L" (inner=" << allInner << L", leaf=" << allLeaf << L")"
                  << L", maxDepth=" << allMaxDepth
                  << L", avgLeafDepth=" << avgLeafDepthAll
                  << L", leafTris(min/avg/max)="
                  << (globalMinLeafTri==UINT32_MAX?0:globalMinLeafTri)
                  << L"/" << avgLeafTrisAll << L"/" << globalMaxLeafTri);
        }
    }

private:
    const std::vector<::LightTriangle>* m_tris = nullptr; Settings m_cfg{};
    const std::vector<InstanceXformCPU>* m_xforms = nullptr;
    std::vector<XMFLOAT4X4> m_worldByInstance;
    std::vector<XMFLOAT3X3> m_normalByInstance;

    std::vector<BLASBuild> m_blas;               // per-instance BLASes
    std::vector<LightTLASNodeGpu> m_tlas;        // TLAS nodes (root at 0)
    GpuBuffers m_gpu{};

    struct TmpTri { uint32_t triIndex; XMFLOAT3 centroid; Aabb aabb; float power; UINT instanceID; Cone cone; };

    // Build/refresh cached matrices from m_xforms
    void rebuildXformCaches(){
        m_worldByInstance.clear();
        m_normalByInstance.clear();
        if (!m_xforms || m_xforms->empty()) return;
        const size_t n = m_xforms->size();
        m_worldByInstance.resize(n);
        m_normalByInstance.resize(n);
        for (size_t i=0; i<n; ++i){
            m_worldByInstance[i] = (*m_xforms)[i].objectToWorld;
            computeNormal33FromWorld(m_worldByInstance[i], m_normalByInstance[i]);
        }
    }
    // Safe accessors with identity fallback
    const XMFLOAT4X4& worldXformFor(UINT id) const {
        if (id < m_worldByInstance.size()) return m_worldByInstance[id];
        if (m_xforms) LT_WARN(L"InstanceID " << id << L" out of range for world matrix; using identity.");
        return LT_IDENTITY_4X4;
    }
    const XMFLOAT3X3& normalXformFor(UINT id) const {
        if (id < m_normalByInstance.size()) return m_normalByInstance[id];
        return LT_IDENTITY_3X3;
    }

    //====================================
    //BUILD BLAS SAOH
    //====================================
    void buildBLASes_SAOH(){
        LT_TIME_SCOPE(L"buildBLASes_SAOH()");
        LT_LOG(L"Grouping " << (m_tris ? m_tris->size() : 0) << L" emissive triangles by instanceID...");
        if (!m_tris){ LT_WARN(L"m_tris == nullptr"); return; }
        std::unordered_map<UINT, std::vector<uint32_t>> groups; groups.reserve(m_tris->size());
        for (uint32_t i=0;i<m_tris->size();++i) groups[(*m_tris)[i].instanceID].push_back(i);
        LT_LOG(L"buildBLASes: groups=" << groups.size());
        m_blas.clear(); m_blas.reserve(groups.size());
        for (auto& kv : groups){ const auto& idxs = kv.second; LT_LOG(L"  BLAS[" << kv.first << L"] tris=" << idxs.size());
            BLASBuild b;
            b.triIndices = idxs;
            // Slightly generous reserve to reduce chance of reallocation during recursion.
            b.nodes.reserve(static_cast<size_t>(idxs.size()) * 2u + 32u);
            std::vector<TmpTri> tmp; tmp.reserve(idxs.size());

            for (uint32_t j=0;j<idxs.size();++j){
                const auto& t = (*m_tris)[idxs[j]];
                const UINT inst = t.instanceID;
                const XMFLOAT4X4& W  = worldXformFor(inst);
                const XMFLOAT3X3& N3 = normalXformFor(inst);

                // World-space vertices
                const XMFLOAT3 Xw = transformPointW(t.x, W);
                const XMFLOAT3 Yw = transformPointW(t.y, W);
                const XMFLOAT3 Zw = transformPointW(t.z, W);

                // Centroid & AABB in world space
                const XMFLOAT3 c{ (Xw.x+Yw.x+Zw.x)/3.f, (Xw.y+Yw.y+Zw.y)/3.f, (Xw.z+Yw.z+Zw.z)/3.f };
                const Aabb a = { min3(Xw, min3(Yw, Zw)), max3(Xw, max3(Yw, Zw)) };

                // Normal cone axis from world-space normal (inverse-transpose)
                const XMFLOAT3 e1L = sub3(t.y, t.x);
                const XMFLOAT3 e2L = sub3(t.z, t.x);
                const XMFLOAT3 nL  = cross3(e1L, e2L);
                XMFLOAT3 nW = transformNormalW(nL, N3);
                float nlen = length3(nW);

                Cone lc;
                if (nlen < 1e-12f){ // degenerate -> isotropic emitter
                    lc.axis = {0,0,1}; lc.theta_o = LT_PI; lc.theta_e = LT_HALF_PI;
                } else {
                    lc.axis = normalize3(nW); lc.theta_o = 0.f; lc.theta_e = LT_HALF_PI;
                }

                tmp.push_back({ idxs[j], c, a, t.weight, inst, lc });
            }

            // SAOH build
            buildBLASRecursive_SAOH(tmp, b, 0, static_cast<uint32_t>(tmp.size()));
            // stats
            uint32_t leafs = 0, inner = 0;
            for (const auto& n : b.nodes) (n.isLeaf() ? leafs : inner)++;
            LT_LOG(L"    nodes=" << b.nodes.size() << L" (inner=" << inner << L", leaf=" << leafs << L")"
                << L", leafTriIndexCount=" << b.leafTriList.size());
            m_blas.push_back(std::move(b));
        }
    }


    struct Agg { bool valid=false; Aabb a; float E=0; Cone cone{}; uint32_t N=0; float sumP=0.f; float sumP2=0.f; };

    static void aggAdd(Agg& A, const TmpTri& t){ if (!A.valid){ A.valid=true; A.a=t.aabb; A.E=t.power; A.cone=t.cone; A.N=1; A.sumP=t.power; A.sumP2=t.power*t.power; return; }
        A.a = unionAabb(A.a, t.aabb); A.E += t.power; A.cone = coneUnion(A.cone, t.cone); A.N++; A.sumP+=t.power; A.sumP2+=t.power*t.power; }

    static void aggMerge(Agg& A, const Agg& B){ if (!B.valid) return; if (!A.valid){ A=B; return; } A.a = unionAabb(A.a, B.a); A.E+=B.E; A.cone=coneUnion(A.cone,B.cone); A.N+=B.N; A.sumP+=B.sumP; A.sumP2+=B.sumP2; }

    uint32_t buildBLASRecursive_SAOH(std::vector<TmpTri>& tmp, BLASBuild& out,
                                 uint32_t begin, uint32_t end)
    {
        const uint32_t nodeIdx = static_cast<uint32_t>(out.nodes.size());
        out.nodes.emplace_back();

        auto nodeAt = [&](uint32_t i) -> BLASNode& { return out.nodes[i]; };
        BLASNode& N0 = nodeAt(nodeIdx);

        // Aggregate parent
        Agg parent{};
        for (uint32_t i = begin; i < end; ++i) aggAdd(parent, tmp[i]);

        // Write aggregate to the freshly created node
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
            for (uint32_t i = begin; i < end; ++i) out.leafTriList.push_back(tmp[i].triIndex);
            N0.firstChild = 0xFFFFFFFF;
            N0.childCount = 0; // leaf
            return nodeIdx;
        }

        // helper: find a binary SAOH split on [b0,e0)
        auto findBinarySplit = [&](uint32_t b0, uint32_t e0,
                                   int& axisOut, float& splitPosOut, uint32_t& midOut)->bool
        {
            Agg parentL{}; for (uint32_t i=b0;i<e0;++i) aggAdd(parentL, tmp[i]);
            const Aabb   aabb = parentL.a;
            const XMFLOAT3 ext = aabbExtent(aabb);
            const float lenX = ext.x, lenY = ext.y, lenZ = ext.z, lenMax = (std::fmax)(lenX,(std::fmax)(lenY,lenZ));
            const float parentMA = (std::fmax)(1e-12f, aabbSurfaceArea(aabb));
            const float parentMO = (std::fmax)(1e-12f, orientationMeasure(parentL.cone));

            struct Best { float cost = std::numeric_limits<float>::infinity(); int axis = -1; float splitPos = 0; } best;
            const uint32_t B = (std::fmax)(4u, (std::fmin)(64u, m_cfg.buildBins));

            for (int axis = 0; axis < 3; ++axis)
            {
                float mn = (&tmp[b0].centroid.x)[axis], mx = mn;
                for (uint32_t i = b0; i < e0; ++i) {
                    float v = (&tmp[i].centroid.x)[axis];
                    mn = (std::fmin)(mn, v); mx = (std::fmax)(mx, v);
                }
                float span = mx - mn;
                if (span <= 1e-20f) continue;

                std::vector<Agg> bins(B);
                float invSpan = 1.0f / span;
                for (uint32_t i = b0; i < e0; ++i) {
                    float v = (&tmp[i].centroid.x)[axis];
                    uint32_t bi = (std::fmin)(B - 1u, (uint32_t)std::floor((v - mn) * invSpan * B));
                    aggAdd(bins[bi], tmp[i]);
                }

                std::vector<Agg> pref(B), suff(B);
                for (uint32_t i = 0; i < B; ++i) {
                    if (i == 0) pref[i] = bins[i]; else { pref[i] = pref[i - 1]; aggMerge(pref[i], bins[i]); }
                }
                for (int i = (int)B - 1; i >= 0; --i) {
                    if ((uint32_t)i == B - 1) suff[i] = bins[i]; else { suff[i] = suff[i + 1]; aggMerge(suff[i], bins[i]); }
                }

                float length_i = (axis == 0 ? lenX : (axis == 1 ? lenY : lenZ));
                float Kr = (length_i > 1e-20f) ? (lenMax / length_i) : 1e6f;

                for (uint32_t s = 1; s < B; ++s) {
                    const Agg& L = pref[s - 1];
                    const Agg& R = suff[s];
                    if (!L.valid || !R.valid) continue;

                    float ML  = aabbSurfaceArea(L.a),  MR  = aabbSurfaceArea(R.a);
                    float MoL = orientationMeasure(L.cone);
                    float MoR = orientationMeasure(R.cone);

                    float cost;
                    if (m_cfg.heuristic == Settings::Heuristic::SAH) {
                        cost = L.N * ML + R.N * MR;
                    } else {
                        cost = Kr * (L.E * ML * MoL + R.E * MR * MoR) / (parentMA * parentMO);
                    }
                    if (cost < best.cost) {
                        best.cost = cost; best.axis = axis;
                        best.splitPos = mn + (span * (float)s / (float)B);
                    }
                }
            }

            if (best.axis < 0 || !std::isfinite(best.cost)) return false;

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

            axisOut = best.axis; splitPosOut = best.splitPos; midOut = mid; return true;
        };

        // split entire set into two buckets
        int ax; float pos; uint32_t mid;
        bool ok = findBinarySplit(begin, end, ax, pos, mid);
        if (!ok) {
            // Fallback to leaf if something degenerate happens
            BLASNode& N = nodeAt(nodeIdx);
            N.triFirst = static_cast<uint32_t>(out.leafTriList.size());
            N.triCount = count;
            for (uint32_t i = begin; i < end; ++i) out.leafTriList.push_back(tmp[i].triIndex);
            N.firstChild = 0xFFFFFFFF; N.childCount = 0;
            return nodeIdx;
        }

        struct Range { uint32_t b, e; };
        Range buckets[4]; uint32_t bucketCount = 0;

        auto pushOrSplitOnce = [&](uint32_t b, uint32_t e){
            if (e <= b) return;
            const uint32_t c = e - b;
            if (c <= m_cfg.maxLeafTris) { buckets[bucketCount++] = {b,e}; return; }
            int ax2; float pos2; uint32_t mid2;
            if (findBinarySplit(b, e, ax2, pos2, mid2) && mid2>b && mid2<e) {
                buckets[bucketCount++] = {b, mid2};
                buckets[bucketCount++] = {mid2, e};
            } else {
                buckets[bucketCount++] = {b, e};
            }
        };

        pushOrSplitOnce(begin, mid);
        pushOrSplitOnce(mid,   end);

        // If we somehow ended up with >4 groups, merge smallest pairs until 4.
        while (bucketCount > 4) {
            uint32_t iMin=0, jMin=1; uint32_t best = UINT32_MAX;
            for (uint32_t i=0;i<bucketCount;i++)
                for (uint32_t j=i+1;j<bucketCount;j++){
                    uint32_t s = (buckets[i].e - buckets[i].b) + (buckets[j].e - buckets[j].b);
                    if (s < best){ best = s; iMin = i; jMin = j; }
                }
            buckets[iMin].b = (std::fmin)(buckets[iMin].b, buckets[jMin].b);
            buckets[iMin].e = (std::fmax)(buckets[iMin].e, buckets[jMin].e);
            for (uint32_t k=jMin+1;k<bucketCount;k++) buckets[k-1] = buckets[k];
            --bucketCount;
        }

        // Allocate children as a contiguous block
        nodeAt(nodeIdx).firstChild = static_cast<uint32_t>(out.nodes.size());
        nodeAt(nodeIdx).childCount = bucketCount;
        for (uint32_t i=0;i<bucketCount;i++) out.nodes.emplace_back();

        // Build each child into its slot
        for (uint32_t c = 0; c < bucketCount; ++c) {
            uint32_t built = buildBLASRecursive_SAOH(tmp, out, buckets[c].b, buckets[c].e);
            uint32_t desired = nodeAt(nodeIdx).firstChild + c;
            if (built != desired) std::swap(out.nodes[built], out.nodes[desired]);
        }

        // Re-fetch parent after recursion and fill the triangle data
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



    //====================================
    //BUILD TLAS SAOH
    //====================================
    void buildTLAS_SAOH(){
        LT_TIME_SCOPE(L"buildTLAS_SAOH()");
        LT_LOG(L"buildTLAS: BLASes=" << m_blas.size());
        std::vector<TItem> items; items.reserve(m_blas.size());
        for (uint32_t i=0;i<m_blas.size();++i){ const auto& b = m_blas[i]; const auto& r = b.nodes[0];
            TItem it; it.idx=i; it.a=r.aabb; it.c=aabbCenter(r.aabb); it.p=r.power; it.cone=r.cone; it.primCount=r.primCount; it.sumP=r.sumPower; it.sumP2=r.sumPowerSq; items.push_back(it);
        }
        m_tlas.clear();
        if (items.empty()){ LT_WARN(L"buildTLAS: no items"); return; }
        // Reserve to minimize reallocations during recursive build
        m_tlas.reserve(items.size() * 2u + 32u);

        buildTLASRecursive_SAOH(items, 0, static_cast<uint32_t>(items.size()));
        m_blasToItem.clear();
        m_blasToItem.resize(m_blas.size(), 0);
        for (uint32_t i = 0; i < items.size(); ++i) {
            m_blasToItem[ items[i].idx ] = i;
        }
        LT_LOG(L"buildTLAS done: TLAS nodes=" << m_tlas.size());
    }

    struct AggT { bool valid=false; Aabb a; float E=0; Cone cone{}; uint32_t N=0; float sumP=0, sumP2=0; };
    static void aggTAdd(AggT& A, const TItem& t){ if (!A.valid){ A.valid=true; A.a=t.a; A.E=t.p; A.cone=t.cone; A.N=t.primCount; A.sumP=t.sumP; A.sumP2=t.sumP2; return; }
        A.a=unionAabb(A.a,t.a); A.E+=t.p; A.cone=coneUnion(A.cone,t.cone); A.N+=t.primCount; A.sumP+=t.sumP; A.sumP2+=t.sumP2; }
    static void aggTMerge(AggT& A, const AggT& B){ if (!B.valid) return; if (!A.valid){ A=B; return; } A.a=unionAabb(A.a,B.a); A.E+=B.E; A.cone=coneUnion(A.cone,B.cone); A.N+=B.N; A.sumP+=B.sumP; A.sumP2+=B.sumP2; }

    uint32_t buildTLASRecursive_SAOH(std::vector<TItem>& it, uint32_t begin, uint32_t end)
    {
        const uint32_t nodeIdx = static_cast<uint32_t>(m_tlas.size());
        m_tlas.push_back({});

        auto nodeAt = [&](uint32_t i) -> LightTLASNodeGpu& { return m_tlas[i]; };
        LightTLASNodeGpu& N0 = nodeAt(nodeIdx);

        // Aggregate
        AggT parent{}; for (uint32_t i=begin;i<end;++i) aggTAdd(parent, it[i]);

        N0.bmin = parent.a.mn; N0.bmax = parent.a.mx; N0.power = parent.E;
        N0.axis = parent.cone.axis;
        N0.cosTheta_o = std::cos(clampf(parent.cone.theta_o, 0.f, LT_PI));
        N0.cosTheta_e = std::cos(clampf(parent.cone.theta_e, 0.f, LT_PI));
        N0.primCount = parent.N; N0.sumPower = parent.sumP; N0.sumPowerSq = parent.sumP2;
        N0.itemFirst = begin; N0.itemCount = end - begin;
        N0.firstChild = 0xFFFFFFFF; N0.childCount = 0;
        N0.blasIndex = UINT32_MAX;

        const uint32_t count = end - begin;
        if (count == 1) {
            N0.blasIndex = it[begin].idx;
            return nodeIdx;
        }

        //helper, binary split on [b0,e0)
        auto findBinarySplit = [&](uint32_t b0, uint32_t e0,
                                   int& axisOut, float& splitPosOut, uint32_t& midOut)->bool
        {
            AggT parentL{}; for (uint32_t i=b0;i<e0;++i) aggTAdd(parentL, it[i]);

            const Aabb aabb = parentL.a;
            const XMFLOAT3 ext = aabbExtent(aabb);
            const float lenX=ext.x, lenY=ext.y, lenZ=ext.z, lenMax=(std::fmax)(lenX,(std::fmax)(lenY,lenZ));
            const float parentMA = (std::fmax)(1e-12f, aabbSurfaceArea(aabb));
            const float parentMO = (std::fmax)(1e-12f, orientationMeasure(parentL.cone));

            struct Best { float cost=std::numeric_limits<float>::infinity(); int axis=-1; float pos=0; } best;
            const uint32_t B = (std::fmax)(4u, (std::fmin)(64u, m_cfg.buildBins));

            for (int axis=0; axis<3; ++axis)
            {
                float mn = (&it[b0].c.x)[axis], mx = mn;
                for (uint32_t i=b0;i<e0;++i){ float v = (&it[i].c.x)[axis]; mn=(std::fmin)(mn,v); mx=(std::fmax)(mx,v); }
                float span = mx - mn; if (span <= 1e-20f) continue;

                std::vector<AggT> bins(B);
                float invSpan = 1.f/span;
                for (uint32_t i=b0;i<e0;++i){
                    float v = (&it[i].c.x)[axis];
                    uint32_t bi = (std::fmin)(B-1u, (uint32_t)std::floor((v - mn)*invSpan*B));
                    aggTAdd(bins[bi], it[i]);
                }

                std::vector<AggT> pref(B), suff(B);
                for (uint32_t i=0;i<B;++i){ if (i==0) pref[i]=bins[i]; else { pref[i]=pref[i-1]; aggTMerge(pref[i], bins[i]); } }
                for (int i=(int)B-1;i>=0;--i){ if ((uint32_t)i==B-1) suff[i]=bins[i]; else { suff[i]=suff[i+1]; aggTMerge(suff[i], bins[i]); } }

                float length_i = (axis==0?lenX:(axis==1?lenY:lenZ));
                float Kr = (length_i>1e-20f) ? (lenMax/length_i) : 1e6f;

                for (uint32_t s=1; s<B; ++s){
                    const AggT& L = pref[s-1]; const AggT& R = suff[s]; if (!L.valid || !R.valid) continue;
                    float ML  = aabbSurfaceArea(L.a),  MR  = aabbSurfaceArea(R.a);
                    float MoL = orientationMeasure(L.cone);
                    float MoR = orientationMeasure(R.cone);

                    float cost;
                    if (m_cfg.heuristic == Settings::Heuristic::SAH) {
                        cost = L.N * ML + R.N * MR;
                    } else {
                        cost = Kr * (L.E * ML * MoL + R.E * MR * MoR) / (parentMA * parentMO);
                    }
                    if (cost < best.cost){ best.cost = cost; best.axis = axis; best.pos = mn + (span * (float)s / (float)B); }
                }
            }

            if (best.axis < 0 || !std::isfinite(best.cost)) return false;

            auto midIter = std::partition(it.begin()+b0, it.begin()+e0,
                [&](const TItem& t){ return (&t.c.x)[best.axis] < best.pos; });

            uint32_t mid = static_cast<uint32_t>(midIter - (it.begin()+b0)) + b0;
            if (mid==b0 || mid==e0){
                mid = (b0+e0)/2;
                std::nth_element(it.begin()+b0, it.begin()+mid, it.begin()+e0,
                    [&](const TItem& A, const TItem& B){ return (&A.c.x)[best.axis] < (&B.c.x)[best.axis]; });
            }

            axisOut = best.axis; splitPosOut = best.pos; midOut = mid; return true;
        };

        int ax; float pos; uint32_t mid;
        bool ok = findBinarySplit(begin, end, ax, pos, mid);
        if (!ok) {
            // fallback
            int widest = 0;
            XMFLOAT3 e = aabbExtent(parent.a);
            if (e.y > e.x && e.y >= e.z) widest = 1;
            else if (e.z > e.x && e.z >= e.y) widest = 2;
            mid = (begin + end) / 2;
            std::nth_element(it.begin()+begin, it.begin()+mid, it.begin()+end,
                             [&](const TItem& A, const TItem& B){ return (&A.c.x)[widest] < (&B.c.x)[widest]; });
        }

        struct Range { uint32_t b, e; };
        Range buckets[4]; uint32_t bucketCount = 0;

        auto pushOrSplitOnce = [&](uint32_t b, uint32_t e){
            if (e <= b) return;
            const uint32_t c = e - b;
            if (c == 1) { buckets[bucketCount++] = {b,e}; return; }
            int ax2; float pos2; uint32_t mid2;
            if (findBinarySplit(b, e, ax2, pos2, mid2) && mid2>b && mid2<e) {
                buckets[bucketCount++] = {b, mid2};
                buckets[bucketCount++] = {mid2, e};
            } else {
                buckets[bucketCount++] = {b, e};
            }
        };

        pushOrSplitOnce(begin, mid);
        pushOrSplitOnce(mid,   end);

        while (bucketCount > 4) {
            uint32_t iMin=0, jMin=1; uint32_t best = UINT32_MAX;
            for (uint32_t i=0;i<bucketCount;i++)
                for (uint32_t j=i+1;j<bucketCount;j++){
                    uint32_t s = (buckets[i].e-buckets[i].b) + (buckets[j].e-buckets[j].b);
                    if (s < best){ best = s; iMin=i; jMin=j; }
                }
            buckets[iMin].b = (std::fmin)(buckets[iMin].b, buckets[jMin].b);
            buckets[iMin].e = (std::fmax)(buckets[iMin].e, buckets[jMin].e);
            for (uint32_t k=jMin+1;k<bucketCount;k++) buckets[k-1] = buckets[k];
            --bucketCount;
        }

        nodeAt(nodeIdx).firstChild = static_cast<uint32_t>(m_tlas.size());
        nodeAt(nodeIdx).childCount = bucketCount;
        for (uint32_t i=0;i<bucketCount;i++) m_tlas.push_back({});

        for (uint32_t c=0;c<bucketCount;c++){
            uint32_t built = buildTLASRecursive_SAOH(it, buckets[c].b, buckets[c].e);
            uint32_t desired = nodeAt(nodeIdx).firstChild + c;
            if (built != desired) std::swap(m_tlas[built], m_tlas[desired]);
        }

        return nodeIdx;
    }



    //====================================
    //UPLOAD HELPERS
    //====================================
    template<typename T>
    ComPtr<ID3D12Resource> uploadVector(ID3D12Device* device, ID3D12GraphicsCommandList* cmd, const std::vector<T>& v){
        if (v.empty()) return {}; const UINT64 bytes = static_cast<UINT64>(v.size()*sizeof(T));
#if LT_ENABLE_TIMING
        auto __t0 = std::chrono::high_resolution_clock::now();
#endif
        ComPtr<ID3D12Resource> upload; CD3DX12_HEAP_PROPERTIES hpU(D3D12_HEAP_TYPE_UPLOAD); auto desc = CD3DX12_RESOURCE_DESC::Buffer(bytes);
        device->CreateCommittedResource(&hpU, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload));
        void* p=nullptr; CD3DX12_RANGE r(0,0); upload->Map(0,&r,&p); memcpy(p, v.data(), bytes); upload->Unmap(0,nullptr);
        ComPtr<ID3D12Resource> gpu; CD3DX12_HEAP_PROPERTIES hpD(D3D12_HEAP_TYPE_DEFAULT);
        device->CreateCommittedResource(&hpD, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&gpu));
        cmd->CopyBufferRegion(gpu.Get(), 0, upload.Get(), 0, bytes);
        auto br = CD3DX12_RESOURCE_BARRIER::Transition(gpu.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_GENERIC_READ); cmd->ResourceBarrier(1, &br);
        m_gpu.staging.push_back(upload);
#if LT_ENABLE_TIMING
        auto __ms = std::chrono::duration<double, std::milli>( std::chrono::high_resolution_clock::now() - __t0 ).count();
        LT_LOG(L"uploadVector: " << (bytes/1024.0) << L" KiB in " << __ms << L" ms");
#endif
        return gpu;
    }

    uint32_t totalBLASNodeCount() const { uint32_t n=0; for (auto& b: m_blas) n += static_cast<uint32_t>(b.nodes.size()); return n; }
    uint32_t totalLeafIndexCount() const { uint32_t n=0; for (auto& b: m_blas) n += static_cast<uint32_t>(b.leafTriList.size()); return n; }

    static LightBLASNodeGpu toGpu(const BLASNode& n)
    {
        LightBLASNodeGpu g{};
        g.bmin = n.aabb.mn; g.bmax = n.aabb.mx; g.power = n.power;
        g.axis = n.cone.axis;
        g.cosTheta_o = std::cos(clampf(n.cone.theta_o, 0.f, LT_PI));
        g.cosTheta_e = std::cos(clampf(n.cone.theta_e, 0.f, LT_PI));

        g.firstChild = n.firstChild;
        g.childCount = n.childCount;

        g.triFirst = n.triFirst;
        g.triCount = n.triCount;

        g.primCount = n.primCount; g._pad0 = 0;
        g.sumPower = n.sumPower; g.sumPowerSq = n.sumPowerSq;
        return g;
    }
};

} // namespace lt
