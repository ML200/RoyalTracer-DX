#include "ClusterGeometry.h"
#include "../Core/DeviceContext.h"
#include "../DXRHelper.h"
#include <algorithm>
#include <cstring>
#include <intrin.h>
#include <stdexcept>

namespace clusters {

namespace {
inline uint32_t spread_bits_3(uint32_t v) {
    v &= 0x3FFu;
    v = (v | (v << 16)) & 0x030000FFu;
    v = (v | (v << 8)) & 0x0300F00Fu;
    v = (v | (v << 4)) & 0x030C30C3u;
    v = (v | (v << 2)) & 0x09249249u;
    return v;
}

#if RDN_CLUSTER_API
// Execute must repeat the prebuild query's inputs, so both come from here.
D3D12_RTAS_CLUSTER_TRIANGLES_INPUTS_DESC build_inputs(const D3D12_RTAS_CLUSTER_LIMITS& limits, bool omm) {
    D3D12_RTAS_CLUSTER_TRIANGLES_INPUTS_DESC d = {};
    d.ClusterLimits = limits;
    d.Flags = D3D12_RTAS_OPERATION_FLAG_FAST_TRACE | (omm ? D3D12_RTAS_OPERATION_FLAG_ALLOW_OMM : D3D12_RTAS_OPERATION_FLAG_NONE);
    d.Mode = D3D12_RTAS_OPERATION_MODE_IMPLICIT_DESTINATIONS;
    d.VertexFormat = D3D12_VERTEX_FORMAT_FLOAT32_3;
    d.IndexFormat = D3D12_INDEX_FORMAT_UINT32;
    d.GeometryIndexAndFlagsIndexFormat = D3D12_INDEX_FORMAT_NONE;
    d.OpacityMicromapIndexFormat = omm ? D3D12_INDEX_FORMAT_UINT32 : D3D12_INDEX_FORMAT_NONE;
    return d;
}

D3D12_RTAS_CLAS_INPUTS_DESC blas_inputs(uint32_t clusterCount, bool omm) {
    D3D12_RTAS_CLAS_INPUTS_DESC d = {};
    d.Flags = D3D12_RTAS_OPERATION_FLAG_FAST_TRACE | (omm ? D3D12_RTAS_OPERATION_FLAG_ALLOW_OMM : D3D12_RTAS_OPERATION_FLAG_NONE);
    d.MaxArgCount = 1;
    d.Mode = D3D12_RTAS_OPERATION_MODE_EXPLICIT_DESTINATIONS;
    d.MaxTotalClasCount = clusterCount;
    d.MaxClasCountPerArg = clusterCount;
    return d;
}

D3D12_RTAS_CLUSTER_MOVES_DESC move_inputs(uint32_t clusterCount, uint64_t bytes, bool omm, bool explicitDest = false) {
    D3D12_RTAS_CLUSTER_MOVES_DESC d = {};
    d.Flags = D3D12_RTAS_CLUSTER_MOVE_OPERATION_FLAG_NO_OVERLAP |
              (omm ? D3D12_RTAS_CLUSTER_MOVE_OPERATION_FLAG_ALLOW_OMM : D3D12_RTAS_CLUSTER_MOVE_OPERATION_FLAG_NONE);
    d.MaxArgCount = clusterCount;
    d.Mode = explicitDest ? D3D12_RTAS_OPERATION_MODE_EXPLICIT_DESTINATIONS : D3D12_RTAS_OPERATION_MODE_IMPLICIT_DESTINATIONS;
    d.Type = D3D12_RTAS_MOVE_OPERATION_TYPE_CLUSTER_LEVEL_ACCELERATION_STRUCTURE;
    d.MaxBytesMoved = bytes;
    return d;
}

D3D12_RTAS_OPERATION_INPUTS rtas_inputs(const D3D12_RTAS_CLUSTER_TRIANGLES_INPUTS_DESC* d) {
    D3D12_RTAS_OPERATION_INPUTS in = {};
    in.Type = D3D12_RTAS_OPERATION_TYPE_BUILD_CLAS_FROM_TRIANGLES;
    in.pClusterTrianglesDesc = d;
    return in;
}
D3D12_RTAS_OPERATION_INPUTS rtas_inputs(const D3D12_RTAS_CLUSTER_MOVES_DESC* d) {
    D3D12_RTAS_OPERATION_INPUTS in = {};
    in.Type = D3D12_RTAS_OPERATION_TYPE_MOVE_CLUSTER_OBJECTS;
    in.pClusterMovesDesc = d;
    return in;
}
D3D12_RTAS_OPERATION_INPUTS rtas_inputs(const D3D12_RTAS_CLAS_INPUTS_DESC* d) {
    D3D12_RTAS_OPERATION_INPUTS in = {};
    in.Type = D3D12_RTAS_OPERATION_TYPE_BUILD_BLAS_FROM_CLAS;
    in.pClasDesc = d;
    return in;
}

constexpr uint64_t BLAS_ARG_BYTES = sizeof(D3D12_RTAS_OPERATION_BUILD_BLAS_FROM_CLAS_ARGS) + sizeof(D3D12_GPU_VIRTUAL_ADDRESS);
inline uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) & ~(a - 1); }
#endif
} // namespace

void morton_cells(const float* centroids, uint32_t unitCount, uint32_t unitTris, uint32_t maxTris,
                  std::vector<uint32_t>& order, std::vector<uint32_t>& cells) {
    order.clear();
    cells.clear();
    if (!unitCount) return;
    float lo[3] = { 1e30f, 1e30f, 1e30f }, hi[3] = { -1e30f, -1e30f, -1e30f };
    for (uint32_t u = 0; u < unitCount; ++u)
        for (int a = 0; a < 3; ++a) {
            lo[a] = std::min(lo[a], centroids[3ull * u + a]);
            hi[a] = std::max(hi[a], centroids[3ull * u + a]);
        }
    float scale[3];
    for (int a = 0; a < 3; ++a) scale[a] = hi[a] > lo[a] ? 1023.0f / (hi[a] - lo[a]) : 0.0f;
    thread_local std::vector<std::pair<uint32_t, uint32_t>> keys; // Morton code, unit
    keys.clear();
    for (uint32_t u = 0; u < unitCount; ++u) {
        uint32_t q[3];
        for (int a = 0; a < 3; ++a) q[a] = (uint32_t)((centroids[3ull * u + a] - lo[a]) * scale[a] + 0.5f);
        keys.emplace_back(spread_bits_3(q[0]) | (spread_bits_3(q[1]) << 1) | (spread_bits_3(q[2]) << 2), u);
    }
    std::sort(keys.begin(), keys.end());
    order.reserve(unitCount);
    for (const auto& kv : keys) order.push_back(kv.second);
    // Split at the highest Morton bit that differs across a cell, as an octree (or LBVH) would.
    const uint32_t maxUnits = std::max(1u, maxTris / std::max(1u, unitTris));
    auto cell = [&](auto& self, uint32_t qb, uint32_t qe) -> void {
        if (qe - qb <= maxUnits) { cells.push_back(qb); return; }
        const uint32_t first = keys[qb].first, last = keys[qe - 1].first;
        uint32_t mid = qb + (qe - qb) / 2u;
        if (first != last) {
            const uint32_t bit = 1u << (31u - (uint32_t)__lzcnt(first ^ last));
            mid = (uint32_t)(std::partition_point(keys.begin() + qb, keys.begin() + qe,
                                                  [&](const auto& kv) { return (kv.first & bit) == 0; }) - keys.begin());
        }
        self(self, qb, mid);
        self(self, mid, qe);
    };
    cell(cell, 0, unitCount);
}

void split(const uint32_t* indices, uint32_t vertexCount, const std::vector<uint32_t>& cuts, uint32_t end,
           std::vector<Span>& out) {
    thread_local std::vector<uint32_t> seen;
    thread_local uint32_t stamp = 0;
    if (seen.size() < vertexCount) seen.resize(vertexCount, 0u);
    for (size_t i = 0; i < cuts.size(); ++i) {
        uint32_t t = cuts[i];
        const uint32_t cellEnd = i + 1 < cuts.size() ? cuts[i + 1] : end;
        while (t < cellEnd) {
            if (++stamp == 0) { std::fill(seen.begin(), seen.end(), 0u); stamp = 1; }
            Span s{ t, 0, 0 };
            for (; t < cellEnd && s.triCount < MAX_TRIS; ++t) {
                const uint32_t* tri = &indices[3ull * t];
                uint32_t fresh = 0;
                for (int k = 0; k < 3; ++k) {
                    bool known = seen[tri[k]] == stamp;
                    for (int j = 0; j < k && !known; ++j) known = tri[j] == tri[k];
                    fresh += known ? 0u : 1u;
                }
                if (s.uniqueVerts + fresh > MAX_VERTS) break;
                for (int k = 0; k < 3; ++k) seen[tri[k]] = stamp;
                s.uniqueVerts = (uint16_t)(s.uniqueVerts + fresh);
                ++s.triCount;
            }
            out.push_back(s);
        }
    }
}

#if RDN_CLUSTER_API
D3D12_RTAS_CLUSTER_LIMITS limits(const Span* spans, uint32_t count, uint32_t maxGeometryIndex, bool omm) {
    D3D12_RTAS_CLUSTER_LIMITS lim = {};
    lim.MaxArgCount = count;
    lim.MaxGeometryIndexValue = maxGeometryIndex;
    lim.MaxUniqueGeometryIndexAndFlagsCountPerCluster = 1;
    for (uint32_t i = 0; i < count; ++i) {
        lim.MaxTriangleCountPerCluster = std::max<UINT>(lim.MaxTriangleCountPerCluster, spans[i].triCount);
        lim.MaxUniqueVertexIndexCountPerCluster = std::max<UINT>(lim.MaxUniqueVertexIndexCountPerCluster, spans[i].uniqueVerts);
        lim.MaxTotalTriangleCount += spans[i].triCount;
        lim.MaxSumOfUniqueVertexIndexCounts += spans[i].uniqueVerts;
    }
    lim.MaxOpacityMicromapIndicesPerCluster = omm ? lim.MaxTriangleCountPerCluster : 0u;
    return lim;
}
#endif

void ClusterGeometry::init(DeviceContext* ctx, uint32_t slots, const Config& cfg) {
    m_enabled = false;
    m_cfg = cfg;
#if RDN_CLUSTER_API
    if (!ctx->ClusteredGeometry()) return;
    ComPtr<ID3D12CommandListRaytracing2> list;
    if (FAILED(ctx->NativeDevice()->QueryInterface(IID_PPV_ARGS(&m_device))) ||
        FAILED(ctx->ComputeList()->QueryInterface(IID_PPV_ARGS(&list))) ||
        FAILED(ctx->CmdList()->QueryInterface(IID_PPV_ARGS(&list)))) {
        m_device.Reset();
        std::printf("[clusters] RTAS interfaces unavailable, classic BLAS only\n");
        return;
    }
    m_ctx = ctx;
    ID3D12Device* device = ctx->Device();
    using nv_helpers_dx12::CreateBuffer;
    D3D12_RANGE noRead{ 0, 0 };
    void* p = nullptr;
    create_id_tables(cfg.clusterCapacity);
    ++slots; // sync_slot()
    m_args.resize(slots); m_argsMapped.assign(slots, nullptr);
    m_sizes.resize(slots); m_sizesReadback.resize(slots);
    m_sizesMapped.assign(slots, nullptr); m_sizesCopied.assign(slots, 0); m_gen.assign(slots, 0);
    m_argCapacity.assign(slots, cfg.argBytesPerSlot); m_argCapacity.back() = cfg.syncArgBytes;
    m_sizeCapacity.assign(slots, cfg.sizeEntriesPerSlot); m_sizeCapacity.back() = cfg.syncSizeEntries;
    for (uint32_t i = 0; i < slots; ++i) {
        m_args[i] = CreateBuffer(device, m_argCapacity[i], D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ,
                                 nv_helpers_dx12::kUploadHeapProps);
        m_args[i]->SetName(L"ClusterArgs");
        if (FAILED(m_args[i]->Map(0, &noRead, &p))) throw std::runtime_error("clusters: argument Map failed");
        m_argsMapped[i] = static_cast<uint8_t*>(p);
        const uint64_t sizeBytes = (uint64_t)m_sizeCapacity[i] * sizeof(uint64_t);
        m_sizes[i] = CreateBuffer(device, sizeBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                  D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nv_helpers_dx12::kDefaultHeapProps);
        m_sizes[i]->SetName(L"ClusterBuiltSizes");
        m_sizesReadback[i] = CreateBuffer(device, sizeBytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST,
                                          nv_helpers_dx12::kReadbackHeapProps);
        m_sizesReadback[i]->SetName(L"ClusterBuiltSizesReadback");
        if (FAILED(m_sizesReadback[i]->Map(0, nullptr, &p))) throw std::runtime_error("clusters: size readback Map failed");
        m_sizesMapped[i] = static_cast<const uint64_t*>(p);
    }
    m_enabled = true;
    std::printf("[clusters] %uM cluster IDs, %u frame slots + 1 sync slot\n", cfg.clusterCapacity >> 20, slots - 1);
#else
    (void)ctx; (void)slots;
#endif
}

void ClusterGeometry::create_id_tables(uint32_t capacity) {
#if RDN_CLUSTER_API
    ID3D12Device* device = m_ctx->Device();
    using nv_helpers_dx12::CreateBuffer;
    const uint64_t cap = capacity;
    D3D12_RANGE noRead{ 0, 0 };
    void* p = nullptr;
    m_records = CreateBuffer(device, cap * sizeof(ClusterRecord), D3D12_RESOURCE_FLAG_NONE,
                             D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
    m_records->SetName(L"ClusterRecords");
    if (FAILED(m_records->Map(0, &noRead, &p))) throw std::runtime_error("clusters: record Map failed");
    m_recordsMapped = static_cast<ClusterRecord*>(p);
    m_built = CreateBuffer(device, cap * sizeof(D3D12_GPU_VIRTUAL_ADDRESS), D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nv_helpers_dx12::kDefaultHeapProps);
    m_built->SetName(L"ClasBuiltAddresses");
    m_builtVa = m_built->GetGPUVirtualAddress();
    m_final = CreateBuffer(device, cap * sizeof(D3D12_GPU_VIRTUAL_ADDRESS), D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nv_helpers_dx12::kDefaultHeapProps);
    m_final->SetName(L"ClasAddresses");
    m_finalVa = m_final->GetGPUVirtualAddress();
    m_cfg.clusterCapacity = capacity;
    m_ids.free.assign(1, { 0u, capacity });
    m_ids.used = 0;
#else
    (void)capacity;
#endif
}

bool ClusterGeometry::reserve_ids(uint32_t count) {
    if (!m_enabled || count <= m_cfg.clusterCapacity) return true;
    if (m_ids.used) return false;
    create_id_tables(count);
    std::printf("[clusters] %.1fM cluster IDs\n", count / 1048576.0);
    return true;
}

bool ClusterGeometry::allocate_ids(uint32_t count, uint32_t& base) {
    std::lock_guard<std::mutex> lk(m_idMutex);
    for (size_t i = 0; i < m_ids.free.size(); ++i) {
        auto& r = m_ids.free[i];
        if (r.second < count) continue;
        base = r.first;
        r.first += count;
        r.second -= count;
        if (!r.second) m_ids.free.erase(m_ids.free.begin() + (ptrdiff_t)i);
        m_ids.used += count;
        return true;
    }
    return false;
}

void ClusterGeometry::free_ids(uint32_t base, uint32_t count) {
    if (!count) return;
    std::lock_guard<std::mutex> lk(m_idMutex);
    auto& f = m_ids.free;
    auto it = std::lower_bound(f.begin(), f.end(), std::make_pair(base, 0u));
    it = f.insert(it, { base, count });
    // Coalesce with the neighbours.
    if (it + 1 != f.end() && it->first + it->second == (it + 1)->first) { it->second += (it + 1)->second; f.erase(it + 1); }
    if (it != f.begin() && (it - 1)->first + (it - 1)->second == it->first) { (it - 1)->second += it->second; f.erase(it); }
    m_ids.used -= count;
}

void ClusterGeometry::write_records(uint32_t base, const Span* spans, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) m_recordsMapped[base + i] = ClusterRecord{ spans[i].firstTri, 0u };
}

void ClusterGeometry::write_info(uint32_t base, const uint32_t* info, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) m_recordsMapped[base + i].info = info[i];
}

#if RDN_CLUSTER_API
bool ClusterGeometry::prebuild(const D3D12_RTAS_CLUSTER_LIMITS& limits, bool omm, Sizes& out) const {
    const D3D12_RTAS_CLUSTER_TRIANGLES_INPUTS_DESC tris = build_inputs(limits, omm);
    const D3D12_RTAS_CLAS_INPUTS_DESC blas = blas_inputs(limits.MaxArgCount, omm);
    const D3D12_RTAS_OPERATION_INPUTS buildIn = rtas_inputs(&tris), blasIn = rtas_inputs(&blas);
    D3D12_RTAS_OPERATION_PREBUILD_INFO buildInfo = {}, blasInfo = {};
    m_device->GetRTASOperationPrebuildInfo(&buildIn, &buildInfo);
    m_device->GetRTASOperationPrebuildInfo(&blasIn, &blasInfo);
    if (!buildInfo.ResultDataMaxSizeInBytes || !blasInfo.ResultDataMaxSizeInBytes) return false;
    out.clas = align_up(buildInfo.ResultDataMaxSizeInBytes, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT);
    out.clasScratch = align_up(buildInfo.ScratchDataSizeInBytes, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT);
    out.blas = align_up(blasInfo.ResultDataMaxSizeInBytes, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT);
    return true;
}

void ClusterGeometry::blas_prebuild(uint32_t clusterCount, bool omm, uint64_t& size, uint64_t& scratch) const {
    const D3D12_RTAS_CLAS_INPUTS_DESC blas = blas_inputs(clusterCount, omm);
    const D3D12_RTAS_OPERATION_INPUTS in = rtas_inputs(&blas);
    D3D12_RTAS_OPERATION_PREBUILD_INFO info = {};
    m_device->GetRTASOperationPrebuildInfo(&in, &info);
    size = align_up(info.ResultDataMaxSizeInBytes, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT);
    scratch = align_up(info.ScratchDataSizeInBytes, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT);
}

uint64_t ClusterGeometry::move_scratch(uint32_t clusterCount, uint64_t clasBytes, bool omm, bool explicitDest) const {
    const D3D12_RTAS_CLUSTER_MOVES_DESC move = move_inputs(clusterCount, clasBytes, omm, explicitDest);
    const D3D12_RTAS_OPERATION_INPUTS in = rtas_inputs(&move);
    D3D12_RTAS_OPERATION_PREBUILD_INFO info = {};
    m_device->GetRTASOperationPrebuildInfo(&in, &info);
    return align_up(info.ScratchDataSizeInBytes, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT);
}

uint64_t ClusterGeometry::finish_scratch(uint32_t clusterCount, uint64_t clasBytes, bool omm) const {
    uint64_t blasSize = 0, blasScratch = 0;
    blas_prebuild(clusterCount, omm, blasSize, blasScratch);
    return move_scratch(clusterCount, clasBytes, omm) + blasScratch;
}
#endif

void ClusterGeometry::begin(uint32_t slot, ID3D12GraphicsCommandList4* cl) {
    m_slot = slot;
    m_argsUsed = 0;
    m_sizesUsed = 0;
    ++m_gen[slot];
#if RDN_CLUSTER_API
    m_builds.clear();
    m_moves.clear();
    m_blas.clear();
    m_builtBlas.clear();
#endif
    if (m_sizesCopied[slot]) {
        const D3D12_RESOURCE_BARRIER tr = CD3DX12_RESOURCE_BARRIER::Transition(
            m_sizes[slot].Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        cl->ResourceBarrier(1, &tr);
        m_sizesCopied[slot] = 0;
    }
}

bool ClusterGeometry::can_queue(uint32_t clusterCount, bool finish) const {
#if RDN_CLUSTER_API
    const uint64_t args = finish ? BLAS_ARG_BYTES : (uint64_t)clusterCount * sizeof(D3D12_RTAS_OPERATION_BUILD_CLAS_FROM_TRIANGLES_ARGS);
    const uint32_t sizes = finish ? 1u : clusterCount;
    return m_argsUsed + args <= m_argCapacity[m_slot] && m_sizesUsed + sizes <= m_sizeCapacity[m_slot];
#else
    (void)clusterCount; (void)finish;
    return false;
#endif
}

bool ClusterGeometry::can_queue_blas(uint32_t clasCount) const {
#if RDN_CLUSTER_API
    const uint64_t args = BLAS_ARG_BYTES + (uint64_t)clasCount * sizeof(D3D12_GPU_VIRTUAL_ADDRESS);
    return m_argsUsed + args <= m_argCapacity[m_slot] && m_sizesUsed + 1 <= m_sizeCapacity[m_slot];
#else
    (void)clasCount;
    return false;
#endif
}

#if RDN_CLUSTER_API
ClusterGeometry::Ticket ClusterGeometry::queue_build(const Mesh& mesh, D3D12_GPU_VIRTUAL_ADDRESS result,
                                                     D3D12_GPU_VIRTUAL_ADDRESS scratch) {
    const uint32_t n = mesh.clusterCount;
    const uint64_t argsOff = m_argsUsed;
    auto* args = reinterpret_cast<D3D12_RTAS_OPERATION_BUILD_CLAS_FROM_TRIANGLES_ARGS*>(m_argsMapped[m_slot] + argsOff);
    const UINT alphaGeometry = mesh.alphaGeometry != ~0u ? mesh.alphaGeometry : (mesh.opaqueTriCount ? 1u : 0u);
    for (uint32_t i = 0; i < n; ++i) {
        const Span& s = mesh.spans[i];
        const bool alpha = s.firstTri >= mesh.opaqueTriCount;
        D3D12_RTAS_OPERATION_BUILD_CLAS_FROM_TRIANGLES_ARGS a = {};
        a.ClusterID = mesh.clusterBase + i;
        a.TriangleCount = s.triCount;
        a.UniqueVertexIndexCount = s.uniqueVerts;
        a.BaseGeometryIndexAndFlags = alpha ? alphaGeometry : (UINT)D3D12_RTAS_CLUSTERED_GEOMETRY_FLAG_OPAQUE;
        a.VertexBufferStride = (UINT16)mesh.vertexStride;
        a.VertexBuffer = mesh.vertices;
        a.IndexBuffer = mesh.indices + 3ull * s.firstTri * sizeof(uint32_t);
        if (alpha && mesh.omm()) {
            a.OpacityMicromapArray = mesh.ommArray;
            a.OpacityMicromapIndexBuffer = mesh.ommIndices + (uint64_t)(s.firstTri - mesh.opaqueTriCount) * sizeof(uint32_t);
        }
        args[i] = a;
    }
    m_argsUsed += (uint64_t)n * sizeof(D3D12_RTAS_OPERATION_BUILD_CLAS_FROM_TRIANGLES_ARGS);
    const Ticket t{ m_slot, m_sizesUsed, m_gen[m_slot] };
    m_sizesUsed += n;

    auto& op = m_builds.emplace_back();
    op.in = build_inputs(mesh.limits, mesh.omm());
    op.data = {};
    op.data.BatchResultData = result;
    op.data.BatchScratchData = scratch;
    op.data.ResultAddressArray = { m_builtVa + (uint64_t)mesh.clusterBase * sizeof(D3D12_GPU_VIRTUAL_ADDRESS),
                                   sizeof(D3D12_GPU_VIRTUAL_ADDRESS) };
    op.data.ResultSizeArray = { m_sizes[m_slot]->GetGPUVirtualAddress() + (uint64_t)t.index * sizeof(uint64_t),
                                sizeof(uint64_t) };
    op.data.IndirectArgumentArray = { m_args[m_slot]->GetGPUVirtualAddress() + argsOff,
                                      sizeof(D3D12_RTAS_OPERATION_BUILD_CLAS_FROM_TRIANGLES_ARGS) };
    return t;
}

void ClusterGeometry::queue_move(uint32_t clusterBase, uint32_t clusterCount, bool omm, D3D12_GPU_VIRTUAL_ADDRESS clasDest,
                                 uint64_t clasBytes, D3D12_GPU_VIRTUAL_ADDRESS scratch) {
    const uint64_t table = (uint64_t)clusterBase * sizeof(D3D12_GPU_VIRTUAL_ADDRESS);
    auto& mv = m_moves.emplace_back();
    mv.in = move_inputs(clusterCount, clasBytes, omm);
    mv.data = {};
    mv.data.BatchResultData = clasDest;
    mv.data.BatchScratchData = scratch;
    mv.data.ResultAddressArray = { m_finalVa + table, sizeof(D3D12_GPU_VIRTUAL_ADDRESS) };
    // A move's argument is its source address, so the builds' address table is the argument array.
    mv.data.IndirectArgumentArray = { m_builtVa + table, sizeof(D3D12_RTAS_OPERATION_MOVE_CLUSTER_OBJECTS_ARGS) };
}

void ClusterGeometry::queue_move_to(uint32_t clusterBase, uint32_t clusterCount, bool omm, const uint64_t* dests,
                                    uint64_t clasBytes, D3D12_GPU_VIRTUAL_ADDRESS scratch) {
    const D3D12_GPU_VIRTUAL_ADDRESS destsVa = m_args[m_slot]->GetGPUVirtualAddress() + m_argsUsed;
    std::memcpy(m_argsMapped[m_slot] + m_argsUsed, dests, (size_t)clusterCount * sizeof(D3D12_GPU_VIRTUAL_ADDRESS));
    m_argsUsed += (uint64_t)clusterCount * sizeof(D3D12_GPU_VIRTUAL_ADDRESS);
    auto& mv = m_moves.emplace_back();
    mv.in = move_inputs(clusterCount, clasBytes, omm, true);
    mv.data = {};
    mv.data.BatchScratchData = scratch;
    mv.data.ResultAddressArray = { destsVa, sizeof(D3D12_GPU_VIRTUAL_ADDRESS) };
    mv.data.IndirectArgumentArray = { m_builtVa + (uint64_t)clusterBase * sizeof(D3D12_GPU_VIRTUAL_ADDRESS),
                                      sizeof(D3D12_RTAS_OPERATION_MOVE_CLUSTER_OBJECTS_ARGS) };
}

ClusterGeometry::Ticket ClusterGeometry::queue_finish(uint32_t clusterBase, uint32_t clusterCount, bool omm,
                                                      D3D12_GPU_VIRTUAL_ADDRESS clasDest, uint64_t clasBytes,
                                                      D3D12_GPU_VIRTUAL_ADDRESS blasDest, D3D12_GPU_VIRTUAL_ADDRESS scratch) {
    const uint64_t moveScratch = move_scratch(clusterCount, clasBytes, omm);
    queue_move(clusterBase, clusterCount, omm, clasDest, clasBytes, scratch);
    return queue_blas(clusterBase, clusterCount, omm, blasDest, scratch + moveScratch);
}

ClusterGeometry::Ticket ClusterGeometry::queue_blas(uint32_t clusterBase, uint32_t clusterCount, bool omm,
                                                    D3D12_GPU_VIRTUAL_ADDRESS blasDest, D3D12_GPU_VIRTUAL_ADDRESS scratch,
                                                    bool moved) {
    const uint64_t table = (uint64_t)clusterBase * sizeof(D3D12_GPU_VIRTUAL_ADDRESS) + (moved ? m_finalVa : m_builtVa);
    uint8_t* args = m_argsMapped[m_slot] + m_argsUsed;
    const D3D12_GPU_VIRTUAL_ADDRESS argsVa = m_args[m_slot]->GetGPUVirtualAddress() + m_argsUsed;
    auto* blasArgs = reinterpret_cast<D3D12_RTAS_OPERATION_BUILD_BLAS_FROM_CLAS_ARGS*>(args);
    blasArgs->ClasAddressCount = clusterCount;
    blasArgs->ClasAddressStride = sizeof(D3D12_GPU_VIRTUAL_ADDRESS);
    blasArgs->ClasAddressArray = table;
    *reinterpret_cast<D3D12_GPU_VIRTUAL_ADDRESS*>(args + sizeof(D3D12_RTAS_OPERATION_BUILD_BLAS_FROM_CLAS_ARGS)) = blasDest;
    m_argsUsed += BLAS_ARG_BYTES;
    const Ticket t{ m_slot, m_sizesUsed++, m_gen[m_slot] };

    auto& op = (moved ? m_blas : m_builtBlas).emplace_back();
    op.in = blas_inputs(clusterCount, omm);
    op.data = {};
    op.data.BatchScratchData = scratch;
    op.data.ResultAddressArray = { argsVa + sizeof(D3D12_RTAS_OPERATION_BUILD_BLAS_FROM_CLAS_ARGS),
                                   sizeof(D3D12_GPU_VIRTUAL_ADDRESS) };
    op.data.ResultSizeArray = { m_sizes[m_slot]->GetGPUVirtualAddress() + (uint64_t)t.index * sizeof(uint64_t),
                                sizeof(uint64_t) };
    op.data.IndirectArgumentArray = { argsVa, sizeof(D3D12_RTAS_OPERATION_BUILD_BLAS_FROM_CLAS_ARGS) };
    return t;
}

ClusterGeometry::Ticket ClusterGeometry::queue_blas_from(const uint64_t* clas, uint32_t count, uint32_t maxCount, bool omm,
                                                         D3D12_GPU_VIRTUAL_ADDRESS blasDest, D3D12_GPU_VIRTUAL_ADDRESS scratch) {
    // The address list follows the arguments in the slot's argument buffer.
    uint8_t* args = m_argsMapped[m_slot] + m_argsUsed;
    const D3D12_GPU_VIRTUAL_ADDRESS argsVa = m_args[m_slot]->GetGPUVirtualAddress() + m_argsUsed;
    auto* blasArgs = reinterpret_cast<D3D12_RTAS_OPERATION_BUILD_BLAS_FROM_CLAS_ARGS*>(args);
    blasArgs->ClasAddressCount = count;
    blasArgs->ClasAddressStride = sizeof(D3D12_GPU_VIRTUAL_ADDRESS);
    blasArgs->ClasAddressArray = argsVa + BLAS_ARG_BYTES;
    *reinterpret_cast<D3D12_GPU_VIRTUAL_ADDRESS*>(args + sizeof(D3D12_RTAS_OPERATION_BUILD_BLAS_FROM_CLAS_ARGS)) = blasDest;
    std::memcpy(args + BLAS_ARG_BYTES, clas, (size_t)count * sizeof(D3D12_GPU_VIRTUAL_ADDRESS));
    m_argsUsed += BLAS_ARG_BYTES + (uint64_t)count * sizeof(D3D12_GPU_VIRTUAL_ADDRESS);
    const Ticket t{ m_slot, m_sizesUsed++, m_gen[m_slot] };

    auto& op = m_blas.emplace_back();
    op.in = blas_inputs(maxCount, omm);
    op.data = {};
    op.data.BatchScratchData = scratch;
    op.data.ResultAddressArray = { argsVa + sizeof(D3D12_RTAS_OPERATION_BUILD_BLAS_FROM_CLAS_ARGS),
                                   sizeof(D3D12_GPU_VIRTUAL_ADDRESS) };
    op.data.ResultSizeArray = { m_sizes[m_slot]->GetGPUVirtualAddress() + (uint64_t)t.index * sizeof(uint64_t),
                                sizeof(uint64_t) };
    op.data.IndirectArgumentArray = { argsVa, sizeof(D3D12_RTAS_OPERATION_BUILD_BLAS_FROM_CLAS_ARGS) };
    return t;
}
#endif

void ClusterGeometry::copy_addresses(ID3D12GraphicsCommandList* cl, uint32_t base, uint32_t count, ID3D12Resource* dest,
                                     uint64_t offset) {
    using CD = CD3DX12_RESOURCE_BARRIER;
    const D3D12_RESOURCE_BARRIER toCopy = CD::Transition(m_final.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                                         D3D12_RESOURCE_STATE_COPY_SOURCE);
    cl->ResourceBarrier(1, &toCopy);
    cl->CopyBufferRegion(dest, offset, m_final.Get(), (uint64_t)base * sizeof(D3D12_GPU_VIRTUAL_ADDRESS),
                         (uint64_t)count * sizeof(D3D12_GPU_VIRTUAL_ADDRESS));
    const D3D12_RESOURCE_BARRIER back = CD::Transition(m_final.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                                                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    cl->ResourceBarrier(1, &back);
}

// One call per stage, each a batch: moves of CLAS built earlier, the cluster BLAS over the moved CLAS, the new CLAS
// builds, then cluster BLAS over CLAS just built in place. A stage's read-only address table is in
// NON_PIXEL_SHADER_RESOURCE while it runs.
void ClusterGeometry::record(ID3D12GraphicsCommandList4* cl) {
#if RDN_CLUSTER_API
    if (m_builds.empty() && m_moves.empty() && m_blas.empty() && m_builtBlas.empty()) return;
    ComPtr<ID3D12CommandListRaytracing2> list;
    if (FAILED(cl->QueryInterface(IID_PPV_ARGS(&list)))) throw std::runtime_error("clusters: RTAS command list unavailable");
    auto execute = [&](const auto& ops) {
        std::vector<D3D12_RTAS_OPERATION_DESC> descs(ops.size());
        for (size_t i = 0; i < ops.size(); ++i) {
            descs[i].Inputs = rtas_inputs(&ops[i].in);
            descs[i].pBatchedOperationData = &ops[i].data;
        }
        list->ExecuteIndirectRTASOperations((UINT)descs.size(), descs.data(), D3D12_EXECUTE_INDIRECT_RTAS_OPERATIONS_FLAG_NONE);
    };
    using CD = CD3DX12_RESOURCE_BARRIER;
    constexpr auto UAV = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    constexpr auto READ = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    // Acceleration structures live in the clients' buffers, so their writes are ordered with all-resource UAV barriers.
    const D3D12_RESOURCE_BARRIER allUav = CD::UAV(nullptr);
    // A second record this frame: the sizes copy of the first left the buffer in COPY_SOURCE.
    if (m_sizesCopied[m_slot]) {
        const D3D12_RESOURCE_BARRIER back = CD::Transition(m_sizes[m_slot].Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, UAV);
        cl->ResourceBarrier(1, &back);
        m_sizesCopied[m_slot] = 0;
    }
    if (!m_moves.empty()) {
        const D3D12_RESOURCE_BARRIER toMove = CD::Transition(m_built.Get(), UAV, READ);
        cl->ResourceBarrier(1, &toMove);
        execute(m_moves);
        const D3D12_RESOURCE_BARRIER moved[2] = { allUav, CD::Transition(m_built.Get(), READ, UAV) };
        cl->ResourceBarrier(2, moved);
    }
    if (!m_blas.empty()) {
        const D3D12_RESOURCE_BARRIER toBlas = CD::Transition(m_final.Get(), UAV, READ);
        cl->ResourceBarrier(1, &toBlas);
        execute(m_blas);
        const D3D12_RESOURCE_BARRIER done[2] = { allUav, CD::Transition(m_final.Get(), READ, UAV) };
        cl->ResourceBarrier(2, done);
    }
    if (!m_builds.empty()) {
        execute(m_builds);
        cl->ResourceBarrier(1, &allUav);
    }
    if (!m_builtBlas.empty()) {
        const D3D12_RESOURCE_BARRIER toBlas = CD::Transition(m_built.Get(), UAV, READ);
        cl->ResourceBarrier(1, &toBlas);
        execute(m_builtBlas);
        const D3D12_RESOURCE_BARRIER done[2] = { allUav, CD::Transition(m_built.Get(), READ, UAV) };
        cl->ResourceBarrier(2, done);
    }
    m_moves.clear();
    m_blas.clear();
    m_builds.clear();
    m_builtBlas.clear();
    // Built sizes, read back by the clients once this list has executed.
    const D3D12_RESOURCE_BARRIER toCopy = CD::Transition(m_sizes[m_slot].Get(), UAV, D3D12_RESOURCE_STATE_COPY_SOURCE);
    cl->ResourceBarrier(1, &toCopy);
    cl->CopyBufferRegion(m_sizesReadback[m_slot].Get(), 0, m_sizes[m_slot].Get(), 0, (uint64_t)m_sizesUsed * sizeof(uint64_t));
    m_sizesCopied[m_slot] = 1;
#else
    (void)cl;
#endif
}

bool ClusterGeometry::sizes(const Ticket& t, uint32_t count, uint64_t* out) const {
    if (!t.valid() || m_gen[t.slot] != t.gen) return false;
    for (uint32_t i = 0; i < count; ++i) out[i] = m_sizesMapped[t.slot][t.index + i];
    return true;
}

} // namespace clusters
