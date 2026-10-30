#include "InstanceSets.h"
#include "../Core/DeviceContext.h"
#include "../DXRHelper.h"
#include "../planet/tlas_builder.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>

namespace clusters {

#if RDN_CLUSTER_API
namespace {
// Layouts of InstanceSets_v8.hlsl.
struct GpuSet {
    uint32_t instanceBase, instanceCount, cellBase, cellCount;
    uint32_t bucketBase, bucketCount, partBase, partCount;
    uint32_t blockBase, blockCount, slotBase, partitionBase; // assemblies: pool blocks, part slots, their partitions
    uint32_t mask, pad[3];
    float center[3], radius;
};
struct GpuBucket {
    float error; // set space
    uint32_t record, hitGroup, flags;
    uint64_t blas;
    uint32_t pad[2];
    float base[12];
};
struct GpuPart {
    uint32_t bucketBase, bucketCount;
    float windWeight, pad;
    float pivot[3], pad2;
};
struct GpuCell {
    float center[3], radius;
    uint32_t first, count, set;
    float minScale, maxScale;
    uint32_t pad[3];
};
struct GpuInstance {
    float position[3], scale;
    uint32_t rotation[2]; // snorm16 x, y | z, w
};
struct Constants {
    float camera[3], pixelsPerUnit;
    float threshold, offscreen, znear;
    uint32_t cells;
    uint32_t instances, maxWrites, maxUpdates, tlasBase;
    uint32_t partitionBase, writeArgs, updateArgs, workLo;
    uint32_t workHi, translate, translateArgs, slots;
    float delta[3], time;
    uint32_t blockArgs, maxBlockWrites, poolTlasBase, poolPartitionBase;
    float windDistance, windAmplitude, windFrequency;
    uint32_t blocks;
    uint32_t sets, freeLists, partStates, poolPartitions;
    float planes[5][4];
};
static_assert(sizeof(GpuSet) == 80 && sizeof(GpuBucket) == 80 && sizeof(GpuPart) == 32 && sizeof(GpuCell) == 48 &&
              sizeof(GpuInstance) == 24 && sizeof(Constants) == 224);
constexpr uint32_t SRVS = 5, UAVS = 4;
// work: counters, the operation count and operations, the readback's copy, then the arguments.
constexpr uint32_t WORK_OP_COUNT = 32, WORK_OPS = 40, WORK_FEEDBACK = 160, FEEDBACK_BYTES = 32, WORK_ARGS = 256;
constexpr uint32_t WRITE_ARG_BYTES = 104, UPDATE_ARG_BYTES = 16, TRANSLATE_ARG_BYTES = 16;
// TLAS operations a frame may write: a new set's instances enter over several frames.
constexpr uint32_t MAX_WRITES = 128u << 10, MAX_UPDATES = 128u << 10, MAX_BLOCK_WRITES = 64u << 10;
constexpr uint32_t GROUP_INSTANCES = 256; // InstanceSets_v8.hlsl's per-instance passes

static_assert(sizeof(D3D12_RTAS_PARTITIONED_TLAS_OPERATION_WRITE_INSTANCE_ARGS) == WRITE_ARG_BYTES &&
              sizeof(D3D12_RTAS_PARTITIONED_TLAS_OPERATION_UPDATE_INSTANCE_ARGS) == UPDATE_ARG_BYTES &&
              sizeof(D3D12_RTAS_PARTITIONED_TLAS_OPERATION_TRANSLATE_PARTITION_ARGS) == TRANSLATE_ARG_BYTES);

inline uint64_t align(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

uint32_t snorm16(float v) { return (uint32_t)(int32_t)std::lround(std::clamp(v, -1.0f, 1.0f) * 32767.0f) & 0xFFFFu; }

// Spreads the low 21 bits of v to every third bit.
uint64_t spread3(uint64_t v) {
    v &= 0x1FFFFF;
    v = (v | v << 32) & 0x1F00000000FFFFull;
    v = (v | v << 16) & 0x1F0000FF0000FFull;
    v = (v | v << 8) & 0x100F00F00F00F00Full;
    v = (v | v << 4) & 0x10C30C30C30C30C3ull;
    v = (v | v << 2) & 0x1249249249249249ull;
    return v;
}

void rotate(const float q[4], const float v[3], float out[3]) {
    // v + 2 w (q x v) + 2 q x (q x v)
    const float t[3] = { 2.0f * (q[1] * v[2] - q[2] * v[1]), 2.0f * (q[2] * v[0] - q[0] * v[2]),
                         2.0f * (q[0] * v[1] - q[1] * v[0]) };
    out[0] = v[0] + q[3] * t[0] + (q[1] * t[2] - q[2] * t[1]);
    out[1] = v[1] + q[3] * t[1] + (q[2] * t[0] - q[0] * t[2]);
    out[2] = v[2] + q[3] * t[2] + (q[0] * t[1] - q[1] * t[0]);
}

// The cut of a resident hierarchy at a uniform object-space error: as select_cut, with a group fine when its error is
// at most error.
void uniform_cut(const Lod& lod, float error, std::vector<uint32_t>& cut) {
    cut.clear();
    auto fine = [&](uint32_t g) { return lod.groups[g].error != FLT_MAX && lod.groups[g].error <= error; };
    for (uint32_t i = 0; i < (uint32_t)lod.clusters.size(); ++i) {
        const LodCluster& c = lod.clusters[i];
        if (!fine(c.group) && (c.refined == LOD_NONE || fine(c.refined))) cut.push_back(i);
    }
}

// A base's largest axis scale: object-space errors grow by it.
float scale_of(const float base[12]) {
    float s = 0.0f;
    for (int a = 0; a < 3; ++a)
        s = std::max(s, std::sqrt(base[a] * base[a] + base[4 + a] * base[4 + a] + base[8 + a] * base[8 + a]));
    return s;
}

template <class T> void append(std::vector<uint32_t>& out, const T& v) {
    const uint32_t* w = reinterpret_cast<const uint32_t*>(&v);
    out.insert(out.end(), w, w + sizeof(T) / 4);
}
} // namespace

void InstanceSets::init(DeviceContext* ctx, ClusterGeometry* geometry, ID3D12DeviceRaytracing2* rtas,
                        uint32_t frameSlots) {
    m_ctx = ctx;
    m_geometry = geometry;
    m_rtas = rtas;
    ID3D12Device* device = ctx->Device();
    std::vector<CD3DX12_ROOT_PARAMETER1> rp(1 + SRVS + UAVS);
    rp[0].InitAsConstantBufferView(0, 0, D3D12_ROOT_DESCRIPTOR_FLAG_DATA_STATIC_WHILE_SET_AT_EXECUTE);
    for (uint32_t i = 0; i < SRVS; ++i) rp[1 + i].InitAsShaderResourceView(i, 0, D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE);
    for (uint32_t i = 0; i < UAVS; ++i) rp[1 + SRVS + i].InitAsUnorderedAccessView(i, 0, D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE);
    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC desc;
    desc.Init_1_1((UINT)rp.size(), rp.data());
    ComPtr<ID3DBlob> sig, err;
    if (FAILED(D3D12SerializeVersionedRootSignature(&desc, &sig, &err))) {
        if (err) OutputDebugStringA((const char*)err->GetBufferPointer());
        throw std::runtime_error("InstanceSets: root signature");
    }
    ThrowIfFailed(device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(), IID_PPV_ARGS(&m_root)));
    m_root->SetName(L"InstanceSetsRoot");
    auto pso = [&](const wchar_t* entry, ComPtr<ID3D12PipelineState>& out) {
        ComPtr<IDxcBlob> cs = nv_helpers_dx12::CompileCS(L"InstanceSets_v8.hlsl", entry);
        D3D12_COMPUTE_PIPELINE_STATE_DESC d = {};
        d.pRootSignature = m_root.Get();
        d.CS = { cs->GetBufferPointer(), cs->GetBufferSize() };
        ThrowIfFailed(device->CreateComputePipelineState(&d, IID_PPV_ARGS(&out)));
        out->SetName(entry);
    };
    pso(L"SetReset", m_reset);
    pso(L"SetCells", m_cells);
    pso(L"SetCollapse", m_collapse);
    pso(L"SetAlloc", m_alloc);
    pso(L"SetInstances", m_instances);
    pso(L"SetFinalize", m_finalize);
    m_slots.resize(std::max(1u, frameSlots));
}

ComPtr<ID3D12Resource> InstanceSets::buffer(uint64_t bytes, bool uav, const wchar_t* name) {
    auto b = nv_helpers_dx12::CreateBuffer(m_ctx->Device(), std::max<uint64_t>(align(bytes, 16), 256),
                                           uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE,
                                           uav ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_COPY_DEST,
                                           nv_helpers_dx12::kDefaultHeapProps);
    b->SetName(name);
    return b;
}

// A default-heap buffer holding data, copied on cl; its upload stays in keep until that list ran.
ComPtr<ID3D12Resource> InstanceSets::upload(const void* data, uint64_t bytes, bool uav, ID3D12GraphicsCommandList4* cl,
                                            std::vector<ComPtr<ID3D12Resource>>& keep, const wchar_t* name) {
    ComPtr<ID3D12Resource> dst = buffer(bytes, uav, name);
    if (!bytes) return dst;
    ComPtr<ID3D12Resource> src = nv_helpers_dx12::CreateBuffer(m_ctx->Device(), align(bytes, 16), D3D12_RESOURCE_FLAG_NONE,
                                                               D3D12_RESOURCE_STATE_GENERIC_READ,
                                                               nv_helpers_dx12::kUploadHeapProps);
    void* p = nullptr;
    const D3D12_RANGE noRead{ 0, 0 };
    ThrowIfFailed(src->Map(0, &noRead, &p));
    std::memcpy(p, data, bytes);
    src->Unmap(0, nullptr);
    using CD = CD3DX12_RESOURCE_BARRIER;
    if (uav) {
        const D3D12_RESOURCE_BARRIER b =
            CD::Transition(dst.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
        cl->ResourceBarrier(1, &b);
    }
    cl->CopyBufferRegion(dst.Get(), 0, src.Get(), 0, bytes);
    const D3D12_RESOURCE_BARRIER b = CD::Transition(dst.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                                    uav ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS
                                                        : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    cl->ResourceBarrier(1, &b);
    keep.push_back(src);
    return dst;
}

uint32_t InstanceSets::add_buckets(const Mesh& m, float standIn, ID3D12GraphicsCommandList4* cl,
                                   const std::function<void()>& flush, uint32_t& count, float& lastError,
                                   uint32_t& fine, uint32_t& coarse) {
    const Lod& lod = *m.lod;
    const float scale = scale_of(m.base);
    // Full detail, then cuts at doubling error up to the coarsest; a step that changes nothing is skipped.
    std::vector<float> errors;
    for (const LodGroup& g : lod.groups)
        if (g.error != FLT_MAX && g.error > 0.0f) errors.push_back(g.error);
    std::sort(errors.begin(), errors.end());
    std::vector<float> thresholds{ 0.0f };
    if (!errors.empty() && standIn <= 0.0f) {
        const float lo = errors.front(), hi = errors.back();
        const float ratio = std::max(2.0f, std::pow(hi / lo, 1.0f / (float)(MAX_BUCKETS - 2)));
        for (float e = lo;; e *= ratio) {
            thresholds.push_back(std::min(e, hi));
            if (e >= hi) break;
        }
    }
    std::vector<std::vector<uint64_t>> lists;
    std::vector<float> bucketError;
    std::vector<uint32_t> cut, last;
    for (float t : thresholds) {
        uniform_cut(lod, t, cut);
        if (!lists.empty() && cut == last) continue;
        last = cut;
        std::vector<uint64_t>& list = lists.emplace_back(lod.clas.begin(), lod.clas.begin() + lod.fixed);
        for (uint32_t c : cut) list.push_back(lod.clas[lod.fixed + c]);
        bucketError.push_back(standIn > 0.0f ? standIn : t * scale);
        if (std::find(list.begin(), list.end(), 0ull) != list.end())
            throw std::runtime_error("InstanceSets: a set's hierarchy is not resident in full");
    }

    std::vector<uint64_t> scratchOff(lists.size());
    uint64_t scratch = 0;
    const size_t blasFirst = m_blas.size();
    for (size_t k = 0; k < lists.size(); ++k) {
        uint64_t size = 0, sc = 0;
        m_geometry->blas_prebuild((uint32_t)lists[k].size(), m.omm, size, sc);
        m_blas.push_back(nv_helpers_dx12::CreateBuffer(m_ctx->Device(), std::max<uint64_t>(size, 256),
                                                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                                       D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
                                                       nv_helpers_dx12::kDefaultHeapProps));
        m_blas.back()->SetName(L"InstanceSetBucketBLAS");
        m_stats.blasBytes += m_blas.back()->GetDesc().Width;
        scratchOff[k] = scratch;
        scratch += align(sc, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT);
    }
    ComPtr<ID3D12Resource> scratchBuf = nv_helpers_dx12::CreateBuffer(
        m_ctx->Device(), std::max<uint64_t>(scratch, 256), D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nv_helpers_dx12::kDefaultHeapProps);
    const uint32_t slot = m_geometry->sync_slot();
    m_geometry->begin(slot, cl);
    for (size_t k = 0; k < lists.size(); ++k) {
        if (!m_geometry->can_queue_blas((uint32_t)lists[k].size())) {
            m_geometry->record(cl);
            flush();
            m_geometry->begin(slot, cl);
        }
        m_geometry->queue_blas_from(lists[k].data(), (uint32_t)lists[k].size(), (uint32_t)lists[k].size(), m.omm,
                                    m_blas[blasFirst + k]->GetGPUVirtualAddress(),
                                    scratchBuf->GetGPUVirtualAddress() + scratchOff[k]);
    }
    m_geometry->record(cl);
    flush();

    const uint32_t first = (uint32_t)(m_bucketData.size() / (sizeof(GpuBucket) / 4));
    for (size_t k = 0; k < lists.size(); ++k) {
        GpuBucket b = {};
        b.error = bucketError[k];
        b.record = m.record;
        b.hitGroup = m.hitGroup;
        b.flags = m.flags;
        b.blas = m_blas[blasFirst + k]->GetGPUVirtualAddress();
        std::memcpy(b.base, m.base, sizeof(b.base));
        append(m_bucketData, b);
    }
    count = (uint32_t)lists.size();
    lastError = bucketError.back();
    fine = (uint32_t)lists.front().size();
    coarse = (uint32_t)lists.back().size();
    m_stats.buckets += count;
    return first;
}

uint32_t InstanceSets::add(const Desc& d, std::vector<Instance> instances, ID3D12GraphicsCommandList4* cl,
                           const std::function<void()>& flush) {
    GpuSet g = {};
    g.mask = 0xFF;

    // The whole's buckets, the stand-in last; each part's own.
    float lastError = 0.0f;
    uint32_t fine = 0, coarse = 0, n = 0;
    g.bucketBase = add_buckets(d.whole, 0.0f, cl, flush, g.bucketCount, lastError, fine, coarse);
    if (d.proxy) {
        float e;
        uint32_t f, c;
        add_buckets(d.stand, 2.0f * std::max(lastError, 1e-6f), cl, flush, n, e, f, c);
        g.bucketCount += n;
    }
    if (d.parts.size() > MAX_PARTS) throw std::runtime_error("InstanceSets: too many parts in an assembly");
    g.partBase = (uint32_t)(m_partData.size() / (sizeof(GpuPart) / 4));
    g.partCount = (uint32_t)d.parts.size();
    uint32_t partClusters = 0;
    for (const Part& p : d.parts) {
        GpuPart gp = {};
        float e;
        uint32_t c;
        uint32_t f;
        gp.bucketBase = add_buckets(p.mesh, 0.0f, cl, flush, gp.bucketCount, e, f, c);
        partClusters += f;
        gp.windWeight = p.windWeight;
        std::memcpy(gp.pivot, p.pivot, sizeof(gp.pivot));
        append(m_partData, gp);
    }
    if (g.partCount) {
        g.blockBase = m_blockCount;
        g.blockCount = std::min<uint32_t>(ASSEMBLY_BLOCKS, (uint32_t)instances.size());
        g.slotBase = m_slotCount;
        g.partitionBase = m_poolPartitions;
        m_blockCount += g.blockCount;
        m_slotCount += g.blockCount * g.partCount;
        m_poolPartitions += (g.blockCount * g.partCount + CELL_INSTANCES - 1) / CELL_INSTANCES;
    }

    // Cells: the instances in Morton order of their positions, cut into runs.
    std::vector<float> spheres(instances.size() * 4);
    float lo[3] = { FLT_MAX, FLT_MAX, FLT_MAX }, hi[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
    for (Instance& in : instances) {
        const float len = std::sqrt(in.rotation[0] * in.rotation[0] + in.rotation[1] * in.rotation[1] +
                                    in.rotation[2] * in.rotation[2] + in.rotation[3] * in.rotation[3]);
        for (float& q : in.rotation) q = len > 0.0f ? q / len : 0.0f;
        if (len <= 0.0f) in.rotation[3] = 1.0f;
        for (int a = 0; a < 3; ++a) {
            lo[a] = std::min(lo[a], in.position[a]);
            hi[a] = std::max(hi[a], in.position[a]);
        }
    }
    std::vector<std::pair<uint64_t, uint32_t>> order(instances.size());
    for (size_t i = 0; i < instances.size(); ++i) {
        uint64_t key = 0;
        for (int a = 0; a < 3; ++a) {
            const float extent = std::max(hi[a] - lo[a], 1e-6f);
            const uint64_t q = (uint64_t)std::clamp((instances[i].position[a] - lo[a]) / extent * 2097151.0f, 0.0f,
                                                    2097151.0f);
            key |= spread3(q) << a;
        }
        order[i] = { key, (uint32_t)i };
    }
    std::sort(order.begin(), order.end());

    const uint32_t setIndex = m_setCount++;
    g.instanceBase = m_instanceCount;
    g.instanceCount = (uint32_t)instances.size();
    g.cellBase = m_cellCount;
    const size_t at = m_instanceData.size();
    m_instanceData.resize(at + instances.size() * sizeof(GpuInstance));
    for (size_t i = 0; i < order.size(); ++i) {
        const Instance& in = instances[order[i].second];
        GpuInstance gi;
        std::memcpy(gi.position, in.position, sizeof(gi.position));
        gi.scale = in.scale;
        gi.rotation[0] = snorm16(in.rotation[0]) | snorm16(in.rotation[1]) << 16;
        gi.rotation[1] = snorm16(in.rotation[2]) | snorm16(in.rotation[3]) << 16;
        std::memcpy(m_instanceData.data() + at + i * sizeof(GpuInstance), &gi, sizeof(gi));
        float c[3];
        const float scaled[3] = { d.center[0] * in.scale, d.center[1] * in.scale, d.center[2] * in.scale };
        rotate(in.rotation, scaled, c);
        for (int a = 0; a < 3; ++a) spheres[i * 4 + a] = in.position[a] + c[a];
        spheres[i * 4 + 3] = d.radius * in.scale;
    }
    for (size_t first = 0; first < order.size(); first += CELL_INSTANCES) {
        const size_t end = std::min(order.size(), first + CELL_INSTANCES);
        float bl[3] = { FLT_MAX, FLT_MAX, FLT_MAX }, bh[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
        GpuCell c = {};
        c.minScale = FLT_MAX;
        for (size_t i = first; i < end; ++i) {
            for (int a = 0; a < 3; ++a) {
                bl[a] = std::min(bl[a], spheres[i * 4 + a] - spheres[i * 4 + 3]);
                bh[a] = std::max(bh[a], spheres[i * 4 + a] + spheres[i * 4 + 3]);
            }
            const float s = instances[order[i].second].scale;
            c.minScale = std::min(c.minScale, s);
            c.maxScale = std::max(c.maxScale, s);
        }
        for (int a = 0; a < 3; ++a) c.center[a] = 0.5f * (bl[a] + bh[a]);
        c.radius = 0.5f * std::sqrt((bh[0] - bl[0]) * (bh[0] - bl[0]) + (bh[1] - bl[1]) * (bh[1] - bl[1]) +
                                    (bh[2] - bl[2]) * (bh[2] - bl[2]));
        c.first = m_instanceCount + (uint32_t)first;
        c.count = (uint32_t)(end - first);
        c.set = setIndex;
        append(m_cellData, c);
        ++m_cellCount;
    }
    g.cellCount = m_cellCount - g.cellBase;
    m_instanceCount += g.instanceCount;
    std::memcpy(g.center, d.center, sizeof(g.center));
    g.radius = d.radius;
    append(m_setData, g);

    m_stats.sets = m_setCount;
    m_stats.cells = m_cellCount;
    m_stats.instances = m_instanceCount;
    m_stats.slots = m_slotCount;
    std::printf("[sets] set %u: %u instances in %u cells; whole in %u buckets (%u to %u clusters, error up to %.3g)%s; "
                "%u parts (%u clusters)\n",
                setIndex, g.instanceCount, g.cellCount, g.bucketCount - n, fine, coarse, (double)lastError,
                d.proxy ? ", then the stand-in" : "", g.partCount, partClusters);
    return setIndex;
}

void InstanceSets::commit(planet::TlasBuilder& tlas, const float origin[3], ID3D12GraphicsCommandList4* cl,
                          const std::function<void()>& flush) {
    if (!m_cellCount) return;
    std::memcpy(m_origin, origin, sizeof(m_origin));
    std::vector<ComPtr<ID3D12Resource>> keep;
    m_setBuf = upload(m_setData.data(), m_setData.size() * 4, false, cl, keep, L"InstanceSets");
    m_bucketBuf = upload(m_bucketData.data(), m_bucketData.size() * 4, false, cl, keep, L"InstanceSetBuckets");
    m_partBuf = upload(m_partData.data(), m_partData.size() * 4, false, cl, keep, L"InstanceSetParts");
    m_cellBuf = upload(m_cellData.data(), m_cellData.size() * 4, false, cl, keep, L"InstanceSetCells");
    m_instanceBuf = upload(m_instanceData.data(), m_instanceData.size(), false, cl, keep, L"InstanceSetInstances");
    m_instanceState = buffer(4ull * m_instanceCount, true, L"InstanceSetInstanceState");
    m_cellState = buffer(8ull * m_cellCount, true, L"InstanceSetCellState");
    // pool: per set its free blocks' count, then the free lists, then each part slot's bucket.
    m_freeLists = (uint32_t)align(4ull * m_setCount, 16);
    m_partStates = (uint32_t)align(m_freeLists + 4ull * m_blockCount, 16);
    m_pool = buffer(m_partStates + 4ull * m_slotCount, true, L"InstanceSetPool");

    m_maxWrites = std::min(MAX_WRITES, m_instanceCount);
    m_maxUpdates = std::min(MAX_UPDATES, m_instanceCount + m_slotCount);
    m_maxBlockWrites = m_slotCount ? MAX_BLOCK_WRITES : 0u;
    m_writeArgs = WORK_ARGS;
    m_updateArgs = (uint32_t)align(m_writeArgs + (uint64_t)WRITE_ARG_BYTES * m_maxWrites, 16);
    m_translateArgs = (uint32_t)align(m_updateArgs + (uint64_t)UPDATE_ARG_BYTES * m_maxUpdates, 16);
    m_blockArgs = (uint32_t)align(m_translateArgs + (uint64_t)TRANSLATE_ARG_BYTES * (m_cellCount + m_poolPartitions), 16);
    m_work = buffer(m_blockArgs + (uint64_t)WRITE_ARG_BYTES * m_maxBlockWrites, true, L"InstanceSetWork");
    for (Slot& s : m_slots) {
        s.constants = nv_helpers_dx12::CreateBuffer(m_ctx->Device(), 256, D3D12_RESOURCE_FLAG_NONE,
                                                    D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
        const D3D12_RANGE noRead{ 0, 0 };
        ThrowIfFailed(s.constants->Map(0, &noRead, reinterpret_cast<void**>(&s.constantsMapped)));
        s.readback = nv_helpers_dx12::CreateBuffer(m_ctx->Device(), 256, D3D12_RESOURCE_FLAG_NONE,
                                                   D3D12_RESOURCE_STATE_COPY_DEST, nv_helpers_dx12::kReadbackHeapProps);
        void* p = nullptr;
        ThrowIfFailed(s.readback->Map(0, nullptr, &p));
        s.readbackMapped = static_cast<const uint32_t*>(p);
        s.pending = s.recorded = false;
    }
    flush();
    m_instanceData.clear();
    m_instanceData.shrink_to_fit();
    m_cellData.clear();
    m_cellData.shrink_to_fit();

    // Their own instances and partitions after the CPU's: the instances, then the part slots; the cells, then the
    // slots' partitions. A new structure, so every instance is written again.
    tlas.reserve_external(m_instanceCount + m_slotCount, m_cellCount + m_poolPartitions);
    m_tlasGeneration = ~0u;
    m_stats.gpuBytes = m_instanceBuf->GetDesc().Width + m_cellBuf->GetDesc().Width + m_instanceState->GetDesc().Width +
                       m_cellState->GetDesc().Width + m_work->GetDesc().Width + m_pool->GetDesc().Width;
    std::printf("[sets] %u sets, %llu instances in %u cells, %u part slots; %.1f MB of tables, %.1f MB of bucket BLAS\n",
                m_setCount, (unsigned long long)m_instanceCount, m_cellCount, m_slotCount,
                (double)m_stats.gpuBytes / 1048576.0, (double)m_stats.blasBytes / 1048576.0);
}

void InstanceSets::collect() {
    if (!m_cellCount) return;
    const uint64_t done = m_ctx->PlanetComputeCompleted();
    for (Slot& s : m_slots) {
        if (!s.pending || done < s.fence) continue;
        s.pending = false;
        const uint32_t* f = s.readbackMapped;
        m_stats.writes = f[0];
        m_stats.updates = f[1];
        m_stats.activeCells = f[2];
        m_stats.written = f[3];
        m_stats.blockWrites = f[4];
        m_stats.expanded = f[5];
        m_stats.valid = true;
    }
}

void InstanceSets::record(ID3D12GraphicsCommandList4* cl, uint32_t frameSlot, planet::TlasBuilder& tlas,
                          const View& view, float offscreenScale, const Wind& wind) {
    if (!m_cellCount || tlas.external_instances() < m_instanceCount + m_slotCount) return;
    m_slot = frameSlot % (uint32_t)m_slots.size();
    Slot& slot = m_slots[m_slot];
    const D3D12_GPU_VIRTUAL_ADDRESS workVa = m_work->GetGPUVirtualAddress();
    const bool fresh = m_tlasGeneration != tlas.generation();

    // The sets keep the frame of the origin they were made at; the TLAS moves their partitions to the present one.
    const float delta[3] = { m_origin[0] - view.origin[0], m_origin[1] - view.origin[1], m_origin[2] - view.origin[2] };
    if (fresh) std::memset(m_translated, 0, sizeof(m_translated)); // a new TLAS translates nothing
    Constants k = {};
    for (int a = 0; a < 3; ++a) k.camera[a] = view.camera[a] - delta[a];
    k.pixelsPerUnit = view.pixelsPerUnit;
    k.threshold = view.threshold;
    k.offscreen = offscreenScale;
    k.znear = view.znear;
    k.cells = m_cellCount;
    k.instances = m_instanceCount;
    k.maxWrites = m_maxWrites;
    k.maxUpdates = m_maxUpdates;
    k.tlasBase = tlas.external_instance_base();
    k.partitionBase = tlas.external_partition_base();
    k.writeArgs = m_writeArgs;
    k.updateArgs = m_updateArgs;
    k.workLo = (uint32_t)workVa;
    k.workHi = (uint32_t)(workVa >> 32);
    k.translate = std::memcmp(delta, m_translated, sizeof(delta)) != 0 ? 1u : 0u;
    k.translateArgs = m_translateArgs;
    k.slots = m_slotCount;
    std::memcpy(k.delta, delta, sizeof(delta));
    std::memcpy(m_translated, delta, sizeof(delta));
    k.time = view.time;
    k.blockArgs = m_blockArgs;
    k.maxBlockWrites = m_maxBlockWrites;
    k.poolTlasBase = k.tlasBase + m_instanceCount;
    k.poolPartitionBase = k.partitionBase + m_cellCount;
    k.windDistance = wind.distance;
    k.windAmplitude = wind.amplitude;
    k.windFrequency = wind.frequency;
    k.blocks = m_blockCount;
    k.sets = m_setCount;
    k.freeLists = m_freeLists;
    k.partStates = m_partStates;
    k.poolPartitions = m_poolPartitions;
    for (int i = 0; i < 5; ++i) {
        std::memcpy(k.planes[i], view.planes[i], sizeof(k.planes[i]));
        k.planes[i][3] += view.planes[i][0] * delta[0] + view.planes[i][1] * delta[1] + view.planes[i][2] * delta[2];
    }
    // This slot's last frame has finished: its constants may change.
    std::memcpy(slot.constantsMapped, &k, sizeof(k));

    cl->SetComputeRootSignature(m_root.Get());
    cl->SetComputeRootConstantBufferView(0, slot.constants->GetGPUVirtualAddress());
    const D3D12_GPU_VIRTUAL_ADDRESS srv[SRVS] = { m_setBuf->GetGPUVirtualAddress(), m_bucketBuf->GetGPUVirtualAddress(),
                                                  m_cellBuf->GetGPUVirtualAddress(),
                                                  m_instanceBuf->GetGPUVirtualAddress(),
                                                  m_partBuf->GetGPUVirtualAddress() };
    for (uint32_t i = 0; i < SRVS; ++i) cl->SetComputeRootShaderResourceView(1 + i, srv[i]);
    ID3D12Resource* uav[UAVS] = { m_instanceState.Get(), m_cellState.Get(), m_work.Get(), m_pool.Get() };
    for (uint32_t i = 0; i < UAVS; ++i) cl->SetComputeRootUnorderedAccessView(1 + SRVS + i, uav[i]->GetGPUVirtualAddress());

    const D3D12_RESOURCE_BARRIER uavAll = CD3DX12_RESOURCE_BARRIER::UAV(nullptr);
    auto pass = [&](ID3D12PipelineState* pso, uint32_t x, uint32_t y) {
        cl->SetPipelineState(pso);
        cl->Dispatch(x, y, 1);
        cl->ResourceBarrier(1, &uavAll);
    };
    // A new TLAS holds none of them: all are written again.
    if (fresh) {
        m_tlasGeneration = tlas.generation();
        pass(m_reset.Get(), (std::max({ m_instanceCount, m_cellCount, m_slotCount, m_blockCount, m_setCount }) + 63) / 64,
             1);
    }
    const uint32_t perCell = CELL_INSTANCES / GROUP_INSTANCES;
    pass(m_cells.Get(), (m_cellCount + m_poolPartitions + 63) / 64, 1);
    // Expansions take pool blocks before collapses give theirs back: a slot is never disabled and written again in one
    // build, whose operations may apply in any order.
    if (m_slotCount) {
        pass(m_alloc.Get(), perCell, m_cellCount);
        pass(m_collapse.Get(), perCell, m_cellCount);
    }
    pass(m_instances.Get(), perCell, m_cellCount);
    cl->SetPipelineState(m_finalize.Get());
    cl->Dispatch(1, 1, 1);

    using CD = CD3DX12_RESOURCE_BARRIER;
    constexpr D3D12_RESOURCE_STATES readState = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE |
                                                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT |
                                                D3D12_RESOURCE_STATE_COPY_SOURCE;
    const D3D12_RESOURCE_BARRIER toRead[2] = {
        uavAll, CD::Transition(m_work.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, readState) };
    cl->ResourceBarrier(2, toRead);
    tlas.build_gpu_ops(cl, workVa + WORK_OPS, workVa + WORK_OP_COUNT);
    cl->CopyBufferRegion(slot.readback.Get(), 0, m_work.Get(), WORK_FEEDBACK, FEEDBACK_BYTES);
    const D3D12_RESOURCE_BARRIER back = CD::Transition(m_work.Get(), readState, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    cl->ResourceBarrier(1, &back);
    slot.recorded = true;
}

void InstanceSets::on_submitted(uint64_t computeFence) {
    if (!m_cellCount) return;
    Slot& s = m_slots[m_slot];
    if (!s.recorded) return;
    s.recorded = false;
    s.fence = computeFence;
    s.pending = true;
}
#endif

} // namespace clusters
