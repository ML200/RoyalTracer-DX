#include "LodGpu.h"
#include "../Core/DeviceContext.h"
#include "../DXRHelper.h"
#include "../planet/tlas_builder.h"
#include <algorithm>
#include <cfloat>
#include <cstring>

namespace clusters {

#if RDN_CLUSTER_API
namespace {
// Layouts of Lod_v8.hlsl.
constexpr uint32_t RING = 4, BASE = ~0u, FEEDBACK_HEADER = 32;
struct GpuMesh {
    uint32_t groupBase, groupCount, clusterBase, clusterCount, clasBase, fixedCount, rootCount, pagerMesh;
    uint32_t baseBlas[2], pad[2];
};
struct GpuInstance {
    uint32_t mesh, cutAddr, cutIdx, tlasIndex, hitGroup, cls, groupWork, clusterWork;
};
struct GpuGroup {
    float center[3], radius, error;
    uint32_t depFirst, depCount, mesh;
};
struct GpuState {
    uint32_t current, retired[RING], hashA, hashB, count, half, currentCount, pad[2];
};
struct GpuUpdate {
    uint32_t index, offset, value, pad;
};
struct Constants {
    uint32_t frame, instances, groupWork, clusterWork;
    float pixelsPerUnit, threshold;
    uint32_t groups, maxRequests, classes, updates, classArgs, classDests, classCounts, classBudgets, tlasArgs, workLo,
        workHi;
};
static_assert(sizeof(GpuMesh) == 48 && sizeof(GpuInstance) == 32 && sizeof(GpuGroup) == 32 && sizeof(GpuState) == 48);
constexpr uint32_t SRVS = 12, UAVS = 11, CONSTANTS = sizeof(Constants) / 4;
// The CLAS references one frame's builds of a class may reach.
constexpr uint32_t CLASS_BUDGET = 1u << 20;

inline uint64_t align(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

D3D12_RTAS_CLAS_INPUTS_DESC class_inputs(uint32_t capacity, uint32_t budget, uint32_t maxPerArg) {
    D3D12_RTAS_CLAS_INPUTS_DESC d = {};
    // Micromap-aware whatever the meshes: an alpha page's CLAS need it, and it costs the others little.
    d.Flags = D3D12_RTAS_OPERATION_FLAG_FAST_TRACE | D3D12_RTAS_OPERATION_FLAG_ALLOW_OMM;
    d.MaxArgCount = capacity;
    d.Mode = D3D12_RTAS_OPERATION_MODE_EXPLICIT_DESTINATIONS;
    d.MaxTotalClasCount = budget;
    d.MaxClasCountPerArg = maxPerArg;
    return d;
}
} // namespace

void LodGpu::init(DeviceContext* ctx, ID3D12DeviceRaytracing2* rtas, uint32_t frameSlots) {
    m_ctx = ctx;
    m_rtas = rtas;
    ID3D12Device* device = ctx->Device();
    std::vector<CD3DX12_ROOT_PARAMETER1> rp(1 + SRVS + UAVS);
    rp[0].InitAsConstants(CONSTANTS, 0);
    for (uint32_t i = 0; i < SRVS; ++i) rp[1 + i].InitAsShaderResourceView(i, 0, D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE);
    for (uint32_t i = 0; i < UAVS; ++i) rp[1 + SRVS + i].InitAsUnorderedAccessView(i, 0, D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE);
    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC desc;
    desc.Init_1_1((UINT)rp.size(), rp.data());
    ComPtr<ID3DBlob> sig, err;
    if (FAILED(D3D12SerializeVersionedRootSignature(&desc, &sig, &err))) {
        if (err) OutputDebugStringA((const char*)err->GetBufferPointer());
        throw std::runtime_error("LodGpu: root signature");
    }
    ThrowIfFailed(device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(), IID_PPV_ARGS(&m_root)));
    m_root->SetName(L"LodGpuRoot");
    auto pso = [&](const wchar_t* entry, ComPtr<ID3D12PipelineState>& out) {
        ComPtr<IDxcBlob> cs = nv_helpers_dx12::CompileCS(L"Lod_v8.hlsl", entry);
        D3D12_COMPUTE_PIPELINE_STATE_DESC d = {};
        d.pRootSignature = m_root.Get();
        d.CS = { cs->GetBufferPointer(), cs->GetBufferSize() };
        ThrowIfFailed(device->CreateComputePipelineState(&d, IID_PPV_ARGS(&out)));
        out->SetName(entry);
    };
    pso(L"LodApply", m_apply);
    pso(L"LodClear", m_clear);
    pso(L"LodGroups", m_groups);
    pso(L"LodClusters", m_clusters);
    pso(L"LodPlan", m_plan);
    pso(L"LodMark", m_mark);
    pso(L"LodFinalize", m_finalize);
    m_slots.resize(std::max(1u, frameSlots));
}

ComPtr<ID3D12Resource> LodGpu::buffer(uint64_t bytes, bool uav, const wchar_t* name) {
    auto b = nv_helpers_dx12::CreateBuffer(m_ctx->Device(), std::max<uint64_t>(align(bytes, 16), 256),
                                           uav ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE,
                                           uav ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_COPY_DEST,
                                           nv_helpers_dx12::kDefaultHeapProps);
    b->SetName(name);
    return b;
}

// A default-heap buffer holding data, copied on cl; its upload stays until that list ran.
template <class T>
ComPtr<ID3D12Resource> LodGpu::upload_static(const std::vector<T>& data, ID3D12GraphicsCommandList4* cl,
                                             const wchar_t* name, bool uav) {
    const uint64_t bytes = std::max<uint64_t>(data.size() * sizeof(T), 16);
    ComPtr<ID3D12Resource> dst = buffer(bytes, uav, name);
    ComPtr<ID3D12Resource> src = nv_helpers_dx12::CreateBuffer(m_ctx->Device(), align(bytes, 16), D3D12_RESOURCE_FLAG_NONE,
                                                               D3D12_RESOURCE_STATE_GENERIC_READ,
                                                               nv_helpers_dx12::kUploadHeapProps);
    void* p = nullptr;
    D3D12_RANGE noRead{ 0, 0 };
    ThrowIfFailed(src->Map(0, &noRead, &p));
    if (!data.empty()) std::memcpy(p, data.data(), data.size() * sizeof(T));
    src->Unmap(0, nullptr);
    const D3D12_RESOURCE_STATES state = uav ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_COPY_DEST;
    if (uav) {
        const D3D12_RESOURCE_BARRIER b = CD3DX12_RESOURCE_BARRIER::Transition(dst.Get(), state, D3D12_RESOURCE_STATE_COPY_DEST);
        cl->ResourceBarrier(1, &b);
    }
    if (!data.empty()) cl->CopyBufferRegion(dst.Get(), 0, src.Get(), 0, data.size() * sizeof(T));
    const D3D12_RESOURCE_BARRIER b = CD3DX12_RESOURCE_BARRIER::Transition(
        dst.Get(), D3D12_RESOURCE_STATE_COPY_DEST, uav ? state : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    cl->ResourceBarrier(1, &b);
    m_keep.push_back(src);
    return dst;
}

void LodGpu::reset(const std::vector<Instance>& instances, ID3D12GraphicsCommandList4* cl, uint64_t frame) {
    // Frames in flight still trace the last set's BLAS and read its tables; every instance goes back to its mesh's BLAS.
    auto keep = [&](ComPtr<ID3D12Resource>& r) {
        if (r) m_keep.push_back(std::move(r));
    };
    keep(m_meshBuf); keep(m_instanceBuf); keep(m_groupBuf); keep(m_clusterBuf); keep(m_depBuf); keep(m_groupWorkBuf);
    keep(m_clusterWorkBuf); keep(m_ringAddrBuf); keep(m_classInfoBuf); keep(m_state); keep(m_finePrime); keep(m_cutIdx);
    keep(m_counters); keep(m_work); keep(m_feedback); keep(m_lastUsed); keep(m_want); keep(m_requested);
    keep(m_residency); keep(m_clasAddr); keep(m_ring); keep(m_scratch);
    for (Slot& s : m_slots) keep(s.readback);
    m_keepPending = true;
    m_rewriteAll = m_instances != 0;
    m_instances = 0;
    m_meshes.clear();
    m_meshOf.clear();
    m_instanceMesh.clear();

    // Meshes, their groups and clusters, in global ranges.
    std::vector<GpuMesh> meshes;
    std::vector<GpuGroup> groups;
    std::vector<uint32_t> deps, rootCount;
    std::vector<uint32_t> clustersFlat; // pairs
    std::vector<uint32_t> residency;
    std::vector<uint64_t> clas;
    std::vector<uint32_t> listOf; // instances' entries in `instances`
    for (size_t i = 0; i < instances.size(); ++i) {
        Lod* lod = instances[i].lod;
        if (!lod || !lod->paged) continue;
        listOf.push_back((uint32_t)i);
        const auto [it, fresh] = m_meshOf.emplace(lod, (uint32_t)m_meshes.size());
        if (!fresh) continue;
        MeshEntry e{ lod, (uint32_t)groups.size(), (uint32_t)(clustersFlat.size() / 2), (uint32_t)clas.size() };
        m_meshes.push_back(e);
        GpuMesh gm = {};
        gm.groupBase = e.groupBase;
        gm.groupCount = (uint32_t)lod->groups.size();
        gm.clusterBase = e.clusterBase;
        gm.clusterCount = (uint32_t)lod->clusters.size();
        gm.clasBase = e.clasBase;
        gm.fixedCount = lod->fixed;
        gm.rootCount = lod->base - lod->fixed;
        gm.pagerMesh = lod->pager;
        gm.baseBlas[0] = (uint32_t)instances[i].baseBlas;
        gm.baseBlas[1] = (uint32_t)(instances[i].baseBlas >> 32);
        meshes.push_back(gm);
        for (uint32_t g = 0; g < gm.groupCount; ++g) {
            const LodGroup& lg = lod->groups[g];
            GpuGroup gg = {};
            std::memcpy(gg.center, lg.center, sizeof(gg.center));
            gg.radius = lg.radius;
            gg.error = lg.error;
            gg.depFirst = (uint32_t)deps.size();
            gg.depCount = lod->depFirst[g + 1] - lod->depFirst[g];
            gg.mesh = (uint32_t)m_meshes.size() - 1;
            for (uint32_t d = lod->depFirst[g]; d < lod->depFirst[g + 1]; ++d) deps.push_back(e.groupBase + lod->deps[d]);
            groups.push_back(gg);
            residency.push_back(lod->resident[g]);
        }
        for (const LodCluster& c : lod->clusters) {
            clustersFlat.push_back(e.groupBase + c.group);
            clustersFlat.push_back(c.refined == LOD_NONE ? LOD_NONE : e.groupBase + c.refined);
        }
        clas.insert(clas.end(), lod->clas.begin(), lod->clas.end());
    }
    m_instances = (uint32_t)listOf.size();
    if (!m_instances) return;
    m_groupCount = (uint32_t)groups.size();

    // Instances: their size class, address list in work, cut halves, work item ranges.
    std::vector<GpuInstance> inst(m_instances);
    std::vector<uint32_t> groupWork(m_instances + 1), clusterWork(m_instances + 1);
    std::vector<uint32_t> classOf(m_instances);
    std::vector<uint32_t> classOfSize(32, ~0u);
    m_classes.clear();
    m_tlasToInstance.clear();
    m_baseClusters = 0;
    m_last = {};
    uint32_t cutAddr = 0, cutIdx = 0;
    m_groupWork = m_clusterWork = 0;
    for (uint32_t k = 0; k < m_instances; ++k) {
        const Instance& src = instances[listOf[k]];
        const uint32_t mi = m_meshOf[src.lod];
        const GpuMesh& gm = meshes[mi];
        const uint32_t refs = gm.fixedCount + gm.clusterCount;
        m_baseClusters += gm.fixedCount + gm.rootCount;
        uint32_t bits = 6;
        while ((1u << bits) < refs) ++bits;
        if (classOfSize[bits] == ~0u) {
            classOfSize[bits] = (uint32_t)m_classes.size();
            m_classes.push_back(Class{ 1u << bits });
        }
        Class& c = m_classes[classOfSize[bits]];
        ++c.capacity;
        c.budget = (uint32_t)std::min<uint64_t>((uint64_t)c.budget + refs, CLASS_BUDGET);
        GpuInstance& g = inst[k];
        g.mesh = mi;
        g.cutAddr = cutAddr;
        g.cutIdx = cutIdx;
        g.tlasIndex = src.tlasIndex;
        g.hitGroup = src.hitGroup;
        g.cls = classOfSize[bits];
        g.groupWork = m_groupWork;
        g.clusterWork = m_clusterWork;
        groupWork[k] = m_groupWork;
        clusterWork[k] = m_clusterWork;
        cutAddr += (uint32_t)align(8ull * refs, 16);
        cutIdx += 2 * gm.clusterCount;
        m_groupWork += gm.groupCount;
        m_clusterWork += gm.clusterCount;
        if (m_tlasToInstance.size() <= src.tlasIndex) m_tlasToInstance.resize(src.tlasIndex + 1, ~0u);
        m_tlasToInstance[src.tlasIndex] = k;
        m_instanceMesh.push_back(mi);
    }
    groupWork[m_instances] = m_groupWork;
    clusterWork[m_instances] = m_clusterWork;
    m_classCount = (uint32_t)m_classes.size();

    // Per class: the BLAS size of one ring entry and the build's scratch.
    std::vector<uint64_t> entrySize(m_classCount);
    uint64_t scratch = 0;
    uint32_t argBase = 0;
    std::vector<uint32_t> classInfo(2 * m_classCount);
    for (uint32_t c = 0; c < m_classCount; ++c) {
        Class& k = m_classes[c];
        const D3D12_RTAS_CLAS_INPUTS_DESC in = class_inputs(k.capacity, k.budget, k.maxPerArg);
        D3D12_RTAS_OPERATION_INPUTS ops = {};
        ops.Type = D3D12_RTAS_OPERATION_TYPE_BUILD_BLAS_FROM_CLAS;
        ops.pClasDesc = &in;
        D3D12_RTAS_OPERATION_PREBUILD_INFO info = {};
        m_rtas->GetRTASOperationPrebuildInfo(&ops, &info);
        entrySize[c] = align(info.ResultDataMaxSizeInBytes, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT);
        k.scratchOff = scratch;
        scratch += align(info.ScratchDataSizeInBytes, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT);
        k.argBase = argBase;
        argBase += k.capacity;
        classInfo[c] = k.budget;
        classInfo[m_classCount + c] = k.argBase;
    }

    // Each instance's ring of cluster BLAS.
    m_ringBytes = 0;
    for (uint32_t k = 0; k < m_instances; ++k) m_ringBytes += RING * entrySize[inst[k].cls];
    // Acceleration structure memory starts in, and never leaves, its state.
    m_ring = nv_helpers_dx12::CreateBuffer(m_ctx->Device(), std::max<uint64_t>(m_ringBytes, 256),
                                           D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                           D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
                                           nv_helpers_dx12::kDefaultHeapProps);
    m_ring->SetName(L"LodGpuBlasRings");
    m_scratch = buffer(scratch, true, L"LodGpuBlasScratch");
    std::vector<uint64_t> ringAddr(m_instances * RING);
    uint64_t at = m_ring->GetGPUVirtualAddress();
    for (uint32_t k = 0; k < m_instances; ++k)
        for (uint32_t e = 0; e < RING; ++e, at += entrySize[inst[k].cls]) ringAddr[k * RING + e] = at;

    // Work: address lists (fixed clusters first, set here once), class arguments and destinations, their counts and
    // budget use, the TLAS update arguments with their count and operation.
    m_classArgs = (uint32_t)align(cutAddr, 16);
    m_classDests = (uint32_t)align(m_classArgs + 16ull * argBase, 16);
    m_classCounts = (uint32_t)align(m_classDests + 8ull * argBase, 16);
    m_classBudgets = (uint32_t)align(m_classCounts + 4ull * m_classCount, 16);
    m_tlasArgs = (uint32_t)align(m_classBudgets + 4ull * m_classCount, 16);
    const uint32_t workBytes = (uint32_t)align(m_tlasArgs + 16ull * m_instances + 16 + 48, 16);
    std::vector<uint8_t> work(workBytes, 0);
    for (uint32_t k = 0; k < m_instances; ++k) {
        const MeshEntry& me = m_meshes[inst[k].mesh];
        std::memcpy(work.data() + inst[k].cutAddr, me.lod->clas.data(), 8ull * me.lod->fixed);
    }

    std::vector<GpuState> state(m_instances);
    for (GpuState& s : state) {
        s = {};
        s.current = BASE;
    }

    m_meshBuf = upload_static(meshes, cl, L"LodGpuMeshes");
    m_instanceBuf = upload_static(inst, cl, L"LodGpuInstances");
    m_groupBuf = upload_static(groups, cl, L"LodGpuGroups");
    m_clusterBuf = upload_static(clustersFlat, cl, L"LodGpuClusters");
    m_depBuf = upload_static(deps, cl, L"LodGpuDeps");
    m_groupWorkBuf = upload_static(groupWork, cl, L"LodGpuGroupWork");
    m_clusterWorkBuf = upload_static(clusterWork, cl, L"LodGpuClusterWork");
    m_ringAddrBuf = upload_static(ringAddr, cl, L"LodGpuRingAddresses");
    m_classInfoBuf = upload_static(classInfo, cl, L"LodGpuClasses");
    m_state = upload_static(state, cl, L"LodGpuState", true);
    m_work = upload_static(work, cl, L"LodGpuWork", true);
    m_residency = upload_static(residency, cl, L"LodGpuResidency", true);
    m_clasAddr = upload_static(clas, cl, L"LodGpuClasAddresses", true);
    m_finePrime = buffer(4ull * m_groupWork, true, L"LodGpuFine");
    m_cutIdx = buffer(4ull * cutIdx, true, L"LodGpuCuts");
    m_counters = buffer(16ull * m_instances, true, L"LodGpuCounters");
    m_feedback = buffer(FEEDBACK_HEADER + 16ull * m_maxRequests, true, L"LodGpuFeedback");
    m_lastUsed = upload_static(std::vector<uint32_t>(m_groupCount, (uint32_t)frame), cl, L"LodGpuLastUsed", true);
    m_want = buffer(4ull * m_groupCount, true, L"LodGpuWant");
    m_requested = upload_static(std::vector<uint32_t>(m_groupCount, 0u), cl, L"LodGpuRequested", true);

    m_readbackBytes = FEEDBACK_HEADER + 16ull * m_maxRequests + 8ull * m_groupCount;
    for (Slot& s : m_slots) {
        s.readback = nv_helpers_dx12::CreateBuffer(m_ctx->Device(), m_readbackBytes, D3D12_RESOURCE_FLAG_NONE,
                                                   D3D12_RESOURCE_STATE_COPY_DEST, nv_helpers_dx12::kReadbackHeapProps);
        s.readback->SetName(L"LodGpuReadback");
        void* p = nullptr;
        ThrowIfFailed(s.readback->Map(0, nullptr, &p));
        s.readbackMapped = static_cast<const uint8_t*>(p);
        s.pending = s.recorded = false;
    }
    std::printf("[lod-gpu] %u instances of %zu meshes, %u groups, %u cluster work items, %u size classes; BLAS rings "
                "%.1f MB, scratch %.1f MB\n",
                m_instances, m_meshes.size(), m_groupCount, m_clusterWork, m_classCount, (double)m_ringBytes / 1048576.0,
                (double)scratch / 1048576.0);
}

void LodGpu::collect(Feedback& out) {
    const uint64_t done = m_ctx->PlanetComputeCompleted();
    if (!m_keepPending && done >= m_keepFence) m_keep.clear();
    if (!m_instances) return;
    for (Slot& s : m_slots) {
        if (!s.pending || done < s.fence) continue;
        s.pending = false;
        const uint32_t* h = reinterpret_cast<const uint32_t*>(s.readbackMapped);
        const uint32_t requests = std::min(h[0], m_maxRequests);
        for (uint32_t i = 0; i < requests; ++i) {
            const uint32_t* r = h + FEEDBACK_HEADER / 4 + 4 * i;
            float priority;
            std::memcpy(&priority, &r[2], 4);
            out.requests.push_back(LodRequest{ r[0], r[1], priority });
        }
        out.builds = m_last.builds = h[1];
        out.buildRefs = m_last.buildRefs = h[2];
        out.tracedClusters = m_last.tracedClusters = h[3];
        out.tlasUpdates = m_last.tlasUpdates = h[4];
        out.valid = m_last.valid = true;
        // When traced cuts last used each group, and how far over budget it is.
        const uint32_t* lastUsed = h + (FEEDBACK_HEADER + 16ull * m_maxRequests) / 4;
        const float* want = reinterpret_cast<const float*>(lastUsed + m_groupCount);
        for (const MeshEntry& me : m_meshes) {
            Lod& lod = *me.lod;
            for (uint32_t g = 0; g < (uint32_t)lod.groups.size(); ++g) {
                lod.lastUsed[g] = std::max<uint64_t>(lod.lastUsed[g], lastUsed[me.groupBase + g]);
                lod.want[g] = want[me.groupBase + g];
            }
        }
    }
}

void LodGpu::record(ID3D12GraphicsCommandList4* cl, uint32_t frameSlot, planet::TlasBuilder& tlas,
                    const std::vector<float>& views, const std::vector<ResidencyChange>& changes, uint64_t frame,
                    float pixelsPerUnit, float threshold) {
    if (!m_instances) return;
    m_slot = frameSlot % (uint32_t)m_slots.size();
    Slot& slot = m_slots[m_slot];

    // This frame's inputs: views, instances the CPU wrote into the TLAS, residency changes. The GPU applies the table
    // updates in parallel, so a group changed twice keeps only its last one.
    std::vector<GpuUpdate> updates;
    std::unordered_map<uint64_t, size_t> updateOf;
    auto update = [&](uint32_t index, uint32_t offset, uint32_t value) {
        const auto [it, fresh] = updateOf.emplace((uint64_t)index << 2 | offset, updates.size());
        if (fresh) updates.push_back(GpuUpdate{ index, offset, value, 0u });
        else updates[it->second].value = value;
    };
    for (const ResidencyChange& c : changes) {
        const auto it = m_meshOf.find(c.lod);
        if (it == m_meshOf.end()) continue;
        const MeshEntry& me = m_meshes[it->second];
        update(me.groupBase + c.group, 0u, c.resident ? 1u : 0u);
        if (!c.resident) continue;
        for (uint32_t k = 0; k < c.clusterCount; ++k) {
            const uint32_t at = c.lod->fixed + c.firstCluster + k;
            const uint64_t va = c.lod->clas[at];
            update(me.clasBase + at, 1u, (uint32_t)va);
            update(me.clasBase + at, 2u, (uint32_t)(va >> 32));
        }
    }
    std::vector<uint32_t> rewritten(m_instances, m_rewriteAll ? 1u : 0u);
    m_rewriteAll = false;
    for (uint32_t id : tlas.written())
        if (id < m_tlasToInstance.size() && m_tlasToInstance[id] != ~0u) rewritten[m_tlasToInstance[id]] = 1u;
    const uint64_t viewBytes = 16ull * m_instances, flagBytes = align(4ull * m_instances, 16);
    const uint64_t need = viewBytes + flagBytes + 16ull * std::max<size_t>(updates.size(), 1);
    // This slot's last frame has finished, so its buffer may be replaced.
    if (!slot.upload || slot.uploadBytes < need) {
        slot.uploadBytes = std::max<uint64_t>(need * 2, 1 << 20);
        slot.upload = nv_helpers_dx12::CreateBuffer(m_ctx->Device(), slot.uploadBytes, D3D12_RESOURCE_FLAG_NONE,
                                                    D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
        slot.upload->SetName(L"LodGpuFrame");
        D3D12_RANGE noRead{ 0, 0 };
        ThrowIfFailed(slot.upload->Map(0, &noRead, reinterpret_cast<void**>(&slot.uploadMapped)));
    }
    std::memcpy(slot.uploadMapped, views.data(), std::min<size_t>(views.size() * sizeof(float), viewBytes));
    std::memcpy(slot.uploadMapped + viewBytes, rewritten.data(), 4ull * m_instances);
    if (!updates.empty()) std::memcpy(slot.uploadMapped + viewBytes + flagBytes, updates.data(), 16ull * updates.size());
    const D3D12_GPU_VIRTUAL_ADDRESS up = slot.upload->GetGPUVirtualAddress();
    const D3D12_GPU_VIRTUAL_ADDRESS workVa = m_work->GetGPUVirtualAddress();

    Constants k = {};
    k.frame = (uint32_t)frame;
    k.instances = m_instances;
    k.groupWork = m_groupWork;
    k.clusterWork = m_clusterWork;
    k.pixelsPerUnit = pixelsPerUnit;
    k.threshold = threshold;
    k.groups = m_groupCount;
    k.maxRequests = m_maxRequests;
    k.classes = m_classCount;
    k.updates = (uint32_t)updates.size();
    k.classArgs = m_classArgs;
    k.classDests = m_classDests;
    k.classCounts = m_classCounts;
    k.classBudgets = m_classBudgets;
    k.tlasArgs = m_tlasArgs;
    k.workLo = (uint32_t)workVa;
    k.workHi = (uint32_t)(workVa >> 32);

    cl->SetComputeRootSignature(m_root.Get());
    cl->SetComputeRoot32BitConstants(0, CONSTANTS, &k, 0);
    const D3D12_GPU_VIRTUAL_ADDRESS srv[SRVS] = {
        m_meshBuf->GetGPUVirtualAddress(), m_instanceBuf->GetGPUVirtualAddress(), m_groupBuf->GetGPUVirtualAddress(),
        m_clusterBuf->GetGPUVirtualAddress(), m_depBuf->GetGPUVirtualAddress(), up,
        m_groupWorkBuf->GetGPUVirtualAddress(), m_clusterWorkBuf->GetGPUVirtualAddress(), up + viewBytes,
        m_ringAddrBuf->GetGPUVirtualAddress(), m_classInfoBuf->GetGPUVirtualAddress(), up + viewBytes + flagBytes };
    for (uint32_t i = 0; i < SRVS; ++i) cl->SetComputeRootShaderResourceView(1 + i, srv[i]);
    ID3D12Resource* uav[UAVS] = { m_state.Get(), m_finePrime.Get(), m_cutIdx.Get(), m_counters.Get(), m_work.Get(),
                                  m_feedback.Get(), m_lastUsed.Get(), m_want.Get(), m_requested.Get(),
                                  m_residency.Get(), m_clasAddr.Get() };
    for (uint32_t i = 0; i < UAVS; ++i) cl->SetComputeRootUnorderedAccessView(1 + SRVS + i, uav[i]->GetGPUVirtualAddress());

    const D3D12_RESOURCE_BARRIER uavAll = CD3DX12_RESOURCE_BARRIER::UAV(nullptr);
    auto dispatch = [&](ID3D12PipelineState* pso, uint32_t threads) {
        if (!threads) return;
        cl->SetPipelineState(pso);
        cl->Dispatch((threads + 63) / 64, 1, 1);
        cl->ResourceBarrier(1, &uavAll);
    };
    dispatch(m_apply.Get(), k.updates);
    dispatch(m_clear.Get(), std::max({ m_instances, m_groupCount, m_classCount }));
    dispatch(m_groups.Get(), m_groupWork);
    dispatch(m_clusters.Get(), m_clusterWork);
    dispatch(m_plan.Get(), m_instances);
    dispatch(m_mark.Get(), m_clusterWork);
    cl->SetPipelineState(m_finalize.Get());
    cl->Dispatch(1, 1, 1);

    // The builds read their arguments, address lists and counts from work; the TLAS update its operation.
    using CD = CD3DX12_RESOURCE_BARRIER;
    constexpr D3D12_RESOURCE_STATES argState =
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
    const D3D12_RESOURCE_BARRIER toRead[2] = {
        uavAll, CD::Transition(m_work.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, argState) };
    cl->ResourceBarrier(2, toRead);
    std::vector<D3D12_RTAS_CLAS_INPUTS_DESC> ins(m_classCount);
    std::vector<D3D12_RTAS_BATCHED_OPERATION_DATA> data(m_classCount);
    std::vector<D3D12_RTAS_OPERATION_DESC> descs(m_classCount);
    for (uint32_t c = 0; c < m_classCount; ++c) {
        const Class& cls = m_classes[c];
        ins[c] = class_inputs(cls.capacity, cls.budget, cls.maxPerArg);
        data[c] = {};
        data[c].BatchScratchData = m_scratch->GetGPUVirtualAddress() + cls.scratchOff;
        data[c].ResultAddressArray = { workVa + m_classDests + 8ull * cls.argBase, sizeof(D3D12_GPU_VIRTUAL_ADDRESS) };
        data[c].IndirectArgumentArray = { workVa + m_classArgs + 16ull * cls.argBase,
                                          sizeof(D3D12_RTAS_OPERATION_BUILD_BLAS_FROM_CLAS_ARGS) };
        data[c].IndirectArgumentArraySize = workVa + m_classCounts + 4ull * c;
        descs[c].Inputs.Type = D3D12_RTAS_OPERATION_TYPE_BUILD_BLAS_FROM_CLAS;
        descs[c].Inputs.pClasDesc = &ins[c];
        descs[c].pBatchedOperationData = &data[c];
    }
    ComPtr<ID3D12CommandListRaytracing2> rt;
    ThrowIfFailed(cl->QueryInterface(IID_PPV_ARGS(&rt)));
    rt->ExecuteIndirectRTASOperations(m_classCount, descs.data(), D3D12_EXECUTE_INDIRECT_RTAS_OPERATIONS_FLAG_NONE);
    cl->ResourceBarrier(1, &uavAll);
    const uint64_t op = m_tlasArgs + 16ull * m_instances + 16;
    tlas.build_gpu_ops(cl, workVa + op, workVa + op + 32);
    const D3D12_RESOURCE_BARRIER back = CD::Transition(m_work.Get(), argState, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    cl->ResourceBarrier(1, &back);

    // Feedback, read back once this list ran.
    const D3D12_RESOURCE_BARRIER toCopy[3] = {
        CD::Transition(m_feedback.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
        CD::Transition(m_lastUsed.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
        CD::Transition(m_want.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE) };
    cl->ResourceBarrier(3, toCopy);
    const uint64_t fb = FEEDBACK_HEADER + 16ull * m_maxRequests;
    cl->CopyBufferRegion(slot.readback.Get(), 0, m_feedback.Get(), 0, fb);
    cl->CopyBufferRegion(slot.readback.Get(), fb, m_lastUsed.Get(), 0, 4ull * m_groupCount);
    cl->CopyBufferRegion(slot.readback.Get(), fb + 4ull * m_groupCount, m_want.Get(), 0, 4ull * m_groupCount);
    const D3D12_RESOURCE_BARRIER toUav[3] = {
        CD::Transition(m_feedback.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        CD::Transition(m_lastUsed.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        CD::Transition(m_want.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS) };
    cl->ResourceBarrier(3, toUav);
    slot.recorded = true;
}

void LodGpu::on_submitted(uint64_t computeFence) {
    if (m_keepPending) {
        m_keepPending = false;
        m_keepFence = computeFence;
    }
    if (!m_instances) return;
    Slot& s = m_slots[m_slot];
    if (!s.recorded) return;
    s.recorded = false;
    s.fence = computeFence;
    s.pending = true;
}
#endif

} // namespace clusters
