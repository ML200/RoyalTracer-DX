#include "voxel_streamer.h"
#include "mc_world.h"
#include "block_registry.h"
#include "voxel_store.h"
#include "../Core/DeviceContext.h"
#include "../planet/worker_pool.h"
#include "../planet/blas_pool.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <intrin.h>
#include <stdexcept>
#include <thread>

namespace mc {

namespace {
constexpr uint64_t BLAS_ALIGN = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT;
constexpr D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS BLAS_FLAGS =
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
constexpr uint32_t LIGHT_DENSE_FLAG = 0x80000000u;
constexpr uint32_t COMPACT_INFO_ENTRIES = 8192;
constexpr double   INITIAL_TRIS_PER_CHUNK = 6000.0;

inline uint64_t align_up_u64(uint64_t v, uint64_t a) { return (v + a - 1) & ~(a - 1); }
inline float luminance(const Vec3f& e) { return 0.2126f * e.x + 0.7152f * e.y + 0.0722f * e.z; }
}

void SpanAllocator::init(uint64_t capacity, uint64_t granularity) {
    m_capacity = capacity;
    m_granularity = granularity ? granularity : 1;
    m_used = 0;
    m_free.clear();
    if (capacity) m_free.push_back(Span{ 0, capacity });
}

// Allocates a best-fit aligned span from the free list.
bool SpanAllocator::allocate(uint64_t count, uint64_t& offset) {
    count = align_up_u64(count, m_granularity);
    if (count == 0) count = m_granularity;
    size_t best = (size_t)-1;
    uint64_t bestCount = ~0ull;
    for (size_t i = 0; i < m_free.size(); ++i) {
        if (m_free[i].count >= count && m_free[i].count < bestCount) { best = i; bestCount = m_free[i].count; }
    }
    if (best == (size_t)-1) return false;
    offset = m_free[best].first;
    m_free[best].first += count;
    m_free[best].count -= count;
    if (m_free[best].count == 0) m_free.erase(m_free.begin() + (ptrdiff_t)best);
    m_used += count;
    return true;
}

// Returns a span and coalesces adjacent free ranges.
void SpanAllocator::free(uint64_t offset, uint64_t count) {
    count = align_up_u64(count, m_granularity);
    if (count == 0) count = m_granularity;
    m_used -= count;
    size_t i = 0;
    while (i < m_free.size() && m_free[i].first < offset) ++i;
    m_free.insert(m_free.begin() + (ptrdiff_t)i, Span{ offset, count });
    if (i > 0 && m_free[i - 1].first + m_free[i - 1].count == m_free[i].first) {
        m_free[i - 1].count += m_free[i].count;
        m_free.erase(m_free.begin() + (ptrdiff_t)i);
        --i;
    }
    if (i + 1 < m_free.size() && m_free[i].first + m_free[i].count == m_free[i + 1].first) {
        m_free[i].count += m_free[i + 1].count;
        m_free.erase(m_free.begin() + (ptrdiff_t)(i + 1));
    }
}

VoxelStreamer::VoxelStreamer() = default;
VoxelStreamer::~VoxelStreamer() {
    m_workers.reset();
    if (m_staging && m_stagingMapped) m_staging->Unmap(0, nullptr);
}

// Creates persistent GPU pools and starts streaming worker resources.
void VoxelStreamer::init(ID3D12Device5* device, DeviceContext* ctx, World* world, const StreamerConfig& cfg) {
    m_device = device;
    m_ctx = ctx;
    m_world = world;
    m_cfg = cfg;
    m_lodFactorNow = cfg.lodFactor;
    for (double& e : m_levelTriEst) e = INITIAL_TRIS_PER_CHUNK;
    m_workers = std::make_unique<planet::WorkerPool>();
    m_lodPool = std::make_unique<planet::WorkerPool>(std::max(2u, std::thread::hardware_concurrency() / 2u));

    m_blasPool = planet::create_buffer(device, cfg.blasPoolBytes,
                                       D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                       D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, planet::HEAP_DEFAULT);
    m_blasPool->SetName(L"MinecraftBlasPool");
    m_blasPoolVa = m_blasPool->GetGPUVirtualAddress();
    m_blasAlloc.init(cfg.blasPoolBytes, BLAS_ALIGN);
    if (cfg.blasCompaction) {
        m_buildPool = planet::create_buffer(device, cfg.blasBuildPoolBytes,
                                            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, planet::HEAP_DEFAULT);
        m_buildPool->SetName(L"MinecraftBlasBuildPool");
        m_buildPoolVa = m_buildPool->GetGPUVirtualAddress();
        m_buildAlloc.init(cfg.blasBuildPoolBytes, BLAS_ALIGN);
    }

    const uint64_t stagingBytes = (uint64_t)cfg.stagingSlots * cfg.stagingSlotBytes;
    m_staging = planet::create_buffer(device, stagingBytes, D3D12_RESOURCE_FLAG_NONE,
                                      D3D12_RESOURCE_STATE_GENERIC_READ, planet::HEAP_UPLOAD);
    m_staging->SetName(L"MinecraftStaging");
    {
        void* p = nullptr;
        D3D12_RANGE noRead{ 0, 0 };
        if (FAILED(m_staging->Map(0, &noRead, &p))) throw std::runtime_error("mc: staging Map failed");
        m_stagingMapped = static_cast<uint8_t*>(p);
    }
    m_stagingSlots.resize(cfg.stagingSlots);
    for (uint32_t i = 0; i < cfg.stagingSlots; ++i) {
        m_stagingSlots[i].offset = (uint64_t)i * cfg.stagingSlotBytes;
        m_stagingSlots[i].mapped = m_stagingMapped + m_stagingSlots[i].offset;
    }

    const uint32_t frames = std::max(1u, ctx->BufferCount());
    m_scratch.resize(frames);
    m_scratchUsed.assign(frames, 0);
    for (uint32_t i = 0; i < frames; ++i) {
        m_scratch[i] = planet::create_buffer(device, cfg.scratchBytesPerFrame,
                                             D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                             D3D12_RESOURCE_STATE_COMMON, planet::HEAP_DEFAULT);
        m_scratch[i]->SetName(L"MinecraftBlasScratch");
    }
    if (cfg.blasCompaction) {
        m_compactInfo.resize(frames); m_compactReadback.resize(frames);
        m_compactMapped.assign(frames, nullptr); m_compactCount.assign(frames, 0); m_compactInfoCopyState.assign(frames, 0);
        for (uint32_t i = 0; i < frames; ++i) {
            m_compactInfo[i] = planet::create_buffer(device, (uint64_t)COMPACT_INFO_ENTRIES * sizeof(uint64_t),
                                                     D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                                     D3D12_RESOURCE_STATE_UNORDERED_ACCESS, planet::HEAP_DEFAULT);
            m_compactInfo[i]->SetName(L"MinecraftCompactedSizes");
            m_compactReadback[i] = planet::create_buffer(device, (uint64_t)COMPACT_INFO_ENTRIES * sizeof(uint64_t),
                                                         D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, planet::HEAP_READBACK);
            m_compactReadback[i]->SetName(L"MinecraftCompactedSizesReadback");
            void* p = nullptr;
            if (FAILED(m_compactReadback[i]->Map(0, nullptr, &p))) throw std::runtime_error("mc: compacted-size readback Map failed");
            m_compactMapped[i] = static_cast<const uint64_t*>(p);
        }
    }
    std::printf("[mc] streamer: %u mesh workers + %u LOD workers, BLAS pool %llu MB%s, staging %llu MB, scratch %u x %llu MB, budget %.1fM tris\n",
                m_workers->thread_count(), m_lodPool->thread_count(), (unsigned long long)(cfg.blasPoolBytes >> 20),
                cfg.blasCompaction ? (" compacted (+" + std::to_string(cfg.blasBuildPoolBytes >> 20) + " MB build pool)").c_str() : "",
                (unsigned long long)(stagingBytes >> 20), frames, (unsigned long long)(cfg.scratchBytesPerFrame >> 20),
                (double)cfg.triangleBudget / 1.0e6);
}

D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS VoxelStreamer::build_flags() const {
    return m_cfg.blasCompaction
        ? (D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS)(BLAS_FLAGS | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION)
        : BLAS_FLAGS;
}

void VoxelStreamer::bind_geometry(ID3D12Resource* vertexGlobal, ID3D12Resource* indexGlobal, uint32_t combinedVertexCount,
                                  uint32_t vertexBaseElems, uint32_t indexBaseElems,
                                  ID3D12Resource* materialIds, uint32_t matIdBaseElems,
                                  uint32_t instancePropsBase) {
    m_vertexGlobal = vertexGlobal;
    m_indexGlobal = indexGlobal;
    m_combinedVertexCount = combinedVertexCount;
    m_vertexBase = vertexBaseElems;
    m_indexBase = indexBaseElems;
    m_materialIds = materialIds;
    m_matIdBase = matIdBaseElems;
    m_propsBase = instancePropsBase;
    m_vertexVa = vertexGlobal->GetGPUVirtualAddress();
    m_indexVa = indexGlobal->GetGPUVirtualAddress();
    {
        void* p = nullptr;
        D3D12_RANGE noRead{ 0, 0 };
        if (FAILED(materialIds->Map(0, &noRead, &p))) throw std::runtime_error("mc: material-id Map failed");
        m_materialIdsMapped = static_cast<uint8_t*>(p);
    }
    m_vtxAlloc.init(m_cfg.vertexCapacity, 64);
    m_idxAlloc.init(m_cfg.indexCapacity, 192);
    m_matAlloc.init(m_cfg.matIdCapacity, 64);
}

void VoxelStreamer::bind_omm(ID3D12Resource* indexBuffer, D3D12_GPU_VIRTUAL_ADDRESS ommArray, const OmmTable* table) {
    m_ommIndexBuffer = indexBuffer;
    m_ommArrayVa = ommArray;
    m_ommTable = table;
    m_ommIndexVa = indexBuffer ? indexBuffer->GetGPUVirtualAddress() : 0;
    m_ommIndicesMapped = nullptr;
    if (indexBuffer) {
        void* p = nullptr;
        D3D12_RANGE noRead{ 0, 0 };
        if (FAILED(indexBuffer->Map(0, &noRead, &p))) throw std::runtime_error("mc: micromap index Map failed");
        m_ommIndicesMapped = static_cast<uint8_t*>(p);
    }
    m_ommAlloc.init(m_cfg.ommIndexCapacity, 64);
}

void VoxelStreamer::bind_lights(const LightBinding& b) {
    {
        std::lock_guard<std::mutex> lk(m_poolMutex);
        m_lightsBound = false;
        m_lights = b;
        m_lightRecAlloc.init(b.recordCapacity, 64);
        m_lightNodeAlloc.init(b.nodeCapacity, 64);
        ++m_lightGen;
        m_lightsBound = b.records && b.nodes && b.leafIndex && b.trails && b.recordCapacity && b.nodeCapacity;
    }
    m_lightTlasHasVoxels = false;
    for (auto& kv : m_chunks) {
        Chunk& c = kv.second;
        if (c.lightSlot != NONE) exclude_light_slot(c);
        if (c.gpu.light.valid() || c.next.light.valid()) {
            c.gpu.light = LightData{};
            c.next.light = LightData{};
            c.lightsDropped = true;
            c.propsWritten = false;
        }
    }
    for (Retired& r : m_retired) r.gpu.light = LightData{};
    m_retiredLights.clear();
    m_lightSetDirty = true;
    std::printf("[mc] lights %s: %u records, %u nodes, slots from %u\n", m_lightsBound ? "bound" : "unbound",
                b.recordCapacity, b.nodeCapacity, b.slotBase);
}

void VoxelStreamer::unbind_lights() {
    bind_lights(LightBinding{});
}

void VoxelStreamer::set_light_slot_base(uint32_t base) {
    if (m_lights.slotBase == base) return;
    m_lights.slotBase = base;
    m_lightTlasHasVoxels = false;
    m_lightSetDirty = true;
}

void VoxelStreamer::on_light_tlas_published(uint32_t version, bool hasVoxelLeaves) {
    if (hasVoxelLeaves) m_lightLiveVersion = std::max(m_lightLiveVersion, version);
    m_lightTlasHasVoxels = hasVoxelLeaves;
}

bool VoxelStreamer::light_slot_live(uint32_t slot) const {
    if (!m_lightTlasHasVoxels || slot >= m_lightSlots.size()) return false;
    const LightSlot& s = m_lightSlots[slot];
    return s.includedAt <= m_lightLiveVersion && m_lightLiveVersion < s.excludedAt;
}

bool VoxelStreamer::light_region_free(uint32_t excludedAt) const {
    return !m_lightTlasHasVoxels || excludedAt <= m_lightLiveVersion;
}

uint32_t VoxelStreamer::exclude_light_slot(Chunk& c) {
    if (c.lightSlot == NONE) return 0;
    LightSlot& s = m_lightSlots[c.lightSlot];
    const uint32_t ex = m_lightVersion + 1;
    s.excludedAt = ex;
    m_retiredLightSlots.push_back(RetiredLightSlot{ c.lightSlot, m_frame + m_cfg.retireFrames, ex });
    c.lightSlot = NONE;
    m_lightSetDirty = true;
    return ex;
}

void VoxelStreamer::light_snapshot(const planet::DVec3& sceneOrigin, std::vector<lt::TLASExtraLeaf>& out,
                                   uint32_t& version, uint32_t& slotCount) const {
    using namespace DirectX;
    out.clear();
    version = m_lightVersion;
    slotCount = m_lights.slotBase + (uint32_t)m_lightSlots.size();
    if (!m_lightsBound) return;
    for (uint32_t i = 0; i < (uint32_t)m_lightSlots.size(); ++i) {
        const LightSlot& s = m_lightSlots[i];
        if (!s.active() || s.includedAt > m_lightVersion || !s.light.valid()) continue;
        const double ob[3] = { (double)s.origin[0], (double)s.origin[1], (double)s.origin[2] };
        double os[3];
        m_placement.to_scene(ob, os);
        const float tx = (float)(os[0] - sceneOrigin.x);
        const float ty = (float)(os[1] - sceneOrigin.y);
        const float tz = (float)(os[2] - sceneOrigin.z);
        lt::TLASExtraLeaf leaf;
        leaf.slot = m_lights.slotBase + i;
        const XMMATRIX world = placement_matrix(tx, ty, tz);
        XMVECTOR mn = XMVectorReplicate(1e30f), mx = XMVectorReplicate(-1e30f);
        for (int k = 0; k < 8; ++k) {
            const XMVECTOR corner = XMVectorSet((k & 1) ? s.light.bmax[0] : s.light.bmin[0],
                                                (k & 2) ? s.light.bmax[1] : s.light.bmin[1],
                                                (k & 4) ? s.light.bmax[2] : s.light.bmin[2], 1.0f);
            const XMVECTOR pt = XMVector3TransformCoord(corner, world);
            mn = XMVectorMin(mn, pt); mx = XMVectorMax(mx, pt);
        }
        XMStoreFloat3(&leaf.aabb.mn, mn);
        XMStoreFloat3(&leaf.aabb.mx, mx);
        const float areaScale = (float)m_placement.area_scale();
        leaf.power = s.light.power * areaScale;
        XMStoreFloat3(&leaf.cone.axis, XMVector3Normalize(XMVector3TransformNormal(XMVectorSet(s.light.axis[0], s.light.axis[1], s.light.axis[2], 0.0f), world)));
        leaf.cone.theta_o = std::acos(std::clamp(s.light.cosTheta, -1.0f, 1.0f));
        leaf.cone.theta_e = lt::LT_HALF_PI;
        leaf.primCount = (uint32_t)s.light.recCount;
        leaf.sumPower = leaf.power;
        leaf.sumPowerSq = leaf.power * leaf.power;
        XMFLOAT4X4 worldF;
        XMStoreFloat4x4(&worldF, world);
        leaf.record = lt::makeSlotRecord(s.instanceID, m_lights.nodeBase + (uint32_t)s.light.nodeOff, worldF);
        out.push_back(leaf);
    }
}

DirectX::XMMATRIX VoxelStreamer::placement_matrix(float tx, float ty, float tz) const {
    using namespace DirectX;
    const double* A = m_placement.a;
    return XMMatrixSet((float)A[0], (float)A[3], (float)A[6], 0.0f,
                       (float)A[1], (float)A[4], (float)A[7], 0.0f,
                       (float)A[2], (float)A[5], (float)A[8], 0.0f,
                       tx, ty, tz, 1.0f);
}

// Reserves all GPU spans needed by one chunk and its optional lights.
bool VoxelStreamer::allocate_gpu(GpuChunk& g, uint32_t vtxCount, uint32_t idxCount, uint32_t triCount, uint64_t blasSize,
                                 uint32_t lightRecs, uint32_t lightNodes, bool* buildPoolFull) {
    std::lock_guard<std::mutex> lk(m_poolMutex);
    if (buildPoolFull) *buildPoolFull = false;
    uint64_t v = 0, i = 0, m = 0, b = 0;
    if (!m_vtxAlloc.allocate(vtxCount, v)) return false;
    if (!m_idxAlloc.allocate(idxCount, i)) { m_vtxAlloc.free(v, vtxCount); return false; }
    if (!m_matAlloc.allocate(triCount, m)) { m_vtxAlloc.free(v, vtxCount); m_idxAlloc.free(i, idxCount); return false; }
    SpanAllocator& blasAlloc = m_cfg.blasCompaction ? m_buildAlloc : m_blasAlloc;
    if (!blasAlloc.allocate(blasSize, b)) {
        m_vtxAlloc.free(v, vtxCount); m_idxAlloc.free(i, idxCount); m_matAlloc.free(m, triCount);
        if (buildPoolFull) *buildPoolFull = m_cfg.blasCompaction;
        return false;
    }
    g.vtxOff = m_vertexBase + v; g.vtxCount = vtxCount;
    g.idxOff = m_indexBase + i;  g.idxCount = idxCount;
    g.matOff = m_matIdBase + m;  g.matCount = triCount;
    if (m_cfg.blasCompaction) { g.buildOff = b; g.buildSize = blasSize; g.buildVa = m_buildPoolVa + b; g.blasOff = g.blasSize = 0; g.blasVa = 0; }
    else                      { g.blasOff = b;  g.blasSize = blasSize;  g.blasVa = m_blasPoolVa + b; }
    g.triCount = triCount;
    g.light = LightData{};
    if (lightRecs && lightNodes && m_lightsBound) {
        lightNodes += m_lights.compactNodes ? 1u : 0u;
        uint64_t r = 0, n = 0;
        if (m_lightRecAlloc.allocate(lightRecs, r)) {
            if (m_lightNodeAlloc.allocate(lightNodes, n)) {
                g.light.recOff = r; g.light.recCount = lightRecs;
                g.light.nodeOff = n; g.light.nodeCount = lightNodes;
                g.light.gen = m_lightGen;
                g.light.compactNodes = m_lights.compactNodes;
            } else {
                m_lightRecAlloc.free(r, lightRecs);
            }
        }
    }
    return true;
}

void VoxelStreamer::free_light(const LightData& l) {
    if (!l.valid() || l.gen != m_lightGen) return;
    m_lightRecAlloc.free(l.recOff, l.recCount);
    m_lightNodeAlloc.free(l.nodeOff, l.nodeCount);
}

void VoxelStreamer::free_gpu(const GpuChunk& g) {
    if (!g.valid()) return;
    std::lock_guard<std::mutex> lk(m_poolMutex);
    m_vtxAlloc.free(g.vtxOff - m_vertexBase, g.vtxCount);
    m_idxAlloc.free(g.idxOff - m_indexBase, g.idxCount);
    m_matAlloc.free(g.matOff - m_matIdBase, g.matCount);
    if (g.blasSize)  m_blasAlloc.free(g.blasOff, g.blasSize);
    if (g.buildSize) m_buildAlloc.free(g.buildOff, g.buildSize);
    if (g.ommCount) m_ommAlloc.free(g.ommOff, g.ommCount);
    free_light(g.light);
}

void VoxelStreamer::retire_gpu(const GpuChunk& g, uint32_t lightExcludedAt) {
    if (!g.valid()) return;
    m_retired.push_back(Retired{ g, m_frame + m_cfg.retireFrames, lightExcludedAt });
}

void VoxelStreamer::retire_light(const LightData& l, uint32_t excludedAt) {
    if (!l.valid()) return;
    m_retiredLights.push_back(RetiredLight{ l, m_frame + m_cfg.retireFrames, excludedAt });
}

int VoxelStreamer::acquire_staging() {
    for (size_t i = 0; i < m_stagingSlots.size(); ++i)
        if (!m_stagingSlots[i].inUse) { m_stagingSlots[i].inUse = true; return (int)i; }
    return -1;
}

void VoxelStreamer::release_staging_after(int slot, uint64_t copyFence) {
    if (slot < 0) return;
    m_stagingSlots[(size_t)slot].copyFence = copyFence;
    m_stagingSlots[(size_t)slot].pendingFence = true;
}

uint32_t VoxelStreamer::acquire_instance_slot() {
    if (!m_freeSlots.empty()) { const uint32_t s = m_freeSlots.back(); m_freeSlots.pop_back(); return s; }
    if (m_nextSlot < m_cfg.maxInstances) return m_nextSlot++;
    return NONE;
}

void VoxelStreamer::release_instance_slot(uint32_t slot) {
    if (slot == NONE) return;
    m_retiredSlots.push_back(RetiredSlot{ slot, m_frame + m_cfg.retireFrames });
}

float VoxelStreamer::priority_of(const NodeKey& k) const {
    return (float)LodTree::node_distance(k, m_cam);
}

uint32_t VoxelStreamer::piece_count(uint32_t count, uint32_t minPerPiece) const {
    if (!m_lodPool || count < 2u * std::max(minPerPiece, 1u)) return 1;
    const uint32_t byWork = count / std::max(minPerPiece, 1u);
    return std::max(1u, std::min(byWork, m_lodPool->thread_count() * 4u));
}

bool VoxelStreamer::build_lights(const ChunkMesh& mesh, MeshJob& job, std::vector<LightTriangle>& records,
                                 lt::LightTreeBuilder::SingleBLAS& blas) {
    using namespace DirectX;
    const uint32_t n = mesh.light_tri_count();
    records.clear();
    if (n == 0) return false;
    const BlockRegistry& reg = m_world->registry();
    records.reserve(n);
    float total = 0.0f;
    for (uint32_t k = 0; k < n; ++k) {
        const uint32_t t = mesh.light_tri_index(k);
        const MeshVertex& a = mesh.vertices[mesh.indices[3 * t + 0]];
        const MeshVertex& b = mesh.vertices[mesh.indices[3 * t + 1]];
        const MeshVertex& c = mesh.vertices[mesh.indices[3 * t + 2]];
        const uint32_t mat = mesh.materials[t];
        const Vec3f e = mat < reg.materialEmission.size() ? reg.materialEmission[mat] : Vec3f{};
        LightTriangle lt{};
        lt.x = XMFLOAT3(a.px, a.py, a.pz);
        lt.y = XMFLOAT3(b.px, b.py, b.pz);
        lt.z = XMFLOAT3(c.px, c.py, c.pz);
        lt.meshID = 0;
        lt.emission = XMFLOAT3(e.x, e.y, e.z);
        const Vec3f cr = cross(Vec3f{ b.px - a.px, b.py - a.py, b.pz - a.pz }, Vec3f{ c.px - a.px, c.py - a.py, c.pz - a.pz });
        const float area = 0.5f * std::sqrt(std::max(0.0f, dot(cr, cr)));
        lt.weight = std::max(area, 1e-10f) * luminance(e);
        lt.triCount = n;
        total += lt.weight;
        records.push_back(lt);
    }
    for (LightTriangle& r : records) r.totalWeight = total;
    try {
        lt::LightTreeBuilder::BuildSingleBLAS(records, blas, 32u);
    } catch (...) {
        blas.nodes.clear();
    }
    if (blas.nodes.empty() || blas.leafTriLocal.size() != records.size()) { records.clear(); return false; }
    (void)job;
    return true;
}

ChunkMesher& VoxelStreamer::thread_mesher() {
    thread_local std::unique_ptr<ChunkMesher> mesher;
    thread_local const World* mesherWorld = nullptr;
    if (!mesher || mesherWorld != m_world) {
        mesher = std::make_unique<ChunkMesher>(m_world->registry(), m_world->store());
        mesherWorld = m_world;
    }
    return *mesher;
}

// Meshes one chunk on a worker before scheduling its upload.
void VoxelStreamer::run_job(const std::shared_ptr<MeshJob>& job) {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    ChunkMesh mesh;
    thread_mesher().mesh(job->key, job->params, mesh);
    if (mesh.empty()) {
        job->meshMs = (float)std::chrono::duration<double, std::milli>(clock::now() - t0).count();
        job->state.store(3, std::memory_order_release);
        return;
    }

    std::vector<LightTriangle> records;
    lt::LightTreeBuilder::SingleBLAS blas;
    const bool wantLights = job->wantLights && m_lightsBound.load() && mesh.light_tri_count() > 0;
    const bool haveLights = wantLights && build_lights(mesh, *job, records, blas);
    job->meshMs = (float)std::chrono::duration<double, std::milli>(clock::now() - t0).count();

    uint64_t ommOff = 0;
    uint32_t ommCount = 0;
    if (m_ommTable && !m_ommTable->empty() && m_ommIndicesMapped && mesh.alphaTriCount) {
        bool any = false;
        for (uint32_t t = mesh.opaqueTriCount; t < mesh.triangle_count() && !any; ++t) any = mesh.ommKeys[t] != 0;
        if (any) {
            std::lock_guard<std::mutex> lk(m_poolMutex);
            if (m_ommAlloc.allocate(mesh.alphaTriCount, ommOff)) ommCount = mesh.alphaTriCount;
        }
    }

    D3D12_RAYTRACING_GEOMETRY_DESC gd[2] = {};
    D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC ommTri = {};
    D3D12_RAYTRACING_GEOMETRY_OMM_LINKAGE_DESC ommLink = {};
    uint32_t geomCount = 0;
    auto geom = [&](uint32_t triCount, bool opaque, bool omm) {
        D3D12_RAYTRACING_GEOMETRY_DESC& g = gd[geomCount++];
        D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC& tri = omm ? ommTri : g.Triangles;
        g.Type  = omm ? D3D12_RAYTRACING_GEOMETRY_TYPE_OMM_TRIANGLES : D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
        g.Flags = opaque ? D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE : D3D12_RAYTRACING_GEOMETRY_FLAG_NONE;
        tri.VertexBuffer.StrideInBytes = sizeof(MeshVertex);
        tri.VertexCount  = m_combinedVertexCount;
        tri.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
        tri.IndexCount   = triCount * 3;
        tri.IndexFormat  = DXGI_FORMAT_R32_UINT;
        if (omm) {
            ommLink.OpacityMicromapIndexBuffer.StrideInBytes = sizeof(int32_t);
            ommLink.OpacityMicromapIndexFormat = DXGI_FORMAT_R32_UINT;
            ommLink.OpacityMicromapArray = m_ommArrayVa;
            g.OmmTriangles.pTriangles  = &ommTri;
            g.OmmTriangles.pOmmLinkage = &ommLink;
        }
    };
    if (mesh.opaqueTriCount) geom(mesh.opaqueTriCount, true, false);
    if (mesh.alphaTriCount)  geom(mesh.alphaTriCount, false, ommCount != 0);
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS in = {};
    in.Type           = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    in.DescsLayout    = D3D12_ELEMENTS_LAYOUT_ARRAY;
    in.Flags          = build_flags();
    in.NumDescs       = geomCount;
    in.pGeometryDescs = gd;
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO info = {};
    m_device->GetRaytracingAccelerationStructurePrebuildInfo(&in, &info);

    GpuChunk g;
    bool buildPoolFull = false;
    if (!allocate_gpu(g, (uint32_t)mesh.vertices.size(), (uint32_t)mesh.indices.size(), mesh.triangle_count(),
                      align_up_u64(info.ResultDataMaxSizeInBytes, BLAS_ALIGN),
                      haveLights ? (uint32_t)records.size() : 0u, haveLights ? (uint32_t)blas.nodes.size() : 0u, &buildPoolFull)) {
        if (ommCount) { std::lock_guard<std::mutex> lk(m_poolMutex); m_ommAlloc.free(ommOff, ommCount); }
        job->buildPoolFull = buildPoolFull;
        job->state.store(2, std::memory_order_release);
        return;
    }
    g.opaqueTriCount = mesh.opaqueTriCount;
    g.ommOff = ommOff; g.ommCount = ommCount;
    job->lightsDropped = mesh.light_tri_count() > 0 && !g.light.valid();
    LightBinding lb;
    if (g.light.valid()) {
        { std::lock_guard<std::mutex> lk(m_poolMutex); lb = m_lights; }
        g.light.litOpaque = mesh.opaqueLightTriCount;
        g.light.litAlpha  = mesh.alphaLightTriCount;
        const lt::LightBLASNodeGpu& root = blas.nodes[0];
        g.light.bmin[0] = root.bmin.x; g.light.bmin[1] = root.bmin.y; g.light.bmin[2] = root.bmin.z;
        g.light.bmax[0] = root.bmax.x; g.light.bmax[1] = root.bmax.y; g.light.bmax[2] = root.bmax.z;
        g.light.power = root.power;
        g.light.axis[0] = root.axis.x; g.light.axis[1] = root.axis.y; g.light.axis[2] = root.axis.z;
        g.light.cosTheta = root.cosTheta_o; g.light.sinTheta = root.sinTheta_o;
    }
    job->gpu = g;
    job->lightGen = g.light.gen;
    job->scratchSize = align_up_u64(info.ScratchDataSizeInBytes, BLAS_ALIGN);

    job->vtxBytes = (uint32_t)(mesh.vertices.size() * sizeof(MeshVertex));
    job->idxBytes = (uint32_t)(mesh.indices.size() * sizeof(uint32_t));
    job->recBytes = job->nodeBytes = job->leafBytes = job->trailBytes = 0;
    if (g.light.valid()) {
        job->recBytes   = (uint32_t)(records.size() * sizeof(LightTriangle));
        job->nodeBytes  = (uint32_t)(g.light.nodeCount * lt::LightBLASNodeStride(g.light.compactNodes));
        job->leafBytes  = (uint32_t)(blas.leafTriLocal.size() * sizeof(uint32_t));
        job->trailBytes = (uint32_t)(blas.trails.size() * sizeof(lt::LightTreeTrail));
    }
    const uint64_t offIdx   = align_up_u64(job->vtxBytes, 256);
    const uint64_t offRec   = align_up_u64(offIdx + job->idxBytes, 256);
    const uint64_t offNode  = align_up_u64(offRec + job->recBytes, 256);
    const uint64_t offLeaf  = align_up_u64(offNode + job->nodeBytes, 256);
    const uint64_t offTrail = align_up_u64(offLeaf + job->leafBytes, 256);
    const uint64_t need     = offTrail + job->trailBytes;
    uint8_t* dst = nullptr;
    if (job->stagingSlot >= 0 && need <= m_cfg.stagingSlotBytes) {
        dst = m_stagingSlots[(size_t)job->stagingSlot].mapped;
        job->srcBuffer = m_staging.Get();
        job->srcVtxOff = m_stagingSlots[(size_t)job->stagingSlot].offset;
    } else {
        try {
            job->privateUpload = planet::create_buffer(m_device, need, D3D12_RESOURCE_FLAG_NONE,
                                                       D3D12_RESOURCE_STATE_GENERIC_READ, planet::HEAP_UPLOAD);
        } catch (...) {
            job->privateUpload.Reset();
        }
        void* p = nullptr;
        D3D12_RANGE noRead{ 0, 0 };
        if (!job->privateUpload || FAILED(job->privateUpload->Map(0, &noRead, &p))) {
            free_gpu(g);
            job->gpu = GpuChunk{};
            job->state.store(2, std::memory_order_release);
            return;
        }
        dst = static_cast<uint8_t*>(p);
        job->srcBuffer = job->privateUpload.Get();
        job->srcVtxOff = 0;
    }
    job->srcMapped  = dst;
    job->srcIdxOff   = job->srcVtxOff + offIdx;
    job->srcRecOff   = job->srcVtxOff + offRec;
    job->srcNodeOff  = job->srcVtxOff + offNode;
    job->srcLeafOff  = job->srcVtxOff + offLeaf;
    job->srcTrailOff = job->srcVtxOff + offTrail;
    std::memcpy(dst, mesh.vertices.data(), job->vtxBytes);
    uint32_t* idx = reinterpret_cast<uint32_t*>(dst + offIdx);
    const uint32_t vbase = (uint32_t)g.vtxOff;
    for (size_t i = 0; i < mesh.indices.size(); ++i) idx[i] = mesh.indices[i] + vbase;
    if (g.light.valid()) {
        const uint32_t recBase = lb.recordBase + (uint32_t)g.light.recOff;
        std::memcpy(dst + offRec, records.data(), job->recBytes);
        const auto packedNodes = lt::EncodeLightBLAS(blas.nodes, g.light.compactNodes, recBase);
        std::memcpy(dst + offNode, packedNodes.data(), job->nodeBytes);
        uint32_t* leaf = reinterpret_cast<uint32_t*>(dst + offLeaf);
        for (size_t i = 0; i < blas.leafTriLocal.size(); ++i) leaf[i] = recBase + blas.leafTriLocal[i];
        std::memcpy(dst + offTrail, blas.trails.data(), job->trailBytes);
    }
    std::memcpy(m_materialIdsMapped + g.matOff * sizeof(uint32_t), mesh.materials.data(), mesh.materials.size() * sizeof(uint32_t));
    if (g.ommCount) {
        int32_t* omm = reinterpret_cast<int32_t*>(m_ommIndicesMapped + g.ommOff * sizeof(int32_t));
        const int32_t unknown = D3D12_RAYTRACING_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_UNKNOWN_OPAQUE;
        for (uint32_t t = mesh.opaqueTriCount; t < mesh.triangle_count(); ++t)
            omm[t - mesh.opaqueTriCount] = mesh.ommKeys[t] ? m_ommTable->lookup(mesh.ommKeys[t], unknown) : unknown;
    }
    if (job->privateUpload) job->privateUpload->Unmap(0, nullptr);
    _mm_sfence();
    job->state.store(1, std::memory_order_release);
}

// Reclaims fence-retired staging, geometry, BLAS, and light allocations.
void VoxelStreamer::reclaim() {
    const uint64_t computeDone = m_ctx->PlanetComputeCompleted();
    for (size_t i = 0; i < m_uploading.size(); ) {
        auto it = m_chunks.find(m_uploading[i]);
        if (it == m_chunks.end() || (it->second.state != State::Uploading && it->second.state != State::Compacting)) {
            m_uploading[i] = m_uploading.back(); m_uploading.pop_back();
            continue;
        }
        Chunk& c = it->second;
        if (c.state == State::Uploading) {
            if (c.uploadFence == 0 || c.uploadFence > computeDone) { ++i; continue; }
            if (m_cfg.blasCompaction && c.next.buildSize) {
                if (!c.compactQueued) {
                    const uint64_t size = (c.compactSlot < m_compactMapped.size() && c.compactIndex < COMPACT_INFO_ENTRIES)
                        ? m_compactMapped[c.compactSlot][c.compactIndex] : 0;
                    const uint64_t aligned = align_up_u64(size, BLAS_ALIGN);
                    if (size == 0 || aligned > c.next.buildSize) {
                        free_gpu(c.next);
                        c.next = GpuChunk{};
                        c.state = State::Pending;
                        c.retryFrame = m_frame + 20;
                        m_uploading[i] = m_uploading.back(); m_uploading.pop_back();
                        continue;
                    }
                    uint64_t off = 0;
                    bool ok = false;
                    { std::lock_guard<std::mutex> lk(m_poolMutex); ok = m_blasAlloc.allocate(aligned, off); }
                    if (!ok) { ++i; continue; }
                    c.next.blasOff = off; c.next.blasSize = aligned; c.next.blasVa = m_blasPoolVa + off;
                    c.compactQueued = true;
                    m_toCompact.push_back(m_uploading[i]);
                }
                ++i;
                continue;
            }
        } else {
            if (c.compactFence == 0 || c.compactFence > computeDone) { ++i; continue; }
            { std::lock_guard<std::mutex> lk(m_poolMutex); m_buildAlloc.free(c.next.buildOff, c.next.buildSize); }
            c.next.buildOff = c.next.buildSize = 0; c.next.buildVa = 0;
            c.compactQueued = false;
        }
        if (c.rebuilding) {
            retire_gpu(c.gpu, exclude_light_slot(c));
            c.rebuilding = false;
        }
        c.gpu = c.next;
        c.next = GpuChunk{};
        if (!m_cfg.lights && c.gpu.light.valid()) {
            retire_light(c.gpu.light, 0);
            c.gpu.light = LightData{};
            c.lightsDropped = true;
        }
        c.state = c.gpu.valid() ? State::Ready : State::Empty;
        c.propsWritten = false;
        set_resident(c, true);
        m_renderListChanged = true;
        if (c.remeshAfterUpload) {
            c.remeshAfterUpload = false;
            c.rebuilding = c.gpu.valid();
            c.state = State::Pending;
            c.retryFrame = 0;
        }
        m_uploading[i] = m_uploading.back(); m_uploading.pop_back();
    }
    for (StagingSlot& s : m_stagingSlots) {
        if (s.inUse && s.pendingFence && s.copyFence <= computeDone) { s.inUse = false; s.pendingFence = false; }
    }
    for (size_t i = 0; i < m_retiredUploads.size(); ) {
        if (m_retiredUploads[i].copyFence <= computeDone) { m_retiredUploads[i] = std::move(m_retiredUploads.back()); m_retiredUploads.pop_back(); }
        else ++i;
    }
    for (size_t i = 0; i < m_retired.size(); ) {
        const Retired& r = m_retired[i];
        if (r.frame <= m_frame && (!r.gpu.light.valid() || light_region_free(r.lightExcludedAt))) {
            free_gpu(r.gpu); m_retired[i] = m_retired.back(); m_retired.pop_back();
        } else ++i;
    }
    for (size_t i = 0; i < m_retiredLights.size(); ) {
        const RetiredLight& r = m_retiredLights[i];
        if (r.frame <= m_frame && light_region_free(r.excludedAt)) {
            { std::lock_guard<std::mutex> lk(m_poolMutex); free_light(r.light); }
            m_retiredLights[i] = m_retiredLights.back(); m_retiredLights.pop_back();
        } else ++i;
    }
    for (size_t i = 0; i < m_retiredSlots.size(); ) {
        if (m_retiredSlots[i].frame <= m_frame) { m_freeSlots.push_back(m_retiredSlots[i].slot); m_retiredSlots[i] = m_retiredSlots.back(); m_retiredSlots.pop_back(); }
        else ++i;
    }
    for (size_t i = 0; i < m_retiredLightSlots.size(); ) {
        const RetiredLightSlot& r = m_retiredLightSlots[i];
        if (r.frame <= m_frame && light_region_free(r.excludedAt)) {
            m_lightSlots[r.slot] = LightSlot{};
            m_freeLightSlots.push_back(r.slot);
            m_retiredLightSlots[i] = m_retiredLightSlots.back(); m_retiredLightSlots.pop_back();
        } else ++i;
    }
}

void VoxelStreamer::process_jobs() {
    for (size_t i = 0; i < m_jobs.size(); ) {
        std::shared_ptr<MeshJob>& job = m_jobs[i];
        const int st = job->state.load(std::memory_order_acquire);
        if (st == 0) { ++i; continue; }
        m_stats.meshMsAvg = m_stats.meshMsAvg <= 0.0f ? job->meshMs : m_stats.meshMsAvg + (job->meshMs - m_stats.meshMsAvg) * 0.05f;
        if (st == 1 || st == 3) {
            const int L = std::clamp((int)job->key.level, 0, MAX_LOD_LEVELS - 1);
            const double tris = st == 1 ? (double)job->gpu.triCount : 0.0;
            m_levelTriEst[L] += (tris - m_levelTriEst[L]) * 0.05;
        }
        auto it = m_chunks.find(job->packed);
        if (it == m_chunks.end() || it->second.version != job->version) {
            if (st == 1) free_gpu(job->gpu);
            if (job->stagingSlot >= 0) m_stagingSlots[(size_t)job->stagingSlot].inUse = false;
        } else {
            Chunk& c = it->second;
            if (st == 1) {
                c.state = State::Meshed;
                c.lightsDropped = job->lightsDropped;
            } else if (st == 3) {
                if (job->stagingSlot >= 0) m_stagingSlots[(size_t)job->stagingSlot].inUse = false;
                if (c.rebuilding) { retire_gpu(c.gpu, exclude_light_slot(c)); c.rebuilding = false; }
                c.gpu = GpuChunk{};
                c.state = State::Empty;
                c.lightsDropped = false;
                c.job.reset();
                set_resident(c, true);
                m_renderListChanged = true;
            } else {
                if (job->stagingSlot >= 0) m_stagingSlots[(size_t)job->stagingSlot].inUse = false;
                ++m_stats.allocFailures;
                c.state = State::Pending;
                c.retryFrame = m_frame + (job->buildPoolFull ? 3u : 20u);
                c.job.reset();
            }
        }
        job = m_jobs.back();
        m_jobs.pop_back();
    }
}

uint64_t VoxelStreamer::estimate_triangles() {
    const uint32_t count = (uint32_t)m_cut.nodes.size();
    const uint32_t pieces = piece_count(count, 2048);
    std::vector<double>   sums(pieces, 0.0);
    std::vector<uint32_t> leaves(pieces, 0), ready(pieces, 0);
    parallel_ranges(count, 2048, [&](uint32_t b, uint32_t e, uint32_t piece) {
        double sum = 0.0;
        uint32_t nLeaves = 0, nReady = 0;
        for (uint32_t i = b; i < e; ++i) {
            const LodCut::Node& n = m_cut.nodes[i];
            if (!n.leaf()) continue;
            ++nLeaves;
            if (const Chunk* c = static_cast<const Chunk*>(n.chunk)) {
                switch (c->state) {
                case State::Ready: case State::Empty: sum += (double)c->gpu.triCount; ++nReady; continue;
                case State::Uploading: case State::Compacting: sum += (double)c->next.triCount; continue;
                case State::Meshed: sum += c->job ? (double)c->job->gpu.triCount : 0.0; continue;
                default: if (c->rebuilding && c->gpu.valid()) { sum += (double)c->gpu.triCount; ++nReady; continue; } break;
                }
            }
            sum += m_levelTriEst[std::clamp((int)n.level, 0, MAX_LOD_LEVELS - 1)];
        }
        sums[piece] = sum; leaves[piece] = nLeaves; ready[piece] = nReady;
    });
    double total = 0.0;
    uint32_t nLeaves = 0, nReady = 0;
    for (uint32_t p = 0; p < pieces; ++p) { total += sums[p]; nLeaves += leaves[p]; nReady += ready[p]; }
    m_cutReadyFraction = nLeaves ? (double)nReady / (double)nLeaves : 1.0;
    return (uint64_t)total;
}

uint64_t VoxelStreamer::effective_budget() const {
    const double resident = (double)std::max<uint64_t>(m_stats.trianglesResident, 1);
    const double vertsPerTri = m_stats.trianglesResident > 100000 ? (double)m_stats.vertexUsed / resident : 2.3;
    const double blasPerTri  = m_stats.trianglesResident > 100000 ? (double)m_stats.blasUsed / resident : 72.0;
    double b = (double)m_cfg.triangleBudget;
    b = std::min(b, 0.85 * (double)m_cfg.vertexCapacity / std::max(vertsPerTri, 1.0));
    b = std::min(b, 0.85 * (double)m_cfg.indexCapacity / 3.0);
    b = std::min(b, 0.85 * (double)m_cfg.matIdCapacity);
    b = std::min(b, 0.85 * (double)m_cfg.blasPoolBytes / std::max(blasPerTri, 16.0));
    return (uint64_t)std::max(b, 1.0e5);
}

void VoxelStreamer::adapt_lod() {
    const uint64_t est = estimate_triangles();
    const uint64_t budget = effective_budget();
    m_stats.trianglesEstimated = est;
    m_stats.triangleBudget = budget;
    if (!m_cfg.adaptiveLod) { m_lodFactorNow = m_cfg.lodFactor; return; }
    const float maxF = std::max(m_cfg.lodFactor, m_cfg.lodFactorMin);
    float f = std::min(m_lodFactorNow, maxF);
    const double ratio = (double)budget / (double)std::max<uint64_t>(est, 1);
    const double ready = m_cutReadyFraction;
    if (est > budget) {
        f = std::max(m_cfg.lodFactorMin, f * (float)std::max(std::sqrt(ratio), 0.8));
    } else if (ready < 0.5 && m_frame > m_cutChangedFrame + 16u) {
        f = std::max(m_cfg.lodFactorMin, f * 0.8f);
    } else if ((double)est < 0.8 * (double)budget && ready > 0.9 && f < maxF) {
        const double step = std::min(std::sqrt(ratio), ready > 0.98 ? 1.25 : 1.1);
        f = std::min(maxF, f * (float)step);
    }
    m_lodFactorNow = f;
}

void VoxelStreamer::set_resident(Chunk& c, bool resident) {
    if (c.resident == resident) return;
    c.resident = resident;
    const int rootLevel = m_world->lod_tree().root_level();
    NodeKey k = c.key;
    while (k.level < rootLevel) {
        k = parent_key(k);
        uint32_t& n = m_residentBelow[pack_node(k)];
        if (resident) ++n;
        else if (n > 0) --n;
    }
}

void VoxelStreamer::adopt_cut(LodCut& cut, const double cam[3], float factor) {
    std::swap(m_cut, cut);
    m_cutValid = true;
    m_cutCam[0] = cam[0]; m_cutCam[1] = cam[1]; m_cutCam[2] = cam[2];
    m_cutLodFactor = factor;
    m_cutChangedFrame = m_frame;
    const uint32_t count = (uint32_t)m_cut.nodes.size();
    std::vector<std::vector<uint32_t>> missing(piece_count(count, 1024));
    parallel_ranges(count, 1024, [&](uint32_t b, uint32_t e, uint32_t piece) {
        for (uint32_t i = b; i < e; ++i) {
            LodCut::Node& n = m_cut.nodes[i];
            const auto it = m_chunks.find(n.key);
            if (it != m_chunks.end()) n.chunk = &it->second;
            else { n.chunk = nullptr; if (n.leaf()) missing[piece].push_back(i); }
        }
    });
    for (const std::vector<uint32_t>& list : missing) {
        for (uint32_t i : list) {
            LodCut::Node& n = m_cut.nodes[i];
            Chunk c;
            c.key = unpack_node(n.key);
            c.createdFrame = m_frame;
            n.chunk = &m_chunks.emplace(n.key, std::move(c)).first->second;
        }
    }
    parallel_ranges(count, 1024, [&](uint32_t b, uint32_t e, uint32_t) {
        for (uint32_t i = b; i < e; ++i) {
            const LodCut::Node& n = m_cut.nodes[i];
            if (Chunk* c = static_cast<Chunk*>(n.chunk)) {
                c->cutNode = i;
                if (n.leaf()) c->lastDesiredFrame = m_frame;
            }
        }
    });
    m_renderListChanged = true;
}

// Adopts the selected cut and schedules missing or stale chunks.
void VoxelStreamer::select_and_schedule() {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    const LodTree& tree = m_world->lod_tree();
    if (!m_cfg.adaptiveLod) m_lodFactorNow = m_cfg.lodFactor;
    m_lodFactorNow = std::clamp(m_lodFactorNow, std::min(m_cfg.lodFactorMin, m_cfg.lodFactor), std::max(m_cfg.lodFactor, m_cfg.lodFactorMin));
    if (m_cutFlatLevel != m_cfg.flatColorLevel) {
        if (m_cutFlatLevel >= 0) {
            const int lo = std::min(m_cutFlatLevel, m_cfg.flatColorLevel), hi = std::max(m_cutFlatLevel, m_cfg.flatColorLevel);
            for (auto& kv : m_chunks) {
                Chunk& c = kv.second;
                if (c.key.level >= lo && c.key.level < hi && (c.state == State::Ready || c.state == State::Empty)) {
                    c.version++; c.rebuilding = c.gpu.valid(); c.state = State::Pending; c.job.reset();
                }
            }
        }
        m_cutFlatLevel = m_cfg.flatColorLevel;
    }
    if (m_lightsApplied < 0) m_lightsApplied = m_cfg.lights ? 1 : 0;
    if (m_lightsApplied != (m_cfg.lights ? 1 : 0)) {
        m_lightsApplied = m_cfg.lights ? 1 : 0;
        for (auto& kv : m_chunks) {
            Chunk& c = kv.second;
            if (!m_cfg.lights) {
                if (c.gpu.light.valid()) {
                    retire_light(c.gpu.light, exclude_light_slot(c));
                    c.gpu.light = LightData{};
                    c.lightsDropped = true;
                    c.propsWritten = false;
                }
            } else if (c.lightsDropped && c.state == State::Ready && !c.job && c.key.level <= m_cfg.lightMaxLevel) {
                c.version++; c.rebuilding = c.gpu.valid(); c.state = State::Pending; c.job.reset(); c.retryFrame = 0; c.lightsDropped = false;
            }
        }
    }
    bool reselected = false;
    m_stats.adoptMs = m_stats.resolveMs = m_stats.adaptMs = 0.0f;
    if (m_selectJob && m_selectJob->state.load(std::memory_order_acquire) == 1) {
        const auto ta = clock::now();
        adopt_cut(m_selectJob->cut, m_selectJob->cam, m_selectJob->factor);
        m_selectJob.reset();
        reselected = true;
        if (m_jumpPending) {
            m_jumpPending = false;
            m_evictNow = true;
            for (const LodCut::Node& n : m_cut.nodes)
                if (Chunk* c = static_cast<Chunk*>(n.chunk)) c->retryFrame = 0;
        }
        m_stats.adoptMs = (float)std::chrono::duration<double, std::milli>(clock::now() - ta).count();
    }
    if (!m_selectJob && !m_cfg.freezeLod) {
        const double moved = std::sqrt((m_cam[0] - m_cutCam[0]) * (m_cam[0] - m_cutCam[0]) +
                                       (m_cam[1] - m_cutCam[1]) * (m_cam[1] - m_cutCam[1]) +
                                       (m_cam[2] - m_cutCam[2]) * (m_cam[2] - m_cutCam[2]));
        const double reselectDistance = std::max(8.0, (double)m_lodFactorNow / 16.0);
        const bool paramsChanged = m_cutLodFactor != m_lodFactorNow;
        if (!m_cutValid) {
            LodCut cut;
            tree.select(m_cam, m_lodFactorNow, cut, m_lodPool.get());
            adopt_cut(cut, m_cam, m_lodFactorNow);
            reselected = true;
        } else if (moved > reselectDistance || paramsChanged) {
            if (moved > (double)m_lodFactorNow) {
                m_lodFactorNow = m_cfg.lodFactorMin;
                m_jumpPending = true;
            }
            auto job = std::make_shared<SelectJob>();
            job->cam[0] = m_cam[0]; job->cam[1] = m_cam[1]; job->cam[2] = m_cam[2];
            job->factor = m_lodFactorNow;
            m_selectJob = job;
            planet::WorkerPool* pool = m_lodPool.get();
            pool->enqueue([&tree, job, pool]() {
                tree.select(job->cam, job->factor, job->cut, pool);
                job->state.store(1, std::memory_order_release);
            });
        }
    }
    const bool resolve = m_renderListChanged;
    m_renderListChanged = false;
    const auto tr = clock::now();
    if (resolve) {
        auto ready = [this](uint64_t k, void*& chunk) {
            Chunk* c = static_cast<Chunk*>(chunk);
            if (!c) {
                const auto it = m_chunks.find(k);
                if (it == m_chunks.end()) return false;
                c = &it->second;
                chunk = c;
            }
            return c->state == State::Ready || c->state == State::Empty || (c->rebuilding && c->gpu.valid());
        };
        auto residentBelow = [this](uint64_t k) {
            const auto it = m_residentBelow.find(k);
            return it != m_residentBelow.end() && it->second > 0;
        };
        tree.render_list(m_cut, ready, m_renderItems, residentBelow, m_lodPool.get());
        const uint32_t count = (uint32_t)m_renderItems.size();
        const bool sizeChanged = count != m_prevRenderKeys.size();
        std::vector<uint64_t> keys(count);
        m_render.resize(count);
        const uint32_t pieces = piece_count(count, 1024);
        std::vector<uint8_t>  differs(pieces, 0);
        std::vector<uint64_t> tris(pieces, 0);
        parallel_ranges(count, 1024, [&](uint32_t b, uint32_t e, uint32_t piece) {
            uint64_t sum = 0;
            bool d = false;
            for (uint32_t i = b; i < e; ++i) {
                const RenderItem& item = m_renderItems[i];
                Chunk& c = *static_cast<Chunk*>(item.chunk);
                keys[i] = item.key;
                if (!sizeChanged && m_prevRenderKeys[i] != item.key) d = true;
                c.lastRenderedFrame = m_frame;
                m_render[i] = RenderEntry{ item.key, &c };
                sum += c.gpu.triCount;
            }
            differs[piece] = d ? 1 : 0;
            tris[piece] = sum;
        });
        bool changed = sizeChanged;
        m_trianglesRendered = 0;
        for (uint32_t p = 0; p < pieces; ++p) { changed = changed || differs[p]; m_trianglesRendered += tris[p]; }
        m_renderListChanged = changed;
        m_prevRenderKeys.swap(keys);
        m_renderListFrame = m_frame;
        m_stats.resolveMs = (float)std::chrono::duration<double, std::milli>(clock::now() - tr).count();
    }
    if (reselected || (m_frame % 8) == 0) {
        const auto tp = clock::now();
        adapt_lod();
        m_stats.adaptMs = (float)std::chrono::duration<double, std::milli>(clock::now() - tp).count();
    }
    m_stats.cutNodes = (uint32_t)m_cut.nodes.size();
    m_stats.selectMs = (float)std::chrono::duration<double, std::milli>(clock::now() - t0).count();
}

void VoxelStreamer::dispatch_jobs() {
    const uint32_t nodeCount = (uint32_t)m_cut.nodes.size();
    const uint32_t nodePieces = piece_count(nodeCount, 2048);
    std::vector<std::vector<std::pair<float, uint64_t>>> parts(nodePieces);
    parallel_ranges(nodeCount, 2048, [&](uint32_t b, uint32_t e, uint32_t piece) {
        for (uint32_t i = b; i < e; ++i) {
            const LodCut::Node& n = m_cut.nodes[i];
            Chunk* c = static_cast<Chunk*>(n.chunk);
            if (!n.leaf() || !c) continue;
            if (c->state != State::Pending || c->job) continue;
            if (c->retryFrame > m_frame) continue;
            c->priority = priority_of(c->key);
            parts[piece].emplace_back(c->priority, n.key);
        }
    });
    std::vector<std::vector<uint64_t>> stale(piece_count((uint32_t)m_chunks.bucket_count(), 512));
    parallel_buckets([&](uint64_t key, const Chunk& c, uint32_t piece) {
        if (c.rebuilding && c.state == State::Pending && !c.job && c.retryFrame <= m_frame) stale[piece].push_back(key);
    });
    std::vector<std::pair<float, uint64_t>> cand;
    for (auto& part : parts) cand.insert(cand.end(), part.begin(), part.end());
    for (auto& list : stale) for (uint64_t k : list) cand.emplace_back(-1.0f, k);
    if (m_lightsBound && m_cfg.lights && (m_frame % 16) == 0) {
        uint64_t recUsed, recCap;
        { std::lock_guard<std::mutex> lk(m_poolMutex); recUsed = m_lightRecAlloc.used(); recCap = m_lightRecAlloc.capacity(); }
        if (recCap && recUsed * 10 < recCap * 7) {
            const uint32_t rc = (uint32_t)m_render.size();
            std::vector<std::vector<std::pair<float, uint64_t>>> dparts(piece_count(rc, 2048));
            parallel_ranges(rc, 2048, [&](uint32_t b, uint32_t e, uint32_t piece) {
                for (uint32_t i = b; i < e; ++i) {
                    const Chunk& c = *m_render[i].chunk;
                    if (!c.lightsDropped || c.state != State::Ready || c.job || c.key.level > m_cfg.lightMaxLevel) continue;
                    dparts[piece].emplace_back(priority_of(c.key), m_render[i].key);
                }
            });
            std::vector<std::pair<float, uint64_t>> dropped;
            for (auto& part : dparts) dropped.insert(dropped.end(), part.begin(), part.end());
            std::sort(dropped.begin(), dropped.end());
            for (size_t i = 0; i < dropped.size() && i < 4; ++i) {
                Chunk& c = m_chunks[dropped[i].second];
                c.version++; c.rebuilding = c.gpu.valid(); c.state = State::Pending; c.retryFrame = 0; c.lightsDropped = false;
                cand.emplace_back(-0.5f, dropped[i].second);
            }
        }
    }
    std::sort(cand.begin(), cand.end());
    const bool behind = m_cutReadyFraction < 0.9;
    const uint32_t limit = std::min((uint32_t)m_stagingSlots.size(), m_unlimitedBudget ? 0xFFFFFFFFu
        : (behind ? std::max(m_cfg.meshJobsInFlight, m_cfg.meshJobsInFlightBurst) : m_cfg.meshJobsInFlight));
    for (const auto& [prio, k] : cand) {
        if ((uint32_t)m_jobs.size() >= limit) break;
        Chunk& c = m_chunks[k];
        if (c.state != State::Pending || c.job) continue;
        const int slot = acquire_staging();
        if (slot < 0) break;
        auto job = std::make_shared<MeshJob>();
        job->key = c.key;
        job->packed = k;
        job->version = c.version;
        job->params.flatMaterials = c.key.level >= m_cfg.flatColorLevel;
        job->wantLights = m_cfg.lights && m_lightsBound && c.key.level <= m_cfg.lightMaxLevel;
        job->stagingSlot = slot;
        c.job = job;
        c.state = State::Meshing;
        m_jobs.push_back(job);
        m_workers->enqueue([this, job]() { run_job(job); });
    }
}

void VoxelStreamer::evict(bool immediate) {
    uint64_t vtxUsed, blasUsed;
    { std::lock_guard<std::mutex> lk(m_poolMutex); vtxUsed = m_vtxAlloc.used(); blasUsed = m_blasAlloc.used(); }
    const bool pressure = vtxUsed > (uint64_t)m_cfg.vertexCapacity * 85 / 100 || blasUsed > m_cfg.blasPoolBytes * 85 / 100;
    const uint32_t fuse = immediate ? 0u : (pressure ? std::min(m_cfg.evictFrames, 30u) : m_cfg.evictFrames);
    std::vector<std::vector<uint64_t>> parts(piece_count((uint32_t)m_chunks.bucket_count(), 512));
    parallel_buckets([&](uint64_t key, const Chunk& c, uint32_t piece) {
        const bool desired = c.lastDesiredFrame == m_cutChangedFrame;
        const bool rendered = c.lastRenderedFrame == m_renderListFrame;
        if (desired || rendered) return;
        const uint32_t idle = m_frame - std::max(c.lastRenderedFrame, std::max(c.lastDesiredFrame, c.createdFrame));
        bool drop = (c.state == State::Pending && !c.job)
            || ((c.state == State::Ready || c.state == State::Empty) && idle > fuse);
        if (immediate && (c.state == State::Meshing || c.state == State::Meshed)) drop = true;
        if (drop) parts[piece].push_back(key);
    });
    for (const std::vector<uint64_t>& list : parts) {
        for (uint64_t key : list) {
            auto it = m_chunks.find(key);
            if (it == m_chunks.end()) continue;
            Chunk& c = it->second;
            if (c.cutNode < m_cut.nodes.size() && m_cut.nodes[c.cutNode].chunk == &c) m_cut.nodes[c.cutNode].chunk = nullptr;
            set_resident(c, false);
            if (c.state == State::Meshed && c.job) {
                free_gpu(c.job->gpu);
                if (c.job->stagingSlot >= 0) m_stagingSlots[(size_t)c.job->stagingSlot].inUse = false;
                c.job.reset();
            }
            if (c.state == State::Ready || c.state == State::Empty) {
                retire_gpu(c.gpu, exclude_light_slot(c));
                release_instance_slot(c.instanceSlot);
                ++m_stats.evicted;
            } else if (c.gpu.valid()) {
                retire_gpu(c.gpu, exclude_light_slot(c));
                release_instance_slot(c.instanceSlot);
            }
            m_chunks.erase(it);
        }
    }
}

void VoxelStreamer::drop_far_lights() {
    if (!m_lightsBound) return;
    uint64_t used, cap;
    { std::lock_guard<std::mutex> lk(m_poolMutex); used = m_lightRecAlloc.used(); cap = m_lightRecAlloc.capacity(); }
    if (!cap || used * 10 < cap * 9) return;
    std::vector<std::vector<std::pair<float, uint64_t>>> parts(piece_count((uint32_t)m_chunks.bucket_count(), 512));
    parallel_buckets([&](uint64_t key, const Chunk& c, uint32_t piece) {
        if (!c.gpu.light.valid() || c.lightSlot != NONE || c.lastRenderedFrame == m_renderListFrame) return;
        if (c.state != State::Ready) return;
        parts[piece].emplace_back(-priority_of(c.key), key);
    });
    std::vector<std::pair<float, uint64_t>> farLights;
    for (auto& part : parts) farLights.insert(farLights.end(), part.begin(), part.end());
    std::sort(farLights.begin(), farLights.end());
    for (const auto& [negDist, k] : farLights) {
        if (used * 10 < cap * 8) break;
        Chunk& c = m_chunks[k];
        used -= c.gpu.light.recCount;
        retire_light(c.gpu.light, 0);
        c.gpu.light = LightData{};
        c.lightsDropped = true;
        c.propsWritten = false;
    }
}

void VoxelStreamer::update_stats() {
    if (!m_world || !m_vertexGlobal) {
        m_stats = StreamerStats{};
        m_statsCensusFrame = m_frame;
        return;
    }
    StreamerStats& s = m_stats;
    s.desired = (uint32_t)m_cut.leaves.size();
    s.rendered = (uint32_t)m_render.size();
    s.trianglesRendered = m_trianglesRendered;
    s.lodFactorNow = m_lodFactorNow;
    s.cutReadyFraction = (float)m_cutReadyFraction;
    {
        std::lock_guard<std::mutex> lk(m_poolMutex);
        const uint64_t vertexUsed = m_vtxAlloc.used(), indexUsed = m_idxAlloc.used(), matUsed = m_matAlloc.used();
        s.geometryBytesUsed = vertexUsed * sizeof(MeshVertex) + (indexUsed + matUsed) * sizeof(uint32_t);
        s.geometryBytesCapacity = m_vtxAlloc.capacity() * sizeof(MeshVertex) +
            (m_idxAlloc.capacity() + m_matAlloc.capacity()) * sizeof(uint32_t);
        s.blasBytesUsed = m_blasAlloc.used();
        s.blasBytesCapacity = m_blasAlloc.capacity();
        s.blasBuildBytesUsed = m_buildAlloc.used();
        s.blasBuildBytesCapacity = m_buildAlloc.capacity();
        s.lightRecUsed = m_lightRecAlloc.used();
        s.lightNodeUsed = m_lightNodeAlloc.used();
    }
    s.scratchBytesCapacity = 0;
    for (const auto& scratch : m_scratch)
        if (scratch) s.scratchBytesCapacity += scratch->GetDesc().Width;
    s.scratchBytesUsed = 0;
    for (uint64_t used : m_scratchUsed) s.scratchBytesUsed += used;
    s.lightSlotsActive = 0;
    s.lightChunksInTree = 0;
    s.lightTrisInTree = 0;
    s.lightNodesInTree = 0;
    for (const LightSlot& ls : m_lightSlots) {
        if (ls.active()) ++s.lightSlotsActive;
        const bool live = m_lightTlasHasVoxels && ls.light.valid()
            && ls.includedAt <= m_lightLiveVersion && m_lightLiveVersion < ls.excludedAt;
        if (!live) continue;
        ++s.lightChunksInTree;
        s.lightTrisInTree += ls.light.recCount;
        s.lightNodesInTree += ls.light.nodeCount;
    }
    s.lightVersion = m_lightVersion;
    s.lightLiveVersion = m_lightLiveVersion;
    if ((m_frame & 3u) != 0u || m_statsCensusFrame == m_frame) {
        s.censusAgeFrames = m_frame >= m_statsCensusFrame ? m_frame - m_statsCensusFrame : 0;
        return;
    }
    struct Census {
        uint32_t byState[7] = {};
        uint64_t resident = 0, lightResident = 0, lightNodesResident = 0;
        uint32_t residentBlas = 0, dropped = 0;
    };
    std::vector<Census> parts(piece_count((uint32_t)m_chunks.bucket_count(), 512));
    parallel_buckets([&](uint64_t, const Chunk& c, uint32_t piece) {
        Census& cs = parts[piece];
        ++cs.byState[(int)c.state];
        if (c.resident) {
            cs.resident += c.gpu.triCount;
            if (c.gpu.valid()) ++cs.residentBlas;
            cs.lightResident += c.gpu.light.recCount;
            cs.lightNodesResident += c.gpu.light.nodeCount;
        }
        if (c.lightsDropped) ++cs.dropped;
    });
    Census total;
    for (const Census& cs : parts) {
        for (int i = 0; i < 7; ++i) total.byState[i] += cs.byState[i];
        total.resident += cs.resident;
        total.lightResident += cs.lightResident;
        total.lightNodesResident += cs.lightNodesResident;
        total.residentBlas += cs.residentBlas;
        total.dropped += cs.dropped;
    }
    s.pending = total.byState[(int)State::Pending];   s.meshing = total.byState[(int)State::Meshing];
    s.meshed = total.byState[(int)State::Meshed];     s.uploading = total.byState[(int)State::Uploading] + total.byState[(int)State::Compacting];
    s.ready = total.byState[(int)State::Ready];       s.empty = total.byState[(int)State::Empty];
    s.trianglesResident = total.resident;
    s.geometryBlasResident = total.residentBlas;
    s.lightTrisResident = total.lightResident;
    s.lightNodesResident = total.lightNodesResident;
    s.lightChunksDropped = total.dropped;
    s.chunksTracked = (uint32_t)m_chunks.size();
    {
        // Keep the LOD estimator's pool usage paired with its triangle census.
        std::lock_guard<std::mutex> lk(m_poolMutex);
        s.vertexUsed = m_vtxAlloc.used();
        s.indexUsed = m_idxAlloc.used();
        s.matIdUsed = m_matAlloc.used();
        s.blasUsed = m_blasAlloc.used();
        s.blasBuildUsed = m_buildAlloc.used();
    }
    m_statsCensusFrame = m_frame;
    s.censusAgeFrames = 0;
}

void VoxelStreamer::begin_frame(const double camWorld[3]) {
    if (!m_world || !m_vertexGlobal) { update_stats(); return; }
    ++m_frame;
    m_placement.to_blocks(camWorld, m_cam);
    reclaim();
    process_jobs();
    select_and_schedule();
    if (m_evictNow) { evict(true); m_evictNow = false; }
    else if ((m_frame & 7u) == 0u) evict(false);
    drop_far_lights();
    if (m_cutChangedFrame == m_frame || (m_frame & 3u) == 0u) dispatch_jobs();
    update_stats();
}

// Records uploads, BLAS builds, compaction, and light-tree updates.
void VoxelStreamer::record_gpu_work(ID3D12GraphicsCommandList* copyList, ID3D12GraphicsCommandList4* computeList) {
    m_stats.buildsThisFrame = 0;
    m_stats.copiesThisFrame = 0;
    m_stats.uploadBytesThisFrame = 0;
    m_uploadingThisFrame.clear();
    if (!m_world || !m_vertexGlobal) { update_stats(); return; }
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    std::vector<std::pair<float, uint64_t>> ready;
    for (auto& kv : m_chunks)
        if (kv.second.state == State::Meshed && kv.second.job) ready.emplace_back(kv.second.priority, kv.first);
    std::sort(ready.begin(), ready.end());

    const uint32_t slot = m_ctx->FrameIndex() % (uint32_t)m_scratch.size();
    uint64_t& scratchUsed = m_scratchUsed[slot];
    scratchUsed = 0;
    const D3D12_GPU_VIRTUAL_ADDRESS scratchVa = m_scratch[slot]->GetGPUVirtualAddress();
    if (m_cfg.blasCompaction) {
        m_compactCount[slot] = 0;
        if (m_compactInfoCopyState[slot]) {
            D3D12_RESOURCE_BARRIER tr = {};
            tr.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            tr.Transition.pResource = m_compactInfo[slot].Get();
            tr.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            tr.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
            tr.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            computeList->ResourceBarrier(1, &tr);
            m_compactInfoCopyState[slot] = 0;
        }
    }
    const bool behind = m_cutReadyFraction < 0.9;
    const uint32_t budget = m_unlimitedBudget ? 0xFFFFFFFFu : (behind ? std::max(m_cfg.buildBudget, m_cfg.buildBudgetBurst) : m_cfg.buildBudget);

    for (const auto& [prio, k] : ready) {
        if (m_stats.buildsThisFrame >= budget) break;
        if (m_cfg.blasCompaction && m_compactCount[slot] >= COMPACT_INFO_ENTRIES) break;
        Chunk& c = m_chunks[k];
        MeshJob& job = *c.job;
        if (scratchUsed + job.scratchSize > m_cfg.scratchBytesPerFrame) {
            if (scratchUsed == 0) {
                std::printf("[mc] chunk needs %llu MB of BLAS scratch, over the %llu MB ring - skipped\n",
                            (unsigned long long)(job.scratchSize >> 20), (unsigned long long)(m_cfg.scratchBytesPerFrame >> 20));
                free_gpu(job.gpu);
                if (job.stagingSlot >= 0) m_stagingSlots[(size_t)job.stagingSlot].inUse = false;
                c.job.reset();
                c.state = State::Empty;
                set_resident(c, true);
                m_renderListChanged = true;
                continue;
            }
            break;
        }
        copyList->CopyBufferRegion(m_vertexGlobal, job.gpu.vtxOff * sizeof(MeshVertex), job.srcBuffer, job.srcVtxOff, job.vtxBytes);
        copyList->CopyBufferRegion(m_indexGlobal,  job.gpu.idxOff * sizeof(uint32_t),   job.srcBuffer, job.srcIdxOff, job.idxBytes);
        m_stats.copiesThisFrame += 2;
        m_stats.uploadBytesThisFrame += job.vtxBytes + job.idxBytes;
        if (job.gpu.light.valid()) {
            if (m_lightsBound && job.gpu.light.gen == m_lightGen) {
                const LightData& l = job.gpu.light;
                copyList->CopyBufferRegion(m_lights.records,   ((uint64_t)m_lights.recordBase + l.recOff) * sizeof(LightTriangle),          job.srcBuffer, job.srcRecOff,   job.recBytes);
                copyList->CopyBufferRegion(m_lights.nodes,     ((uint64_t)m_lights.nodeBase + l.nodeOff) * lt::LightBLASNodeStride(l.compactNodes), job.srcBuffer, job.srcNodeOff, job.nodeBytes);
                copyList->CopyBufferRegion(m_lights.leafIndex, ((uint64_t)m_lights.recordBase + l.recOff) * sizeof(uint32_t),               job.srcBuffer, job.srcLeafOff,  job.leafBytes);
                copyList->CopyBufferRegion(m_lights.trails,    ((uint64_t)m_lights.recordBase + l.recOff) * sizeof(lt::LightTreeTrail),      job.srcBuffer, job.srcTrailOff, job.trailBytes);
                m_stats.copiesThisFrame += 4;
                m_stats.uploadBytesThisFrame += job.recBytes + job.nodeBytes + job.leafBytes + job.trailBytes;
            } else {
                job.gpu.light = LightData{};
                c.lightsDropped = true;
            }
        }

        D3D12_RAYTRACING_GEOMETRY_DESC gd[2] = {};
        D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC ommTri = {};
        D3D12_RAYTRACING_GEOMETRY_OMM_LINKAGE_DESC ommLink = {};
        uint32_t geomCount = 0;
        uint64_t idxElem = job.gpu.idxOff;
        auto geom = [&](uint32_t triCount, bool opaque, bool omm) {
            D3D12_RAYTRACING_GEOMETRY_DESC& g = gd[geomCount++];
            D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC& tri = omm ? ommTri : g.Triangles;
            g.Type  = omm ? D3D12_RAYTRACING_GEOMETRY_TYPE_OMM_TRIANGLES : D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
            g.Flags = opaque ? D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE : D3D12_RAYTRACING_GEOMETRY_FLAG_NONE;
            tri.VertexBuffer.StartAddress  = m_vertexVa;
            tri.VertexBuffer.StrideInBytes = sizeof(MeshVertex);
            tri.VertexCount  = m_combinedVertexCount;
            tri.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
            tri.IndexBuffer  = m_indexVa + idxElem * sizeof(uint32_t);
            tri.IndexCount   = triCount * 3;
            tri.IndexFormat  = DXGI_FORMAT_R32_UINT;
            if (omm) {
                ommLink.OpacityMicromapIndexBuffer.StartAddress  = m_ommIndexVa + job.gpu.ommOff * sizeof(int32_t);
                ommLink.OpacityMicromapIndexBuffer.StrideInBytes = sizeof(int32_t);
                ommLink.OpacityMicromapIndexFormat  = DXGI_FORMAT_R32_UINT;
                ommLink.OpacityMicromapBaseLocation = 0;
                ommLink.OpacityMicromapArray        = m_ommArrayVa;
                g.OmmTriangles.pTriangles  = &ommTri;
                g.OmmTriangles.pOmmLinkage = &ommLink;
            }
            idxElem += (uint64_t)triCount * 3;
        };
        if (job.gpu.opaqueTriCount) geom(job.gpu.opaqueTriCount, true, false);
        const uint32_t alphaTris = job.gpu.triCount - job.gpu.opaqueTriCount;
        if (alphaTris) geom(alphaTris, false, job.gpu.ommCount != 0);

        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC desc = {};
        desc.Inputs.Type           = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        desc.Inputs.DescsLayout    = D3D12_ELEMENTS_LAYOUT_ARRAY;
        desc.Inputs.Flags          = build_flags();
        desc.Inputs.NumDescs       = geomCount;
        desc.Inputs.pGeometryDescs = gd;
        desc.DestAccelerationStructureData    = m_cfg.blasCompaction ? job.gpu.buildVa : job.gpu.blasVa;
        desc.ScratchAccelerationStructureData = scratchVa + scratchUsed;
        if (m_cfg.blasCompaction) {
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC pb = {};
            pb.DestBuffer = m_compactInfo[slot]->GetGPUVirtualAddress() + (uint64_t)m_compactCount[slot] * sizeof(uint64_t);
            pb.InfoType   = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE;
            computeList->BuildRaytracingAccelerationStructure(&desc, 1, &pb);
            c.compactSlot = slot;
            c.compactIndex = m_compactCount[slot]++;
            c.compactQueued = false;
        } else {
            computeList->BuildRaytracingAccelerationStructure(&desc, 0, nullptr);
        }
        scratchUsed += job.scratchSize;

        c.next = job.gpu;
        c.state = State::Uploading;
        c.uploadFence = 0;
        m_uploadingThisFrame.push_back(k);
        m_uploading.push_back(k);
        ++m_stats.buildsThisFrame;
    }
    if (m_stats.buildsThisFrame) {
        D3D12_RESOURCE_BARRIER uav = {};
        uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        uav.UAV.pResource = m_cfg.blasCompaction ? m_buildPool.Get() : m_blasPool.Get();
        computeList->ResourceBarrier(1, &uav);
        if (m_cfg.blasCompaction) {
            D3D12_RESOURCE_BARRIER tr = {};
            tr.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            tr.Transition.pResource = m_compactInfo[slot].Get();
            tr.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            tr.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            tr.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
            computeList->ResourceBarrier(1, &tr);
            computeList->CopyBufferRegion(m_compactReadback[slot].Get(), 0, m_compactInfo[slot].Get(), 0,
                                          (uint64_t)m_compactCount[slot] * sizeof(uint64_t));
            m_compactInfoCopyState[slot] = 1;
        }
    }
    m_stats.compactionsThisFrame = 0;
    for (uint64_t k : m_toCompact) {
        auto it = m_chunks.find(k);
        if (it == m_chunks.end() || it->second.state != State::Uploading || !it->second.compactQueued) continue;
        Chunk& c = it->second;
        computeList->CopyRaytracingAccelerationStructure(c.next.blasVa, c.next.buildVa,
                                                         D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT);
        c.state = State::Compacting;
        c.compactFence = 0;
        m_compactingThisFrame.push_back(k);
        ++m_stats.compactionsThisFrame;
    }
    m_toCompact.clear();
    if (m_stats.compactionsThisFrame) {
        D3D12_RESOURCE_BARRIER uav = {};
        uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        uav.UAV.pResource = m_blasPool.Get();
        computeList->ResourceBarrier(1, &uav);
    }
    m_stats.recordMs = (float)std::chrono::duration<double, std::milli>(clock::now() - t0).count();
}

void VoxelStreamer::append_instances(planet::TlasBuilder& tlas, InstanceProperties* props, const planet::DVec3& sceneOrigin,
                                     uint32_t hitGroup, bool& forceRebuild) {
    if (!m_world || !m_vertexGlobal) return;
    const bool originChanged = sceneOrigin.x != m_lastOrigin.x || sceneOrigin.y != m_lastOrigin.y || sceneOrigin.z != m_lastOrigin.z;
    m_lastOrigin = sceneOrigin;
    if (m_renderListChanged || originChanged) forceRebuild = true;
    if (originChanged) m_lightSetDirty = true;

    m_stats.geometryBlasRendered = 0;
    m_stats.trianglesInTlas = 0;
    for (const RenderEntry& e : m_render) {
        Chunk& c = *e.chunk;
        if (!c.gpu.valid()) continue;
        if (c.instanceSlot == NONE) {
            c.instanceSlot = acquire_instance_slot();
            if (c.instanceSlot == NONE) continue;
            c.propsWritten = false;
        }
        int64_t o[3];
        node_origin_blocks(c.key, o);
        const double ob[3] = { (double)o[0], (double)o[1], (double)o[2] };
        double os[3];
        m_placement.to_scene(ob, os);
        const float tx = (float)(os[0] - sceneOrigin.x);
        const float ty = (float)(os[1] - sceneOrigin.y);
        const float tz = (float)(os[2] - sceneOrigin.z);
        const double* A = m_placement.a;
        const float xform[12] = { (float)A[0], (float)A[1], (float)A[2], tx,
                                  (float)A[3], (float)A[4], (float)A[5], ty,
                                  (float)A[6], (float)A[7], (float)A[8], tz };
        const uint32_t instId = m_propsBase + c.instanceSlot;
        const D3D12_RAYTRACING_INSTANCE_FLAGS flags = (c.gpu.opaqueTriCount == c.gpu.triCount)
            ? D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_OPAQUE : D3D12_RAYTRACING_INSTANCE_FLAG_NONE;
        const uint32_t hg = (c.gpu.opaqueTriCount == 0 && c.gpu.triCount) ? hitGroup + 1u : hitGroup;
        tlas.add_instance(c.gpu.blasVa, xform, instId, hg, flags);
        ++m_stats.geometryBlasRendered;
        m_stats.trianglesInTlas += c.gpu.triCount;
        const uint32_t wantSlot = (c.lightSlot != NONE && light_slot_live(c.lightSlot)) ? m_lights.slotBase + c.lightSlot : NONE;
        const bool full = !c.propsWritten || originChanged;
        if (props && (full || c.propsLightSlot != wantSlot)) {
            using namespace DirectX;
            InstanceProperties& p = props[instId];
            if (full) {
                const XMMATRIX M = placement_matrix(tx, ty, tz);
                XMVECTOR det;
                const XMMATRIX Minv = XMMatrixInverse(&det, M);
                XMMATRIX upper = M;
                upper.r[3] = XMVectorSet(0.0f, 0.0f, 0.0f, 1.0f);
                p.objectToWorld        = MakeFloat3x4(M);
                p.objectToWorldInverse = MakeFloat3x4(Minv);
                p.objectToWorldNormal  = MakeFloat3x4(m_placement.identity ? XMMatrixIdentity() : XMMatrixTranspose(XMMatrixInverse(&det, upper)));
                p.prevObjectToWorld    = MakeFloat3x4(M);
                p.indexBase      = (UINT)c.gpu.idxOff;
                p.vertexBase     = (UINT)c.gpu.vtxOff;
                p.materialBase   = (UINT)c.gpu.matOff;
                p.opaqueTriCount = c.gpu.opaqueTriCount;
                if (c.gpu.light.valid()) {
                    p.triToLightBase = LIGHT_DENSE_FLAG | (m_lights.recordBase + (uint32_t)c.gpu.light.recOff);
                    p._pad[0] = c.gpu.light.litOpaque;
                    p._pad[1] = c.gpu.light.litAlpha;
                } else {
                    p.triToLightBase = NONE;
                    p._pad[0] = 0u; p._pad[1] = 0u;
                }
            }
            p.lightSlot = wantSlot;
            c.propsLightSlot = wantSlot;
            c.propsWritten = true;
        }
    }
    if (m_renderListFrame == m_frame) {
        for (auto& kv : m_chunks) {
            Chunk& c = kv.second;
            if (c.instanceSlot != NONE && c.lastRenderedFrame != m_renderListFrame) {
                release_instance_slot(c.instanceSlot);
                c.instanceSlot = NONE;
            }
        }
    }
    update_light_set(m_lightLiveVersion);
    update_stats();
}

void VoxelStreamer::update_light_set(uint32_t) {
    if (m_lightsBound && m_cfg.lights) {
        const uint32_t rc = (uint32_t)m_render.size();
        std::vector<std::vector<std::pair<float, uint64_t>>> parts(piece_count(rc, 2048));
        parallel_ranges(rc, 2048, [&](uint32_t b, uint32_t e, uint32_t piece) {
            for (uint32_t i = b; i < e; ++i) {
                const Chunk& c = *m_render[i].chunk;
                if (!c.gpu.valid() || !c.gpu.light.valid() || c.instanceSlot == NONE) continue;
                if (c.key.level > m_cfg.lightMaxLevel) continue;
                parts[piece].emplace_back(priority_of(c.key), m_render[i].key);
            }
        });
        std::vector<std::pair<float, uint64_t>> cand;
        for (auto& part : parts) cand.insert(cand.end(), part.begin(), part.end());
        std::sort(cand.begin(), cand.end());
        uint64_t total = 0;
        for (const auto& [dist, k] : cand) {
            Chunk& c = m_chunks[k];
            if (total + c.gpu.light.recCount > m_cfg.maxLightTris) break;
            const uint32_t instId = m_propsBase + c.instanceSlot;
            if (c.lightSlot != NONE && m_lightSlots[c.lightSlot].instanceID != instId) exclude_light_slot(c);
            if (c.lightSlot == NONE) {
                uint32_t s = NONE;
                if (m_lightSlots.size() < m_cfg.maxLightSlots) { s = (uint32_t)m_lightSlots.size(); m_lightSlots.emplace_back(); }
                else if (!m_freeLightSlots.empty()) { s = m_freeLightSlots.front(); m_freeLightSlots.pop_front(); }
                if (s == NONE) continue;
                LightSlot& ls = m_lightSlots[s];
                ls.chunk = k;
                ls.instanceID = instId;
                ls.includedAt = m_lightVersion + 1;
                ls.excludedAt = NONE;
                node_origin_blocks(c.key, ls.origin);
                ls.light = c.gpu.light;
                c.lightSlot = s;
                m_lightSetDirty = true;
            }
            c.lightIncludedFrame = m_frame;
            total += c.gpu.light.recCount;
        }
    }
    for (uint32_t s = 0; s < (uint32_t)m_lightSlots.size(); ++s) {
        LightSlot& ls = m_lightSlots[s];
        if (!ls.active()) continue;
        auto it = m_chunks.find(ls.chunk);
        if (it == m_chunks.end() || it->second.lightSlot != s) {
            ls.excludedAt = m_lightVersion + 1;
            m_retiredLightSlots.push_back(RetiredLightSlot{ s, m_frame + m_cfg.retireFrames, ls.excludedAt });
            m_lightSetDirty = true;
            continue;
        }
        if (it->second.lightIncludedFrame != m_frame) exclude_light_slot(it->second);
    }
    if (m_lightSetDirty) { ++m_lightVersion; m_lightSetDirty = false; }
}

// Records copy and compute fences for pending uploads and builds.
void VoxelStreamer::on_submitted(uint64_t copyFence, uint64_t computeFence) {
    m_lastCopyFence = copyFence;
    m_lastComputeFence = computeFence;
    for (uint64_t k : m_uploadingThisFrame) {
        auto it = m_chunks.find(k);
        if (it == m_chunks.end()) continue;
        Chunk& c = it->second;
        c.uploadFence = computeFence;
        if (c.job) {
            if (c.job->stagingSlot >= 0) release_staging_after(c.job->stagingSlot, computeFence);
            if (c.job->privateUpload) m_retiredUploads.push_back(RetiredUpload{ c.job->privateUpload, computeFence });
            c.job.reset();
        }
    }
    m_uploadingThisFrame.clear();
    for (uint64_t k : m_compactingThisFrame) {
        auto it = m_chunks.find(k);
        if (it != m_chunks.end()) it->second.compactFence = computeFence;
    }
    m_compactingThisFrame.clear();
}

void VoxelStreamer::calibrate_estimates() {
    LodCut cut;
    m_world->lod_tree().select(m_cam, std::max(m_cfg.lodFactor, m_cfg.lodFactorMin), cut, m_lodPool.get());
    std::vector<uint64_t> perLevel[MAX_LOD_LEVELS];
    for (uint64_t k : cut.leafList) perLevel[std::clamp((int)unpack_node(k).level, 0, MAX_LOD_LEVELS - 1)].push_back(k);
    for (int L = 0; L < MAX_LOD_LEVELS; ++L) {
        const std::vector<uint64_t>& keys = perLevel[L];
        if (keys.empty()) continue;
        const size_t n = std::min<size_t>(keys.size(), 48);
        std::vector<uint64_t> sample(n);
        for (size_t i = 0; i < n; ++i) sample[i] = keys[i * keys.size() / n];
        std::vector<uint32_t> tris(n, 0);
        MeshParams params;
        params.flatMaterials = L >= m_cfg.flatColorLevel;
        m_workers->parallel_for((uint32_t)n, [&](uint32_t i) {
            ChunkMesh mesh;
            thread_mesher().mesh(unpack_node(sample[i]), params, mesh);
            tris[i] = mesh.triangle_count();
        });
        double sum = 0.0;
        for (uint32_t t : tris) sum += (double)t;
        m_levelTriEst[L] = sum / (double)n;
    }
}

void VoxelStreamer::warm_up(const double camWorld[3]) {
    if (!m_world || !m_vertexGlobal || !m_cfg.warmUp) return;
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    m_unlimitedBudget = true;
    m_placement.to_blocks(camWorld, m_cam);
    calibrate_estimates();
    for (int i = 0; i < 24; ++i) {
        const float before = m_lodFactorNow;
        m_world->lod_tree().select(m_cam, m_lodFactorNow, m_cut, m_lodPool.get());
        adapt_lod();
        if (m_lodFactorNow == before) break;
    }
    m_cutValid = false;
    m_selectJob.reset();
    std::printf("[mc] warm-up: detail distance %.0f blocks for %.1fM estimated triangles (budget %.1fM; per chunk L0 %.0f, L1 %.0f, L2 %.0f, L3 %.0f)\n",
                m_lodFactorNow * 2.0f, (double)m_stats.trianglesEstimated / 1.0e6, (double)m_stats.triangleBudget / 1.0e6,
                m_levelTriEst[0], m_levelTriEst[1], m_levelTriEst[2], m_levelTriEst[3]);
    for (int iter = 0; ; ++iter) {
        begin_frame(camWorld);
        m_ctx->ResetPlanetLists();
        record_gpu_work(m_ctx->CopyList(), m_ctx->ComputeList());
        const uint64_t copyVal = m_ctx->SubmitPlanetCopy();
        const uint64_t cv = m_ctx->SubmitPlanetCompute(copyVal);
        on_submitted(copyVal, cv);
        m_ctx->PlanetComputeCpuWait(cv);
        bool allReady = true;
        for (uint64_t k : m_cut.leafList) {
            const auto it = m_chunks.find(k);
            if (it == m_chunks.end() || (it->second.state != State::Ready && it->second.state != State::Empty)) { allReady = false; break; }
        }
        const double elapsed = std::chrono::duration<double>(clock::now() - t0).count();
        if (allReady || elapsed > m_cfg.warmUpSeconds) {
            m_stats.warmUpSeconds = elapsed;
            m_stats.warmUpComplete = allReady;
            std::printf("[mc] warm-up: %u chunks resident (%u empty), %llu triangles, %llu light triangles, %.1f s%s\n",
                        m_stats.ready, m_stats.empty, (unsigned long long)m_stats.trianglesRendered,
                        (unsigned long long)m_stats.lightTrisResident, elapsed,
                        allReady ? "" : " (time budget hit, streaming continues)");
            break;
        }
        if (m_stats.buildsThisFrame == 0) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    m_unlimitedBudget = false;
    m_prevRenderKeys.clear();
    m_renderListChanged = true;
}

void VoxelStreamer::set_block(int x, int y, int z, BlockId id) {
    if (!m_world) return;
    std::vector<uint64_t> stale;
    m_world->store().set_block(x, y, z, id, m_world->registry(), stale);
    for (uint64_t k : stale) {
        if (m_world->store().chunk_occupied(unpack_node(k))) m_world->lod_tree_mut().add_occupied(k);
        auto it = m_chunks.find(k);
        if (it == m_chunks.end()) continue;
        Chunk& c = it->second;
        c.version++;
        c.retryFrame = 0;
        switch (c.state) {
        case State::Uploading:
        case State::Compacting:
            c.remeshAfterUpload = true;
            continue;
        case State::Meshed:
            if (c.job) {
                free_gpu(c.job->gpu);
                if (c.job->stagingSlot >= 0) m_stagingSlots[(size_t)c.job->stagingSlot].inUse = false;
            }
            break;
        case State::Ready:
        case State::Empty:
            c.rebuilding = c.gpu.valid();
            break;
        default:
            break;
        }
        c.job.reset();
        c.state = State::Pending;
    }
}

}
