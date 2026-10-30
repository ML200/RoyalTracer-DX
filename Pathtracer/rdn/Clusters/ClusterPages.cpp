#include "ClusterPages.h"
#include "../Core/DeviceContext.h"
#include "../DXRHelper.h"
#include <algorithm>
#include <cfloat>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <unordered_map>
#include <DirectXPackedVector.h>

namespace clusters {

namespace {
constexpr char MAGIC[8] = { 'R', 'D', 'N', 'P', 'A', 'G', 'E', 'S' };
constexpr uint32_t VERSION = 1;
// File: header, LodGroup per group, FilePage per group, FileCluster per hierarchy cluster (by offset), the deps, then
// the pages: BTriVertex per page vertex, then three page-local uint16 indices per triangle.
struct FileHeader {
    char magic[8];
    uint32_t version, groups, clusters, deps;
    uint64_t dataOffset;
};
struct FilePage {
    uint64_t offset;
    uint32_t bytes, firstCluster, clusterCount, vertexCount, triCount, depFirst, depCount, pad;
};
struct FileCluster {
    uint32_t group, refined, firstTri, material;
    uint16_t triCount, uniqueVerts;
    uint32_t pad;
};
static_assert(sizeof(FileHeader) == 32 && sizeof(FilePage) == 40 && sizeof(FileCluster) == 24);
static_assert(sizeof(BTriVertex) == 20);
#if RDN_CLUSTER_API
constexpr uint64_t CLAS_UNIT = D3D12_RAYTRACING_CLAS_BYTE_ALIGNMENT;
#endif

uint64_t env_u64(const char* name, uint64_t fallback) {
    char v[32] = {};
    return GetEnvironmentVariableA(name, v, sizeof(v)) > 0 ? std::strtoull(v, nullptr, 10) : fallback;
}

bool read_at(HANDLE file, uint64_t offset, void* dst, uint32_t bytes) {
    OVERLAPPED ov = {};
    ov.Offset = (DWORD)offset;
    ov.OffsetHigh = (DWORD)(offset >> 32);
    DWORD got = 0;
    return ReadFile(file, dst, bytes, &got, &ov) && got == bytes;
}

HANDLE open_read(const std::string& path) {
    return CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
}
} // namespace

void RangeAllocator::init(uint64_t capacity) {
    m_capacity = capacity;
    m_used = 0;
    m_free.assign(capacity ? 1 : 0, { 0ull, capacity });
}

bool RangeAllocator::allocate(uint64_t count, uint64_t& offset) {
    if (!count) { offset = 0; return true; }
    for (size_t i = 0; i < m_free.size(); ++i) {
        auto& r = m_free[i];
        if (r.second < count) continue;
        offset = r.first;
        r.first += count;
        r.second -= count;
        if (!r.second) m_free.erase(m_free.begin() + (ptrdiff_t)i);
        m_used += count;
        return true;
    }
    return false;
}

void RangeAllocator::free(uint64_t offset, uint64_t count) {
    if (!count) return;
    auto it = std::lower_bound(m_free.begin(), m_free.end(), std::make_pair(offset, 0ull));
    it = m_free.insert(it, { offset, count });
    if (it + 1 != m_free.end() && it->first + it->second == (it + 1)->first) { it->second += (it + 1)->second; m_free.erase(it + 1); }
    if (it != m_free.begin() && (it - 1)->first + (it - 1)->second == it->first) { (it - 1)->second += it->second; m_free.erase(it); }
    m_used -= count;
}

bool paging_enabled() {
    static const bool off = [] {
        char v[4] = {};
        return GetEnvironmentVariableA("RDN_PAGING", v, sizeof(v)) > 0 && v[0] == '0';
    }();
    return lod_enabled() && !off;
}

std::string cache_dir() {
    char v[MAX_PATH] = {};
    if (GetEnvironmentVariableA("RDN_GEOMETRY_CACHE", v, sizeof(v)) > 0) return v;
    if (GetEnvironmentVariableA("LOCALAPPDATA", v, sizeof(v)) > 0) return std::string(v) + "\\RoyalTracer\\GeometryCache";
    return "GeometryCache";
}

PoolSizes pool_sizes() {
    PoolSizes s;
    s.vertices = (uint32_t)env_u64("RDN_PAGE_POOL_VERTS", 4u << 20);
    s.tris = (uint32_t)env_u64("RDN_PAGE_POOL_TRIS", 4u << 20);
    s.clasBytes = env_u64("RDN_PAGE_POOL_CLAS_MB", 512) << 20;
    return s;
}

bool write_pages(const std::string& path, const LodBuild& b, const BakeMesh& mesh) {
    using namespace DirectX;
    const uint32_t groups = (uint32_t)b.groups.size(), clusters = (uint32_t)b.clusters.size();
    std::vector<std::vector<uint32_t>> members(groups), deps(groups);
    for (uint32_t c = 0; c < clusters; ++c) {
        members[b.clusters[c].group].push_back(c);
        if (b.clusters[c].refined != LOD_NONE) deps[b.clusters[c].refined].push_back(b.clusters[c].group);
    }
    for (auto& d : deps) {
        std::sort(d.begin(), d.end());
        d.erase(std::unique(d.begin(), d.end()), d.end());
    }
    // Roots first: they load together at startup, and their clusters make the coarsest cut, [fixed, base).
    std::vector<uint32_t> order(groups);
    std::iota(order.begin(), order.end(), 0u);
    std::stable_partition(order.begin(), order.end(), [&](uint32_t g) { return b.groups[g].error == FLT_MAX; });

    std::vector<FilePage> pages(groups);
    std::vector<FileCluster> fileClusters(clusters);
    std::vector<uint32_t> depList;
    std::vector<uint8_t> data;
    std::unordered_map<uint32_t, uint16_t> local;
    std::vector<uint32_t> verts;
    uint32_t offset = 0;
    for (uint32_t g : order) {
        FilePage& p = pages[g];
        p = {};
        p.offset = data.size();
        p.firstCluster = offset;
        p.clusterCount = (uint32_t)members[g].size();
        p.depFirst = (uint32_t)depList.size();
        p.depCount = (uint32_t)deps[g].size();
        depList.insert(depList.end(), deps[g].begin(), deps[g].end());
        local.clear();
        verts.clear();
        std::vector<uint16_t> idx;
        for (uint32_t c : members[g]) {
            const Span& s = b.spans[c];
            FileCluster& fc = fileClusters[offset++];
            fc = {};
            fc.group = g;
            fc.refined = b.clusters[c].refined;
            fc.firstTri = (uint32_t)(idx.size() / 3);
            fc.material = b.regions[s.firstTri] - mesh.materialBase;
            fc.triCount = s.triCount;
            fc.uniqueVerts = s.uniqueVerts;
            for (uint32_t i = 3 * s.firstTri; i < 3 * (s.firstTri + s.triCount); ++i) {
                const auto [it, fresh] = local.emplace(b.indices[i], (uint16_t)verts.size());
                if (fresh) {
                    if (verts.size() == 0xFFFF) return false;
                    verts.push_back(b.indices[i]);
                }
                idx.push_back(it->second);
            }
        }
        p.vertexCount = (uint32_t)verts.size();
        p.triCount = (uint32_t)(idx.size() / 3);
        const size_t at = data.size();
        data.resize(at + verts.size() * sizeof(BTriVertex) + idx.size() * sizeof(uint16_t));
        auto* out = reinterpret_cast<BTriVertex*>(data.data() + at);
        for (size_t i = 0; i < verts.size(); ++i) {
            const uint8_t* v = mesh.vertices + (size_t)verts[i] * mesh.stride;
            XMFLOAT3 pos, n;
            XMFLOAT2 uv;
            std::memcpy(&pos, v, 12);
            std::memcpy(&n, v + mesh.normalOffset, 12);
            std::memcpy(&uv, v + mesh.uvOffset, 8);
            out[i].vertex = pos;
            out[i].packedNormal = EncodeNormalOct(XMVector3Normalize(XMLoadFloat3(&n)));
            PackedVector::XMStoreHalf2(&out[i].texCoord, XMLoadFloat2(&uv));
        }
        std::memcpy(data.data() + at + verts.size() * sizeof(BTriVertex), idx.data(), idx.size() * sizeof(uint16_t));
        p.bytes = (uint32_t)(data.size() - at);
        data.resize((data.size() + 15) & ~size_t(15));
    }

    FileHeader h = {};
    std::memcpy(h.magic, MAGIC, sizeof(MAGIC));
    h.version = VERSION;
    h.groups = groups;
    h.clusters = clusters;
    h.deps = (uint32_t)depList.size();
    const uint64_t tables = sizeof(h) + groups * (sizeof(LodGroup) + sizeof(FilePage)) + clusters * sizeof(FileCluster) +
                            depList.size() * sizeof(uint32_t);
    h.dataOffset = (tables + 4095) & ~4095ull;
    for (FilePage& p : pages) p.offset += h.dataOffset;

    // Written aside and renamed, so a crash leaves no half file under the real name.
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
    const std::string tmp = path + ".tmp";
    FILE* f = nullptr;
    if (fopen_s(&f, tmp.c_str(), "wb") || !f) return false;
    bool ok = fwrite(&h, sizeof(h), 1, f) == 1;
    ok = ok && fwrite(b.groups.data(), sizeof(LodGroup), groups, f) == groups;
    ok = ok && fwrite(pages.data(), sizeof(FilePage), groups, f) == groups;
    ok = ok && fwrite(fileClusters.data(), sizeof(FileCluster), clusters, f) == clusters;
    ok = ok && (depList.empty() || fwrite(depList.data(), sizeof(uint32_t), depList.size(), f) == depList.size());
    const std::vector<uint8_t> pad(h.dataOffset - tables, 0);
    ok = ok && (pad.empty() || fwrite(pad.data(), 1, pad.size(), f) == pad.size());
    ok = ok && fwrite(data.data(), 1, data.size(), f) == data.size();
    ok = fclose(f) == 0 && ok;
    if (ok) {
        std::filesystem::rename(tmp, path, ec);
        ok = !ec;
    }
    if (!ok) std::filesystem::remove(tmp, ec);
    return ok;
}

bool read_pages(const std::string& path, Lod& lod, PagedSource& src) {
    const HANDLE file = open_read(path);
    if (file == INVALID_HANDLE_VALUE) return false;
    FileHeader h = {};
    bool ok = read_at(file, 0, &h, sizeof(h)) && std::memcmp(h.magic, MAGIC, sizeof(MAGIC)) == 0 && h.version == VERSION;
    std::vector<FilePage> pages;
    std::vector<FileCluster> fc;
    if (ok) {
        lod.groups.resize(h.groups);
        pages.resize(h.groups);
        fc.resize(h.clusters);
        lod.deps.resize(h.deps);
        uint64_t at = sizeof(h);
        ok = read_at(file, at, lod.groups.data(), h.groups * (uint32_t)sizeof(LodGroup));
        at += h.groups * sizeof(LodGroup);
        ok = ok && read_at(file, at, pages.data(), h.groups * (uint32_t)sizeof(FilePage));
        at += h.groups * sizeof(FilePage);
        ok = ok && read_at(file, at, fc.data(), h.clusters * (uint32_t)sizeof(FileCluster));
        at += h.clusters * sizeof(FileCluster);
        ok = ok && (h.deps == 0 || read_at(file, at, lod.deps.data(), h.deps * (uint32_t)sizeof(uint32_t)));
    }
    CloseHandle(file);
    if (!ok) return false;

    src.path = path;
    src.pages.resize(h.groups);
    lod.depFirst.resize(h.groups + 1);
    uint32_t roots = 0;
    for (uint32_t g = 0; g < h.groups; ++g) {
        const FilePage& p = pages[g];
        src.pages[g] = PageEntry{ p.offset, p.bytes, p.firstCluster, p.clusterCount, p.vertexCount, p.triCount };
        lod.depFirst[g] = p.depFirst;
        if (lod.groups[g].error == FLT_MAX) roots += p.clusterCount;
    }
    lod.depFirst[h.groups] = h.deps;
    lod.clusters.resize(h.clusters);
    src.spans.resize(h.clusters);
    src.materials.resize(h.clusters);
    for (uint32_t c = 0; c < h.clusters; ++c) {
        lod.clusters[c] = LodCluster{ fc[c].group, fc[c].refined };
        src.spans[c] = Span{ fc[c].firstTri, fc[c].triCount, fc[c].uniqueVerts };
        src.materials[c] = fc[c].material;
    }
    lod.paged = true;
    lod.base = lod.fixed + roots;
    lod.resident.assign(h.groups, 0);
    lod.lastUsed.assign(h.groups, 0);
    lod.want.assign(h.groups, 0.0f);
    return true;
}

#if RDN_CLUSTER_API
Pager::~Pager() {
    {
        std::lock_guard<std::mutex> lk(m_ioMutex);
        m_ioStop = true;
    }
    m_ioWake.notify_all();
    for (auto& t : m_ioThreads) t.join();
    for (Mesh& m : m_meshes)
        if (m.file != INVALID_HANDLE_VALUE) CloseHandle(m.file);
}

void Pager::init(DeviceContext* ctx, ClusterGeometry* geometry, const Geometry& g, const Config& cfg) {
    m_ctx = ctx;
    m_geometry = geometry;
    m_geo = g;
    m_cfg = cfg;
    using nv_helpers_dx12::CreateBuffer;
    ID3D12Device* device = ctx->Device();
    m_clasPool = CreateBuffer(device, cfg.clasPoolBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                              D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nv_helpers_dx12::kDefaultHeapProps);
    m_clasPool->SetName(L"PageClasPool");
    m_clasAlloc.init(cfg.clasPoolBytes / CLAS_UNIT);
    m_arenas.resize(geometry->frame_slots());
    for (Arena& a : m_arenas) {
        a.buffer = CreateBuffer(device, cfg.arenaBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nv_helpers_dx12::kDefaultHeapProps);
        a.buffer->SetName(L"PageClasArena");
        a.scratch = CreateBuffer(device, cfg.arenaBytes / 4, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                 D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nv_helpers_dx12::kDefaultHeapProps);
        a.scratch->SetName(L"PageScratch");
        a.staging = CreateBuffer(device, cfg.stagingBytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ,
                                 nv_helpers_dx12::kUploadHeapProps);
        a.staging->SetName(L"PageStaging");
        D3D12_RANGE noRead{ 0, 0 };
        ThrowIfFailed(a.staging->Map(0, &noRead, reinterpret_cast<void**>(&a.stagingMapped)));
    }
    for (uint32_t i = 0; i < cfg.ioThreads; ++i) m_ioThreads.emplace_back([this] { io_loop(); });
}

void Pager::add_mesh(Lod* lod, PagedSource&& source, uint32_t triBase, bool omm) {
    Mesh& m = m_meshes.emplace_back();
    m.omm = omm;
    m.lod = lod;
    m.src = std::move(source);
    m.file = open_read(m.src.path);
    if (m.file == INVALID_HANDLE_VALUE) throw std::runtime_error("pager: cannot open " + m.src.path);
    m.triBase = triBase;
    m.materialBase = m.src.materialBase;
    m.pages.resize(lod->groups.size());
    for (uint32_t g = 0; g < (uint32_t)lod->groups.size(); ++g) m.pages[g].pinned = lod->groups[g].error == FLT_MAX;
    lod->pager = (uint32_t)m_meshes.size() - 1;
    if (!m_geometry->allocate_ids((uint32_t)lod->clusters.size(), lod->idBase))
        throw std::runtime_error("pager: out of cluster IDs");
    lod->clas.resize(lod->count(), 0);
    m_stats.meshes = (uint32_t)m_meshes.size();
    m_stats.pages += (uint32_t)m.pages.size();
}

bool Pager::alloc_geometry(Mesh& m, uint32_t g, bool evict) {
    const PageEntry& e = m.src.pages[g];
    Page& p = m.pages[g];
    auto fits = [&] {
        uint64_t v = 0, t = 0;
        if (!m_geo.vertexAlloc->allocate(e.vertexCount, v)) return false;
        if (!m_geo.triAlloc->allocate(e.triCount, t)) {
            m_geo.vertexAlloc->free(v, e.vertexCount);
            return false;
        }
        p.vertex = v;
        p.tri = t;
        return true;
    };
    return fits() || (evict && make_room(fits));
}

void Pager::free_geometry(Mesh& m, uint32_t g) {
    const PageEntry& e = m.src.pages[g];
    Page& p = m.pages[g];
    m_geo.vertexAlloc->free(p.vertex, e.vertexCount);
    m_geo.triAlloc->free(p.tri, e.triCount);
}

void Pager::upload(Mesh& m, uint32_t g, const uint8_t* data, ID3D12GraphicsCommandList* copy, ID3D12Resource* staging,
                   uint8_t* stagingMapped, uint64_t stagingOff, uint32_t* gpuMaterials) {
    const PageEntry& e = m.src.pages[g];
    const Page& p = m.pages[g];
    const uint64_t vbytes = (uint64_t)e.vertexCount * sizeof(BTriVertex), ibytes = 3ull * e.triCount * sizeof(uint32_t);
    std::memcpy(stagingMapped + stagingOff, data, vbytes);
    const auto* local = reinterpret_cast<const uint16_t*>(data + vbytes);
    auto* idx = reinterpret_cast<uint32_t*>(stagingMapped + stagingOff + vbytes);
    for (uint32_t i = 0; i < 3 * e.triCount; ++i) idx[i] = (uint32_t)p.vertex + local[i];
    copy->CopyBufferRegion(m_geo.vertices, p.vertex * sizeof(BTriVertex), staging, stagingOff, vbytes);
    copy->CopyBufferRegion(m_geo.indices, p.tri * 3 * sizeof(uint32_t), staging, stagingOff + vbytes, ibytes);
    // Material IDs at the triangles' positions; records as flat IDs from the mesh's own range, which wrap below it.
    std::vector<uint32_t>& mats = *m_geo.materialIds;
    thread_local std::vector<Span> records;
    records.resize(e.clusterCount);
    for (uint32_t k = 0; k < e.clusterCount; ++k) {
        const Span& s = m.src.spans[e.firstCluster + k];
        const uint32_t material = m.materialBase + m.src.materials[e.firstCluster + k];
        for (uint32_t t = 0; t < s.triCount; ++t) {
            mats[p.tri + s.firstTri + t] = material;
            if (gpuMaterials) gpuMaterials[p.tri + s.firstTri + t] = material;
        }
        records[k] = Span{ (uint32_t)p.tri + s.firstTri - m.triBase, s.triCount, s.uniqueVerts };
    }
    m_geometry->write_records(m.lod->idBase + e.firstCluster, records.data(), e.clusterCount);
}

clusters::Mesh Pager::build_desc(const Mesh& m, uint32_t g) const {
    const PageEntry& e = m.src.pages[g];
    clusters::Mesh d;
    d.vertices = m_geo.vertices->GetGPUVirtualAddress();
    d.vertexStride = sizeof(BTriVertex);
    d.indices = m_geo.indices->GetGPUVirtualAddress() + m.pages[g].tri * 3 * sizeof(uint32_t);
    d.spans = m.src.spans.data() + e.firstCluster;
    d.clusterCount = e.clusterCount;
    d.clusterBase = m.lod->idBase + e.firstCluster;
    d.opaqueTriCount = e.triCount; // hierarchies are opaque
    d.limits = limits(d.spans, e.clusterCount, 0, false);
    return d;
}

// Built, then moved by their built sizes: on Bistro (driver 621.01) per-cluster explicit-destination prebuild sizes came
// to 2.9x the built ones (5.6x with 256/256 limits), so building straight into the pool would nearly triple it.
bool Pager::place_clas(uint32_t mi, uint32_t g, D3D12_GPU_VIRTUAL_ADDRESS scratch, bool evict) {
    Mesh& m = m_meshes[mi];
    const PageEntry& e = m.src.pages[g];
    Page& p = m.pages[g];
    uint64_t bytes = 0, units = 0, off = 0;
    for (uint64_t s : p.sizes) {
        bytes += s;
        units += (s + CLAS_UNIT - 1) / CLAS_UNIT;
    }
    auto fits = [&] { return m_clasAlloc.allocate(units, off); };
    if (!fits() && !(evict && make_room(fits))) {
        m_starved = true;
        return false;
    }
    thread_local std::vector<uint64_t> dests;
    dests.resize(e.clusterCount);
    D3D12_GPU_VIRTUAL_ADDRESS at = m_clasPool->GetGPUVirtualAddress() + off * CLAS_UNIT;
    for (uint32_t k = 0; k < e.clusterCount; ++k) {
        dests[k] = at;
        at += (p.sizes[k] + CLAS_UNIT - 1) / CLAS_UNIT * CLAS_UNIT;
        m.lod->clas[m.lod->fixed + e.firstCluster + k] = dests[k];
    }
    m_geometry->queue_move_to(m.lod->idBase + e.firstCluster, e.clusterCount, false, dests.data(), bytes, scratch);
    p.clas = off;
    p.clasUnits = units;
    std::vector<uint64_t>().swap(p.sizes);
    make_resident(mi, g);
    return true;
}

bool Pager::move(uint32_t mi, uint32_t g, Arena& a) {
    const PageEntry& e = m_meshes[mi].src.pages[g];
    uint64_t bytes = 0;
    for (uint64_t s : m_meshes[mi].pages[g].sizes) bytes += s;
    const uint64_t scratch = m_geometry->move_scratch(e.clusterCount, bytes, false, true);
    if (a.scratchUsed + scratch > m_cfg.arenaBytes / 4 || !m_geometry->can_queue_blas(e.clusterCount)) return false;
    if (!place_clas(mi, g, a.scratch->GetGPUVirtualAddress() + a.scratchUsed, true)) return false;
    a.scratchUsed += scratch;
    return true;
}

void Pager::make_resident(uint32_t mi, uint32_t g) {
    Mesh& m = m_meshes[mi];
    const PageEntry& e = m.src.pages[g];
    m.pages[g].state = State::Resident;
    m.lod->resident[g] = 1;
    m.lod->lastUsed[g] = m_frame;
    for (uint32_t i = m.lod->depFirst[g]; i < m.lod->depFirst[g + 1]; ++i) ++m.pages[m.lod->deps[i]].dependents;
    ++m_stats.resident;
    m_stats.residentClusters += e.clusterCount;
    m_stats.residentTris += e.triCount;
}

void Pager::evict(uint32_t mi, uint32_t g) {
    Mesh& m = m_meshes[mi];
    const PageEntry& e = m.src.pages[g];
    Page& p = m.pages[g];
    free_geometry(m, g);
    m_clasAlloc.free(p.clas, p.clasUnits);
    p.clas = p.clasUnits = 0;
    std::fill(m.lod->clas.begin() + m.lod->fixed + e.firstCluster,
              m.lod->clas.begin() + m.lod->fixed + e.firstCluster + e.clusterCount, 0ull);
    if (p.state == State::Resident) retire(mi, g);
    p.state = State::Absent;
    ++m_stats.evicted;
    ++m_stats.evictedTotal;
}

void Pager::retire(uint32_t mi, uint32_t g) {
    Mesh& m = m_meshes[mi];
    const PageEntry& e = m.src.pages[g];
    m.pages[g].state = State::Retiring;
    m.lod->resident[g] = 0;
    for (uint32_t i = m.lod->depFirst[g]; i < m.lod->depFirst[g + 1]; ++i) --m.pages[m.lod->deps[i]].dependents;
    --m_stats.resident;
    m_stats.residentClusters -= e.clusterCount;
    m_stats.residentTris -= e.triCount;
}

void Pager::rebalance(float pressure) {
    // Leaves wanted far less than the best page waiting for room leave the cuts now; their memory follows once no
    // frame in flight traces them (make_room).
    thread_local std::vector<std::tuple<float, uint32_t, uint32_t>> cold;
    cold.clear();
    for (uint32_t mi = 0; mi < (uint32_t)m_meshes.size(); ++mi) {
        const Mesh& m = m_meshes[mi];
        for (uint32_t g = 0; g < (uint32_t)m.pages.size(); ++g) {
            const Page& p = m.pages[g];
            if (p.state == State::Resident && !p.pinned && !p.dependents && m.lod->want[g] * 2.0f < pressure)
                cold.emplace_back(m.lod->want[g], mi, g);
        }
    }
    const size_t n = std::min<size_t>(cold.size(), m_cfg.retiresPerFrame);
    std::partial_sort(cold.begin(), cold.begin() + (ptrdiff_t)n, cold.end());
    for (size_t i = 0; i < n; ++i) {
        retire(std::get<1>(cold[i]), std::get<2>(cold[i]));
        ++m_stats.retiredTotal;
    }
}

bool Pager::make_room(const std::function<bool()>& test) {
    // Candidates once a frame: resident, not a root, no resident page depending on it, and no frame in flight tracing it.
    if (!m_lruBuilt) {
        m_lruBuilt = true;
        m_lru.clear();
        m_lruNext = 0;
        const uint64_t margin = m_geometry->frame_slots() + 2;
        for (uint32_t mi = 0; mi < (uint32_t)m_meshes.size(); ++mi) {
            const Mesh& m = m_meshes[mi];
            for (uint32_t g = 0; g < (uint32_t)m.pages.size(); ++g) {
                const Page& p = m.pages[g];
                if ((p.state == State::Resident || p.state == State::Retiring) && !p.pinned && !p.dependents &&
                    m_frame >= m.lod->lastUsed[g] + margin)
                    m_lru.emplace_back(p.state == State::Retiring ? 0 : m.lod->lastUsed[g], mi, g);
            }
        }
        std::sort(m_lru.begin(), m_lru.end());
    }
    while (m_lruNext < m_lru.size()) {
        const auto [used, mi, g] = m_lru[m_lruNext++];
        const Page& p = m_meshes[mi].pages[g];
        if ((p.state != State::Resident && p.state != State::Retiring) || p.dependents) continue;
        evict(mi, g);
        if (test()) return true;
    }
    return false;
}

void Pager::load_roots(ID3D12GraphicsCommandList4* cl, const std::function<void()>& flush,
                       const std::function<void(uint32_t, ComPtr<ID3D12Resource>)>& blas) {
    using nv_helpers_dx12::CreateBuffer;
    ID3D12Device* device = m_ctx->Device();
    const uint32_t slot = m_geometry->sync_slot();
    auto rtasBuffer = [&](uint64_t bytes, const wchar_t* name) {
        auto b = CreateBuffer(device, std::max<uint64_t>(bytes, 256), D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                              D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nv_helpers_dx12::kDefaultHeapProps);
        b->SetName(name);
        return b;
    };
    auto scratchBuffer = [&](uint64_t bytes) {
        return CreateBuffer(device, std::max<uint64_t>(bytes, 256), D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nv_helpers_dx12::kDefaultHeapProps);
    };
    auto stagingBytes = [](const PageEntry& e) {
        return ((uint64_t)e.vertexCount * sizeof(BTriVertex) + 3ull * e.triCount * sizeof(uint32_t) + 15) & ~15ull;
    };
    std::vector<std::pair<uint32_t, uint32_t>> roots;
    for (uint32_t mi = 0; mi < (uint32_t)m_meshes.size(); ++mi)
        for (uint32_t g = 0; g < (uint32_t)m_meshes[mi].pages.size(); ++g)
            if (m_meshes[mi].pages[g].pinned) roots.emplace_back(mi, g);

    // In batches: read and place, build, then move tightly into the CLAS pool.
    constexpr uint64_t BATCH_STAGING = 256ull << 20;
    constexpr uint32_t BATCH_CLUSTERS = 1u << 16;
    std::vector<uint8_t> data;
    for (size_t next = 0; next < roots.size();) {
        size_t end = next;
        uint64_t staging = 0, arena = 0, scratch = 0;
        uint32_t batchClusters = 0;
        std::vector<ClusterGeometry::Sizes> sz;
        while (end < roots.size()) {
            const Mesh& m = m_meshes[roots[end].first];
            const PageEntry& e = m.src.pages[roots[end].second];
            if (end > next && (staging + stagingBytes(e) > BATCH_STAGING || batchClusters + e.clusterCount > BATCH_CLUSTERS))
                break;
            ClusterGeometry::Sizes s;
            if (!m_geometry->prebuild(limits(m.src.spans.data() + e.firstCluster, e.clusterCount, 0, false), false, s))
                throw std::runtime_error("pager: CLAS prebuild query failed");
            sz.push_back(s);
            staging += stagingBytes(e);
            arena += s.clas;
            scratch += s.clasScratch;
            batchClusters += e.clusterCount;
            ++end;
        }
        ComPtr<ID3D12Resource> stage = CreateBuffer(device, std::max<uint64_t>(staging, 256), D3D12_RESOURCE_FLAG_NONE,
                                                    D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
        uint8_t* mapped = nullptr;
        uint32_t* gpuMaterials = nullptr;
        D3D12_RANGE noRead{ 0, 0 };
        ThrowIfFailed(stage->Map(0, &noRead, reinterpret_cast<void**>(&mapped)));
        ID3D12Resource* matBuf = m_geo.materialIdBuffer ? m_geo.materialIdBuffer->Get() : nullptr;
        if (matBuf) ThrowIfFailed(matBuf->Map(0, &noRead, reinterpret_cast<void**>(&gpuMaterials)));
        using CD = CD3DX12_RESOURCE_BARRIER;
        const D3D12_RESOURCE_BARRIER toCopy[2] = {
            CD::Transition(m_geo.vertices, m_geo.readState, D3D12_RESOURCE_STATE_COPY_DEST),
            CD::Transition(m_geo.indices, m_geo.readState, D3D12_RESOURCE_STATE_COPY_DEST) };
        cl->ResourceBarrier(2, toCopy);
        uint64_t at = 0;
        for (size_t i = next; i < end; ++i) {
            const auto [mi, g] = roots[i];
            Mesh& m = m_meshes[mi];
            const PageEntry& e = m.src.pages[g];
            data.resize(e.bytes);
            if (!read_at(m.file, e.offset, data.data(), e.bytes)) throw std::runtime_error("pager: cannot read " + m.src.path);
            m_stats.readBytes += e.bytes;
            if (!alloc_geometry(m, g, false))
                throw std::runtime_error("pager: the page pool cannot hold the roots; raise RDN_PAGE_POOL_VERTS / RDN_PAGE_POOL_TRIS");
            upload(m, g, data.data(), cl, stage.Get(), mapped, at, gpuMaterials);
            at += stagingBytes(e);
        }
        const D3D12_RESOURCE_BARRIER toRead[2] = {
            CD::Transition(m_geo.vertices, D3D12_RESOURCE_STATE_COPY_DEST, m_geo.readState),
            CD::Transition(m_geo.indices, D3D12_RESOURCE_STATE_COPY_DEST, m_geo.readState) };
        cl->ResourceBarrier(2, toRead);
        stage->Unmap(0, nullptr);
        if (matBuf) matBuf->Unmap(0, nullptr);

        ComPtr<ID3D12Resource> arenaBuf = rtasBuffer(arena, L"PageRootArena"), scratchBuf = scratchBuffer(scratch);
        m_geometry->begin(slot, cl);
        uint64_t ao = 0, so = 0;
        for (size_t i = next; i < end; ++i) {
            const auto [mi, g] = roots[i];
            m_meshes[mi].pages[g].ticket = m_geometry->queue_build(build_desc(m_meshes[mi], g),
                                                                   arenaBuf->GetGPUVirtualAddress() + ao,
                                                                   scratchBuf->GetGPUVirtualAddress() + so);
            ao += sz[i - next].clas;
            so += sz[i - next].clasScratch;
        }
        m_geometry->record(cl);
        flush();

        std::vector<uint64_t> moveOff(end - next);
        uint64_t moveScratch = 0;
        for (size_t i = next; i < end; ++i) {
            const auto [mi, g] = roots[i];
            Page& p = m_meshes[mi].pages[g];
            const PageEntry& e = m_meshes[mi].src.pages[g];
            p.sizes.resize(e.clusterCount);
            if (!m_geometry->sizes(p.ticket, e.clusterCount, p.sizes.data()))
                throw std::runtime_error("pager: root build sizes lost");
            uint64_t bytes = 0;
            for (uint64_t s : p.sizes) bytes += s;
            moveOff[i - next] = moveScratch;
            moveScratch += m_geometry->move_scratch(e.clusterCount, bytes, false, true);
        }
        scratchBuf = scratchBuffer(moveScratch);
        m_geometry->begin(slot, cl);
        for (size_t i = next; i < end; ++i)
            if (!place_clas(roots[i].first, roots[i].second, scratchBuf->GetGPUVirtualAddress() + moveOff[i - next], false))
                throw std::runtime_error("pager: the CLAS pool cannot hold the roots; raise RDN_PAGE_POOL_CLAS_MB");
        m_geometry->record(cl);
        flush();
        next = end;
    }

    // Each mesh's coarsest cut: its fixed clusters and its roots, [0, base).
    std::vector<ComPtr<ID3D12Resource>> out(m_meshes.size());
    std::vector<uint64_t> scratchOff(m_meshes.size());
    uint64_t scratch = 0;
    for (size_t mi = 0; mi < m_meshes.size(); ++mi) {
        const Lod& lod = *m_meshes[mi].lod;
        if (!lod.base) continue;
        uint64_t size = 0, sc = 0;
        m_geometry->blas_prebuild(lod.base, m_meshes[mi].omm, size, sc);
        out[mi] = rtasBuffer(size, L"PagedMeshBLAS");
        scratchOff[mi] = scratch;
        scratch += sc;
    }
    ComPtr<ID3D12Resource> scratchBuf = scratchBuffer(scratch);
    m_geometry->begin(slot, cl);
    for (size_t mi = 0; mi < m_meshes.size(); ++mi) {
        const Lod& lod = *m_meshes[mi].lod;
        if (!out[mi]) continue;
        m_geometry->queue_blas_from(lod.clas.data(), lod.base, lod.base, m_meshes[mi].omm, out[mi]->GetGPUVirtualAddress(),
                                    scratchBuf->GetGPUVirtualAddress() + scratchOff[mi]);
    }
    m_geometry->record(cl);
    flush();
    for (uint32_t mi = 0; mi < (uint32_t)m_meshes.size(); ++mi)
        if (out[mi]) blas(mi, out[mi]);
    m_stats.pinned = (uint32_t)roots.size();
    std::printf("[pager] %u meshes, %u pages; %u roots loaded (%.2fM triangles), CLAS pool %.1f of %.0f MB\n",
                m_stats.meshes, m_stats.pages, m_stats.pinned, (double)m_stats.residentTris / 1e6,
                (double)(m_clasAlloc.used() * CLAS_UNIT) / 1048576.0, (double)m_cfg.clasPoolBytes / 1048576.0);
}

void Pager::before_begin() {
    if (m_inFlight.empty()) return;
    const uint64_t done = m_ctx->PlanetComputeCompleted();
    for (const auto& [mi, g] : m_inFlight) {
        Page& p = m_meshes[mi].pages[g];
        if (p.state != State::Building || !p.fence || done < p.fence) continue;
        const uint32_t count = m_meshes[mi].src.pages[g].clusterCount;
        p.sizes.resize(count);
        if (m_geometry->sizes(p.ticket, count, p.sizes.data())) {
            p.state = State::Built;
        } else {
            free_geometry(m_meshes[mi], g);
            p.state = State::Absent;
            ++m_stats.dropped;
        }
    }
}

void Pager::moves(uint64_t frame) {
    m_frame = frame;
    m_stats.installed = m_stats.evicted = m_stats.dropped = 0;
    m_lruBuilt = false;
    m_starved = false;
    for (Mesh& m : m_meshes) std::fill(m.lod->want.begin(), m.lod->want.end(), 0.0f);
    const uint32_t slot = m_geometry->slot();
    Arena& a = m_arenas[slot % m_arenas.size()];
    a.used = a.scratchUsed = a.stagingUsed = 0;
    // Pages built in this slot's arena move now or never: this frame's builds reuse it, after the moves in the list.
    for (auto it = m_inFlight.begin(); it != m_inFlight.end();) {
        const auto [mi, g] = *it;
        Page& p = m_meshes[mi].pages[g];
        const bool here = p.slot == slot;
        if ((p.state == State::Built && !move(mi, g, a) && here) || (p.state == State::Building && here)) {
            free_geometry(m_meshes[mi], g);
            std::vector<uint64_t>().swap(p.sizes);
            p.state = State::Absent;
            ++m_stats.dropped;
        }
        if (p.state == State::Building || p.state == State::Built) ++it;
        else it = m_inFlight.erase(it);
    }
}

bool Pager::install(uint32_t mi, uint32_t g, ID3D12GraphicsCommandList* copy, Arena& a, uint32_t* gpuMaterials) {
    Mesh& m = m_meshes[mi];
    const PageEntry& e = m.src.pages[g];
    Page& p = m.pages[g];
    // This frame's room first: arena, scratch, staging, arguments.
    ClusterGeometry::Sizes sz;
    if (!m_geometry->prebuild(limits(m.src.spans.data() + e.firstCluster, e.clusterCount, 0, false), false, sz)) return false;
    const uint64_t staging = ((uint64_t)e.vertexCount * sizeof(BTriVertex) + 3ull * e.triCount * sizeof(uint32_t) + 15) & ~15ull;
    if (a.used + sz.clas > m_cfg.arenaBytes || a.scratchUsed + sz.clasScratch > m_cfg.arenaBytes / 4 ||
        a.stagingUsed + staging > m_cfg.stagingBytes || !m_geometry->can_queue(e.clusterCount, false))
        return false;
    if (!alloc_geometry(m, g, true)) {
        m_starved = true;
        return false;
    }
    upload(m, g, p.data.data(), copy, a.staging.Get(), a.stagingMapped, a.stagingUsed, gpuMaterials);
    a.stagingUsed += staging;
    p.ticket = m_geometry->queue_build(build_desc(m, g), a.buffer->GetGPUVirtualAddress() + a.used,
                                       a.scratch->GetGPUVirtualAddress() + a.scratchUsed);
    a.used += sz.clas;
    a.scratchUsed += sz.clasScratch;
    p.slot = m_geometry->slot();
    p.state = State::Building;
    p.fence = 0;
    std::vector<uint8_t>().swap(p.data);
    m_building.emplace_back(mi, g);
    m_inFlight.emplace_back(mi, g);
    return true;
}

void Pager::installs(std::vector<LodRequest>& requests, ID3D12GraphicsCommandList* copy) {
    std::deque<Read> done;
    {
        std::lock_guard<std::mutex> lk(m_ioMutex);
        done.swap(m_ioDone);
    }
    for (Read& r : done) {
        Page& p = m_meshes[r.mesh].pages[r.group];
        --m_reading;
        if (!r.ok) {
            p.state = State::Absent;
            continue;
        }
        m_stats.readBytes += r.bytes;
        p.data = std::move(r.data);
        p.state = State::Read;
        m_read.emplace_back(r.mesh, r.group);
    }

    // New reads, the most wanted first.
    std::sort(requests.begin(), requests.end(), [](const LodRequest& a, const LodRequest& b) { return a.priority > b.priority; });
    uint32_t issued = 0;
    {
        std::lock_guard<std::mutex> lk(m_ioMutex);
        for (const LodRequest& q : requests) {
            if (m_reading + m_read.size() >= m_cfg.readsInFlight) break;
            Mesh& m = m_meshes[q.mesh];
            Page& p = m.pages[q.group];
            if (p.state == State::Retiring) { // still in memory: back into the cuts
                make_resident(q.mesh, q.group);
                continue;
            }
            if (p.state != State::Absent) continue;
            const PageEntry& e = m.src.pages[q.group];
            p.state = State::Reading;
            ++m_reading;
            m_ioQueue.push_back(Read{ q.mesh, q.group, m.file, e.offset, e.bytes, {}, false });
            ++issued;
        }
    }
    if (issued) m_ioWake.notify_all();

    // Installs, the oldest read first; one whose deps went away meanwhile waits for them to come back.
    if (!m_read.empty()) {
        Arena& a = m_arenas[m_geometry->slot() % m_arenas.size()];
        uint32_t* gpuMaterials = nullptr;
        ID3D12Resource* matBuf = m_geo.materialIdBuffer ? m_geo.materialIdBuffer->Get() : nullptr;
        D3D12_RANGE noRead{ 0, 0 };
        if (matBuf && FAILED(matBuf->Map(0, &noRead, reinterpret_cast<void**>(&gpuMaterials)))) matBuf = nullptr;
        // Read pages wait while the frame or the pools are full; once the pools are, the ones no view wants any more
        // give their read slots back.
        bool full = false;
        for (size_t i = 0; i < m_read.size();) {
            const auto [mi, g] = m_read[i];
            Mesh& m = m_meshes[mi];
            Page& p = m.pages[g];
            bool ready = !m_starved || m.lod->want[g] > 0.0f;
            for (uint32_t d = m.lod->depFirst[g]; d < m.lod->depFirst[g + 1] && ready; ++d) ready = m.lod->resident[m.lod->deps[d]] != 0;
            if (!ready) {
                std::vector<uint8_t>().swap(p.data);
                p.state = State::Absent;
                ++m_stats.dropped;
                m_read.erase(m_read.begin() + (ptrdiff_t)i);
                continue;
            }
            if (full || m_starved || m_stats.installed >= m_cfg.installsPerFrame ||
                !install(mi, g, copy, a, gpuMaterials)) {
                full = !m_starved;
                ++i;
                continue;
            }
            ++m_stats.installed;
            ++m_stats.installedTotal;
            m_read.erase(m_read.begin() + (ptrdiff_t)i);
        }
        if (matBuf) matBuf->Unmap(0, nullptr);
    }
    if (m_starved) {
        float pressure = 0.0f;
        for (const LodRequest& q : requests)
            if (m_meshes[q.mesh].pages[q.group].state != State::Resident) {
                pressure = q.priority;
                break;
            }
        if (pressure > 0.0f) rebalance(pressure);
    }
    m_stats.droppedTotal += m_stats.dropped;
    m_stats.reading = m_reading + (uint32_t)m_read.size();
    m_stats.building = (uint32_t)m_inFlight.size();
    m_stats.clasUsed = m_clasAlloc.used() * CLAS_UNIT;
    m_stats.clasCapacity = m_cfg.clasPoolBytes;
    m_stats.trisUsed = m_geo.triAlloc->used();
    m_stats.trisCapacity = m_geo.triAlloc->capacity();
    m_stats.vertsUsed = m_geo.vertexAlloc->used();
    m_stats.vertsCapacity = m_geo.vertexAlloc->capacity();
}

void Pager::on_submitted(uint64_t computeFence) {
    for (const auto& [mi, g] : m_building) m_meshes[mi].pages[g].fence = computeFence;
    m_building.clear();
}

void Pager::io_loop() {
    for (;;) {
        Read r;
        {
            std::unique_lock<std::mutex> lk(m_ioMutex);
            m_ioWake.wait(lk, [&] { return m_ioStop || !m_ioQueue.empty(); });
            if (m_ioStop) return;
            r = std::move(m_ioQueue.front());
            m_ioQueue.pop_front();
        }
        r.data.resize(r.bytes);
        r.ok = read_at(r.file, r.offset, r.data.data(), r.bytes);
        std::lock_guard<std::mutex> lk(m_ioMutex);
        m_ioDone.push_back(std::move(r));
    }
}
#endif

} // namespace clusters
