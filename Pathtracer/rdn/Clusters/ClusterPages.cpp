#include "ClusterPages.h"
#include "../Core/DeviceContext.h"
#include "../DXRHelper.h"
#include <algorithm>
#include <cfloat>
#include <cstring>
#include <filesystem>
#include <map>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <DirectXPackedVector.h>

namespace clusters {

namespace {
constexpr char MAGIC[8] = { 'R', 'D', 'N', 'P', 'A', 'G', 'E', 'S' };
constexpr uint32_t VERSION = 4, VERSION_3 = 3;
// File: header, LodGroup per group, FilePage per group, FileCluster per hierarchy cluster (by offset), the deps, the
// mesh's micromaps (OmmHeader, descs, histogram, raw data), the extra bytes, then the pages: BTriVertex per page vertex,
// three page-local uint16 indices per triangle, in a coded file a uint16 material per triangle, and for an alpha page
// (flags bit 0), 4-byte aligned, an int32 OMM index per triangle. Version 3 files lack the extra bytes and codes.
constexpr uint32_t PAGE_ALPHA = 1;
constexpr uint32_t FILE_CODED = 1;
// A page index: no page data of its own; after the deps the data files' names (a count, then per name its length and
// characters, relative to the index's folder). FilePage::flags bits 16 and up name a page's file, FileCluster::pad a
// cluster's cell.
constexpr uint32_t FILE_INDEX = 2;
constexpr uint32_t PAGE_FILE_SHIFT = 16;
// Micromap subdivisions: full detail as the texture asks, where equal micromaps are shared widely; coarser levels' larger
// triangles rarely share theirs and are seen from further away.
constexpr int OMM_LEVEL_FULL = 8, OMM_LEVEL_COARSE = 5;
struct OmmHeader {
    uint32_t descCount, histogramCount, rawBytes, pad;
};
struct FileHeader {
    char magic[8];
    uint32_t version, groups, clusters, deps;
    uint64_t dataOffset, ommOffset;
    uint32_t ommBytes, flags;
    uint64_t extraOffset, extraBytes; // version 4
};
constexpr uint32_t HEADER_3_BYTES = 48;
struct FilePage {
    uint64_t offset;
    uint32_t bytes, firstCluster, clusterCount, vertexCount, triCount, depFirst, depCount, flags;
};
struct FileCluster {
    uint32_t group, refined, firstTri, material;
    uint16_t triCount, uniqueVerts;
    uint32_t pad;
};
static_assert(sizeof(FileHeader) == 64 && sizeof(FilePage) == 40 && sizeof(FileCluster) == 24);
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

void RangeAllocator::grow(uint64_t capacity) {
    if (capacity <= m_capacity) return;
    const uint64_t at = m_capacity;
    m_capacity = capacity;
    m_used += capacity - at;
    free(at, capacity - at);
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
    s.vertices = (uint32_t)env_u64("RDN_PAGE_POOL_VERTS", 8u << 20); // foliage cards need about 2 per triangle
    s.tris = (uint32_t)env_u64("RDN_PAGE_POOL_TRIS", 4u << 20);
    s.clasBytes = env_u64("RDN_PAGE_POOL_CLAS_MB", 512) << 20;
    s.ommBytes = env_u64("RDN_PAGE_POOL_OMM_MB", 256) << 20;
    char v[4] = {};
    s.fixed = GetEnvironmentVariableA("RDN_PAGE_POOL_TRIS", v, sizeof(v)) > 0 ||
              GetEnvironmentVariableA("RDN_PAGE_POOL_VERTS", v, sizeof(v)) > 0;
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
    auto region_of = [&](uint32_t g) { return b.regions[b.spans[members[g][0]].firstTri]; };

    // Micromaps: baked once per alpha region over all its levels, so equal ones are shared, and kept with the mesh; an
    // alpha page holds an index per triangle into them.
    std::vector<std::vector<int32_t>> pageOmm(groups);
    PageOmm meshOmm;
    const bool coded = !b.codes.empty();
    // Coded with materials for codes: micromaps per material over its triangles of every page (full detail and the coarser
    // ones apart), an index per triangle into the mesh's; a page is alpha when any of its triangles' materials is.
    std::vector<int32_t> triOmm;
    std::vector<uint8_t> triAlpha;
    if (mesh.opacity && coded) {
        const size_t tris = b.indices.size() / 3;
        triOmm.assign(tris, D3D12_RAYTRACING_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_OPAQUE);
        triAlpha.assign(tris, 0);
        std::vector<uint8_t> coarse(tris, 0);
        for (uint32_t c = 0; c < clusters; ++c)
            if (b.clusters[c].refined != LOD_NONE)
                std::fill(coarse.begin() + b.spans[c].firstTri, coarse.begin() + b.spans[c].firstTri + b.spans[c].triCount, 1);
        std::map<uint32_t, std::vector<uint32_t>> byMaterial[2];
        for (uint32_t t = 0; t < (uint32_t)tris; ++t) byMaterial[coarse[t]][b.codes[t]].push_back(t);
        std::unordered_map<uint32_t, uint32_t> local;
        std::vector<float> uvs;
        std::vector<uint32_t> idx;
        for (int pass = 0; pass < 2; ++pass)
            for (const auto& [code, list] : byMaterial[pass]) {
                local.clear();
                uvs.clear();
                idx.clear();
                for (uint32_t t : list)
                    for (uint32_t i = 3 * t; i < 3 * t + 3; ++i) {
                        const auto [it, fresh] = local.emplace(b.indices[i], (uint32_t)(uvs.size() / 2));
                        if (fresh) {
                            float uv[2];
                            std::memcpy(uv, mesh.vertices + (size_t)b.indices[i] * mesh.stride + mesh.uvOffset, 8);
                            uvs.insert(uvs.end(), uv, uv + 2);
                        }
                        idx.push_back(it->second);
                    }
                PageOmm r;
                if (!mesh.opacity(mesh.materialBase + code, pass ? OMM_LEVEL_COARSE : OMM_LEVEL_FULL, uvs.data(),
                                  (uint32_t)(uvs.size() / 2), idx.data(), (uint32_t)list.size(), r))
                    continue;
                r.indices.resize(list.size(), D3D12_RAYTRACING_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_OPAQUE);
                const int32_t descBase = (int32_t)meshOmm.descs.size();
                for (D3D12_RAYTRACING_OPACITY_MICROMAP_DESC d : r.descs) {
                    d.ByteOffset += (UINT)meshOmm.raw.size();
                    meshOmm.descs.push_back(d);
                }
                meshOmm.raw.insert(meshOmm.raw.end(), r.raw.begin(), r.raw.end());
                for (const auto& e : r.histogram) {
                    auto same = std::find_if(meshOmm.histogram.begin(), meshOmm.histogram.end(), [&](const auto& x) {
                        return x.SubdivisionLevel == e.SubdivisionLevel && x.Format == e.Format;
                    });
                    if (same != meshOmm.histogram.end()) same->Count += e.Count;
                    else meshOmm.histogram.push_back(e);
                }
                for (size_t i = 0; i < list.size(); ++i) {
                    triOmm[list[i]] = r.indices[i] >= 0 ? r.indices[i] + descBase : r.indices[i];
                    triAlpha[list[i]] = 1;
                }
            }
    }
    if (mesh.opacity && !coded) {
        std::unordered_map<uint32_t, std::vector<uint32_t>> regionPages;
        std::vector<uint32_t> regionOrder;
        for (uint32_t g : order) {
            if (members[g].empty()) continue;
            const auto [it, fresh] = regionPages.try_emplace(region_of(g));
            if (fresh) regionOrder.push_back(region_of(g));
            it->second.push_back(g);
        }
        std::unordered_map<uint32_t, uint32_t> local;
        std::vector<float> uvs;
        std::vector<uint32_t> idx;
        std::vector<int32_t> regionIndices;
        for (uint32_t region : regionOrder) {
            // Per triangle in page order, its index; the full-detail and the coarser triangles bake apart.
            size_t regionTris = 0;
            for (uint32_t g : regionPages[region])
                for (uint32_t c : members[g]) regionTris += b.spans[c].triCount;
            regionIndices.assign(regionTris, D3D12_RAYTRACING_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_OPAQUE);
            bool alpha = false;
            for (int pass = 0; pass < 2; ++pass) {
                const bool coarse = pass == 1;
                local.clear();
                uvs.clear();
                idx.clear();
                std::vector<size_t> where;
                size_t t = 0;
                for (uint32_t g : regionPages[region])
                    for (uint32_t c : members[g]) {
                        const bool clusterCoarse = b.clusters[c].refined != LOD_NONE;
                        for (uint32_t k = 0; k < b.spans[c].triCount; ++k, ++t) {
                            if (clusterCoarse != coarse) continue;
                            where.push_back(t);
                            for (uint32_t i = 3 * (b.spans[c].firstTri + k); i < 3 * (b.spans[c].firstTri + k + 1); ++i) {
                                const auto [it, fresh] = local.emplace(b.indices[i], (uint32_t)(uvs.size() / 2));
                                if (fresh) {
                                    float uv[2];
                                    std::memcpy(uv, mesh.vertices + (size_t)b.indices[i] * mesh.stride + mesh.uvOffset, 8);
                                    uvs.insert(uvs.end(), uv, uv + 2);
                                }
                                idx.push_back(it->second);
                            }
                        }
                    }
                if (where.empty()) continue;
                PageOmm r;
                if (!mesh.opacity(region, coarse ? OMM_LEVEL_COARSE : OMM_LEVEL_FULL, uvs.data(), (uint32_t)(uvs.size() / 2),
                                  idx.data(), (uint32_t)where.size(), r))
                    break;
                alpha = true;
                r.indices.resize(where.size(), D3D12_RAYTRACING_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_OPAQUE);
                const int32_t descBase = (int32_t)meshOmm.descs.size();
                for (D3D12_RAYTRACING_OPACITY_MICROMAP_DESC d : r.descs) {
                    d.ByteOffset += (UINT)meshOmm.raw.size();
                    meshOmm.descs.push_back(d);
                }
                meshOmm.raw.insert(meshOmm.raw.end(), r.raw.begin(), r.raw.end());
                for (const auto& e : r.histogram) {
                    auto same = std::find_if(meshOmm.histogram.begin(), meshOmm.histogram.end(), [&](const auto& x) {
                        return x.SubdivisionLevel == e.SubdivisionLevel && x.Format == e.Format;
                    });
                    if (same != meshOmm.histogram.end()) same->Count += e.Count;
                    else meshOmm.histogram.push_back(e);
                }
                for (size_t i = 0; i < where.size(); ++i)
                    regionIndices[where[i]] = r.indices[i] >= 0 ? r.indices[i] + descBase : r.indices[i];
            }
            if (!alpha) continue;
            size_t t = 0;
            for (uint32_t g : regionPages[region]) {
                std::vector<int32_t>& out = pageOmm[g];
                for (uint32_t c : members[g])
                    for (uint32_t k = 0; k < b.spans[c].triCount; ++k) out.push_back(regionIndices[t++]);
            }
        }
    }

    std::vector<FilePage> pages(groups);
    std::vector<FileCluster> fileClusters(clusters);
    std::vector<uint32_t> depList;
    std::vector<uint8_t> data;
    std::unordered_map<uint32_t, uint16_t> local;
    std::vector<uint32_t> verts, corner; // per page vertex: mesh vertex, and an index entry using it
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
        corner.clear();
        std::vector<uint16_t> idx;
        for (uint32_t c : members[g]) {
            const Span& s = b.spans[c];
            FileCluster& fc = fileClusters[offset++];
            fc = {};
            fc.group = g;
            fc.refined = b.clusters[c].refined;
            fc.firstTri = (uint32_t)(idx.size() / 3);
            fc.material = coded ? b.codes[s.firstTri] : b.regions[s.firstTri] - mesh.materialBase;
            fc.triCount = s.triCount;
            fc.uniqueVerts = s.uniqueVerts;
            for (uint32_t i = 3 * s.firstTri; i < 3 * (s.firstTri + s.triCount); ++i) {
                const auto [it, fresh] = local.emplace(b.indices[i], (uint16_t)verts.size());
                if (fresh) {
                    if (verts.size() == 0xFFFF) return false;
                    verts.push_back(b.indices[i]);
                    corner.push_back(i);
                }
                idx.push_back(it->second);
            }
        }
        p.vertexCount = (uint32_t)verts.size();
        p.triCount = (uint32_t)(idx.size() / 3);
        std::vector<uint16_t> codes;
        if (coded) {
            bool alpha = false;
            for (uint32_t c : members[g])
                for (uint32_t t = b.spans[c].firstTri; t < b.spans[c].firstTri + b.spans[c].triCount; ++t) {
                    codes.push_back(mesh.codeMaterial ? mesh.codeMaterial(b.codes[t]) : (uint16_t)b.codes[t]);
                    alpha = alpha || (mesh.opacity ? triAlpha[t] != 0 : mesh.codeAlpha && mesh.codeAlpha(b.codes[t]));
                }
            if (alpha)
                for (uint32_t c : members[g])
                    for (uint32_t t = b.spans[c].firstTri; t < b.spans[c].firstTri + b.spans[c].triCount; ++t)
                        pageOmm[g].push_back(mesh.opacity ? triOmm[t] : mesh.codeOmm(b.codes[t]));
        }
        const size_t at = data.size();
        data.resize(at + verts.size() * sizeof(BTriVertex) + (idx.size() + codes.size()) * sizeof(uint16_t));
        auto* out = reinterpret_cast<BTriVertex*>(data.data() + at);
        for (size_t i = 0; i < verts.size(); ++i) {
            const uint8_t* v = mesh.vertices + (size_t)verts[i] * mesh.stride;
            XMFLOAT3 pos, n;
            XMFLOAT2 uv;
            std::memcpy(&pos, b.positions.empty() ? (const void*)v : (const void*)&b.positions[3ull * corner[i]], 12);
            std::memcpy(&n, v + mesh.normalOffset, 12);
            std::memcpy(&uv, v + mesh.uvOffset, 8);
            out[i].vertex = pos;
            out[i].packedNormal = EncodeNormalOct(XMVector3Normalize(XMLoadFloat3(&n)));
            PackedVector::XMStoreHalf2(&out[i].texCoord, XMLoadFloat2(&uv));
        }
        std::memcpy(data.data() + at + verts.size() * sizeof(BTriVertex), idx.data(), idx.size() * sizeof(uint16_t));
        if (!codes.empty())
            std::memcpy(data.data() + at + verts.size() * sizeof(BTriVertex) + idx.size() * sizeof(uint16_t), codes.data(),
                        codes.size() * sizeof(uint16_t));
        if (!pageOmm[g].empty()) {
            p.flags |= PAGE_ALPHA;
            data.resize((data.size() + 3) & ~size_t(3));
            const size_t o = data.size();
            data.resize(o + pageOmm[g].size() * sizeof(int32_t));
            std::memcpy(data.data() + o, pageOmm[g].data(), pageOmm[g].size() * sizeof(int32_t));
        }
        p.bytes = (uint32_t)(data.size() - at);
        data.resize((data.size() + 15) & ~size_t(15));
    }

    // The mesh's micromaps: OmmHeader, descs, histogram, raw data.
    std::vector<uint8_t> ommBlock;
    if (!meshOmm.descs.empty()) {
        const OmmHeader oh{ (uint32_t)meshOmm.descs.size(), (uint32_t)meshOmm.histogram.size(), (uint32_t)meshOmm.raw.size(), 0 };
        auto put = [&](const void* src, size_t bytes) {
            const size_t o = ommBlock.size();
            ommBlock.resize(o + bytes);
            if (bytes) std::memcpy(ommBlock.data() + o, src, bytes);
        };
        put(&oh, sizeof(oh));
        put(meshOmm.descs.data(), meshOmm.descs.size() * sizeof(meshOmm.descs[0]));
        put(meshOmm.histogram.data(), meshOmm.histogram.size() * sizeof(meshOmm.histogram[0]));
        put(meshOmm.raw.data(), meshOmm.raw.size());
    }

    FileHeader h = {};
    std::memcpy(h.magic, MAGIC, sizeof(MAGIC));
    h.version = VERSION;
    h.groups = groups;
    h.clusters = clusters;
    h.deps = (uint32_t)depList.size();
    const uint64_t tables = sizeof(h) + groups * (sizeof(LodGroup) + sizeof(FilePage)) + clusters * sizeof(FileCluster) +
                            depList.size() * sizeof(uint32_t);
    h.ommOffset = (tables + 15) & ~15ull;
    h.ommBytes = (uint32_t)ommBlock.size();
    h.flags = coded ? FILE_CODED : 0u;
    h.extraOffset = (h.ommOffset + h.ommBytes + 15) & ~15ull;
    h.extraBytes = mesh.extra.size();
    h.dataOffset = (h.extraOffset + h.extraBytes + 4095) & ~4095ull;
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
    auto pad = [&](uint64_t from, uint64_t to) {
        const std::vector<uint8_t> zeros(to - from, 0);
        return zeros.empty() || fwrite(zeros.data(), 1, zeros.size(), f) == zeros.size();
    };
    ok = ok && pad(tables, h.ommOffset);
    ok = ok && (ommBlock.empty() || fwrite(ommBlock.data(), 1, ommBlock.size(), f) == ommBlock.size());
    ok = ok && pad(h.ommOffset + h.ommBytes, h.extraOffset);
    ok = ok && (mesh.extra.empty() || fwrite(mesh.extra.data(), 1, mesh.extra.size(), f) == mesh.extra.size());
    ok = ok && pad(h.extraOffset + h.extraBytes, h.dataOffset);
    ok = ok && fwrite(data.data(), 1, data.size(), f) == data.size();
    ok = fclose(f) == 0 && ok;
    if (ok) {
        std::filesystem::rename(tmp, path, ec);
        ok = !ec;
    }
    if (!ok) std::filesystem::remove(tmp, ec);
    return ok;
}

namespace {
void encode_vertex(BTriVertex& out, const BakeMesh& mesh, uint32_t v, const float* position) {
    using namespace DirectX;
    const uint8_t* src = mesh.vertices + (size_t)v * mesh.stride;
    XMFLOAT3 pos, n;
    XMFLOAT2 uv;
    std::memcpy(&pos, position ? (const void*)position : (const void*)src, 12);
    std::memcpy(&n, src + mesh.normalOffset, 12);
    std::memcpy(&uv, src + mesh.uvOffset, 8);
    out.vertex = pos;
    out.packedNormal = EncodeNormalOct(XMVector3Normalize(XMLoadFloat3(&n)));
    PackedVector::XMStoreHalf2(&out.texCoord, XMLoadFloat2(&uv));
}
} // namespace

bool serialize_page(const PageCluster* clusters, uint32_t count, const BakeMesh& mesh, std::vector<uint8_t>& out,
                    PageLayout& layout) {
    std::unordered_map<uint32_t, uint16_t> local;
    std::vector<uint32_t> verts;
    std::vector<uint16_t> idx, codes;
    std::vector<int32_t> omm;
    layout = PageLayout{};
    for (uint32_t c = 0; c < count; ++c) {
        const PageCluster& pc = clusters[c];
        const size_t first = verts.size();
        layout.spans.push_back(Span{ (uint32_t)(idx.size() / 3), (uint16_t)pc.triCount, 0 });
        for (uint32_t i = 0; i < 3 * pc.triCount; ++i) {
            const auto [it, fresh] = local.emplace(pc.indices[i], (uint16_t)verts.size());
            if (fresh) {
                if (verts.size() == 0xFFFF) return false;
                verts.push_back(pc.indices[i]);
            }
            idx.push_back(it->second);
        }
        // Unique vertices of the cluster alone.
        std::vector<uint32_t> u(pc.indices, pc.indices + 3 * pc.triCount);
        std::sort(u.begin(), u.end());
        layout.spans.back().uniqueVerts = (uint16_t)(std::unique(u.begin(), u.end()) - u.begin());
        (void)first;
        for (uint32_t t = 0; t < pc.triCount; ++t) {
            codes.push_back(mesh.codeMaterial(pc.codes[t]));
            layout.alpha = layout.alpha || mesh.codeAlpha(pc.codes[t]);
        }
    }
    if (layout.alpha)
        for (uint32_t c = 0; c < count; ++c)
            for (uint32_t t = 0; t < clusters[c].triCount; ++t) omm.push_back(mesh.codeOmm(clusters[c].codes[t]));
    layout.vertexCount = (uint32_t)verts.size();
    layout.triCount = (uint32_t)(idx.size() / 3);
    out.resize(verts.size() * sizeof(BTriVertex) + (idx.size() + codes.size()) * sizeof(uint16_t));
    auto* v = reinterpret_cast<BTriVertex*>(out.data());
    for (size_t i = 0; i < verts.size(); ++i) encode_vertex(v[i], mesh, verts[i], nullptr);
    std::memcpy(out.data() + verts.size() * sizeof(BTriVertex), idx.data(), idx.size() * sizeof(uint16_t));
    std::memcpy(out.data() + verts.size() * sizeof(BTriVertex) + idx.size() * sizeof(uint16_t), codes.data(),
                codes.size() * sizeof(uint16_t));
    if (!omm.empty()) {
        out.resize((out.size() + 3) & ~size_t(3));
        const size_t o = out.size();
        out.resize(o + omm.size() * sizeof(int32_t));
        std::memcpy(out.data() + o, omm.data(), omm.size() * sizeof(int32_t));
    }
    return true;
}

bool write_page_index(const std::string& path, const std::vector<LodGroup>& groups, const std::vector<IndexPage>& pages,
                      const std::vector<LodCluster>& clusters, const std::vector<Span>& spans,
                      const std::vector<std::string>& files, const std::vector<uint32_t>& cells,
                      const std::vector<uint8_t>& extra) {
    const uint32_t groupCount = (uint32_t)groups.size(), clusterCount = (uint32_t)clusters.size();
    // Clusters by group, a group's in the order given.
    std::vector<uint32_t> order(clusterCount);
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return clusters[a].group < clusters[b].group; });
    std::vector<std::vector<uint32_t>> deps(groupCount);
    for (const LodCluster& c : clusters)
        if (c.refined != LOD_NONE) deps[c.refined].push_back(c.group);
    std::vector<FilePage> fp(groupCount);
    std::vector<FileCluster> fc(clusterCount);
    std::vector<uint32_t> depList;
    std::vector<uint8_t> seen(groupCount, 0);
    for (uint32_t i = 0; i < clusterCount; ++i) {
        const uint32_t c = order[i], g = clusters[c].group;
        if (!seen[g]) {
            seen[g] = 1;
            fp[g].firstCluster = i;
        }
        ++fp[g].clusterCount;
        fc[i] = FileCluster{ g, clusters[c].refined, spans[c].firstTri, 0u, spans[c].triCount, spans[c].uniqueVerts, cells[c] };
    }
    for (uint32_t g = 0; g < groupCount; ++g) {
        std::sort(deps[g].begin(), deps[g].end());
        deps[g].erase(std::unique(deps[g].begin(), deps[g].end()), deps[g].end());
        FilePage& p = fp[g];
        p.offset = pages[g].offset;
        p.bytes = pages[g].bytes;
        p.vertexCount = pages[g].vertexCount;
        p.triCount = pages[g].triCount;
        p.depFirst = (uint32_t)depList.size();
        p.depCount = (uint32_t)deps[g].size();
        p.flags = (pages[g].alpha ? PAGE_ALPHA : 0u) | pages[g].file << PAGE_FILE_SHIFT;
        depList.insert(depList.end(), deps[g].begin(), deps[g].end());
    }
    std::vector<uint8_t> names;
    auto put = [&](const void* src, size_t n) {
        const size_t o = names.size();
        names.resize(o + n);
        if (n) std::memcpy(names.data() + o, src, n);
    };
    const uint32_t fileCount = (uint32_t)files.size();
    put(&fileCount, 4);
    for (const std::string& f : files) {
        const uint16_t n = (uint16_t)f.size();
        put(&n, 2);
        put(f.data(), n);
    }

    FileHeader h = {};
    std::memcpy(h.magic, MAGIC, sizeof(MAGIC));
    h.version = VERSION;
    h.groups = groupCount;
    h.clusters = clusterCount;
    h.deps = (uint32_t)depList.size();
    h.flags = FILE_CODED | FILE_INDEX;
    const uint64_t tables = sizeof(h) + groupCount * (sizeof(LodGroup) + sizeof(FilePage)) + clusterCount * sizeof(FileCluster) +
                            depList.size() * sizeof(uint32_t);
    h.ommOffset = tables + names.size();
    h.ommBytes = 0;
    h.extraOffset = (h.ommOffset + 15) & ~15ull;
    h.extraBytes = extra.size();
    h.dataOffset = h.extraOffset + h.extraBytes;
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
    const std::string tmp = path + ".tmp";
    FILE* f = nullptr;
    if (fopen_s(&f, tmp.c_str(), "wb") || !f) return false;
    bool ok = fwrite(&h, sizeof(h), 1, f) == 1;
    ok = ok && fwrite(groups.data(), sizeof(LodGroup), groupCount, f) == groupCount;
    ok = ok && fwrite(fp.data(), sizeof(FilePage), groupCount, f) == groupCount;
    ok = ok && fwrite(fc.data(), sizeof(FileCluster), clusterCount, f) == clusterCount;
    ok = ok && (depList.empty() || fwrite(depList.data(), sizeof(uint32_t), depList.size(), f) == depList.size());
    ok = ok && fwrite(names.data(), 1, names.size(), f) == names.size();
    const std::vector<uint8_t> zeros((size_t)(h.extraOffset - h.ommOffset), 0);
    ok = ok && (zeros.empty() || fwrite(zeros.data(), 1, zeros.size(), f) == zeros.size());
    ok = ok && (extra.empty() || fwrite(extra.data(), 1, extra.size(), f) == extra.size());
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
    bool ok = read_at(file, 0, &h, HEADER_3_BYTES) && std::memcmp(h.magic, MAGIC, sizeof(MAGIC)) == 0 &&
              (h.version == VERSION || h.version == VERSION_3);
    if (ok && h.version == VERSION_3) h.flags = 0;
    ok = ok && (h.version == VERSION_3 || read_at(file, HEADER_3_BYTES, (uint8_t*)&h + HEADER_3_BYTES, sizeof(h) - HEADER_3_BYTES));
    std::vector<FilePage> pages;
    std::vector<FileCluster> fc;
    if (ok) {
        lod.groups.resize(h.groups);
        pages.resize(h.groups);
        fc.resize(h.clusters);
        lod.deps.resize(h.deps);
        uint64_t at = h.version == VERSION_3 ? HEADER_3_BYTES : sizeof(h);
        ok = read_at(file, at, lod.groups.data(), h.groups * (uint32_t)sizeof(LodGroup));
        at += h.groups * sizeof(LodGroup);
        ok = ok && read_at(file, at, pages.data(), h.groups * (uint32_t)sizeof(FilePage));
        at += h.groups * sizeof(FilePage);
        ok = ok && read_at(file, at, fc.data(), h.clusters * (uint32_t)sizeof(FileCluster));
        at += h.clusters * sizeof(FileCluster);
        ok = ok && (h.deps == 0 || read_at(file, at, lod.deps.data(), h.deps * (uint32_t)sizeof(uint32_t)));
    }
    std::vector<uint8_t> omm(ok && !(h.flags & FILE_INDEX) ? h.ommBytes : 0);
    ok = ok && (omm.empty() || read_at(file, h.ommOffset, omm.data(), h.ommBytes));
    // An index's data files and cells.
    src.files.clear();
    src.cells.clear();
    if (ok && (h.flags & FILE_INDEX)) {
        const uint64_t at = sizeof(h) + h.groups * (sizeof(LodGroup) + sizeof(FilePage)) + h.clusters * sizeof(FileCluster) +
                            (uint64_t)h.deps * sizeof(uint32_t);
        std::vector<uint8_t> names((size_t)(h.ommOffset - at));
        ok = read_at(file, at, names.data(), (uint32_t)names.size()) && names.size() >= 4;
        size_t o = 0;
        uint32_t count = 0;
        if (ok) std::memcpy(&count, names.data(), 4), o = 4;
        const std::filesystem::path dir = std::filesystem::path(path).parent_path();
        for (uint32_t i = 0; ok && i < count; ++i) {
            uint16_t n = 0;
            ok = o + 2 <= names.size();
            if (ok) std::memcpy(&n, names.data() + o, 2), o += 2;
            ok = ok && o + n <= names.size();
            if (ok) src.files.push_back((dir / std::string(names.data() + o, names.data() + o + n)).string()), o += n;
        }
    }
    src.extra.assign(ok && h.version != VERSION_3 ? (size_t)h.extraBytes : 0, 0);
    ok = ok && (src.extra.empty() || read_at(file, h.extraOffset, src.extra.data(), (uint32_t)src.extra.size()));
    CloseHandle(file);
    if (!ok) return false;
    src.coded = (h.flags & FILE_CODED) != 0;
    src.omm = PageOmm{};
    if (omm.size() >= sizeof(OmmHeader)) {
        OmmHeader oh;
        std::memcpy(&oh, omm.data(), sizeof(oh));
        const uint8_t* at = omm.data() + sizeof(oh);
        auto take = [&](auto& v, size_t n) {
            v.resize(n);
            std::memcpy(v.data(), at, n * sizeof(v[0]));
            at += n * sizeof(v[0]);
        };
        take(src.omm.descs, oh.descCount);
        take(src.omm.histogram, oh.histogramCount);
        take(src.omm.raw, oh.rawBytes);
    }

    src.path = path;
    src.pages.resize(h.groups);
    lod.depFirst.resize(h.groups + 1);
    // The file lists the deps in page order; the hierarchy keeps them in group order.
    std::vector<uint32_t> fileDeps;
    fileDeps.swap(lod.deps);
    lod.deps.reserve(fileDeps.size());
    uint32_t roots = 0;
    for (uint32_t g = 0; g < h.groups; ++g) {
        const FilePage& p = pages[g];
        src.pages[g] = PageEntry{ p.offset, p.bytes, p.firstCluster, p.clusterCount, p.vertexCount, p.triCount,
                                  (p.flags & PAGE_ALPHA) != 0, (uint16_t)(p.flags >> PAGE_FILE_SHIFT) };
        if ((uint64_t)p.depFirst + p.depCount > fileDeps.size()) return false;
        lod.depFirst[g] = (uint32_t)lod.deps.size();
        lod.deps.insert(lod.deps.end(), fileDeps.begin() + p.depFirst, fileDeps.begin() + p.depFirst + p.depCount);
        if (lod.groups[g].error == FLT_MAX && !(h.flags & FILE_INDEX)) roots += p.clusterCount;
    }
    lod.depFirst[h.groups] = (uint32_t)lod.deps.size();
    lod.clusters.resize(h.clusters);
    src.spans.resize(h.clusters);
    src.materials.resize(h.clusters);
    for (uint32_t c = 0; c < h.clusters; ++c) {
        lod.clusters[c] = LodCluster{ fc[c].group, fc[c].refined };
        src.spans[c] = Span{ fc[c].firstTri, fc[c].triCount, fc[c].uniqueVerts };
        src.materials[c] = fc[c].material;
    }
    if (h.flags & FILE_INDEX) {
        src.cells.resize(h.clusters);
        for (uint32_t c = 0; c < h.clusters; ++c) src.cells[c] = fc[c].pad;
    }
    lod.paged = true;
    lod.base = lod.fixed + roots;
    lod.resident.assign(h.groups, 0);
    lod.lastUsed.assign(h.groups, 0);
    lod.want.assign(h.groups, 0.0f);
    return true;
}

#if RDN_CLUSTER_API
namespace {
constexpr uint64_t OMM_UNIT = D3D12_RAYTRACING_OPACITY_MICROMAP_ARRAY_BYTE_ALIGNMENT;
inline uint64_t align(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

// A page's CLAS limits: an alpha page is geometry 1 (the alpha hit group) with micromaps.
D3D12_RTAS_CLUSTER_LIMITS page_limits(const PagedSource& src, uint32_t g) {
    const PageEntry& e = src.pages[g];
    return limits(src.spans.data() + e.firstCluster, e.clusterCount, e.alpha ? 1u : 0u, e.alpha);
}
} // namespace

const int32_t* Pager::omm_indices(const PageEntry& e, const uint8_t* data, bool coded) {
    if (!e.alpha) return nullptr;
    return reinterpret_cast<const int32_t*>(
        data + align((uint64_t)e.vertexCount * sizeof(BTriVertex) + (coded ? 4ull : 3ull) * e.triCount * sizeof(uint16_t), 4));
}

void Pager::omm_prebuild(const PageOmm& o, uint64_t& result, uint64_t& scratch) const {
    D3D12_RAYTRACING_OPACITY_MICROMAP_ARRAY_DESC desc{};
    desc.NumOmmHistogramEntries = (UINT)o.histogram.size();
    desc.pOmmHistogram = o.histogram.data();
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS in{};
    in.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_OPACITY_MICROMAP_ARRAY;
    in.NumDescs = 1;
    in.pOpacityMicromapArrayDesc = &desc;
    in.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO info{};
    m_ctx->Device()->GetRaytracingAccelerationStructurePrebuildInfo(&in, &info);
    result = align(info.ResultDataMaxSizeInBytes, OMM_UNIT);
    scratch = align(info.ScratchDataSizeInBytes, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT);
}

void Pager::omm_build(ID3D12GraphicsCommandList4* cl, const PageOmm& o, D3D12_GPU_VIRTUAL_ADDRESS rawVa,
                      D3D12_GPU_VIRTUAL_ADDRESS descsVa, D3D12_GPU_VIRTUAL_ADDRESS dest, D3D12_GPU_VIRTUAL_ADDRESS scratch) {
    D3D12_RAYTRACING_OPACITY_MICROMAP_ARRAY_DESC desc{};
    desc.NumOmmHistogramEntries = (UINT)o.histogram.size();
    desc.pOmmHistogram = o.histogram.data();
    desc.InputBuffer = rawVa;
    desc.PerOmmDescs = { descsVa, sizeof(D3D12_RAYTRACING_OPACITY_MICROMAP_DESC) };
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build{};
    build.Inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_OPACITY_MICROMAP_ARRAY;
    build.Inputs.NumDescs = 1;
    build.Inputs.pOpacityMicromapArrayDesc = &desc;
    build.Inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
    build.DestAccelerationStructureData = dest;
    build.ScratchAccelerationStructureData = scratch;
    cl->BuildRaytracingAccelerationStructure(&build, 0, nullptr);
}

Pager::~Pager() {
    {
        std::lock_guard<std::mutex> lk(m_ioMutex);
        m_ioStop = true;
    }
    m_ioWake.notify_all();
    for (auto& t : m_ioThreads) t.join();
    for (Mesh& m : m_meshes)
        for (HANDLE f : m.files)
            if (f != INVALID_HANDLE_VALUE) CloseHandle(f);
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
    m_ommPool = CreateBuffer(device, cfg.ommPoolBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                             D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nv_helpers_dx12::kDefaultHeapProps);
    m_ommPool->SetName(L"PageOmmPool");
    m_ommAlloc.init(cfg.ommPoolBytes / OMM_UNIT);
    m_ommIndexPool = CreateBuffer(device, std::max<uint64_t>(g.triAlloc->capacity() * sizeof(int32_t), 256),
                                  D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
    m_ommIndexPool->SetName(L"PageOmmIndices");
    {
        D3D12_RANGE noRead{ 0, 0 };
        ThrowIfFailed(m_ommIndexPool->Map(0, &noRead, reinterpret_cast<void**>(&m_ommIndexMapped)));
    }
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
    for (const std::string& path : m.src.files.empty() ? std::vector<std::string>{ m.src.path } : m.src.files) {
        m.files.push_back(open_read(path));
        if (m.files.back() == INVALID_HANDLE_VALUE) throw std::runtime_error("pager: cannot open " + path);
    }
    m.triBase = triBase;
    m.materialBase = m.src.materialBase;
    for (const PageEntry& e : m.src.pages) m.omm = m.omm || e.alpha;
    m.pages.resize(lod->groups.size());
    for (uint32_t g = 0; g < (uint32_t)lod->groups.size(); ++g) m.pages[g].pinned = lod->groups[g].error == FLT_MAX;
    lod->pager = (uint32_t)m_meshes.size() - 1;
    lod_levels(*lod, m.levels);
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
    const uint16_t* codes = m.src.coded ? local + 3ull * e.triCount : nullptr;
    thread_local std::vector<Span> records;
    records.resize(e.clusterCount);
    for (uint32_t k = 0; k < e.clusterCount; ++k) {
        const Span& s = m.src.spans[e.firstCluster + k];
        const uint32_t material = m.materialBase + m.src.materials[e.firstCluster + k];
        for (uint32_t t = 0; t < s.triCount; ++t) {
            const uint32_t mt = codes ? m.materialBase + codes[s.firstTri + t] : material;
            mats[p.tri + s.firstTri + t] = mt;
            if (gpuMaterials) gpuMaterials[p.tri + s.firstTri + t] = mt;
        }
        records[k] = Span{ (uint32_t)p.tri + s.firstTri - m.triBase, s.triCount, s.uniqueVerts };
    }
    m_geometry->write_records(m.lod->idBase + e.firstCluster, records.data(), e.clusterCount);
    thread_local std::vector<uint32_t> info;
    info.resize(e.clusterCount);
    for (uint32_t k = 0; k < e.clusterCount; ++k)
        info[k] = cluster_info(p.pinned ? CLUSTER_KIND_ROOT : CLUSTER_KIND_STREAMED, m.levels[e.firstCluster + k]);
    m_geometry->write_info(m.lod->idBase + e.firstCluster, info.data(), e.clusterCount);
    if (e.alpha) {
        const int32_t* src = omm_indices(e, data, m.src.coded);
        const bool array = m.ommUnits || m.src.ommArray;
        for (uint32_t t = 0; t < e.triCount; ++t)
            m_ommIndexMapped[p.tri + t] = src[t] >= 0 && !array
                                              ? D3D12_RAYTRACING_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_UNKNOWN_OPAQUE
                                              : src[t];
    }
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
    d.limits = page_limits(m.src, g);
    if (e.alpha) {
        d.opaqueTriCount = 0;
        d.alphaGeometry = 1;
        d.ommArray = m.src.ommArray ? m.src.ommArray
                     : m.ommUnits   ? m_ommPool->GetGPUVirtualAddress() + m.ommArray * OMM_UNIT
                                    : 0;
        d.ommIndices = m_ommIndexPool->GetGPUVirtualAddress() + m.pages[g].tri * sizeof(int32_t);
    } else {
        d.opaqueTriCount = e.triCount;
    }
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
    m_geometry->queue_move_to(m.lod->idBase + e.firstCluster, e.clusterCount, e.alpha, dests.data(), bytes, scratch);
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
    const uint64_t scratch = m_geometry->move_scratch(e.clusterCount, bytes, e.alpha, true);
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
    if (m_external) m_changes.push_back(ResidencyChange{ m.lod, g, e.firstCluster, e.clusterCount, true });
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
    m.pages[g].retiredAt = m_frame;
    m.lod->resident[g] = 0;
    if (m_external) m_changes.push_back(ResidencyChange{ m.lod, g, e.firstCluster, e.clusterCount, false });
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
    m_reviveBar = pressure;
    std::partial_sort(cold.begin(), cold.begin() + (ptrdiff_t)n, cold.end());
    for (size_t i = 0; i < n; ++i) {
        retire(std::get<1>(cold[i]), std::get<2>(cold[i]));
        ++m_stats.retiredTotal;
    }
}

bool Pager::make_room(const std::function<bool()>& test) {
    // Candidates once a frame: resident, not a root, no resident page depending on it, and no frame in flight tracing it.
    // With external feedback, lastUsed lags by the readback (a traced cut marks it every frame, so it stays within the
    // lag while traced), and frames in flight may have taken any resident page into their cuts: a resident page only
    // retires here, and its memory is freed once the frames before that have finished and no traced cut has held it
    // for a while (an instance whose rebuild waits keeps tracing its last cut).
    const uint64_t margin = m_geometry->frame_slots() + 2;
    if (!m_lruBuilt) {
        m_lruBuilt = true;
        m_lru.clear();
        m_lruNext = 0;
        const uint64_t unused = m_external ? 2 * margin : margin;
        for (uint32_t mi = 0; mi < (uint32_t)m_meshes.size(); ++mi) {
            const Mesh& m = m_meshes[mi];
            for (uint32_t g = 0; g < (uint32_t)m.pages.size(); ++g) {
                const Page& p = m.pages[g];
                if ((p.state != State::Resident && p.state != State::Retiring) || p.pinned || p.dependents) continue;
                if (m_frame < m.lod->lastUsed[g] + unused) continue;
                if (p.state == State::Retiring && m_external && m_frame < p.retiredAt + margin) continue;
                m_lru.emplace_back(p.state == State::Retiring ? 0 : m.lod->lastUsed[g], mi, g);
            }
        }
        std::sort(m_lru.begin(), m_lru.end());
    }
    while (m_lruNext < m_lru.size()) {
        const auto [used, mi, g] = m_lru[m_lruNext];
        const Page& p = m_meshes[mi].pages[g];
        if ((p.state != State::Resident && p.state != State::Retiring) || p.dependents) {
            ++m_lruNext;
            continue;
        }
        if (m_external && p.state == State::Resident) {
            if (!m_retiresLeft) return false;
            --m_retiresLeft;
            ++m_lruNext;
            retire(mi, g);
            ++m_stats.retiredTotal;
            continue;
        }
        ++m_lruNext;
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
    // Every mesh's micromaps, before any CLAS that references them; they stay.
    uint64_t ommWanted = 0;
    uint32_t ommMissing = 0;
    {
        uint64_t staging = 0, scratch = 0;
        std::vector<uint64_t> at(m_meshes.size()), scratchAt(m_meshes.size());
        for (size_t mi = 0; mi < m_meshes.size(); ++mi) {
            Mesh& m = m_meshes[mi];
            if (m.src.omm.descs.empty()) continue;
            uint64_t result = 0, sc = 0, off = 0;
            omm_prebuild(m.src.omm, result, sc);
            ommWanted += result;
            if (!m_ommAlloc.allocate(result / OMM_UNIT, off)) {
                // Its alpha triangles fall back to any-hit tests.
                ++ommMissing;
                m.src.omm = PageOmm{};
                continue;
            }
            m.ommArray = off;
            m.ommUnits = result / OMM_UNIT;
            at[mi] = staging = align(staging, OMM_UNIT);
            staging += align(m.src.omm.raw.size(), 16) + m.src.omm.descs.size() * sizeof(D3D12_RAYTRACING_OPACITY_MICROMAP_DESC);
            scratchAt[mi] = scratch;
            scratch += sc;
        }
        if (staging) {
            ComPtr<ID3D12Resource> stage = CreateBuffer(device, staging, D3D12_RESOURCE_FLAG_NONE,
                                                        D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
            ComPtr<ID3D12Resource> scratchBuf = scratchBuffer(scratch);
            uint8_t* mapped = nullptr;
            D3D12_RANGE noRead{ 0, 0 };
            ThrowIfFailed(stage->Map(0, &noRead, reinterpret_cast<void**>(&mapped)));
            for (size_t mi = 0; mi < m_meshes.size(); ++mi) {
                Mesh& m = m_meshes[mi];
                if (m.src.omm.descs.empty()) continue;
                const PageOmm& o = m.src.omm;
                const uint64_t descs = at[mi] + align(o.raw.size(), 16);
                std::memcpy(mapped + at[mi], o.raw.data(), o.raw.size());
                std::memcpy(mapped + descs, o.descs.data(), o.descs.size() * sizeof(o.descs[0]));
                omm_build(cl, o, stage->GetGPUVirtualAddress() + at[mi], stage->GetGPUVirtualAddress() + descs,
                          m_ommPool->GetGPUVirtualAddress() + m.ommArray * OMM_UNIT,
                          scratchBuf->GetGPUVirtualAddress() + scratchAt[mi]);
            }
            stage->Unmap(0, nullptr);
            const D3D12_RESOURCE_BARRIER b = CD3DX12_RESOURCE_BARRIER::UAV(nullptr);
            cl->ResourceBarrier(1, &b);
            flush();
        }
        for (Mesh& m : m_meshes) m.src.omm = PageOmm{};
    }

    for (size_t next = 0; next < roots.size();) {
        size_t end = next;
        uint64_t staging = 0, arena = 0, scratch = 0;
        uint32_t batchClusters = 0;
        std::vector<ClusterGeometry::Sizes> sz;
        std::vector<std::vector<uint8_t>> data;
        while (end < roots.size()) {
            Mesh& m = m_meshes[roots[end].first];
            const PageEntry& e = m.src.pages[roots[end].second];
            if (end > next && (staging + stagingBytes(e) > BATCH_STAGING || batchClusters + e.clusterCount > BATCH_CLUSTERS))
                break;
            ClusterGeometry::Sizes s;
            if (!m_geometry->prebuild(page_limits(m.src, roots[end].second), e.alpha, s))
                throw std::runtime_error("pager: CLAS prebuild query failed");
            sz.push_back(s);
            std::vector<uint8_t>& d = data.emplace_back(e.bytes);
            if (!read_at(m.files[e.file], e.offset, d.data(), e.bytes)) throw std::runtime_error("pager: cannot read " + m.src.path);
            m_stats.readBytes += e.bytes;
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
            if (!alloc_geometry(m, g, false))
                throw std::runtime_error("pager: the page pool cannot hold the roots; raise RDN_PAGE_POOL_VERTS / RDN_PAGE_POOL_TRIS");
            upload(m, g, data[i - next].data(), cl, stage.Get(), mapped, at, gpuMaterials);
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
            moveScratch += m_geometry->move_scratch(e.clusterCount, bytes, e.alpha, true);
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
    std::printf("[pager] micromaps %.1f MB in a %.0f MB pool%s\n", (double)ommWanted / 1048576.0,
                (double)m_cfg.ommPoolBytes / 1048576.0, ommMissing ? "; some meshes did not fit and test alpha in any-hit" : "");
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
    m_retiresLeft = m_cfg.retiresPerFrame;
    if (!m_external)
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

bool Pager::install(uint32_t mi, uint32_t g, ID3D12GraphicsCommandList* copy, ID3D12GraphicsCommandList4* compute,
                    Arena& a, uint32_t* gpuMaterials) {
    Mesh& m = m_meshes[mi];
    const PageEntry& e = m.src.pages[g];
    Page& p = m.pages[g];
    // This frame's room first: arena, scratch, staging, arguments.
    (void)compute;
    ClusterGeometry::Sizes sz;
    if (!m_geometry->prebuild(page_limits(m.src, g), e.alpha, sz)) return false;
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

void Pager::installs(std::vector<LodRequest>& requests, ID3D12GraphicsCommandList* copy, ID3D12GraphicsCommandList4* compute) {
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
            if (p.state == State::Retiring) { // still in memory: back into the cuts, once its deps are
                bool ready = true;
                for (uint32_t d = m.lod->depFirst[q.group]; d < m.lod->depFirst[q.group + 1] && ready; ++d)
                    ready = m.lod->resident[m.lod->deps[d]] != 0;
                // Not one the pools just gave up for better pages.
                if (ready && q.priority * 2.0f >= m_reviveBar) make_resident(q.mesh, q.group);
                continue;
            }
            if (p.state != State::Absent) continue;
            const PageEntry& e = m.src.pages[q.group];
            p.state = State::Reading;
            ++m_reading;
            m_ioQueue.push_back(Read{ q.mesh, q.group, m.files[e.file], e.offset, e.bytes, {}, false });
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
                !install(mi, g, copy, compute, a, gpuMaterials)) {
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
    m_stats.droppedTotal += m_stats.dropped;
    m_stats.reading = m_reading + (uint32_t)m_read.size();
    m_stats.building = (uint32_t)m_inFlight.size();
    m_stats.clasUsed = m_clasAlloc.used() * CLAS_UNIT;
    m_stats.clasCapacity = m_cfg.clasPoolBytes;
    m_stats.trisUsed = m_geo.triAlloc->used();
    m_stats.trisCapacity = m_geo.triAlloc->capacity();
    m_stats.vertsUsed = m_geo.vertexAlloc->used();
    m_stats.vertsCapacity = m_geo.vertexAlloc->capacity();

    // What the cuts need: the pages they used lately or bring in with the deps that hold them up, and the absent ones
    // they ask for (their CLAS at the pool's present bytes per triangle). Pages kept only because nothing needed their
    // room yet do not count.
    const uint64_t recent = (m_geometry->frame_slots() + 2) * (m_external ? 2 : 1);
    uint64_t tris = 0, verts = 0, clasUnits = 0, wantedTris = 0;
    uint32_t retiring = 0;
    thread_local std::vector<uint8_t> needed;
    thread_local std::vector<uint32_t> stack;
    for (const Mesh& m : m_meshes) {
        needed.assign(m.pages.size(), 0);
        stack.clear();
        for (uint32_t g = 0; g < (uint32_t)m.pages.size(); ++g) {
            const Page& p = m.pages[g];
            retiring += p.state == State::Retiring;
            if (p.state == State::Building || p.state == State::Built ||
                (p.state == State::Resident && (p.pinned || m_frame < m.lod->lastUsed[g] + recent))) {
                needed[g] = 1;
                stack.push_back(g);
            }
        }
        while (!stack.empty()) {
            const uint32_t g = stack.back();
            stack.pop_back();
            for (uint32_t i = m.lod->depFirst[g]; i < m.lod->depFirst[g + 1]; ++i)
                if (!needed[m.lod->deps[i]] && m.pages[m.lod->deps[i]].state == State::Resident) {
                    needed[m.lod->deps[i]] = 1;
                    stack.push_back(m.lod->deps[i]);
                }
        }
        for (uint32_t g = 0; g < (uint32_t)m.pages.size(); ++g)
            if (needed[g]) {
                tris += m.src.pages[g].triCount;
                verts += m.src.pages[g].vertexCount;
                clasUnits += m.pages[g].clasUnits;
            }
    }
    thread_local std::unordered_set<uint64_t> counted;
    counted.clear();
    for (const LodRequest& q : requests) {
        const State st = m_meshes[q.mesh].pages[q.group].state;
        if ((st != State::Absent && st != State::Reading && st != State::Read) ||
            !counted.insert((uint64_t)q.mesh << 32 | q.group).second)
            continue;
        wantedTris += m_meshes[q.mesh].src.pages[q.group].triCount;
        verts += m_meshes[q.mesh].src.pages[q.group].vertexCount;
    }
    const double clasPerTri = m_stats.trisUsed ? (double)m_stats.clasUsed / (double)m_stats.trisUsed : 0.0;
    auto share = [](double used, uint64_t capacity) { return capacity ? (float)(used / (double)capacity) : 0.0f; };
    m_stats.pressure = std::max({ share((double)(tris + wantedTris), m_stats.trisCapacity),
                                  share((double)verts, m_stats.vertsCapacity),
                                  share((double)clasUnits * CLAS_UNIT + (double)wantedTris * clasPerTri,
                                        m_stats.clasCapacity) });

    // Starved: leaves wanted far less than the best waiting page leave the cuts. With external feedback, not while
    // pages already retired wait out the frames in flight before their memory frees.
    if (m_starved && !(m_external && retiring)) {
        float pressure = 0.0f;
        for (const LodRequest& q : requests)
            if (m_meshes[q.mesh].pages[q.group].state != State::Resident) {
                pressure = q.priority;
                break;
            }
        if (pressure > 0.0f) rebalance(pressure);
    }
    if (!m_starved) m_reviveBar = 0.0f;
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
