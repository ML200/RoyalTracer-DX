#include "../stdafx.h"
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <unordered_map>
#include "Scene.h"
#include <meshoptimizer.h>
#include "../ocean/OceanCommon.h"
#include "../DXRHelper.h"

void MeshGPU::CreateBlasBuildInputs(ID3D12Device* device) {
    auto upload = [&](const void* data, size_t bytes, ComPtr<ID3D12Resource>& resource) {
        if (resource)
            return;
        resource = nv_helpers_dx12::CreateBuffer(device, bytes, D3D12_RESOURCE_FLAG_NONE,
                                                 D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
        void* mapped = nullptr;
        CD3DX12_RANGE noRead(0, 0);
        ThrowIfFailed(resource->Map(0, &noRead, &mapped));
        if (bytes)
            memcpy(mapped, data, bytes);
        resource->Unmap(0, nullptr);
    };
    upload(cpuVertices.data(), cpuVertices.size() * sizeof(Vertex), vertexBuffer);
    upload(cpuIndices.data(), cpuIndices.size() * sizeof(UINT), indexBuffer);
}

namespace {
// Reorders triangles [b, e) into compact octree cells, appending each cell's first triangle to cuts.
void order_cells(const std::vector<Vertex>& vertices, std::vector<UINT>& indices, std::vector<UINT>& materials, UINT b,
                 UINT e, std::vector<UINT>& cuts) {
    if (b == e) return;
    std::vector<float> centroids(3ull * (e - b), 0.0f);
    std::vector<uint32_t> order, cells;
    for (UINT t = b; t < e; ++t)
        for (int k = 0; k < 3; ++k) {
            const XMFLOAT3& p = vertices[indices[3ull * t + k]].position;
            centroids[3ull * (t - b) + 0] += p.x / 3.0f;
            centroids[3ull * (t - b) + 1] += p.y / 3.0f;
            centroids[3ull * (t - b) + 2] += p.z / 3.0f;
        }
    clusters::morton_cells(centroids.data(), e - b, 1, clusters::MAX_TRIS, order, cells);
    const std::vector<UINT> idx(indices.begin() + 3ull * b, indices.begin() + 3ull * e);
    const std::vector<UINT> mat(materials.begin() + b, materials.begin() + e);
    for (UINT i = 0; i < e - b; ++i) {
        for (int k = 0; k < 3; ++k) indices[3ull * (b + i) + k] = idx[3ull * order[i] + k];
        materials[b + i] = mat[order[i]];
    }
    for (uint32_t c : cells) cuts.push_back(b + c);
}
} // namespace

bool MeshGPU::BuildPagedLod(std::vector<uint32_t> regions, uint32_t materialBase, std::function<bool(uint32_t)> foliage,
                            decltype(clusters::BakeMesh::opacity) opacity) {
    using clusters::NO_REGION;
    const UINT triCount = (UINT)(cpuIndices.size() / 3);
    regions.resize(triCount, NO_REGION);
    std::unordered_map<uint32_t, uint32_t> regionTris;
    for (UINT t = 0; t < triCount; ++t)
        if (regions[t] != NO_REGION) ++regionTris[regions[t]];
    UINT lodSourceTris = 0;
    for (UINT t = 0; t < triCount; ++t) {
        if (regions[t] != NO_REGION && regionTris[regions[t]] < clusters::MIN_LOD_TRIS) regions[t] = NO_REGION;
        lodSourceTris += regions[t] != NO_REGION ? 1u : 0u;
    }
    if (!lodSourceTris) {
        OrderForClusters();
        return false;
    }

    // Materials simplify together, each triangle keeping its own (a simplified one the one covering most around it), as
    // Nanite's clusters mix them: a material border is no lock. Only foliage and the rest are regions apart, as they
    // simplify differently.
    constexpr uint32_t REGION_SURFACE = 0, REGION_FOLIAGE = 1;
    std::vector<uint32_t> codes(triCount, 0), classes(triCount, NO_REGION);
    for (UINT t = 0; t < triCount; ++t)
        if (regions[t] != NO_REGION) {
            codes[t] = regions[t] - materialBase;
            classes[t] = foliage && foliage(regions[t]) ? REGION_FOLIAGE : REGION_SURFACE;
        }

    // Material borders as seams: a vertex shared across materials splits into one per material, and every vertex at such a
    // position is protected, so a border moves only along itself and no material spreads over another's surface.
    std::vector<Vertex> lodVertices;
    std::vector<UINT> lodIndices(cpuIndices.size());
    std::vector<uint8_t> lodFlags;
    {
        std::unordered_map<uint64_t, UINT> split;
        for (UINT t = 0; t < triCount; ++t)
            for (int k = 0; k < 3; ++k) {
                const UINT v = cpuIndices[3ull * t + k];
                const auto [it, fresh] = split.emplace((uint64_t)v << 32 | codes[t], (UINT)lodVertices.size());
                if (fresh) lodVertices.push_back(cpuVertices[v]);
                lodIndices[3ull * t + k] = it->second;
            }
        const size_t n = lodVertices.size();
        std::vector<uint32_t> pos(n);
        meshopt_generatePositionRemap(pos.data(), &lodVertices[0].position.x, n, sizeof(Vertex));
        constexpr uint32_t NONE = ~0u, MIXED = ~0u - 1;
        std::vector<uint32_t> owner(n, NONE);
        for (UINT t = 0; t < triCount; ++t)
            for (int k = 0; k < 3; ++k) {
                uint32_t& o = owner[pos[lodIndices[3ull * t + k]]];
                o = o == NONE ? codes[t] : o == codes[t] ? o : MIXED;
            }
        lodFlags.assign(n, 0);
        for (size_t i = 0; i < n; ++i)
            if (owner[pos[i]] == MIXED) lodFlags[i] = meshopt_SimplifyVertex_Protect;
    }

    // The page file is named after what it is built from: vertices, triangles, their model-relative materials and regions.
    uint64_t key = 0xcbf29ce484222325ull ^ 7u; // FNV-1a over words; the xor is the page format and builder version
    auto mix = [&](const void* data, size_t bytes) {
        const uint8_t* p = static_cast<const uint8_t*>(data);
        for (; bytes >= 8; bytes -= 8, p += 8) {
            uint64_t w;
            std::memcpy(&w, p, 8);
            key = (key ^ w) * 0x100000001b3ull;
        }
        for (; bytes; --bytes, ++p) key = (key ^ *p) * 0x100000001b3ull;
    };
    mix(cpuVertices.data(), cpuVertices.size() * sizeof(Vertex));
    mix(cpuIndices.data(), cpuIndices.size() * sizeof(UINT));
    mix(codes.data(), codes.size() * sizeof(uint32_t));
    mix(classes.data(), classes.size() * sizeof(uint32_t));
    char name[32];
    std::snprintf(name, sizeof(name), "%016llx.rdnpages", (unsigned long long)key);
    const std::string path = clusters::cache_dir() + "\\" + name;

    clusters::LodSource src;
    src.vertices = reinterpret_cast<const uint8_t*>(lodVertices.data());
    src.stride = sizeof(Vertex);
    src.normalOffset = offsetof(Vertex, normal_material);
    src.uvOffset = offsetof(Vertex, texCoord);
    src.vertexCount = (uint32_t)lodVertices.size();
    src.indices = lodIndices.data();
    src.vertexFlags = lodFlags.data();
    src.regions = classes.data();
    src.triCount = triCount;
    src.foliage = [](uint32_t region) { return region == REGION_FOLIAGE; };
    src.codes = codes.data();
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
        clusters::LodBuild b;
        clusters::build_lod(src, b);
        if (b.clusters.empty()) {
            OrderForClusters();
            return false;
        }
        const clusters::BakeMesh bake{ src.vertices, src.stride, src.normalOffset, src.uvOffset, materialBase, opacity };
        if (!clusters::write_pages(path, b, bake)) {
            std::printf("[pager] cannot write %s; this mesh keeps every level resident\n", path.c_str());
            return BuildClusterLod(std::move(regions));
        }
    }

    // The mesh keeps the triangles outside the hierarchy, on the vertices they use: opaque, then alpha, in cells.
    std::vector<UINT> remap(cpuVertices.size(), ~0u), idx, mat;
    std::vector<Vertex> verts;
    UINT fixedOpaque = 0;
    for (UINT t = 0; t < triCount; ++t) {
        if (regions[t] != NO_REGION) continue;
        for (int k = 0; k < 3; ++k) {
            UINT& r = remap[cpuIndices[3ull * t + k]];
            if (r == ~0u) {
                r = (UINT)verts.size();
                verts.push_back(cpuVertices[cpuIndices[3ull * t + k]]);
            }
            idx.push_back(r);
        }
        mat.push_back(cpuMaterialIDs[t]);
        fixedOpaque += t < opaqueTriCount ? 1u : 0u;
    }
    cpuVertices = std::move(verts);
    cpuIndices = std::move(idx);
    cpuMaterialIDs = std::move(mat);
    vertexCount = (UINT)cpuVertices.size();
    indexCount = (UINT)cpuIndices.size();
    opaqueTriCount = fixedOpaque;
    alphaTriCount = indexCount / 3 - fixedOpaque;
    clusterCuts.clear();
    std::vector<UINT> cuts;
    order_cells(cpuVertices, cpuIndices, cpuMaterialIDs, 0, opaqueTriCount, cuts);
    order_cells(cpuVertices, cpuIndices, cpuMaterialIDs, opaqueTriCount, indexCount / 3, cuts);
    clusterSpans.clear();
    if (!cuts.empty()) clusters::split(cpuIndices.data(), vertexCount, cuts, indexCount / 3, clusterSpans);
    lod = clusters::Lod{};
    lod.fixed = (uint32_t)clusterSpans.size();
    pagedSource = std::make_shared<clusters::PagedSource>();
    if (!clusters::read_pages(path, lod, *pagedSource)) throw std::runtime_error("cannot read page file " + path);
    pagedSource->materialBase = materialBase;
    lodAlpha = std::any_of(pagedSource->pages.begin(), pagedSource->pages.end(), [](const clusters::PageEntry& e) { return e.alpha; });
    paged = true;
    return true;
}

void MeshGPU::AdoptPages(clusters::Lod&& pages, std::shared_ptr<clusters::PagedSource> source, std::vector<Vertex> vertices,
                         std::vector<UINT> indices, std::vector<UINT> materials, UINT opaqueTris) {
    cpuVertices = std::move(vertices);
    cpuIndices = std::move(indices);
    cpuMaterialIDs = std::move(materials);
    vertexCount = (UINT)cpuVertices.size();
    indexCount = (UINT)cpuIndices.size();
    opaqueTriCount = opaqueTris;
    alphaTriCount = indexCount / 3 - opaqueTris;
    clusterCuts.clear();
    std::vector<UINT> cuts;
    order_cells(cpuVertices, cpuIndices, cpuMaterialIDs, 0, opaqueTriCount, cuts);
    order_cells(cpuVertices, cpuIndices, cpuMaterialIDs, opaqueTriCount, indexCount / 3, cuts);
    clusterSpans.clear();
    if (!cuts.empty()) clusters::split(cpuIndices.data(), vertexCount, cuts, indexCount / 3, clusterSpans);
    lod = std::move(pages);
    lod.fixed = (uint32_t)clusterSpans.size();
    lod.base += lod.fixed;
    pagedSource = std::move(source);
    lodAlpha = std::any_of(pagedSource->pages.begin(), pagedSource->pages.end(), [](const clusters::PageEntry& e) { return e.alpha; });
    paged = true;
}

void MeshGPU::OrderForClusters() {
    clusterCuts.clear();
    const UINT triCount = (UINT)(cpuIndices.size() / 3);
    order_cells(cpuVertices, cpuIndices, cpuMaterialIDs, 0, opaqueTriCount, clusterCuts);
    order_cells(cpuVertices, cpuIndices, cpuMaterialIDs, opaqueTriCount, triCount, clusterCuts);
}

bool MeshGPU::BuildClusterLod(std::vector<uint32_t> regions) {
    using clusters::NO_REGION;
    const UINT triCount = (UINT)(cpuIndices.size() / 3);
    regions.resize(triCount, NO_REGION);
    std::fill(regions.begin() + opaqueTriCount, regions.end(), NO_REGION);
    std::unordered_map<uint32_t, uint32_t> regionTris;
    for (UINT t = 0; t < opaqueTriCount; ++t)
        if (regions[t] != NO_REGION) ++regionTris[regions[t]];
    UINT lodSourceTris = 0;
    for (UINT t = 0; t < opaqueTriCount; ++t) {
        if (regions[t] != NO_REGION && regionTris[regions[t]] < clusters::MIN_LOD_TRIS) regions[t] = NO_REGION;
        lodSourceTris += regions[t] != NO_REGION ? 1u : 0u;
    }
    clusters::LodBuild b;
    if (lodSourceTris) {
        clusters::LodSource src;
        src.vertices = reinterpret_cast<const uint8_t*>(cpuVertices.data());
        src.stride = sizeof(Vertex);
        src.normalOffset = offsetof(Vertex, normal_material);
        src.uvOffset = offsetof(Vertex, texCoord);
        src.vertexCount = (uint32_t)cpuVertices.size();
        src.indices = cpuIndices.data();
        src.regions = regions.data();
        src.triCount = triCount;
        clusters::build_lod(src, b);
    }
    if (b.clusters.empty()) {
        OrderForClusters();
        return false;
    }
    // The hierarchy's triangles, then the opaque ones outside it, then the alpha ones; the latter two in octree cells.
    std::vector<UINT> idx = std::move(b.indices), mat = std::move(b.regions);
    const UINT lodTris = (UINT)(idx.size() / 3);
    for (UINT t = 0; t < triCount; ++t) {
        if (regions[t] != NO_REGION) continue;
        idx.insert(idx.end(), cpuIndices.begin() + 3ull * t, cpuIndices.begin() + 3ull * (t + 1));
        mat.push_back(cpuMaterialIDs[t]);
    }
    cpuIndices = std::move(idx);
    cpuMaterialIDs = std::move(mat);
    const UINT total = (UINT)(cpuIndices.size() / 3);
    opaqueTriCount = lodTris + (opaqueTriCount - lodSourceTris);
    indexCount = (UINT)cpuIndices.size();
    clusterCuts.clear();
    std::vector<UINT> cuts;
    order_cells(cpuVertices, cpuIndices, cpuMaterialIDs, lodTris, opaqueTriCount, cuts);
    order_cells(cpuVertices, cpuIndices, cpuMaterialIDs, opaqueTriCount, total, cuts);
    clusterSpans.clear();
    if (!cuts.empty()) clusters::split(cpuIndices.data(), vertexCount, cuts, total, clusterSpans);
    lod.fixed = (uint32_t)clusterSpans.size();
    lod.base = lod.fixed + b.originalClusters;
    lod.coarseTriBase = b.originalClusters ? b.spans[b.originalClusters - 1].firstTri + b.spans[b.originalClusters - 1].triCount : 0;
    lod.coarseTriCount = lodTris - lod.coarseTriBase;
    clusterSpans.insert(clusterSpans.end(), b.spans.begin(), b.spans.end()); // the hierarchy's triangles come first
    lod.clusters = std::move(b.clusters);
    lod.groups = std::move(b.groups);
    lod.clas.clear();
    return true;
}

void Scene::MarkModelMoved(UINT modelIndex) {
    if (modelIndex >= models.size())
        return;
    auto& model = models[modelIndex];
    model.RebuildTransform();

    for (UINT i = model.instanceStart; i < model.instanceStart + model.instanceCount; ++i) {
        instances[i].worldTransform = instances[i].localTransform * model.worldTransform;
        MarkInstanceDirty(i);
    }

    tlasDirty = true;
    lightTreeDirty = true;
}

void Scene::MarkMaterialsDirty(bool emissionChanged) {
    materialsDirty = true;
    if (emissionChanged) {
        lightTreeDirty = true;
        emissivesDirty = true;
    }
}

void Scene::MarkInstanceDirty(UINT instanceIndex) {
    if (instanceIndex < instanceDirty.size())
        instanceDirty[instanceIndex] = 1;
}

void Scene::MarkAllInstancesDirty() {
    std::fill(instanceDirty.begin(), instanceDirty.end(), 1);
}

void Scene::ReserveTerrain(UINT vertexElems, UINT indexElems, UINT matIDElems, UINT triLightElems, UINT instanceSlots,
                           UINT propsBase) {
    terrainVertexElems = vertexElems;
    terrainIndexElems = indexElems;
    terrainMatIDElems = matIDElems;
    terrainTriLightElems = triLightElems;
    terrainInstanceSlots = instanceSlots;
    terrainPropsBase = propsBase;

    Material tm;
    tm.Kd = XMFLOAT4(0.42f, 0.36f, 0.30f, 1.0f);
    tm.Ke = XMFLOAT3(0.0f, 0.0f, 0.0f);
    tm.Ni = 1.0f;
    tm.Pr_Pm_Ps_Pc = XMFLOAT4(1.0f, 0.0f, 0.0f, 0.0f);
    tm.albedoTexID = -1;
    tm.normalTexID = -1;
    tm.rmaTexID = -1;
    tm.alphaThreshold = 1.0f;
    materials.push_back(tm);
    terrainMatIndex = (UINT)materials.size() - 1;
}

void Scene::ReserveOcean(UINT vertexElems, UINT indexElems, UINT matIDElems, UINT instanceSlots,
                         const Material& mat, float foamAlbedo) {
    // Read before the ocean's slots count, so it lands last.
    oceanPropsBase = instancePropsCount();
    oceanVertexElems = vertexElems;
    oceanIndexElems = indexElems;
    oceanMatIDElems = matIDElems;
    oceanInstanceSlots = instanceSlots;

    oceanMatIndex = (UINT)materials.size();
    oceanMaterialEdited = false;
    // Foam ramps per bubble density, then the diagnostic slot.
    for (uint32_t i = 0; i < OCEAN_MATERIAL_COUNT; ++i) {
        Material layer = mat;
        ocean::WriteMaterialSlot(i, mat.Kd, mat.Tf, mat.sssWeight, mat.sssEnable, foamAlbedo, layer.Kd,
                                 layer.sssWeight, layer.sssEnable);
        materials.push_back(layer);
    }
    materialNames.resize(materials.size());
    materialNames[oceanMatIndex] = "Water";
}

void Scene::ReserveInstanceSets(UINT slots) {
    if (setSlots >= slots)
        return;
    setSlots = 0;
    setPropsBase = instancePropsCount();
    setSlots = slots;
}

void Scene::WriteSetRecord(UINT slot, UINT meshIndex) {
    const MeshGPU& mesh = meshes[meshIndex];
    InstanceProperties p = {};
    p.objectToWorld = MakeFloat3x4(XMMatrixIdentity());
    p.objectToWorldInverse = p.objectToWorld;
    p.objectToWorldNormal = p.objectToWorld;
    p.prevObjectToWorld = p.objectToWorld;
    p.indexBase = mesh.globalIndexBase;
    p.vertexBase = mesh.globalVertexBase;
    p.materialBase = mesh.materialIDBase;
    p.opaqueTriCount = mesh.opaqueTriCount;
    p.triToLightBase = 0xFFFFFFFFu;
    p.lightSlot = 0xFFFFFFFFu;
    uint8_t* dst = nullptr;
    const CD3DX12_RANGE noRead(0, 0);
    ThrowIfFailed(instanceProperties->Map(0, &noRead, reinterpret_cast<void**>(&dst)));
    std::memcpy(dst + (size_t)(setPropsBase + slot) * sizeof(InstanceProperties), &p, sizeof(p));
    instanceProperties->Unmap(0, nullptr);
}

void Scene::ReserveRocks(UINT instanceSlots) {
    rockInstanceSlots = instanceSlots;
    rockPropsBase = terrainPropsBase + terrainInstanceSlots;
}

void Scene::ReserveVoxels(UINT vertexElems, UINT indexElems, UINT matIDElems, UINT instanceSlots, UINT minPropsBase) {
    voxelVertexElems = vertexElems;
    voxelIndexElems = indexElems;
    voxelMatIDElems = matIDElems;
    voxelInstanceSlots = instanceSlots;

    const UINT afterRocks = terrainInstanceSlots ? (rockPropsBase + rockInstanceSlots) : 0u;
    voxelPropsBase = std::max(minPropsBase, afterRocks);
}

bool Scene::ReservePagePool() { return ReservePagePool(clusters::pool_sizes()); }

bool Scene::ReservePagePool(const clusters::PoolSizes& s) {
    if (!meshes.empty() || !materialIDs.empty())
        return false;
    pagePoolVertices = s.vertices;
    pagePoolTris = s.tris;
    pagePoolVertexAlloc.init(s.vertices);
    pagePoolTriAlloc.init(s.tris);
    materialIDs.assign(s.tris, 0u);
    return true;
}

void Scene::FitPagePool() {
    if (!pagePoolTris || clusters::pool_sizes().fixed)
        return;
    uint64_t tris = 0, verts = 0;
    for (const MeshGPU& m : meshes) {
        if (!m.paged || !m.pagedSource)
            continue;
        tris += m.indexCount / 3;
        verts += m.vertexCount;
        // Full detail needs every coarser page resident too (a page stays only while its deps do): the whole hierarchy.
        for (const clusters::PageEntry& e : m.pagedSource->pages) {
            tris += e.triCount;
            verts += e.vertexCount;
        }
    }
    // An eighth more for pages on their way in and out; the vertices stay within the 4 GiB view.
    constexpr uint64_t MAX_TRIS = 96ull << 20, MAX_VERTS = 120ull << 20;
    const UINT wantTris = (UINT)std::min<uint64_t>(tris + tris / 8, MAX_TRIS);
    const UINT wantVerts = (UINT)std::min<uint64_t>(verts + verts / 8, MAX_VERTS);
    if (wantTris <= pagePoolTris && wantVerts <= pagePoolVertices)
        return;
    const UINT addTris = std::max(wantTris, pagePoolTris) - pagePoolTris;
    // Material IDs past the pool (meshes outside it) move up with it.
    materialIDs.insert(materialIDs.begin() + pagePoolTris, addTris, 0u);
    for (MeshGPU& m : meshes)
        if (!m.paged && m.cellOf < 0)
            m.materialIDBase += addTris;
    pagePoolTris += addTris;
    pagePoolVertices = std::max(wantVerts, pagePoolVertices);
    pagePoolTriAlloc.grow(pagePoolTris);
    pagePoolVertexAlloc.grow(pagePoolVertices);
    std::printf("[pager] page pool fitted to the scene: %.1fM triangles, %.1fM vertices\n", pagePoolTris / 1048576.0,
                pagePoolVertices / 1048576.0);
}

void Scene::ReleasePagePool() {
    if (!pagePoolTris)
        return;
    materialIDs.erase(materialIDs.begin(), materialIDs.begin() + pagePoolTris);
    for (MeshGPU& m : meshes)
        m.materialIDBase -= pagePoolTris;
    pagePoolVertices = pagePoolTris = 0;
    pagePoolVertexAlloc.init(0);
    pagePoolTriAlloc.init(0);
}

void Scene::BuildGlobalMeshBuffers(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList) {
    SCOPE_TIMER("BuildGlobalMeshBuffers");

    // All geometry shares one buffer, after the page pool; paged meshes live in the pool.
    geoOffsets.resize(meshes.size());
    size_t totalV = pagePoolVertices, totalI = 3ull * pagePoolTris;

    for (size_t m = 0; m < meshes.size(); ++m) {
        if (meshes[m].cellOf >= 0) {
            const MeshGPU& owner = meshes[meshes[m].cellOf];
            meshes[m].globalVertexBase = owner.globalVertexBase;
            meshes[m].globalIndexBase = owner.globalIndexBase;
            meshes[m].materialIDBase = owner.materialIDBase;
            geoOffsets[m] = {owner.globalVertexBase, owner.globalIndexBase, owner.materialIDBase};
            continue;
        }
        if (meshes[m].paged) {
            meshes[m].globalVertexBase = meshes[m].poolVertexBase;
            meshes[m].globalIndexBase = 3u * meshes[m].poolTriBase;
            geoOffsets[m] = {meshes[m].globalVertexBase, meshes[m].globalIndexBase, meshes[m].materialIDBase};
            continue;
        }
        meshes[m].globalVertexBase = (UINT)totalV;
        meshes[m].globalIndexBase = (UINT)totalI;
        geoOffsets[m] = {(UINT)totalV, (UINT)totalI, meshes[m].materialIDBase};
        totalV += meshes[m].vertexCount;
        totalI += meshes[m].indexCount;
    }
    totalVertexCount = (UINT)totalV;
    totalIndexCount = (UINT)totalI;

    terrainVertexBase = totalVertexCount;
    terrainIndexBase = totalIndexCount;

    voxelVertexBase = totalVertexCount + terrainVertexElems;
    voxelIndexBase = totalIndexCount + terrainIndexElems;

    oceanVertexBase = voxelVertexBase + voxelVertexElems;
    oceanIndexBase = voxelIndexBase + voxelIndexElems;

    const uint64_t vbBytes = (uint64_t)combinedVertexCount() * sizeof(BTriVertex);
    const uint64_t ibBytes = (uint64_t)combinedIndexCount() * sizeof(uint32_t);

    // 32-bit view extent; overruns surface later as device removal.
    constexpr uint64_t kMaxViewBytes = 1ull << 32;
    if (vbBytes >= kMaxViewBytes) {
        char msg[512];
        snprintf(msg, sizeof(msg),
                 "Global vertex buffer needs %.2f GiB, past the 4 GiB a structured-buffer view can address "
                 "(scene %u + terrain %u + voxels %u + ocean %u vertices). Lower the Minecraft streamer's "
                 "vertexCapacity.",
                 (double)vbBytes / (1024.0 * 1024.0 * 1024.0), totalVertexCount, terrainVertexElems,
                 voxelVertexElems, oceanVertexElems);
        throw std::runtime_error(msg);
    }

    const uint64_t sceneVbBytes = (uint64_t)totalVertexCount * sizeof(BTriVertex);
    const uint64_t sceneIbBytes = (uint64_t)totalIndexCount * sizeof(uint32_t);

    // Only CPU-written terrain needs an upload heap.
    const bool hasTerrain = terrainVertexElems > 0 || terrainIndexElems > 0;

    // Ocean writes vertices via UAV; upload heaps can't be UAVs.
    const bool hasOcean = oceanVertexElems > 0;
    if (hasTerrain && (hasOcean || hostWritesGeometry))
        throw std::runtime_error("Streamed terrain cannot share the global vertex buffer with the ocean or a host");
    const bool computeWrites = hasOcean || hostWritesGeometry;
    const D3D12_RESOURCE_FLAGS vbFlags =
        computeWrites ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
    // The compute queue rejects GENERIC_READ.
    const D3D12_RESOURCE_STATES vbReadState =
        computeWrites ? D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE : D3D12_RESOURCE_STATE_GENERIC_READ;
    globalReadState = hasTerrain ? D3D12_RESOURCE_STATE_GENERIC_READ : vbReadState;

    uint8_t* dstVertsRaw;
    uint8_t* dstIdxRaw;
    if (hasTerrain) {
        vertexGlobal =
            nv_helpers_dx12::CreateBuffer(device, vbBytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ,
                                          nv_helpers_dx12::kUploadHeapProps);
        vertexGlobal->SetName(L"GlobalVertexBuffer");
        indexGlobal =
            nv_helpers_dx12::CreateBuffer(device, ibBytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ,
                                          nv_helpers_dx12::kUploadHeapProps);
        indexGlobal->SetName(L"GlobalIndexBuffer");

        CD3DX12_RANGE noRead(0, 0);
        vertexGlobal->Map(0, &noRead, (void**)&vertexGlobalMapped);
        indexGlobal->Map(0, &noRead, (void**)&indexGlobalMapped);
        dstVertsRaw = vertexGlobalMapped;
        dstIdxRaw = indexGlobalMapped;
    } else {
        vertexGlobal = nv_helpers_dx12::CreateBuffer(device, vbBytes, vbFlags, D3D12_RESOURCE_STATE_COPY_DEST,
                                                     nv_helpers_dx12::kDefaultHeapProps);
        vertexGlobal->SetName(L"GlobalVertexBuffer");
        indexGlobal = nv_helpers_dx12::CreateBuffer(device, ibBytes, D3D12_RESOURCE_FLAG_NONE,
                                                    D3D12_RESOURCE_STATE_COPY_DEST, nv_helpers_dx12::kDefaultHeapProps);
        indexGlobal->SetName(L"GlobalIndexBuffer");

        // Static ocean indices, staged once.
        const uint64_t stagedVbBytes = sceneVbBytes;
        const uint64_t stagedIbBytes =
            sceneIbBytes + (uint64_t)oceanIndexElems * sizeof(uint32_t);

        vertexGlobalUpload =
            nv_helpers_dx12::CreateBuffer(device, std::max<uint64_t>(stagedVbBytes, 256), D3D12_RESOURCE_FLAG_NONE,
                                          D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
        indexGlobalUpload =
            nv_helpers_dx12::CreateBuffer(device, std::max<uint64_t>(stagedIbBytes, 256), D3D12_RESOURCE_FLAG_NONE,
                                          D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);

        CD3DX12_RANGE noRead(0, 0);
        vertexGlobalUpload->Map(0, &noRead, (void**)&dstVertsRaw);
        indexGlobalUpload->Map(0, &noRead, (void**)&dstIdxRaw);
        vertexGlobalMapped = nullptr;
        indexGlobalMapped = nullptr;
    }

    auto* dstVerts = reinterpret_cast<BTriVertex*>(dstVertsRaw);
    auto* dstIdx = reinterpret_cast<uint32_t*>(dstIdxRaw);

    for (size_t m = 0; m < meshes.size(); ++m) {
        const auto& mesh = meshes[m];
        const UINT vBase = mesh.globalVertexBase;

        BTriVertex* outV = dstVerts + vBase;
        for (UINT i = 0; i < mesh.vertexCount; ++i) {
            const Vertex& sv = mesh.cpuVertices[i];
            outV[i].vertex = sv.position;
            XMVECTOR normal =
                XMVector3Normalize(XMVectorSet(sv.normal_material.x, sv.normal_material.y, sv.normal_material.z, 0.0f));
            outV[i].packedNormal = EncodeNormalOct(normal);
            PackedVector::XMStoreHalf2(&outV[i].texCoord, XMLoadFloat2(&sv.texCoord));
        }

        uint32_t* outI = dstIdx + mesh.globalIndexBase;
        for (UINT i = 0; i < mesh.indexCount; ++i)
            outI[i] = mesh.cpuIndices[i] + vBase;
    }

    // Static topology; each tile indexes only its own block.
    if (hasOcean) {
        uint32_t* out = dstIdx + totalIndexCount;
        constexpr uint32_t edge = OCEAN_TILE_EDGE_VERTS;
        for (uint32_t t = 0; t < oceanInstanceSlots; ++t) {
            const uint32_t base = oceanVertexBase + t * OCEAN_TILE_VERTS;
            for (uint32_t j = 0; j < OCEAN_TILE_GRID; ++j) {
                for (uint32_t i = 0; i < OCEAN_TILE_GRID; ++i) {
                    const uint32_t v00 = base + j * edge + i;
                    const uint32_t v10 = v00 + 1;
                    const uint32_t v01 = v00 + edge;
                    const uint32_t v11 = v01 + 1;
                    // Wound so geometric normals face the sky.
                    *out++ = v00;
                    *out++ = v01;
                    *out++ = v11;
                    *out++ = v00;
                    *out++ = v11;
                    *out++ = v10;
                }
            }
        }
    }

    if (!hasTerrain) {
        vertexGlobalUpload->Unmap(0, nullptr);
        indexGlobalUpload->Unmap(0, nullptr);
        if (sceneVbBytes)
            cmdList->CopyBufferRegion(vertexGlobal.Get(), 0, vertexGlobalUpload.Get(), 0, sceneVbBytes);
        if (sceneIbBytes)
            cmdList->CopyBufferRegion(indexGlobal.Get(), 0, indexGlobalUpload.Get(), 0, sceneIbBytes);
        if (hasOcean)
            cmdList->CopyBufferRegion(indexGlobal.Get(), (uint64_t)oceanIndexBase * sizeof(uint32_t),
                                      indexGlobalUpload.Get(), sceneIbBytes,
                                      (uint64_t)oceanIndexElems * sizeof(uint32_t));
        const D3D12_RESOURCE_BARRIER toRead[2] = {
            CD3DX12_RESOURCE_BARRIER::Transition(vertexGlobal.Get(), D3D12_RESOURCE_STATE_COPY_DEST, vbReadState),
            CD3DX12_RESOURCE_BARRIER::Transition(indexGlobal.Get(), D3D12_RESOURCE_STATE_COPY_DEST, vbReadState),
        };
        cmdList->ResourceBarrier(2, toRead);
    }
}

void Scene::CreateInstancePropertiesBuffer(ID3D12Device* device) {
    const uint32_t count = instancePropsCount();
    const uint32_t size = ROUND_UP(count * sizeof(InstanceProperties), D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT);

    if (instanceProperties) {
        const D3D12_RESOURCE_DESC d = instanceProperties->GetDesc();
        if (d.Width >= size)
            return;
    }
    instanceProperties = nv_helpers_dx12::CreateBuffer(
        device, size, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
}

void Scene::PrepareInstanceProperties() {
    const size_t count = instances.size();

    cpuInstanceProps.resize(count);
    instanceDirty.resize(count, 1);
    instanceInitialized.resize(count, 0);

    dirtyInstanceList.clear();
    for (size_t i = 0; i < count; ++i) {
        if (!instanceDirty[i])
            continue;
        dirtyInstanceList.push_back(static_cast<uint32_t>(i));
    }

    // Relative to the floating origin.
    const XMVECTOR shift = XMVectorSet(sceneOriginWorld.x, sceneOriginWorld.y, sceneOriginWorld.z, 0.0f);

    const XMVECTOR prevShift =
        XMVectorSet(prevSceneOriginWorld.x, prevSceneOriginWorld.y, prevSceneOriginWorld.z, 0.0f);
    const XMVECTOR originDelta = XMVectorSubtract(shift, prevShift);
    const bool originShifted = XMVector3LengthSq(originDelta).m128_f32[0] > 0.0f;

    for (uint32_t idx : dirtyInstanceList) {
        auto& dst = cpuInstanceProps[idx];
        auto& si = instances[idx];
        XMMATRIX M = si.worldTransform;

        M.r[3] = XMVectorSubtract(M.r[3], shift);
        XMVECTOR det;

        bool isNew = !instanceInitialized[idx];

        if (!isNew) {
            dst.prevObjectToWorld = dst.objectToWorld;
            dst.prevObjectToWorldInverse = dst.objectToWorldInverse;
            dst.prevObjectToWorldNormal = dst.objectToWorldNormal;

            if (originShifted) {
                dst.prevObjectToWorld.r[3] = XMVectorSubtract(dst.prevObjectToWorld.r[3], originDelta);
                dst.prevObjectToWorldInverse = XMMatrixInverse(&det, dst.prevObjectToWorld);
            }
        }

        dst.objectToWorld = M;
        dst.objectToWorldInverse = XMMatrixInverse(&det, M);

        XMMATRIX upper3x3 = M;
        upper3x3.r[0].m128_f32[3] = upper3x3.r[1].m128_f32[3] = upper3x3.r[2].m128_f32[3] = 0.f;
        upper3x3.r[3] = {0, 0, 0, 1.f};
        dst.objectToWorldNormal = XMMatrixTranspose(XMMatrixInverse(&det, upper3x3));

        if (isNew) {
            dst.prevObjectToWorld = dst.objectToWorld;
            dst.prevObjectToWorldInverse = dst.objectToWorldInverse;
            dst.prevObjectToWorldNormal = dst.objectToWorldNormal;
            instanceInitialized[idx] = 1;
        }

        const MeshGPU& mesh = meshes[si.meshIndex];
        dst.opaqueTriCount = mesh.opaqueTriCount;
        dst.indexBase = mesh.globalIndexBase;
        dst.vertexBase = mesh.globalVertexBase;
        dst.materialBase = mesh.materialIDBase;

        const bool lit = lightClassEnabled[MeshLightClass(si.meshIndex)];
        dst.triToLightBase = (lit && si.meshIndex < meshLightBase.size()) ? meshLightBase[si.meshIndex] : 0xFFFFFFFFu;
        dst.lightSlot = (lit && idx < instanceLightSlot.size()) ? instanceLightSlot[idx] : 0xFFFFFFFFu;
        // A paged mesh's streamed triangles count on from its own, which alone have light records: ranged lookup.
        if (mesh.paged && dst.triToLightBase != 0xFFFFFFFFu) {
            dst.triToLightBase |= 0x40000000u;
            dst._pad[1] = mesh.indexCount / 3;
        }

        if (idx < tlasInstances.size())
            tlasInstances[idx].transform = M;
    }

    prevSceneOriginWorld = sceneOriginWorld;
}

void Scene::UploadInstanceProperties() {
    if (!dirtyInstanceList.empty()) {
        uint8_t* gpuDst = nullptr;
        CD3DX12_RANGE readRange(0, 0);
        ThrowIfFailed(instanceProperties->Map(0, &readRange, reinterpret_cast<void**>(&gpuDst)));
        for (uint32_t idx : dirtyInstanceList) {
            const InstancePropertiesCpu& src = cpuInstanceProps[idx];
            InstanceProperties gp;
            gp.objectToWorld = MakeFloat3x4(src.objectToWorld);
            gp.objectToWorldInverse = MakeFloat3x4(src.objectToWorldInverse);
            gp.objectToWorldNormal = MakeFloat3x4(src.objectToWorldNormal);
            gp.indexBase = src.indexBase;
            gp.vertexBase = src.vertexBase;
            gp.materialBase = src.materialBase;
            gp.triToLightBase = src.triToLightBase;
            gp.opaqueTriCount = src.opaqueTriCount;
            gp._pad[0] = src._pad[0];
            gp._pad[1] = src._pad[1];
            gp.lightSlot = src.lightSlot;
            gp.prevObjectToWorld = MakeFloat3x4(src.prevObjectToWorld);
            memcpy(gpuDst + idx * sizeof(InstanceProperties), &gp, sizeof(gp));
        }
        instanceProperties->Unmap(0, nullptr);
    }

    for (uint32_t idx : dirtyInstanceList) {
        auto& p = cpuInstanceProps[idx];
        bool alreadySettled = (memcmp(&p.prevObjectToWorld, &p.objectToWorld, sizeof(XMMATRIX)) == 0);
        p.prevObjectToWorld = p.objectToWorld;
        p.prevObjectToWorldInverse = p.objectToWorldInverse;
        p.prevObjectToWorldNormal = p.objectToWorldNormal;

        instanceDirty[idx] = alreadySettled ? 0 : 1;
    }
}

void Scene::RebuildTLASInstanceList() {
    tlasInstances.clear();
    tlasInstances.reserve(instances.size());

    const XMVECTOR shift = XMVectorSet(sceneOriginWorld.x, sceneOriginWorld.y, sceneOriginWorld.z, 0.0f);
    for (size_t i = 0; i < instances.size(); ++i) {
        const auto& si = instances[i];
        const auto& mesh = meshes[si.meshIndex];

        UINT hitGroupContrib = static_cast<UINT>(i) * 2;

        auto flags = (mesh.alphaTriCount == 0 && !mesh.lodAlpha) ? D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_OPAQUE
                                                                 : D3D12_RAYTRACING_INSTANCE_FLAG_NONE;

        XMMATRIX shifted = si.worldTransform;
        shifted.r[3] = XMVectorSubtract(shifted.r[3], shift);

        tlasInstances.push_back({mesh.blas, shifted, hitGroupContrib, flags});
    }
}

void Scene::CollectEmissiveTriangles() {
    emissiveTriangles.clear();
    triToLightId.clear();
    meshLightBase.assign(meshes.size(), 0xFFFFFFFFu);
    meshRecordBase.assign(meshes.size(), 0xFFFFFFFFu);

    terrainTriLightBase = 0xFFFFFFFFu;

    for (size_t m = 0; m < meshes.size(); ++m) {
        const auto& mesh = meshes[m];
        const UINT tris = mesh.indexCount / 3;
        const auto coarse = [&](UINT t) { return t - mesh.lod.coarseTriBase < mesh.lod.coarseTriCount; };
        bool anyEmissive = false;
        for (UINT t = 0; t < tris && !anyEmissive; ++t) {
            if (coarse(t))
                continue;
            const XMFLOAT3& Ke = materials.Ke[mesh.cpuMaterialIDs[t]];
            anyEmissive = Ke.x + Ke.y + Ke.z > 0.0f;
        }
        if (!anyEmissive)
            continue;

        meshLightBase[m] = (uint32_t)triToLightId.size();
        meshRecordBase[m] = (uint32_t)emissiveTriangles.size();
        triToLightId.resize(triToLightId.size() + tris, 0xFFFFFFFFu);
        for (UINT t = 0; t < tris; ++t) {
            const XMFLOAT3& Ke = materials.Ke[mesh.cpuMaterialIDs[t]];
            if (Ke.x + Ke.y + Ke.z <= 0.0f || coarse(t))
                continue;

            LightTriangle lt{};
            lt.x = mesh.cpuVertices[mesh.cpuIndices[3 * t + 0]].position;
            lt.y = mesh.cpuVertices[mesh.cpuIndices[3 * t + 1]].position;
            lt.z = mesh.cpuVertices[mesh.cpuIndices[3 * t + 2]].position;
            lt.meshID = (UINT)m;

            lt.weight = ComputeTriangleWeight(lt.x, lt.y, lt.z, Ke, XMMatrixIdentity());
            lt.emission = Ke;

            triToLightId[(size_t)meshLightBase[m] + t] = (uint32_t)emissiveTriangles.size();
            emissiveTriangles.push_back(lt);
        }
    }

    lightInstances.clear();
    instanceLightSlot.assign(instances.size(), 0xFFFFFFFFu);
    for (size_t i = 0; i < instances.size(); ++i) {
        const UINT m = instances[i].meshIndex;
        if (m >= meshes.size() || meshLightBase[m] == 0xFFFFFFFFu)
            continue;
        instanceLightSlot[i] = (uint32_t)lightInstances.size();
        lightInstances.push_back({(UINT)i, m});
    }

    if (emissiveTriangles.empty())
        emissiveTriangles.push_back(LightTriangle{});

    LOG(L"Emissive triangles: " << emissiveTriangles.size() << L" unique records, " << lightInstances.size()
                                << L" light instances of " << instances.size());
}

float Scene::ComputeTriangleWeight(const XMFLOAT3& v0, const XMFLOAT3& v1, const XMFLOAT3& v2, const XMFLOAT3& emissive,
                                   const XMMATRIX& M) {
    XMVECTOR p0 = XMVector3TransformCoord(XMLoadFloat3(&v0), M);
    XMVECTOR p1 = XMVector3TransformCoord(XMLoadFloat3(&v1), M);
    XMVECTOR p2 = XMVector3TransformCoord(XMLoadFloat3(&v2), M);
    float area = 0.5f * XMVectorGetX(XMVector3Length(XMVector3Cross(p1 - p0, p2 - p0)));
    return std::max(area, 1e-10f) * Luminance(emissive);
}

void Scene::UploadMaterials(ID3D12Device* device) {
    if (terrainMatIDElems && terrainMatIDBase == 0) {
        terrainMatIDBase = (UINT)materialIDs.size();
        materialIDs.resize((size_t)terrainMatIDBase + terrainMatIDElems, terrainMatIndex);
    }

    if (voxelMatIDElems && !voxelMatIDReserved) {
        voxelMatIDBase = (UINT)materialIDs.size();
        materialIDs.resize((size_t)voxelMatIDBase + voxelMatIDElems, 0u);
        voxelMatIDReserved = true;
    }

    // One tile's IDs serve all tiles: a single material.
    if (oceanMatIDElems && !oceanMatIDReserved) {
        oceanMatIDBase = (UINT)materialIDs.size();
        materialIDs.resize((size_t)oceanMatIDBase + oceanMatIDElems, oceanMatIndex);
        oceanMatIDReserved = true;
    }

    materials.BuildGpuPacked(materialPacked);

    {
        const UINT sz = (UINT)materialPacked.size() * sizeof(uint32_t);
        auto d = CD3DX12_RESOURCE_DESC::Buffer(sz);
        ThrowIfFailed(device->CreateCommittedResource(&nv_helpers_dx12::kUploadHeapProps, D3D12_HEAP_FLAG_NONE, &d,
                                                      D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                      IID_PPV_ARGS(&materialBuffer)));
        UINT8* p;
        materialBuffer->Map(0, nullptr, (void**)&p);
        memcpy(p, materialPacked.data(), sz);
        materialBuffer->Unmap(0, nullptr);
    }

    {
        const size_t idCount = materialIDs.size();
        const uint64_t sz = std::max<uint64_t>((uint64_t)idCount * sizeof(UINT), 256u);
        auto d = CD3DX12_RESOURCE_DESC::Buffer(sz);
        ThrowIfFailed(device->CreateCommittedResource(&nv_helpers_dx12::kUploadHeapProps, D3D12_HEAP_FLAG_NONE, &d,
                                                      D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                      IID_PPV_ARGS(&materialIndexBuffer)));
        UINT8* p;
        materialIndexBuffer->Map(0, nullptr, (void**)&p);
        if (idCount)
            memcpy(p, materialIDs.data(), idCount * sizeof(UINT));
        materialIndexBuffer->Unmap(0, nullptr);
    }
}

void Scene::UpdateMaterialBuffer() {
    if (!materialBuffer || materials.empty())
        return;

    materials.BuildGpuPacked(materialPacked);

    const UINT sz = (UINT)materialPacked.size() * sizeof(uint32_t);
    UINT8* p = nullptr;
    CD3DX12_RANGE readRange(0, 0);
    if (SUCCEEDED(materialBuffer->Map(0, &readRange, (void**)&p))) {
        memcpy(p, materialPacked.data(), sz);
        materialBuffer->Unmap(0, nullptr);
    }
}

void Scene::CreateTriToLightIdBuffer(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList) {
    if (triToLightId.empty())
        return;
    const uint64_t bytes = (uint64_t)triToLightId.size() * sizeof(uint32_t);
    ComPtr<ID3D12Resource> upload = nv_helpers_dx12::CreateBuffer(
        device, bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ, nv_helpers_dx12::kUploadHeapProps);
    {
        void* p = nullptr;
        CD3DX12_RANGE r(0, 0);
        ThrowIfFailed(upload->Map(0, &r, &p));
        memcpy(p, triToLightId.data(), bytes);
        upload->Unmap(0, nullptr);
    }
    triToLightIdBuffer = nv_helpers_dx12::CreateBuffer(
        device, bytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, nv_helpers_dx12::kDefaultHeapProps);
    cmdList->CopyBufferRegion(triToLightIdBuffer.Get(), 0, upload.Get(), 0, bytes);
    auto br = CD3DX12_RESOURCE_BARRIER::Transition(triToLightIdBuffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                                   D3D12_RESOURCE_STATE_GENERIC_READ);
    cmdList->ResourceBarrier(1, &br);
    // Kept until the GPU copy completes.
    pendingLightUploads.push_back(std::move(upload));
}

void Scene::CreateEmissiveTrianglesBuffer(ID3D12Device* device, ID3D12GraphicsCommandList* cmdList,
                                          ID3D12CommandQueue* queue, ID3D12CommandAllocator* alloc) {
    size_t bufferSize = emissiveTriangles.size() * sizeof(LightTriangle);
    for (auto& t : emissiveTriangles)
        t.triCount = (UINT)emissiveTriangles.size();
    ComPtr<ID3D12Resource> upload =
        nv_helpers_dx12::CreateBuffer(device, bufferSize, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ,
                                      nv_helpers_dx12::kUploadHeapProps);
    {
        void* p = nullptr;
        CD3DX12_RANGE r(0, 0);
        ThrowIfFailed(upload->Map(0, &r, &p));
        memcpy(p, emissiveTriangles.data(), bufferSize);
        upload->Unmap(0, nullptr);
    }
    emissiveTrianglesBuffer =
        nv_helpers_dx12::CreateBuffer(device, bufferSize, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST,
                                      nv_helpers_dx12::kDefaultHeapProps);
    cmdList->CopyBufferRegion(emissiveTrianglesBuffer.Get(), 0, upload.Get(), 0, bufferSize);
    auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(emissiveTrianglesBuffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                                        D3D12_RESOURCE_STATE_GENERIC_READ);
    cmdList->ResourceBarrier(1, &barrier);
    // Kept through the async submission.
    pendingLightUploads.push_back(std::move(upload));
}
