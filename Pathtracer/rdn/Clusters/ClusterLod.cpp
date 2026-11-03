#include "ClusterLod.h"
#include "../DXRHelper.h"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <numeric>
#include <thread>
#include <unordered_map>
#include <meshoptimizer.h>
#define CLUSTERLOD_IMPLEMENTATION
#include "ClodBuild.h"

namespace clusters {

namespace {
struct RegionCluster {
    uint32_t firstTri, triCount, vertexCount, group, refined;
};
struct RegionLod {
    uint32_t region = NO_REGION;
    std::vector<uint32_t> tris;    // the region's triangles in the mesh
    std::vector<uint32_t> indices; // the hierarchy's triangles, in mesh vertex indices
    std::vector<float> positions;  // per entry of indices, as its group was emitted
    std::vector<RegionCluster> clusters;
    std::vector<LodGroup> groups;
    std::vector<uint32_t> codes;   // per triangle of indices, with the source's codes
};

// One region's hierarchy, over a compact copy of its vertices.
void build_region(const LodSource& src, const std::vector<uint8_t>& locked, const std::vector<uint32_t>& remap,
                  RegionLod& r) {
    std::vector<uint32_t> verts;
    verts.reserve(r.tris.size() * 3);
    for (uint32_t t : r.tris)
        for (int k = 0; k < 3; ++k) verts.push_back(src.indices[3ull * t + k]);
    std::sort(verts.begin(), verts.end());
    verts.erase(std::unique(verts.begin(), verts.end()), verts.end());
    const auto local = [&](uint32_t v) { return (uint32_t)(std::lower_bound(verts.begin(), verts.end(), v) - verts.begin()); };

    const size_t n = verts.size();
    std::vector<float> pos(n * 3), attr(n * 5); // normal, uv
    std::vector<uint8_t> lock(n);
    for (size_t i = 0; i < n; ++i) {
        const uint8_t* v = src.vertices + (size_t)verts[i] * src.stride;
        std::memcpy(&pos[i * 3], v, 12);
        std::memcpy(&attr[i * 5], v + src.normalOffset, 12);
        std::memcpy(&attr[i * 5 + 3], v + src.uvOffset, 8);
        lock[i] = (locked[remap[verts[i]]] ? (uint8_t)meshopt_SimplifyVertex_Lock : (uint8_t)0) |
                  (src.vertexFlags ? src.vertexFlags[verts[i]] : (uint8_t)0);
    }
    std::vector<uint32_t> idx(r.tris.size() * 3);
    for (size_t i = 0; i < r.tris.size(); ++i)
        for (int k = 0; k < 3; ++k) idx[i * 3 + k] = local(src.indices[3ull * r.tris[i] + k]);

    std::vector<uint32_t> triCodes;
    if (src.codes) {
        triCodes.resize(r.tris.size());
        for (size_t i = 0; i < r.tris.size(); ++i) triCodes[i] = src.codes[r.tris[i]];
    }

    // The meshoptimizer demo's setup: normals weigh in, UV seams are protected from permissive collapses.
    const float* weights = src.attributeWeights;
    clodMesh mesh = {};
    mesh.indices = idx.data();
    mesh.index_count = idx.size();
    mesh.vertex_count = n;
    mesh.vertex_positions = pos.data();
    mesh.vertex_positions_stride = sizeof(float) * 3;
    mesh.vertex_attributes = attr.data();
    mesh.vertex_attributes_stride = sizeof(float) * 5;
    mesh.vertex_lock = lock.data();
    mesh.attribute_weights = weights;
    mesh.attribute_count = 3;
    mesh.attribute_protect_mask = src.protectUvSeams ? (1u << 3) | (1u << 4) : 0u;
    clodConfig config = clodDefaultConfigRT(MAX_TRIS);
    // Groups that fail to simplify try again with other neighbours, so a hierarchy ends at a single root more often; but
    // never sloppily (vertex clustering, blind to UVs and the shape's detail), which Nanite does not either.
    config.simplify_carry = 3;
    config.simplify_fallback_sloppy = false;
    if (src.foliage && src.foliage(r.region)) {
        // Dilation moves positions in place, so each group's are copied as it is emitted.
        config.simplify_dilate_borders = true;
        config.simplify_preserve_folds = true;
    } else if (src.aggregate && src.aggregate(r.region)) {
        // Growth moves positions in place, too.
        config.simplify_prune_aggregate = true;
    }

    if (src.codes) {
        mesh.triangle_codes = triCodes.data();
        mesh.coarsen = [](void* ctx, unsigned int code, int level) {
            const LodSource& s = *static_cast<const LodSource*>(ctx);
            return s.coarsen ? s.coarsen(code, (uint32_t)level) : code;
        };
        mesh.coarsen_context = const_cast<LodSource*>(&src);
    }

    clodBuild(config, mesh, [&](clodGroup group, const clodCluster* cl, size_t count) -> int {
        const uint32_t g = (uint32_t)r.groups.size();
        const clodBounds& b = group.simplified;
        r.groups.push_back(LodGroup{ { b.center[0], b.center[1], b.center[2] }, b.radius, b.error });
        for (size_t i = 0; i < count; ++i) {
            r.clusters.push_back(RegionCluster{ (uint32_t)(r.indices.size() / 3), (uint32_t)(cl[i].index_count / 3),
                                                (uint32_t)cl[i].vertex_count, g,
                                                cl[i].refined < 0 ? LOD_NONE : (uint32_t)cl[i].refined });
            if (cl[i].codes) r.codes.insert(r.codes.end(), cl[i].codes, cl[i].codes + cl[i].index_count / 3);
            for (size_t k = 0; k < cl[i].index_count; ++k) {
                r.indices.push_back(verts[cl[i].indices[k]]);
                r.positions.insert(r.positions.end(), &pos[cl[i].indices[k] * 3ull], &pos[cl[i].indices[k] * 3ull] + 3);
            }
        }
        return (int)g;
    });
}
} // namespace

bool lod_enabled() {
    static const bool off = [] {
        char v[4] = {};
        return GetEnvironmentVariableA("RDN_LOD", v, sizeof(v)) > 0 && v[0] == '0';
    }();
    return nv_helpers_dx12::g_clusteredGeometry && !off;
}

void build_lod(const LodSource& src, LodBuild& out) {
    out = LodBuild{};
    // Positions shared across regions (or with triangles outside) are locked: the regions' cuts are independent.
    std::vector<uint32_t> remap(src.vertexCount);
    meshopt_generatePositionRemap(remap.data(), reinterpret_cast<const float*>(src.vertices), src.vertexCount, src.stride);
    constexpr uint32_t UNSEEN = NO_REGION - 1;
    std::vector<uint32_t> owner(src.vertexCount, UNSEEN);
    std::vector<uint8_t> locked(src.vertexCount, 0);
    std::unordered_map<uint32_t, uint32_t> regionIndex;
    std::vector<RegionLod> regions;
    for (uint32_t t = 0; t < src.triCount; ++t) {
        const uint32_t reg = src.regions[t];
        if (reg == NO_REGION && !src.lockOutside) continue;
        for (int k = 0; k < 3; ++k) {
            const uint32_t p = remap[src.indices[3ull * t + k]];
            if (owner[p] == UNSEEN) owner[p] = reg;
            else if (owner[p] != reg) locked[p] = 1;
        }
        if (reg == NO_REGION) continue;
        const auto [it, fresh] = regionIndex.emplace(reg, (uint32_t)regions.size());
        if (fresh) regions.emplace_back().region = reg;
        regions[it->second].tris.push_back(t);
    }
    if (regions.empty()) return;

    // Largest first, so the long ones start early.
    std::vector<uint32_t> order(regions.size());
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return regions[a].tris.size() > regions[b].tris.size(); });
    std::atomic<uint32_t> next{ 0 };
    auto work = [&] {
        for (uint32_t i; (i = next.fetch_add(1)) < order.size();) build_region(src, locked, remap, regions[order[i]]);
    };
    const uint32_t threads = std::min<uint32_t>((uint32_t)regions.size(), std::max(1u, std::thread::hardware_concurrency()));
    std::vector<std::thread> pool;
    for (uint32_t i = 1; i < threads; ++i) pool.emplace_back(work);
    work();
    for (auto& th : pool) th.join();

    // Merged in region order, the original clusters of every region first.
    uint32_t groupBase = 0;
    std::vector<std::pair<const RegionLod*, uint32_t>> later; // simplified clusters, with their region's group base
    auto emit = [&](const RegionLod& r, const RegionCluster& c, uint32_t base) {
        out.spans.push_back(Span{ (uint32_t)(out.indices.size() / 3), (uint16_t)c.triCount, (uint16_t)c.vertexCount });
        out.clusters.push_back(LodCluster{ base + c.group, c.refined == LOD_NONE ? LOD_NONE : base + c.refined });
        out.indices.insert(out.indices.end(), r.indices.begin() + 3ull * c.firstTri, r.indices.begin() + 3ull * (c.firstTri + c.triCount));
        out.positions.insert(out.positions.end(), r.positions.begin() + 9ull * c.firstTri,
                             r.positions.begin() + 9ull * (c.firstTri + c.triCount));
        out.regions.insert(out.regions.end(), c.triCount, r.region);
        if (!r.codes.empty())
            out.codes.insert(out.codes.end(), r.codes.begin() + c.firstTri, r.codes.begin() + c.firstTri + c.triCount);
    };
    for (const RegionLod& r : regions) {
        for (const RegionCluster& c : r.clusters)
            if (c.refined == LOD_NONE) emit(r, c, groupBase);
        later.emplace_back(&r, groupBase);
        out.groups.insert(out.groups.end(), r.groups.begin(), r.groups.end());
        groupBase += (uint32_t)r.groups.size();
    }
    out.originalClusters = (uint32_t)out.clusters.size();
    for (const auto& [r, base] : later)
        for (const RegionCluster& c : r->clusters)
            if (c.refined != LOD_NONE) emit(*r, c, base);
}

void build_lod_stage(const LodStageMesh& m, const std::vector<LodSurvivor>* in, const LodStageEmit& emit,
                     std::vector<LodSurvivor>* out) {
    static const float weights[3] = { 0.5f, 0.5f, 0.5f };
    clodMesh mesh = {};
    mesh.indices = m.indices;
    mesh.index_count = 3ull * m.triCount;
    mesh.vertex_count = m.vertexCount;
    mesh.vertex_positions = m.positions;
    mesh.vertex_positions_stride = sizeof(float) * 3;
    mesh.vertex_attributes = m.attributes;
    mesh.vertex_attributes_stride = sizeof(float) * 5;
    mesh.vertex_lock = m.flags;
    mesh.attribute_weights = weights;
    mesh.attribute_count = 3;
    mesh.triangle_codes = m.codes;
    if (m.coarsen) {
        mesh.coarsen = [](void* ctx, unsigned int code, int level) {
            return (*static_cast<const std::function<uint32_t(uint32_t, uint32_t)>*>(ctx))(code, (uint32_t)level);
        };
        mesh.coarsen_context = const_cast<std::function<uint32_t(uint32_t, uint32_t)>*>(&m.coarsen);
    }
    clodConfig config = clodDefaultConfigRT(MAX_TRIS);
    config.simplify_carry = 3;
    config.simplify_fallback_sloppy = false;
    std::vector<clodSurvivor> survivorsIn, survivorsOut;
    if (in) {
        survivorsIn.resize(in->size());
        for (size_t i = 0; i < in->size(); ++i) {
            const LodSurvivor& v = (*in)[i];
            clodSurvivor& c = survivorsIn[i];
            c.indices = v.indices;
            c.codes = v.codes;
            c.refined = v.refined == LOD_NONE ? -1 : (int)v.refined;
            c.level = (int)v.level;
            c.stuck = 0;
            c.bounds = clodBounds{ { v.bounds.center[0], v.bounds.center[1], v.bounds.center[2] }, v.bounds.radius, v.bounds.error };
        }
    }
    std::vector<LodStageCluster> clusters;
    clodBuildStage(config, mesh, survivorsIn.data(), survivorsIn.size(), (void*)&emit,
                   [](void* ctx, clodGroup group, const clodCluster* cl, size_t count) -> int {
                       thread_local std::vector<LodStageCluster> list;
                       list.resize(count);
                       for (size_t i = 0; i < count; ++i)
                           list[i] = LodStageCluster{ cl[i].indices, cl[i].codes, (uint32_t)(cl[i].index_count / 3),
                                                      (uint32_t)cl[i].vertex_count,
                                                      cl[i].refined < 0 ? LOD_NONE : (uint32_t)cl[i].refined,
                                                      (uint32_t)cl[i].level };
                       const clodBounds& b = group.simplified;
                       const LodGroup g{ { b.center[0], b.center[1], b.center[2] }, b.radius, b.error };
                       return (int)(*static_cast<const LodStageEmit*>(ctx))(g, list.data(), (uint32_t)count);
                   },
                   out ? &survivorsOut : nullptr);
    if (out)
        for (clodSurvivor& c : survivorsOut) {
            LodSurvivor v;
            v.indices = std::move(c.indices);
            v.codes = std::move(c.codes);
            v.refined = c.refined < 0 ? LOD_NONE : (uint32_t)c.refined;
            v.level = (uint32_t)c.level;
            v.bounds = LodGroup{ { c.bounds.center[0], c.bounds.center[1], c.bounds.center[2] }, c.bounds.radius, c.bounds.error };
            out->push_back(std::move(v));
        }
}

void lod_levels(const Lod& lod, std::vector<uint8_t>& levels) {
    // A cluster joins a group output after the one it was simplified from, so groups in order see final levels.
    const uint32_t n = (uint32_t)lod.clusters.size(), groups = (uint32_t)lod.groups.size();
    std::vector<uint32_t> first(groups + 1, 0), members(n);
    for (const LodCluster& c : lod.clusters) ++first[c.group + 1];
    for (uint32_t g = 0; g < groups; ++g) first[g + 1] += first[g];
    std::vector<uint32_t> at(first.begin(), first.end() - 1);
    for (uint32_t c = 0; c < n; ++c) members[at[lod.clusters[c].group]++] = c;
    std::vector<uint8_t> groupLevel(groups, 0);
    levels.assign(n, 0);
    for (uint32_t g = 0; g < groups; ++g)
        for (uint32_t i = first[g]; i < first[g + 1]; ++i) {
            const uint32_t c = members[i], r = lod.clusters[c].refined;
            levels[c] = r == LOD_NONE ? 0 : (uint8_t)std::min(255, groupLevel[r] + 1);
            groupLevel[g] = std::max(groupLevel[g], levels[c]);
        }
}

void select_cut(const Lod& lod, const LodView& v, std::vector<uint32_t>& cut, std::vector<LodRequest>* requests) {
    cut.clear();
    thread_local std::vector<uint8_t> fine;
    fine.resize(lod.groups.size());
    for (uint32_t g = 0; g < (uint32_t)lod.groups.size(); ++g) {
        const LodGroup& b = lod.groups[g];
        if (b.error == FLT_MAX) { fine[g] = 0; continue; }
        const float dx = b.center[0] - v.camera[0], dy = b.center[1] - v.camera[1], dz = b.center[2] - v.camera[2];
        const float d = std::max(std::sqrt(dx * dx + dy * dy + dz * dz) - b.radius, v.znear);
        fine[g] = b.error * v.pixelsPerUnit <= v.threshold * d;
        if (fine[g]) continue;
        const float over = b.error * v.pixelsPerUnit / (v.threshold * d);
        if (requests) {
            float& w = const_cast<Lod&>(lod).want[g];
            w = std::max(w, over);
        }
        if (lod.is_resident(g)) continue;
        // Wanted but absent: the coarser level stands in, so the page is requested once that level is resident.
        fine[g] = 1;
        if (!requests) continue;
        bool ready = true;
        for (uint32_t i = lod.depFirst[g]; i < lod.depFirst[g + 1] && ready; ++i) ready = lod.resident[lod.deps[i]] != 0;
        if (ready) requests->push_back(LodRequest{ lod.pager, g, over });
    }
    for (uint32_t i = 0; i < (uint32_t)lod.clusters.size(); ++i) {
        const LodCluster& c = lod.clusters[i];
        if (!fine[c.group] && (c.refined == LOD_NONE || fine[c.refined])) cut.push_back(i);
    }
}

#if RDN_CLUSTER_API
void LodInstances::init(ClusterGeometry* geometry, ID3D12Device* device, const Budget& budget) {
    m_geometry = geometry;
    m_device = device;
    m_budget = budget;
}

void LodInstances::reset(const std::vector<Instance>& instances, uint64_t frame) {
    if (m_pool) m_retired.push_back({ std::move(m_pool), frame });
    for (auto& s : m_scratch) m_retired.push_back({ std::move(s), frame });
    m_scratch.clear();
    m_states.assign(instances.size(), State{});
    m_active = false;
    m_next = 0;
    m_stats = {};
    if (!m_geometry || !m_geometry->enabled()) return;
    m_ring = std::min(MAX_RING, m_geometry->frame_slots() + 1);
    uint64_t total = 0, maxScratch = 0;
    for (size_t i = 0; i < instances.size(); ++i) {
        Lod* lod = instances[i].lod;
        if (!lod || lod->empty() || lod->clas.size() != lod->count()) continue;
        State& s = m_states[i];
        s.lod = lod;
        s.omm = instances[i].omm;
        m_geometry->blas_prebuild(lod->count(), s.omm, s.blasSize, s.scratch);
        s.ringOff = total;
        total += s.blasSize * m_ring;
        maxScratch = std::max(maxScratch, s.scratch);
        s.cut.resize(lod->base - lod->fixed);
        std::iota(s.cut.begin(), s.cut.end(), 0u);
        ++m_stats.instances;
        m_stats.fullClusters += lod->base;
    }
    if (!total) return;
    using nv_helpers_dx12::CreateBuffer;
    m_pool = CreateBuffer(m_device, total, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                          D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE, nv_helpers_dx12::kDefaultHeapProps);
    m_pool->SetName(L"LodClusterBlasRings");
    m_scratchCapacity = std::max<uint64_t>(16ull << 20, 2 * maxScratch);
    for (uint32_t i = 0; i < m_geometry->frame_slots(); ++i) {
        m_scratch.push_back(CreateBuffer(m_device, m_scratchCapacity, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                         D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nv_helpers_dx12::kDefaultHeapProps));
        m_scratch.back()->SetName(L"LodClusterBlasScratch");
    }
    m_stats.ringBytes = total;
    m_stats.cutClusters = m_stats.fullClusters;
    m_active = true;
    std::printf("[lod] %u instances with a cluster hierarchy, %llu clusters in their meshes' own BLAS; rings of %u, %.1f MB\n",
                m_stats.instances, (unsigned long long)m_stats.fullClusters, m_ring, (double)total / 1048576.0);
}

void LodInstances::update(const LodView* views, uint64_t frame, std::vector<LodRequest>* requests) {
    m_retired.erase(std::remove_if(m_retired.begin(), m_retired.end(),
                                   [&](const Retired& r) { return frame >= r.frame + m_ring; }),
                    m_retired.end());
    m_stats.builds = m_stats.deferred = 0;
    if (!m_active) return;
    const uint32_t slot = m_geometry->slot();
    const D3D12_GPU_VIRTUAL_ADDRESS scratchVa = m_scratch[slot % m_scratch.size()]->GetGPUVirtualAddress();
    const D3D12_GPU_VIRTUAL_ADDRESS poolVa = m_pool->GetGPUVirtualAddress();
    uint64_t scratchUsed = 0, cutClusters = 0;
    uint32_t queued = 0, firstDeferred = LOD_NONE;
    const uint32_t n = (uint32_t)m_states.size();
    for (uint32_t k = 0; k < n; ++k) {
        const uint32_t i = (m_next + k) % n;
        State& s = m_states[i];
        if (!s.lod) continue;
        Lod& lod = *s.lod;
        select_cut(lod, views[i], m_cut, lod.paged ? requests : nullptr);
        // The traced cut's groups stay resident while frames in flight may trace it.
        if (lod.paged)
            for (uint32_t c : s.cut) lod.lastUsed[lod.clusters[c].group] = frame;
        if (m_cut == s.cut) { cutClusters += lod.fixed + s.cut.size(); continue; }
        const uint32_t count = lod.fixed + (uint32_t)m_cut.size();
        const uint32_t entry = s.current == LOD_NONE ? 0u : (s.current + 1) % m_ring;
        const bool fits = m_stats.builds < m_budget.builds && (m_stats.builds == 0 || queued + count <= m_budget.clusters) &&
                          scratchUsed + s.scratch <= m_scratchCapacity && m_geometry->can_queue_blas(count) &&
                          (s.retired[entry] == 0 || frame >= s.retired[entry] + m_ring);
        if (!fits) {
            ++m_stats.deferred;
            if (firstDeferred == LOD_NONE) firstDeferred = i;
            cutClusters += lod.fixed + s.cut.size();
            continue;
        }
        m_addresses.assign(lod.clas.begin(), lod.clas.begin() + lod.fixed);
        for (uint32_t c : m_cut) m_addresses.push_back(lod.clas[lod.fixed + c]);
        m_geometry->queue_blas_from(m_addresses.data(), count, lod.count(), s.omm,
                                    poolVa + s.ringOff + entry * s.blasSize, scratchVa + scratchUsed);
        scratchUsed += s.scratch;
        queued += count;
        ++m_stats.builds;
        if (s.current != LOD_NONE) s.retired[s.current] = frame;
        s.current = entry;
        s.cut.swap(m_cut);
        if (lod.paged)
            for (uint32_t c : s.cut) lod.lastUsed[lod.clusters[c].group] = frame;
        cutClusters += count;
    }
    // Deferred instances go first next frame.
    if (firstDeferred != LOD_NONE) m_next = firstDeferred;
    m_stats.cutClusters = cutClusters;
}

D3D12_GPU_VIRTUAL_ADDRESS LodInstances::blas(uint32_t instance) const {
    if (instance >= m_states.size()) return 0;
    const State& s = m_states[instance];
    if (!s.lod || s.current == LOD_NONE) return 0;
    return m_pool->GetGPUVirtualAddress() + s.ringOff + s.current * s.blasSize;
}
#endif

} // namespace clusters
