#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <chrono>

#include "coordinate_system.h"
#include "cube_sphere.h"
#include "chunk_mesh.h"
#include "heightmap_source.h"
#include "heightmap_procedural.h"
#include "worker_pool.h"
#include "tessellator.h"
#include "restricted_quadtree.h"
#include "blas_cells.h"

using namespace planet;

static int g_checks   = 0;
static int g_failures = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        ++g_checks;                                                            \
        if (!(cond)) {                                                         \
            ++g_failures;                                                      \
            std::printf("  FAIL  %s   (line %d)\n", #cond, __LINE__);          \
        }                                                                      \
    } while (0)

static bool near_eq(double a, double b, double tol) { return std::fabs(a - b) <= tol; }

#define CHECK_NEAR(a, b, tol)                                                  \
    do {                                                                       \
        ++g_checks;                                                            \
        const double a_ = double(a), b_ = double(b);                           \
        if (!near_eq(a_, b_, double(tol))) {                                   \
            ++g_failures;                                                      \
            std::printf("  FAIL  %s ~= %s  (got %.6f want %.6f, line %d)\n",   \
                        #a, #b, a_, b_, __LINE__);                             \
        }                                                                      \
    } while (0)

static bool is_ancestor(uint64_t a_id, uint64_t b_id) {
    const QuadNode a = unpack_node_id(a_id);
    const QuadNode b = unpack_node_id(b_id);
    if (a.face != b.face || a.lod >= b.lod) return false;
    const int d = int(b.lod) - int(a.lod);
    return (b.x >> d) == a.x && (b.y >> d) == a.y;
}

// Decodes the mesh's signed octahedral normal representation.
static Vec3f decode_signed_normal(uint32_t packed) {
    int32_t x = static_cast<int32_t>(packed & 0xFFFFu);
    int32_t y = static_cast<int32_t>((packed >> 16) & 0xFFFFu);
    if (x & 0x8000) x -= 0x10000;
    if (y & 0x8000) y -= 0x10000;

    float ox = static_cast<float>(x) / 32767.0f;
    float oy = static_cast<float>(y) / 32767.0f;
    float oz = 1.0f - std::fabs(ox) - std::fabs(oy);
    if (oz < 0.0f) {
        const float oldX = ox;
        const float oldY = oy;
        ox = (1.0f - std::fabs(oldY)) * (oldX >= 0.0f ? 1.0f : -1.0f);
        oy = (1.0f - std::fabs(oldX)) * (oldY >= 0.0f ? 1.0f : -1.0f);
    }
    return normalize(Vec3f{ ox, oy, oz });
}

static void dump_obj(const char* path, const ChunkVertex* v, uint32_t vcount,
                     const int32_t* idx, uint32_t icount) {
    std::FILE* f = std::fopen(path, "w");
    if (!f) { std::printf("  (could not open %s for writing)\n", path); return; }
    std::fprintf(f, "# planet chunk: %u verts, %u tris\n", vcount, icount / 3);
    for (uint32_t k = 0; k < vcount; ++k)
        std::fprintf(f, "v %.6f %.6f %.6f\n", v[k].px, v[k].py, v[k].pz);
    for (uint32_t k = 0; k + 2 < icount; k += 3)
        std::fprintf(f, "f %u %u %u\n",
                     idx[k] + 1u, idx[k + 1] + 1u, idx[k + 2] + 1u);
    std::fclose(f);
}

static void test_coords() {
    std::printf("[test_coords]\n");

    CHECK_NEAR(length(DVec3{ 0, 3, 4 }), 5.0, 1e-12);
    CHECK_NEAR(normalize(DVec3{ 3, 0, 0 }).x, 1.0, 1e-12);
    CHECK_NEAR(length(normalize(DVec3{ 1, 2, 3 })), 1.0, 1e-12);

    const DVec3 z = normalize(DVec3{ 0, 0, 0 });
    CHECK(z.x == 0.0 && z.y == 0.0 && z.z == 0.0);

    const DVec3 cx = cross(DVec3{ 1, 0, 0 }, DVec3{ 0, 1, 0 });
    CHECK(near_eq(cx.x, 0, 1e-12) && near_eq(cx.y, 0, 1e-12) && near_eq(cx.z, 1, 1e-12));

    const double base = 6.371e6, off = 1234.567;
    const Vec3f rel   = to_camera_relative(DVec3{ base + off, 0, 0 }, DVec3{ base, 0, 0 });
    const float naive = float(base + off) - float(base);
    std::printf("  camera-relative: ours=%.4f  naive(float-first)=%.4f  want=%.4f\n",
                rel.x, naive, off);
    CHECK_NEAR(rel.x, off, 0.01);

    CameraView cam;
    cam.position_world = { 0, 0, 0 };
    cam.forward = { 0, 0, 1 };
    cam.up      = { 0, 1, 0 };
    cam.fov_y = 1.0f; cam.aspect = 1.0f;
    cam.near_plane = 1.0f; cam.far_plane = 1.0e6f;
    const Frustum fr = Frustum::from_camera(cam);
    CHECK( fr.intersects_sphere(Vec3f{ 0, 0, 100 },   1.0f));
    CHECK(!fr.intersects_sphere(Vec3f{ 0, 0, -100 },  1.0f));
    CHECK(!fr.intersects_sphere(Vec3f{ 5000, 0, 100 },1.0f));
    CHECK(!fr.intersects_sphere(Vec3f{ 0, 0, 2.0e6f },1.0f));
    std::printf("\n");
}

static void test_cube_sphere() {
    std::printf("[test_cube_sphere]\n");

    const QuadNode samples[] = {
        { 0, 0, 0, 0 }, { 5, 12, 4095, 4095 }, { 3, 7, 100, 50 }, { 2, 1, 1, 0 },
        { 4, 24, 16777215, 16777215 }, { 1, 20, 1048575, 700000 },
    };
    for (const QuadNode& s : samples) {
        const QuadNode r = unpack_node_id(pack_node_id(s));
        CHECK(r.face == s.face && r.lod == s.lod && r.x == s.x && r.y == s.y);
    }

    const QuadNode root{ 4, 2, 1, 1 };
    for (int q = 0; q < 4; ++q) {
        const QuadNode c = child_node(root, q);
        CHECK(c.lod == root.lod + 1);
        const QuadNode p = parent_node(c);
        CHECK(p.face == root.face && p.lod == root.lod && p.x == root.x && p.y == root.y);
    }

    for (uint8_t f = 0; f < CUBE_FACES; ++f)
        CHECK_NEAR(length(cube_to_sphere_dir(f, 0.0, 0.0)), 1.0, 1e-12);
    CHECK_NEAR(cube_to_sphere_dir(0, 0, 0).x,  1.0, 1e-12);
    CHECK_NEAR(cube_to_sphere_dir(3, 0, 0).y, -1.0, 1e-12);

    PlanetGeometry unit; unit.center = { 0, 0, 0 }; unit.radius = 1.0;
    const NodeGeometry g0 = compute_node_geometry(QuadNode{ 0, 0, 0, 0 }, unit);
    CHECK_NEAR(g0.center_dir.x, 1.0, 1e-9);
    CHECK_NEAR(g0.edge_length,    2.0 / std::sqrt(3.0),          1e-6);
    CHECK_NEAR(g0.angular_radius, std::acos(1.0 / std::sqrt(3.0)), 1e-6);
    CHECK(g0.bounding_radius > 0.0);

    const NodeGeometry g1 = compute_node_geometry(child_node(QuadNode{ 0, 0, 0, 0 }, 0), unit);
    CHECK(g1.edge_length    < g0.edge_length);
    CHECK(g1.angular_radius < g0.angular_radius);
    std::printf("  lod0 face0: edge=%.6f  angRadius=%.6f  boundR=%.6f\n",
                g0.edge_length, g0.angular_radius, g0.bounding_radius);
    std::printf("\n");
}

static void test_cube_neighbors() {
    std::printf("[test_cube_neighbors]\n");

    bool wellformed = true;
    for (uint8_t lod = 0; lod <= 4; ++lod) {
        const uint32_t N = 1u << lod;
        for (uint8_t f = 0; f < CUBE_FACES; ++f)
            for (uint32_t y = 0; y < N; ++y)
                for (uint32_t x = 0; x < N; ++x)
                    for (int e = 0; e < 4; ++e) {
                        const QuadNode m = neighbor_node(
                            QuadNode{ f, lod, (uint32_t)x, (uint32_t)y }, (QuadEdge)e);
                        if (m.face >= CUBE_FACES || m.lod != lod
                                                 || m.x >= N || m.y >= N)
                            wellformed = false;
                    }
    }
    CHECK(wellformed);

    bool reciprocal = true;
    for (uint8_t lod = 0; lod <= 4; ++lod) {
        const uint32_t N = 1u << lod;
        for (uint8_t f = 0; f < CUBE_FACES; ++f)
            for (uint32_t y = 0; y < N; ++y)
                for (uint32_t x = 0; x < N; ++x) {
                    const QuadNode n{ f, lod, (uint32_t)x, (uint32_t)y };
                    for (int e = 0; e < 4; ++e) {
                        const QuadNode m = neighbor_node(n, (QuadEdge)e);
                        int back = 0;
                        for (int e2 = 0; e2 < 4; ++e2) {
                            const QuadNode b = neighbor_node(m, (QuadEdge)e2);
                            back += (b.face == n.face && b.lod == n.lod
                                  && b.x == n.x && b.y == n.y);
                        }
                        if (back != 1) reciprocal = false;
                    }
                }
    }
    CHECK(reciprocal);

    PlanetGeometry unit; unit.center = { 0, 0, 0 }; unit.radius = 1.0;
    bool corners_ok = true;
    int  cross_face = 0;
    for (uint8_t lod = 1; lod <= 3; ++lod) {
        const uint32_t N = 1u << lod;
        for (uint8_t f = 0; f < CUBE_FACES; ++f)
            for (uint32_t y = 0; y < N; ++y)
                for (uint32_t x = 0; x < N; ++x) {
                    const QuadNode     n{ f, lod, (uint32_t)x, (uint32_t)y };
                    const NodeGeometry gn = compute_node_geometry(n, unit);
                    for (int e = 0; e < 4; ++e) {
                        const QuadNode     m  = neighbor_node(n, (QuadEdge)e);
                        const NodeGeometry gm = compute_node_geometry(m, unit);
                        int shared = 0;
                        for (int i = 0; i < 4; ++i)
                            for (int j = 0; j < 4; ++j)
                                if (length(gn.corners[i] - gm.corners[j]) < 1e-9)
                                    ++shared;
                        if (shared != 2)      corners_ok = false;
                        if (m.face != n.face) ++cross_face;
                    }
                }
    }
    CHECK(corners_ok);
    CHECK(cross_face > 0);
    std::printf("  well-formed + reciprocal; %d cross-face links share an edge\n\n",
                cross_face);
}

enum RqCov { RQ_HOLE, RQ_OK, RQ_INVALID };

static RqCov rq_check_cov(const RestrictedQuadtree& qt, const QuadNode& node, uint8_t cap) {
    if (qt.is_leaf(pack_node_id(node))) return RQ_OK;
    if (node.lod >= cap)               return RQ_HOLE;

    int holes = 0, oks = 0, bad = 0;
    for (int q = 0; q < 4; ++q) {
        const RqCov c = rq_check_cov(qt, child_node(node, q), cap);
        holes += (c == RQ_HOLE);
        oks   += (c == RQ_OK);
        bad   += (c == RQ_INVALID);
    }
    if (bad)        return RQ_INVALID;
    if (holes == 4) return RQ_HOLE;
    if (oks == 4)   return RQ_OK;
    return RQ_INVALID;
}

static void test_restricted_quadtree() {
    std::printf("[test_restricted_quadtree]\n");

    QuadtreeParams p;
    p.planet.center = { 0, 0, 0 };
    p.planet.radius = 1000.0;
    p.min_lod       = 1;
    p.max_lod       = 9;
    p.max_leaves    = 4000u;

    CameraView cam;
    cam.position_world = { 0, 0, 1030.0 };

    RestrictedQuadtree qt;
    qt.select(p, cam);
    const std::vector<uint64_t>& L = qt.leaves();
    CHECK(!L.empty());

    bool partition_ok = true;
    for (uint8_t f = 0; f < CUBE_FACES; ++f)
        if (rq_check_cov(qt, QuadNode{ f, 0, 0, 0 }, p.max_lod) != RQ_OK)
            partition_ok = false;
    CHECK(partition_ok);

    bool no_overlap = true;
    for (uint64_t id : L) {
        QuadNode a = unpack_node_id(id);
        while (a.lod > 0) {
            a = parent_node(a);
            if (qt.is_leaf(pack_node_id(a))) no_overlap = false;
        }
    }
    CHECK(no_overlap);

    int per_face[CUBE_FACES] = {};
    int lo = 99, hi = 0;
    for (uint64_t id : L) {
        const QuadNode n = unpack_node_id(id);
        ++per_face[n.face];
        lo = std::min(lo, int(n.lod));
        hi = std::max(hi, int(n.lod));
    }
    bool all_faces = true;
    for (int f = 0; f < CUBE_FACES; ++f) all_faces &= (per_face[f] > 0);
    CHECK(all_faces);
    CHECK(lo >= int(p.min_lod));
    CHECK(hi <= int(p.max_lod));

    bool balanced  = true;
    int  max_delta = 0;
    for (uint64_t id : L) {
        const QuadNode n = unpack_node_id(id);
        for (int e = 0; e < 4; ++e) {
            const int delta = std::abs(int(qt.neighbor_lod(n, (QuadEdge)e)) - int(n.lod));
            max_delta = std::max(max_delta, delta);
            if (delta > 1) balanced = false;
        }
    }
    CHECK(balanced);

    CHECK(hi >= 6);
    int far_min = 99;
    for (uint64_t id : L) {
        const QuadNode n = unpack_node_id(id);
        const NodeGeometry g = compute_node_geometry(n, p.planet);
        const DVec3 d = normalize(g.center_world - p.planet.center);
        if (d.z < -0.5) far_min = std::min(far_min, int(n.lod));
    }
    CHECK(far_min <= hi - 3);

    std::printf("  leaves=%zu  lod range=[%d..%d]  far-side coarsest=%d  max delta=%d\n\n",
                L.size(), lo, hi, far_min, max_delta);
}

static void test_blas_cells() {
    std::printf("[test_blas_cells]\n");

    QuadtreeParams qp;
    qp.planet.center = { 0, 0, 0 };
    qp.planet.radius = 1000.0;
    qp.min_lod       = 1;
    qp.max_lod       = 9;
    qp.max_leaves    = 4000u;

    CellCutParams cp;
    cp.planet              = qp.planet;
    cp.max_leaves_per_cell = 8;
    cp.max_cell_radius_m   = 1e30;

    auto build_gen = [&](DVec3 campos, RestrictedQuadtree& qt, BlasCellSet& cells) {
        CameraView cam;
        cam.position_world = campos;
        qt.select(qp, cam);
        cells.build(qt, cp);
    };

    RestrictedQuadtree liveQt;
    BlasCellSet        liveCells;
    build_gen(DVec3{ 0, 0, 1030.0 }, liveQt, liveCells);
    CHECK(liveCells.cell_count() > 0);

    bool counts_ok = true;
    uint32_t leaves_in_cells = 0;
    for (const BlasCellSet::Cell& c : liveCells.cells()) {
        if (c.leaf_count == 0 || c.leaf_count > cp.max_leaves_per_cell) counts_ok = false;
        leaves_in_cells += c.leaf_count;
    }
    CHECK(counts_ok);
    CHECK(leaves_in_cells == liveQt.leaf_count());

    bool owner_ok = true;
    for (uint64_t leaf : liveQt.leaves())
        if (liveCells.cell_of_leaf(leaf) < 0) owner_ok = false;
    CHECK(owner_ok);

    bool disjoint = true;
    const std::vector<BlasCellSet::Cell>& cl = liveCells.cells();
    for (size_t a = 0; a < cl.size(); ++a)
        for (size_t b = 0; b < cl.size(); ++b)
            if (a != b && is_ancestor(cl[a].node_id, cl[b].node_id)) disjoint = false;
    CHECK(disjoint);

    {
        RestrictedQuadtree q2;
        BlasCellSet        c2;
        build_gen(DVec3{ 0, 0, 1030.0 }, q2, c2);
        std::vector<uint8_t> dirty;
        diff_generations(liveQt, liveCells, q2, c2, dirty);
        int nd = 0;
        for (uint8_t d : dirty) nd += d;
        CHECK(nd == 0);
    }

    {
        RestrictedQuadtree tgtQt;
        BlasCellSet        tgtCells;
        build_gen(DVec3{ 60.0, 0, 1029.0 }, tgtQt, tgtCells);
        std::vector<uint8_t> dirty;
        diff_generations(liveQt, liveCells, tgtQt, tgtCells, dirty);

        int nd = 0;
        for (uint8_t d : dirty) nd += d;
        std::printf("  cells: live=%u  target=%u  dirty=%d  reused=%d\n",
                    liveCells.cell_count(), tgtCells.cell_count(),
                    nd, (int)tgtCells.cell_count() - nd);
        CHECK(nd > 0);
        CHECK(nd < (int)tgtCells.cell_count());

        bool clean_ok = true;
        const std::vector<BlasCellSet::Cell>& tc = tgtCells.cells();
        const std::vector<uint64_t>&          tl = tgtCells.cell_leaves();
        for (size_t i = 0; i < tc.size(); ++i) {
            if (dirty[i]) continue;
            if (!liveCells.is_cell(tc[i].node_id)) clean_ok = false;
            for (uint32_t k = 0; k < tc[i].leaf_count; ++k) {
                const uint64_t leaf = tl[tc[i].leaf_begin + k];
                if (!liveQt.is_leaf(leaf)) clean_ok = false;
                const QuadNode L = unpack_node_id(leaf);
                for (int e = 0; e < 4; ++e) {
                    uint64_t adj[2];
                    const int n = tgtQt.adjacent_leaves(L, (QuadEdge)e, adj);
                    for (int j = 0; j < n; ++j)
                        if (!liveQt.is_leaf(adj[j])) clean_ok = false;
                }
            }
        }
        CHECK(clean_ok);
    }

    std::printf("\n");
}

static void test_tessellation() {
    std::printf("[test_tessellation]\n");

    const Vec3f normals[] = {
        { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 0, 0, -1 },
        normalize(Vec3f{ 1, 1, 1 }), normalize(Vec3f{ -1, 2, -3 }),
        normalize(Vec3f{ 0.2f, -0.9f, 0.3f }),
    };
    for (const Vec3f& n : normals)
        CHECK(dot(n, oct_decode(oct_encode(n))) > 0.999f);
    CHECK(float_to_half(0.0f) == 0x0000);
    CHECK(float_to_half(1.0f) == 0x3C00);
    CHECK(float_to_half(0.5f) == 0x3800);

    PlanetGeometry planet;
    planet.center = { 0, 0, 0 };
    planet.radius = 1000.0;

    HeightmapProcedural hm;
    hm.amplitude = 20.0f;
    hm.frequency = 8.0f;

    const QuadNode     node{ 4, 2, 1, 1 };
    const NodeGeometry g = compute_node_geometry(node, planet);

    std::vector<uint8_t> vbuf(CHUNK_VERTEX_BYTES);
    std::vector<uint8_t> ibuf(CHUNK_INDEX_BYTES);

    TessJob job;
    job.node            = node;
    job.planet          = planet;
    job.anchor_world    = g.center_world;
    job.grid            = CHUNK_GRID;
    job.vertex_dest     = vbuf.data();
    job.index_dest      = ibuf.data();
    job.vertex_capacity = (uint32_t)vbuf.size();
    job.index_capacity  = (uint32_t)ibuf.size();

    const TessResult res = tessellate_chunk(job, hm);
    CHECK(res.ok);
    CHECK(res.vertex_count == MAX_CHUNK_VERTS);
    CHECK(res.index_count  == MAX_CHUNK_TRIS * 3);

    const ChunkVertex* verts = reinterpret_cast<const ChunkVertex*>(vbuf.data());
    const int32_t*     idx   = reinterpret_cast<const int32_t*>(ibuf.data());

    bool idx_ok = true, pos_ok = true;
    for (uint32_t k = 0; k < res.index_count; ++k)
        if (idx[k] < 0 || idx[k] >= static_cast<int32_t>(res.vertex_count)) idx_ok = false;
    for (uint32_t k = 0; k < res.vertex_count; ++k)
        if (!std::isfinite(verts[k].px) || !std::isfinite(verts[k].py)
                                        || !std::isfinite(verts[k].pz)) pos_ok = false;
    CHECK(idx_ok);
    CHECK(pos_ok);

    const Vec3f n_center  = decode_signed_normal(verts[MAX_CHUNK_VERTS / 2].normal_oct);
    const DVec3 outward_d = normalize(g.center_world - planet.center);
    CHECK(dot(n_center, Vec3f{ float(outward_d.x), float(outward_d.y),
                               float(outward_d.z) }) > 0.5f);

    dump_obj("planet_chunk.obj", verts, res.vertex_count, idx, res.index_count);
    std::printf("  tessellated face=%u lod=%u: %u verts, %u tris -> planet_chunk.obj\n",
                node.face, node.lod, res.vertex_count, res.index_count / 3);
    std::printf("\n");
}

static void test_tessellation_threaded() {
    std::printf("[test_tessellation_threaded]\n");

    PlanetGeometry planet;
    planet.center = { 0, 0, 0 };
    planet.radius = 6.371e6;
    HeightmapProcedural hm;

    constexpr uint32_t N = 64;
    std::vector<TessJob>   jobs(N);
    std::vector<TessResult> rpar(N), rser(N);
    std::vector<std::vector<uint8_t>> vpar(N), ipar(N), vser(N), iser(N);

    for (uint32_t k = 0; k < N; ++k) {
        QuadNode node;
        node.face = (uint8_t )(k % 6);
        node.lod  = 3;
        node.x    = (uint32_t)(k % 8);
        node.y    = (uint32_t)((k / 8) % 8);
        const NodeGeometry g = compute_node_geometry(node, planet);

        vpar[k].resize(CHUNK_VERTEX_BYTES); ipar[k].resize(CHUNK_INDEX_BYTES);
        vser[k].resize(CHUNK_VERTEX_BYTES); iser[k].resize(CHUNK_INDEX_BYTES);

        jobs[k].node            = node;
        jobs[k].planet          = planet;
        jobs[k].anchor_world    = g.center_world;
        jobs[k].grid            = CHUNK_GRID;
        jobs[k].vertex_capacity = CHUNK_VERTEX_BYTES;
        jobs[k].index_capacity  = CHUNK_INDEX_BYTES;
    }

    WorkerPool pool;
    const auto t0 = std::chrono::high_resolution_clock::now();
    pool.parallel_for(N, [&](uint32_t k) {
        TessJob j = jobs[k];
        j.vertex_dest = vpar[k].data();
        j.index_dest  = ipar[k].data();
        rpar[k] = tessellate_chunk(j, hm);
    });
    const auto t1 = std::chrono::high_resolution_clock::now();

    for (uint32_t k = 0; k < N; ++k) {
        TessJob j = jobs[k];
        j.vertex_dest = vser[k].data();
        j.index_dest  = iser[k].data();
        rser[k] = tessellate_chunk(j, hm);
    }

    bool all_ok = true, identical = true;
    for (uint32_t k = 0; k < N; ++k) {
        if (!rpar[k].ok || rpar[k].vertex_count != MAX_CHUNK_VERTS
                        || rpar[k].index_count  != MAX_CHUNK_TRIS * 3) all_ok = false;
        if (vpar[k] != vser[k] || ipar[k] != iser[k]) identical = false;
    }
    CHECK(all_ok);
    CHECK(identical);

    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double cpms = (ms > 0.0) ? double(N) / ms : 0.0;
    std::printf("  %u chunks on %u worker threads (+caller): %.2f ms = %.1f chunks/ms\n",
                N, pool.thread_count(), ms, cpms);
    std::printf("  throughput target >= 20 chunks/ms: %s\n",
                cpms >= 20.0 ? "MET" : "below target (tessellator can be optimized later)");
    std::printf("\n");
}

static bool check_seam(const ChunkVertex* A, uint32_t aBase, uint32_t aStep,
                       const ChunkVertex* B, uint32_t bBase, uint32_t bStep,
                       uint32_t G) {
    auto ap = [&](uint32_t k) {
        const ChunkVertex& v = A[aBase + k * aStep];
        return Vec3f{ v.px, v.py, v.pz };
    };
    auto bp = [&](uint32_t j) {
        const ChunkVertex& v = B[bBase + j * bStep];
        return Vec3f{ v.px, v.py, v.pz };
    };
    bool ok = true;
    for (uint32_t j = 0; j <= G / 2; ++j)
        if (length(ap(2 * j) - bp(j)) > 1e-3f) ok = false;
    for (uint32_t k = 1; k < G; k += 2) {
        const Vec3f mid = (ap(k - 1) + ap(k + 1)) * 0.5f;
        if (length(ap(k) - mid) > 1e-4f) ok = false;
    }
    return ok;
}

static void test_tessellation_seams() {
    std::printf("[test_tessellation_seams]\n");

    PlanetGeometry planet;
    planet.center = { 0, 0, 0 };
    planet.radius = 1000.0;
    HeightmapProcedural hm;
    hm.amplitude = 20.0f;
    hm.frequency = 8.0f;

    const uint32_t G    = CHUNK_GRID;
    const uint32_t edge = G + 1;
    std::vector<uint8_t> vA(CHUNK_VERTEX_BYTES), iA(CHUNK_INDEX_BYTES);
    std::vector<uint8_t> vB(CHUNK_VERTEX_BYTES), iB(CHUNK_INDEX_BYTES);

    auto tess = [&](QuadNode n, uint8_t mask,
                    std::vector<uint8_t>& vb, std::vector<uint8_t>& ib) {
        TessJob j;
        j.node            = n;
        j.planet          = planet;
        j.anchor_world    = planet.center;
        j.grid            = G;
        j.stitch_mask     = mask;
        j.vertex_dest     = vb.data();
        j.index_dest      = ib.data();
        j.vertex_capacity = (uint32_t)vb.size();
        j.index_capacity  = (uint32_t)ib.size();
        return tessellate_chunk(j, hm);
    };

    const ChunkVertex* A = reinterpret_cast<const ChunkVertex*>(vA.data());
    const ChunkVertex* B = reinterpret_cast<const ChunkVertex*>(vB.data());

    {
        const TessResult rA = tess(QuadNode{ 4, 3, 4, 2 }, 1u << EDGE_NEG_S, vA, iA);
        const TessResult rB = tess(QuadNode{ 4, 2, 1, 1 }, 0,                vB, iB);
        CHECK(rA.ok && rB.ok);
        CHECK(check_seam(A, 0, edge, B, G, edge, G));
    }

    {
        tess(QuadNode{ 4, 3, 4, 2 }, 0, vA, iA);
        tess(QuadNode{ 4, 2, 1, 1 }, 0, vB, iB);
        CHECK(!check_seam(A, 0, edge, B, G, edge, G));
    }

    {
        const TessResult rA = tess(QuadNode{ 4, 3, 7, 2 }, 1u << EDGE_POS_S, vA, iA);
        const TessResult rB = tess(QuadNode{ 0, 2, 0, 1 }, 0,                vB, iB);
        CHECK(rA.ok && rB.ok);
        CHECK(check_seam(A, G, edge, B, 0, edge, G));
    }

    std::printf("  1:2 stitch crack-free within-face and across a cube-face seam\n\n");
}

int main() {
    std::printf("=== Planet CPU tests ===\n\n");
    test_coords();
    test_cube_sphere();
    test_cube_neighbors();
    test_restricted_quadtree();
    test_blas_cells();
    test_tessellation();
    test_tessellation_threaded();
    test_tessellation_seams();
    std::printf("=== %d checks, %d failure(s) ===\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
