#pragma once
//====================================
//PLANET - PROCEDURAL ROCK SCATTER
//====================================
//Generates a few boulder meshes (deformed icospheres) and scatters instances of
//them on the terrain surface in a disk that FOLLOWS the camera. Pure CPU /
//geometry - no DXR types, just plain descriptors the renderer fills with BLAS
//addresses and global-buffer bases. Contract with the renderer:
//
//  init:      generate_rock_variants() -> upload each as an ordinary scene mesh
//             (the engine's global vertex/index buffers + one BLAS per variant)
//             and record, per variant, its blas_va + vertexBase/indexBase/
//             materialBase/triCount.
//  per frame: RockScatter::update(camera_world, heightmap) refreshes the live
//             instance set when the camera ground-point drifts; record_tlas then
//             adds one TLAS instance + InstanceProperties per live rock into a
//             reserved props range [rockPropsBase, rockPropsBase + max_rocks).
//
//Placement is hashed off a WORLD-anchored grid, so a given patch of ground always
//grows the same rock - the live set only changes membership at the disk edge; the
//rocks themselves never swim as the camera moves. Rocks shade through the unified
//EvalSurfaceState path as normal scene meshes (their own rock material, _pad[0]==0),
//so they get NO terrain tint and NO terrain detail-normal.

#include "coordinate_system.h"   // Vec3<T>, DVec3, Vec3f, dot/cross/normalize/length
#include "cube_sphere.h"         // PlanetGeometry
#include <vector>
#include <cstdint>

namespace planet {

//One generated rock vertex. The renderer copies these into its engine Vertex /
//MeshGPU (position, normal, texCoord) before BuildGlobalMeshBuffers packs them
//into the global BTriVertex buffer.
struct RockVertex {
    Vec3f position;   // local space, ~unit radius (instance scale applied by the transform)
    Vec3f normal;
    float u = 0.0f, v = 0.0f;
};

struct RockMesh {
    std::vector<RockVertex> vertices;
    std::vector<uint32_t>   indices;   // triangle list
};

//Generate `count` boulder variants. Deterministic in `seed`. subdiv 1 -> ~80 tris,
//2 -> ~320 tris per variant. Each variant is an icosphere radially displaced by
//multi-octave value noise plus a mild anisotropic squash for variety.
std::vector<RockMesh> generate_rock_variants(int count, int subdiv, uint32_t seed);

//A live rock placement. anchor_world is FP64 (the origin-relative translation is
//formed at record time in FP64, exactly like terrain cells). rot_scale is the
//row-major 3x3 local->world rotation * uniform-scale.
struct RockInstance {
    DVec3    anchor_world{};
    float    rot_scale[9] = { 1,0,0, 0,1,0, 0,0,1 };
    uint32_t variant   = 0;
    uint32_t stable_id = 0;   // slot in [0, max_rocks)
};

struct RockScatterConfig {
    PlanetGeometry planet{};            // center + surface radius (m)
    double   region_radius_m  = 120.0;  // scatter-disk radius around the camera ground point
    double   cell_size_m      = 7.0;    // one candidate rock per ground cell of this size
    float    coverage         = 0.16f;  // fraction of cells that actually spawn a rock
    float    min_scale_m      = 0.35f;  // rock half-size range (multiplies the ~unit mesh)
    float    max_scale_m      = 2.4f;
    double   retrigger_move_m = 25.0;   // rebuild the set when the ground point moves this far (arc len)
    uint32_t max_rocks        = 4096;   // hard cap == reserved instanceProps slots
    uint32_t seed             = 0x90CCEE17u;
};

//Heightmap interface. The renderer's planet::HeightmapCubemap already returns a
//metre offset from sample(dir, lod); wrap it in this so the scatter can snap
//rocks to the terrain surface without depending on the cubemap type directly.
struct IRockHeight {
    virtual float sample_height_m(const DVec3& dir_unit) const = 0;
    virtual ~IRockHeight() = default;
};

class RockScatter {
public:
    void configure(const RockScatterConfig& cfg, int variant_count);

    //Rebuilds the live set if the camera ground point drifted past the hysteresis
    //threshold (or on first call). Returns true if the set changed this call.
    bool update(const DVec3& camera_world, const IRockHeight& height);

    const std::vector<RockInstance>& live()   const { return m_live; }
    const RockScatterConfig&         config() const { return m_cfg; }

private:
    void rebuild(const DVec3& ground_dir, const IRockHeight& height);

    RockScatterConfig         m_cfg{};
    int                       m_variantCount = 1;
    bool                      m_have = false;
    DVec3                     m_lastGroundDir{};
    std::vector<RockInstance> m_live;
};

} // namespace planet
