#pragma once

#include "coordinate_system.h"
#include "cube_sphere.h"
#include <vector>
#include <cstdint>

namespace planet {

struct RockVertex {
    Vec3f position;
    Vec3f normal;
    float u = 0.0f, v = 0.0f;
};

struct RockMesh {
    std::vector<RockVertex> vertices;
    std::vector<uint32_t>   indices;
};

// Deformed icospheres.
std::vector<RockMesh> generate_rock_variants(int count, int subdiv, uint32_t seed);

struct RockInstance {
    DVec3    anchor_world{};
    float    rot_scale[9] = { 1,0,0, 0,1,0, 0,0,1 };
    uint32_t variant   = 0;
    uint32_t stable_id = 0;
};

struct RockScatterConfig {
    PlanetGeometry planet{};
    double   region_radius_m  = 120.0;
    double   cell_size_m      = 7.0;
    float    coverage         = 0.16f;
    float    min_scale_m      = 0.35f;
    float    max_scale_m      = 2.4f;
    double   retrigger_move_m = 25.0;
    uint32_t max_rocks        = 4096;
    uint32_t seed             = 0x90CCEE17u;
};

struct IRockHeight {
    virtual float sample_height_m(const DVec3& dir_unit) const = 0;
    virtual ~IRockHeight() = default;
};

class RockScatter {
public:
    void configure(const RockScatterConfig& cfg, int variant_count);

    // Rebuilds only past the movement threshold.
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

}
