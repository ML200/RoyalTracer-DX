#pragma once

#include <cstdint>
#include "coordinate_system.h"
#include "cube_sphere.h"
#include "chunk_mesh.h"
#include "heightmap_source.h"

namespace planet {

struct TessJob {
    QuadNode       node;
    PlanetGeometry planet;
    DVec3          anchor_world{}; // Output positions use this local origin.
    DVec3          scene_origin{};
    uint32_t       grid = CHUNK_GRID;
    uint8_t        stitch_mask = 0;

    void*    vertex_dest     = nullptr; // Caller-owned output buffers.
    void*    index_dest      = nullptr;
    uint32_t vertex_capacity = 0;
    uint32_t index_capacity  = 0;
    uint32_t index_vertex_base = 0;
};

struct TessResult {
    uint32_t vertex_count = 0;
    uint32_t index_count  = 0;
    bool     ok           = false;
};

TessResult tessellate_chunk(const TessJob& job, const IHeightmapSource& heightmap);

// Packs unit normals into the mesh's 16:16 octahedral format.
uint32_t oct_encode(const Vec3f& n);
Vec3f    oct_decode(uint32_t e);
uint16_t float_to_half(float f);

}
