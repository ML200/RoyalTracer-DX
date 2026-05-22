#pragma once
//====================================
//PLANET - TESSELLATOR
//====================================
//Generates one chunk's triangle mesh on a worker thread: a CHUNK_GRID x
//CHUNK_GRID quad grid over a cube-sphere patch, displaced by the heightmap,
//written straight into caller-provided output memory. Pure CPU; no GPU types,
//no heap allocation.

#include <cstdint>
#include "coordinate_system.h"
#include "cube_sphere.h"
#include "chunk_mesh.h"
#include "heightmap_source.h"

namespace planet {

//One tessellation job. The Phase 4 orchestrator fills this and points the
//destinations at an UploadRing::Region; the standalone test points them at
//plain memory. (The plan sketched UploadRing::Region fields here - kept as raw
//pointers so the tessellator stays GPU-free and unit-testable, per its own
//OBJ-dump test requirement.)
struct TessJob {
    QuadNode       node;                  // which cube-sphere quadtree node
    PlanetGeometry planet;
    DVec3          anchor_world{};        // output positions are relative to this
    uint32_t       grid = CHUNK_GRID;     // quads per edge
    //Per-edge seam-stitch mask: bit (1u << QuadEdge) set => that edge faces a
    //COARSER neighbour (one LOD level down), so its odd boundary vertices are
    //snapped onto the coarse neighbour's chord for a crack-free 1:2 boundary.
    //An edge facing a same-LOD neighbour needs no stitch (it already matches);
    //a finer neighbour stitches its own side. 0 = uniform chunk (default).
    uint8_t        stitch_mask = 0;

    void*    vertex_dest     = nullptr;   // ChunkVertex[] destination
    void*    index_dest      = nullptr;   // uint16[]     destination
    uint32_t vertex_capacity = 0;         // bytes available at vertex_dest
    uint32_t index_capacity  = 0;         // bytes available at index_dest
};

struct TessResult {
    uint32_t vertex_count = 0;
    uint32_t index_count  = 0;
    bool     ok           = false;        // false if a destination was too small
};

//Tessellate one chunk. Thread-safe: distinct jobs write distinct destinations
//and the heightmap is sampled read-only.
TessResult tessellate_chunk(const TessJob& job, const IHeightmapSource& heightmap);

//----- vertex-attribute codecs (shared with the test + a reference for HLSL) -----
uint32_t oct_encode(const Vec3f& n);      // unit normal -> 16:16 octahedral uint32
Vec3f    oct_decode(uint32_t e);
uint16_t float_to_half(float f);          // IEEE-754 binary16 pack

} // namespace planet
