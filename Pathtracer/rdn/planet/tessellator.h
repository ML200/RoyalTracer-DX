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
    //FP64 floating-origin the renderer will use when building the unified-TLAS
    //instance transform for this cell. The tessellator pre-compensates each
    //vertex by the quantisation error of FP32_cast(anchor_world - scene_origin),
    //so the GPU sum (instance_translation + vertex_FP32) reproduces the
    //precise world-relative position without the planet-scale FP32 grid snap
    //that otherwise pulls seam chunks to a 0.5 m grid. The compensation is
    //invariant under the renderer's 1 km scene-origin snaps (the snap delta is
    //FP32-exact), so a chunk tessellated for one scene origin renders
    //correctly across any subsequent snap until it is rebuilt for another
    //reason.
    DVec3          scene_origin{};
    uint32_t       grid = CHUNK_GRID;     // quads per edge
    //Per-edge + per-corner seam-stitch mask:
    //  bits 0..3 (one per QuadEdge): edge faces a COARSER neighbour. The fine
    //            chunk's odd boundary vertices are snapped onto the chord
    //            between their even neighbours for a crack-free 1:2 boundary,
    //            and EVERY vertex on that edge evaluates noise at the
    //            coarser neighbour's footprint so the heights agree.
    //  bits 4..7 (one per corner, encoding (i,j)=(0,0),(G,0),(0,G),(G,G)):
    //            the chunk's DIAGONAL neighbour at that corner is COARSER,
    //            even though both adjacent edges are at the same LOD. The
    //            corner vertex evaluates noise at the coarser footprint
    //            (same factor as edge stitching) so all 4 chunks meeting at
    //            that point agree on the height. The corner is never snapped
    //            (its position is fixed by the chunk's (s,t) parametrisation);
    //            we only adjust which noise octaves it sees. Without this
    //            bit, an interior fine chunk with a same-LOD edge neighbour
    //            and a coarser diagonal would carry one extra octave at that
    //            corner, producing a sub-metre crack visible as a dark spot
    //            at every LOD-transition corner (a ring around the camera).
    //  0 = uniform chunk (default).
    uint8_t        stitch_mask = 0;

    void*    vertex_dest     = nullptr;   // ChunkVertex[] destination
    void*    index_dest      = nullptr;   // int32[]      destination
    uint32_t vertex_capacity = 0;         // bytes available at vertex_dest
    uint32_t index_capacity  = 0;         // bytes available at index_dest
    //Absolute index of THIS leaf's vertex 0 in the unified global vertex buffer.
    //tessellate_chunk emits indices as (index_vertex_base + local_vertex), so
    //the cell's merged single-geometry BLAS and the shader both read the
    //combined buffer with no per-leaf rebasing. The orchestrator sets this to
    //cell_vertex_base_elems + leaf_slot * MAX_CHUNK_VERTS.
    uint32_t index_vertex_base = 0;
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
