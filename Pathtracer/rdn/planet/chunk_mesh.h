#pragma once
//====================================
//PLANET - CHUNK MESH
//====================================
//Chunk tessellation grid, worst-case mesh budget, and the GPU vertex layout.
//Pure CPU (cstdint only) - deliberately free of <d3d12.h> so it can be shared
//by both the GPU resource pools (Phase 2) and the CPU tessellator (Phase 3).

#include <cstdint>

namespace planet {

//A chunk is a CHUNK_GRID x CHUNK_GRID grid of quads -> (CHUNK_GRID+1)^2 vertices.
//Every chunk uses the same grid resolution regardless of its LOD (finer LOD =
//smaller world patch, same vertex count = more detail).
constexpr uint32_t CHUNK_GRID       = 32;
constexpr uint32_t CHUNK_EDGE_VERTS = CHUNK_GRID + 1;                       // 33
constexpr uint32_t MAX_CHUNK_VERTS  = CHUNK_EDGE_VERTS * CHUNK_EDGE_VERTS;   // 1089
constexpr uint32_t MAX_CHUNK_TRIS   = CHUNK_GRID * CHUNK_GRID * 2;           // 2048

constexpr uint32_t MAX_BUILDS_PER_FRAME = 16;   // orchestrator build cap (Phase 4)

//GPU vertex: position 3xf32 (12 B) + octahedral normal uint32 (4 B)
//          + uv 2x f16 (4 B) = 20 B. uint16 indices (MAX_CHUNK_VERTS < 65536).
constexpr uint32_t CHUNK_VERTEX_STRIDE = 20;
constexpr uint32_t CHUNK_INDEX_STRIDE  = 2;
constexpr uint32_t CHUNK_VERTEX_BYTES  = MAX_CHUNK_VERTS * CHUNK_VERTEX_STRIDE;
constexpr uint32_t CHUNK_INDEX_BYTES   = MAX_CHUNK_TRIS * 3 * CHUNK_INDEX_STRIDE;

//exact GPU layout the tessellator writes and the BLAS / terrain shader read.
struct ChunkVertex {
    float    px, py, pz;   // chunk-local position (relative to the chunk anchor)
    uint32_t normal_oct;   // octahedral-encoded unit normal
    uint16_t u, v;         // surface uv, IEEE-754 half-float
};
static_assert(sizeof(ChunkVertex) == CHUNK_VERTEX_STRIDE, "ChunkVertex must be 20 bytes");

} // namespace planet
