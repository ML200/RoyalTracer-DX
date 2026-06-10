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
//          + uv 2x f16 (4 B) = 20 B. This layout is BYTE-IDENTICAL to the
//scene's STriVertex {float3 vertex; uint packedNormal; half2 texCoord}, so a
//terrain vertex IS a scene vertex - it shades through the unified
//EvalSurfaceState with no terrain branch. The normal is packed in the scene's
//signed-int16 octahedral format (UnpackNormal_INT / EncodeNormalOct).
//
//Indices are 32-bit and store ABSOLUTE indices into the unified global vertex
//buffer (a cell is one BLAS geometry over its K contiguous leaf slots), so they
//match the scene index convention: the BLAS builds with R32_UINT and
//VertexBuffer.StartAddress = the combined buffer base, and the shader reads the
//same buffer.
constexpr uint32_t CHUNK_VERTEX_STRIDE = 20;
constexpr uint32_t CHUNK_INDEX_STRIDE  = 4;   // 32-bit absolute indices
constexpr uint32_t CHUNK_VERTEX_BYTES  = MAX_CHUNK_VERTS * CHUNK_VERTEX_STRIDE;
constexpr uint32_t CHUNK_INDEX_BYTES   = MAX_CHUNK_TRIS * 3 * CHUNK_INDEX_STRIDE;

//exact GPU layout the tessellator writes and the BLAS / shader read.
struct ChunkVertex {
    float    px, py, pz;   // chunk-local position (relative to the chunk anchor)
    uint32_t normal_oct;   // octahedral-encoded unit normal (scene packedNormal format)
    uint16_t u, v;         // equiangular cubed-sphere FACE uv [0,1], IEEE-754 half
                           // (indexes the per-face surface_color/normal cubemaps)
};
static_assert(sizeof(ChunkVertex) == CHUNK_VERTEX_STRIDE, "ChunkVertex must be 20 bytes");

} // namespace planet
