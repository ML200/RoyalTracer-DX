#pragma once

#include <cstdint>

namespace planet {

constexpr uint32_t CHUNK_GRID       = 32;
constexpr uint32_t CHUNK_EDGE_VERTS = CHUNK_GRID + 1;
constexpr uint32_t MAX_CHUNK_VERTS  = CHUNK_EDGE_VERTS * CHUNK_EDGE_VERTS;
constexpr uint32_t MAX_CHUNK_TRIS   = CHUNK_GRID * CHUNK_GRID * 2;

constexpr uint32_t CHUNK_VERTEX_STRIDE = 20;
constexpr uint32_t CHUNK_INDEX_STRIDE  = 4;
constexpr uint32_t CHUNK_VERTEX_BYTES  = MAX_CHUNK_VERTS * CHUNK_VERTEX_STRIDE;
constexpr uint32_t CHUNK_INDEX_BYTES   = MAX_CHUNK_TRIS * 3 * CHUNK_INDEX_STRIDE;

struct ChunkVertex {
    float    px, py, pz;
    uint32_t normal_oct;
    uint16_t u, v;
};
static_assert(sizeof(ChunkVertex) == CHUNK_VERTEX_STRIDE, "ChunkVertex must be 20 bytes");

}
