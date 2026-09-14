#pragma once

#include <cstdint>
#include <vector>
#include "mc_types.h"

namespace mc {

class BlockRegistry;
class VoxelStore;

struct MeshVertex {
    float    px, py, pz;
    uint32_t packedNormal;
    uint16_t u, v;
};
static_assert(sizeof(MeshVertex) == 20, "MeshVertex must match BTriVertex (20 bytes)");

struct ChunkMesh {
    std::vector<MeshVertex> vertices;
    std::vector<uint32_t>   indices;
    std::vector<uint32_t>   materials;
    std::vector<uint64_t>   ommKeys;
    uint32_t opaqueTriCount      = 0;
    uint32_t alphaTriCount       = 0;
    uint32_t opaqueLightTriCount = 0;
    uint32_t alphaLightTriCount  = 0;
    uint32_t quadCount           = 0;
    void clear() {
        vertices.clear(); indices.clear(); materials.clear(); ommKeys.clear();
        opaqueTriCount = alphaTriCount = opaqueLightTriCount = alphaLightTriCount = quadCount = 0;
    }
    bool empty() const { return indices.empty(); }
    uint32_t triangle_count() const { return (uint32_t)(indices.size() / 3); }
    uint32_t light_tri_count() const { return opaqueLightTriCount + alphaLightTriCount; }
    uint32_t light_tri_index(uint32_t k) const {
        return k < opaqueLightTriCount ? k : opaqueTriCount + (k - opaqueLightTriCount);
    }
};

struct MeshParams {
    bool flatMaterials = false;
};

class ChunkMesher {
public:
    ChunkMesher(const BlockRegistry& reg, const VoxelStore& store);

    // Reads neighbors for culling and writes one renderable chunk mesh.
    void mesh(const NodeKey& key, const MeshParams& params, ChunkMesh& out);

    static void face_quad_uvs(int face, int w, int h, float uvScale, float uv[4][2]);

private:
    static constexpr int W  = CHUNK_SIZE + 2;
    static constexpr int CW = 2 * CHUNK_SIZE + 2;

    Voxel at(int x, int y, int z) const {
        return m_window[((size_t)(z + 1) * W + (size_t)(y + 1)) * W + (size_t)(x + 1)];
    }
    Voxel child_at(int x, int y, int z) const {
        return m_child[((size_t)(z + 1) * CW + (size_t)(y + 1)) * CW + (size_t)(x + 1)];
    }
    uint16_t face_material(int level, const int c[3], int f, const struct BlockInfo& parent, bool flat) const;
    bool renderable(Voxel v, int level, const struct BlockInfo*& info) const;
    bool occludes(Voxel neighbour, Voxel self, int level, bool selfCullSame) const;
    void greedy_faces(int level, const MeshParams& p, ChunkMesh& out);
    void model_quads(const MeshParams& p, ChunkMesh& out);
    void lamp_cubes(int level, const MeshParams& p, ChunkMesh& out);
    static bool lamp_voxel(Voxel v, int level, const struct BlockInfo& info, float& sideBlocks);
    void emit_quad(const Vec3f p[4], const float uv[4][2], const Vec3f& n, uint16_t material, ChunkMesh& out, uint64_t omm0 = 0, uint64_t omm1 = 0);
    void emit_face_quad(int face, const int corner[4][3], float s, float uvScale, float insetVoxels, uint16_t material, ChunkMesh& out, uint64_t omm0 = 0, uint64_t omm1 = 0);
    void push_triangles(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3, uint16_t material, ChunkMesh& out, uint64_t omm0, uint64_t omm1);

    const BlockRegistry& m_reg;
    const VoxelStore&    m_store;
    std::vector<Voxel>   m_window;
    std::vector<Voxel>   m_child;
    std::vector<uint32_t> m_mask;
    struct Bucket { std::vector<uint32_t> idx, mat; std::vector<uint64_t> omm; void clear() { idx.clear(); mat.clear(); omm.clear(); } };
    Bucket m_opaqueLit, m_opaque, m_alphaLit, m_alpha;
    static constexpr int CORNERS = CHUNK_SIZE + 1;
    std::vector<uint32_t> m_cornerVertex, m_cornerGen;
    uint32_t m_gen = 0;
};

}
