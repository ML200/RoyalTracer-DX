#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include "../../src/Components/Vertex.h"
#include "mc_types.h"
#include "mc_zip.h"
#include "block_registry.h"

namespace mc { struct RawQuad; }
namespace mc {

class BlockRegistry;
class ModelResolver;

struct MaterialBuildStats {
    uint32_t bakedFaces     = 0;
    uint32_t statesResolved = 0;
    uint32_t statesMissing  = 0;
    uint32_t cubes          = 0;
    uint32_t modelBlocks    = 0;
    uint32_t invisible      = 0;
    uint32_t textures       = 0;
    uint32_t texturesMissing = 0;
    uint32_t materials      = 0;
    double   seconds        = 0.0;
};

class MaterialBuilder {
public:
    float lodOpaqueCoverage = 0.5f;
    bool opaqueLeaves = false;

    // Resolves textures and appends renderer materials for every block state.
    bool build(BlockRegistry& reg, IResourceProvider& res,
               MaterialSoA& materials, std::vector<std::string>& materialNames,
               std::vector<TextureData>& textures, int texIdBase,
               MaterialBuildStats& stats, std::string* err = nullptr);

private:
    enum class Kind : uint8_t { Textured, Cutout, Glass, Water, Flat };
    enum class Profile : uint8_t { Default, Foliage, ClearGlass };

    struct TexEntry {
        int   index      = -1;
        bool  hasAlpha   = false;
        float avg[3]     = { 0.5f, 0.5f, 0.5f };
        bool  missing    = false;
    };
    struct MaterialKey {
        int      tex;
        uint8_t  kind;
        uint8_t  profile;
        uint8_t  metal;
        uint8_t  gloss;
        uint32_t emissive;
        float    emissionScale;
        bool operator==(const MaterialKey& o) const {
            return tex == o.tex && kind == o.kind && profile == o.profile && metal == o.metal && gloss == o.gloss && emissive == o.emissive && emissionScale == o.emissionScale;
        }
    };
    struct KeyHash {
        size_t operator()(const MaterialKey& k) const noexcept {
            return (size_t)hash_u64(((uint64_t)(uint32_t)k.tex << 32) ^ ((uint64_t)k.kind << 8) ^ ((uint64_t)k.profile << 56) ^ ((uint64_t)k.metal << 16) ^ ((uint64_t)k.gloss << 24) ^ ((uint64_t)k.emissive << 32) ^ (uint64_t)(k.emissionScale * 100.0f));
        }
    };

    const TexEntry& texture(const std::string& name, uint32_t tintRgb, const std::string& overlay, uint32_t overlayTint);
    bool load_png(const std::string& name, std::vector<uint8_t>& rgba, int& w, int& h);
    TexEntry make_texture(const std::vector<uint8_t>& rgba, int w, int h, bool missing);
    const TexEntry& baked_texture(const std::vector<uint8_t>& rgba, int w, int h);
    uint16_t opaque_lod_material(const TexEntry& te, bool metal, bool gloss, float emit, const std::string& name, uint16_t fallback, bool force = false);
    bool quad_texels_opaque(const TexEntry& te, const RawQuad& q) const;
    uint16_t material_for(const TexEntry& te, Kind kind, bool metal, bool gloss, uint32_t emissiveRgb, float emissionScale, const std::string& debugName, bool allowFoliage = true);
    std::vector<Vec3f>* m_materialEmission = nullptr;
    std::vector<int32_t>* m_materialCutoutTexture = nullptr;
    std::vector<BlockRegistry::AlphaMask>* m_textureAlpha = nullptr;

    IResourceProvider*       m_res = nullptr;
    MaterialSoA*             m_materials = nullptr;
    std::vector<std::string>* m_materialNames = nullptr;
    std::vector<TextureData>* m_textures = nullptr;
    int                      m_texIdBase = 0;
    std::vector<uint8_t>*    m_materialAlpha = nullptr;
    std::unordered_map<std::string, TexEntry>                 m_texCache;
    std::unordered_map<MaterialKey, uint16_t, KeyHash>        m_matCache;
    std::unordered_map<std::string, std::vector<uint8_t>>     m_pngCache;
    std::unordered_map<std::string, std::pair<int, int>>      m_pngSize;
    std::unordered_map<uint64_t, TexEntry>                    m_bakedCache;
    MaterialBuildStats* m_stats = nullptr;
};

}
