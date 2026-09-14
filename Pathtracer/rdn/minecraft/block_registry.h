#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>
#include "mc_types.h"

namespace mc {

struct NbtValue;

constexpr BlockId INVALID_BLOCK = 0xFFFFu;

struct BlockStateDesc {
    std::string name;
    std::vector<std::pair<std::string, std::string>> props;
    std::string canonical;
    std::string_view prop(std::string_view key) const {
        for (const auto& p : props) if (p.first == key) return p.second;
        return {};
    }
    std::string_view path() const {
        const size_t c = name.find(':');
        return c == std::string::npos ? std::string_view(name) : std::string_view(name).substr(c + 1);
    }
};

enum class Significance : uint8_t {
    None    = 0,
    Partial = 1,
    Full    = 2,
};

enum class TintKind : uint8_t { None = 0, Grass = 1, Foliage = 2, Water = 3 };

struct BlockQuad {
    Vec3f    pos[4];
    float    uv[4][2];
    Vec3f    normal;
    uint16_t material   = 0;
    uint8_t  cullFace   = FACE_NONE;
    uint8_t  alphaGeom  = 0;
};

constexpr uint16_t NO_MATERIAL = 0xFFFFu;

struct BlockInfo {
    bool         isCube      = false;
    bool         fullOpaque  = false;
    bool         cullSameId  = false;
    bool         hasQuads    = false;
    Significance sig         = Significance::None;
    TintKind     tint        = TintKind::None;
    uint8_t      lightLevel  = 0;
    uint8_t      emissive    = 0;
    uint8_t      lodFaceSolid = 0;
    uint8_t      volume      = 0;
    uint16_t     faceMaterial[6]     = { NO_MATERIAL, NO_MATERIAL, NO_MATERIAL, NO_MATERIAL, NO_MATERIAL, NO_MATERIAL };
    uint16_t     lodFaceMaterial[6]  = { NO_MATERIAL, NO_MATERIAL, NO_MATERIAL, NO_MATERIAL, NO_MATERIAL, NO_MATERIAL };
    uint16_t     flatFaceMaterial[6] = { NO_MATERIAL, NO_MATERIAL, NO_MATERIAL, NO_MATERIAL, NO_MATERIAL, NO_MATERIAL };
    uint32_t     quadBegin = 0, quadCount = 0;
};

class BlockRegistry {
public:
    BlockRegistry();

    // Interns a canonical state and assigns its stable dense ID.
    BlockId intern(const std::string& canonical, const std::string& name,
                   const std::vector<std::pair<std::string, std::string>>& props);
    BlockId intern(const std::string& name,
                   std::vector<std::pair<std::string, std::string>> props);
    // Looks up a state without mutating the registry.
    BlockId find(const std::string& canonical) const;

    size_t count() const { return m_states.size(); }
    const BlockStateDesc& desc(BlockId id) const { return m_states[id]; }
    const BlockInfo&      info(BlockId id) const { return m_info[id]; }
    BlockInfo&            info(BlockId id)       { return m_info[id]; }

    std::vector<BlockQuad> quads;
    std::vector<uint8_t> materialAlpha;
    std::vector<Vec3f> materialEmission;
    std::vector<int32_t> materialCutoutTexture;
    struct AlphaMask {
        int width = 0, height = 0;
        std::vector<uint8_t> alpha;
        bool empty() const { return alpha.empty(); }
    };
    std::vector<AlphaMask> textureAlpha;

    static std::string make_canonical(const std::string& name,
                                      const std::vector<std::pair<std::string, std::string>>& sortedProps);
    static bool is_air_name(std::string_view name);

private:
    mutable std::mutex                        m_mutex;
    std::vector<BlockStateDesc>               m_states;
    std::vector<BlockInfo>                    m_info;
    std::unordered_map<std::string, BlockId>  m_index;
};

class InternCache {
public:
    explicit InternCache(BlockRegistry& reg) : m_reg(reg) {}
    BlockId intern(std::string_view name, const NbtValue* props);
private:
    BlockRegistry& m_reg;
    std::unordered_map<std::string, BlockId> m_local;
    std::string m_scratch;
    std::vector<std::pair<std::string, std::string>> m_props;
};

}
