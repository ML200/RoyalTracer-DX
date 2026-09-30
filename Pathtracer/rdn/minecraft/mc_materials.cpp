#include "mc_materials.h"
#include "block_registry.h"
#include "mc_models.h"
#include "mc_bake.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "../../src/Util/stb_image.h"
#include <DirectXTex.h>

namespace mc {

namespace {

constexpr uint32_t TINT_NONE    = 0xFFFFFFu;
constexpr uint32_t TINT_GRASS   = 0x91BD59u;
constexpr uint32_t TINT_FOLIAGE = 0x77AB2Fu;
constexpr uint32_t TINT_BIRCH   = 0x80A755u;
constexpr uint32_t TINT_SPRUCE  = 0x619961u;
constexpr uint32_t TINT_WATER   = 0x3F76E4u;

bool contains(const std::string_view s, const char* sub) { return s.find(sub) != std::string_view::npos; }

uint64_t hash_rgba(const std::vector<uint8_t>& rgba, int w) {
    uint64_t h = 1469598103934665603ull ^ (uint64_t)(uint32_t)w;
    for (uint8_t b : rgba) { h ^= b; h *= 1099511628211ull; }
    return h;
}

float srgb_to_linear(float c) {
    return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

bool foliage_block(std::string_view path) {
    if (path.starts_with("potted_")) path.remove_prefix(7);
    if (path.ends_with("_leaves") || path.ends_with("_sapling") || path.ends_with("_flower") ||
        path.ends_with("_flowers") || path.ends_with("_tulip") || path.ends_with("_orchid") ||
        path.ends_with("_grass") || path.ends_with("_fern") || path.ends_with("_bush") ||
        path.ends_with("_vines") || path.ends_with("_vines_plant") || path.ends_with("_roots") ||
        path.ends_with("_fungus") || path.ends_with("_mushroom") || path.ends_with("_sprouts") ||
        path.ends_with("_plant") || path.ends_with("_crop") || path.ends_with("_blossom") ||
        path.ends_with("_coral") || path.ends_with("_coral_fan") || path.ends_with("_coral_wall_fan") ||
        path.ends_with("_dripleaf") || path.ends_with("_eyeblossom") || path.ends_with("_moss_carpet") ||
        path.ends_with("_hanging_moss")) return true;
    static constexpr std::string_view plants[] = {
        "leaves", "grass", "fern", "vine", "bush", "azalea", "flowering_azalea",
        "dandelion", "poppy", "allium", "azure_bluet", "oxeye_daisy", "cornflower",
        "lily_of_the_valley", "wither_rose", "sunflower", "lilac", "peony", "torchflower",
        "wildflowers", "pink_petals", "leaf_litter", "lily_pad", "glow_lichen", "moss_carpet",
        "sugar_cane", "bamboo", "mangrove_propagule", "seagrass", "tall_seagrass", "kelp", "kelp_plant", "sea_pickle",
        "wheat", "carrots", "potatoes", "beetroots", "nether_wart", "cocoa",
        "melon_stem", "pumpkin_stem", "attached_melon_stem", "attached_pumpkin_stem",
        "torchflower_crop", "pitcher_crop", "big_dripleaf_stem"
    };
    for (const std::string_view plant : plants) if (path == plant) return true;
    return false;
}

// Decorations dropped at coarse LODs.
bool insignificant_block(std::string_view path) {
    static const char* subs[] = {
        "torch", "flower", "sapling", "tall_grass", "fern", "rail", "carpet", "pressure_plate", "button",
        "lever", "sign", "banner", "ladder", "vine", "redstone_wire", "tripwire", "cobweb", "flower_pot",
        "potted_", "candle", "dead_bush", "kelp", "seagrass", "lichen", "sugar_cane", "lily_pad", "cocoa",
        "wheat", "carrots", "potatoes", "beetroots", "nether_wart", "sweet_berry", "fire", "skull",
        "_head", "coral_fan", "coral_wall", "end_rod", "chain", "bamboo", "dripleaf", "hanging_roots",
        "spore_blossom", "glow_lichen", "sculk_vein", "pointed_dripstone", "azalea", "moss_carpet",
        "snow", "string", "comparator", "repeater", "scaffolding", "lantern", "bell", "conduit",
        "cake", "brewing_stand", "hopper", "cauldron", "grindstone", "lectern", "campfire",
        "structure_void", "light", "barrier", "air", "moving_piston", "piston_head", "frame",
        "attached_", "pumpkin_stem", "melon_stem", "tube_coral", "brain_coral", "bubble_coral",
        "fire_coral", "horn_coral", "big_dripleaf", "small_dripleaf", "sea_pickle", "turtle_egg",
        "frogspawn", "mangrove_propagule", "pink_petals", "torchflower", "pitcher", "wildflowers",
        "bush", "firefly", "leaf_litter", "cactus_flower",
    };
    for (const char* s : subs) if (contains(path, s)) {
        if (contains(path, "grass_block") || contains(path, "snow_block") || contains(path, "moss_block") ||
            contains(path, "lantern") && (contains(path, "sea_lantern") || contains(path, "jack_o_lantern")) ||
            contains(path, "chain_command") || contains(path, "cake") && contains(path, "block") ||
            contains(path, "light_") && (contains(path, "concrete") || contains(path, "wool") || contains(path, "terracotta") || contains(path, "glass") || contains(path, "candle") || contains(path, "bed") || contains(path, "banner") || contains(path, "carpet") || contains(path, "shulker") || contains(path, "weighted")))
            continue;
        return true;
    }
    return path == "grass" || path == "short_grass";
}

bool cube_fallback_block(std::string_view path) {
    return contains(path, "chest") || contains(path, "shulker_box") || contains(path, "bed") && !contains(path, "bedrock")
        || contains(path, "barrel") || contains(path, "furnace") || contains(path, "smoker") || contains(path, "beehive")
        || contains(path, "decorated_pot") || path == "spawner";
}

bool metal_block(std::string_view path) {
    static const char* subs[] = { "iron_block", "gold_block", "copper", "netherite_block", "diamond_block", "emerald_block",
                                  "iron_bars", "anvil", "lightning_rod", "iron_door", "iron_trapdoor", "raw_iron_block", "raw_gold_block",
                                  "chain", "cauldron", "hopper", "rail", "lantern", "bell", "grindstone", "smithing_table", "brewing_stand" };
    for (const char* s : subs) if (contains(path, s)) return true;
    return false;
}
bool glossy_block(std::string_view path) {
    static const char* subs[] = { "polished_", "smooth_", "quartz", "glazed_terracotta", "prismarine", "purpur", "marble", "tile" };
    for (const char* s : subs) if (contains(path, s)) return true;
    return false;
}
bool glass_block(std::string_view path) {
    if (contains(path, "packed_ice") || contains(path, "blue_ice")) return false;
    return contains(path, "glass") || path == "ice" || contains(path, "tinted_glass");
}
float emission_scale(const BlockStateDesc& d) {
    // Night City imports: emission independent of block colour.
    const std::string_view importedEmission = d.prop("nightcity_emission");
    if (importedEmission == "2") return 2.0f;
    if (importedEmission == "6") return 6.0f;
    if (importedEmission == "12") return 12.0f;
    const std::string_view p = d.path();
    if (contains(p, "coral") || contains(p, "torchflower") || contains(p, "firefly")) return 0.0f;
    if (contains(p, "glowstone") || contains(p, "sea_lantern") || contains(p, "shroomlight") || contains(p, "froglight")) return 6.0f;
    if (p == "lava" || contains(p, "magma_block")) return 4.0f;
    if (contains(p, "jack_o_lantern") || contains(p, "end_rod") || contains(p, "beacon")) return 5.0f;
    if (contains(p, "redstone_lamp") || contains(p, "campfire") || contains(p, "furnace") || contains(p, "smoker"))
        return d.prop("lit") == "true" ? 5.0f : 0.0f;
    if (contains(p, "torch") || contains(p, "lantern") || contains(p, "fire") || contains(p, "candle") && d.prop("lit") == "true"
        || contains(p, "glow_lichen") || contains(p, "crying_obsidian") || contains(p, "respawn_anchor") || contains(p, "amethyst_cluster"))
        return 4.0f;
    return 0.0f;
}

uint32_t tint_for(const BlockStateDesc& d, int tintIndex) {
    if (tintIndex < 0) return TINT_NONE;
    const std::string_view p = d.path();
    if (contains(p, "birch_leaves"))  return TINT_BIRCH;
    if (contains(p, "spruce_leaves")) return TINT_SPRUCE;
    if (contains(p, "cherry_leaves") || contains(p, "azalea_leaves") || contains(p, "pale_oak_leaves")) return TINT_NONE;
    if (contains(p, "leaves") || contains(p, "vine") || contains(p, "lily_pad") || contains(p, "mangrove")) return TINT_FOLIAGE;
    if (contains(p, "grass") || contains(p, "fern") || contains(p, "sugar_cane") || contains(p, "pink_petals") || contains(p, "bush") || contains(p, "leaf_litter")) return TINT_GRASS;
    if (contains(p, "water") || contains(p, "bubble_column")) return TINT_WATER;
    return TINT_NONE;
}

int face_of_normal(const Vec3f& n) {
    const float ax = std::fabs(n.x), ay = std::fabs(n.y), az = std::fabs(n.z);
    if (ay >= ax && ay >= az) return n.y > 0 ? FACE_UP : FACE_DOWN;
    if (ax >= az)             return n.x > 0 ? FACE_EAST : FACE_WEST;
    return n.z > 0 ? FACE_SOUTH : FACE_NORTH;
}

bool full_cube_face(const RawQuad& q, int& face) {
    face = face_of_normal(q.normal);
    if (std::fabs(std::fabs(q.normal.x) + std::fabs(q.normal.y) + std::fabs(q.normal.z) - 1.0f) > 1e-3f) return false;
    const int axis = (face == FACE_DOWN || face == FACE_UP) ? 1 : ((face == FACE_WEST || face == FACE_EAST) ? 0 : 2);
    const float plane = FACE_DIR[face][axis] > 0 ? 16.0f : 0.0f;
    float lo[3] = { 1e9f, 1e9f, 1e9f }, hi[3] = { -1e9f, -1e9f, -1e9f };
    for (int k = 0; k < 4; ++k) {
        const float p[3] = { q.pos[k].x, q.pos[k].y, q.pos[k].z };
        if (std::fabs(p[axis] - plane) > 1e-3f) return false;
        for (int a = 0; a < 3; ++a) { lo[a] = std::min(lo[a], p[a]); hi[a] = std::max(hi[a], p[a]); }
    }
    for (int a = 0; a < 3; ++a) {
        if (a == axis) continue;
        if (std::fabs(lo[a]) > 1e-3f || std::fabs(hi[a] - 16.0f) > 1e-3f) return false;
    }
    return true;
}

}

bool MaterialBuilder::load_png(const std::string& name, std::vector<uint8_t>& rgba, int& w, int& h) {
    const auto it = m_pngCache.find(name);
    if (it != m_pngCache.end()) {
        rgba = it->second;
        const auto sz = m_pngSize[name];
        w = sz.first; h = sz.second;
        return !rgba.empty();
    }
    std::vector<uint8_t> bytes;
    rgba.clear();
    w = h = 0;
    if (m_res->read(ModelResolver::texture_resource_path(name), bytes)) {
        int c = 0;
        unsigned char* px = stbi_load_from_memory(bytes.data(), (int)bytes.size(), &w, &h, &c, 4);
        if (px) {
            if (h > w && w > 0 && (h % w) == 0) h = w;
            rgba.assign(px, px + (size_t)w * h * 4);
            stbi_image_free(px);
        }
    }
    m_pngCache[name] = rgba;
    m_pngSize[name] = { w, h };
    return !rgba.empty();
}

const MaterialBuilder::TexEntry& MaterialBuilder::texture(const std::string& name, uint32_t tintRgb,
                                                          const std::string& overlay, uint32_t overlayTint) {
    char keyBuf[64];
    std::snprintf(keyBuf, sizeof(keyBuf), "|%06x|%06x", tintRgb & 0xFFFFFFu, overlayTint & 0xFFFFFFu);
    const std::string key = name + keyBuf + "|" + overlay;
    const auto it = m_texCache.find(key);
    if (it != m_texCache.end()) return it->second;

    TexEntry te;
    std::vector<uint8_t> rgba;
    int w = 0, h = 0;
    if (!load_png(name, rgba, w, h)) {
        w = h = 16;
        rgba.assign((size_t)w * h * 4, 255);
        for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
            uint8_t* p = &rgba[((size_t)y * w + x) * 4];
            const bool m = ((x >> 3) ^ (y >> 3)) & 1;
            p[0] = m ? 255 : 0; p[1] = 0; p[2] = m ? 255 : 0; p[3] = 255;
        }
        te.missing = true;
        ++m_stats->texturesMissing;
    }
    if (tintRgb != TINT_NONE) {
        const float tr = ((tintRgb >> 16) & 255) / 255.0f, tg = ((tintRgb >> 8) & 255) / 255.0f, tb = (tintRgb & 255) / 255.0f;
        for (size_t i = 0; i < rgba.size(); i += 4) {
            rgba[i + 0] = (uint8_t)std::lround(rgba[i + 0] * tr);
            rgba[i + 1] = (uint8_t)std::lround(rgba[i + 1] * tg);
            rgba[i + 2] = (uint8_t)std::lround(rgba[i + 2] * tb);
        }
    }
    if (!overlay.empty()) {
        std::vector<uint8_t> ov;
        int ow = 0, oh = 0;
        if (load_png(overlay, ov, ow, oh) && ow > 0 && oh > 0) {
            const float tr = overlayTint == TINT_NONE ? 1.0f : ((overlayTint >> 16) & 255) / 255.0f;
            const float tg = overlayTint == TINT_NONE ? 1.0f : ((overlayTint >> 8) & 255) / 255.0f;
            const float tb = overlayTint == TINT_NONE ? 1.0f : (overlayTint & 255) / 255.0f;
            for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
                const int ox = x * ow / w, oy = y * oh / h;
                const uint8_t* o = &ov[((size_t)oy * ow + ox) * 4];
                uint8_t* p = &rgba[((size_t)y * w + x) * 4];
                const float a = o[3] / 255.0f;
                p[0] = (uint8_t)std::lround(p[0] * (1 - a) + o[0] * tr * a);
                p[1] = (uint8_t)std::lround(p[1] * (1 - a) + o[1] * tg * a);
                p[2] = (uint8_t)std::lround(p[2] * (1 - a) + o[2] * tb * a);
            }
        }
    }
    te = make_texture(rgba, w, h, te.missing);
    if (te.index >= 0) m_bakedCache.emplace(hash_rgba(rgba, w), te);
    return m_texCache.emplace(key, te).first->second;
}

MaterialBuilder::TexEntry MaterialBuilder::make_texture(const std::vector<uint8_t>& rgba, int w, int h, bool missing) {
    TexEntry te;
    te.missing = missing;
    double sum[3] = { 0, 0, 0 };
    size_t n = 0;
    for (size_t i = 0; i < rgba.size(); i += 4) {
        if (rgba[i + 3] < 255) te.hasAlpha = true;
        if (rgba[i + 3] < 128) continue;
        sum[0] += srgb_to_linear(rgba[i + 0] / 255.0f);
        sum[1] += srgb_to_linear(rgba[i + 1] / 255.0f);
        sum[2] += srgb_to_linear(rgba[i + 2] / 255.0f);
        ++n;
    }
    if (n) for (int c = 0; c < 3; ++c) te.avg[c] = (float)(sum[c] / (double)n);

    DirectX::ScratchImage img;
    if (SUCCEEDED(img.Initialize2D(DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, (size_t)w, (size_t)h, 1, 1))) {
        std::memcpy(img.GetPixels(), rgba.data(), std::min(img.GetPixelsSize(), rgba.size()));
        DirectX::ScratchImage mips;
        TextureData td;
        if (SUCCEEDED(DirectX::GenerateMipMaps(*img.GetImage(0, 0, 0), DirectX::TEX_FILTER_BOX | DirectX::TEX_FILTER_SEPARATE_ALPHA, 0, mips))) {
            td.image = std::move(mips);
        } else {
            td.image = std::move(img);
        }
        td.width = w; td.height = h; td.channels = 4;
        td.original_width = w; td.original_height = h;
        te.index = (int)m_textures->size();
        m_textures->push_back(std::move(td));
        ++m_stats->textures;
    }
    return te;
}

const MaterialBuilder::TexEntry& MaterialBuilder::baked_texture(const std::vector<uint8_t>& rgba, int w, int h) {
    const uint64_t key = hash_rgba(rgba, w);
    const auto it = m_bakedCache.find(key);
    if (it != m_bakedCache.end()) return it->second;
    TexEntry te = make_texture(rgba, w, h, false);
    ++m_stats->bakedFaces;
    return m_bakedCache.emplace(key, te).first->second;
}

bool MaterialBuilder::quad_texels_opaque(const TexEntry& te, const RawQuad& q) const {
    if (te.index < 0) return false;
    const DirectX::Image* img = (*m_textures)[te.index].image.GetImage(0, 0, 0);
    if (!img) return false;
    float u0 = 1e9f, u1 = -1e9f, v0 = 1e9f, v1 = -1e9f;
    for (int k = 0; k < 4; ++k) {
        u0 = std::min(u0, q.uv[k][0]); u1 = std::max(u1, q.uv[k][0]);
        v0 = std::min(v0, q.uv[k][1]); v1 = std::max(v1, q.uv[k][1]);
    }
    const float sx = (float)img->width / 16.0f, sy = (float)img->height / 16.0f;
    const int x0 = std::max(0, (int)std::floor(u0 * sx)), x1 = std::min((int)img->width - 1, (int)std::ceil(u1 * sx) - 1);
    const int y0 = std::max(0, (int)std::floor(v0 * sy)), y1 = std::min((int)img->height - 1, (int)std::ceil(v1 * sy) - 1);
    if (x1 < x0 || y1 < y0) return false;
    for (int y = y0; y <= y1; ++y)
        for (int x = x0; x <= x1; ++x)
            if (img->pixels[(size_t)y * img->rowPitch + (size_t)x * 4 + 3] < 128) return false;
    return true;
}

uint16_t MaterialBuilder::opaque_lod_material(const TexEntry& te, bool metal, bool gloss, float emit, const std::string& name, uint16_t fallback, bool force) {
    if (te.index < 0 || (!force && lodOpaqueCoverage > 1.0f)) return fallback;
    const DirectX::Image* img = (*m_textures)[te.index].image.GetImage(0, 0, 0);
    if (!img) return fallback;
    const int w = (int)img->width, h = (int)img->height;
    std::vector<uint8_t> rgba((size_t)w * h * 4);
    for (int y = 0; y < h; ++y) std::memcpy(&rgba[(size_t)y * w * 4], img->pixels + (size_t)y * img->rowPitch, (size_t)w * 4);
    const double coverage = fill_holes(rgba, w, h);
    if (coverage <= 0.0 || (!force && coverage < lodOpaqueCoverage)) return fallback;
    const TexEntry& fte = baked_texture(rgba, w, h);
    const uint32_t rgb = ((uint32_t)std::lround(fte.avg[0] * 255) << 16) | ((uint32_t)std::lround(fte.avg[1] * 255) << 8) | (uint32_t)std::lround(fte.avg[2] * 255);
    return material_for(fte, Kind::Textured, metal, gloss, emit > 0 ? rgb : 0, emit, name + "#lod");
}

uint16_t MaterialBuilder::material_for(const TexEntry& te, Kind kind, bool metal, bool gloss,
                                       uint32_t emissiveRgb, float emissionScale, const std::string& debugName,
                                       bool allowFoliage) {
    // Keyed by profile: shared textures must not merge foliage/solid or clear/tinted.
    const std::string_view block = std::string_view(debugName).substr(0, debugName.find('#'));
    const bool foliage = allowFoliage && foliage_block(block);
    const Profile profile = foliage ? Profile::Foliage :
        (kind == Kind::Glass && (block == "glass" || block == "glass_pane") ? Profile::ClearGlass : Profile::Default);
    MaterialKey key{ te.index, (uint8_t)kind, (uint8_t)profile, (uint8_t)metal, (uint8_t)gloss, emissiveRgb, emissionScale };
    const auto it = m_matCache.find(key);
    if (it != m_matCache.end()) return it->second;

    Material m;
    m.Kd = DirectX::XMFLOAT4(1.0f, 1.0f, 1.0f, 1.0f);
    m.Ni = 1.0f;
    m.Pr_Pm_Ps_Pc = DirectX::XMFLOAT4(gloss ? 0.45f : 0.85f, 0.0f, 0.0f, 0.0f);
    m.albedoTexID = -1;
    m.normalTexID = -1;
    m.rmaTexID = -1;
    m.alphaThreshold = 1.0f;
    m.invertAlpha = false;
    bool alphaGeom = false;
    const float tintR = std::min(1.0f, te.avg[0] * 1.4f + 0.15f);
    const float tintG = std::min(1.0f, te.avg[1] * 1.4f + 0.15f);
    const float tintB = std::min(1.0f, te.avg[2] * 1.4f + 0.15f);
    switch (kind) {
    case Kind::Textured:
        m.albedoTexID = te.index >= 0 ? m_texIdBase + te.index : -1;
        break;
    case Kind::Cutout:
        m.albedoTexID = te.index >= 0 ? m_texIdBase + te.index : -1;
        m.alphaThreshold = 0.5f;
        alphaGeom = true;
        break;
    case Kind::Glass:
        m.thinGlass = 1;
        m.Ni = 1.5f;
        m.Pr_Pm_Ps_Pc = DirectX::XMFLOAT4(0.03f, 0.0f, 0.0f, 0.0f);
        m.Tf = DirectX::XMFLOAT3(tintR, tintG, tintB);
        m.Kd = DirectX::XMFLOAT4(tintR, tintG, tintB, 0.0f);
        if (profile == Profile::ClearGlass) {
            m.Tf = DirectX::XMFLOAT3(1.0f, 1.0f, 1.0f);
            m.Kd = DirectX::XMFLOAT4(1.0f, 1.0f, 1.0f, 0.0f);
        }
        alphaGeom = true;
        break;
    case Kind::Water:
        m.thinGlass = 0;
        m.Ni = 1.33f;
        m.Pr_Pm_Ps_Pc = DirectX::XMFLOAT4(0.0f, 0.0f, 0.0f, 0.0f);
        m.Kd = DirectX::XMFLOAT4(0.012f, 0.07f, 0.22f, 1.0f);
        break;
    case Kind::Flat:
        m.Kd = DirectX::XMFLOAT4(te.avg[0], te.avg[1], te.avg[2], 1.0f);
        break;
    }
    if (metal && kind != Kind::Glass && kind != Kind::Water) {
        m.Pr_Pm_Ps_Pc = DirectX::XMFLOAT4(gloss ? 0.2f : 0.35f, 1.0f, 0.0f, 0.0f);
    }
    if (profile == Profile::Foliage) {
        const float peak = std::max({ te.avg[0], te.avg[1], te.avg[2], 0.01f });
        m.sssEnable = 1;
        m.sssAlbedo = DirectX::XMFLOAT3(0.6f + 0.35f * te.avg[0] / peak,
                                      0.6f + 0.35f * te.avg[1] / peak,
                                      0.6f + 0.35f * te.avg[2] / peak);
        m.sssRadius = 1.0f; // mean free path: one block
        m.sssWeight = 0.65f;
        m.sssPhaseG = 0.35f;
        m.Ni = 1.4f;
    }
    if (emissionScale > 0.0f) {
        const float r = ((emissiveRgb >> 16) & 255) / 255.0f, g = ((emissiveRgb >> 8) & 255) / 255.0f, b = (emissiveRgb & 255) / 255.0f;
        m.Ke = DirectX::XMFLOAT3(r * emissionScale, g * emissionScale, b * emissionScale);
    }
    const uint16_t id = (uint16_t)m_materials->size();
    m_materials->push_back(m);
    m_materialNames->push_back(debugName);
    m_materialAlpha->resize(m_materials->size(), 0);
    (*m_materialAlpha)[id] = alphaGeom ? 1 : 0;
    m_materialEmission->resize(m_materials->size(), Vec3f{});
    (*m_materialEmission)[id] = Vec3f{ m.Ke.x, m.Ke.y, m.Ke.z };
    m_materialCutoutTexture->resize(m_materials->size(), -1);
    if (kind == Kind::Cutout && te.index >= 0) {
        (*m_materialCutoutTexture)[id] = te.index;
        if ((size_t)te.index >= m_textureAlpha->size()) m_textureAlpha->resize((size_t)te.index + 1);
        BlockRegistry::AlphaMask& am = (*m_textureAlpha)[(size_t)te.index];
        if (am.empty()) {
            const DirectX::Image* img = (*m_textures)[te.index].image.GetImage(0, 0, 0);
            if (img) {
                am.width = (int)img->width; am.height = (int)img->height;
                am.alpha.resize((size_t)am.width * am.height);
                for (int y = 0; y < am.height; ++y)
                    for (int x = 0; x < am.width; ++x)
                        am.alpha[(size_t)y * am.width + x] = img->pixels[(size_t)y * img->rowPitch + (size_t)x * 4 + 3];
            }
        }
    }
    m_matCache.emplace(key, id);
    ++m_stats->materials;
    return id;
}

bool MaterialBuilder::build(BlockRegistry& reg, IResourceProvider& res,
                            MaterialSoA& materials, std::vector<std::string>& materialNames,
                            std::vector<TextureData>& textures, int texIdBase,
                            MaterialBuildStats& stats, std::string* err) {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    stats = MaterialBuildStats{};
    m_res = &res; m_materials = &materials; m_materialNames = &materialNames;
    m_textures = &textures; m_texIdBase = texIdBase; m_stats = &stats;
    m_materialAlpha = &reg.materialAlpha;
    reg.materialAlpha.resize(materials.size(), 0);
    m_materialEmission = &reg.materialEmission;
    reg.materialEmission.resize(materials.size(), Vec3f{});
    m_materialCutoutTexture = &reg.materialCutoutTexture;
    reg.materialCutoutTexture.resize(materials.size(), -1);
    m_textureAlpha = &reg.textureAlpha;
    reg.quads.clear();
    if (!materialNames.empty() && materialNames.size() != materials.size()) materialNames.resize(materials.size());

    ModelResolver resolver(res);

    for (size_t i = 1; i < reg.count(); ++i) {
        const BlockId id = (BlockId)i;
        const BlockStateDesc& d = reg.desc(id);
        BlockInfo& bi = reg.info(id);
        bi = BlockInfo{};
        const std::string_view path = d.path();
        const float emit = emission_scale(d);
        const bool metal = metal_block(path);
        const bool gloss = glossy_block(path);
        bi.emissive = emit > 0.0f ? 1 : 0;

        if (path == "water" || path == "bubble_column" || path == "lava" || path == "flowing_water" || path == "flowing_lava") {
            const bool lava = contains(path, "lava");
            const TexEntry& te = texture(lava ? "minecraft:block/lava_still" : "minecraft:block/water_still",
                                         lava ? TINT_NONE : TINT_WATER, "", TINT_NONE);
            const uint32_t emRgb = ((uint32_t)std::lround(te.avg[0] * 255) << 16) | ((uint32_t)std::lround(te.avg[1] * 255) << 8) | (uint32_t)std::lround(te.avg[2] * 255);
            const uint16_t m    = material_for(te, lava ? Kind::Textured : Kind::Water, false, false, lava ? emRgb : 0, lava ? emit : 0.0f, std::string(path));
            const uint16_t flat = lava ? material_for(te, Kind::Flat, false, false, emRgb, emit, std::string(path) + "#flat") : m;
            bi.isCube = true; bi.fullOpaque = true; bi.cullSameId = true; bi.sig = Significance::Full;
            bi.lodFaceSolid = 0x3F;
            bi.tint = lava ? TintKind::None : TintKind::Water;
            bi.water = !lava && !bi.emissive;
            for (int f = 0; f < 6; ++f) { bi.faceMaterial[f] = m; bi.lodFaceMaterial[f] = m; bi.flatFaceMaterial[f] = flat; }
            ++stats.statesResolved; ++stats.cubes;
            continue;
        }

        ResolvedShape shape;
        std::string rerr;
        if (!resolver.resolve(d, shape, &rerr)) {
            ++stats.statesMissing;
            bi.sig = Significance::None;
            continue;
        }
        ++stats.statesResolved;
        if (shape.quads.empty()) {
            if (!shape.particle.empty() && cube_fallback_block(path)) {
                const TexEntry& te = texture(shape.particle, TINT_NONE, "", TINT_NONE);
                const uint16_t m = material_for(te, te.hasAlpha ? Kind::Cutout : Kind::Textured, metal, gloss, 0, 0.0f, std::string(path));
                const uint16_t flat = material_for(te, Kind::Flat, metal, gloss, 0, 0.0f, std::string(path) + "#flat");
                bi.isCube = true; bi.fullOpaque = false; bi.sig = Significance::Partial;
                bi.lodFaceSolid = 0x3F;
                for (int f = 0; f < 6; ++f) { bi.faceMaterial[f] = m; bi.lodFaceMaterial[f] = m; bi.flatFaceMaterial[f] = flat; }
                ++stats.cubes;
            } else {
                ++stats.invisible;
            }
            continue;
        }

        struct QuadMat { uint16_t mat, flat; int face; bool full; bool opaqueTex; const TexEntry* te = nullptr; };
        std::vector<QuadMat> qm(shape.quads.size());
        const bool glass = glass_block(path);
        for (size_t q = 0; q < shape.quads.size(); ++q) {
            const RawQuad& rq = shape.quads[q];
            const uint32_t tint = tint_for(d, rq.tintIndex);
            const uint32_t otint = rq.overlay.empty() ? TINT_NONE : tint_for(d, rq.overlayTint);
            const TexEntry& te = texture(rq.texture, tint, rq.overlay, otint);
            Kind kind = Kind::Textured;
            if (glass) kind = Kind::Glass;
            else if (te.hasAlpha && !quad_texels_opaque(te, rq)) kind = Kind::Cutout;
            const uint32_t emRgb = ((uint32_t)std::lround(te.avg[0] * 255) << 16) | ((uint32_t)std::lround(te.avg[1] * 255) << 8) | (uint32_t)std::lround(te.avg[2] * 255);
            // Pot soil and ceramic are not foliage.
            const bool allowFoliage = !path.starts_with("potted_") ||
                (!rq.texture.ends_with("/flower_pot") && !rq.texture.ends_with("/dirt"));
            qm[q].mat  = material_for(te, kind, metal, gloss, emit > 0 ? emRgb : 0, emit, std::string(path), allowFoliage);
            qm[q].flat = glass ? qm[q].mat : material_for(te, Kind::Flat, metal, gloss, emit > 0 ? emRgb : 0, emit, std::string(path) + "#flat", allowFoliage);
            qm[q].full = full_cube_face(rq, qm[q].face);
            qm[q].opaqueTex = kind == Kind::Textured;
            qm[q].te = &te;
        }
        if (tint_for(d, 0) == TINT_GRASS) bi.tint = TintKind::Grass;
        else if (tint_for(d, 0) == TINT_FOLIAGE) bi.tint = TintKind::Foliage;

        bool cube = shape.quads.size() == 6;
        if (cube) {
            bool seen[6] = {};
            for (const QuadMat& m : qm) { if (!m.full || seen[m.face]) { cube = false; break; } seen[m.face] = true; }
        }
        if (cube) {
            bool opaque = true;
            for (const QuadMat& m : qm) {
                bi.faceMaterial[m.face] = m.mat;
                bi.lodFaceMaterial[m.face] = (!m.opaqueTex && !glass) ? opaque_lod_material(*m.te, metal, gloss, emit, std::string(path), m.mat) : m.mat;
                bi.flatFaceMaterial[m.face] = m.flat;
                opaque = opaque && m.opaqueTex;
            }
            if (opaqueLeaves && !glass && contains(path, "leaves")) {
                for (const QuadMat& m : qm)
                    if (!m.opaqueTex) bi.faceMaterial[m.face] = bi.lodFaceMaterial[m.face] = opaque_lod_material(*m.te, metal, gloss, emit, std::string(path), m.mat, true);
                opaque = true;
            }
            bi.isCube = true;
            bi.lodFaceSolid = 0x3F;
            bi.fullOpaque = opaque;
            bi.cullSameId = !opaque;
            bi.sig = Significance::Full;
            ++stats.cubes;
            continue;
        }

        bi.isCube = false;
        bi.hasQuads = true;
        bi.sig = insignificant_block(path) ? Significance::None : Significance::Partial;
        bi.quadBegin = (uint32_t)reg.quads.size();
        int bakeSize = 16;
        for (size_t q = 0; q < shape.quads.size(); ++q) {
            const RawQuad& rq = shape.quads[q];
            BlockQuad bq;
            for (int k = 0; k < 4; ++k) {
                bq.pos[k] = rq.pos[k] * (1.0f / 16.0f);
                bq.uv[k][0] = rq.uv[k][0] / 16.0f;
                bq.uv[k][1] = rq.uv[k][1] / 16.0f;
            }
            bq.normal = rq.normal;
            bq.material = qm[q].mat;
            bq.cullFace = rq.cullFace >= 0 ? (uint8_t)rq.cullFace : FACE_NONE;
            bq.alphaGeom = reg.materialAlpha[qm[q].mat];
            reg.quads.push_back(bq);
            const TexEntry& te = texture(rq.texture, tint_for(d, rq.tintIndex), rq.overlay, rq.overlay.empty() ? TINT_NONE : tint_for(d, rq.overlayTint));
            if (te.index >= 0) bakeSize = std::max(bakeSize, std::min(64, (*m_textures)[te.index].width));
        }
        bi.quadCount = (uint32_t)(reg.quads.size() - bi.quadBegin);
        {
            BakedFace faces[6];
            auto lookup = [&](const RawQuad& rq) -> BakeTexture {
                const TexEntry& te = texture(rq.texture, tint_for(d, rq.tintIndex), rq.overlay, rq.overlay.empty() ? TINT_NONE : tint_for(d, rq.overlayTint));
                if (te.index < 0) return BakeTexture{};
                const DirectX::Image* img = (*m_textures)[te.index].image.GetImage(0, 0, 0);
                if (!img) return BakeTexture{};
                return BakeTexture{ img->pixels, (int)img->width, (int)img->height, img->rowPitch };
            };
            bake_block_faces(shape.quads, bakeSize, lookup, faces);
            for (int f = 0; f < 6; ++f) {
                if (faces[f].coverage < 0.04f) { bi.lodFaceMaterial[f] = NO_MATERIAL; bi.flatFaceMaterial[f] = NO_MATERIAL; continue; }
                if (faces[f].coverage >= 0.9f) bi.lodFaceSolid |= (uint8_t)(1u << f);
                if (!glass && faces[f].coverage >= lodOpaqueCoverage) fill_holes(faces[f].rgba, faces[f].size, faces[f].size);
                const TexEntry& bte = baked_texture(faces[f].rgba, faces[f].size, faces[f].size);
                const uint32_t bRgb = ((uint32_t)std::lround(bte.avg[0] * 255) << 16) | ((uint32_t)std::lround(bte.avg[1] * 255) << 8) | (uint32_t)std::lround(bte.avg[2] * 255);
                const Kind lodKind = glass ? Kind::Glass : (bte.hasAlpha ? Kind::Cutout : Kind::Textured);
                // Coarse potted faces mix plant, soil and pot colours.
                const bool allowFoliage = !path.starts_with("potted_");
                bi.lodFaceMaterial[f]  = material_for(bte, lodKind, metal, gloss, emit > 0 ? bRgb : 0, emit, std::string(path) + "#lod", allowFoliage);
                bi.flatFaceMaterial[f] = glass ? bi.lodFaceMaterial[f] : material_for(bte, Kind::Flat, metal, gloss, emit > 0 ? bRgb : 0, emit, std::string(path) + "#flat", allowFoliage);
            }
        }
        ++stats.modelBlocks;
    }

    stats.seconds = std::chrono::duration<double>(clock::now() - t0).count();
    if (err) err->clear();
    std::printf("[mc] materials: %u states resolved (%u cubes, %u model blocks, %u invisible, %u without blockstate), %u textures (%u missing, %u baked faces), %u materials, %.2f s\n",
                stats.statesResolved, stats.cubes, stats.modelBlocks, stats.invisible, stats.statesMissing,
                stats.textures, stats.texturesMissing, stats.bakedFaces, stats.materials, stats.seconds);
    return true;
}

}
