#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "mc_types.h"
#include "mc_zip.h"

namespace mc {

struct BlockStateDesc;

struct RawQuad {
    Vec3f    pos[4];
    float    uv[4][2];
    Vec3f    normal;
    int      cullFace  = -1;
    int      tintIndex = -1;
    bool     shade     = true;
    std::string texture;
    std::string overlay;
    int      overlayTint = -1;
};

struct ResolvedShape {
    std::vector<RawQuad> quads;
    std::string particle;
    bool found = false;
};

class ModelResolver {
public:
    explicit ModelResolver(IResourceProvider& res) : m_res(res) {}

    bool resolve(const BlockStateDesc& state, ResolvedShape& out, std::string* err = nullptr);

    static std::string texture_resource_path(const std::string& textureName);
    static std::string normalize_name(const std::string& name);

private:
    struct FaceDef {
        bool  present = false;
        bool  hasUv   = false;
        float uv[4]   = { 0, 0, 16, 16 };
        std::string texture;
        int   cullFace = -1;
        int   rotation = 0;
        int   tintIndex = -1;
    };
    struct ElementDef {
        float from[3] = { 0, 0, 0 };
        float to[3]   = { 16, 16, 16 };
        bool  hasRotation = false;
        float rotOrigin[3] = { 8, 8, 8 };
        int   rotAxis = 1;
        float rotAngle = 0.0f;
        bool  rescale = false;
        bool  shade = true;
        FaceDef faces[6];
    };
    struct Model {
        std::string parent;
        std::unordered_map<std::string, std::string> textures;
        std::vector<ElementDef> elements;
        bool hasElements = false;
        bool loaded = false;
    };
    struct Variant {
        std::string model;
        int  x = 0, y = 0;
        bool uvlock = false;
    };

    const Model* load_model(const std::string& name);
    bool parse_model(const std::vector<uint8_t>& json, Model& out, std::string* err);
    bool resolve_textures(const Model* m, std::unordered_map<std::string, std::string>& outTex,
                          const std::vector<ElementDef>*& outElements, std::string& particle) const;
    void emit_model(const Variant& v, ResolvedShape& out);
    static void merge_overlays(ResolvedShape& shape);

    IResourceProvider& m_res;
    std::unordered_map<std::string, std::unique_ptr<Model>> m_models;
};

void rotate_variant_point(float p[3], int xDeg, int yDeg);
int rotate_variant_face(int face, int xDeg, int yDeg);

}
