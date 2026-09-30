#include "mc_models.h"
#include "block_registry.h"
#include <algorithm>
#include <cmath>
#include <cstring>

#include "../../lib/tinygltf_json.h"

namespace mc {

using J = tinygltf_json;

namespace {

const J* jfind(const J& o, const char* key) {
    if (!o.is_object()) return nullptr;
    const tinygltf_json_member* m = o.find_member_(key);
    return m ? &m->val : nullptr;
}
std::string jstr(const J* v, const char* def = "") {
    if (!v || !v->is_string() || !v->str_) return def;
    return std::string(v->str_, v->str_len_);
}
double jnum(const J* v, double def = 0.0) {
    if (!v) return def;
    if (v->type_ == CJ_INT)  return (double)v->i_;
    if (v->type_ == CJ_REAL) return v->d_;
    if (v->type_ == CJ_BOOL) return v->b_ ? 1.0 : 0.0;
    return def;
}
bool jbool(const J* v, bool def = false) {
    if (!v) return def;
    if (v->type_ == CJ_BOOL) return v->b_ != 0;
    if (v->type_ == CJ_INT)  return v->i_ != 0;
    if (v->is_string())      return jstr(v) == "true";
    return def;
}
bool jarr3(const J* v, float out[3]) {
    if (!v || !v->is_array() || v->arr_size_ < 3) return false;
    for (int i = 0; i < 3; ++i) out[i] = (float)jnum(&v->arr_data_[i]);
    return true;
}

constexpr float PI_F = 3.14159265358979f;

void rotate_about(float p[3], const float origin[3], int axis, float deg) {
    const float a = deg * PI_F / 180.0f;
    const float c = std::cos(a), s = std::sin(a);
    float v[3] = { p[0] - origin[0], p[1] - origin[1], p[2] - origin[2] };
    float r[3] = { v[0], v[1], v[2] };
    switch (axis) {
    case 0: r[1] = v[1] * c - v[2] * s; r[2] = v[1] * s + v[2] * c; break;
    case 1: r[0] = v[0] * c - v[2] * s; r[2] = v[0] * s + v[2] * c; break;
    default: r[0] = v[0] * c - v[1] * s; r[1] = v[0] * s + v[1] * c; break;
    }
    p[0] = r[0] + origin[0]; p[1] = r[1] + origin[1]; p[2] = r[2] + origin[2];
}

// Vanilla rotation convention.
void rotate_element(float p[3], const float origin[3], int axis, float deg) {
    rotate_about(p, origin, axis, axis == 1 ? deg : -deg);
}

int face_of_normal(const Vec3f& n) {
    const float ax = std::fabs(n.x), ay = std::fabs(n.y), az = std::fabs(n.z);
    if (ay >= ax && ay >= az) return n.y > 0 ? FACE_UP : FACE_DOWN;
    if (ax >= az)             return n.x > 0 ? FACE_EAST : FACE_WEST;
    return n.z > 0 ? FACE_SOUTH : FACE_NORTH;
}

void default_uv(int face, const float from[3], const float to[3], float uv[4]) {
    switch (face) {
    case FACE_DOWN:  uv[0] = from[0]; uv[1] = 16 - to[2]; uv[2] = to[0]; uv[3] = 16 - from[2]; break;
    case FACE_UP:    uv[0] = from[0]; uv[1] = from[2];    uv[2] = to[0]; uv[3] = to[2];        break;
    case FACE_NORTH: uv[0] = 16 - to[0]; uv[1] = 16 - to[1]; uv[2] = 16 - from[0]; uv[3] = 16 - from[1]; break;
    case FACE_SOUTH: uv[0] = from[0]; uv[1] = 16 - to[1]; uv[2] = to[0]; uv[3] = 16 - from[1]; break;
    case FACE_WEST:  uv[0] = from[2]; uv[1] = 16 - to[1]; uv[2] = to[2]; uv[3] = 16 - from[1]; break;
    default:         uv[0] = 16 - to[2]; uv[1] = 16 - to[1]; uv[2] = 16 - from[2]; uv[3] = 16 - from[1]; break;
    }
}

void project_uv(int face, const float p[3], float& u, float& v) {
    switch (face) {
    case FACE_DOWN:  u = p[0];      v = 16 - p[2]; break;
    case FACE_UP:    u = p[0];      v = p[2];      break;
    case FACE_NORTH: u = 16 - p[0]; v = 16 - p[1]; break;
    case FACE_SOUTH: u = p[0];      v = 16 - p[1]; break;
    case FACE_WEST:  u = p[2];      v = 16 - p[1]; break;
    default:         u = 16 - p[2]; v = 16 - p[1]; break;
    }
}

}

void rotate_variant_point(float p[3], int xDeg, int yDeg) {
    static const float c[3] = { 8, 8, 8 };
    if (xDeg) rotate_about(p, c, 0, (float)xDeg);
    if (yDeg) rotate_about(p, c, 1, (float)yDeg);
}
int rotate_variant_face(int face, int xDeg, int yDeg) {
    if (face < 0 || face >= 6) return face;
    float p[3] = { 8 + 8.0f * FACE_DIR[face][0], 8 + 8.0f * FACE_DIR[face][1], 8 + 8.0f * FACE_DIR[face][2] };
    rotate_variant_point(p, xDeg, yDeg);
    return face_of_normal(Vec3f{ p[0] - 8, p[1] - 8, p[2] - 8 });
}

std::string ModelResolver::normalize_name(const std::string& name) {
    if (name.find(':') == std::string::npos) return "minecraft:" + name;
    return name;
}
std::string ModelResolver::texture_resource_path(const std::string& textureName) {
    const std::string n = normalize_name(textureName);
    const size_t c = n.find(':');
    return "assets/" + n.substr(0, c) + "/textures/" + n.substr(c + 1) + ".png";
}
static std::string model_resource_path(const std::string& modelName) {
    const std::string n = ModelResolver::normalize_name(modelName);
    const size_t c = n.find(':');
    return "assets/" + n.substr(0, c) + "/models/" + n.substr(c + 1) + ".json";
}
static std::string blockstate_resource_path(const std::string& blockName) {
    const std::string n = ModelResolver::normalize_name(blockName);
    const size_t c = n.find(':');
    return "assets/" + n.substr(0, c) + "/blockstates/" + n.substr(c + 1) + ".json";
}

bool ModelResolver::parse_model(const std::vector<uint8_t>& json, Model& out, std::string* err) {
    const J root = J::parse((const char*)json.data(), (const char*)json.data() + json.size());
    if (!root.is_object()) { if (err) *err = "model json is not an object"; return false; }
    out.parent = jstr(jfind(root, "parent"));
    if (!out.parent.empty()) out.parent = normalize_name(out.parent);
    if (const J* tex = jfind(root, "textures"); tex && tex->is_object()) {
        for (size_t i = 0; i < tex->obj_size_; ++i) {
            const tinygltf_json_member& m = tex->obj_data_[i];
            out.textures[std::string(m.key, m.key_len)] = jstr(&m.val);
        }
    }
    if (const J* els = jfind(root, "elements"); els && els->is_array()) {
        out.hasElements = true;
        for (size_t i = 0; i < els->arr_size_; ++i) {
            const J& e = els->arr_data_[i];
            if (!e.is_object()) continue;
            ElementDef d;
            jarr3(jfind(e, "from"), d.from);
            jarr3(jfind(e, "to"), d.to);
            d.shade = jbool(jfind(e, "shade"), true);
            if (const J* rot = jfind(e, "rotation"); rot && rot->is_object()) {
                d.hasRotation = true;
                jarr3(jfind(*rot, "origin"), d.rotOrigin);
                const std::string ax = jstr(jfind(*rot, "axis"), "y");
                d.rotAxis = (ax == "x") ? 0 : (ax == "z" ? 2 : 1);
                d.rotAngle = (float)jnum(jfind(*rot, "angle"), 0.0);
                d.rescale = jbool(jfind(*rot, "rescale"), false);
            }
            if (const J* faces = jfind(e, "faces"); faces && faces->is_object()) {
                for (size_t k = 0; k < faces->obj_size_; ++k) {
                    const tinygltf_json_member& fm = faces->obj_data_[k];
                    const int f = face_from_name(std::string(fm.key, fm.key_len).c_str());
                    if (f < 0 || !fm.val.is_object()) continue;
                    FaceDef& fd = d.faces[f];
                    fd.present = true;
                    if (const J* uv = jfind(fm.val, "uv"); uv && uv->is_array() && uv->arr_size_ >= 4) {
                        fd.hasUv = true;
                        for (int q = 0; q < 4; ++q) fd.uv[q] = (float)jnum(&uv->arr_data_[q]);
                    }
                    fd.texture = jstr(jfind(fm.val, "texture"));
                    const std::string cull = jstr(jfind(fm.val, "cullface"));
                    fd.cullFace = cull.empty() ? -1 : face_from_name(cull.c_str());
                    fd.rotation = (int)jnum(jfind(fm.val, "rotation"), 0.0);
                    fd.tintIndex = (int)jnum(jfind(fm.val, "tintindex"), -1.0);
                }
            }
            out.elements.push_back(d);
        }
    }
    out.loaded = true;
    return true;
}

const ModelResolver::Model* ModelResolver::load_model(const std::string& nameIn) {
    const std::string name = normalize_name(nameIn);
    const auto it = m_models.find(name);
    if (it != m_models.end()) return it->second->loaded ? it->second.get() : nullptr;
    auto m = std::make_unique<Model>();
    std::vector<uint8_t> bytes;
    Model* raw = m.get();
    m_models.emplace(name, std::move(m));
    if (!m_res.read(model_resource_path(name), bytes)) return nullptr;
    std::string err;
    if (!parse_model(bytes, *raw, &err)) return nullptr;
    return raw;
}

bool ModelResolver::resolve_textures(const Model* m, std::unordered_map<std::string, std::string>& outTex,
                                     const std::vector<ElementDef>*& outElements, std::string& particle) const {
    outElements = nullptr;
    const Model* cur = m;
    int hops = 0;
    while (cur && hops++ < 32) {
        for (const auto& kv : cur->textures)
            if (!outTex.count(kv.first)) outTex[kv.first] = kv.second;
        if (!outElements && cur->hasElements) outElements = &cur->elements;
        if (cur->parent.empty()) break;
        const auto it = m_models.find(cur->parent);
        cur = (it != m_models.end() && it->second->loaded) ? it->second.get() : nullptr;
    }
    const auto p = outTex.find("particle");
    if (p != outTex.end()) {
        std::string t = p->second;
        int guard = 0;
        while (!t.empty() && t[0] == '#' && guard++ < 16) {
            const auto r = outTex.find(t.substr(1));
            if (r == outTex.end()) { t.clear(); break; }
            t = r->second;
        }
        if (!t.empty() && t[0] != '#') particle = normalize_name(t);
    }
    return true;
}

void ModelResolver::emit_model(const Variant& v, ResolvedShape& out) {
    const Model* m = load_model(v.model);
    if (!m) return;
    {
        const Model* cur = m;
        int hops = 0;
        while (cur && !cur->parent.empty() && hops++ < 32) cur = load_model(cur->parent);
    }
    std::unordered_map<std::string, std::string> tex;
    const std::vector<ElementDef>* elements = nullptr;
    std::string particle;
    resolve_textures(m, tex, elements, particle);
    if (out.particle.empty()) out.particle = particle;
    if (!elements) return;

    auto resolveTex = [&](const std::string& ref) -> std::string {
        std::string t = ref;
        int guard = 0;
        while (!t.empty() && t[0] == '#' && guard++ < 16) {
            const auto r = tex.find(t.substr(1));
            if (r == tex.end()) return std::string();
            t = r->second;
        }
        if (t.empty() || t[0] == '#') return std::string();
        return normalize_name(t);
    };

    for (const ElementDef& e : *elements) {
        for (int f = 0; f < 6; ++f) {
            const FaceDef& fd = e.faces[f];
            if (!fd.present) continue;
            const std::string texName = resolveTex(fd.texture);
            if (texName.empty()) continue;

            float uv[4];
            if (fd.hasUv) std::memcpy(uv, fd.uv, sizeof(uv)); else default_uv(f, e.from, e.to, uv);
            float corner[4][3];
            {
                const int axis = (f == FACE_DOWN || f == FACE_UP) ? 1 : ((f == FACE_WEST || f == FACE_EAST) ? 0 : 2);
                const float plane = (FACE_DIR[f][axis] > 0) ? e.to[axis] : e.from[axis];
                int ua, va; bool uFlip, vFlip;
                switch (f) {
                case FACE_DOWN:  ua = 0; va = 2; uFlip = false; vFlip = true;  break;
                case FACE_UP:    ua = 0; va = 2; uFlip = false; vFlip = false; break;
                case FACE_NORTH: ua = 0; va = 1; uFlip = true;  vFlip = true;  break;
                case FACE_SOUTH: ua = 0; va = 1; uFlip = false; vFlip = true;  break;
                case FACE_WEST:  ua = 2; va = 1; uFlip = false; vFlip = true;  break;
                default:         ua = 2; va = 1; uFlip = true;  vFlip = true;  break;
                }
                const float uLo = uFlip ? e.to[ua] : e.from[ua];
                const float uHi = uFlip ? e.from[ua] : e.to[ua];
                const float vLo = vFlip ? e.to[va] : e.from[va];
                const float vHi = vFlip ? e.from[va] : e.to[va];
                const float us[4] = { uLo, uHi, uHi, uLo };
                const float vs[4] = { vLo, vLo, vHi, vHi };
                for (int k = 0; k < 4; ++k) {
                    corner[k][axis] = plane;
                    corner[k][ua] = us[k];
                    corner[k][va] = vs[k];
                }
            }
            float cuv[4][2] = { { uv[0], uv[1] }, { uv[2], uv[1] }, { uv[2], uv[3] }, { uv[0], uv[3] } };
            const int rot = ((fd.rotation % 360) + 360) % 360 / 90;

            RawQuad q;
            for (int k = 0; k < 4; ++k) {
                float p[3] = { corner[k][0], corner[k][1], corner[k][2] };
                if (e.hasRotation) {
                    rotate_element(p, e.rotOrigin, e.rotAxis, e.rotAngle);
                    if (e.rescale) {
                        const float s = 1.0f / std::cos(e.rotAngle * PI_F / 180.0f);
                        for (int a = 0; a < 3; ++a)
                            if (a != e.rotAxis) p[a] = (p[a] - e.rotOrigin[a]) * s + e.rotOrigin[a];
                    }
                }
                rotate_variant_point(p, v.x, v.y);
                q.pos[k] = Vec3f{ p[0], p[1], p[2] };
                const int src = (k + rot) & 3;
                q.uv[k][0] = cuv[src][0];
                q.uv[k][1] = cuv[src][1];
            }
            {
                float n[3] = { 8 + 8.0f * FACE_DIR[f][0], 8 + 8.0f * FACE_DIR[f][1], 8 + 8.0f * FACE_DIR[f][2] };
                float o[3] = { 8, 8, 8 };
                if (e.hasRotation) { rotate_element(n, o, e.rotAxis, e.rotAngle); }
                rotate_variant_point(n, v.x, v.y);
                rotate_variant_point(o, v.x, v.y);
                q.normal = normalize(Vec3f{ n[0] - o[0], n[1] - o[1], n[2] - o[2] });
            }
            {
                const Vec3f g = cross(q.pos[1] - q.pos[0], q.pos[2] - q.pos[0]);
                if (dot(g, q.normal) < 0.0f) {
                    std::swap(q.pos[1], q.pos[3]);
                    std::swap(q.uv[1][0], q.uv[3][0]);
                    std::swap(q.uv[1][1], q.uv[3][1]);
                }
            }
            if (v.uvlock && (v.x || v.y)) {
                const int wf = face_of_normal(q.normal);
                for (int k = 0; k < 4; ++k) {
                    const float p[3] = { q.pos[k].x, q.pos[k].y, q.pos[k].z };
                    project_uv(wf, p, q.uv[k][0], q.uv[k][1]);
                }
            }
            q.cullFace  = (fd.cullFace >= 0) ? rotate_variant_face(fd.cullFace, v.x, v.y) : -1;
            q.tintIndex = fd.tintIndex;
            q.shade     = e.shade;
            q.texture   = texName;
            out.quads.push_back(std::move(q));
        }
    }
}

void ModelResolver::merge_overlays(ResolvedShape& shape) {
    auto same = [](const RawQuad& a, const RawQuad& b) {
        if (dot(a.normal, b.normal) < 0.999f) return false;
        for (int i = 0; i < 4; ++i) {
            bool hit = false;
            for (int j = 0; j < 4; ++j) {
                const Vec3f d = a.pos[i] - b.pos[j];
                if (std::fabs(d.x) < 1e-3f && std::fabs(d.y) < 1e-3f && std::fabs(d.z) < 1e-3f) { hit = true; break; }
            }
            if (!hit) return false;
        }
        return true;
    };
    std::vector<RawQuad> out;
    out.reserve(shape.quads.size());
    for (RawQuad& q : shape.quads) {
        bool merged = false;
        for (RawQuad& o : out) {
            if (!o.overlay.empty() || !same(o, q)) continue;
            o.overlay = q.texture;
            o.overlayTint = q.tintIndex;
            merged = true;
            break;
        }
        if (!merged) out.push_back(std::move(q));
    }
    shape.quads = std::move(out);
}

namespace {

bool prop_matches(const BlockStateDesc& s, const std::string& key, const std::string& wanted) {
    const std::string_view have = s.prop(key);
    size_t start = 0;
    while (start <= wanted.size()) {
        size_t bar = wanted.find('|', start);
        if (bar == std::string::npos) bar = wanted.size();
        if (have == std::string_view(wanted).substr(start, bar - start)) return true;
        start = bar + 1;
    }
    return false;
}

bool variant_key_matches(const BlockStateDesc& s, const std::string& key) {
    if (key.empty()) return true;
    size_t start = 0;
    while (start < key.size()) {
        size_t comma = key.find(',', start);
        if (comma == std::string::npos) comma = key.size();
        const std::string pair = key.substr(start, comma - start);
        const size_t eq = pair.find('=');
        if (eq == std::string::npos) return false;
        if (!prop_matches(s, pair.substr(0, eq), pair.substr(eq + 1))) return false;
        start = comma + 1;
    }
    return true;
}

bool when_matches(const BlockStateDesc& s, const J& when) {
    if (!when.is_object()) return true;
    if (const J* orList = jfind(when, "OR"); orList && orList->is_array()) {
        for (size_t i = 0; i < orList->arr_size_; ++i)
            if (when_matches(s, orList->arr_data_[i])) return true;
        return false;
    }
    if (const J* andList = jfind(when, "AND"); andList && andList->is_array()) {
        for (size_t i = 0; i < andList->arr_size_; ++i)
            if (!when_matches(s, andList->arr_data_[i])) return false;
        return true;
    }
    for (size_t i = 0; i < when.obj_size_; ++i) {
        const tinygltf_json_member& m = when.obj_data_[i];
        std::string wanted;
        if (m.val.is_string()) wanted = jstr(&m.val);
        else if (m.val.type_ == CJ_BOOL) wanted = m.val.b_ ? "true" : "false";
        else wanted = std::to_string((long long)jnum(&m.val));
        if (!prop_matches(s, std::string(m.key, m.key_len), wanted)) return false;
    }
    return true;
}

}

bool ModelResolver::resolve(const BlockStateDesc& state, ResolvedShape& out, std::string* err) {
    out = ResolvedShape{};
    std::vector<uint8_t> bytes;
    if (!m_res.read(blockstate_resource_path(state.name), bytes)) {
        if (err) *err = "no blockstate file for " + state.name;
        return false;
    }
    out.found = true;
    const J root = J::parse((const char*)bytes.data(), (const char*)bytes.data() + bytes.size());
    if (!root.is_object()) { if (err) *err = "blockstate json is not an object"; return false; }

    auto readVariant = [&](const J& v) -> Variant {
        Variant out{};
        const J* pick = &v;
        if (v.is_array()) {
            if (v.arr_size_ == 0) return out;
            pick = &v.arr_data_[0];
        }
        if (!pick->is_object()) return out;
        out.model  = normalize_name(jstr(jfind(*pick, "model")));
        out.x      = (int)jnum(jfind(*pick, "x"), 0.0);
        out.y      = (int)jnum(jfind(*pick, "y"), 0.0);
        out.uvlock = jbool(jfind(*pick, "uvlock"), false);
        return out;
    };

    if (const J* variants = jfind(root, "variants"); variants && variants->is_object()) {
        const J* chosen = nullptr;
        for (size_t i = 0; i < variants->obj_size_; ++i) {
            const tinygltf_json_member& m = variants->obj_data_[i];
            const std::string key(m.key, m.key_len);
            if (key.empty()) continue;
            if (variant_key_matches(state, key)) { chosen = &m.val; break; }
        }
        if (!chosen) chosen = jfind(*variants, "");
        if (!chosen) {
            if (variants->obj_size_ > 0) chosen = &variants->obj_data_[0].val;
        }
        if (chosen) {
            const Variant v = readVariant(*chosen);
            if (!v.model.empty()) emit_model(v, out);
        }
    } else if (const J* parts = jfind(root, "multipart"); parts && parts->is_array()) {
        for (size_t i = 0; i < parts->arr_size_; ++i) {
            const J& part = parts->arr_data_[i];
            if (!part.is_object()) continue;
            const J* when = jfind(part, "when");
            if (when && !when_matches(state, *when)) continue;
            const J* apply = jfind(part, "apply");
            if (!apply) continue;
            const Variant v = readVariant(*apply);
            if (!v.model.empty()) emit_model(v, out);
        }
    } else {
        if (err) *err = "blockstate has neither variants nor multipart";
        return false;
    }
    merge_overlays(out);
    return true;
}

}
