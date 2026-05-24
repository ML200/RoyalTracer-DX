#include "render/preview.h"

#include "render/sphere_mesh.h"
#include "render/shader.h"
#include "render/camera.h"
#include "core/cubed_sphere.h"
#include "core/field.h"
#include "core/log.h"
#include "core/param_registry.h"
#include "core/pass.h"
#include "core/pipeline.h"
#include "core/planet_state.h"
#include "gpu/preview_field.h"

#include <GL/glew.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace pb {

static const char* kVertSrc = R"GLSL(
#version 450 core
layout(location = 0) in vec3 a_position;
uniform mat4 u_view_proj;
uniform float u_displace_scale;

uniform sampler2D u_face0;
uniform sampler2D u_face1;
uniform sampler2D u_face2;
uniform sampler2D u_face3;
uniform sampler2D u_face4;
uniform sampler2D u_face5;

out vec3 v_pos;
out vec3 v_world;

void sphere_to_face_uv(vec3 p, out int face, out vec2 uv) {
    vec3 ap = abs(p);
    float ut, vt;
    if (ap.x >= ap.y && ap.x >= ap.z) {
        if (p.x > 0.0) { face = 0; ut = -p.z / ap.x; vt = -p.y / ap.x; }
        else           { face = 1; ut =  p.z / ap.x; vt = -p.y / ap.x; }
    } else if (ap.y >= ap.x && ap.y >= ap.z) {
        if (p.y > 0.0) { face = 2; ut =  p.x / ap.y; vt =  p.z / ap.y; }
        else           { face = 3; ut =  p.x / ap.y; vt = -p.z / ap.y; }
    } else {
        if (p.z > 0.0) { face = 4; ut =  p.x / ap.z; vt = -p.y / ap.z; }
        else           { face = 5; ut = -p.x / ap.z; vt = -p.y / ap.z; }
    }
    const float PI = 3.14159265358979;
    float u = atan(ut) * 4.0 / PI;
    float v = atan(vt) * 4.0 / PI;
    uv = vec2(u, v) * 0.5 + 0.5;
}

float sample_lod0(int face, vec2 uv) {
    if (face == 0) return textureLod(u_face0, uv, 0.0).r;
    if (face == 1) return textureLod(u_face1, uv, 0.0).r;
    if (face == 2) return textureLod(u_face2, uv, 0.0).r;
    if (face == 3) return textureLod(u_face3, uv, 0.0).r;
    if (face == 4) return textureLod(u_face4, uv, 0.0).r;
    return textureLod(u_face5, uv, 0.0).r;
}

void main() {
    vec3 dir = normalize(a_position);
    float val = 0.0;
    if (u_displace_scale != 0.0) {
        int face;
        vec2 uv;
        sphere_to_face_uv(dir, face, uv);
        val = sample_lod0(face, uv);
    }
    vec3 displaced = dir * (1.0 + val * u_displace_scale);

    v_pos   = dir;
    v_world = displaced;
    gl_Position = u_view_proj * vec4(displaced, 1.0);
}
)GLSL";

static const char* kFragSrc = R"GLSL(
#version 450 core
in vec3 v_pos;
in vec3 v_world;
out vec4 frag_color;

uniform sampler2D u_face0;
uniform sampler2D u_face1;
uniform sampler2D u_face2;
uniform sampler2D u_face3;
uniform sampler2D u_face4;
uniform sampler2D u_face5;

uniform float u_value_min;
uniform float u_value_max;
uniform int   u_show_seams;
uniform int   u_lit;
uniform float u_displace_scale;
uniform float u_face_resolution;

void sphere_to_face_uv(vec3 p, out int face, out vec2 uv) {
    vec3 ap = abs(p);
    float ut, vt;
    if (ap.x >= ap.y && ap.x >= ap.z) {
        if (p.x > 0.0) { face = 0; ut = -p.z / ap.x; vt = -p.y / ap.x; }
        else           { face = 1; ut =  p.z / ap.x; vt = -p.y / ap.x; }
    } else if (ap.y >= ap.x && ap.y >= ap.z) {
        if (p.y > 0.0) { face = 2; ut =  p.x / ap.y; vt =  p.z / ap.y; }
        else           { face = 3; ut =  p.x / ap.y; vt = -p.z / ap.y; }
    } else {
        if (p.z > 0.0) { face = 4; ut =  p.x / ap.z; vt = -p.y / ap.z; }
        else           { face = 5; ut = -p.x / ap.z; vt = -p.y / ap.z; }
    }
    const float PI = 3.14159265358979;
    float u = atan(ut) * 4.0 / PI;
    float v = atan(vt) * 4.0 / PI;
    uv = vec2(u, v) * 0.5 + 0.5;
}

float sample_face(int face, vec2 uv) {
    if (face == 0) return texture(u_face0, uv).r;
    if (face == 1) return texture(u_face1, uv).r;
    if (face == 2) return texture(u_face2, uv).r;
    if (face == 3) return texture(u_face3, uv).r;
    if (face == 4) return texture(u_face4, uv).r;
    return texture(u_face5, uv).r;
}

//Inverse of sphere_to_face_uv: face + uv (in [0,1]) -> sphere direction.
//Mirror of host face_uv_to_sphere in core/cubed_sphere.h.
vec3 face_uv_to_sphere(int face, vec2 uv) {
    vec2 uv2 = uv * 2.0 - 1.0;
    const float PI = 3.14159265358979;
    float ut = tan(uv2.x * PI * 0.25);
    float vt = tan(uv2.y * PI * 0.25);
    vec3 p;
    if      (face == 0) p = vec3( 1.0, -vt, -ut);
    else if (face == 1) p = vec3(-1.0, -vt,  ut);
    else if (face == 2) p = vec3( ut,  1.0,  vt);
    else if (face == 3) p = vec3( ut, -1.0, -vt);
    else if (face == 4) p = vec3( ut, -vt,  1.0);
    else                p = vec3(-ut, -vt, -1.0);
    return normalize(p);
}

//Position on the displaced surface at a given face+uv. Used for per-pixel
//normal computation - sampling at offset uvs and crossing the tangents
//gives a normal that reflects TEXTURE-resolution detail (~20 km/texel at
//N=1024 on Earth), not just the icosphere mesh resolution (~55 km/vert at
//subdivision 6). That's how the fine_freq / peak / lowland_rough layers
//become visible in the relief shading.
vec3 displaced_at(int face, vec2 uv, float scale) {
    vec3  dir = face_uv_to_sphere(face, uv);
    float val = sample_face(face, uv);
    return dir * (1.0 + val * scale);
}

vec3 viridis(float t) {
    t = clamp(t, 0.0, 1.0);
    const vec3 c0 = vec3(0.2777273472,  0.005499649973, 0.3340998125);
    const vec3 c1 = vec3(0.1050930431,  1.404613554,    1.384523068);
    const vec3 c2 = vec3(-0.3308618287, 0.214847635,    0.09410307832);
    const vec3 c3 = vec3(-4.634084144, -5.799146325,  -19.33241851);
    const vec3 c4 = vec3( 6.228718209, 14.17999401,    56.69052851);
    const vec3 c5 = vec3( 4.776341660,-13.74514243,   -65.35303927);
    const vec3 c6 = vec3(-5.435450183,  4.645852464,   26.31242318);
    return c0 + t*(c1 + t*(c2 + t*(c3 + t*(c4 + t*(c5 + t*c6)))));
}

vec3 face_tint(int face) {
    if (face == 0) return vec3(1.0, 0.4, 0.4);
    if (face == 1) return vec3(0.4, 1.0, 0.4);
    if (face == 2) return vec3(0.4, 0.4, 1.0);
    if (face == 3) return vec3(1.0, 1.0, 0.4);
    if (face == 4) return vec3(1.0, 0.4, 1.0);
    return vec3(0.4, 1.0, 1.0);
}

void main() {
    vec3 dir = normalize(v_pos);
    int face;
    vec2 uv;
    sphere_to_face_uv(dir, face, uv);

    float val = sample_face(face, uv);
    float t   = (val - u_value_min) / max(u_value_max - u_value_min, 1e-6);
    vec3 col  = viridis(t);

    if (u_lit == 1) {
        //Per-pixel normal via central difference on texture-rate elevation
        //samples. dFdx/dFdy on v_world would give per-triangle face normals
        //which only resolves features above the mesh vertex spacing - the
        //fine_freq / peak / lowland_rough layers get washed out. Sampling
        //the elevation field directly at 4 uv-offsets per pixel and
        //crossing the resulting tangent vectors gives a normal at TEXTURE
        //resolution (~20 km / texel at N=1024 on Earth).
        float eps = 1.5 / max(u_face_resolution, 1.0);
        vec3 px = displaced_at(face, uv + vec2(eps, 0.0), u_displace_scale);
        vec3 mx = displaced_at(face, uv - vec2(eps, 0.0), u_displace_scale);
        vec3 py = displaced_at(face, uv + vec2(0.0, eps), u_displace_scale);
        vec3 my = displaced_at(face, uv - vec2(0.0, eps), u_displace_scale);
        vec3 tu = px - mx;
        vec3 tv = py - my;
        vec3 n  = normalize(cross(tu, tv));
        if (dot(n, dir) < 0.0) n = -n;

        vec3  L       = normalize(vec3(0.55, 0.65, 0.40));
        float lambert = max(dot(n, L), 0.0);
        float ambient = 0.30;
        col           = col * (ambient + (1.0 - ambient) * lambert);
    }

    if (u_show_seams == 1) {
        col = mix(col, face_tint(face), 0.35);
    }

    frag_color = vec4(col, 1.0);
}
)GLSL";

//====================================
//Helpers
//====================================

namespace {

//Check whether the field a descriptor points to has been allocated. In
//PlanetFieldSet::BedrockOnly mode the climate / biome / cloud Field<T>
//members of PlanetState are default-constructed (grid.n == 0) and must
//not be sampled - the dropdown and refresh paths skip them.
bool field_allocated(const FieldDescriptor& d, PlanetState& s) {
    void* p = d.field_ptr(s);
    switch (d.kind) {
        case FieldKind::F1: return static_cast<Field<float>*>(p)->grid().n > 0;
        case FieldKind::F2: return static_cast<Field<float2>*>(p)->grid().n > 0;
        case FieldKind::F4: return static_cast<Field<float4>*>(p)->grid().n > 0;
        case FieldKind::U8: return static_cast<Field<std::uint8_t>*>(p)->grid().n > 0;
    }
    return false;
}

float intersect_unit_sphere(const glm::vec3& origin, const glm::vec3& dir) {
    float b    = glm::dot(origin, dir);
    float c    = glm::dot(origin, origin) - 1.0f;
    float disc = b * b - c;
    if (disc < 0.0f) return -1.0f;
    float t = -b - std::sqrt(disc);
    return (t >= 0.0f) ? t : -1.0f;
}

float to_degrees(float r) {
    return r * (180.0f / 3.14159265358979323846f);
}

ImVec4 status_colour(PassStatus s) {
    switch (s) {
        case PassStatus::Dirty:   return ImVec4(0.55f, 0.55f, 0.55f, 1.0f);
        case PassStatus::Running: return ImVec4(0.30f, 0.60f, 0.95f, 1.0f);
        case PassStatus::Clean:   return ImVec4(0.30f, 0.85f, 0.40f, 1.0f);
        case PassStatus::Error:   return ImVec4(0.95f, 0.30f, 0.30f, 1.0f);
    }
    return ImVec4(1, 1, 1, 1);
}

const char* status_text(PassStatus s) {
    switch (s) {
        case PassStatus::Dirty:   return "DIRTY";
        case PassStatus::Running: return "RUN..";
        case PassStatus::Clean:   return "CLEAN";
        case PassStatus::Error:   return "ERROR";
    }
    return "?????";
}

const char* source_text(LastSource s) {
    switch (s) {
        case LastSource::None:     return "-";
        case LastSource::Cache:    return "cache";
        case LastSource::Computed: return "computed";
    }
    return "?";
}

ImVec4 log_level_colour(LogLevel l) {
    switch (l) {
        case LogLevel::Info:  return ImVec4(0.85f, 0.85f, 0.85f, 1.0f);
        case LogLevel::Warn:  return ImVec4(0.95f, 0.85f, 0.30f, 1.0f);
        case LogLevel::Error: return ImVec4(0.95f, 0.40f, 0.40f, 1.0f);
    }
    return ImVec4(1, 1, 1, 1);
}

}

struct HoverInfo {
    bool   valid    = false;
    int    face     = -1;
    int    i        = 0;
    int    j        = 0;
    float  lat_deg  = 0.0f;
    float  lon_deg  = 0.0f;
    float  value    = 0.0f;
    Vec3f  position{};
};

class ViewerProgressSink final : public ProgressSink {
public:
    void stage(std::string_view label) override {
        current_stage_.assign(label);
    }
    void fraction(float f) override { current_fraction_ = f; }

    const std::string& stage() const     { return current_stage_; }
    float              fraction() const  { return current_fraction_; }

private:
    std::string current_stage_;
    float       current_fraction_ = 0.0f;
};

//====================================
//Viewer impl
//====================================

struct PreviewViewer::Impl {
    GLFWwindow*                    window   = nullptr;
    PlanetState*                   state    = nullptr;
    ParamRegistry*                 registry = nullptr;
    Pipeline*                      pipeline = nullptr;

    OrbitCamera                    cam;
    SphereMesh                     mesh;
    std::unique_ptr<ShaderProgram> prog;
    std::unique_ptr<PreviewField>  preview;

    ViewerProgressSink             progress;

    GLuint  vao = 0, vbo = 0, ebo = 0;
    GLsizei index_count = 0;

    //Display selection
    int    field_index   = 0;
    int    channel       = 0;
    float  value_min     = -3.0f;
    float  value_max     = 3.0f;
    bool   show_seams    = false;
    bool   show_pattern  = false;
    bool   show_relief   = false;
    float  relief_scale  = 0.02f;
    float  test_freq     = 4.0f;
    float  test_warp     = 0.3f;

    std::vector<float> cpu_mirror;
    bool               texture_dirty   = true;
    bool               mirror_dirty    = true;
    int                last_field_idx  = -1;
    int                last_channel    = -1;

    //Input
    double last_x = 0.0, last_y = 0.0;
    bool   dragging = false;
    bool   have_last_pos = false;

    HoverInfo hover;

    glm::mat4 cached_view = glm::mat4(1.0f);
    glm::mat4 cached_proj = glm::mat4(1.0f);
    int       cached_fb_w = 1;
    int       cached_fb_h = 1;

    Impl(GLFWwindow* w, PlanetState& s, ParamRegistry& reg, Pipeline& pl)
        : window(w),
          state(&s),
          registry(&reg),
          pipeline(&pl),
          mesh(make_icosphere(6)),
          prog(std::make_unique<ShaderProgram>(kVertSrc, kFragSrc)),
          preview(std::make_unique<PreviewField>(s.grid().n))
    {
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glGenBuffers(1, &ebo);

        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER,
                     static_cast<GLsizeiptr>(mesh.positions.size() * sizeof(glm::vec3)),
                     mesh.positions.data(), GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     static_cast<GLsizeiptr>(mesh.indices.size() * sizeof(uint32_t)),
                     mesh.indices.data(), GL_STATIC_DRAW);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), nullptr);
        glEnableVertexAttribArray(0);
        glBindVertexArray(0);

        index_count = static_cast<GLsizei>(mesh.indices.size());

        apply_descriptor_defaults();
    }

    ~Impl() {
        if (vao) glDeleteVertexArrays(1, &vao);
        if (vbo) glDeleteBuffers(1, &vbo);
        if (ebo) glDeleteBuffers(1, &ebo);
    }

    const FieldDescriptor& current_desc() const {
        auto descs = PlanetState::descriptors();
        return descs[field_index];
    }

    void apply_descriptor_defaults() {
        const auto& d = current_desc();
        value_min = d.default_min;
        value_max = d.default_max;
        channel   = std::clamp(channel, 0, d.channels - 1);
    }

    void refresh_display() {
        if (show_pattern) {
            preview->fill_test_pattern(test_freq, test_warp);
            cpu_mirror.clear();
            mirror_dirty  = false;
            texture_dirty = false;
            return;
        }

        const auto& d = current_desc();
        if (!field_allocated(d, *state)) {
            //Selected field isn't allocated (BedrockOnly mode). Show a test
            //pattern as a stand-in so the viewer doesn't sample garbage.
            preview->fill_test_pattern(test_freq, test_warp);
            cpu_mirror.clear();
            mirror_dirty  = false;
            texture_dirty = false;
            return;
        }
        void* p = d.field_ptr(*state);

        switch (d.kind) {
            case FieldKind::F1: {
                auto& f = *static_cast<Field<float>*>(p);
                preview->fill_from(f);
                f.download(cpu_mirror);
                break;
            }
            case FieldKind::F2: {
                auto& f = *static_cast<Field<float2>*>(p);
                preview->fill_from(f, channel);
                std::vector<float2> tmp;
                f.download(tmp);
                cpu_mirror.resize(tmp.size());
                const int c = channel & 0x1;
                for (size_t k = 0; k < tmp.size(); ++k) {
                    cpu_mirror[k] = (c == 0) ? tmp[k].x : tmp[k].y;
                }
                break;
            }
            case FieldKind::F4: {
                auto& f = *static_cast<Field<float4>*>(p);
                preview->fill_from(f, channel);
                std::vector<float4> tmp;
                f.download(tmp);
                cpu_mirror.resize(tmp.size());
                const int c = channel & 0x3;
                for (size_t k = 0; k < tmp.size(); ++k) {
                    float v = 0.0f;
                    switch (c) {
                        case 0: v = tmp[k].x; break;
                        case 1: v = tmp[k].y; break;
                        case 2: v = tmp[k].z; break;
                        default: v = tmp[k].w; break;
                    }
                    cpu_mirror[k] = v;
                }
                break;
            }
            case FieldKind::U8: {
                auto& f = *static_cast<Field<std::uint8_t>*>(p);
                preview->fill_from(f, 1.0f);
                std::vector<std::uint8_t> tmp;
                f.download(tmp);
                cpu_mirror.resize(tmp.size());
                for (size_t k = 0; k < tmp.size(); ++k) {
                    cpu_mirror[k] = static_cast<float>(tmp[k]);
                }
                break;
            }
        }
        texture_dirty = false;
        mirror_dirty  = false;
    }

    void compute_hover(double cursor_x, double cursor_y) {
        hover = HoverInfo{};
        if (cpu_mirror.empty()) return;

        int win_w = 1, win_h = 1;
        glfwGetWindowSize(window, &win_w, &win_h);
        if (win_w <= 0 || win_h <= 0) return;

        float ndc_x =        static_cast<float>(cursor_x) / static_cast<float>(win_w) * 2.0f - 1.0f;
        float ndc_y = 1.0f - static_cast<float>(cursor_y) / static_cast<float>(win_h) * 2.0f;

        glm::mat4 vp_inv = glm::inverse(cached_proj * cached_view);
        glm::vec4 near_h = vp_inv * glm::vec4(ndc_x, ndc_y, -1.0f, 1.0f);
        glm::vec4 far_h  = vp_inv * glm::vec4(ndc_x, ndc_y,  1.0f, 1.0f);
        glm::vec3 near_p = glm::vec3(near_h) / near_h.w;
        glm::vec3 far_p  = glm::vec3(far_h)  / far_h.w;
        glm::vec3 origin = near_p;
        glm::vec3 dir    = glm::normalize(far_p - near_p);

        float t = intersect_unit_sphere(origin, dir);
        if (t < 0.0f) return;

        glm::vec3 hit = origin + t * dir;
        Vec3f p{hit.x, hit.y, hit.z};

        int   face = -1;
        float u = 0.0f, v = 0.0f;
        sphere_to_face_uv(p, face, u, v);

        const auto& g = state->grid();
        int i = g.i_of_u(u);
        int j = g.j_of_v(v);

        hover.valid   = true;
        hover.face    = face;
        hover.i       = i;
        hover.j       = j;
        hover.position = p;
        hover.lat_deg = to_degrees(std::asin(std::clamp(p.y, -1.0f, 1.0f)));
        hover.lon_deg = to_degrees(std::atan2(p.z, p.x));
        int idx = g.global_index(face, i, j);
        if (idx >= 0 && static_cast<size_t>(idx) < cpu_mirror.size()) {
            hover.value = cpu_mirror[idx];
        }
    }

    void run_pipeline() {
        pipeline->run(*state, *registry, progress);
        texture_dirty = true;
        mirror_dirty  = true;
    }

    void run_from(std::string_view pass_name) {
        pipeline->invalidate_from(pass_name);
        run_pipeline();
    }

    //====================================
    //UI sub-panels
    //====================================

    void draw_fields_panel() {
        ImGui::Begin("Fields");

        auto descs = PlanetState::descriptors();
        const auto& cur = descs[field_index];

        if (ImGui::BeginCombo("field", cur.name)) {
            for (int k = 0; k < static_cast<int>(descs.size()); ++k) {
                if (!field_allocated(descs[k], *state)) continue;
                bool sel = (k == field_index);
                if (ImGui::Selectable(descs[k].name, sel)) field_index = k;
                if (sel) ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        if (cur.channels > 1) {
            ImGui::SliderInt("channel", &channel, 0, cur.channels - 1);
        }

        ImGui::Separator();
        ImGui::TextUnformatted("Display range");
        float pad = std::max(1.0f, (cur.default_max - cur.default_min) * 0.5f);
        float lo  = cur.default_min - pad;
        float hi  = cur.default_max + pad;
        ImGui::SliderFloat("min", &value_min, lo, hi);
        ImGui::SliderFloat("max", &value_max, lo, hi);
        if (ImGui::Button("Reset range")) {
            value_min = cur.default_min;
            value_max = cur.default_max;
        }

        ImGui::Separator();
        ImGui::Checkbox("Tint by face (debug seams)", &show_seams);

        ImGui::Separator();
        ImGui::Checkbox("Show 3D relief (displaced + lit)", &show_relief);
        if (show_relief) {
            ImGui::SliderFloat("Relief scale", &relief_scale, 0.0f, 0.10f,
                               "%.3f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Vertex displacement = field_value * scale. "
                                  "0.02 makes a 5 km mountain ~10%% of the "
                                  "unit-sphere radius. Set to 0 to flatten "
                                  "while keeping the lighting.");
            }
        }

        ImGui::Separator();
        if (ImGui::Checkbox("Show test pattern instead", &show_pattern)) {
            texture_dirty = true;
        }
        if (show_pattern) {
            if (ImGui::SliderFloat("freq", &test_freq, 0.5f, 20.0f)) texture_dirty = true;
            if (ImGui::SliderFloat("warp", &test_warp, 0.0f,  1.0f)) texture_dirty = true;
        }

        ImGui::End();
    }

    void draw_param_row(std::string_view path, const ParamSlot& slot) {
        std::string label = slot.label.empty() ? std::string(path) : slot.label;
        if (!slot.units.empty()) { label += " ("; label += slot.units; label += ")"; }
        const std::string id_label = label + "##" + std::string(path);

        if (auto* ip = std::get_if<IntSlot>(&slot.v)) {
            int v = ip->value;
            bool huge_range = (ip->lo == std::numeric_limits<int>::min())
                           || (ip->hi == std::numeric_limits<int>::max())
                           || (static_cast<std::int64_t>(ip->hi) - ip->lo > 1000000);
            bool changed = false;
            if (huge_range) {
                changed = ImGui::InputInt(id_label.c_str(), &v);
            } else {
                changed = ImGui::SliderInt(id_label.c_str(), &v, ip->lo, ip->hi);
            }
            if (changed) registry->set_int(path, v);
        } else if (auto* fp = std::get_if<FloatSlot>(&slot.v)) {
            float v = fp->value;
            if (ImGui::SliderFloat(id_label.c_str(), &v, fp->lo, fp->hi)) {
                registry->set_float(path, v);
            }
        } else if (auto* bp = std::get_if<BoolSlot>(&slot.v)) {
            bool v = bp->value;
            if (ImGui::Checkbox(id_label.c_str(), &v)) {
                registry->set_bool(path, v);
            }
        }
        if (!slot.tooltip.empty() && ImGui::IsItemHovered()) {
            ImGui::SetTooltip("%s", slot.tooltip.c_str());
        }
    }

    void draw_pipeline_panel() {
        ImGui::Begin("Pipeline");

        if (ImGui::Button("Run pipeline")) {
            run_pipeline();
        }
        ImGui::SameLine();
        ImGui::TextDisabled("(runs every Dirty pass; Clean passes skipped)");

        if (!progress.stage().empty()) {
            ImGui::Text("stage: %s", progress.stage().c_str());
            ImGui::ProgressBar(progress.fraction(), ImVec2(-1, 0));
        }

        ImGui::Separator();

        auto entries = pipeline->entries();
        for (std::size_t i = 0; i < entries.size(); ++i) {
            const auto& e = entries[i];
            ImGui::PushID(static_cast<int>(i));

            ImGui::TextColored(status_colour(e.status), "[%s]", status_text(e.status));
            ImGui::SameLine();
            ImGui::Text("%s", e.name.c_str());
            ImGui::SameLine();
            ImGui::TextDisabled("%.1f ms (%s)", e.last_ms, source_text(e.last_source));
            if (!e.last_error.empty()) {
                ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1), "  err: %s", e.last_error.c_str());
            }

            ImGui::SameLine();
            if (ImGui::SmallButton("run from here")) {
                run_from(e.name);
            }

            std::string header = "params: " + e.name;
            if (ImGui::TreeNodeEx(header.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
                std::string prefix = e.name + ".";
                registry->prefix_iter(prefix,
                    [this](std::string_view path, const ParamSlot& slot) {
                        draw_param_row(path, slot);
                    });
                ImGui::TreePop();
            }

            ImGui::Separator();
            ImGui::PopID();
        }

        ImGui::End();
    }

    void draw_log_panel() {
        ImGui::Begin("Log");
        if (ImGui::Button("Clear")) Log::reset();
        ImGui::Separator();

        ImGui::BeginChild("log_scroll", ImVec2(0, 0), false,
                          ImGuiWindowFlags_HorizontalScrollbar);
        Log::drain([](const LogEntry& e) {
            ImGui::TextColored(log_level_colour(e.level),
                               "[%s] %s",
                               e.category.c_str(),
                               e.message.c_str());
        });
        if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 1.0f) {
            ImGui::SetScrollHereY(1.0f);
        }
        ImGui::EndChild();
        ImGui::End();
    }

    void draw_view_panel() {
        ImGui::Begin("View");
        ImGui::TextUnformatted("Drag viewport: orbit. Wheel: zoom.");
        if (ImGui::Button("Reset camera")) cam.reset();
        ImGui::SliderFloat("FOV", &cam.fov_deg, 20.0f, 90.0f);

        ImGui::Separator();
        ImGui::Text("Grid: 6 x %d x %d (stride %d, halo %d)",
                    state->grid().n, state->grid().n,
                    state->grid().stride(), CubedSphereGrid::HALO);
        ImGui::Text("Preview texture: %d x %d per face",
                    preview->resolution(), preview->resolution());
        ImGui::Text("Camera: yaw %.2f, pitch %.2f, dist %.2f",
                    cam.yaw, cam.pitch, cam.distance);

        ImGui::Separator();
        if (hover.valid) {
            ImGui::Text("Hover: lat %+7.3f deg  lon %+8.3f deg",
                        hover.lat_deg, hover.lon_deg);
            ImGui::Text("Face %d  cell (%d, %d)", hover.face, hover.i, hover.j);
            ImGui::Text("Value: %.4f", hover.value);
            ImGui::Text("Dir: (%+.3f, %+.3f, %+.3f)",
                        hover.position.x, hover.position.y, hover.position.z);
        } else {
            ImGui::TextUnformatted("Hover the sphere for cell info.");
        }
        ImGui::End();
    }
};

PreviewViewer::PreviewViewer(GLFWwindow* window,
                             PlanetState& state,
                             ParamRegistry& registry,
                             Pipeline& pipeline)
    : impl_(std::make_unique<Impl>(window, state, registry, pipeline)) {}

PreviewViewer::~PreviewViewer() = default;

ProgressSink& PreviewViewer::progress_sink() {
    return impl_->progress;
}

void PreviewViewer::update_input() {
    auto& im = *impl_;
    ImGuiIO& io = ImGui::GetIO();

    double mx = 0.0, my = 0.0;
    glfwGetCursorPos(im.window, &mx, &my);

    if (!io.WantCaptureMouse) {
        bool lmb = glfwGetMouseButton(im.window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        if (lmb && im.dragging && im.have_last_pos) {
            im.cam.rotate(static_cast<float>(mx - im.last_x),
                          static_cast<float>(my - im.last_y));
        }
        im.dragging = lmb;
        if (std::abs(io.MouseWheel) > 0.0f) {
            im.cam.zoom(io.MouseWheel);
        }
        im.compute_hover(mx, my);
    } else {
        im.dragging  = false;
        im.hover     = HoverInfo{};
    }

    im.last_x = mx;
    im.last_y = my;
    im.have_last_pos = true;
}

void PreviewViewer::render(int fb_w, int fb_h) {
    auto& im = *impl_;

    bool selection_changed = (im.last_field_idx != im.field_index)
                          || (im.last_channel   != im.channel);
    if (selection_changed) {
        im.apply_descriptor_defaults();
        im.last_field_idx = im.field_index;
        im.last_channel   = im.channel;
        im.texture_dirty  = true;
        im.mirror_dirty   = true;
    }
    if (im.texture_dirty) {
        im.refresh_display();
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);

    float aspect = (fb_h > 0) ? float(fb_w) / float(fb_h) : 1.0f;
    glm::mat4 view = im.cam.view();
    glm::mat4 proj = im.cam.proj(aspect);
    glm::mat4 vp   = proj * view;
    im.cached_view = view;
    im.cached_proj = proj;
    im.cached_fb_w = fb_w;
    im.cached_fb_h = fb_h;

    im.prog->use();
    glUniformMatrix4fv(im.prog->uniform("u_view_proj"), 1, GL_FALSE, glm::value_ptr(vp));
    glUniform1f(im.prog->uniform("u_value_min"), im.value_min);
    glUniform1f(im.prog->uniform("u_value_max"), im.value_max);
    glUniform1i(im.prog->uniform("u_show_seams"), im.show_seams ? 1 : 0);
    glUniform1f(im.prog->uniform("u_displace_scale"),
                im.show_relief ? im.relief_scale : 0.0f);
    glUniform1i(im.prog->uniform("u_lit"), im.show_relief ? 1 : 0);
    glUniform1f(im.prog->uniform("u_face_resolution"),
                static_cast<float>(im.preview->resolution()));

    for (int f = 0; f < 6; ++f) {
        glActiveTexture(GL_TEXTURE0 + f);
        glBindTexture(GL_TEXTURE_2D, im.preview->gl_texture(f));
    }
    glUniform1i(im.prog->uniform("u_face0"), 0);
    glUniform1i(im.prog->uniform("u_face1"), 1);
    glUniform1i(im.prog->uniform("u_face2"), 2);
    glUniform1i(im.prog->uniform("u_face3"), 3);
    glUniform1i(im.prog->uniform("u_face4"), 4);
    glUniform1i(im.prog->uniform("u_face5"), 5);

    glBindVertexArray(im.vao);
    glDrawElements(GL_TRIANGLES, im.index_count, GL_UNSIGNED_INT, nullptr);
    glBindVertexArray(0);
}

void PreviewViewer::draw_ui() {
    auto& im = *impl_;
    im.draw_fields_panel();
    im.draw_pipeline_panel();
    im.draw_log_panel();
    im.draw_view_panel();
}

}
