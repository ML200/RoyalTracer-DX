#include "render/preview.h"

#include "render/sphere_mesh.h"
#include "render/shader.h"
#include "render/camera.h"
#include "gpu/preview_field.h"

#include <GL/glew.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>

namespace pb {

static const char* kVertSrc = R"GLSL(
#version 450 core
layout(location = 0) in vec3 a_position;
uniform mat4 u_view_proj;
out vec3 v_pos;
void main() {
    v_pos = a_position;
    gl_Position = u_view_proj * vec4(a_position, 1.0);
}
)GLSL";

static const char* kFragSrc = R"GLSL(
#version 450 core
in vec3 v_pos;
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

    if (u_show_seams == 1) {
        col = mix(col, face_tint(face), 0.35);
    }

    frag_color = vec4(col, 1.0);
}
)GLSL";

struct PreviewViewer::Impl {
    GLFWwindow*    window = nullptr;
    OrbitCamera    cam;
    SphereMesh     mesh;
    std::unique_ptr<ShaderProgram> prog;
    std::unique_ptr<PreviewField>  field;

    GLuint   vao = 0, vbo = 0, ebo = 0;
    GLsizei  index_count = 0;

    float test_freq = 4.0f;
    float test_warp = 0.3f;
    float value_min = 0.0f;
    float value_max = 1.0f;
    bool  show_seams = false;
    bool  field_dirty = true;

    double last_x = 0.0, last_y = 0.0;
    bool   dragging = false;
    bool   have_last_pos = false;

    Impl(GLFWwindow* w, int n)
        : window(w),
          mesh(make_icosphere(5)),
          prog(std::make_unique<ShaderProgram>(kVertSrc, kFragSrc)),
          field(std::make_unique<PreviewField>(n))
    {
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glGenBuffers(1, &ebo);

        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER,
                     static_cast<GLsizeiptr>(mesh.positions.size() * sizeof(glm::vec3)),
                     mesh.positions.data(),
                     GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     static_cast<GLsizeiptr>(mesh.indices.size() * sizeof(uint32_t)),
                     mesh.indices.data(),
                     GL_STATIC_DRAW);

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), nullptr);
        glEnableVertexAttribArray(0);
        glBindVertexArray(0);

        index_count = static_cast<GLsizei>(mesh.indices.size());
    }

    ~Impl() {
        if (vao) glDeleteVertexArrays(1, &vao);
        if (vbo) glDeleteBuffers(1, &vbo);
        if (ebo) glDeleteBuffers(1, &ebo);
    }
};

PreviewViewer::PreviewViewer(GLFWwindow* window, int face_resolution)
    : impl_(std::make_unique<Impl>(window, face_resolution)) {}

PreviewViewer::~PreviewViewer() = default;

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
    } else {
        im.dragging = false;
    }

    im.last_x = mx;
    im.last_y = my;
    im.have_last_pos = true;
}

void PreviewViewer::render(int fb_w, int fb_h) {
    auto& im = *impl_;

    if (im.field_dirty) {
        im.field->fill_test_pattern(im.test_freq, im.test_warp);
        im.field_dirty = false;
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);

    float aspect = (fb_h > 0) ? float(fb_w) / float(fb_h) : 1.0f;
    glm::mat4 vp = im.cam.proj(aspect) * im.cam.view();

    im.prog->use();
    glUniformMatrix4fv(im.prog->uniform("u_view_proj"), 1, GL_FALSE, glm::value_ptr(vp));
    glUniform1f(im.prog->uniform("u_value_min"), im.value_min);
    glUniform1f(im.prog->uniform("u_value_max"), im.value_max);
    glUniform1i(im.prog->uniform("u_show_seams"), im.show_seams ? 1 : 0);

    for (int f = 0; f < 6; ++f) {
        glActiveTexture(GL_TEXTURE0 + f);
        glBindTexture(GL_TEXTURE_2D, im.field->gl_texture(f));
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

    ImGui::Begin("Preview");

    ImGui::TextUnformatted("Drag viewport: orbit. Wheel: zoom.");
    if (ImGui::Button("Reset camera")) im.cam.reset();
    ImGui::SliderFloat("FOV", &im.cam.fov_deg, 20.0f, 90.0f);

    ImGui::Separator();
    ImGui::TextUnformatted("Test pattern (continuous on the sphere)");
    if (ImGui::SliderFloat("frequency", &im.test_freq, 0.5f, 20.0f)) im.field_dirty = true;
    if (ImGui::SliderFloat("warp",      &im.test_warp, 0.0f,  1.0f)) im.field_dirty = true;
    if (ImGui::Button("Refill")) im.field_dirty = true;

    ImGui::Separator();
    ImGui::TextUnformatted("Colormap range");
    ImGui::SliderFloat("min", &im.value_min, -1.0f, 1.0f);
    ImGui::SliderFloat("max", &im.value_max, -1.0f, 2.0f);

    ImGui::Separator();
    ImGui::Checkbox("Tint by face (debug seams)", &im.show_seams);

    ImGui::Separator();
    ImGui::Text("Per-face resolution: %d x %d", im.field->resolution(), im.field->resolution());
    ImGui::Text("Camera: yaw %.2f, pitch %.2f, dist %.2f", im.cam.yaw, im.cam.pitch, im.cam.distance);

    ImGui::End();
}

}
