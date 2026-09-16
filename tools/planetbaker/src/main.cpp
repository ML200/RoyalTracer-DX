#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>

#include <GL/glew.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include "core/log.h"
#include "core/param_registry.h"
#include "core/pass.h"
#include "core/pipeline.h"
#include "core/planet_state.h"
#include "passes/bake_pass.h"
#include "passes/bedrock_noise.h"
#include "passes/hydraulic_erosion_pass.h"
#include "passes/impacts_pass.h"
#include "passes/surface_color_pass.h"
#include "passes/thermal_erosion_pass.h"
#include "render/preview.h"

namespace pb {
    void print_cuda_device_info();
    void run_hello_kernel();
}

static void glfw_error_callback(int error, const char* description) {
    std::fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

int main(int argc, char** argv) {
    //--preview-resolution=N: cells per face. Cap is 16384; on a 32 GB card
    //the BedrockOnly field set fits all the way up. Default 2048 is the
    //middle ground that picks up most of the bedrock-noise frequency
    //content while keeping cold-cache pipeline runs interactive (~few
    //seconds rather than the multi-minute 8k cost). Bump via CLI to
    //--preview-resolution=8192 to make the viewer match the bake exactly.
    int preview_n = 2048;
    for (int a = 1; a < argc; ++a) {
        const char* arg = argv[a];
        constexpr char kPrefix[] = "--preview-resolution=";
        constexpr std::size_t kPLen = sizeof(kPrefix) - 1;
        if (std::strncmp(arg, kPrefix, kPLen) == 0) {
            int v = std::atoi(arg + kPLen);
            if (v >= 64 && v <= 16384) preview_n = v;
            else std::fprintf(stderr,
                "[main] preview-resolution %d out of [64,16384], keeping %d\n",
                v, preview_n);
        }
    }
    //BedrockOnly is the only field set we need now that the climate / cloud /
    //biome / wind / ice / hydraulic passes are gone - the surface heightmap
    //pipeline (bedrock + impacts + thermal + bake) writes only the bedrock
    //fields. PlanetFieldSet::All still exists in case future passes need it.
    const pb::PlanetFieldSet field_set = pb::PlanetFieldSet::BedrockOnly;
    std::printf("[main] preview resolution: %d per face (bedrock-only)\n", preview_n);

    //Estimate VRAM / DRAM up front so the user can see what they're about
    //to ask CUDA for. PlanetState + GL preview textures share VRAM; the
    //CPU mirror buffer for hover is DRAM.
    const std::size_t ps_bytes = pb::estimate_planet_state_bytes(preview_n, field_set);
    const std::size_t gl_bytes =
        static_cast<std::size_t>(6) * preview_n * preview_n * sizeof(float);
    const std::size_t cpu_bytes =
        static_cast<std::size_t>(6) * (preview_n + 4) * (preview_n + 4) * sizeof(float);
    const double gib = static_cast<double>(1ull << 30);
    std::printf("[main] estimated VRAM: %.2f GB (PlanetState %.2f + GL preview %.2f)\n",
                (ps_bytes + gl_bytes) / gib, ps_bytes / gib, gl_bytes / gib);
    std::printf("[main] estimated DRAM: %.2f GB (CPU hover mirror)\n",
                cpu_bytes / gib);

    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) {
        std::fprintf(stderr, "Failed to initialize GLFW\n");
        return EXIT_FAILURE;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
    glfwWindowHint(GLFW_DEPTH_BITS, 24);

    GLFWwindow* window = glfwCreateWindow(1600, 1000, "planet_bake", nullptr, nullptr);
    if (!window) {
        std::fprintf(stderr, "Failed to create window\n");
        glfwTerminate();
        return EXIT_FAILURE;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    glewExperimental = GL_TRUE;
    GLenum glew_err = glewInit();
    if (glew_err != GLEW_OK) {
        std::fprintf(stderr, "Failed to initialize GLEW: %s\n",
                     reinterpret_cast<const char*>(glewGetErrorString(glew_err)));
        glfwDestroyWindow(window);
        glfwTerminate();
        return EXIT_FAILURE;
    }
    glGetError();

    std::printf("[GL] %s, GLSL %s\n",
                reinterpret_cast<const char*>(glGetString(GL_VERSION)),
                reinterpret_cast<const char*>(glGetString(GL_SHADING_LANGUAGE_VERSION)));

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 450");

    pb::print_cuda_device_info();
    pb::run_hello_kernel();

    //M3: pipeline + registry + cache wiring.
    pb::PlanetState   state(preview_n, field_set);
    pb::ParamRegistry registry;
    pb::Pipeline      pipeline(std::filesystem::path("./cache"));

    //Minimal surface-heightmap pipeline. Climate / clouds / biome / wind /
    //ice / hydraulic were stripped after the bedrock pass landed; the
    //heightmap (bedrock + impact craters + thermal relaxation) is the only
    //product the renderer needs right now. The BakePass at the end writes
    //the final cubemap to disk on demand (gated by bake.enabled).
    pipeline.add_pass(std::make_unique<pb::BedrockNoisePass>());
    pipeline.add_pass(std::make_unique<pb::ImpactsPass>());
    pipeline.add_pass(std::make_unique<pb::ThermalErosionPass>());
    pipeline.add_pass(std::make_unique<pb::HydraulicErosionPass>());
    pipeline.add_pass(std::make_unique<pb::SurfaceColorPass>());
    pipeline.add_pass(std::make_unique<pb::BakePass>());
    pipeline.declare_all(registry);

    const std::filesystem::path config_path = "./config.json";
    registry.load(config_path);

    //Wire dirty tracking AFTER load so the load doesn't fire cascading
    //invalidations (each set_ inside load() would otherwise mark passes
    //dirty even though their values match what's already cached).
    pipeline.wire_dirty_tracking(registry);

    pb::PreviewViewer viewer(window, state, registry, pipeline);

    //Startup run: a cold cache will compute and persist; a warm cache will
    //load from disk and skip kernels.
    pipeline.run(state, registry, viewer.progress_sink());

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        viewer.update_input();

        ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport(),
                                     ImGuiDockNodeFlags_PassthruCentralNode);
        viewer.draw_ui();

        int fb_w = 0, fb_h = 0;
        glfwGetFramebufferSize(window, &fb_w, &fb_h);
        glViewport(0, 0, fb_w, fb_h);
        glClearColor(0.05f, 0.05f, 0.08f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        viewer.render(fb_w, fb_h);

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    registry.save(config_path);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
    return EXIT_SUCCESS;
}
