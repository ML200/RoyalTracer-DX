#include "../stdafx.h"
#include "Editor.h"
#include "../../shaders/OceanLayout.h"
#include <unordered_set>

namespace {
void SetInitialPanelPosition(ImVec2 offset) {
    // Multi-viewports use desktop coordinates; keep new panels inside the main window.
    const ImVec2 origin = ImGui::GetMainViewport()->Pos;
    ImGui::SetNextWindowPos(ImVec2(origin.x + offset.x, origin.y + offset.y), ImGuiCond_FirstUseEver);
}

// Beaufort force from the 10 m wind, so the speed slider reads as a sea state rather than as a
// bare number. Upper bound of each force, m/s.
int BeaufortForce(float windSpeed) {
    static const float kUpper[] = {0.5f, 1.5f, 3.3f, 5.5f, 7.9f, 10.7f, 13.8f, 17.1f, 20.7f, 24.4f, 28.4f, 32.6f};
    for (int i = 0; i < (int)(sizeof(kUpper) / sizeof(kUpper[0])); ++i)
        if (windSpeed < kUpper[i])
            return i;
    return 12;
}

const char* BeaufortName(int force) {
    static const char* kNames[] = {"calm",      "light air",       "light breeze", "gentle breeze",
                                   "moderate breeze", "fresh breeze", "strong breeze", "near gale",
                                   "gale",      "strong gale",     "storm",        "violent storm",
                                   "hurricane"};
    return kNames[std::clamp(force, 0, 12)];
}

// The ocean owns one block of material slots: a whitecap-coverage ramp crossed with the anisotropy
// and direction axes the shader picks from. An edit that reached only the first slot would change
// one level out of four thousand, so every edit is broadcast across the block exactly as the
// generated path writes it, leaving the per-slot anisotropy axes alone.
void BroadcastWaterMaterial(Scene& scene, UINT base) {
    auto& m = scene.materials;
    if ((size_t)base + OCEAN_MATERIAL_COUNT > m.size() || m.sssEnable.size() < m.size())
        return;
    const XMFLOAT4 kd = m.Kd[base];
    const float weight = m.sssWeight[base];
    const uint8_t enable = m.sssEnable[base];
    for (UINT i = 1; i < OCEAN_MATERIAL_COUNT; ++i) {
        const float coverage = float(i % OCEAN_MATERIAL_LEVELS) / float(OCEAN_MATERIAL_LEVELS - 1);
        m.Kd[base + i] = XMFLOAT4{kd.x, kd.y, kd.z, kd.w + (1.0f - kd.w) * coverage};
        m.Ni[base + i] = m.Ni[base];
        m.Tf[base + i] = m.Tf[base];
        m.sssAlbedo[base + i] = m.sssAlbedo[base];
        m.sssRadius[base + i] = m.sssRadius[base];
        m.sssPhaseG[base + i] = m.sssPhaseG[base];
        m.sssWeight[base + i] = weight * (1.0f - coverage);
        m.sssEnable[base + i] = coverage < 1.0f ? enable : uint8_t(0u);
    }
}
}

void Editor::Init(HWND hwnd, ID3D12Device* device, UINT numFramesInFlight, ID3D12DescriptorHeap* srvHeap,
                  D3D12_CPU_DESCRIPTOR_HANDLE fontCpu, D3D12_GPU_DESCRIPTOR_HANDLE fontGpu) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

    ImGui::StyleColorsDark();
    auto& style = ImGui::GetStyle();
    style.WindowRounding = 0.0f;
    style.FrameRounding = 2.0f;
    style.GrabRounding = 2.0f;
    style.Colors[ImGuiCol_WindowBg].w = 1.0f;

    ImGui_ImplWin32_Init(hwnd);
    ImGui_ImplDX12_Init(device, numFramesInFlight, DXGI_FORMAT_R8G8B8A8_UNORM, srvHeap, fontCpu, fontGpu);
}

void Editor::Shutdown() {
    ImGui_ImplDX12_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();
}

void Editor::Draw(Scene& scene, Camera& camera, FlyCamController& flyCam, PassSystem& passes, DLSSManager& dlss,
                  DLSSNRManager& dlssNR, DLSSGSettings& dlssG, IntegratorSettings& restir, float fps,
                  const FrameStats& stats, const planet::StreamOrchestrator::Stats& planetStats,
                  mc::VoxelStreamer* voxels, ocean::OceanSystem* ocean) {
    // Keep history advancing while panels are hidden.
    if (!m_performanceHistory.paused) {
        m_performanceHistory.push(stats);
        m_performanceFrame.frame = stats;
        m_performanceFrame.stream = planetStats;
        m_performanceFrame.fps = fps;
        m_performanceFrame.hasMinecraft = voxels && voxels->world();
        m_performanceFrame.minecraft = m_performanceFrame.hasMinecraft ? voxels->stats() : mc::StreamerStats{};
    }

    ImGui_ImplDX12_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();

    // An empty frame lets ImGui close detached windows when the editor is hidden.
    if (!m_visible) {
        ImGui::Render();
        return;
    }

    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("View")) {
            ImGui::MenuItem("Scene", nullptr, &m_showScene);
            ImGui::MenuItem("Camera", nullptr, &m_showCamera);
            ImGui::MenuItem("Materials", nullptr, &m_showMaterials);
            ImGui::MenuItem("Environment", nullptr, &m_showSun);
            ImGui::Separator();
            ImGui::MenuItem("Integrator", nullptr, &m_showIntegrator);
            ImGui::MenuItem("DLSS", nullptr, &m_showDLSS);
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Diagnostics")) {
            ImGui::MenuItem("Render passes", nullptr, &m_showPipeline);
            ImGui::MenuItem("DLSS buffers", nullptr, &m_showDlssInputs);
            ImGui::MenuItem("Performance", nullptr, &m_showPerformance);
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Experimental")) {
            if (voxels)
                ImGui::MenuItem("Minecraft", nullptr, &m_showMinecraft);
            if (ocean && ocean->Enabled())
                ImGui::MenuItem("Water", nullptr, &m_showWater);
            ImGui::MenuItem("DLSS 5 Neural Rendering", nullptr, &m_showDLSSNR);
            ImGui::EndMenu();
        }
        ImGui::Separator();
        if (dlssG.enabled && dlssG.framesToGenerate > 0) {
            float presentedFps = fps * (1 + dlssG.framesToGenerate);
            ImGui::Text("%.1f fps (%.1f rendered + %dx FG) | %.2f ms", presentedFps, fps, 1 + dlssG.framesToGenerate,
                        fps > 0 ? 1000.0f / fps : 0.0f);
        } else {
            ImGui::Text("%.1f fps | %.2f ms", fps, fps > 0 ? 1000.0f / fps : 0.0f);
        }
        ImGui::Separator();
        ImGui::TextDisabled("%s", restir.sharcEnabled ? "Path tracer + SHARC" : "Path tracer");
        ImGui::SetItemTooltip("%u instances | %u meshes | CPU %.2f ms | GPU wait %.2f ms", stats.instanceCount,
                              stats.meshCount, stats.cpuFrameMs, stats.gpuWaitMs);
        ImGui::EndMainMenuBar();
    }

    if (m_showScene)
        DrawScenePanel(scene);
    if (m_showCamera)
        DrawCameraPanel(camera, flyCam);
    if (m_showPipeline)
        DrawPassPipelinePanel(passes);
    if (m_showDLSS)
        DrawDLSSPanel(camera, dlss, dlssG);
    if (m_showDLSSNR)
        DrawDLSSNRPanel(dlssNR);
    if (m_showDlssInputs)
        DrawDlssInputsPanel(restir, dlss);
    if (!m_showDlssInputs)
        restir.dlssDebugLayer = 0;
    if (m_showIntegrator)
        DrawIntegratorPanel(restir, stats);
    if (m_showSun)
        DrawSunPanel(scene, camera, stats, voxels);
    if (m_showMaterials)
        DrawMaterialInspector(scene, camera, restir, voxels);
    if (m_showPerformance)
        DrawPerformancePanel(m_performanceFrame.stream, m_performanceFrame.frame, m_performanceFrame.fps,
                             m_performanceFrame.hasMinecraft ? &m_performanceFrame.minecraft : nullptr);
    if (m_showMinecraft && voxels)
        DrawMinecraftPanel(*voxels);
    if (m_showWater && ocean && ocean->Enabled())
        DrawWaterPanel(*ocean, scene);

    ImGui::Render();
}

void Editor::DrawMinecraftPanel(mc::VoxelStreamer& v) {
    SetInitialPanelPosition(ImVec2(380, 30));
    ImGui::SetNextWindowSize(ImVec2(440, 560), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Minecraft###Minecraft World", &m_showMinecraft)) {
        ImGui::End();
        return;
    }

    mc::StreamerConfig& cfg = v.config();
    const mc::StreamerStats& st = v.stats();
    if (const mc::World* w = v.world()) {
        const auto& ws = w->stats();
        ImGui::Text("%s", w->level().name.c_str());
        ImGui::TextDisabled("%u chunks | %u sections | %u block states | %d LOD levels | %.0f MB in RAM", ws.chunks,
                            ws.sections, ws.blockStates, w->lod_levels(), (double)ws.storeBytes / (1024.0 * 1024.0));
        ImGui::TextDisabled("load %.1f s + LOD %.1f s; warm-up %.1f s%s", ws.loadSeconds, ws.lodSeconds,
                            st.warmUpSeconds, st.warmUpComplete ? "" : " (incomplete)");
    }

    ImGui::SeparatorText("Level of detail");
    float budgetM = (float)((double)cfg.triangleBudget / 1.0e6);
    if (ImGui::SliderFloat("Triangle budget (M)", &budgetM, 5.0f, 300.0f, "%.0f"))
        cfg.triangleBudget = (uint64_t)(budgetM * 1.0e6);
    ImGui::SetItemTooltip("Triangles the desired LOD cut may reach. The detail distance shrinks until the cut fits\n"
                          "this budget and the GPU pools, and grows back towards its maximum when there is room.");
    ImGui::Checkbox("Adapt detail distance to the budget", &cfg.adaptiveLod);
    ImGui::SliderFloat("Detail distance (max)", &cfg.lodFactor, 32.0f, 2048.0f, "%.0f", ImGuiSliderFlags_Logarithmic);
    ImGui::SetItemTooltip("A chunk refines while the camera is closer than this many blocks per voxel size:\n"
                          "1-block voxels within 2x this distance, 2-block voxels within 4x, and so on.");
    ImGui::Text("in use %.0f: full detail within %.0f blocks | cut %.1fM of %.1fM tris, %.0f%% built", st.lodFactorNow,
                st.lodFactorNow * 2.0f, (double)st.trianglesEstimated / 1.0e6, (double)st.triangleBudget / 1.0e6,
                st.cutReadyFraction * 100.0f);
    ImGui::SetItemTooltip(
        "The detail distance also shrinks while the desired cut is mostly unbuilt (a camera jump, very fast\n"
        "flight) and grows back once it is resident, so the surroundings fill in coarse first.");
    ImGui::SliderInt("Flat colour from level", &cfg.flatColorLevel, 0, 10);
    ImGui::SetItemTooltip(
        "Chunks at this LOD level and coarser use one average colour per block face instead of textures\n"
        "(finer levels tile the block textures once per block, whatever the voxel size).");
    ImGui::Checkbox("Freeze LOD selection", &cfg.freezeLod);
    int budget = (int)cfg.buildBudget;
    if (ImGui::SliderInt("BLAS builds per frame", &budget, 1, 64))
        cfg.buildBudget = (uint32_t)budget;
    int evict = (int)cfg.evictFrames;
    if (ImGui::SliderInt("Evict unused after (frames)", &evict, 30, 2000))
        cfg.evictFrames = (uint32_t)evict;

    ImGui::SeparatorText("Lights");
    ImGui::Checkbox("Emissive blocks light the scene", &cfg.lights);
    ImGui::SetItemTooltip(
        "Emissive block faces become light-tree lights: one light BLAS per chunk, built with the mesh,\n"
        "and a light TLAS rebuilt in the background whenever the set of lit chunks changes.");
    int maxLightK = (int)(cfg.maxLightTris / 1000u);
    if (ImGui::SliderInt("Light triangles in the tree (k)", &maxLightK, 50, 4000))
        cfg.maxLightTris = (uint32_t)maxLightK * 1000u;
    ImGui::SetItemTooltip(
        "Nearest chunks first; the rest keep glowing through BSDF hits but are not sampled directly.");
    ImGui::SliderInt("Lights up to LOD level", &cfg.lightMaxLevel, 0, 10);
    ImGui::Text("in tree: %.2fM tris in %u chunks | resident %.2fM | dropped %u | set v%u, tree v%u%s",
                (double)st.lightTrisInTree / 1.0e6, st.lightChunksInTree, (double)st.lightTrisResident / 1.0e6,
                st.lightChunksDropped, st.lightVersion, st.lightLiveVersion,
                v.has_live_lights() ? "" : " (no voxel lights in use)");

    ImGui::SeparatorText("Streaming");
    ImGui::Text("desired %u | on screen %u | ready %u (+%u empty)", st.desired, st.rendered, st.ready, st.empty);
    ImGui::Text("pending %u | meshing %u | meshed %u | uploading %u", st.pending, st.meshing, st.meshed, st.uploading);
    ImGui::Text("triangles on screen: %.2f M (resident %.2f M)", (double)st.trianglesRendered / 1.0e6,
                (double)st.trianglesResident / 1.0e6);
    ImGui::Text("this frame: %u BLAS builds, %.1f MB uploaded", st.buildsThisFrame,
                (double)st.uploadBytesThisFrame / (1024.0 * 1024.0));
    ImGui::Text("mesh job %.2f ms avg | select %.2f ms | record %.2f ms", st.meshMsAvg, st.selectMs, st.recordMs);
    ImGui::Text("tracked %u | evicted %u | no-space retries %u", st.chunksTracked, st.evicted, st.allocFailures);

    ImGui::SeparatorText("GPU pools");
    auto bar = [](const char* label, double used, double cap, const char* unit) {
        char txt[96];
        snprintf(txt, sizeof(txt), "%.1f / %.1f %s", used, cap, unit);
        ImGui::ProgressBar(cap > 0.0 ? (float)(used / cap) : 0.0f, ImVec2(-1.0f, 0.0f), txt);
        ImGui::SameLine(0.0f, 8.0f);
        ImGui::TextUnformatted(label);
    };
    bar("vertices", st.vertexUsed / 1.0e6, cfg.vertexCapacity / 1.0e6, "M");
    bar("indices", st.indexUsed / 1.0e6, cfg.indexCapacity / 1.0e6, "M");
    bar("material ids", st.matIdUsed / 1.0e6, cfg.matIdCapacity / 1.0e6, "M");
    bar("BLAS pool", (double)st.blasUsed / (1024.0 * 1024.0), (double)cfg.blasPoolBytes / (1024.0 * 1024.0), "MB");
    if (cfg.blasCompaction)
        bar("BLAS build pool", (double)st.blasBuildUsed / (1024.0 * 1024.0),
            (double)cfg.blasBuildPoolBytes / (1024.0 * 1024.0), "MB");
    bar("light records", st.lightRecUsed / 1.0e6, cfg.lightRecordCapacity / 1.0e6, "M");
    bar("light nodes", st.lightNodeUsed / 1.0e6, cfg.lightNodeCapacity / 1.0e6, "M");
    ImGui::End();
}

void Editor::DrawWaterPanel(ocean::OceanSystem& oceanSystem, Scene& scene) {
    SetInitialPanelPosition(ImVec2(420, 60));
    ImGui::SetNextWindowSize(ImVec2(470, 700), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Water###Ocean", &m_showWater)) {
        ImGui::End();
        return;
    }

    // Anything staged here stays staged until it is committed; outside the staging window the
    // panel follows the sea state the system actually holds, so a scene or fixture edit shows up.
    if (!m_waterRespecPending && !ImGui::IsAnyItemActive())
        m_waterParams = oceanSystem.GetParams();

    ocean::Params& p = m_waterParams;
    const auto& st = oceanSystem.GetStats();
    bool cheap = false;  // takes effect on the next frame
    bool respec = false; // re-bakes the spectrum, so it commits on release

    const int force = BeaufortForce(p.windSpeed);
    ImGui::Text("Beaufort %d, %s", force, BeaufortName(force));
    ImGui::TextDisabled("Hs %.2f m | sea level %.2f m | %u tiles, %.2fM tris | bake %.0f ms", st.significantWaveHeight,
                        st.surfaceY, st.tiles, (double)st.triangles / 1.0e6, st.bakeMs);

    // This work sits on the streaming compute queue the graphics queue waits on, so it never shows
    // up in a per-pass profile of the render passes - it lands in the frame's GPU wait instead.
    if (st.gpuTotalMs > 0.0f) {
        ImGui::Text("GPU %.2f ms per frame, before a ray is traced", st.gpuTotalMs);
        if (ImGui::IsItemHovered()) {
            ImGui::BeginTooltip();
            ImGui::TextUnformatted("Recorded on the streaming compute queue, which the graphics queue waits on\n"
                                   "before it traces. A profiler that times the render passes cannot see it:\n"
                                   "it turns up as the gap between the passes and the frame's GPU wait.");
            ImGui::Separator();
            for (int i = 0; i < 6; ++i) {
                char name[32];
                const wchar_t* w = ocean::OceanSystem::kGpuStageNames[i];
                size_t n = 0;
                wcstombs_s(&n, name, sizeof(name), w, _TRUNCATE);
                ImGui::Text("%-14s %6.2f ms", name, st.gpuStageMs[i]);
            }
            ImGui::EndTooltip();
        }
        // The two that scale with the tile budget, called out because they are the ones a scene
        // can do something about. Every resident tile is rebuilt or refitted every frame, its
        // vertices having moved, so this triangle count is paid in full each time.
        ImGui::TextDisabled("  tessellation %.2f ms + structures %.2f ms: %.2fM triangles every frame",
                            st.gpuStageMs[4], st.gpuStageMs[5], (double)st.triangles / 1.0e6);
        ImGui::SetItemTooltip("Tiles times %u triangles each (OCEAN_TILE_GRID is %u quads per edge).\n"
                              "Halving the grid quarters this; halving the tile budget halves it.",
                              (uint32_t)OCEAN_TILE_TRIS, (uint32_t)OCEAN_TILE_GRID);
        ImGui::TextDisabled("  %u refits + %u rebuilds this frame", st.refits, st.builds);
        ImGui::SetItemTooltip("A refit keeps the tree the structure was built around and only moves its\n"
                              "vertices; a rebuild starts again and costs several times as much. A tile is\n"
                              "rebuilt once its refits have had about a second to drift, spread across the\n"
                              "interval so they do not all fall due on the same frame.");
    }

    ImGui::SeparatorText("Placement");
    respec |= ImGui::DragFloat("Sea level", &p.seaLevelY, 0.25f, -1000.0f, 10000.0f, "%.2f");
    ImGui::SetItemTooltip("Height the waves swing about, in world units. In a Minecraft world this is where\n"
                          "the water line is: the top of the surface blocks, once the world's own water is\n"
                          "replaced from the Materials panel.");
    respec |= ImGui::Checkbox("Keep troughs above zero", &p.keepAboveZero);
    ImGui::SetItemTooltip("Lifts the level until the deepest trough clears the ground plane, because the\n"
                          "atmosphere treats anything below it as underground and renders it black. Turn it\n"
                          "off when the sea level is set deliberately and already sits well clear.");
    if (p.keepAboveZero)
        ImGui::TextDisabled("in use %.2f (lifted to clear a %.2f m trough)", st.surfaceY,
                            st.surfaceY - p.seaLevelY);

    ImGui::SeparatorText("Wind");
    respec |= ImGui::SliderFloat("Speed (m/s)", &p.windSpeed, 0.0f, 32.0f, "%.1f", ImGuiSliderFlags_AlwaysClamp);
    ImGui::SetItemTooltip("Wind at 10 m. It sets the wave height through the JONSWAP spectrum and, with the\n"
                          "turbulence below, how hard the crests are whipped over - the same turbulence\n"
                          "setting reads far calmer in a breeze than in a gale.");
    respec |=
        ImGui::SliderFloat("Direction (deg)", &p.windDirectionDeg, 0.0f, 360.0f, "%.0f", ImGuiSliderFlags_AlwaysClamp);
    ImGui::SetItemTooltip("Bearing the wind blows towards, clockwise from +Z.");
    float fetchKm = p.fetch / 1000.0f;
    if (ImGui::SliderFloat("Fetch (km)", &fetchKm, 1.0f, 500.0f, "%.0f",
                           ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp)) {
        p.fetch = fetchKm * 1000.0f;
        respec = true;
    }
    ImGui::SetItemTooltip("How far the wind has blown over open water. A short fetch keeps the sea young:\n"
                          "shorter and steeper for the same wind. 200 km is already nearly fully developed.");
    respec |= ImGui::SliderFloat("Height scale", &p.amplitudeScale, 0.0f, 3.0f, "%.2fx", ImGuiSliderFlags_AlwaysClamp);
    ImGui::SetItemTooltip("Artistic gain over the whole wave field; 1 is the physical JONSWAP height.");

    bool fixedHeight = p.significantHeight >= 0.0f;
    if (ImGui::Checkbox("Sea height (m)##fix", &fixedHeight)) {
        p.significantHeight = fixedHeight ? std::max(0.05f, (float)st.significantWaveHeight) : -1.0f;
        respec = true;
    }
    ImGui::SetItemTooltip("Drive the spectrum from a significant wave height instead of from the wind's own\n"
                          "energy. The wind still sets the direction, the spreading and the shape.");
    if (fixedHeight) {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(-FLT_MIN);
        respec |= ImGui::SliderFloat("##Hs", &p.significantHeight, 0.0f, 14.0f, "%.2f m",
                                     ImGuiSliderFlags_AlwaysClamp);
    }
    bool fixedPeriod = p.peakPeriod > 0.0f;
    if (ImGui::Checkbox("Peak period (s)##fix", &fixedPeriod)) {
        p.peakPeriod = fixedPeriod ? 8.0f : -1.0f;
        respec = true;
    }
    ImGui::SetItemTooltip("Period of the energy-carrying waves. Left off, the wind and fetch set it.");
    if (fixedPeriod) {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(-FLT_MIN);
        respec |= ImGui::SliderFloat("##Tp", &p.peakPeriod, 2.0f, 22.0f, "%.1f s", ImGuiSliderFlags_AlwaysClamp);
    }

    ImGui::SeparatorText("Turbulence");
    respec |= ImGui::SliderFloat("Turbulence", &p.turbulence, 0.0f, 3.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
    ImGui::SetItemTooltip("How whipped the water is, on top of the height the wind already gives it. It drives\n"
                          "the horizontal displacement that pulls crests into peaks, the gain on the short wind\n"
                          "waves and the crest sharpening below, so the small detail runs from rounded swell at\n"
                          "0 to a hard broken chop at 2. Wind speed carries part of it on its own, so a gale is\n"
                          "whipped harder than a breeze at the same setting.");
    cheap |= ImGui::SliderFloat("Crest sharpening", &p.crestSharpening, 0.0f, 8.0f, "%.2f",
                                ImGuiSliderFlags_AlwaysClamp);
    ImGui::SetItemTooltip("Second-order Stokes sharpening, as a multiple of the physical bound harmonic. A\n"
                          "linear spectrum is Gaussian and therefore symmetric - every trough mirrors a crest,\n"
                          "which is what makes an FFT sea read as rolling rather than as a real one. This pulls\n"
                          "each band's crests into narrow peaks and leaves long shallow troughs behind them.\n"
                          "0 is the symmetric sea, 1 the physical wave; past that each band runs up against its\n"
                          "own steepness limit and stops. It costs nothing against the no-fold bound, so it is\n"
                          "what buys a sharp crest once the horizontal displacement has saturated.");
    // The wind's own contribution is invisible in the slider values, so show what the sea gets.
    ImGui::TextDisabled("in use: crest displacement x%.2f, short-wave gain x%.2f", ocean::EffectiveChoppiness(p),
                        ocean::ShortWaveAmplitude(p, 6.283185307179586 / 2.0));
    ImGui::TextDisabled("crest steepness %.3f of %.2f%s | no-fold gain %.3f", st.crestSteepness,
                        ocean::kMaxSkewSteepness,
                        st.crestSteepness >= ocean::kMaxSkewSteepness * 0.999 ? " (at the limit)" : "",
                        st.conditioningGain);
    ImGui::SetItemTooltip("The sharpest band's skew against the point where its troughs would turn back up, and\n"
                          "the uniform gain the GPU applies to the horizontal displacement to keep the surface\n"
                          "from folding into itself. A no-fold gain well below 1 means the chop is already\n"
                          "saturated and more choppiness buys nothing - reach for crest sharpening instead.");

    respec |= ImGui::SliderFloat("Patch variation", &p.turbulenceVariation, 0.0f, 0.9f, "%.2f",
                                 ImGuiSliderFlags_AlwaysClamp);
    ImGui::SetItemTooltip("Kilometre-scale variation in sea state. A real ocean is not one sea everywhere:\n"
                          "currents shear the surface and the wind arrives in gusts and lulls, leaving patches\n"
                          "of steeper, more broken water drifting between calmer lanes. 0 is the uniform sea;\n"
                          "0.4 means the roughest patches carry 40%% more wave than the mean and the calmest\n"
                          "40%% less. Because the crest warp is quadratic, a rough patch is more peaked as well\n"
                          "as taller.");
    float patchKm = p.turbulencePeriod / 1000.0f;
    if (ImGui::SliderFloat("Patch period (km)", &patchKm, 0.5f, 40.0f, "%.1f",
                           ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp)) {
        p.turbulencePeriod = patchKm * 1000.0f;
        respec = true;
    }
    ImGui::SetItemTooltip("Tiling period of that field. The octaves inside it run from half this down to a\n"
                          "thirty-second, so the patches themselves are a few hundred metres to a few km.");

    if (ImGui::CollapsingHeader("Turbulence detail")) {
        respec |= ImGui::SliderFloat("Crest displacement", &p.choppiness, 0.0f, 2.0f, "%.2f",
                                     ImGuiSliderFlags_AlwaysClamp);
        ImGui::SetItemTooltip("Base horizontal displacement gain, before turbulence and wind scale it. The GPU\n"
                              "still bounds the composite deformation each frame, so this is a request.");
        respec |= ImGui::SliderFloat("Short-wave amplitude", &p.shortWaveAmplitude, 0.0f, 3.0f, "%.2f",
                                     ImGuiSliderFlags_AlwaysClamp);
        ImGui::SetItemTooltip("Base gain on wind waves shorter than 8 m, reaching full gain below 2 m. The long\n"
                              "wind sea and the independent swell are left alone.");
    }

    ImGui::SeparatorText("Swell");
    respec |= ImGui::SliderFloat("Swell height (m)", &p.swellHeight, 0.0f, 8.0f, "%.2f",
                                 ImGuiSliderFlags_AlwaysClamp);
    ImGui::SetItemTooltip("Independent long waves arriving from a distant storm, on their own bearing.");
    respec |= ImGui::SliderFloat("Swell period (s)", &p.swellPeriod, 2.0f, 25.0f, "%.1f",
                                 ImGuiSliderFlags_AlwaysClamp);
    respec |= ImGui::SliderFloat("Swell direction (deg)", &p.swellDirectionDeg, 0.0f, 360.0f, "%.0f",
                                 ImGuiSliderFlags_AlwaysClamp);
    respec |= ImGui::SliderFloat("Swell spread (deg)", &p.swellSpreadDeg, 2.0f, 90.0f, "%.0f",
                                 ImGuiSliderFlags_AlwaysClamp);
    ImGui::SetItemTooltip("Angular width of the swell. Narrow spread gives long parallel crests.");
    respec |= ImGui::SliderFloat("Wind-sea narrowing", &p.swell, 0.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
    ImGui::SetItemTooltip("Narrows the wind sea itself towards long-crested swell. Pushed up it turns the\n"
                          "surface into parallel corrugations, so the default stays low.");
    respec |= ImGui::SliderFloat("Downwind alignment", &p.windAlign, 0.0f, 4.0f, "%.2f",
                                 ImGuiSliderFlags_AlwaysClamp);
    ImGui::SetItemTooltip("Suppresses waves travelling into the wind; higher values make the crests more\n"
                          "parallel to each other.");

    ImGui::SeparatorText("Water body");
    cheap |= ImGui::SliderFloat("Chlorophyll (mg/m^3)", &p.chlorophyll, 0.001f, 10.0f, "%.3f",
                                ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
    ImGui::SetItemTooltip("Morel's Case-1 water. 0.03 is clear open ocean and reads deep indigo; 1-10 is\n"
                          "coastal green. Drives the absorption and scattering below.");
    cheap |= ImGui::SliderFloat("Turbidity", &p.turbidity, 0.0f, 10.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
    ImGui::SetItemTooltip("Extra scattering for sediment-laden water, over the Case-1 model.");
    cheap |= ImGui::SliderFloat("Scattering strength", &p.subsurfaceStrength, 0.0f, 1.0f, "%.2f",
                                ImGuiSliderFlags_AlwaysClamp);
    cheap |= ImGui::SliderFloat("Mean free path scale", &p.subsurfaceRadiusScale, 0.01f, 10.0f, "%.3f",
                                ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
    cheap |= ImGui::SliderFloat("Forward scattering", &p.subsurfacePhaseG, -0.95f, 0.95f, "%.2f",
                                ImGuiSliderFlags_AlwaysClamp);
    cheap |= ImGui::SliderFloat("Surface body weight", &p.bodyWeight, 0.0f, 1.0f, "%.2f",
                                ImGuiSliderFlags_AlwaysClamp);
    ImGui::SetItemTooltip("Optional diffuse contribution at the surface. Clear water needs none: the colour\n"
                          "comes from transmission into the volume.");

    // Material adjustments, moved here from the material inspector: the ocean's slots are
    // generated from the optics above, and hand edits take the whole block over.
    const int waterMat = scene.oceanInstanceSlots && scene.oceanMatIndex < scene.materials.size()
                             ? (int)scene.oceanMatIndex
                             : -1;
    if (waterMat >= 0 && ImGui::CollapsingHeader("Material", ImGuiTreeNodeFlags_DefaultOpen)) {
        auto& mats = scene.materials;
        const int i = waterMat;
        bool matChanged = false;

        if (scene.oceanMaterialEdited) {
            ImGui::TextDisabled("Edited by hand; the optics above no longer regenerate it.");
            ImGui::SameLine();
            if (ImGui::SmallButton("Revert")) {
                const Material generated = ocean::OceanSystem::MakeMaterial(p);
                mats.Kd[i] = generated.Kd;
                mats.Ni[i] = generated.Ni;
                mats.Tf[i] = generated.Tf;
                mats.sssAlbedo[i] = generated.sssAlbedo;
                mats.sssRadius[i] = generated.sssRadius;
                mats.sssPhaseG[i] = generated.sssPhaseG;
                mats.sssWeight[i] = generated.sssWeight;
                mats.sssEnable[i] = generated.sssEnable;
                BroadcastWaterMaterial(scene, (UINT)i);
                scene.oceanMaterialEdited = false;
                scene.MarkMaterialsDirty();
            }
        } else {
            ImGui::TextDisabled("Generated from the optics above; any edit here takes it over.");
        }

        matChanged |= ImGui::DragFloat3("Absorption (1/m)", &mats.Tf[i].x, 0.001f, 0.0f, 100.0f, "%.4f");
        ImGui::SetItemTooltip("Red, green and blue absorption per metre.\n"
                              "0 = no absorption; higher values absorb that channel faster.");
        matChanged |= ImGui::DragFloat("IOR", &mats.Ni[i], 0.001f, 1.0f, 2.0f, "%.3f");
        ImGui::SetItemTooltip("1.333 is sea water at visible wavelengths.");

        bool en = mats.sssEnable[i] != 0u;
        if (ImGui::Checkbox("Water volume scattering", &en)) {
            mats.sssEnable[i] = en ? 1u : 0u;
            matChanged = true;
        }
        matChanged |= ImGui::ColorEdit3("Scattering color", &mats.sssAlbedo[i].x, ImGuiColorEditFlags_Float);
        ImGui::SetItemTooltip("Relative RGB scattering coefficients in the water volume.\n"
                              "One volume event redirects the path; absorption determines the depth color.");
        matChanged |= ImGui::SliderFloat("Scattering mean free path (m)", &mats.sssRadius[i], 0.0005f, 1000.0f,
                                         "%.4f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
        ImGui::SetItemTooltip("Mean free path in metres for the strongest color channel. Larger values mean\n"
                              "clearer water; thickness comes from the traced geometry.");
        matChanged |= ImGui::SliderFloat("Scattering forward g", &mats.sssPhaseG[i], -0.95f, 0.95f, "%.2f",
                                         ImGuiSliderFlags_AlwaysClamp);
        ImGui::SetItemTooltip("Higher values concentrate volume scattering in the forward direction.");
        matChanged |= ImGui::SliderFloat("Scattering density", &mats.sssWeight[i], 0.0f, 1.0f, "%.2f",
                                         ImGuiSliderFlags_AlwaysClamp);
        ImGui::SetItemTooltip("Scales the scattering coefficient, independently of absorption.\n"
                              "0 = absorption only; higher values increase underwater haze.\n"
                              "Surface Fresnel reflectance is unchanged.");

        if (matChanged) {
            scene.oceanMaterialEdited = true;
            BroadcastWaterMaterial(scene, (UINT)i);
            scene.MarkMaterialsDirty();
        }
    }

    ImGui::SeparatorText("Detail");
    ImGui::TextDisabled("tile budget %u of %u (restart to change)", p.maxTiles, (uint32_t)OCEAN_MAX_TILES);
    ImGui::SetItemTooltip("Tiles the surface may keep resident. Every one is an acceleration structure the\n"
                          "GPU rebuilds or refits each frame and an extra instance in the scene's top level,\n"
                          "so this is what the ocean costs in both time and memory. It sizes the geometry\n"
                          "buffers at load, which is why it cannot move now - set it on the scene's sea state.");
    cheap |= ImGui::SliderFloat("Tile size / distance", &p.lodFactor, 0.05f, 2.0f, "%.3f",
                                ImGuiSliderFlags_AlwaysClamp);
    ImGui::SetItemTooltip("Tile edge as a fraction of the distance to the camera. Lower is finer and costs\n"
                          "proportionally more acceleration-structure builds.");
    cheap |= ImGui::SliderFloat("Smallest tile (m)", &p.minTileSize, 1.0f, 256.0f, "%.0f",
                                ImGuiSliderFlags_AlwaysClamp);
    ImGui::SetItemTooltip("Every tile carries the same fixed grid, so this is really the size of the quads the\n"
                          "surface is built from - and the number of tiles it takes to cover the near field,\n"
                          "which is what traversal pays for. Waves shorter than a couple of quads are carried\n"
                          "by the BRDF anyway, so there is nothing to gain below that.");
    ImGui::TextDisabled("%.1f cm quads, %u triangles per tile", p.minTileSize / OCEAN_TILE_GRID * 100.0f,
                        (uint32_t)OCEAN_TILE_TRIS);
    cheap |= ImGui::SliderFloat("Near keep radius (m)", &p.nearKeepRadius, 0.0f, 4000.0f, "%.0f",
                                ImGuiSliderFlags_AlwaysClamp);
    ImGui::SetItemTooltip("Tiles nearer than this survive the frustum cull, so reflections and shadows of\n"
                          "nearby waves stay correct.");
    cheap |= ImGui::SliderFloat("Ray footprint", &p.filterScale, 0.1f, 4.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
    ImGui::SetItemTooltip("Scales the footprint that splits wave detail between geometry and the BRDF. Raise\n"
                          "it if the horizon shimmers, lower it if the near surface looks over-smoothed.");
    cheap |= ImGui::SliderFloat("Sun highlight roughness", &p.sunLobeRoughness, 0.0f, 0.5f, "%.3f",
                                ImGuiSliderFlags_AlwaysClamp);
    ImGui::SetItemTooltip("Lobe width the sun sampler and next-event estimation widen the water to, and nothing\n"
                          "else - continuation rays, environment reflections and the reconstruction guides keep\n"
                          "the authored roughness and the full-resolution wave normal.\n"
                          "Clear water is mirror-flat, which leaves direct lighting a delta lobe: the glitter\n"
                          "track then arrives as isolated fireflies instead of a sun path. Lower is a sharper,\n"
                          "sparklier highlight and more noise; higher is a softer, calmer one. 0 hands the sun\n"
                          "sampler the true delta lobe.");
    float extentKm = p.extent / 1000.0f;
    if (ImGui::SliderFloat("Extent (km)", &extentKm, 1.0f, 200.0f, "%.0f", ImGuiSliderFlags_AlwaysClamp)) {
        p.extent = extentKm * 1000.0f;
        cheap = true;
    }
    ImGui::SetItemTooltip("Half-width of the simulated ocean; the quadtree root spans twice this.");
    cheap |= ImGui::Checkbox("Earth curvature", &p.curvature);
    ImGui::SetItemTooltip("Without it the horizon sits at infinity and distant ships never drop below it.");

    ImGui::SeparatorText("Simulation");
    cheap |= ImGui::Checkbox("Pause", &p.paused);
    int seed = (int)p.seed;
    if (ImGui::SliderInt("Seed", &seed, 0, 9999, "%d", ImGuiSliderFlags_AlwaysClamp)) {
        p.seed = (uint32_t)std::max(0, seed);
        respec = true;
    }
    ImGui::SetItemTooltip("Picks a different realisation of the same sea state.");
    static const char* kDebugModes[] = {"Beauty",    "Normals",   "Compression",  "Covariance",
                                        "Roughness", "Foam", "Mip level"};
    int debug = (int)std::min<uint32_t>(p.debugMode, 6u);
    if (ImGui::Combo("Debug view", &debug, kDebugModes, IM_ARRAYSIZE(kDebugModes))) {
        p.debugMode = (uint32_t)std::clamp(debug, 0, 6);
        cheap = true;
    }

    if (ImGui::CollapsingHeader("Statistics")) {
        ImGui::Text("tiles %u of %u leaves, %u dropped", st.tiles, st.leaves, st.dropped);
        ImGui::Text("builds %u, refits %u | %.2fM triangles", st.builds, st.refits, (double)st.triangles / 1.0e6);
        ImGui::Text("BLAS %.1f MB | resources %.1f MB", (double)st.blasBytes / (1024.0 * 1024.0),
                    (double)st.resourceBytes / (1024.0 * 1024.0));
        ImGui::Text("slope variance: spectrum %.4f, Cox-Munk %.4f", st.slopeVarSpectrum, st.slopeVarCoxMunk);
        ImGui::SetItemTooltip("The synthesised spectrum is calibrated against the measured Cox & Munk totals;\n"
                              "the gap is the energy above the finest cascade's Nyquist limit.");
        ImGui::Text("conditioning gain %.3f | bake %.0f ms", st.conditioningGain, st.bakeMs);
    }

    // Respectral changes re-bake four 1024^2 cascades on the CPU, so they are staged while a
    // widget is held and committed once it is let go. The cheap ones land the same frame.
    if (m_waterRespecPending || respec) {
        m_waterRespecPending = true;
        if (!ImGui::IsAnyItemActive()) {
            oceanSystem.Configure(p);
            m_waterRespecPending = false;
        }
    } else if (cheap) {
        oceanSystem.Configure(p);
    }

    ImGui::End();
}

void Editor::Render(ID3D12GraphicsCommandList* cmdList) {
    if (!m_visible)
        return;
    ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), cmdList);
}

void Editor::RenderPlatformWindows() {
    if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
        ImGui::UpdatePlatformWindows();
        ImGui::RenderPlatformWindowsDefault();
    }
}

void Editor::DrawScenePanel(Scene& scene) {
    SetInitialPanelPosition(ImVec2(10, 30));
    ImGui::SetNextWindowSize(ImVec2(360, 450), ImGuiCond_FirstUseEver);

    if (!ImGui::Begin("Scene###Scene Hierarchy", &m_showScene)) {
        ImGui::End();
        return;
    }

    for (int mi = 0; mi < (int)scene.models.size(); ++mi) {
        auto& model = scene.models[mi];
        bool selected = (m_selectedModel == mi);

        char label[256];
        snprintf(label, sizeof(label), "%s  (%u meshes, %u instances)##model%d", model.name.c_str(), model.meshCount,
                 model.instanceCount, mi);

        if (ImGui::Selectable(label, selected))
            m_selectedModel = mi;
    }

    ImGui::Separator();

    if (m_selectedModel >= 0 && m_selectedModel < (int)scene.models.size()) {
        auto& model = scene.models[m_selectedModel];
        ImGui::Text("Edit: %s", model.name.c_str());
        ImGui::TextDisabled("File: %s", model.filePath.c_str());

        bool changed = false;
        changed |= ImGui::DragFloat3("Position", &model.position.x, 0.05f);
        changed |= ImGui::DragFloat3("Rotation", &model.rotation.x, 0.5f);
        changed |= ImGui::DragFloat3("Scale", &model.scale.x, 0.01f, 0.001f, 100.0f);

        if (changed) {
            scene.MarkModelMoved((UINT)m_selectedModel);
        }

        ImGui::Separator();
        ImGui::TextDisabled("%u meshes | %u instances", model.meshCount, model.instanceCount);

        if (m_cachedMatModel != m_selectedModel) {
            m_cachedMatModel = m_selectedModel;
            std::unordered_set<UINT> seen;
            m_cachedUniqueMats.clear();
            for (UINT i = model.meshStart; i < model.meshStart + model.meshCount; ++i) {
                if (i >= scene.meshes.size())
                    break;
                for (UINT mid : scene.meshes[i].cpuMaterialIDs) {
                    if (seen.insert(mid).second)
                        m_cachedUniqueMats.push_back(mid);
                }
            }
        }
        if (!m_cachedUniqueMats.empty()) {
            ImGui::Text("Materials (%zu):", m_cachedUniqueMats.size());
            for (UINT mid : m_cachedUniqueMats) {
                char btn[32];
                snprintf(btn, sizeof(btn), "Mat %u", mid);
                if (ImGui::SmallButton(btn)) {
                    m_selectedMat = (int)mid;
                    m_showMaterials = true;
                }
                if (ImGui::GetItemRectMax().x + 90.0f <
                    ImGui::GetWindowPos().x + ImGui::GetWindowWidth() - ImGui::GetStyle().WindowPadding.x)
                    ImGui::SameLine();
            }
            ImGui::NewLine();
        }

        if (ImGui::TreeNode("Sub-instances")) {
            for (UINT i = model.instanceStart; i < model.instanceStart + model.instanceCount; ++i) {
                if (i >= scene.instances.size())
                    break;
                auto& inst = scene.instances[i];
                ImGui::TextDisabled("[%u] %s (mesh %u)", i, inst.name.c_str(), inst.meshIndex);
            }
            ImGui::TreePop();
        }
    }

    ImGui::End();
}

void Editor::DrawCameraPanel(Camera& camera, FlyCamController& flyCam) {
    SetInitialPanelPosition(ImVec2(10, 490));
    ImGui::SetNextWindowSize(ImVec2(360, 180), ImGuiCond_FirstUseEver);

    if (!ImGui::Begin("Camera", &m_showCamera)) {
        ImGui::End();
        return;
    }

    ImGui::DragFloat("FOV", &camera.fovDegrees, 0.5f, 10.0f, 170.0f);
    ImGui::SliderFloat("Move Speed", &flyCam.moveSpeed, 0.01f, 1000000.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
    ImGui::DragFloat("Mouse Sensitivity", &flyCam.mouseSensitivity, 0.01f, 0.01f, 2.0f, "%.2f");

    if (ImGui::Button("Reset Camera")) {
        camera.ResetView();
        flyCam.Reset();
    }

    ImGui::Separator();
    ImGui::TextUnformatted("Depth of Field");
    ImGui::DragFloat("Aperture Radius", &camera.apertureRadius, 0.001f, 0.0f, 1.0f, "%.4f");
    ImGui::DragFloat("Focus Distance", &camera.focusDistance, 0.05f, 0.01f, 10000.0f, "%.3f",
                     ImGuiSliderFlags_Logarithmic);

    if (ImGui::CollapsingHeader("Clipping")) {
        ImGui::DragFloat("Near plane", &camera.nearPlane, 0.001f, 0.001f, 10.0f, "%.3f");
        ImGui::DragFloat("Far plane", &camera.farPlane, 1000.0f, 100.0f, 1.0e9f, "%.0f");
    }

    ImGui::End();
}

void Editor::DrawPassPipelinePanel(PassSystem& passes) {
    SetInitialPanelPosition(ImVec2(380, 30));
    ImGui::SetNextWindowSize(ImVec2(350, 400), ImGuiCond_FirstUseEver);

    if (!ImGui::Begin("Render passes###Pass Pipeline", &m_showPipeline)) {
        ImGui::End();
        return;
    }
    ImGui::Checkbox("Show inactive passes", &m_showInactivePasses);
    ImGui::TextDisabled("Previous frame");

    const char* stageNames[] = {"RayGen", "Compute", "FixedCompute", "Barrier", "LoopStart", "LoopEnd", "DLSS"};

    for (size_t i = 0; i < passes.Passes().size(); ++i) {
        auto& p = passes.Passes()[i];
        if (p.stage == Stage::Barrier || (!m_showInactivePasses && !p.executedLastFrame))
            continue;
        int stageIdx = static_cast<int>(p.stage);
        const char* stageName = (stageIdx < _countof(stageNames)) ? stageNames[stageIdx] : "?";

        ImVec4 color(0.8f, 0.8f, 0.8f, 1.0f);
        switch (p.stage) {
        case Stage::RayGen:
            color = ImVec4(0.3f, 0.9f, 0.3f, 1);
            break;
        case Stage::Compute:
            color = ImVec4(0.3f, 0.6f, 0.9f, 1);
            break;
        case Stage::Barrier:
            color = ImVec4(0.6f, 0.6f, 0.6f, 1);
            break;
        case Stage::DLSS:
            color = ImVec4(0.9f, 0.6f, 0.2f, 1);
            break;
        case Stage::LoopStart:
        case Stage::LoopEnd:
            color = ImVec4(0.9f, 0.9f, 0.3f, 1);
            break;
        default:
            break;
        }

        if (!p.executedLastFrame)
            color = ImVec4(0.4f, 0.4f, 0.4f, 1);
        ImGui::PushStyleColor(ImGuiCol_Text, color);
        if (p.file.empty()) {
            ImGui::Text("[%2zu] %s", i, stageName);
        } else {
            char fileStr[256];
            WideCharToMultiByte(CP_UTF8, 0, p.file.c_str(), -1, fileStr, 256, nullptr, nullptr);
            ImGui::Text("[%2zu] %s: %s", i, stageName, fileStr);
            if (p.stage == Stage::Compute)
                ImGui::SameLine(), ImGui::TextDisabled("(%ux%u)", p.groupX, p.groupY);
        }
        ImGui::PopStyleColor();
    }

    ImGui::End();
}

void Editor::DrawDLSSPanel(Camera& camera, DLSSManager& dlss, DLSSGSettings& dlssG) {
    ImGui::SetNextWindowSize(ImVec2(390, 380), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("DLSS", &m_showDLSS)) {
        ImGui::End();
        return;
    }
    ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.46f);
    const char* labels[] = {"Off", "DLAA", "Quality", "Balanced", "Performance", "Ultra performance"};
    const sl::DLSSMode modes[] = {
        sl::DLSSMode::eOff,      sl::DLSSMode::eDLAA,           sl::DLSSMode::eMaxQuality,
        sl::DLSSMode::eBalanced, sl::DLSSMode::eMaxPerformance, sl::DLSSMode::eUltraPerformance};
    int selected = 0;
    for (int i = 0; i < IM_ARRAYSIZE(modes); ++i)
        if (dlss.mode == modes[i])
            selected = i;
    if (ImGui::Combo("Ray reconstruction", &selected, labels, IM_ARRAYSIZE(labels)))
        dlss.mode = modes[selected];
    ImGui::TextDisabled("%u x %u -> %u x %u", dlss.RenderWidth(), dlss.RenderHeight(), dlss.DisplayWidth(),
                        dlss.DisplayHeight());
    ImGui::BeginDisabled(dlss.mode == sl::DLSSMode::eOff);
    ImGui::SliderFloat("Sharpness", &dlss.sharpness, 0.0f, 1.0f, "%.2f");
    if (ImGui::CollapsingHeader("Reconstruction tuning")) {
        const char* presets[] = {"Default", "D", "E", "F"};
        const uint32_t values[] = {0, 4, 5, 6};
        int preset = 0;
        for (int i = 0; i < IM_ARRAYSIZE(values); ++i)
            if ((uint32_t)dlss.rrPresets[DLSSManager::kPresetDLAA] == values[i])
                preset = i;
        if (ImGui::Combo("Model", &preset, presets, IM_ARRAYSIZE(presets))) {
            dlss.rrLinkPresets = true;
            for (auto& p : dlss.rrPresets)
                p = (sl::DLSSDPreset)values[preset];
        }
        ImGui::SliderFloat("Temporal response (rough)", &dlss.rrResponsivityRough, -1.0f, 1.0f, "%.3f",
                           ImGuiSliderFlags_AlwaysClamp);
        if (ImGui::IsItemDeactivatedAfterEdit())
            dlss.ForceReset();
        ImGui::SetItemTooltip("How readily reconstruction drops accumulated history: -1 accumulates the longest,\n"
                              "+1 is the most responsive. The mask is written per pixel, and an ordinary surface\n"
                              "interpolates between this value at roughness 1 and the mirror one at roughness 0.\n"
                              "Rough shading barely changes between frames, so it accumulates freely.");
        ImGui::SliderFloat("Temporal response (mirror)", &dlss.rrResponsivityMirror, -1.0f, 1.0f, "%.3f",
                           ImGuiSliderFlags_AlwaysClamp);
        if (ImGui::IsItemDeactivatedAfterEdit())
            dlss.ForceReset();
        ImGui::SetItemTooltip("The other end of that ramp, at roughness 0: a sharp reflection slides across a\n"
                              "smooth surface as the camera moves, so it cannot lean on history as hard.");
        ImGui::SliderFloat("Temporal response (water)", &dlss.rrWaterResponsivity, -1.0f, 1.0f, "%.3f",
                           ImGuiSliderFlags_AlwaysClamp);
        if (ImGui::IsItemDeactivatedAfterEdit())
            dlss.ForceReset();
        ImGui::SetItemTooltip("Flat value for pixels whose primary ray landed on the ocean, off the ramp above. A\n"
                              "moving sea presents a different set of crests every frame, so the history length\n"
                              "that resolves a static surface blends its sun glitter into streaks and\n"
                              "misestimates it; water wants the responsive end while the scene keeps\n"
                              "accumulating. All three at 0 leaves the mask unbound and reconstruction uses its\n"
                              "own default.");
        ImGui::SliderFloat("Camera jitter", &camera.jitterScale, 0.0f, 1.0f, "%.3f");
        if (ImGui::IsItemDeactivatedAfterEdit())
            dlss.ForceReset();
        ImGui::SetItemTooltip("1 uses the full subpixel sampling pattern; 0 disables jitter.");
        if (ImGui::Checkbox("Limit emitter spikes", &dlss.clampEmitterSpikes))
            dlss.ForceReset();
        if (ImGui::Button("Reset reconstruction history"))
            dlss.ForceReset();
    }
    ImGui::EndDisabled();
    ImGui::SeparatorText("Frame generation");
    ImGui::BeginDisabled(!dlssG.available);
    ImGui::Checkbox("Enabled", &dlssG.enabled);
    if (dlssG.available && dlssG.enabled) {
        const char* multipliers[] = {"2x", "3x", "4x"};
        const int count = std::clamp((int)dlssG.maxFrames, 1, IM_ARRAYSIZE(multipliers));
        int multiplier = std::clamp((int)dlssG.framesToGenerate - 1, 0, count - 1);
        if (ImGui::Combo("Multiplier", &multiplier, multipliers, count))
            dlssG.framesToGenerate = multiplier + 1;
    }
    ImGui::EndDisabled();
    if (!dlssG.available)
        ImGui::TextDisabled("Unavailable on this GPU");
    ImGui::PopItemWidth();
    ImGui::End();
}

void Editor::PerformanceHistory::push(const FrameStats& fs) {
    const int i = write;
    frameCpuMs[i] = fs.cpuFrameMs;
    frameGpuMs[i] = fs.gpuTimingsValid ? fs.gpuFrameMs : 0.0f;
    streamingCpuMs[i] = fs.cpuStreamingMs;
    write = (write + 1) % N;
    if (filled < N)
        ++filled;
}

namespace {
float historyMax(const float* values, int count) {
    float result = 0.0f;
    for (int i = 0; i < count; ++i)
        result = std::max(result, values[i]);
    return result;
}

float historyAverage(const float* values, int count) {
    if (count == 0)
        return 0.0f;
    double total = 0.0;
    for (int i = 0; i < count; ++i)
        total += values[i];
    return static_cast<float>(total / count);
}

float historyLast(const float* values, int write, int count) {
    if (count == 0)
        return 0.0f;
    return values[(write + Editor::PerformanceHistory::N - 1) % Editor::PerformanceHistory::N];
}

void plotHistory(const char* id, const char* label, const float* values, int count, int offset, int write,
                 ImVec4 color) {
    const float maximum = historyMax(values, count);
    char overlay[96];
    snprintf(overlay, sizeof(overlay), "last %.3f | avg %.3f | max %.3f ms", historyLast(values, write, count),
             historyAverage(values, count), maximum);
    ImGui::TextUnformatted(label);
    ImGui::PushStyleColor(ImGuiCol_PlotLines, color);
    ImGui::PlotLines(id, values, count, offset, overlay, 0.0f, maximum > 0.0f ? maximum * 1.1f : 1.0f,
                     ImVec2(-1.0f, 58.0f));
    ImGui::PopStyleColor();
}

void showBytes(const char* label, UINT64 bytes) {
    ImGui::Text("%s: %.2f MiB", label, static_cast<double>(bytes) / (1024.0 * 1024.0));
    ImGui::SetItemTooltip("%s: %llu bytes", label, static_cast<unsigned long long>(bytes));
}

void showByteUsage(const char* label, UINT64 used, UINT64 capacity) {
    ImGui::Text("%s: %.2f / %.2f MiB", label, static_cast<double>(used) / (1024.0 * 1024.0),
                static_cast<double>(capacity) / (1024.0 * 1024.0));
    ImGui::SetItemTooltip("%s: %llu / %llu bytes (used / capacity)", label, static_cast<unsigned long long>(used),
                          static_cast<unsigned long long>(capacity));
}
}

void Editor::DrawPerformancePanel(const planet::StreamOrchestrator::Stats& ps, const FrameStats& fs, float fps,
                                  const mc::StreamerStats* minecraft) {
    SetInitialPanelPosition(ImVec2(20, 60));
    ImGui::SetNextWindowSize(ImVec2(650, 760), ImGuiCond_FirstUseEver);

    if (!ImGui::Begin("Performance###Performance", &m_showPerformance)) {
        ImGui::End();
        return;
    }

    ImGui::TextDisabled("Latest CPU and completed GPU measurements");
    ImGui::Text("%.1f fps | CPU %.2f ms", fps, fs.cpuFrameMs);
    ImGui::SameLine();
    if (fs.gpuTimingsValid)
        ImGui::Text("| GPU %.2f ms", fs.gpuFrameMs);
    else
        ImGui::TextDisabled("| GPU unavailable");
    ImGui::SameLine();
    ImGui::Checkbox("Pause", &m_performanceHistory.paused);
    ImGui::SameLine();
    if (ImGui::Button("Clear")) {
        const bool paused = m_performanceHistory.paused;
        m_performanceHistory = PerformanceHistory{};
        m_performanceHistory.paused = paused;
    }
    const int count = m_performanceHistory.filled;
    const int offset = count < PerformanceHistory::N ? 0 : m_performanceHistory.write;
    const int write = m_performanceHistory.write;

    ImGui::TextDisabled("Direct queue timings exclude async queues and presentation.");

    if (ImGui::CollapsingHeader("Frame history")) {
        plotHistory("##frameCpu", "CPU frame", m_performanceHistory.frameCpuMs, count, offset, write,
                    ImVec4(0.40f, 0.85f, 0.40f, 1.0f));
        plotHistory("##frameGpu", "GPU frame", m_performanceHistory.frameGpuMs, count, offset, write,
                    ImVec4(0.95f, 0.55f, 0.20f, 1.0f));
        plotHistory("##streamingCpu", "CPU streaming", m_performanceHistory.streamingCpuMs, count, offset, write,
                    ImVec4(0.30f, 0.70f, 1.00f, 1.0f));
        ImGui::Text("CPU GPU wait: %.3f ms", fs.gpuWaitMs);
    }

    if (ImGui::BeginTabBar("##performanceViews")) {
        if (ImGui::BeginTabItem("GPU passes")) {
            if (!fs.gpuTimingsValid || fs.gpuPasses.empty()) {
                ImGui::TextDisabled("No valid pass timings for this frame.");
            } else {
                if (fs.gpuTimingsTruncated)
                    ImGui::TextDisabled("Pass list truncated at the profiler limit.");
                const float totalMs = fs.gpuFrameMs > 0.0f ? fs.gpuFrameMs : [&] {
                    float total = 0.0f;
                    for (const GpuPassTiming& pass : fs.gpuPasses)
                        total += pass.gpuMs;
                    return total;
                }();
                if (ImGui::BeginTable("##gpuPasses", 4,
                                      ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                          ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_ScrollY |
                                          ImGuiTableFlags_Resizable,
                                      ImVec2(0, std::max(160.0f, ImGui::GetContentRegionAvail().y)))) {
                    ImGui::TableSetupColumn("Pass");
                    ImGui::TableSetupColumn("Last ms", ImGuiTableColumnFlags_WidthFixed, 72.0f);
                    ImGui::TableSetupColumn("Calls", ImGuiTableColumnFlags_WidthFixed, 56.0f);
                    ImGui::TableSetupColumn("% total", ImGuiTableColumnFlags_WidthFixed, 72.0f);
                    ImGui::TableSetupScrollFreeze(0, 1);
                    ImGui::TableHeadersRow();
                    for (const GpuPassTiming& pass : fs.gpuPasses) {
                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::TextUnformatted(pass.name.c_str());
                        ImGui::SetItemTooltip("%s", pass.name.c_str());
                        ImGui::TableNextColumn();
                        ImGui::Text("%.3f", pass.gpuMs);
                        ImGui::TableNextColumn();
                        ImGui::Text("%u", pass.calls);
                        ImGui::TableNextColumn();
                        ImGui::Text("%.1f%%", totalMs > 0.0f ? 100.0f * pass.gpuMs / totalMs : 0.0f);
                    }
                    ImGui::EndTable();
                }
            }

            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Minecraft / BVHs")) {
            ImGui::SeparatorText("Minecraft geometry");
            if (minecraft) {
                ImGui::Text("Selected chunks: %u | resident: %u | tracked: %u", minecraft->desired, minecraft->ready,
                            minecraft->chunksTracked);
                ImGui::TextDisabled("Resident counts sampled %u frames ago", minecraft->censusAgeFrames);
                if (ImGui::BeginTable("##geometryCounts", 3,
                                      ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
                    ImGui::TableSetupColumn("Geometry");
                    ImGui::TableSetupColumn("In scene BVH");
                    ImGui::TableSetupColumn("Resident");
                    ImGui::TableHeadersRow();
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted("Chunk BLAS");
                    ImGui::TableNextColumn();
                    ImGui::Text("%u", minecraft->geometryBlasRendered);
                    ImGui::TableNextColumn();
                    ImGui::Text("%u", minecraft->geometryBlasResident);
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::TextUnformatted("Triangles");
                    ImGui::TableNextColumn();
                    ImGui::Text("%llu", static_cast<unsigned long long>(minecraft->trianglesInTlas));
                    ImGui::TableNextColumn();
                    ImGui::Text("%llu", static_cast<unsigned long long>(minecraft->trianglesResident));
                    ImGui::EndTable();
                }
                showByteUsage("Geometry buffers", minecraft->geometryBytesUsed, minecraft->geometryBytesCapacity);
                showByteUsage("BLAS pool", minecraft->blasBytesUsed, minecraft->blasBytesCapacity);
                showByteUsage("BLAS build pool", minecraft->blasBuildBytesUsed, minecraft->blasBuildBytesCapacity);
                showByteUsage("Scratch ring high-water", minecraft->scratchBytesUsed, minecraft->scratchBytesCapacity);

                ImGui::SeparatorText("Streaming");
                ImGui::Text("Pending %u | meshing %u | meshed %u | uploading %u", minecraft->pending,
                            minecraft->meshing, minecraft->meshed, minecraft->uploading);
                ImGui::Text("This frame: %u BLAS builds | %u compactions | %u copies", minecraft->buildsThisFrame,
                            minecraft->compactionsThisFrame, minecraft->copiesThisFrame);
                showBytes("Uploaded this frame", minecraft->uploadBytesThisFrame);
                ImGui::Text("CPU selection %.3f ms | recording %.3f ms", minecraft->selectMs, minecraft->recordMs);
                ImGui::Text("Light selection: %.3f ms CPU (%s)", minecraft->lightSelectionMs,
                            minecraft->lightSelectionReused ? "reused" : "updated");
                ImGui::Text("Mesh job average %.3f ms | allocation retries %u", minecraft->meshMsAvg,
                            minecraft->allocFailures);
            } else {
                ImGui::TextDisabled("No Minecraft world loaded.");
            }

            ImGui::SeparatorText("Shared scene BVH (TLAS)");
            ImGui::Text("Instances: %u / %u | %s", ps.tlas_instances, ps.tlas_instance_capacity,
                        ps.tlas_build_recorded ? "rebuilt this frame" : "reused this frame");
            showBytes("TLAS allocation", ps.tlas_result_bytes);
            showBytes("TLAS scratch", ps.tlas_scratch_bytes);
            if (ps.gpu_timing_valid) {
                ImGui::Text("Minecraft builds + compaction: %.3f ms GPU", ps.external_blas_gpu_ms);
                ImGui::Text("Scene TLAS: %.3f ms GPU (%s, %u instances)", ps.tlas_gpu_ms,
                            ps.gpu_timing_tlas_build_recorded ? "rebuilt" : "reused", ps.gpu_timing_tlas_instances);
                ImGui::TextDisabled("Compute queue sample: %u frames old; excludes copy queue",
                                    ps.gpu_timing_sample_age);
            } else {
                ImGui::TextDisabled("Waiting for completed BVH GPU timings.");
            }

            ImGui::SeparatorText("Published light BVH");
            ImGui::Text("TLAS nodes %u | slot entries %u | Minecraft leaves %u", fs.lightBvh.nodes, fs.lightBvh.slots,
                        fs.lightBvh.voxelLeaves);
            showBytes("TLAS node allocation", fs.lightBvh.nodeBytes);
            showBytes("Slot allocation", fs.lightBvh.slotBytes);
            showBytes("Traversal trails", fs.lightBvh.trailBytes);
            if (fs.lightBvh.buildMeasured) {
                ImGui::Text("Last worker update: %.3f ms CPU (%s)", fs.lightBvh.buildCpuMs,
                            fs.lightBvh.incremental ? "incremental" : "full rebuild");
            } else {
                ImGui::TextDisabled("No completed light BVH worker timing.");
            }
            if (fs.lightBvh.pending)
                ImGui::TextDisabled("Next light BVH update pending");
            if (minecraft &&
                ImGui::BeginTable("##lightCounts", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
                ImGui::TableSetupColumn("Minecraft lights");
                ImGui::TableSetupColumn("Published tree");
                ImGui::TableSetupColumn("Resident");
                ImGui::TableHeadersRow();
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted("BLAS nodes");
                ImGui::TableNextColumn();
                ImGui::Text("%llu", static_cast<unsigned long long>(minecraft->lightNodesInTree));
                ImGui::TableNextColumn();
                ImGui::Text("%llu", static_cast<unsigned long long>(minecraft->lightNodesResident));
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted("Triangles");
                ImGui::TableNextColumn();
                ImGui::Text("%llu", static_cast<unsigned long long>(minecraft->lightTrisInTree));
                ImGui::TableNextColumn();
                ImGui::Text("%llu", static_cast<unsigned long long>(minecraft->lightTrisResident));
                ImGui::EndTable();
                ImGui::Text("Published chunks %u | light data dropped from %u chunks", minecraft->lightChunksInTree,
                            minecraft->lightChunksDropped);
            }
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }

    ImGui::End();
}
void Editor::DrawDLSSNRPanel(DLSSNRManager& nr) {
    SetInitialPanelPosition(ImVec2(380, 60));
    ImGui::SetNextWindowSize(ImVec2(450, 470), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("DLSS 5 Neural Rendering", &m_showDLSSNR)) {
        ImGui::End();
        return;
    }
    const auto& status = nr.GetStatus();

    ImGui::TextWrapped("%s", status.backendText.c_str());
    const bool available = status.backend != DLSSNRManager::BackendState::eStubNoSdk &&
                           status.backend != DLSSNRManager::BackendState::eRuntimeMissing;
    ImGui::BeginDisabled(!available);
    ImGui::Checkbox("Enable", &nr.settings.enabled);
    static const char* modelStyles[] = {"Default", "Natural", "Cinematic"};
    static const char* renderPresets[] = {"Default", "Preset 1", "Preset 2", "Preset 3"};
    static_assert(IM_ARRAYSIZE(modelStyles) == dlssnr::kModelStyleCount);
    static_assert(IM_ARRAYSIZE(renderPresets) == dlssnr::kRenderPresetCount);
    ImGui::Combo("Model style", &nr.settings.modelStyle, modelStyles, IM_ARRAYSIZE(modelStyles));
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Selects the NR rendering style. Changing it restarts NR and clears its history.");
    ImGui::Combo("Render preset", &nr.settings.renderPreset, renderPresets, IM_ARRAYSIZE(renderPresets));
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "Requests an embedded NR model preset. A preset absent from the runtime may use its default model.");
    ImGui::EndDisabled();

    ImGui::BeginDisabled(!nr.settings.enabled);
    ImGui::SliderFloat("Intensity", &nr.settings.intensity, 0.0f, 1.0f, "%.2f");
    ImGui::SliderFloat("Local Tone", &nr.settings.localToneStrength, 0.0f, 1.0f, "%.2f");
    ImGui::SliderFloat("Local Structure", &nr.settings.localStructureStrength, 0.0f, 1.0f, "%.2f");
    ImGui::SliderFloat("Skin Structure", &nr.settings.skinStructureStrength, -1.0f, 1.0f, "%.2f");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("-1 uses the model default.");
    ImGui::Checkbox("Auto Skin Mask", &nr.settings.useAutoMask);
    if (ImGui::Button("Reset history"))
        nr.ForceReset();
    ImGui::EndDisabled();

    ImGui::Text("Successful evaluations: %llu", (unsigned long long)status.evalCount);
    if (!status.lastResult.empty())
        ImGui::TextWrapped("%s", status.lastResult.c_str());
    if (ImGui::TreeNode("Runtime details")) {
        ImGui::TextWrapped("Path: %s", status.runtimePath.c_str());
        ImGui::Text("Version: %s", status.runtimeVersion.c_str());
        ImGui::TextWrapped("%s", status.signatureText.c_str());

        ImGui::TreePop();
    }
    ImGui::End();
}

void Editor::DrawMaterialInspector(Scene& scene, Camera& camera, IntegratorSettings& restir, mc::VoxelStreamer* voxels) {
    SetInitialPanelPosition(ImVec2(740, 30));
    ImGui::SetNextWindowSize(ImVec2(620, 700), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(ImVec2(540, 300), ImVec2(FLT_MAX, FLT_MAX));

    if (!ImGui::Begin("Materials", &m_showMaterials)) {
        ImGui::End();
        return;
    }

    bool texInterp = (restir.texturePointFilter == 0);
    if (ImGui::Checkbox("Texture interpolation", &texInterp))
        restir.texturePointFilter = texInterp ? 0 : 1;
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Off = nearest-texel point sampling for every texture\n"
                          "(crisp pixel-art / Minecraft look). On = bilinear/aniso.");

    if (ImGui::CollapsingHeader("Preview overrides"))
        ImGui::Checkbox("Diffuse materials only", &restir.forceDiffuseMats);

    if (voxels && voxels->world()) {
        bool hideWater = voxels->hide_water();
        if (ImGui::Checkbox("Replace world water with the wave surface", &hideWater))
            voxels->set_hide_water(hideWater);
        ImGui::SetItemTooltip(
            "Leaves every water block out of the world's mesh, so the renderer's own wave surface\n"
            "is what a ray meets instead of a stack of flat block tops. Place that surface at the\n"
            "water line with the sea level in Experimental > Water.\n"
            "Anything else the world marks as water goes with it, so a waterfall or a cauldron\n"
            "empties too. Toggling rebuilds every resident chunk, which streams back in over a\n"
            "few seconds.");
    }
    ImGui::Separator();

    ImGui::SliderFloat("Global Emission", &camera.sunSettings.globalEmissionStrength, 0.0f, 10.0f, "%.2fx");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Scales every emissive material equally.\n1.0 = authored values.");
    ImGui::Separator();

    ImGui::BeginChild("MatList", ImVec2(180, 0), true);

    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::InputTextWithHint("##matFilter", "filter...", m_matFilter, sizeof(m_matFilter));

    auto toLower = [](char c) -> char { return (c >= 'A' && c <= 'Z') ? (char)(c - 'A' + 'a') : c; };
    auto containsCI = [&](const char* hay, const char* needle) -> bool {
        if (!needle || !*needle)
            return true;
        if (!hay)
            return false;
        const size_t nLen = strlen(needle);
        for (const char* p = hay; *p; ++p) {
            size_t j = 0;
            while (j < nLen && p[j] && toLower(p[j]) == toLower(needle[j]))
                ++j;
            if (j == nLen)
                return true;
        }
        return false;
    };

    bool numericQuery = false;
    int numericValue = 0;
    if (m_matFilter[0]) {
        numericQuery = true;
        for (const char* p = m_matFilter; *p; ++p) {
            if (*p < '0' || *p > '9') {
                numericQuery = false;
                break;
            }
            numericValue = numericValue * 10 + (*p - '0');
        }
    }

    // The ocean's block of materials is generated, not authored: it is edited from the water panel
    // (Experimental > Water), which broadcasts every change across the block. Listing 4096
    // near-identical slots here would only get in the way.
    const int waterMat = scene.oceanInstanceSlots && scene.oceanMatIndex < scene.materials.size()
        ? (int)scene.oceanMatIndex : -1;
    if (waterMat >= 0 && m_selectedMat >= waterMat && m_selectedMat < waterMat + OCEAN_MATERIAL_COUNT)
        m_selectedMat = -1;
    int matchCount = 0;
    auto drawMaterial = [&](int i) {
        const char* name = (i < (int)scene.materialNames.size() && !scene.materialNames[i].empty())
                               ? scene.materialNames[i].c_str()
                               : nullptr;

        if (m_matFilter[0]) {
            const bool nameMatch = containsCI(name, m_matFilter);
            const bool idxMatch = numericQuery && (i == numericValue);
            if (!nameMatch && !idxMatch)
                return;
        }
        ++matchCount;

        const XMFLOAT4& kd = scene.materials.Kd[i];
        ImVec4 preview(kd.x, kd.y, kd.z, 1.0f);
        ImGui::PushStyleColor(ImGuiCol_Text, preview);
        ImGui::Text("\xe2\x96\xa0");
        ImGui::PopStyleColor();
        ImGui::SameLine();
        char label[128];
        if (name)
            snprintf(label, sizeof(label), "%d %s##mat", i, name);
        else
            snprintf(label, sizeof(label), "%d##mat", i);
        if (ImGui::Selectable(label, m_selectedMat == i))
            m_selectedMat = i;
    };
    for (int i = 0; i < (int)scene.materials.size(); ++i) {
        if (waterMat >= 0 && i >= waterMat && i < waterMat + OCEAN_MATERIAL_COUNT)
            continue;
        drawMaterial(i);
    }

    if (m_matFilter[0] && matchCount == 0) {
        ImGui::TextDisabled("(no matches)");
    }

    ImGui::EndChild();

    ImGui::SameLine();

    ImGui::BeginChild("MatProps", ImVec2(0, 0), false);
    if (m_selectedMat >= 0 && m_selectedMat < (int)scene.materials.size()) {
        const int i = m_selectedMat;
        auto& mats = scene.materials;
        const char* matName = (i < (int)scene.materialNames.size() && !scene.materialNames[i].empty())
                                  ? scene.materialNames[i].c_str()
                                  : nullptr;
        if (matName)
            ImGui::Text("Material %d: %s", i, matName);
        else
            ImGui::Text("Material %d", i);
        ImGui::Separator();

        bool changed = false;
        bool emissionChanged = false;

        if (ImGui::CollapsingHeader("Surface", ImGuiTreeNodeFlags_DefaultOpen)) {
            changed |= ImGui::ColorEdit3("Albedo", &mats.Kd[i].x, ImGuiColorEditFlags_Float);
            changed |= ImGui::SliderFloat("Opacity", &mats.Kd[i].w, 0.0f, 1.0f, "%.3f");
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("0 = fully transparent (glass)\n1 = fully opaque");
            changed |= ImGui::DragFloat("IOR", &mats.Ni[i], 0.01f, 1.0f, 3.0f, "%.3f");
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Index of Refraction\n1.0 = air, 1.33 = water, 1.5 = glass");
        }

        if (ImGui::CollapsingHeader("Emission", ImGuiTreeNodeFlags_DefaultOpen)) {
            XMFLOAT3& Ke = mats.Ke[i];
            bool emEdit = ImGui::ColorEdit3("Emission", &Ke.x, ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR);
            if (emEdit) {
                changed = true;
                emissionChanged = true;
            }

            if (Ke.x > 0 || Ke.y > 0 || Ke.z > 0) {
                float intensity = std::max({Ke.x, Ke.y, Ke.z});
                float prevIntensity = intensity;
                if (ImGui::DragFloat("Intensity", &intensity, 0.1f, 0.0f, 1000.0f)) {
                    if (prevIntensity > 0.001f) {
                        float s = intensity / prevIntensity;
                        Ke.x *= s;
                        Ke.y *= s;
                        Ke.z *= s;
                        changed = true;
                        emissionChanged = true;
                    }
                }
            }
        }

        if (ImGui::CollapsingHeader("PBR", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (mats.diffuseRoughness.size() < mats.size())
                mats.diffuseRoughness.resize(mats.size(), .5f);
            changed |= ImGui::SliderFloat("Roughness", &mats.Pr_Pm_Ps_Pc[i].x, 0.0f, 1.0f);
            changed |= ImGui::SliderFloat("Diffuse roughness", &mats.diffuseRoughness[i], 0.0f, 1.0f);
            changed |= ImGui::SliderFloat("Metallic", &mats.Pr_Pm_Ps_Pc[i].y, 0.0f, 1.0f);
            changed |= ImGui::SliderFloat("Sheen", &mats.Pr_Pm_Ps_Pc[i].z, 0.0f, 1.0f);
        }

        if (ImGui::CollapsingHeader("Clearcoat")) {
            ImGui::PushID("coat");
            changed |= ImGui::SliderFloat("Strength", &mats.Pr_Pm_Ps_Pc[i].w, 0.0f, 1.0f);
            changed |= ImGui::SliderFloat("Roughness", &mats.Pcr_aniso_anisor[i].x, 0.0f, 1.0f);
            ImGui::PopID();
        }

        if (ImGui::CollapsingHeader("Anisotropy")) {
            ImGui::PushID("aniso");
            changed |= ImGui::SliderFloat("Strength", &mats.Pcr_aniso_anisor[i].y, -1.0f, 1.0f);
            changed |= ImGui::SliderFloat("Rotation", &mats.Pcr_aniso_anisor[i].z, 0.0f, 1.0f);
            ImGui::PopID();
        }

        if (ImGui::CollapsingHeader("Transmission")) {
            if (i >= (int)mats.thinGlass.size())
                mats.thinGlass.resize(i + 1, 0u);
            changed |= ImGui::ColorEdit3("Filter (Tf)", &mats.Tf[i].x,
                                         ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Volume absorption color (solid glass)\n"
                                  "Thin glass: flat per-surface transmission tint\nWhite = no absorption");

            bool thin = mats.thinGlass[i] != 0u;
            if (ImGui::Checkbox("Thin glass (Fresnel only)", &thin)) {
                mats.thinGlass[i] = thin ? 1u : 0u;
                changed = true;
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Single-surface glass: reflects a roughness-aware Fresnel lobe and transmits\n"
                                  "the rest straight through (no refraction, no medium/absorption). Shadow rays\n"
                                  "attenuate by (1-F)*Tf instead of blocking.\n"
                                  "Works live as long as Opacity < 1 (any transmissive material is already\n"
                                  "non-opaque geometry). A material that was fully opaque (Opacity = 1) at load\n"
                                  "needs a scene reload to become shadow-passable.");
        }

        if (ImGui::CollapsingHeader("Subsurface")) {
            if (i >= mats.sssEnable.size())
                mats.sssEnable.resize(i + 1, 0u);
            if (i >= mats.sssAlbedo.size())
                mats.sssAlbedo.resize(i + 1, DirectX::XMFLOAT3(1.0f, 1.0f, 1.0f));
            if (i >= mats.sssRadius.size())
                mats.sssRadius.resize(i + 1, 0.0f);
            if (i >= mats.sssPhaseG.size())
                mats.sssPhaseG.resize(i + 1, 0.0f);
            if (i >= mats.sssWeight.size())
                mats.sssWeight.resize(i + 1, 1.0f);

            bool en = mats.sssEnable[i] != 0u;
            if (ImGui::Checkbox("Enable SSS", &en)) {
                mats.sssEnable[i] = en ? 1u : 0u;
                changed = true;
            }
            changed |= ImGui::ColorEdit3("SSS Color", &mats.sssAlbedo[i].x, ImGuiColorEditFlags_Float);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "Single-scattering albedo (the inside colour).\nCarried by the random walk's albedo product.");
            changed |= ImGui::SliderFloat("Radius", &mats.sssRadius[i], 0.0005f, 50.0f, "%.4f",
                                          ImGuiSliderFlags_Logarithmic);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "Scatter distance / mean free path (world units), log scale.\nsigma_t = 1/radius. Small = "
                    "dense/opaque, large = translucent.\nUseful range is relative to the object's thickness.");
            changed |= ImGui::SliderFloat("Phase g", &mats.sssPhaseG[i], -0.95f, 0.95f);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Henyey-Greenstein anisotropy.\n0 = isotropic, >0 forward, <0 backward scattering.");
            changed |= ImGui::SliderFloat("Entry weight", &mats.sssWeight[i], 0.0f, 1.0f);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Probability scale for entering the medium vs reflecting.\np_enter = weight * "
                                  "Fresnel-transmittance. 0 = pure reflection, 1 = mostly subsurface.");
        }

        if (ImGui::CollapsingHeader("Alpha Test")) {
            changed |= ImGui::SliderFloat("Threshold", &mats.alphaThreshold[i], 0.0f, 1.0f);

            if (i >= mats.invertAlpha.size())
                mats.invertAlpha.resize(i + 1, 0u);
            bool inv = mats.invertAlpha[i] != 0u;
            if (ImGui::Checkbox("Invert (sample is transparency)", &inv)) {
                mats.invertAlpha[i] = inv ? 1u : 0u;
                changed = true;
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Top of the invert-alpha hierarchy (L3, manual override).\n"
                                  "Auto-detection at load: filename hint (L1) and brightness check (L2).\n"
                                  "Toggle on if the texture's alpha encodes 1 = transparent (map_Tr style).");
        }

        if (ImGui::CollapsingHeader("Textures")) {
            ImGui::TextDisabled("Assigned texture IDs:");
            ImGui::Text("  Albedo: %s",
                        mats.albedoTexID[i] >= 0 ? std::to_string(mats.albedoTexID[i]).c_str() : "none");
            ImGui::Text("  Normal: %s",
                        mats.normalTexID[i] >= 0 ? std::to_string(mats.normalTexID[i]).c_str() : "none");
            ImGui::Text("  RMA:    %s", mats.rmaTexID[i] >= 0 ? std::to_string(mats.rmaTexID[i]).c_str() : "none");
        }

        if (changed)
            scene.MarkMaterialsDirty(emissionChanged);
    } else {
        ImGui::TextDisabled("Select a material.");
    }
    ImGui::EndChild();

    ImGui::End();
}

void Editor::DrawIntegratorPanel(IntegratorSettings& rs, const FrameStats& stats) {
    ImGui::SetNextWindowSize(ImVec2(420, 600), ImGuiCond_FirstUseEver);

    if (!ImGui::Begin("Integrator###ReSTIR", &m_showIntegrator)) {
        ImGui::End();
        return;
    }
    ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.46f);

    ImGui::SliderInt("Samples per pixel", &rs.initialSamples, 1, 8);
    if (ImGui::CollapsingHeader("Path limits")) {
        ImGui::SliderInt("Maximum depth", &rs.maxBounces, 2, 32);
        ImGui::SliderInt("Diffuse bounces", &rs.maxDiffuseBounces, 1, rs.maxBounces);
        ImGui::SliderInt("Roulette starts", &rs.rrStartDepth, 1, rs.maxBounces);
        ImGui::SliderFloat("Caustic roughness floor", &rs.regularizeRoughness, 0.0f, 0.6f, "%.2f");
        ImGui::SetItemTooltip(
            "A smooth lobe reached after a diffuse or glossy bounce is rendered this rough.\n"
            "Caustics blur to what such a surface casts; direct views and mirror chains stay sharp. 0 = off.");
        ImGui::SetItemTooltip("Earlier termination saves rays but increases noise.");
    }

    ImGui::SeparatorText("Light sampling");
    ImGui::Checkbox("Compact light tree", &rs.compactLightTree);
    ImGui::SetItemTooltip("Uses less GPU memory; performance depends on the scene. Changing this rebuilds light buffers.");
    ImGui::Checkbox("Learn light clusters", &rs.lightTreeLearning);
    ImGui::SetItemTooltip("Learns visible light contributions per receiver cell at all distances; every receiver "
                          "samples from the learned cuts.");
    if (rs.lightTreeLearning) {
        ImGui::Text("Lighting capacity: %u cells (%.0f MiB)", LT_GRID_CAPACITY,
                    double(LT_LEARNING_BYTES) / (1024.0 * 1024.0));
        ImGui::SliderFloat("Training roughness floor", &rs.lightTreeLearnRoughness, 0.0f, 0.8f, "%.2f");
        ImGui::SetItemTooltip("Glossy lobes at least this rough train the learning with their own response; "
                              "smoother ones train it as a white diffuse surface would, because what they "
                              "mirror changes with the view and shows up as patches between cells. The "
                              "diffuse lobe always trains with its own response.\n"
                              "Every receiver samples from the learned cuts (one MIS technique: this only trades "
                              "variance, never bias).");
        ImGui::SliderInt("Minimum lighting cell size (log2 m)", &rs.lightTreeCellExponent, -4, 8);
        ImGui::SliderFloat("Lighting cell growth", &rs.lightTreeLodScale, 0.005f, 0.2f, "%.3f");
        ImGui::SetItemTooltip("Cells grow with distance from the camera. Coarser learned cells cover new regions "
                              "while finer cells are prepared.");
        ImGui::Checkbox("Show learning coverage", &rs.lightTreeDebug);
        if (rs.lightTreeDebug)
            ImGui::TextWrapped("Green: requested detail. Blue: coarser cell. Orange: shared fallback. Gray: "
                               "ordinary light tree; fading color shows partial learned sampling. Brightness shows "
                               "completed updates. Red: unavailable.");
        if (ImGui::Button("Reset learned lighting"))
            rs.lightTreeReset = true;
    }
    ImGui::SeparatorText("SHARC");
    ImGui::Checkbox("Radiance cache", &rs.sharcEnabled);
    ImGui::BeginDisabled(!rs.sharcEnabled);
    ImGui::Checkbox("Path guiding", &rs.sharcGuideEnabled);
    ImGui::SliderInt("Training tile width", &rs.sharcUpdateStride, 2, 8);
    ImGui::SetItemTooltip("One training path per tile. Smaller tiles fill the cache faster.");
    if (ImGui::Button("Clear cache"))
        rs.sharcReset = true;

    if (ImGui::CollapsingHeader("Cache tuning")) {
        ImGui::SliderInt("Cell size (log2 m)", &rs.sharcCellSizeExponent, -6, 4);
        ImGui::SliderFloat("Distance scale", &rs.sharcLodScale, 0.001f, 0.1f, "%.3f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderInt("Training depth", &rs.sharcTrainBounces, 4, 64);
        ImGui::SliderInt("Training roulette", &rs.sharcTrainRrDepth, 2, rs.sharcTrainBounces);
        ImGui::SliderInt("Minimum samples", &rs.sharcMinSamples, 8, 256);
        ImGui::SliderInt("History length", &rs.sharcHistoryFrames, 8, 256);
        ImGui::SliderInt("Retention (frames)", &rs.sharcMaxAge, 32, 4096);
        ImGui::SliderFloat("Query footprint", &rs.sharcQueryFootprint, 0.5f, 8.0f, "%.1f");
        ImGui::SetItemTooltip("How many cache cells the lobe that reaches a surface must span (by solid angle) before "
                              "the cache may answer there. Higher values trace further.");
    }
    if (rs.sharcGuideEnabled && ImGui::CollapsingHeader("Guiding tuning")) {
        ImGui::SliderFloat("Maximum guide probability", &rs.sharcGuideMax, 0.0f, 0.9f, "%.2f");
        ImGui::SetItemTooltip(
            "Share of broad-lobe samples the learned lobes may take; the BSDF sampler keeps the rest.");
        ImGui::SliderInt("Receiver level", &rs.sharcGuideLevelOffset, 1, 6);
        ImGui::SliderInt("Guided vertices", &rs.sharcGuideDepth, 1, 7);
        ImGui::SliderInt("Freshness half-life", &rs.sharcGuideFreshness, 8, 2048);
        ImGui::SetItemTooltip(
            "Frames without new training evidence after which a receiver guides at half strength.");
        ImGui::Checkbox("Guide training paths", &rs.sharcGuideTrain);
        ImGui::SetItemTooltip(
            "Training paths sample the learned mixture too, so newly discovered lobes are measured faster.");
    }
    if (ImGui::CollapsingHeader("Cache inspection")) {
        const char* views[] = {"Off", "Cells", "Cell lighting", "Guiding"};
        ImGui::Combo("View", &rs.sharcDebugMode, views, IM_ARRAYSIZE(views));
        if (rs.sharcDebugMode != 0) {
            ImGui::Checkbox(rs.sharcDebugMode == SHARC_DEBUG_GUIDING ? "Show lobe directions" : "Other query level",
                            &rs.sharcDebugCoarse);
        }
    }
    ImGui::EndDisabled();

    if (ImGui::CollapsingHeader("Diffuse resampling")) {
        ImGui::Checkbox("Resample direct and indirect diffuse", &rs.liteEnabled);
        ImGui::BeginDisabled(!rs.liteEnabled);
        ImGui::Checkbox("Spatial reuse", &rs.liteSpatial);
        ImGui::BeginDisabled(!rs.liteSpatial);
        ImGui::SliderInt("Partners", &rs.liteSpatSlots, 0, 3);
        ImGui::SliderFloat("Pair radius (px)", &rs.liteReuseSigma, 2.0f, 40.0f, "%.1f");
        ImGui::SliderInt("Confidence cap", &rs.liteSpatMcap, 1, 64);
        ImGui::EndDisabled();
        ImGui::SliderFloat("Normal similarity", &rs.liteNormalSimCos, -1.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("Plane tolerance", &rs.litePlaneDist, 0.0f, 0.5f, "%.3f");
        ImGui::Checkbox("Show resampled contribution", &rs.liteDebugView);
        ImGui::EndDisabled();
    }

    if (stats.cacheTimingMask != 0 && ImGui::CollapsingHeader("GPU timings")) {
        const char* labels[] = {"Cache prepare",    "Cache training",      "Cache resolve", "Path tracing",
                                "Diffuse shift",    "Diffuse merge",       "Sky / atmosphere", "Light cluster update"};
        static_assert(IM_ARRAYSIZE(labels) == FrameStats::GpuTimingCount);
        for (int i = 0; i < IM_ARRAYSIZE(labels); ++i)
            if (stats.cacheTimingMask & (1u << i))
                ImGui::Text("%s: %.3f ms", labels[i], stats.cachePassMs[i]);
        if (stats.cacheTimingMask & (1u << 7)) {
            ImGui::TextDisabled("Light sampling and feedback are included in the path/cache times.");
        }
    }
    ImGui::PopItemWidth();
    ImGui::End();
}
void Editor::DrawDlssInputsPanel(IntegratorSettings& rs, DLSSManager& dlss) {
    ImGui::SetNextWindowSize(ImVec2(430, 480), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("DLSS buffers###DLSS Inputs", &m_showDlssInputs)) {
        ImGui::End();
        return;
    }

    static const char* layers[] = {"Off",
                                   "Color input",
                                   "Denoised output",
                                   "Depth",
                                   "Motion vectors",
                                   "Normals",
                                   "Diffuse albedo",
                                   "Specular albedo",
                                   "Roughness",
                                   "Specular motion",
                                   "Specular distance",
                                   "Transparency",
                                   "Color before transparency",
                                   "Bias hint"};
    ImGui::Combo("Layer", &rs.dlssDebugLayer, layers, IM_ARRAYSIZE(layers));
    if (rs.dlssDebugLayer == 3 || rs.dlssDebugLayer == 10)
        ImGui::DragFloatRange2("Range (m)", &rs.dlssDebugDepthNear, &rs.dlssDebugDepthFar, 0.25f, 0.0f, 65000.0f,
                               "near %.2f", "far %.2f");
    if (rs.sharcEnabled && rs.sharcDebugMode != 0) {
        ImGui::TextDisabled("SHARC inspection is active.");
        if (ImGui::Button("Show DLSS buffer instead"))
            rs.sharcDebugMode = 0;
    }
    if (ImGui::CollapsingHeader("Guide overrides")) {
        auto guide = [&](const char* label, bool& off) {
            bool enabled = !off;
            if (ImGui::Checkbox(label, &enabled)) {
                off = !enabled;
                dlss.ForceReset();
            }
        };
        guide("Depth", dlss.guideOffDepth);
        guide("Motion", dlss.guideOffMV);
        guide("Normals", dlss.guideOffNormals);
        guide("Roughness", dlss.guideOffRough);
        guide("Diffuse albedo", dlss.guideOffAlbedo);
        guide("Specular albedo", dlss.guideOffSpecAlb);
        guide("Specular motion", dlss.guideOffSpecMV);
        guide("Surface replacement", dlss.guideOffPsr);
        guide("Motion blend", dlss.guideOffMvBlend);
        if (ImGui::Button("Restore all guides")) {
            dlss.guideOffDepth = dlss.guideOffMV = dlss.guideOffNormals = dlss.guideOffRough = false;
            dlss.guideOffAlbedo = dlss.guideOffSpecAlb = dlss.guideOffSpecMV = dlss.untagSpecMV = false;
            dlss.guideOffPsr = dlss.guideOffMvBlend = false;
            dlss.ForceReset();
        }
    }

    if (ImGui::CollapsingHeader("Guide sentinel", ImGuiTreeNodeFlags_DefaultOpen)) {
        const auto& gs = dlss.sentinel;
        ImGui::Text("frame %llu  |  max luma %.3f  max |MV| %.2f px  max |specMV| %.2f px",
                    (unsigned long long)gs.frame, gs.maxLuma, gs.maxMV, gs.maxSpecMV);
        ImGui::Text("pixels at luma cap: %u", gs.capCount);
        if (gs.mask != 0) {
            const uint32_t bx = gs.firstBad & 0xFFFFu, by = gs.firstBad >> 16;
            ImGui::TextColored(ImVec4(1, 0.3f, 0.3f, 1), "ANOMALY NOW: mask 0x%03X, %u px, first (%u,%u)", gs.mask,
                               gs.badCount, bx ? bx - 1 : 0, by ? by - 1 : 0);
        } else {
            ImGui::TextColored(ImVec4(0.4f, 1, 0.4f, 1), "guides clean");
        }
        if (gs.lastFrame != 0) {
            const uint32_t lx = gs.lastBad & 0xFFFFu, ly = gs.lastBad >> 16;
            ImGui::TextColored(ImVec4(1, 0.8f, 0.3f, 1), "last anomaly: frame %llu, mask 0x%03X, first (%u,%u)",
                               (unsigned long long)gs.lastFrame, gs.lastMask, lx ? lx - 1 : 0, ly ? ly - 1 : 0);
            ImGui::SetItemTooltip("Mask bits: 0x001 color NaN/Inf  0x002 depth bad  0x004 MV NaN/Inf\n"
                                  "0x008 normal/rough NaN/Inf  0x010 roughness out of [0,1]\n"
                                  "0x020 specMV NaN/Inf  0x040/0x080 albedo NaN/Inf\n"
                                  "0x100 |MV|>256px  0x200 |specMV|>256px\n"
                                  "If an instability onset happens while this stays clean, the\n"
                                  "poison is NOT in the guide data — it's options/execution side.");
        } else {
            ImGui::TextDisabled("no anomaly seen this session");
        }
    }

    if (ImGui::CollapsingHeader("D3D12 messages during slEvaluateFeature")) {
        ImGui::Text("total captured: %llu", (unsigned long long)dlss.evalDxMessageTotal);
        if (dlss.evalDxMessages.empty()) {
            ImGui::TextDisabled("none — the evaluate window is validation-clean");
        } else {
            for (const auto& m : dlss.evalDxMessages)
                ImGui::TextWrapped("%s", m.c_str());
        }
    }
    ImGui::End();
}

void Editor::DrawSunPanel(Scene& scene, Camera& camera, const FrameStats& stats, mc::VoxelStreamer* voxels) {
    ImGui::SetNextWindowSize(ImVec2(430, 600), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Environment###Sun / Time of Day", &m_showSun)) {
        ImGui::End();
        return;
    }
    ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.46f);
    auto& s = camera.sunSettings;
    ImGui::SeparatorText("Light sources");
    ImGui::Checkbox("Mesh lights", &scene.lightClassEnabled[Scene::LightClassScene]);
    ImGui::SetItemTooltip("Emissive triangles of the loaded meshes. Off: they leave the light tree and stop glowing.");
    ImGui::Checkbox("Emissive cubes", &scene.lightClassEnabled[Scene::LightClassCubes]);
    ImGui::SetItemTooltip("The scene's animated emissive cubes (EmissiveCubes).");
    if (voxels) {
        ImGui::Checkbox("Block lights", &voxels->config().lights);
        ImGui::SetItemTooltip(
            "Emissive Minecraft blocks (glowstone, lanterns, torches...). Off retires every chunk light at once,\n"
            "on re-meshes the lit chunks so they come back with their lights.");
    }
    ImGui::SeparatorText("Daylight");
    ImGui::SliderFloat("Time (UTC)", &s.startUTCHours, 0.0f, 24.0f, "%.1f h");
    ImGui::SliderFloat("Sun intensity", &s.sunIntensity, 0.0f, 100.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderFloat("Sky intensity", &s.skyIntensity, 0.0f, 100.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderFloat("Turbidity", &s.turbidity, 1.0f, 10.0f, "%.1f");
    if (ImGui::CollapsingHeader("Location and animation")) {
        ImGui::SliderFloat("Latitude", &s.latitude, -90.0f, 90.0f, "%.2f deg");
        ImGui::SliderFloat("Longitude", &s.longitude, -180.0f, 180.0f, "%.2f deg");
        ImGui::SliderFloat("Day of year", &s.dayOfYear, 1.0f, 365.0f, "%.0f");
        ImGui::SliderFloat("Time speed", &s.simSpeed, 0.0f, 10000.0f, "%.1fx", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Night speedup", &s.nightSpeedup, 1.0f, 10.0f, "%.1fx");
    }
    if (ImGui::CollapsingHeader("Night sky")) {
        ImGui::SliderFloat("Stars", &s.skyStarIntensity, 0.0f, 5.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Star contrast", &s.skyStarGamma, 1.0f, 4.0f, "%.2f");
        ImGui::SliderFloat("Star detail", &s.skyStarLodBias, -1.0f, 3.0f, "%.2f");
        ImGui::SliderFloat("Star threshold", &s.skyStarThreshold, 0.0f, 0.5f, "%.3f");
        ImGui::SliderFloat("Night brightness", &s.skyNightBaseIntensity, 0.0f, 50.0f, "%.2f",
                           ImGuiSliderFlags_Logarithmic);
    }
    if (ImGui::CollapsingHeader("Atmosphere quality")) {
        int view = (int)s.atmosViewSteps, light = (int)s.atmosLightSteps;
        int aerial = (int)s.atmosAerialViewSteps, aerialLight = (int)s.atmosAerialLightSteps;
        if (ImGui::SliderInt("Sky steps", &view, 4, 32))
            s.atmosViewSteps = (float)view;
        if (ImGui::SliderInt("Light steps", &light, 2, 16))
            s.atmosLightSteps = (float)light;
        if (ImGui::SliderInt("Haze steps", &aerial, 2, 16))
            s.atmosAerialViewSteps = (float)aerial;
        if (ImGui::SliderInt("Haze light steps", &aerialLight, 2, 16))
            s.atmosAerialLightSteps = (float)aerialLight;
        ImGui::SliderFloat("Scattering scale", &s.atmosMultiScatterFactor, 0.5f, 3.0f, "%.2f");
        ImGui::SliderFloat("Shadow softness", &s.atmosEarthShadowSoftness, 0.0f, 0.05f, "%.4f");
        ImGui::SliderFloat("Halo depth (km)", &s.atmosHaloDistanceKm, 0.0f, 20.0f, "%.2f",
                           ImGuiSliderFlags_AlwaysClamp);
        ImGui::SetItemTooltip("Distance over which the halo around the sun fades in. That halo is the forward\n"
                              "lobe of the aerosol phase function, which knows only which way a ray points and\n"
                              "not how much air is in front of the surface it ends on - so without this it\n"
                              "brightens a wall a few metres away as much as the sky behind it.\n"
                              "The lobe is faded towards its isotropic average over this distance, which moves\n"
                              "the in-scatter around rather than removing it: near surfaces get plain haze,\n"
                              "distant ones get the halo, and the sky is untouched either way.\n"
                              "0 restores the undamped lobe.");
    }
    ImGui::PopItemWidth();
    ImGui::End();
}
