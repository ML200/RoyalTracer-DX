//====================================
//EDITOR
//====================================
//model-level scene hierarchy, live materials

#include "../stdafx.h"
#include "Editor.h"
#include <unordered_set>

void Editor::Init(HWND hwnd, ID3D12Device* device, UINT numFramesInFlight,
                  ID3D12DescriptorHeap* srvHeap,
                  D3D12_CPU_DESCRIPTOR_HANDLE fontCpu,
                  D3D12_GPU_DESCRIPTOR_HANDLE fontGpu)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();
    auto& style = ImGui::GetStyle();
    style.WindowRounding   = 4.0f;
    style.FrameRounding    = 2.0f;
    style.GrabRounding     = 2.0f;
    style.Colors[ImGuiCol_WindowBg].w = 0.92f;

    ImGui_ImplWin32_Init(hwnd);
    ImGui_ImplDX12_Init(device, numFramesInFlight,
        DXGI_FORMAT_R8G8B8A8_UNORM, srvHeap, fontCpu, fontGpu);
}

void Editor::Shutdown() {
    ImGui_ImplDX12_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();
}

// ─────────────────────────────────────────────────────────────────
void Editor::Draw(Scene& scene, Camera& camera, FlyCamController& flyCam,
                  PassSystem& passes, DLSSManager& dlss, DLSSGSettings& dlssG,
                  ReSTIRSettings& restir, nrc::Settings& nrc,
                  float fps, const FrameStats& stats,
                  const planet::StreamOrchestrator::Stats& planetStats)
{
    //Push the planet perf sample every frame, even when hidden, so opening
    //the panel doesn't show an empty graph. Pausing freezes the rings.
    if (!m_planetHist.paused) m_planetHist.push(planetStats, stats);

    if (!m_visible) return;

    ImGui_ImplDX12_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();

    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("View")) {
            ImGui::MenuItem("Scene Hierarchy", nullptr, &m_showScene);
            ImGui::MenuItem("Camera",          nullptr, &m_showCamera);
            ImGui::MenuItem("Pass Pipeline",   nullptr, &m_showPipeline);
            ImGui::MenuItem("DLSS",            nullptr, &m_showDLSS);
            ImGui::MenuItem("ReSTIR",          nullptr, &m_showReSTIR);
            //ImGui::MenuItem("NRC",             nullptr, &m_showNRC);   // NRC disabled (planet bring-up)
            ImGui::MenuItem("Sun / Time of Day", nullptr, &m_showSun);
            ImGui::MenuItem("Clouds",          nullptr, &m_showClouds);
            ImGui::MenuItem("Materials",       nullptr, &m_showMaterials);
            ImGui::MenuItem("Planet Perf",     nullptr, &m_showPlanetPerf);
            ImGui::EndMenu();
        }
        ImGui::Separator();
        if (dlssG.enabled && dlssG.framesToGenerate > 0) {
            float presentedFps = fps * (1 + dlssG.framesToGenerate);
            ImGui::Text("%.1f fps (%.1f rendered + %dx FG) | %.2f ms",
                presentedFps, fps, 1 + dlssG.framesToGenerate,
                fps > 0 ? 1000.0f / fps : 0.0f);
        } else {
            ImGui::Text("%.1f fps | %.2f ms", fps, fps > 0 ? 1000.0f / fps : 0.0f);
        }
        ImGui::Separator();
        ImGui::Text("CPU: %.1f ms (upd %.1f | inst %.1f | pop %.1f | tlas %.2f)",
            stats.cpuFrameMs, stats.cpuUpdateMs, stats.cpuInstanceMs,
            stats.cpuPopulateMs, stats.tlasMs);
        ImGui::Separator();
        ImGui::Text("GPU: %.1f ms", stats.gpuMs);
        ImGui::Separator();
        ImGui::Text("%u inst | %u mesh", stats.instanceCount, stats.meshCount);
        if (stats.tlasWasRebuilt) { ImGui::SameLine(); ImGui::TextColored(ImVec4(1,0.3f,0.3f,1), "[TLAS REBUILD]"); }
        else if (stats.tlasWasRefit) { ImGui::SameLine(); ImGui::TextColored(ImVec4(1,0.8f,0,1), "[TLAS refit]"); }
        ImGui::EndMainMenuBar();
    }

    if (m_showScene)     DrawScenePanel(scene);
    if (m_showCamera)    DrawCameraPanel(camera, flyCam);
    if (m_showPipeline)  DrawPassPipelinePanel(passes);
    if (m_showDLSS)      DrawDLSSPanel(dlss, dlssG);
    if (m_showReSTIR)    DrawReSTIRPanel(restir);
    //if (m_showNRC)       DrawNRCPanel(nrc);   // NRC disabled (planet bring-up)
    if (m_showSun)       DrawSunPanel(camera);
    if (m_showClouds)    DrawCloudPanel(camera);
    if (m_showMaterials) DrawMaterialInspector(scene, camera);
    if (m_showPlanetPerf) DrawPlanetPerfPanel(planetStats, stats, fps);

    // Material re-upload happens via dirty flag checked in Renderer
    ImGui::Render();
}

void Editor::Render(ID3D12GraphicsCommandList* cmdList) {
    if (!m_visible) return;
    ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), cmdList);
}

//====================================
//SCENE PANEL
//====================================
//one entry per loaded model
void Editor::DrawScenePanel(Scene& scene) {
    ImGui::SetNextWindowPos(ImVec2(10, 30), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(360, 450), ImGuiCond_FirstUseEver);

    if (!ImGui::Begin("Scene Hierarchy")) { ImGui::End(); return; }

    // ── Model list ───────────────────────────────────────────────
    for (int mi = 0; mi < (int)scene.models.size(); ++mi) {
        auto& model = scene.models[mi];
        bool selected = (m_selectedModel == mi);

        char label[256];
        snprintf(label, sizeof(label), "%s  (%u meshes, %u instances)##model%d",
            model.name.c_str(), model.meshCount, model.instanceCount, mi);

        if (ImGui::Selectable(label, selected))
            m_selectedModel = mi;
    }

    ImGui::Separator();

    // ── Selected model transform ─────────────────────────────────
    if (m_selectedModel >= 0 && m_selectedModel < (int)scene.models.size()) {
        auto& model = scene.models[m_selectedModel];
        ImGui::Text("Edit: %s", model.name.c_str());
        ImGui::TextDisabled("File: %s", model.filePath.c_str());

        bool changed = false;
        changed |= ImGui::DragFloat3("Position", &model.position.x, 0.05f);
        changed |= ImGui::DragFloat3("Rotation", &model.rotation.x, 0.5f);
        changed |= ImGui::DragFloat3("Scale",    &model.scale.x,    0.01f, 0.001f, 100.0f);

        if (changed) {
            scene.MarkModelMoved((UINT)m_selectedModel);
        }

        ImGui::Separator();
        ImGui::TextDisabled("Meshes %u-%u | Instances %u-%u",
            model.meshStart, model.meshStart + model.meshCount - 1,
            model.instanceStart, model.instanceStart + model.instanceCount - 1);

        // Show unique materials used by this model (cached — only recomputed on selection change)
        if (m_cachedMatModel != m_selectedModel) {
            m_cachedMatModel = m_selectedModel;
            std::unordered_set<UINT> seen;
            m_cachedUniqueMats.clear();
            for (UINT i = model.meshStart; i < model.meshStart + model.meshCount; ++i) {
                if (i >= scene.meshes.size()) break;
                for (UINT mid : scene.meshes[i].cpuMaterialIDs) {
                    if (seen.insert(mid).second)
                        m_cachedUniqueMats.push_back(mid);
                }
            }
        }
        if (!m_cachedUniqueMats.empty()) {
            ImGui::Text("Materials (%zu):", m_cachedUniqueMats.size());
            for (UINT mid : m_cachedUniqueMats) {
                char btn[32]; snprintf(btn, sizeof(btn), "Mat %u", mid);
                if (ImGui::SmallButton(btn)) { m_selectedMat = (int)mid; m_showMaterials = true; }
                ImGui::SameLine();
            }
            ImGui::NewLine();
        }

        // Expandable sub-instances (collapsed by default)
        if (ImGui::TreeNode("Sub-instances")) {
            for (UINT i = model.instanceStart; i < model.instanceStart + model.instanceCount; ++i) {
                if (i >= scene.instances.size()) break;
                auto& inst = scene.instances[i];
                ImGui::TextDisabled("[%u] %s (mesh %u)", i, inst.name.c_str(), inst.meshIndex);
            }
            ImGui::TreePop();
        }
    }

    ImGui::End();
}

//====================================
//CAMERA PANEL
//====================================
void Editor::DrawCameraPanel(Camera& camera, FlyCamController& flyCam) {
    ImGui::SetNextWindowPos(ImVec2(10, 490), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(360, 180), ImGuiCond_FirstUseEver);

    if (!ImGui::Begin("Camera")) { ImGui::End(); return; }

    ImGui::DragFloat("FOV",              &camera.fovDegrees, 0.5f, 10.0f, 170.0f);
    ImGui::DragFloat("Near Plane",       &camera.nearPlane,  0.001f, 0.001f, 10.0f, "%.3f");
    ImGui::DragFloat("Far Plane",        &camera.farPlane,   1000.0f, 100.0f, 1.0e9f, "%.0f");
    ImGui::SliderFloat("Move Speed",     &flyCam.moveSpeed,  0.01f, 1000000.0f, "%.3f",
                       ImGuiSliderFlags_Logarithmic);
    ImGui::DragFloat("Mouse Sensitivity",&flyCam.mouseSensitivity, 0.01f, 0.01f, 2.0f, "%.2f");

    if (ImGui::Button("Reset Camera")) {
        camera.ResetView();
        flyCam.Reset();
    }

    ImGui::Separator();
    ImGui::TextUnformatted("Depth of Field");
    ImGui::DragFloat("Aperture Radius",  &camera.apertureRadius, 0.001f, 0.0f, 1.0f, "%.4f");
    ImGui::DragFloat("Focus Distance",   &camera.focusDistance,  0.05f, 0.01f, 10000.0f, "%.3f",
                     ImGuiSliderFlags_Logarithmic);

    ImGui::Separator();
    ImGui::Text("Jitter frame: %u", camera.JitterFrame());

    ImGui::End();
}

//====================================
//PASS PIPELINE PANEL
//====================================
void Editor::DrawPassPipelinePanel(PassSystem& passes) {
    ImGui::SetNextWindowPos(ImVec2(380, 30), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(350, 400), ImGuiCond_FirstUseEver);

    if (!ImGui::Begin("Pass Pipeline")) { ImGui::End(); return; }

    const char* stageNames[] = {
        "RayGen", "Compute", "FixedCompute", "Wavefront", "Barrier",
        "LoopStart", "LoopEnd", "PingSwap", "ClearSort", "Callable", "DLSS",
        "CudaOp"
    };

    for (size_t i = 0; i < passes.Passes().size(); ++i) {
        auto& p = passes.Passes()[i];
        int stageIdx = static_cast<int>(p.stage);
        const char* stageName = (stageIdx < _countof(stageNames)) ? stageNames[stageIdx] : "?";

        ImVec4 color(0.8f, 0.8f, 0.8f, 1.0f);
        switch (p.stage) {
            case Stage::RayGen:  color = ImVec4(0.3f,0.9f,0.3f,1); break;
            case Stage::Compute: color = ImVec4(0.3f,0.6f,0.9f,1); break;
            case Stage::Barrier: color = ImVec4(0.6f,0.6f,0.6f,1); break;
            case Stage::DLSS:    color = ImVec4(0.9f,0.6f,0.2f,1); break;
            case Stage::CudaOp:  color = ImVec4(0.76f,0.46f,0.87f,1); break;
            case Stage::LoopStart: case Stage::LoopEnd: color = ImVec4(0.9f,0.9f,0.3f,1); break;
            default: break;
        }

        ImGui::PushStyleColor(ImGuiCol_Text, color);
        if (p.file.empty()) {
            ImGui::Text("[%2zu] %s", i, stageName);
        } else {
            char fileStr[256];
            WideCharToMultiByte(CP_UTF8, 0, p.file.c_str(), -1, fileStr, 256, nullptr, nullptr);
            ImGui::Text("[%2zu] %s: %s", i, stageName, fileStr);
            if (p.stage == Stage::Compute && !p.isWorkGraph)
                ImGui::SameLine(), ImGui::TextDisabled("(%ux%u)", p.groupX, p.groupY);
        }
        ImGui::PopStyleColor();
    }

    ImGui::End();
}

//====================================
//DLSS PANEL
//====================================
void Editor::DrawDLSSPanel(DLSSManager& dlss, DLSSGSettings& dlssG) {
    ImGui::SetNextWindowPos(ImVec2(380, 440), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(350, 200), ImGuiCond_FirstUseEver);

    if (!ImGui::Begin("DLSS")) { ImGui::End(); return; }

    // ── DLSS-RR (Ray Reconstruction) ────────────────────────────
    ImGui::SeparatorText("Ray Reconstruction");

    const char* modeLabels[] = { "Off", "DLAA", "Quality", "Balanced" };
    const sl::DLSSMode modeValues[] = {
        sl::DLSSMode::eOff,
        sl::DLSSMode::eDLAA,
        sl::DLSSMode::eMaxQuality,
        sl::DLSSMode::eBalanced
    };
    constexpr int modeCount = IM_ARRAYSIZE(modeLabels);

    int currentIdx = 1;
    for (int i = 0; i < modeCount; ++i) {
        if (dlss.mode == modeValues[i]) { currentIdx = i; break; }
    }

    if (ImGui::Combo("RR Mode", &currentIdx, modeLabels, modeCount))
        dlss.mode = modeValues[currentIdx];

    ImGui::TextDisabled("Render: %ux%u -> Display: %ux%u",
        dlss.RenderWidth(), dlss.RenderHeight(),
        dlss.DisplayWidth(), dlss.DisplayHeight());

    // ── DLSS-G (Frame Generation) ───────────────────────────────
    ImGui::SeparatorText("Frame Generation");

    if (!dlssG.available) {
        ImGui::TextDisabled("Not available on this GPU");
    } else {
        ImGui::Checkbox("Enabled", &dlssG.enabled);

        if (dlssG.enabled) {
            // Multiplier labels based on hardware max
            const char* fgLabels[] = { "2x", "3x", "4x" };
            // framesToGenerate: 1=2x, 2=3x, 3=4x  ->  combo index = framesToGenerate - 1
            int fgIdx = dlssG.framesToGenerate - 1;
            if (ImGui::Combo("Multiplier", &fgIdx, fgLabels, dlssG.maxFrames))
                dlssG.framesToGenerate = fgIdx + 1;
        }
    }

    ImGui::End();
}

//====================================
//MATERIAL INSPECTOR
//====================================
void Editor::DrawMaterialInspector(Scene& scene, Camera& camera) {
    ImGui::SetNextWindowPos(ImVec2(740, 30), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(420, 700), ImGuiCond_FirstUseEver);

    if (!ImGui::Begin("Materials", &m_showMaterials)) { ImGui::End(); return; }

    // Lives on SunSettings only because that struct is the tail of the camera CB.
    // Applied at GPU emission read sites, leaves authored Ke untouched.
    ImGui::SliderFloat("Global Emission", &camera.sunSettings.globalEmissionStrength,
                       0.0f, 10.0f, "%.2fx");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Scales every emissive material equally.\n1.0 = authored values.");
    ImGui::Separator();

    // Material list (left)
    ImGui::BeginChild("MatList", ImVec2(180, 0), true);

    // Filter box. Case-insensitive substring match against the name; a
    // purely numeric query also matches the material index. Empty query
    // shows everything.
    ImGui::SetNextItemWidth(-FLT_MIN);
    ImGui::InputTextWithHint("##matFilter", "filter...", m_matFilter, sizeof(m_matFilter));

    auto toLower = [](char c) -> char {
        return (c >= 'A' && c <= 'Z') ? (char)(c - 'A' + 'a') : c;
    };
    auto containsCI = [&](const char* hay, const char* needle) -> bool {
        if (!needle || !*needle) return true;
        if (!hay) return false;
        const size_t nLen = strlen(needle);
        for (const char* p = hay; *p; ++p) {
            size_t j = 0;
            while (j < nLen && p[j] && toLower(p[j]) == toLower(needle[j])) ++j;
            if (j == nLen) return true;
        }
        return false;
    };

    // Numeric-query shortcut: "12" matches material index 12 directly,
    // without requiring the index to appear in the name string.
    bool numericQuery = false;
    int  numericValue = 0;
    if (m_matFilter[0]) {
        numericQuery = true;
        for (const char* p = m_matFilter; *p; ++p) {
            if (*p < '0' || *p > '9') { numericQuery = false; break; }
            numericValue = numericValue * 10 + (*p - '0');
        }
    }

    int matchCount = 0;
    for (int i = 0; i < (int)scene.materials.size(); ++i) {
        const char* name = (i < (int)scene.materialNames.size() && !scene.materialNames[i].empty())
            ? scene.materialNames[i].c_str() : nullptr;

        if (m_matFilter[0]) {
            const bool nameMatch = containsCI(name, m_matFilter);
            const bool idxMatch  = numericQuery && (i == numericValue);
            if (!nameMatch && !idxMatch) continue;
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
    }

    if (m_matFilter[0] && matchCount == 0) {
        ImGui::TextDisabled("(no matches)");
    }

    ImGui::EndChild();

    ImGui::SameLine();

    // Properties (right)
    ImGui::BeginChild("MatProps", ImVec2(0, 0), false);
    if (m_selectedMat >= 0 && m_selectedMat < (int)scene.materials.size()) {
        const int i = m_selectedMat;
        auto& mats = scene.materials;
        const char* matName = (i < (int)scene.materialNames.size() && !scene.materialNames[i].empty())
            ? scene.materialNames[i].c_str() : nullptr;
        if (matName)
            ImGui::Text("Material %d: %s", i, matName);
        else
            ImGui::Text("Material %d", i);
        ImGui::Separator();

        bool changed = false;
        bool emissionChanged = false;

        // ── Surface ──────────────────────────────────────────────
        if (ImGui::CollapsingHeader("Surface", ImGuiTreeNodeFlags_DefaultOpen)) {
            changed |= ImGui::ColorEdit3("Albedo", &mats.Kd[i].x, ImGuiColorEditFlags_Float);
            changed |= ImGui::SliderFloat("Opacity", &mats.Kd[i].w, 0.0f, 1.0f, "%.3f");
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("0 = fully transparent (glass)\n1 = fully opaque");
            changed |= ImGui::DragFloat("IOR", &mats.Ni[i], 0.01f, 1.0f, 3.0f, "%.3f");
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Index of Refraction\n1.0 = air, 1.33 = water, 1.5 = glass");
        }

        // ── Emission ─────────────────────────────────────────────
        if (ImGui::CollapsingHeader("Emission", ImGuiTreeNodeFlags_DefaultOpen)) {
            XMFLOAT3& Ke = mats.Ke[i];
            bool emEdit = ImGui::ColorEdit3("Emission", &Ke.x,
                ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR);
            if (emEdit) { changed = true; emissionChanged = true; }

            if (Ke.x > 0 || Ke.y > 0 || Ke.z > 0) {
                float intensity = std::max({Ke.x, Ke.y, Ke.z});
                float prevIntensity = intensity;
                if (ImGui::DragFloat("Intensity", &intensity, 0.1f, 0.0f, 1000.0f)) {
                    if (prevIntensity > 0.001f) {
                        float s = intensity / prevIntensity;
                        Ke.x *= s; Ke.y *= s; Ke.z *= s;
                        changed = true; emissionChanged = true;
                    }
                }
            }
        }

        // ── PBR ──────────────────────────────────────────────────
        if (ImGui::CollapsingHeader("PBR", ImGuiTreeNodeFlags_DefaultOpen)) {
            changed |= ImGui::SliderFloat("Roughness", &mats.Pr_Pm_Ps_Pc[i].x, 0.0f, 1.0f);
            changed |= ImGui::SliderFloat("Metallic",  &mats.Pr_Pm_Ps_Pc[i].y, 0.0f, 1.0f);
            changed |= ImGui::SliderFloat("Sheen",     &mats.Pr_Pm_Ps_Pc[i].z, 0.0f, 1.0f);
        }

        // ── Clearcoat ────────────────────────────────────────────
        if (ImGui::CollapsingHeader("Clearcoat")) {
            ImGui::PushID("coat");
            changed |= ImGui::SliderFloat("Strength",   &mats.Pr_Pm_Ps_Pc[i].w,      0.0f, 1.0f);
            changed |= ImGui::SliderFloat("Roughness",  &mats.Pcr_aniso_anisor[i].x,  0.0f, 1.0f);
            ImGui::PopID();
        }

        // ── Anisotropy ───────────────────────────────────────────
        if (ImGui::CollapsingHeader("Anisotropy")) {
            ImGui::PushID("aniso");
            changed |= ImGui::SliderFloat("Strength",   &mats.Pcr_aniso_anisor[i].y, -1.0f, 1.0f);
            changed |= ImGui::SliderFloat("Rotation",   &mats.Pcr_aniso_anisor[i].z,  0.0f, 1.0f);
            ImGui::PopID();
        }

        // ── Transmission ─────────────────────────────────────────
        if (ImGui::CollapsingHeader("Transmission")) {
            changed |= ImGui::ColorEdit3("Filter (Tf)", &mats.Tf[i].x, ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Volume absorption color\nWhite = no absorption");
        }

        // ── Alpha ────────────────────────────────────────────────
        if (ImGui::CollapsingHeader("Alpha Test")) {
            changed |= ImGui::SliderFloat("Threshold", &mats.alphaThreshold[i], 0.0f, 1.0f);

            //ensure the SoA slot exists (older materials predate this field)
            if (i >= mats.invertAlpha.size()) mats.invertAlpha.resize(i + 1, 0u);
            bool inv = mats.invertAlpha[i] != 0u;
            if (ImGui::Checkbox("Invert (sample is transparency)", &inv)) {
                mats.invertAlpha[i] = inv ? 1u : 0u;
                changed = true;
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "Top of the invert-alpha hierarchy (L3, manual override).\n"
                    "Auto-detection at load: filename hint (L1) and brightness check (L2).\n"
                    "Toggle on if the texture's alpha encodes 1 = transparent (map_Tr style).");
        }

        // ── Textures (read-only info) ────────────────────────────
        if (ImGui::CollapsingHeader("Textures")) {
            ImGui::TextDisabled("Assigned texture IDs:");
            ImGui::Text("  Albedo: %s", mats.albedoTexID[i] >= 0 ? std::to_string(mats.albedoTexID[i]).c_str() : "none");
            ImGui::Text("  Normal: %s", mats.normalTexID[i] >= 0 ? std::to_string(mats.normalTexID[i]).c_str() : "none");
            ImGui::Text("  RMA:    %s", mats.rmaTexID[i]    >= 0 ? std::to_string(mats.rmaTexID[i]).c_str()    : "none");
        }

        if (changed)
            scene.MarkMaterialsDirty(emissionChanged);
    } else {
        ImGui::TextDisabled("Select a material.");
    }
    ImGui::EndChild();

    ImGui::End();
}

// ─────────────────────────────────────────────────────────────────
void Editor::DrawReSTIRPanel(ReSTIRSettings& rs) {
    ImGui::SetNextWindowSize(ImVec2(340, 460), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("ReSTIR")) { ImGui::End(); return; }

    if (ImGui::CollapsingHeader("Temporal", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Enable##Temp", &rs.enableTempGI);
        ImGui::SliderInt("M-cap##Temp", &rs.tempMcapGI, 0, 128);
    }
    if (ImGui::CollapsingHeader("Spatial", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Enable##Spat", &rs.enableSpatGI);
        ImGui::SliderInt("Radius Max##Spat", &rs.spatRadMaxGI, 4, 128);
        ImGui::SliderInt("Radius Min##Spat", &rs.spatRadMinGI, 4, 128);
        ImGui::SliderInt("Tries##Spat",      &rs.spatTriesGI, 2, 16);
    }
    if (ImGui::CollapsingHeader("Neighbor Rejection", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat("Normal dot min", &rs.rejNormalDot, 0.0f, 1.0f);
        ImGui::SetItemTooltip("Reject neighbor if dot(nA, nB) falls below this.");
        ImGui::SliderFloat("Distance max",   &rs.rejDistance,  0.001f, 1.0f, "%.3f");
        ImGui::SetItemTooltip("Reject neighbor if |proj onto normal| exceeds this (world units).");
    }
    if (ImGui::CollapsingHeader("Roughness Reuse")) {
        ImGui::SliderFloat("Min##Rough", &rs.reuseRoughnessMin, 0.0f, 1.0f);
        ImGui::SliderFloat("Max##Rough", &rs.reuseRoughnessMax, 0.0f, 1.0f);
    }
    ImGui::End();
}

// ─────────────────────────────────────────────────────────────────
void Editor::DrawNRCPanel(nrc::Settings& n) {
    ImGui::SetNextWindowSize(ImVec2(340, 300), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("NRC")) { ImGui::End(); return; }

    if (ImGui::CollapsingHeader("Pipeline", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Enable cache (terminate + resolve)", &n.enabled);
        ImGui::SetItemTooltip("Off = pure ReSTIR PT. Cache termination + resolve are short-circuited.");

        ImGui::Checkbox("Train", &n.trainingEnabled);
        ImGui::SetItemTooltip("Off = weights frozen, inference still runs against whatever state was last trained.");

        if (ImGui::Button("Reinitialize weights")) {
            n.requestReinit = true;
        }
        ImGui::SetItemTooltip(
            "Reseed the MLP and clear the EMA. Use this when the cache has\n"
            "collapsed to all-zero / all-black and won't recover on its own.");
    }

    if (ImGui::CollapsingHeader("Debug view", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Show cache at primary vertex", &n.debugView);
        ImGui::SetItemTooltip(
            "Queries L̂_s at x1 per pixel, writes to gOutput slice 3.\n"
            "Cycle to slice 3 with 'C' to view it.\n"
            "Overrides cache termination in raygen while on.");
    }

    if (ImGui::CollapsingHeader("Termination", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat("Area-spread c", &n.areaSpreadC, 0.001f, 0.1f, "%.4f", ImGuiSliderFlags_Logarithmic);
        ImGui::SetItemTooltip("Paper's c (eq. 3-4). Smaller = terminate earlier (more cache, more bias).");
    }

    if (ImGui::CollapsingHeader("Scene bounds (position encoding)")) {
        // Auto-recomputed every frame from the scene AABB; shown here
        // purely for diagnostic purposes, edits get overwritten.
        ImGui::BeginDisabled(true);
        ImGui::DragFloat3("Center (auto)",     &n.sceneCenter.x, 0.0f);
        ImGui::DragFloat ("Half extent (auto)", &n.sceneExtent,  0.0f);
        ImGui::EndDisabled();
        ImGui::TextDisabled(
            "Derived from mesh localAabbs × live instance transforms.");
    }

    if (ImGui::CollapsingHeader("Optimizer")) {
        ImGui::SliderFloat("LR scale", &n.learningRateScale, 0.01f, 10.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
        ImGui::SetItemTooltip("Reserved — a later turn will feed this into tcnn's Adam LR.");
    }

    ImGui::End();
}

// ─────────────────────────────────────────────────────────────────
void Editor::DrawSunPanel(Camera& camera) {
    ImGui::SetNextWindowSize(ImVec2(320, 300), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Sun / Time of Day")) { ImGui::End(); return; }

    auto& s = camera.sunSettings;

    if (ImGui::CollapsingHeader("Location / Date", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat("Latitude",    &s.latitude,  -90.0f, 90.0f, "%.2f deg");
        ImGui::SliderFloat("Longitude",   &s.longitude, -180.0f, 180.0f, "%.2f deg");
        ImGui::SliderFloat("Day of Year", &s.dayOfYear, 1.0f, 365.0f, "%.0f");
    }
    if (ImGui::CollapsingHeader("Simulation", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat("Sim Speed",       &s.simSpeed, 0.0f, 10000.0f, "%.1fx",
                           ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Start UTC Hours", &s.startUTCHours, 0.0f, 24.0f, "%.1f h");
        ImGui::SliderFloat("Night Speedup",   &s.nightSpeedup, 1.0f, 10.0f, "%.1fx");
    }
    if (ImGui::CollapsingHeader("Appearance", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat("Turbidity",      &s.turbidity, 1.0f, 10.0f, "%.1f");
        ImGui::SliderFloat("Sun Intensity",  &s.sunIntensity, 0.0f, 100.0f, "%.2f",
                           ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Sky Intensity",  &s.skyIntensity, 0.0f, 100.0f, "%.2f",
                           ImGuiSliderFlags_Logarithmic);
    }
    if (ImGui::CollapsingHeader("Stars / Milky Way", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat("Star Intensity", &s.skyStarIntensity, 0.0f, 5.0f, "%.3f",
                           ImGuiSliderFlags_Logarithmic);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Final brightness multiplier on the star texture sample,\n"
                              "applied after the gamma curve.");

        ImGui::SliderFloat("Star Gamma",     &s.skyStarGamma, 1.0f, 4.0f, "%.2f");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Luminance power curve. >1 crushes the bilinear mip\n"
                              "smear (faint dim averaged pixels) into near black\n"
                              "while preserving peak star centres. 1.0 = linear,\n"
                              "2.0 = balanced, 3.0+ = aggressive sparkle.");

        ImGui::SliderFloat("Star LOD Bias",  &s.skyStarLodBias, -1.0f, 3.0f, "%.2f");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Extra mip offset on top of the footprint based pick.\n"
                              "Higher = blurrier + more temporally stable under jitter;\n"
                              "lower = sharper but may flicker on single texel bright\n"
                              "stars. 0 = exactly pixel = texel.");

        ImGui::SliderFloat("Star Threshold", &s.skyStarThreshold, 0.0f, 0.5f, "%.3f");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Black level subtraction applied before the gamma curve.\n"
                              "Cuts the bilinear halo around each star so the visible\n"
                              "footprint shrinks to the bright centre, fixing the\n"
                              "\"large blob\" look. Raise for sharper / sparser stars,\n"
                              "but very high values start clipping faint real stars.");

        ImGui::SliderFloat("Night Base", &s.skyNightBaseIntensity, 0.0f, 50.0f, "%.2f",
                           ImGuiSliderFlags_Logarithmic);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Brightness of the residual airglow / integrated faint\n"
                              "starlight that fills the night sky between resolved\n"
                              "stars. Independent of sun intensity (was a bug before).\n"
                              "0 = pitch black night, 5 = previous default look,\n"
                              "higher = stylized brighter night sky.");

        if (ImGui::Button("Reset Star Defaults")) {
            s.skyStarIntensity      = 0.047f;
            s.skyStarGamma          = 1.57f;
            s.skyStarLodBias        = -1.0f;
            s.skyStarThreshold      = 0.0f;
            s.skyNightBaseIntensity = 0.57f;
        }
    }

    if (ImGui::CollapsingHeader("Atmosphere", ImGuiTreeNodeFlags_DefaultOpen)) {
        //Bruneton atmosphere march quality. View / light steps drive the
        //dominant cost of every sky / cloud pixel (each cloud shell phase
        //integrates ATMOS_VIEW_STEPS atmospheric samples + ATMOS_LIGHT_STEPS
        //sun ray taps per sample). Aerial perspective is a separate cheap
        //march for the haze in front of meshes.
        int v;

        v = (int)s.atmosViewSteps;
        if (ImGui::SliderInt("View Steps", &v, 4, 32))
            s.atmosViewSteps = (float)v;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Per ray atmosphere sample count. Dominant cost of\n"
                              "the sky / unified cloud march. 12 = Bruneton\n"
                              "baseline; raise for smoother gradients on long\n"
                              "horizon rays, drop to 6..8 for cheap previews.");

        v = (int)s.atmosLightSteps;
        if (ImGui::SliderInt("Light Steps", &v, 2, 16))
            s.atmosLightSteps = (float)v;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Sun ray transmittance step count for each\n"
                              "atmosphere sample. Mostly affects the spectral\n"
                              "accuracy of the sunset tint; 8 is plenty for\n"
                              "smooth gradients, raise only if you see banding\n"
                              "in the orange band at low sun.");

        v = (int)s.atmosAerialViewSteps;
        if (ImGui::SliderInt("Aerial View Steps", &v, 2, 16))
            s.atmosAerialViewSteps = (float)v;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("View ray sample count inside ComputeAerialPerspective\n"
                              "(haze in front of meshes). 4 is the baseline;\n"
                              "doubling smooths long mesh ray haze gradients.");

        v = (int)s.atmosAerialLightSteps;
        if (ImGui::SliderInt("Aerial Light Steps", &v, 2, 16))
            s.atmosAerialLightSteps = (float)v;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Sun ray step count inside ComputeAerialPerspective.\n"
                              "Same rationale as the main Light Steps but for the\n"
                              "aerial perspective march only.");

        ImGui::SliderFloat("Multi Scatter Factor", &s.atmosMultiScatterFactor, 0.5f, 3.0f, "%.2f");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Artistic boost on the DIRECTIONAL single scatter\n"
                              "term only. Real 2nd+ order scattering now comes\n"
                              "from the per frame Hillaire Psi_ms LUT, which this\n"
                              "slider deliberately does not touch (the old 1.1\n"
                              "default was the flat stand-in for that term).\n"
                              "1.0 = physical, 1.2..1.5 = stylized brighter sky.");

        ImGui::SliderFloat("Cloud Shadow Cone (deg)", &s.atmosCloudShadowConeDeg, 0.0f, 15.0f, "%.2f");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Half angle of the cone the atmospheric cloud\n"
                              "shadow tap samples. Wider = softer shafts of\n"
                              "light through cloud gaps, more bleed across\n"
                              "cloud edges; narrower = sharper shafts but more\n"
                              "visible per pixel stepping until DLSS RR resolves\n"
                              "the cone jitter. 5 degrees was the previous hard\n"
                              "coded default.");

        ImGui::SliderFloat("Cloud Shadow Floor", &s.atmosCloudShadowFloor, 0.0f, 0.5f, "%.3f");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Safety floor on the cloud shadow tap for\n"
                              "atmospheric samples. Shadowed air is now lit by\n"
                              "the physically based through deck diffuse source,\n"
                              "so this only guards numeric corner cases (0.01).\n"
                              "Raising it re-tints under deck air with the\n"
                              "clear sky spectrum - the old sunset-at-noon\n"
                              "band - so treat values above ~0.05 as a look,\n"
                              "not a fix.");

        ImGui::SliderFloat("Earth Shadow Softness", &s.atmosEarthShadowSoftness, 0.0f, 0.05f, "%.4f");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Half width (cosine units) of the planet shadow\n"
                              "penumbra at the horizon. 0.005 cos ≈ 0.57 degrees\n"
                              "angular, comparable to the sun's apparent\n"
                              "diameter. Larger = wider soft band, smaller =\n"
                              "sharper terminator on the horizon haze.");

        if (ImGui::Button("Reset Atmosphere Defaults")) {
            s.atmosViewSteps              = 12.0f;
            s.atmosLightSteps             = 8.0f;
            s.atmosAerialViewSteps        = 4.0f;
            s.atmosAerialLightSteps       = 4.0f;
            s.atmosMultiScatterFactor     = 1.0f;
            s.atmosCloudShadowConeDeg     = 5.0f;
            s.atmosCloudShadowFloor       = 0.01f;
            s.atmosEarthShadowSoftness    = 0.005f;
        }
    }

    ImGui::End();
}

// ─────────────────────────────────────────────────────────────────
//CLOUD PANEL
//Live controls for the volumetric cloud system in Clouds_v8.hlsli.
//Every field maps 1:1 onto a CloudSettings member which Camera uploads
//into the camera cbuffer tail; the shader macros in Includes_v8.hlsli
//redirect the Clouds_v8 CLOUD_* identifiers to those cbuffer fields so
//edits take effect on the next frame without a recompile.
void Editor::DrawCloudPanel(Camera& camera) {
    ImGui::SetNextWindowSize(ImVec2(360, 520), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Clouds")) { ImGui::End(); return; }

    auto& c = camera.cloudSettings;

    //----- Master toggle (mirrors cloud_enabled, sampled as <0.5/>=0.5)
    bool enabled = c.enabled >= 0.5f;
    if (ImGui::Checkbox("Enabled", &enabled)) c.enabled = enabled ? 1.0f : 0.0f;
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Runtime master switch. The shader takes an\n"
                          "early-terrain path when off, so the cost is\n"
                          "essentially free. ENABLE_CLOUDS in\n"
                          "Clouds_v8.hlsli is the compile-time kill\n"
                          "switch that dead-codes the integrator.");

    if (ImGui::CollapsingHeader("Coverage", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat("Coverage##amount",    &c.coverage,           0.0f, 1.0f, "%.2f");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Fraction of sky filled with cumulus. 0 = clear,\n"
                              "~0.5 = scattered, 1.0 = overcast. Uses\n"
                              "Schneider's coverage-threshold remap so the\n"
                              "field stays sharp instead of fading uniformly.");
    }

    if (ImGui::CollapsingHeader("Shell Geometry", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat("Layer Bottom",   &c.layerBotKm,    0.0f, 20.0f, "%.2f km");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Altitude of the cloud base above the planet\n"
                              "surface. Real cumulus base sits at 1..2 km in\n"
                              "fair weather, 0.5..1 km in maritime air.");

        ImGui::SliderFloat("Layer Top",      &c.layerTopKm,    0.0f, 30.0f, "%.2f km");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Top of the cloud layer. Cumulus tops vary from\n"
                              "3 km (small) to 12 km (cumulonimbus). Must be\n"
                              "above Layer Bottom or the shell is empty.");

        // (Horizon Fade slider removed 2026-06-11 — cloud_horizonFadeKm has
        // had no shader consumer since the unified-march refactor. The
        // struct field stays for cbuffer layout; see Common.h.)

        ImGui::SliderFloat("Top Variation",  &c.topVariationKm, 0.0f, 10.0f, "%.2f km");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Per cloud top altitude jitter. Each column's\n"
                              "effective top is Layer Top + variation * noise,\n"
                              "so 0 collapses to a flat slab top and larger\n"
                              "values produce towering cumulus reaching well\n"
                              "above Layer Top.");

        ImGui::SliderFloat("Top Frequency",  &c.topFrequency,   0.0f, 0.5f, "%.3f /km",
                           ImGuiSliderFlags_Logarithmic);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Horizontal frequency of the per cloud top altitude\n"
                              "noise. Lower = neighbouring cumulus share top\n"
                              "altitudes (long thunderstorm fronts), higher =\n"
                              "tall and short cumulus alternate cloud to cloud.");
    }

    if (ImGui::CollapsingHeader("Density Field", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat("Extinction",     &c.extinction,    1.0f, 200.0f, "%.1f /km",
                           ImGuiSliderFlags_Logarithmic);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Sigma_t at unit density (per km). Real water\n"
                              "cumulus runs 25..60 /km. Higher = more opaque,\n"
                              "lower = wispier and more translucent.");

        ImGui::SliderFloat("Base Frequency", &c.baseFrequency, 0.05f, 5.0f, "%.3f /km",
                           ImGuiSliderFlags_Logarithmic);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Frequency of the low-frequency Worley field that\n"
                              "defines cumulus blob spacing. Lower = bigger\n"
                              "clouds, higher = smaller more numerous puffs.");

        ImGui::SliderFloat("HF Frequency",   &c.hfFrequency,   0.5f, 30.0f, "%.2f /km",
                           ImGuiSliderFlags_Logarithmic);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Erosion noise frequency — controls the wispy\n"
                              "edge detail size. Higher = finer whisps.");

        ImGui::SliderFloat("HF Amount",      &c.hfAmount,      0.0f, 1.0f, "%.2f");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("How aggressively the HF noise eats edges. 0 =\n"
                              "smooth Worley blobs, 1 = heavily eroded whispy\n"
                              "stratocumulus. Cores stay intact regardless.");

        ImGui::SliderFloat("Coverage Edge Width", &c.covModFilterWidth, 0.01f, 1.0f, "%.2f");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Schneider coverage threshold remap edge width.\n"
                              "Smaller = sharper cloud silhouettes (hard edged\n"
                              "cumulus), larger = softer transition between cloud\n"
                              "and clear sky.");

        ImGui::SliderFloat("Domain Warp",    &c.warpAmpKm,     0.0f, 3.0f, "%.2f km");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Low frequency domain warp amplitude. Pushes the\n"
                              "base shape around so cumulus don't look like\n"
                              "stamps on a grid. 0 disables warp (slightly\n"
                              "faster, more obvious tiling). Auto attenuated\n"
                              "with distance via the LOD blend.");
    }

    if (ImGui::CollapsingHeader("Phase Function (Nubis-3)", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat("Droplet Size", &c.silverIntensity, 0.0f, 1.0f, "%.2f");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Effective cloud droplet diameter (Jendersie & d'Eon\n"
                              "2023 Mie approximation), remapped 0..1 -> 5..30 um.\n"
                              "Small (haze, drizzle ~5 um) = broad forward halo.\n"
                              "Mid (cumulus ~10..15 um) = concentrated silver lining\n"
                              "with proper Mie shape.\n"
                              "Large (large droplets ~30 um) = very sharp silver\n"
                              "lining peak within ~0.5 deg of the sun.");

        ImGui::SliderFloat("Silver Spread (unused)", &c.silverSpread, 0.01f, 0.3f, "%.3f");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("LEGACY: the dual-lobe HG silver-spread slider.\n"
                              "No longer wired into the direct phase since the\n"
                              "switch to the Jendersie-d'Eon Mie approximation.\n"
                              "Kept in the cbuffer for layout compatibility.");

        ImGui::SliderFloat("Shadow Cone (deg)", &c.shadowConeDeg,  0.0f, 15.0f, "%.2f");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Half angle of the sun shadow defocus cone (deg).\n"
                              "0 = strict sun direction (cheapest single ray).\n"
                              "2..6 = visibly softer self shadow when paired\n"
                              "with Shadow Cone Samples > 1.");

        // Albedo is stored as three consecutive scalar floats in the cbuffer
        // (HLSL scalar packing). C++ guarantees no padding between consecutive
        // float members, so &albedoR is a valid float[3] for ColorEdit3.
        ImGui::ColorEdit3("Single Scatter Albedo", &c.albedoR);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Per channel single scattering albedo. 0.995\n"
                              "white = real water cumulus (almost lossless).\n"
                              "Drop all three for pollution / dust loaded\n"
                              "clouds, tint asymmetric for sunset rim experiments.");

        ImGui::SliderFloat("Sun Tau Multiplier", &c.sunTauMult, 0.0f, 5.0f, "%.2f");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Multiplier on the optical depth accumulated along\n"
                              "the sun shadow ray. >1 deepens self shadow,\n"
                              "<1 lifts the shadow side of cumulus. 1.0 keeps\n"
                              "the integrator physically calibrated.");
    }

    if (ImGui::CollapsingHeader("Multi-Scatter Fill")) {
        //MS model selector. 0 = current Nubis sqrt(Tdir) shortcut (one
        //isotropic-ish MS lobe, cheapest). 1..4 = Wrenninge multi octave
        //(Hillaire 2016 §5.8) which adds N extra extinction evals per cloud
        //sample with progressively attenuated extinction (a^n = 0.5^n) and
        //isotropic phase. No extra shadow taps so perf cost is small.
        //Octaves 3/4 reach exp(-tau/8) and exp(-tau/16) — the similarity-
        //theory diffusion scale that keeps thick-cloud undersides from
        //going exponentially black.
        const char* msModeNames[] = {
            "0: Nubis shortcut (1 lobe, cheapest)",
            "1: Wrenninge 2 octave",
            "2: Wrenninge 3 octave",
            "3: Wrenninge 4 octave (deep)",
            "4: Wrenninge 5 octave (deepest)",
        };
        int msModeIdx = (int)c.msMode;
        if (ImGui::Combo("MS Model", &msModeIdx, msModeNames, IM_ARRAYSIZE(msModeNames)))
            c.msMode = (float)msModeIdx;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Multi-scatter model:\n"
                              "0 = current Nubis Evolved single term following\n"
                              "    sqrt(Tdir). Cheapest, one phase function eval\n"
                              "    beyond direct. MS Strength + Floor sliders apply.\n"
                              "1 = Wrenninge 2 octave (direct + one extra octave\n"
                              "    with a^n / b^n / c^n per Hillaire 2016 §5.8).\n"
                              "    Soft fill in deep cores the shortcut misses.\n"
                              "2 = Wrenninge 3 octave (direct + two extra). Deep\n"
                              "    cumulus cores read as illuminated rather than\n"
                              "    just dark. Two extra phase evals per sample,\n"
                              "    no extra shadow taps.\n\n"
                              "MS Strength + Floor still scale octaves 1..N in\n"
                              "modes 1 and 2 so the artist knobs keep working.\n"
                              "Secondary Strength and Secondary G apply only in\n"
                              "mode 0 (they parametrise the shortcut's MS lobe).");

        ImGui::SliderFloat("Secondary Strength", &c.secondaryStrength, 0.0f, 1.5f, "%.2f");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Amplitude of the secondary multi-scatter phase.\n"
                              "Modulated by density × height × sun atten. Lifts\n"
                              "the shadow side of cumulus without the cost of\n"
                              "a full octave loop. 0 = no fill, 1.0 = bright.");

        ImGui::SliderFloat("Secondary G",        &c.secondaryG,        0.0f, 0.6f, "%.2f");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("HG eccentricity of the secondary lobe. Smaller\n"
                              "= more isotropic (more fill across the volume),\n"
                              "larger = still forward biased like the primary.");

        ImGui::SliderFloat("MS Strength",        &c.msStrength,        0.0f, 20.0f, "%.2f");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Global multiplier on the multi-scatter contribution.\n"
                              "4.0 = Nubis Evolved baseline. 0 disables MS and\n"
                              "clouds collapse to pure single scatter (very dark\n"
                              "shadow sides and cores).");

        ImGui::SliderFloat("MS Base Floor",      &c.msHeightFloor,     0.0f, 1.0f, "%.2f");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Minimum MS amplitude floor at cloud base (h=0).\n"
                              "Without this real cumulus bases read as black;\n"
                              "0.18 keeps them 'shaded white' the way real\n"
                              "stratocumulus bases look from below.");
    }

    if (ImGui::CollapsingHeader("Animation")) {
        ImGui::SliderFloat("Wind X", &c.windX, -1.0f, 1.0f, "%.3f km/s");
        ImGui::SliderFloat("Wind Z", &c.windZ, -1.0f, 1.0f, "%.3f km/s");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Horizontal drift of the cloud field over time.\n"
                              "Driven by walltime, so paused sim freezes the\n"
                              "field. ±0.05 km/s is a gentle breeze.");
    }

    if (ImGui::CollapsingHeader("Indirect Lighting", ImGuiTreeNodeFlags_DefaultOpen)) {
        bool skyAmb = c.skyAmbient >= 0.5f;
        if (ImGui::Checkbox("Sky Ambient", &skyAmb))
            c.skyAmbient = skyAmb ? 1.0f : 0.0f;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Hemispherical sky-dome illumination on cloud\n"
                              "samples. Single zenith probe per pixel, biased\n"
                              "but cheap. Lifts cloud shadow sides from inky\n"
                              "to natural blue-gray, the dominant fix for\n"
                              "the dim-looking underside complaint.");

        ImGui::SliderFloat("Sky Ambient Scale", &c.skyAmbientScale, 0.0f, 4.0f, "%.2f");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Artistic multiplier on the sky ambient term.\n"
                              "1.0 is physically scaled to the Bruneton sky.");

        ImGui::SliderFloat("Sky Ambient Intensity", &c.ambientIntensity, 0.0f, 4.0f, "%.2f");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Brightness multiplier on the sky dome contribution\n"
                              "applied at every cloud sample. Stacks with Sky\n"
                              "Ambient Scale (this controls per sample weight,\n"
                              "the scale controls overall mix).");

        ImGui::SliderFloat("Sky AO Scale",       &c.ambientAOScale,    0.0f, 2.0f, "%.2f");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Scales how much the column density above a sample\n"
                              "occludes the sky probe. 0 = sky reaches every\n"
                              "sample regardless of overhead cloud, 1.0 = full\n"
                              "physical attenuation through the overhead column.");

        ImGui::SliderFloat("Sky AO Max OD",      &c.ambientODMax,      0.5f, 50.0f, "%.2f",
                           ImGuiSliderFlags_Logarithmic);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Optical depth cap on the sky AO term. Default 8\n"
                              "lets dense overhead columns actually shut the sky\n"
                              "term down (exp(-8) ~ 0.03%%). Lowering it floors\n"
                              "the leakage — at 2 every thick base kept ~13%%\n"
                              "sky light, a thickness-independent brightness\n"
                              "floor.");

        bool gnd = c.groundBounce >= 0.5f;
        if (ImGui::Checkbox("Ground Bounce", &gnd))
            c.groundBounce = gnd ? 1.0f : 0.0f;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Reflected sun off the ground proxy lighting\n"
                              "cloud bases from below. Snow / desert / ocean\n"
                              "scenes need this for cloud bottoms to read\n"
                              "as bright instead of gray-flat.");

        ImGui::SliderFloat("Ground Albedo", &c.groundAlbedo, 0.0f, 1.0f, "%.2f");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Gray Lambertian reflectance of the terrain.\n"
                              "0.18 grass/forest, 0.30 desert, 0.06 ocean,\n"
                              "0.85 fresh snow. Biased single scalar until a\n"
                              "ground irradiance map is wired in.");

        ImGui::SliderFloat("Ground Scale", &c.groundScale, 0.0f, 4.0f, "%.2f");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Artistic multiplier on the ground bounce term.");
    }

    if (ImGui::CollapsingHeader("Surface Interaction")) {
        bool surf = c.cloudShadowOnSurfaces >= 0.5f;
        if (ImGui::Checkbox("Shadow On Surfaces", &surf))
            c.cloudShadowOnSurfaces = surf ? 1.0f : 0.0f;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Cloud transmittance applied to the sun NEE on\n"
                              "scene surfaces, producing live cloud shadows on\n"
                              "terrain and props. Each surface NEE pays for a\n"
                              "short cloud march, expensive without a shadow\n"
                              "map. Off by default until the shadow map pass\n"
                              "lands.");
    }

    if (ImGui::CollapsingHeader("Quality / Performance")) {
        int viewMax = (int)c.viewStepsMax;
        if (ImGui::SliderInt("View Steps Max", &viewMax, 16, 256))
            c.viewStepsMax = (float)viewMax;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Hard upper loop bound on the main view march.\n"
                              "The adaptive stepper usually exits early via\n"
                              "t >= tFar — this is the runaway guard. 128 =\n"
                              "Nubis baseline; 64 buys ~30%% on grazing orbital\n"
                              "views; 32 for cheap previews.");

        ImGui::SliderFloat("Target Step Size", &c.targetStepKm, 0.1f, 3.0f, "%.2f km");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Base step size for the fine portion of the\n"
                              "adaptive view march. Smaller = denser sampling\n"
                              "= better quality, worse perf. 0.6 km tuned\n"
                              "for stratocumulus; raise to 1.0..1.5 for ~half\n"
                              "the sample count when HF noise hides banding.");

        int shadowSteps = (int)c.shadowSteps;
        if (ImGui::SliderInt("Surface Shadow Steps", &shadowSteps, 1, 6))
            c.shadowSteps = (float)shadowSteps;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Sample count for the surface-shadow march (cloud\n"
                              "shadows on terrain via CloudSunVisibility).\n"
                              "1 = fast single-sample sphere-intersect path,\n"
                              "~4-5x cheaper than multi-tap and visually\n"
                              "indistinguishable for overhead cumulus.\n"
                              "2..6 = multi-tap shell march for softer edges /\n"
                              "low sun angles at proportional cost. This is\n"
                              "called per-pixel per-bounce when 'Shadow On\n"
                              "Surfaces' is on, so it's the biggest single knob\n"
                              "for that feature's cost.");

        int cheapSteps = (int)c.cheapSteps;
        if (ImGui::SliderInt("Bounce Cheap Steps", &cheapSteps, 4, 32))
            c.cheapSteps = (float)cheapSteps;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Step count for the cheap volume march used on\n"
                              "indirect bounces (specular / transmission).\n"
                              "10 = Nubis baseline; drop to 6 if bounce-ray\n"
                              "clouds are an indirect-illumination niche.");

        ImGui::SliderFloat("Bounce Cheap Max Length", &c.cheapMaxLenKm,
                           10.0f, 500.0f, "%.0f km", ImGuiSliderFlags_Logarithmic);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Maximum march length along the bounce ray (km).\n"
                              "Clamps the cheap path so bounce rays don't pay\n"
                              "for orbital-distance clouds.");

        int shadowK = (int)c.shadowConeSamples;
        if (ImGui::SliderInt("Shadow Cone Samples", &shadowK, 1, 5))
            c.shadowConeSamples = (float)shadowK;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Number of jittered shadow rays inside the sun\n"
                              "defocus cone per scattering event. 1 = single\n"
                              "jittered sample (cheapest), 3..5 = visibly softer\n"
                              "self shadow at proportional cost.");

        int ambientK = (int)c.ambientSteps;
        if (ImGui::SliderInt("Ambient Occlusion Steps", &ambientK, 1, 6))
            c.ambientSteps = (float)ambientK;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Upward density march sample count per scattering\n"
                              "event, used to estimate sky ambient occlusion\n"
                              "from cloud overhead. 2 = Nubis baseline (cheap).\n"
                              "4..6 = smoother but proportional cost. Two sky\n"
                              "anchor colours (zenith + horizon) are sampled\n"
                              "once per pixel regardless of this setting.");

        ImGui::SliderFloat("Tr Cutoff", &c.trEps, 1e-4f, 0.1f, "%.4f",
                           ImGuiSliderFlags_Logarithmic);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("View transmittance threshold below which the\n"
                              "march exits early. Lower = more accurate (less\n"
                              "early exit), higher = faster but slight banding\n"
                              "behind dense clouds.");

        ImGui::SliderFloat("RR Threshold", &c.rrThreshold, 0.01f, 0.5f, "%.2f");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Russian roulette termination boundary on view\n"
                              "throughput. Samples above this never terminate,\n"
                              "samples below survive with probability\n"
                              "throughput/threshold. Lower = less variance,\n"
                              "higher = faster.");

        ImGui::SliderFloat("Max Step",         &c.maxStepKm,    0.01f, 5.0f, "%.3f km",
                           ImGuiSliderFlags_Logarithmic);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Cap on a single empty space step before adaptive\n"
                              "growth kicks in. Smaller = more sample density in\n"
                              "near empty regions but more steps wasted; larger\n"
                              "= fewer wasted steps but risk of missing thin\n"
                              "clouds at the start of the march.");

        ImGui::SliderFloat("Step Growth",      &c.stepGrowth,   1.0f, 1.5f, "%.3f");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Geometric growth factor on the in cloud step.\n"
                              "Each fine step multiplies stride by this until\n"
                              "the Max Fine Step ceiling. 1.0 = constant step,\n"
                              "1.1 = aggressive growth (cheap, banding at edges).");

        ImGui::SliderFloat("Zero Density",     &c.effectiveZeroDensity,
                           1e-5f, 0.1f, "%.5f", ImGuiSliderFlags_Logarithmic);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Density floor below which a sample is treated as\n"
                              "empty (no in scatter, no transmittance update).\n"
                              "Higher = skip more thin cloud edges (faster, more\n"
                              "visible silhouette steps); lower = capture every\n"
                              "wisp.");

        ImGui::SliderFloat("Max Empty Step",   &c.maxEmptyStepKm,
                           1.0f, 500.0f, "%.1f km", ImGuiSliderFlags_Logarithmic);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Absolute cap on the big empty space step. The\n"
                              "march takes huge strides through distant clear\n"
                              "sky; this clamps the stride so even at long\n"
                              "distances we don't skip an entire cloud field\n"
                              "in one step.");

        ImGui::SliderFloat("Empty Growth/Km",  &c.emptyStepGrowthPerKm,
                           0.0f, 2.0f, "%.3f");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Empty step grows as (1 + t * this) with view ray\n"
                              "distance. 0 = empty step never grows, 0.1 = ~10x\n"
                              "growth per 100 km, 1.0 = aggressive (great for\n"
                              "orbital views over deserts).");

        ImGui::SliderFloat("Max Fine Step",    &c.maxFineStepKm, 0.1f, 10.0f, "%.2f km");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Ceiling on the in cloud step after geometric\n"
                              "growth. Smaller = denser sampling deep in thick\n"
                              "clouds, larger = lets the step balloon for cheap\n"
                              "interiors at the cost of banding.");
    }

    if (ImGui::CollapsingHeader("Distance / LOD")) {
        ImGui::SliderFloat("Fade Distance", &c.fadeDistanceKm,
                           50.0f, 10000.0f, "%.0f km", ImGuiSliderFlags_Logarithmic);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Distance along the view ray at which clouds\n"
                              "start fading terrain. Must be < Render Distance.\n"
                              "Drop to ~200 km for ground-level scenes where\n"
                              "the horizon hides anything beyond.");

        ImGui::SliderFloat("Render Distance", &c.renderDistanceKm,
                           100.0f, 10000.0f, "%.0f km", ImGuiSliderFlags_Logarithmic);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Hard clamp on the view march distance (km).\n"
                              "Beyond this the march exits early. 3000 km\n"
                              "= orbital baseline; 300 km for ground level\n"
                              "gives a big perf win because the march stops\n"
                              "at the horizon instead of integrating\n"
                              "through dead pixels.");

        // Keep fade < render so the smoothstep doesn't invert
        if (c.fadeDistanceKm >= c.renderDistanceKm)
            c.fadeDistanceKm = c.renderDistanceKm * 0.9f;

        // (Haze Strength slider removed 2026-06-11 — cloud_hazeStrength has
        // had no shader consumer since the unified-march refactor folded
        // atmosphere and cloud into one integral. The struct field stays
        // for cbuffer layout; see Common.h.)

        ImGui::SliderFloat("LOD Near",  &c.lodNearKm, 0.0f, 100.0f, "%.1f km",
                           ImGuiSliderFlags_Logarithmic);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Below this distance the cloud noise runs at full\n"
                              "quality (full domain warp, both HF erosion taps,\n"
                              "cauliflower mid-frequency). Above LOD Far the noise\n"
                              "drops to the simplified far-field path.");

        ImGui::SliderFloat("LOD Far",   &c.lodFarKm,  1.0f, 500.0f, "%.1f km",
                           ImGuiSliderFlags_Logarithmic);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Distance at which the cloud noise reaches its\n"
                              "simplified far-field state. Smaller band (Near→Far)\n"
                              "= sharper LOD step, larger = smoother quality\n"
                              "transition.");

        // Keep LOD near <= far so the saturate( (d-near)/(far-near) ) blend
        // doesn't divide by zero or invert.
        if (c.lodNearKm >= c.lodFarKm)
            c.lodNearKm = c.lodFarKm * 0.5f;
    }

    if (ImGui::Button("Reset Cloud Defaults")) {
        c = CloudSettings{};
    }

    ImGui::End();
}

//====================================
//PLANET PERFORMANCE
//====================================
//Per-frame ring sample + the panel that visualises it. Frame pacing (total
//frame CPU vs GPU), planet CPU/GPU breakdowns, and the async pipeline
//(Pending/Ready/BLAS-recorded/Built cell counts) all plotted as wrapping
//PlotLines so you can spot spikes at a glance.

void Editor::PlanetPerfHistory::push(const planet::StreamOrchestrator::Stats& ps,
                                     const FrameStats& fs)
{
    const int i = write;
    frame_total_ms    [i] = fs.cpuFrameMs;
    frame_gpu_ms      [i] = fs.gpuMs;
    planet_cpu_ms     [i] = ps.blas_record_cpu_ms;
    planet_plan_ms    [i] = ps.plan_ms;
    planet_blas_gpu_ms[i] = ps.blas_gpu_ms;
    planet_tlas_gpu_ms[i] = ps.tlas_gpu_ms;
    cells_recorded    [i] = (float)ps.cells_recorded;
    pipe_pending      [i] = (float)ps.cells_pending;
    pipe_ready        [i] = (float)ps.cells_ready;
    pipe_blas_pending [i] = (float)ps.cells_recorded_total;
    pipe_built        [i] = (float)ps.dirty_built;

    write  = (write + 1) % N;
    if (filled < N) ++filled;
}

namespace {
//Largest sample in a wrapping ring; used for auto-scale labels.
inline float ring_max(const float* v, int n) {
    float m = 0.0f;
    for (int i = 0; i < n; ++i) if (v[i] > m) m = v[i];
    return m;
}
inline float ring_avg(const float* v, int n) {
    if (n == 0) return 0.0f;
    double s = 0.0; for (int i = 0; i < n; ++i) s += v[i];
    return (float)(s / n);
}
//Last sample written into a wrapping ring (i.e. the most recent value).
inline float ring_last(const float* v, int write, int filled) {
    if (filled == 0) return 0.0f;
    const int idx = (write + Editor::PlanetPerfHistory::N - 1)
                  % Editor::PlanetPerfHistory::N;
    return v[idx];
}

//Plot a single metric with a min/max/avg/now readout next to it. Auto-scaled
//to [0, max(samples)] with a small headroom so flat traces don't look like
//noise on the y axis. 'offset' is the ring's OLDEST sample (= write index).
void plot_metric(const char* label, const float* values, int count, int offset,
                 int write, const char* unit, ImVec4 colour)
{
    const float vmax = ring_max(values, count);
    const float vavg = ring_avg(values, count);
    const float vnow = ring_last(values, write, count);
    const float scale_max = vmax > 0.0f ? vmax * 1.1f : 1.0f;

    char overlay[64];
    snprintf(overlay, sizeof(overlay), "now %.3f  avg %.3f  max %.3f %s",
             vnow, vavg, vmax, unit);

    ImGui::PushStyleColor(ImGuiCol_PlotLines, colour);
    ImGui::PlotLines(label, values, count, offset, overlay,
                     0.0f, scale_max, ImVec2(-1, 60));
    ImGui::PopStyleColor();
}
} // namespace

void Editor::DrawPlanetPerfPanel(const planet::StreamOrchestrator::Stats& ps,
                                 const FrameStats& fs, float fps)
{
    ImGui::SetNextWindowPos(ImVec2(20, 60),   ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(560, 760), ImGuiCond_FirstUseEver);

    if (!ImGui::Begin("Planet Performance", &m_showPlanetPerf)) {
        ImGui::End();
        return;
    }

    //ImGui::PlotLines wraps a ring buffer when 'offset' = oldest-sample index.
    //When the ring isn't yet full, the oldest is index 0; once full, it's the
    //write cursor.
    const int   count  = m_planetHist.filled;
    const int   offset = (m_planetHist.filled < PlanetPerfHistory::N) ? 0
                                                                      : m_planetHist.write;
    const int   write  = m_planetHist.write;

    //--- top bar: live readout + pause ---
    ImGui::Text("%.1f fps  (%.2f ms frame)", fps, fps > 0 ? 1000.0f / fps : 0.0f);
    ImGui::SameLine();
    ImGui::Checkbox("Pause", &m_planetHist.paused);
    ImGui::SameLine();
    if (ImGui::Button("Clear")) {
        m_planetHist = PlanetPerfHistory{};
    }
    ImGui::Separator();

    //--- LIVE generation state (static numbers, no graph) ---
    ImGui::SeparatorText("LIVE generation");
    ImGui::Text("built=%d  cells=%u  leaves=%u  tris=%llu  tlas_instances=%u",
                (int)ps.built, ps.cell_count, ps.leaf_count,
                (unsigned long long)ps.triangle_count, ps.tlas_instances);

    ImGui::SeparatorText("Rebuild");
    if (ps.rebuilding) {
        ImGui::Text("ACTIVE  dirty=%u/%u  recorded=%u  recorded_pending=%u  ready=%u  pending=%u",
                    ps.dirty_built, ps.dirty_total,
                    ps.cells_recorded, ps.cells_recorded_total,
                    ps.cells_ready, ps.cells_pending);
    } else {
        ImGui::TextDisabled("idle  (last rebuild %u frames, est %.1f f)",
                            ps.last_rebuild_frames, ps.rebuild_frames_est);
    }

    //--- frame pacing ---
    ImGui::SeparatorText("Frame pacing");
    plot_metric("##frame_cpu", m_planetHist.frame_total_ms, count, offset, write,
                "ms", ImVec4(0.40f, 0.85f, 0.40f, 1.0f));
    ImGui::SameLine(); ImGui::TextUnformatted("CPU frame");
    plot_metric("##frame_gpu", m_planetHist.frame_gpu_ms,   count, offset, write,
                "ms", ImVec4(0.95f, 0.55f, 0.20f, 1.0f));
    ImGui::SameLine(); ImGui::TextUnformatted("GPU frame");

    //--- planet CPU (render-thread cost: BLAS recording + plan job) ---
    ImGui::SeparatorText("Planet CPU (render thread)");
    plot_metric("##blas_rec_cpu", m_planetHist.planet_cpu_ms, count, offset, write,
                "ms", ImVec4(0.30f, 0.70f, 1.00f, 1.0f));
    ImGui::SameLine(); ImGui::TextUnformatted("blas_record");
    ImGui::TextDisabled("plan job runs on the worker pool - this is the "
                        "render thread cost of recording BLAS commands.");

    //--- plan job (worker thread - one shot per rebuild) ---
    ImGui::SeparatorText("Planet plan job (worker thread)");
    plot_metric("##plan", m_planetHist.planet_plan_ms, count, offset, write,
                "ms", ImVec4(0.85f, 0.40f, 0.85f, 1.0f));
    ImGui::SameLine(); ImGui::TextUnformatted("plan_ms");
    ImGui::TextDisabled("non-zero only on the frame after a rebuild was "
                        "triggered (the plan job's LOD select + cell cut + "
                        "diff). All other frames read the cached last value.");

    //--- planet GPU (BLAS builds + TLAS rebuild on the compute queue) ---
    ImGui::SeparatorText("Planet GPU (compute queue)");
    plot_metric("##blas_gpu", m_planetHist.planet_blas_gpu_ms, count, offset, write,
                "ms", ImVec4(1.00f, 0.50f, 0.50f, 1.0f));
    ImGui::SameLine(); ImGui::TextUnformatted("BLAS builds");
    plot_metric("##tlas_gpu", m_planetHist.planet_tlas_gpu_ms, count, offset, write,
                "ms", ImVec4(1.00f, 0.85f, 0.30f, 1.0f));
    ImGui::SameLine(); ImGui::TextUnformatted("TLAS rebuild");
    ImGui::TextDisabled("GPU timestamps lag ~4 frames (fence-gated readback).");

    //--- async pipeline state (cells in each stage) ---
    ImGui::SeparatorText("Async pipeline (cells per stage)");
    plot_metric("##pending", m_planetHist.pipe_pending, count, offset, write,
                "", ImVec4(0.60f, 0.60f, 0.60f, 1.0f));
    ImGui::SameLine(); ImGui::TextUnformatted("Pending  (tess in flight)");
    plot_metric("##ready",   m_planetHist.pipe_ready,   count, offset, write,
                "", ImVec4(0.30f, 0.80f, 1.00f, 1.0f));
    ImGui::SameLine(); ImGui::TextUnformatted("Ready    (waiting for BLAS record)");
    plot_metric("##blasrec", m_planetHist.pipe_blas_pending, count, offset, write,
                "", ImVec4(1.00f, 0.55f, 0.30f, 1.0f));
    ImGui::SameLine(); ImGui::TextUnformatted("Recorded (BLAS fence pending)");
    plot_metric("##built",   m_planetHist.pipe_built,   count, offset, write,
                "", ImVec4(0.40f, 0.95f, 0.40f, 1.0f));
    ImGui::SameLine(); ImGui::TextUnformatted("Built    (BLAS done)");

    //--- BLAS recordings per frame (throughput at the render-thread side) ---
    ImGui::SeparatorText("BLAS recordings per frame");
    plot_metric("##rec_count", m_planetHist.cells_recorded, count, offset, write,
                "cells", ImVec4(0.80f, 0.80f, 0.30f, 1.0f));
    ImGui::TextDisabled("capped by StreamConfig::build_budget - controls how "
                        "many BLAS the compute queue does per frame.");

    ImGui::End();
}

