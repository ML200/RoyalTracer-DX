// ═══════════════════════════════════════════════════════════════════
// Editor/Editor.cpp — Model-level scene hierarchy, live materials
// ═══════════════════════════════════════════════════════════════════

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
                  ReSTIRSettings& restir, float fps, const FrameStats& stats)
{
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
            ImGui::MenuItem("Sun / Time of Day", nullptr, &m_showSun);
            ImGui::MenuItem("Materials",       nullptr, &m_showMaterials);
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
    if (m_showSun)       DrawSunPanel(camera);
    if (m_showMaterials) DrawMaterialInspector(scene);

    // Material re-upload happens via dirty flag checked in Renderer
    ImGui::Render();
}

void Editor::Render(ID3D12GraphicsCommandList* cmdList) {
    if (!m_visible) return;
    ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), cmdList);
}

// ═════════════════════════════════════════════════════════════════
// Scene Panel — one entry per loaded model
// ═════════════════════════════════════════════════════════════════
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

// ═════════════════════════════════════════════════════════════════
void Editor::DrawCameraPanel(Camera& camera, FlyCamController& flyCam) {
    ImGui::SetNextWindowPos(ImVec2(10, 490), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(360, 180), ImGuiCond_FirstUseEver);

    if (!ImGui::Begin("Camera")) { ImGui::End(); return; }

    ImGui::DragFloat("FOV",              &camera.fovDegrees, 0.5f, 10.0f, 170.0f);
    ImGui::DragFloat("Near Plane",       &camera.nearPlane,  0.00001f, 0.00001f, 1.0f, "%.5f");
    ImGui::DragFloat("Far Plane",        &camera.farPlane,   10.0f, 100.0f, 100000.0f);
    ImGui::DragFloat("Move Speed",       &flyCam.moveSpeed,  0.1f, 0.1f, 100.0f);
    ImGui::DragFloat("Mouse Sensitivity",&flyCam.mouseSensitivity, 0.01f, 0.01f, 2.0f, "%.2f");

    ImGui::Separator();
    ImGui::Text("Jitter frame: %u", camera.JitterFrame());

    ImGui::End();
}

// ═════════════════════════════════════════════════════════════════
void Editor::DrawPassPipelinePanel(PassSystem& passes) {
    ImGui::SetNextWindowPos(ImVec2(380, 30), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(350, 400), ImGuiCond_FirstUseEver);

    if (!ImGui::Begin("Pass Pipeline")) { ImGui::End(); return; }

    const char* stageNames[] = {
        "RayGen", "Compute", "FixedCompute", "Wavefront", "Barrier",
        "LoopStart", "LoopEnd", "PingSwap", "ClearSort", "Callable", "DLSS"
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

// ═════════════════════════════════════════════════════════════════
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

// ═════════════════════════════════════════════════════════════════
// Material Inspector
// ═════════════════════════════════════════════════════════════════
void Editor::DrawMaterialInspector(Scene& scene) {
    ImGui::SetNextWindowPos(ImVec2(740, 30), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(420, 700), ImGuiCond_FirstUseEver);

    if (!ImGui::Begin("Materials", &m_showMaterials)) { ImGui::End(); return; }

    // Material list (left)
    ImGui::BeginChild("MatList", ImVec2(180, 0), true);
    for (int i = 0; i < (int)scene.materials.size(); ++i) {
        const XMFLOAT4& kd = scene.materials.Kd[i];
        ImVec4 preview(kd.x, kd.y, kd.z, 1.0f);
        ImGui::PushStyleColor(ImGuiCol_Text, preview);
        ImGui::Text("\xe2\x96\xa0");
        ImGui::PopStyleColor();
        ImGui::SameLine();
        const char* name = (i < (int)scene.materialNames.size() && !scene.materialNames[i].empty())
            ? scene.materialNames[i].c_str() : nullptr;
        char label[128];
        if (name)
            snprintf(label, sizeof(label), "%d %s##mat", i, name);
        else
            snprintf(label, sizeof(label), "%d##mat", i);
        if (ImGui::Selectable(label, m_selectedMat == i))
            m_selectedMat = i;
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
        ImGui::SliderInt("M-cap##Temp", &rs.tempMcapGI, 0, 32);
    }
    if (ImGui::CollapsingHeader("Spatial", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Enable##Spat", &rs.enableSpatGI);
        ImGui::SliderInt("Radius Max##Spat", &rs.spatRadMaxGI, 4, 128);
        ImGui::SliderInt("Radius Min##Spat", &rs.spatRadMinGI, 4, 128);
        ImGui::SliderInt("Tries##Spat",      &rs.spatTriesGI, 2, 16);
    }
    if (ImGui::CollapsingHeader("Neighbor Rejection", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat("Normal dot min",    &rs.rejNormalDot,   0.0f, 1.0f);
        ImGui::SetItemTooltip("Reject neighbor if dot(nA, nB) falls below this.");
        ImGui::SliderFloat("Distance max",      &rs.rejDistance,    0.001f, 1.0f, "%.3f");
        ImGui::SetItemTooltip("Reject neighbor if |proj onto normal| exceeds this (world units).");
        ImGui::SliderFloat("Jacobian ratio min",&rs.rejJacobianMin, 0.0001f, 1.0f, "%.4f");
        ImGui::SetItemTooltip("Reject if Jn/Jc < this on either shift direction.");
        ImGui::SliderFloat("Jacobian ratio max",&rs.rejJacobianMax, 1.0f, 100.0f, "%.2f");
        ImGui::SetItemTooltip("Reject if Jn/Jc > this on either shift direction.");
    }
    if (ImGui::CollapsingHeader("Roughness Reuse")) {
        ImGui::SliderFloat("Min##Rough", &rs.reuseRoughnessMin, 0.0f, 1.0f);
        ImGui::SliderFloat("Max##Rough", &rs.reuseRoughnessMax, 0.0f, 1.0f);
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
        ImGui::SliderFloat("Sim Speed",       &s.simSpeed, 0.0f, 100.0f, "%.1fx");
        ImGui::SliderFloat("Start UTC Hours", &s.startUTCHours, 0.0f, 24.0f, "%.1f h");
        ImGui::SliderFloat("Night Speedup",   &s.nightSpeedup, 1.0f, 10.0f, "%.1fx");
    }
    if (ImGui::CollapsingHeader("Appearance", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat("Turbidity",      &s.turbidity, 1.0f, 10.0f, "%.1f");
        ImGui::SliderFloat("Sun Intensity",  &s.sunIntensity, 0.0f, 20.0f, "%.1f");
        ImGui::SliderFloat("Sky Intensity",  &s.skyIntensity, 0.0f, 20.0f, "%.1f");
    }

    ImGui::End();
}
