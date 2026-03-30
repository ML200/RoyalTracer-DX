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
void Editor::Draw(Scene& scene, Camera& camera, PassSystem& passes,
                  DLSSManager& dlss, float fps)
{
    if (!m_visible) return;

    ImGui_ImplDX12_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();

    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("View")) {
            ImGui::MenuItem("Scene",     nullptr, &m_visible);
            ImGui::MenuItem("Materials", nullptr, &m_showMaterials);
            ImGui::EndMenu();
        }
        ImGui::Separator();
        ImGui::Text("%.1f fps | %.2f ms", fps, fps > 0 ? 1000.0f / fps : 0.0f);
        ImGui::Text("  |  %zu models, %zu instances, %zu materials",
            scene.models.size(), scene.instances.size(), scene.materials.size());
        if (scene.tlasDirty)      { ImGui::SameLine(); ImGui::TextColored(ImVec4(1,0.6f,0,1), "[TLAS dirty]"); }
        if (scene.lightTreeDirty) { ImGui::SameLine(); ImGui::TextColored(ImVec4(1,0.3f,0.3f,1), "[LightTree dirty]"); }
        ImGui::EndMainMenuBar();
    }

    DrawScenePanel(scene);
    DrawCameraPanel(camera);
    DrawPassPipelinePanel(passes);
    DrawDLSSPanel(dlss);
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
void Editor::DrawCameraPanel(Camera& camera) {
    ImGui::SetNextWindowPos(ImVec2(10, 490), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(360, 160), ImGuiCond_FirstUseEver);

    if (!ImGui::Begin("Camera")) { ImGui::End(); return; }

    ImGui::DragFloat("FOV",        &camera.fovDegrees, 0.5f, 10.0f, 170.0f);
    ImGui::DragFloat("Near Plane", &camera.nearPlane,  0.00001f, 0.00001f, 1.0f, "%.5f");
    ImGui::DragFloat("Far Plane",  &camera.farPlane,   10.0f, 100.0f, 100000.0f);
    ImGui::DragFloat("Move Speed", &camera.moveSpeed,  0.1f, 0.1f, 100.0f);

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
void Editor::DrawDLSSPanel(DLSSManager& dlss) {
    ImGui::SetNextWindowPos(ImVec2(380, 440), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(350, 120), ImGuiCond_FirstUseEver);

    if (!ImGui::Begin("DLSS-RR")) { ImGui::End(); return; }

    // DLSS-RR only supports Off, DLAA, Quality (MaxQuality), and Balanced.
    // Map combo indices to actual sl::DLSSMode enum values (which may not be contiguous).
    const char* modeLabels[] = { "Off", "DLAA", "Quality", "Balanced" };
    const sl::DLSSMode modeValues[] = {
        sl::DLSSMode::eOff,
        sl::DLSSMode::eDLAA,
        sl::DLSSMode::eMaxQuality,
        sl::DLSSMode::eBalanced
    };
    constexpr int modeCount = IM_ARRAYSIZE(modeLabels);

    // Find current combo index from active mode
    int currentIdx = 1; // default to DLAA
    for (int i = 0; i < modeCount; ++i) {
        if (dlss.mode == modeValues[i]) { currentIdx = i; break; }
    }

    if (ImGui::Combo("Mode", &currentIdx, modeLabels, modeCount))
        dlss.mode = modeValues[currentIdx];

    ImGui::TextDisabled("Render: %ux%u -> Display: %ux%u",
        dlss.RenderWidth(), dlss.RenderHeight(),
        dlss.DisplayWidth(), dlss.DisplayHeight());

    ImGui::End();
}

// ═════════════════════════════════════════════════════════════════
// Material Inspector
// ═════════════════════════════════════════════════════════════════
void Editor::DrawMaterialInspector(Scene& scene) {
    ImGui::SetNextWindowPos(ImVec2(740, 30), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(380, 550), ImGuiCond_FirstUseEver);

    if (!ImGui::Begin("Materials", &m_showMaterials)) { ImGui::End(); return; }

    // Material list (left)
    ImGui::BeginChild("MatList", ImVec2(120, 0), true);
    for (int i = 0; i < (int)scene.materials.size(); ++i) {
        auto& mat = scene.materials[i];
        ImVec4 preview(mat.Kd.x, mat.Kd.y, mat.Kd.z, 1.0f);
        ImGui::PushStyleColor(ImGuiCol_Text, preview);
        ImGui::Text("\xe2\x96\xa0");
        ImGui::PopStyleColor();
        ImGui::SameLine();
        char label[32]; snprintf(label, sizeof(label), "%d##mat", i);
        if (ImGui::Selectable(label, m_selectedMat == i))
            m_selectedMat = i;
    }
    ImGui::EndChild();

    ImGui::SameLine();

    // Properties (right)
    ImGui::BeginChild("MatProps", ImVec2(0, 0), false);
    if (m_selectedMat >= 0 && m_selectedMat < (int)scene.materials.size()) {
        auto& mat = scene.materials[m_selectedMat];
        ImGui::Text("Material %d", m_selectedMat);
        ImGui::Separator();

        bool changed = false;
        bool emissionChanged = false;

        changed |= ImGui::ColorEdit3("Albedo", &mat.Kd.x, ImGuiColorEditFlags_Float);

        // Emission (track separately for light tree)
        XMFLOAT3 prevKe = mat.Ke;
        bool emEdit = ImGui::ColorEdit3("Emission", &mat.Ke.x,
            ImGuiColorEditFlags_Float | ImGuiColorEditFlags_HDR);
        if (emEdit) { changed = true; emissionChanged = true; }

        if (mat.Ke.x > 0 || mat.Ke.y > 0 || mat.Ke.z > 0) {
            float intensity = std::max({mat.Ke.x, mat.Ke.y, mat.Ke.z});
            float prevIntensity = intensity;
            if (ImGui::DragFloat("Intensity", &intensity, 0.1f, 0.0f, 1000.0f)) {
                if (prevIntensity > 0.001f) {
                    float s = intensity / prevIntensity;
                    mat.Ke.x *= s; mat.Ke.y *= s; mat.Ke.z *= s;
                    changed = true; emissionChanged = true;
                }
            }
        }

        ImGui::Separator();
        changed |= ImGui::SliderFloat("Roughness", &mat.Pr_Pm_Ps_Pc.x, 0.0f, 1.0f);
        changed |= ImGui::SliderFloat("Metallic",  &mat.Pr_Pm_Ps_Pc.y, 0.0f, 1.0f);
        changed |= ImGui::SliderFloat("Specular",  &mat.Pr_Pm_Ps_Pc.z, 0.0f, 1.0f);
        changed |= ImGui::SliderFloat("Clearcoat", &mat.Pr_Pm_Ps_Pc.w, 0.0f, 1.0f);

        ImGui::Separator();
        changed |= ImGui::SliderFloat("Alpha", &mat.alphaThreshold, 0.0f, 1.0f);

        ImGui::Separator();
        ImGui::TextDisabled("Textures:");
        ImGui::Text("  Albedo: %s", mat.albedoTexID >= 0 ? std::to_string(mat.albedoTexID).c_str() : "none");
        ImGui::Text("  Normal: %s", mat.normalTexID >= 0 ? std::to_string(mat.normalTexID).c_str() : "none");
        ImGui::Text("  RMA:    %s", mat.rmaTexID    >= 0 ? std::to_string(mat.rmaTexID).c_str()    : "none");

        if (changed)
            scene.MarkMaterialsDirty(emissionChanged);
    } else {
        ImGui::TextDisabled("Select a material.");
    }
    ImGui::EndChild();

    ImGui::End();
}
