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
                  PassSystem& passes, DLSSManager& dlss, DLSSNRManager& dlssNR,
                  DLSSGSettings& dlssG,
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
            ImGui::MenuItem("DLSS 5 Neural Rendering", nullptr, &m_showDLSSNR);
            ImGui::MenuItem("DLSS Inputs",     nullptr, &m_showDlssInputs);
            ImGui::MenuItem("ReSTIR",          nullptr, &m_showReSTIR);
            ImGui::MenuItem("Initial Sampling", nullptr, &m_showInitialSampling);
            //ImGui::MenuItem("NRC",             nullptr, &m_showNRC); // NRC removed — UI panel disabled; restore for NIRC
            ImGui::MenuItem("Sun / Time of Day", nullptr, &m_showSun);
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
    if (m_showDLSS)      DrawDLSSPanel(camera, dlss, dlssG);
    if (m_showDLSSNR)    DrawDLSSNRPanel(dlssNR);
    if (m_showDlssInputs) DrawDlssInputsPanel(restir, dlss);
    if (m_showReSTIR)    DrawReSTIRPanel(restir, stats);
    if (m_showInitialSampling) DrawInitialSamplingPanel(restir);
    if (m_showNRC)       DrawNRCPanel(nrc);
    if (m_showSun)       DrawSunPanel(camera, stats);
    if (m_showMaterials) DrawMaterialInspector(scene, camera, restir);
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
void Editor::DrawDLSSPanel(Camera& camera, DLSSManager& dlss, DLSSGSettings& dlssG) {
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

    // ── RR model preset ─────────────────────────────────────────
    // sl::DLSSDPreset is a dense uint32: eDefault=0, then A=1 .. O=15, so the
    // combo index IS the enum value. Retain the full range for diagnostics;
    // SL 2.14.1 documents D/E/F models, with F the latest/default.
    static const char* kPresetLabels[] = {
        "Default (OTA)", "A", "B", "C", "D", "E", "F", "G",
        "H", "I", "J", "K", "L", "M", "N", "O"
    };
    constexpr int kPresetCount = IM_ARRAYSIZE(kPresetLabels);

    auto presetCombo = [&](const char* label, DLSSManager::PresetSlot slot) {
        int idx = (int)dlss.rrPresets[slot];
        if (idx < 0 || idx >= kPresetCount) idx = 0;
        if (ImGui::Combo(label, &idx, kPresetLabels, kPresetCount)) {
            const auto p = (sl::DLSSDPreset)idx;
            if (dlss.rrLinkPresets) {
                for (int i = 0; i < DLSSManager::kPresetSlotCount; ++i)
                    dlss.rrPresets[i] = p;
            } else {
                dlss.rrPresets[slot] = p;
            }
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "DLSS-RR denoiser/upscaler model.\n"
                "The bundled Streamline 2.14.1 SDK documents F as its latest/default\n"
                "RR transformer; D and E are earlier transformer models. Removed\n"
                "or unsupported letters may fall back to the default or fail.\n"
                "Switching preset drops temporal history for one frame.");
    };

    // Linked is the common case: one letter for every quality mode. Unlink to give
    // each mode its own, which is what DLSSDOptions actually models.
    ImGui::Checkbox("Link presets across quality modes", &dlss.rrLinkPresets);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "On: one preset applies to every quality mode.\n"
            "Off: DLAA / Quality / Balanced / Performance each get their own.");

    if (dlss.rrLinkPresets) {
        // Any slot works while linked — the write fans out to all six.
        presetCombo("RR Preset", DLSSManager::kPresetDLAA);
    } else {
        presetCombo("DLAA",          DLSSManager::kPresetDLAA);
        presetCombo("Quality",       DLSSManager::kPresetQuality);
        presetCombo("Balanced",      DLSSManager::kPresetBalanced);
        presetCombo("Performance",   DLSSManager::kPresetPerformance);
        presetCombo("Ultra Perf",    DLSSManager::kPresetUltraPerformance);
        presetCombo("Ultra Quality", DLSSManager::kPresetUltraQuality);
    }

    ImGui::SliderFloat("RR Responsivity", &dlss.rrResponsivity, -1.0f, 1.0f,
        "%.3f", ImGuiSliderFlags_AlwaysClamp);
    if (ImGui::IsItemDeactivatedAfterEdit())
        dlss.ForceReset();
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "Adjusts temporal response across the whole image.\n"
            "0 disables the override. Intended for RR preset F.\n"
            "Updates live; history resets once after you finish adjusting.");

    // One-frame RR history flush. The creeping-instability diagnostic: if an
    // established creep clears INSTANTLY on press and then slowly rebuilds,
    // the corruption lives in RR's recursive accumulator state (poisoned or
    // diverging history), not in the current-frame inputs.
    if (ImGui::Button("Reset RR history"))
        dlss.ForceReset();
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "Passes reset=eTrue to DLSS-RR for one frame, dropping all\n"
            "temporal history. Inputs and guides are untouched.");

    // The ACTUAL sub-pixel jitter amplitude. Scales the Halton offset at the
    // camera source, so the raygen samples AND the jitterOffset reported to
    // DLSS shrink together — truthful at every value, unlike the report-only
    // diagnostic below. History from a different amplitude is still valid
    // (nothing lied), so no reset per drag tick; one ForceReset on release
    // gives a clean-slate stability measurement at the new amplitude.
    ImGui::SliderFloat("Jitter offset", &camera.jitterScale, 0.0f, 1.0f, "%.3f");
    if (ImGui::IsItemDeactivatedAfterEdit())
        dlss.ForceReset();
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "Amplitude of the REAL camera jitter (raygen sampling offset).\n"
            "The offset reported to DLSS scales with it, so real and reported\n"
            "jitter stay matched at every value.\n"
            "1.0 = full Halton [-0.5,+0.5] (correct AA/upscaling input).\n"
            "0.0 = no jitter: primary rays hit identical sub-pixel positions\n"
            "every frame — aliased, but grazing-angle surfaces stop being\n"
            "resampled metres apart each frame. If the preset-F wobble dies\n"
            "as this approaches 0, the instability is jitter-driven surface\n"
            "undersampling, not a jitter-transform bug.\n"
            "DLSS history resets once on slider release.");

    // Diagnostic: scales the jitter REPORTED to DLSS without touching the offset
    // the raygen samples with. Off {1,1} the two disagree on purpose — see
    // DLSSManager::jitterScale. History is invalid across a change, so drop it.
    // PER-AXIS: {-1,-1} is the full sign-convention experiment; {1,-1} the
    // Y-only inversion (pixel-y-down vs NDC-y-up trap). A misregistered report
    // shows as stair-stepping on shallow edges that BUILDS under a static
    // camera and washes out under motion (history dies), preferentially on
    // horizontal edges for a Y error.
    if (ImGui::SliderFloat2("DLSS jitter scale", dlss.jitterScale, -2.0f, 2.0f, "%.3f"))
        dlss.ForceReset();
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "DIAGNOSTIC. Scales only the jitterOffset handed to DLSS, per axis.\n"
            "Rendering still samples at the camera's actual offset, so anything\n"
            "except {1,1} reports an offset we did not actually sample with.\n"
            "{1,1} = truthful, and the only value correct by construction.\n"
            "{-1,-1} = full sign flip, {1,-1} = Y-only flip: if static-camera\n"
            "stair-steps on shallow edges straighten at one of these, the\n"
            "baked report convention is wrong on that axis and should be fixed.\n"
            "{0,0} tells DLSS there is no jitter at all.");

    // RCAS sharpening on the DLSS output. Deliberately not DLSSDOptions::sharpness,
    // which DLSS-RR ignores — this runs in Pass_postprocess_v8 after AgX.
    ImGui::SliderFloat("Sharpness", &dlss.sharpness, 0.0f, 1.0f, "%.2f");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "RCAS sharpening applied to the DLSS output after tonemapping.\n"
            "0 = off (the pass skips the taps entirely).\n"
            "Contrast-adaptive, so it re-tightens upscaler softness without\n"
            "ringing on high-contrast edges. Affects the live DLSS view only,\n"
            "not the debug slices.\n"
            "DLSS-RR ignores Streamline's own sharpness field, so this is a\n"
            "post-process step rather than a DLSS setting.");

    // Clamp emitter radiance fed to DLSS RR so big bright emitters don't sit on
    // the Reinhard rail (where the denoiser/inverse amplify error into artefacts).
    ImGui::Checkbox("Clamp emitter spikes (DLSS RR)", &dlss.clampEmitterSpikes);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "Luminance-clamps emitter radiance before the DLSS pre-tonemap.\n"
            "Fixes DLSS RR artefacts around a large bright emitter (e.g. a lamp)\n"
            "in an otherwise dark scene. Cap = DLSS_EMITTER_CAP in Pass_shading.");

    // ── RR guide kill-switches ──────────────────────────────────
    // Ticked = guide fed normally. Unticked = Pass_shading overwrites it with
    // a neutral constant field (tag stays — RR requires these buffers), so the
    // creeping-instability hunt can bisect which guide it follows. Any change
    // invalidates history semantics -> ForceReset.
    ImGui::SeparatorText("Guide inputs");
    ImGui::TextDisabled("Untick to feed a neutral field instead");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(
            "Diagnostic bisection for the creeping preset-F instability.\n"
            "Unticked guides are overwritten with neutral values at the end of\n"
            "Pass_shading: depth 0 (far), MV 0, normals 0, roughness 1,\n"
            "diffuse albedo white, spec albedo black, spec MV 0. Color input\n"
            "cannot be disabled (it is the signal being denoised).\n"
            "History resets on every change.");
    {
        auto guideToggle = [&](const char* label, bool& off) {
            bool on = !off;
            if (ImGui::Checkbox(label, &on)) { off = !on; dlss.ForceReset(); }
        };
        guideToggle("Depth##guide",            dlss.guideOffDepth);
        guideToggle("Motion vectors##guide",   dlss.guideOffMV);
        guideToggle("Normals##guide",          dlss.guideOffNormals);
        guideToggle("Roughness##guide",        dlss.guideOffRough);
        guideToggle("Diffuse albedo##guide",   dlss.guideOffAlbedo);
        guideToggle("Specular albedo##guide",  dlss.guideOffSpecAlb);
        guideToggle("Specular MV##guide",      dlss.guideOffSpecMV);

        // The one guide that may be truly ABSENT: drop the tag instead of
        // zeroing, RR then uses its internal specular tracking.
        bool tagged = !dlss.untagSpecMV;
        if (ImGui::Checkbox("Specular MV tag##guide", &tagged)) {
            dlss.untagSpecMV = !tagged;
            dlss.ForceReset();
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Unticked: the spec-MV buffer is not tagged at all (null tag),\n"
                "RR falls back to internal specular tracking. A different\n"
                "experiment than feeding zero spec MVs via the box above.");
    }

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

            // DLSSGOptions exposes no preset field — the interpolation model is
            // picked by the driver/OTA, so there is nothing to select here.
            ImGui::TextDisabled("Preset: driver-selected (no app control)");
        }
    }

    ImGui::End();
}

//====================================
//DLSS NEURAL RENDERING PANEL
//====================================
//Status + settings for the optional NGX DLSS-NR post-process. This function
//only reads status and writes DLSSNRManager::settings / request flags — every
//NGX call happens inside the manager on the render thread.
void Editor::DrawDLSSNRPanel(DLSSNRManager& nr) {
    ImGui::SetNextWindowPos(ImVec2(380, 60), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(450, 470), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("DLSS 5 Neural Rendering", &m_showDLSSNR)) { ImGui::End(); return; }
    const auto& status = nr.GetStatus();
    ImGui::TextWrapped("Experimental native integration using the signed 310.8.0 runtime.");
    ImGui::TextWrapped("%s", status.backendText.c_str());
    const bool available = status.backend != DLSSNRManager::BackendState::eStubNoSdk &&
                           status.backend != DLSSNRManager::BackendState::eRuntimeMissing;
    ImGui::BeginDisabled(!available);
    ImGui::Checkbox("Enable", &nr.settings.enabled);
    static const char* modelStyles[] = { "Default", "Natural", "Cinematic" };
    static const char* renderPresets[] = { "Default", "Preset 1", "Preset 2", "Preset 3" };
    static_assert(IM_ARRAYSIZE(modelStyles) == dlssnr::kModelStyleCount);
    static_assert(IM_ARRAYSIZE(renderPresets) == dlssnr::kRenderPresetCount);
    ImGui::Combo("Model style", &nr.settings.modelStyle, modelStyles, IM_ARRAYSIZE(modelStyles));
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Selects the NR rendering style. Changing it restarts NR and clears its history.");
    ImGui::Combo("Render preset", &nr.settings.renderPreset, renderPresets, IM_ARRAYSIZE(renderPresets));
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Requests an embedded NR model preset. A preset absent from the runtime may use its default model.");
    ImGui::EndDisabled();
    ImGui::TextWrapped("Applies to the reconstructed scene view, after tone mapping and before the editor.");
    ImGui::BeginDisabled(!nr.settings.enabled);
    ImGui::SliderFloat("Intensity", &nr.settings.intensity, 0.0f, 1.0f, "%.2f");
    ImGui::SliderFloat("Local Tone", &nr.settings.localToneStrength, 0.0f, 1.0f, "%.2f");
    ImGui::SliderFloat("Local Structure", &nr.settings.localStructureStrength, 0.0f, 1.0f, "%.2f");
    ImGui::SliderFloat("Skin Structure", &nr.settings.skinStructureStrength, -1.0f, 1.0f, "%.2f");
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("-1 uses the model default.");
    ImGui::Checkbox("Auto Skin Mask", &nr.settings.useAutoMask);
    if (ImGui::Button("Reset history")) nr.ForceReset();
    ImGui::EndDisabled();
    ImGui::TextDisabled("Control changes restart the model on the next frame.");
    ImGui::Text("Successful evaluations: %llu", (unsigned long long)status.evalCount);
    if (!status.lastResult.empty()) ImGui::TextWrapped("%s", status.lastResult.c_str());
    if (ImGui::TreeNode("Runtime details")) {
        ImGui::TextWrapped("Path: %s", status.runtimePath.c_str());
        ImGui::Text("Version: %s", status.runtimeVersion.c_str());
        ImGui::TextWrapped("%s", status.signatureText.c_str());
        ImGui::TextWrapped("The native bridge accepts only the exact tested runtime hash.");
        ImGui::TreePop();
    }
    ImGui::End();
}


//====================================
//MATERIAL INSPECTOR
//====================================
void Editor::DrawMaterialInspector(Scene& scene, Camera& camera, ReSTIRSettings& restir) {
    ImGui::SetNextWindowPos(ImVec2(740, 30), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(420, 700), ImGuiCond_FirstUseEver);

    if (!ImGui::Begin("Materials", &m_showMaterials)) { ImGui::End(); return; }

    // Global texture filtering. Off = nearest-texel point sampling for ALL material
    // textures (crisp pixel-art / Minecraft); on = hardware bilinear/aniso. Rides
    // the rs-consts block (slot 42 -> pt_pointFilter) consumed by SampleMaterialTex.
    bool texInterp = (restir.texturePointFilter == 0);
    if (ImGui::Checkbox("Texture interpolation", &texInterp))
        restir.texturePointFilter = texInterp ? 0 : 1;
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Off = nearest-texel point sampling for every texture\n"
                          "(crisp pixel-art / Minecraft look). On = bilinear/aniso.");

    ImGui::Checkbox("Force diffuse opaque (debug)", &restir.forceDiffuseMats);
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Every material becomes an opaque Lambertian surface: albedo and\n"
                          "emission are kept, transmission / specular / metal / coat / sheen /\n"
                          "thin glass / SSS are forced off at the material decoder (all passes\n"
                          "see the same forced material, so ReSTIR targets stay consistent).\n"
                          "Alpha cutout is untouched. Live-toggleable; history re-settles in a\n"
                          "frame like any material edit.");
    ImGui::Separator();

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
                ImGui::SetTooltip("Volume absorption color (solid glass)\n"
                                  "Thin glass: flat per-surface transmission tint\nWhite = no absorption");

            //ensure the SoA slot exists (older materials predate this field)
            if (i >= (int)mats.thinGlass.size()) mats.thinGlass.resize(i + 1, 0u);
            bool thin = mats.thinGlass[i] != 0u;
            if (ImGui::Checkbox("Thin glass (Fresnel only)", &thin)) {
                mats.thinGlass[i] = thin ? 1u : 0u;
                changed = true;
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "Single-surface glass: reflects a roughness-aware Fresnel lobe and transmits\n"
                    "the rest straight through (no refraction, no medium/absorption). Shadow rays\n"
                    "attenuate by (1-F)*Tf instead of blocking.\n"
                    "Works live as long as Opacity < 1 (any transmissive material is already\n"
                    "non-opaque geometry). A material that was fully opaque (Opacity = 1) at load\n"
                    "needs a scene reload to become shadow-passable.");
        }

        // ── Subsurface scattering ────────────────────────────────
        if (ImGui::CollapsingHeader("Subsurface")) {
            //ensure the SoA slots exist (older materials predate these fields)
            if (i >= mats.sssEnable.size()) mats.sssEnable.resize(i + 1, 0u);
            if (i >= mats.sssAlbedo.size()) mats.sssAlbedo.resize(i + 1, DirectX::XMFLOAT3(1.0f, 1.0f, 1.0f));
            if (i >= mats.sssRadius.size()) mats.sssRadius.resize(i + 1, 0.0f);
            if (i >= mats.sssPhaseG.size()) mats.sssPhaseG.resize(i + 1, 0.0f);
            if (i >= mats.sssWeight.size()) mats.sssWeight.resize(i + 1, 1.0f);

            bool en = mats.sssEnable[i] != 0u;
            if (ImGui::Checkbox("Enable SSS", &en)) {
                mats.sssEnable[i] = en ? 1u : 0u;
                changed = true;
            }
            changed |= ImGui::ColorEdit3("SSS Color", &mats.sssAlbedo[i].x, ImGuiColorEditFlags_Float);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Single-scattering albedo (the inside colour).\nCarried by the random walk's albedo product.");
            changed |= ImGui::SliderFloat("Radius", &mats.sssRadius[i], 0.0005f, 50.0f, "%.4f", ImGuiSliderFlags_Logarithmic);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Scatter distance / mean free path (world units), log scale.\nsigma_t = 1/radius. Small = dense/opaque, large = translucent.\nUseful range is relative to the object's thickness.");
            changed |= ImGui::SliderFloat("Phase g", &mats.sssPhaseG[i], -0.95f, 0.95f);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Henyey-Greenstein anisotropy.\n0 = isotropic, >0 forward, <0 backward scattering.");
            changed |= ImGui::SliderFloat("Entry weight", &mats.sssWeight[i], 0.0f, 1.0f);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Probability scale for entering the medium vs reflecting.\np_enter = weight * Fresnel-transmittance. 0 = pure reflection, 1 = mostly subsurface.");
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
void Editor::DrawReSTIRPanel(ReSTIRSettings& rs, const FrameStats& stats) {
    ImGui::SetNextWindowSize(ImVec2(340, 460), ImGuiCond_FirstUseEver);
    //"###ReSTIR" keeps the saved window layout/ID while the title reflects the
    //panel's widened role (integrator select + the deprecated ReSTIR controls).
    if (!ImGui::Begin("Integrator###ReSTIR")) { ImGui::End(); return; }

    //====================================
    //INTEGRATOR SELECT
    //====================================
    {
        const char* kIntegrators[] = { "Path tracer", "ReSTIR (deprecated)" };
        ImGui::Combo("Integrator", &rs.integratorMode, kIntegrators, 2);
        ImGui::SetItemTooltip("Path tracer: the clean RIS-free kernel (Pass_pt_v8) — per-vertex light-"
                              "tree NEE + sun NEE with balance-heuristic MIS, same transport math as the "
                              "ReSTIR raygen but accumulating radiance directly. Every reservoir pass is "
                              "skipped; disable SHaRC to use it as a reference (its sampling knobs live in "
                              "the Initial Sampling panel: samples/pixel, bounce caps, RR).\nReSTIR: the "
                              "deprecated reservoir pipeline, kept selectable for A/B comparison.");
    }
    const bool ptActive = (rs.integratorMode == 0);
    if (ptActive) {
        ImGui::SeparatorText("SHaRC indirect lighting");
        if (stats.cacheTimingMask != 0u) {
            ImGui::Text("GPU ms: prepare %.2f | train %.2f | resolve %.2f | PT %.2f",
                stats.cachePassMs[0], stats.cachePassMs[1], stats.cachePassMs[2], stats.cachePassMs[3]);
            ImGui::SetItemTooltip("Actual GPU dispatch timestamps from the previous completed frame. "
                "PT includes cache lookup and any active cache debug view. Training keeps running independently.");
        }
        ImGui::Checkbox("Enable radiance cache", &rs.sharcEnabled);
        ImGui::SetItemTooltip("Caches the DIFFUSE lobe's outgoing radiance at secondary vertices of any "
            "diffuse-bearing material; a hit supplies that lobe and the vertex continues with its sheen, "
            "coat and GGX layers on the exact tracer, so nothing specular is ever cached. The primary "
            "vertex always stays exact. Uncertain entries keep tracing. Disable for the uncached reference.");
        ImGui::BeginDisabled(!rs.sharcEnabled);
        const char* cacheViews[] = { "Off", "Cells", "Cell lighting", "Guiding" };
        ImGui::Combo("Cache debug view", &rs.sharcDebugMode, cacheViews, IM_ARRAYSIZE(cacheViews));
        ImGui::SetItemTooltip("Projects the nearest stored cache cells onto visible surfaces. "
            "Bypasses denoising and interpolation; training and normal path tracing keep running. "
            "Off restores the previous display view.");
        if (rs.sharcDebugMode != 0) {
            if (rs.sharcDebugMode == SHARC_DEBUG_GUIDING) {
                ImGui::Checkbox("Show receivers instead of targets", &rs.sharcDebugCoarse);
                ImGui::SetItemTooltip("Targets: surfaces guided samples are aimed at. Receivers: how "
                    "strongly each surface itself guides.");
            } else {
                ImGui::Checkbox("Show other query level", &rs.sharcDebugCoarse);
                ImGui::SetItemTooltip("By default the view shows the distance level most rendering queries "
                    "sample at each pixel. This shows the other level of the pair, which receives "
                    "proportionally fewer training deposits.");
            }
            if (rs.sharcDebugMode == SHARC_DEBUG_CELLS)
                ImGui::TextWrapped("Cell colors: dim = warming, bright = confident. Dark grey = missing, "
                    "dark red = bucket full (insert pending), slate = surface never cached "
                    "(no broad lobe: glossy without a diffuse share, transmitting, SSS, "
                    "inside a medium, or steep normal map).");
            else if (rs.sharcDebugMode == SHARC_DEBUG_LIGHTING)
                ImGui::TextWrapped("Stored broad-share outgoing radiance (direct + indirect through the diffuse "
                    "and rough-GGX lobes, normal-incidence view), including untrusted samples. Magenta = missing, "
                    "dark red = bucket full, slate = no broad lobe here, amber = no resolved samples, black = stored zero.");
            else if (!rs.sharcDebugCoarse)
                ImGui::TextWrapped("Guiding targets: surfaces whose coarse patch is held by some receiver, "
                    "shown at the brightness guiding believes them to have (normal exposure), fading as "
                    "receivers stop referencing them. Dark grey = not a target, slate = no diffuse lobe "
                    "here. Targets are recorded only while this view is on, so allow a few frames.");
            else
                ImGui::TextWrapped("Guiding receivers at the primary vertex: green rises with the guided "
                    "fraction of diffuse samples (relative to the cap), blue with the number of stored "
                    "patches. Magenta = no receiver cell trained yet, amber = too few irradiance "
                    "observations, slate = no diffuse lobe here.");
        }
        ImGui::SliderInt("Minimum cell size (log2 metres)", &rs.sharcCellSizeExponent, -6, 4);
        ImGui::SliderFloat("Distance grid scale", &rs.sharcLodScale, 0.001f, 0.1f, "%.3f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderInt("Training tile width", &rs.sharcUpdateStride, 2, 8);
        ImGui::SetItemTooltip("One rotating training path per tile each frame. 4 means 1/16 of the pixels. "
            "Smaller tiles refill new regions faster.");
        ImGui::SliderInt("Cache path depth", &rs.sharcTrainBounces, 4, 64);
        ImGui::SetItemTooltip("Depth cap for training paths only; cache misses render with the Initial Sampling "
            "budget. Training paths normally end earlier by cache resampling or by roulette on their "
            "suffix throughput.");
        ImGui::SliderInt("Training roulette starts", &rs.sharcTrainRrDepth, 2, 32);
        ImGui::SetItemTooltip("From this depth, training survival follows the BSDF throughput accumulated "
            "since the last registered vertex (floor 0.1), so dark suffixes stop early.");
        ImGui::SliderInt("Minimum effective samples", &rs.sharcMinSamples, 8, 256);
        ImGui::SliderInt("History updates per cell", &rs.sharcHistoryFrames, 8, 256);
        ImGui::SetItemTooltip("History decays when the cell receives new samples. Sparse cells retain "
            "their learning between visits instead of losing it every frame.");
        ImGui::SliderInt("Unused cell lifetime", &rs.sharcMaxAge, 32, 4096);
        ImGui::SetItemTooltip("Base lifetime in frames. Sparse cells adapt up to four times this "
            "based on revisit intervals. Obsolete distance levels are still reclaimed immediately.");
        ImGui::SliderFloat("Footprint / cell width", &rs.sharcQueryFootprint, 0.5f, 8.0f, "%.1f");
        ImGui::SetItemTooltip("Path spread a secondary vertex needs before it may terminate into the cache, in "
            "coarser-cell widths (full acceptance at twice this). Cells scale with camera distance, so high "
            "values make distant first bounces trace on instead of terminating.");
        ImGui::SeparatorText("Cache-driven path guiding");
        ImGui::Checkbox("Guide diffuse samples toward bright cached patches", &rs.sharcGuideEnabled);
        ImGui::SetItemTooltip("Training paths record the bright cached patches they see from each coarse "
            "receiver cell. Diffuse-lobe samples then aim at those patches as a MIS-weighted mixture "
            "with cosine sampling, so small bright openings such as a lit doorway are found far more "
            "often. Unbiased: only the sampling density changes. Off = plain cosine sampling.");
        ImGui::BeginDisabled(!rs.sharcGuideEnabled);
        ImGui::SliderFloat("Guided fraction cap", &rs.sharcGuideMax, 0.0f, 0.9f, "%.2f");
        ImGui::SetItemTooltip("Upper bound on the share of diffuse samples that follow the patches. The "
            "actual share is this times the fraction of the cell's mean incident light the patches "
            "explain, so open areas fall back to cosine sampling on their own.");
        ImGui::SliderInt("Receiver cell (log2 x cache cell)", &rs.sharcGuideLevelOffset, 1, 6);
        ImGui::SetItemTooltip("Receiver cells and patches are this many doublings coarser than the cache "
            "grid. 2 = 0.5 m near the camera. Changing it resets the cache.");
        ImGui::SliderFloat("Patch radius / cell width", &rs.sharcGuideRadius, 0.25f, 2.0f, "%.2f");
        ImGui::SetItemTooltip("Bounding radius of a patch cone in patch-cell widths. Larger cones cover more "
            "of an opening but concentrate samples less.");
        ImGui::SliderInt("Guided path vertices", &rs.sharcGuideDepth, 1, 7);
        ImGui::SetItemTooltip("Deepest path vertex that is guided and trained (1 = primary only). The first "
            "two carry the visible noise; deeper vertices mostly end in the cache and cost training time.");
        ImGui::SliderInt("Patch lifetime (frames)", &rs.sharcGuideLifetime, 8, 2048);
        ImGui::SetItemTooltip("An unseen patch loses half its weight every this many frames and is dropped "
            "after four lifetimes, so closed doors and switched-off lights stop attracting samples.");
        ImGui::Checkbox("Guide training paths", &rs.sharcGuideTrain);
        ImGui::SetItemTooltip("Training paths use the same mixture, so a patch found once is revisited and "
            "refreshed quickly and the cache itself learns bright openings faster. Patches are still "
            "discovered with this off.");
        ImGui::EndDisabled();
        if (ImGui::Button("Reset radiance cache")) rs.sharcReset = true;
        ImGui::TextDisabled("205 MiB persistent cache + guide table; regular path tracer only");
        ImGui::EndDisabled();
        ImGui::SeparatorText("ReSTIR lite (diffuse primary vertex)");
        if ((stats.cacheTimingMask & 0x30u) != 0u) {
            ImGui::Text("GPU ms: shift %.3f | merge %.3f  (sum %.3f)",
                stats.cachePassMs[4], stats.cachePassMs[5], stats.cachePassMs[4] + stats.cachePassMs[5]);
            ImGui::SetItemTooltip("GPU dispatch timestamps of the lite passes from the previous completed frame.");
        }
        ImGui::Checkbox("Resample the primary diffuse lobe", &rs.liteEnabled);
        ImGui::SetItemTooltip("The primary vertex's diffuse lobe leaves the path tracer: its NEE samples and "
            "whatever its scatter ray finds (an emitter, the sky, or the secondary vertex with its cached or "
            "path-traced outgoing radiance) "
            "become reservoir candidates that paired spatial reuse resamples before shading. "
            "Every other lobe and every cache miss stays on the path tracer. The winner is shaded with the "
            "exact lobe and its own traced visibility.");
        ImGui::BeginDisabled(!rs.liteEnabled);
        ImGui::Checkbox("Spatial reuse", &rs.liteSpatial);
        ImGui::SetItemTooltip("Paired spatial reuse only; temporal reuse was removed because spatial alone "
            "suffices and reprojected history left artifacts after denoising.");
        ImGui::SliderInt("Spatial confidence cap", &rs.liteSpatMcap, 1, 64);
        ImGui::SliderInt("Spatial partners", &rs.liteSpatSlots, 0, 3);
        ImGui::SetItemTooltip("Self-inverting pair tables: a partner's partner is the pixel itself, so one "
            "shadow ray per pair serves both sides of the pairwise MIS.");
        ImGui::SliderFloat("Pair distance (px std dev)", &rs.liteReuseSigma, 2.0f, 40.0f, "%.1f");
        ImGui::SetItemTooltip("Regenerates the pair tables when changed.");
        ImGui::SliderFloat("Reuse normal cone (cos)", &rs.tempNormalSimCos, -1.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("Reuse plane distance / camera distance", &rs.tempPlaneDist, 0.0f, 0.5f, "%.3f");
        ImGui::SliderFloat("Contribution weight clamp", &rs.ucwClampMax, 0.0f, 100000.0f, "%.0f", ImGuiSliderFlags_Logarithmic);
        ImGui::Checkbox("Unshadowed reuse targets", &rs.liteUnshadowedTargets);
        ImGui::SetItemTooltip("Off (default): every reuse target traces its own ray (one per partner) and the "
            "estimator is exact. On (A/B): reuse targets ignore visibility and only the winner traces a "
            "shadow ray (about 0.2 ms cheaper); generation still drops occluded NEE candidates, so the "
            "reuse MIS can slightly over-credit a neighbour right at its own shadow edges (the classic "
            "visibility-reuse compromise, a thin band).");
        ImGui::SameLine();
        ImGui::Checkbox("Show lite only", &rs.liteDebugView);
        ImGui::SetItemTooltip("Displays only the resampled diffuse contribution (the path tracer's part is dropped).");
        ImGui::EndDisabled();
    }
    ImGui::SeparatorText(ptActive ? "ReSTIR (deprecated, inactive)" : "ReSTIR (deprecated)");
    ImGui::BeginDisabled(ptActive);

    if (ImGui::CollapsingHeader("Temporal", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Enable##Temp", &rs.enableTempGI);
        ImGui::SliderInt("M-cap##Temp", &rs.tempMcapGI, 0, 128);
        ImGui::Checkbox("Disable correlation reduction (A/B)##Temp", &rs.disableCorrReduction);
        ImGui::SetItemTooltip("Turns off the duplication-map decorrelation. Normally the temporal "
                              "confidence cap collapses toward 1 as a sample is shared across "
                              "neighbours (lerp(M-cap,1,pow(D,corr-power)), see the slider below). "
                              "Cell reuse spreads good samples, raising D, so this caps them back down "
                              "and fights the reuse. Disable to let well-reused samples keep "
                              "accumulating confidence (suspected cause of weak cell-reuse quality).");
        ImGui::SliderFloat("Corr-reduction power##Temp", &rs.corrReductionPow, 0.005f, 0.5f,
                           "%.3f", ImGuiSliderFlags_Logarithmic);
        ImGui::SetItemTooltip("Dup-map exponent e: effMcap = lerp(M-cap, 1, pow(D, e)). SMALLER = "
                              "stronger decorrelation (each halving doubles the strength in log "
                              "space). 0.1 = original tuning; at 0.025 with M-cap 20 a single "
                              "17x17 duplicate caps history at ~3.5 and D=0.1 at ~2.1. Unique "
                              "samples (D=0) always keep the full M-cap.");
        //("Disable x1 direct" diagnostic removed — §6.1 made x1 sun/env reservoir
        // candidates, so the toggle had nothing left to zero.)
        ImGui::Checkbox("Surface reproj only (no specular MV)##Temp", &rs.noSpecReproj);
        ImGui::SetItemTooltip("Forces temporal reuse to self-reproject the DI reservoir instead of the "
                              "stochastic specular reprojection. The default flips per-pixel-per-frame "
                              "between self and the reflection's virtual position (rSpec < specularity); "
                              "that coin flip gives a random subset of pixels inconsistent history lineage "
                              "so their confidence M never accumulates - the flat high-variance band layered "
                              "over the clean (self-reprojected) pixels, worst at grazing angles / near lights. "
                              "Enable to give every pixel a stable history so M accumulates uniformly.");
        ImGui::Checkbox("Disable reuse visibility (defer to resolve)##Temp", &rs.disableReuseVis);
        ImGui::SetItemTooltip("Takes the reconnection shadow ray OUT of temporal AND spatial reuse and "
                              "applies it ONCE where the spatial pass resolves F*W. Small/far lights behind "
                              "thin high-frequency occluders (fences, poles) make x1->x2 visibility a near-"
                              "binary function of the shading point; sub-pixel jitter slides that point across "
                              "the occluder so vis flips 1->0, p_n=ph*vis collapses and the reservoir resets - "
                              "reuse dies (clean top vs noisy side of a container). With this on the reuse "
                              "target is UNSHADOWED so light selection keeps accumulating M under jitter, and "
                              "the true visibility multiplies the final contribution once (unbiased). Residual "
                              "single-ray shadow noise is left for DLSS RR. Raygen's NEE/sun visibility is "
                              "untouched.");
        ImGui::BeginDisabled(!rs.disableReuseVis);
        ImGui::Checkbox("    + disable final resolve vis (unshadowed GI)##Temp", &rs.disableFinalVis);
        ImGui::SetItemTooltip("Diagnostic. Also skips the ONE deferred reconnection shadow ray at the "
                              "spatial resolve, so the reservoir GI is shown fully UNSHADOWED (the upper-bound "
                              "brightness / what the unshadowed selection picks). Only meaningful with the "
                              "toggle above on. The NRC gather follows it, training on the same unshadowed GI "
                              "it displays. Expect light to leak through thin occluders - it's for isolating "
                              "how much of the side-face variance is reuse-reset vs intrinsic shadow-ray noise.");
        ImGui::EndDisabled();

        ImGui::Separator();
        ImGui::SliderFloat("Reproj normal cone (cos)##Temp", &rs.tempNormalSimCos, -1.0f, 1.0f, "%.2f");
        ImGui::SetItemTooltip("Reject the reprojected temporal candidate when dot(normal, reprojNormal) <= this "
                              "- stops reuse across a corner / different face. -1 = accept all normals.");
        ImGui::SliderFloat("Reproj plane dist (xCamDist)##Temp", &rs.tempPlaneDist, 0.0f, 1.0f, "%.3f");
        ImGui::SetItemTooltip("Reject the reprojected candidate when its hit is off this pixel's tangent plane by "
                              "more than this FRACTION of the distance to camera (depth-discontinuity reject, same "
                              "form as the SPMIS pass). Lower = stricter; ~1.0 = off.");
        ImGui::SliderFloat("Jacobian clamp T##Temp", &rs.tempJacClamp, 1.0f, 100.0f, "%.1f");
        ImGui::SetItemTooltip("Reconnection-shift Jacobian ratio is clamped to [1/T, T]. A grazing reconnection "
                              "(cos -> 0 near a corner) blows the ratio up; this bounds it so the resampling "
                              "weight can't spike into fireflies. Lower = stronger suppression, more bias.");
        ImGui::SliderFloat("UCW clamp (W)##Temp", &rs.ucwClampMax, 0.0f, 100000.0f, "%.0f",
                           ImGuiSliderFlags_Logarithmic);
        ImGui::SetItemTooltip("Clamps the reservoir weight W = w_sum / p_hat for temporal AND SPMIS reuse outputs. "
                              "W is unbounded; a tiny target luminance near a grazing/occluded surface spikes it and "
                              "the spike feeds back every frame into a diverging firefly ('exploding W'). Lower = "
                              "stronger firefly suppression / more bias on valid low-pdf samples. 0 = off. "
                              "(Raygen's single-frame W is left unclamped so small/far lights aren't darkened.)");
    }
    if (ImGui::CollapsingHeader("Spatial (SPMIS)", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Enable##Spat", &rs.enableSpatGI);
        ImGui::SetItemTooltip("Spatial reuse via the SPMIS global hash grid: raygen inserts each pixel's "
                              "cell, then materialized per-cell lists feed stochastic pairwise-MIS reuse "
                              "(Pass_spmis_* pipeline). The sliders below tune it.");
        ImGui::SliderInt("Reuse N (Ntilde)##SPMIS", &rs.spmisReuseN, 1, 32);
        ImGui::SetItemTooltip("Non-canonical reuse draws per pixel (NOT the inner-RIS count below). Higher "
                              "dilutes the canonical weight (collapse factor 1/(N+1) where reuse fails).");
        ImGui::SliderInt("RIS steps##SPMIS", &rs.spmisRisN, 1, 32);
        ImGui::SetItemTooltip("Inner-RIS candidate count per draw: each reuse draw RIS-picks one non-zero "
                              "cell member from this many uniform samples.");
        ImGui::SliderInt("Tile size (px)##SPMIS", &rs.spmisTileSize, 1, 64);
        ImGui::SetItemTooltip("Screen-space cell tile size: cell = (pixel/tile, jittered-quantized normal). "
                              "Larger = bigger, coarser cells (more reuse, less locality).");
        ImGui::SliderInt("M-cap##SPMIS", &rs.spmisMcap, 0, 200);
        ImGui::SetItemTooltip("Output confidence cap applied to the spatial result. 0 disables.");
        ImGui::SliderFloat("Jacobian reject##SPMIS", &rs.spmisJacThreshold, 1.0f, 100.0f, "%.1f");
        ImGui::SetItemTooltip("Reconnection-shift jacobian reject band: a shift whose jacobian falls outside "
                              "[1/T, T] is dropped (anti-firefly).");
        ImGui::SliderFloat("Normal cone (cos)##SPMIS", &rs.spmisNormalSimCos, -1.0f, 1.0f, "%.2f");
        ImGui::SetItemTooltip("Neighbor-similarity normal cone for the cell search: a different cell is only "
                              "hopped to if dot(n, n_neighbor) > this. Higher = stricter / more local; "
                              "-1 = accept all normals.");
        ImGui::SliderFloat("Plane dist (xCamDist)##SPMIS", &rs.spmisPlaneDist, 0.0f, 1.0f, "%.3f");
        ImGui::SetItemTooltip("Cell-search plane-distance rejection as a FRACTION of the distance to camera "
                              "(reject bad cells by geometry, same as regular ReSTIR). Rejects neighbour cells "
                              "whose primary hit is off the query's tangent plane (a depth discontinuity - "
                              "container edge vs background). Lower = stricter; ~1.0 = off.");
        ImGui::Checkbox("Confidence scaling (§4.3)##SPMIS", &rs.spmisConfidenceAdjust);
        ImGui::SetItemTooltip("§4.3 non-canonical confidence scaling (Ntilde/cell_size). OFF gives the "
                              "canonical ~no weight -> maximal neighbour reuse / equalization. ON boosts the "
                              "canonical (~1/(N+1)) -> weaker reuse but less dark-pepper.");

        ImGui::SeparatorText("Hash grid / cell search");
        ImGui::Checkbox("Cell jitter##SPMIS", &rs.spmisCellJitter);
        ImGui::SetItemTooltip("Per-frame jitter of the screen-tile origin (from time): cell boundaries "
                              "move every frame and average out under accumulation/temporal reuse. OFF = "
                              "static grid - boundary pixels keep the same cell neighbours every frame "
                              "(isolates boundary artifacts vs jitter noise). Grid rebuilds per frame, "
                              "always safe to toggle.");
        ImGui::SliderFloat("Normal fuzz##SPMIS", &rs.spmisNormalFuzz, 0.0f, 1.0f, "%.2f");
        ImGui::SetItemTooltip("Deterministic per-position tangent-plane jitter applied to the normal "
                              "before quantization in the cell hash - anti-aliases the discrete normal "
                              "buckets so curved surfaces don't band into cells. 0 = hard buckets.");
        ImGui::SliderInt("Normal bits##SPMIS", &rs.spmisNormalBits, 1, 4);
        ImGui::SetItemTooltip("Normal quantization bits per component in the cell key. More bits = finer "
                              "normal buckets = smaller, more orientation-local cells (less reuse); "
                              "1 bit = very coarse (walls and floor may share cells at fuzz > 0).");
        ImGui::SliderFloat("Search radius (px)##SPMIS", &rs.spmisSearchR0, 1.0f, 256.0f, "%.0f",
                           ImGuiSliderFlags_Logarithmic);
        ImGui::SetItemTooltip("Cell-search initial probe radius. Probes draw uniform offsets in a square "
                              "of this half-size, growing per probe (below).");
        ImGui::SliderFloat("Search growth##SPMIS", &rs.spmisSearchGrow, 1.0f, 2.0f, "%.2f");
        ImGui::SetItemTooltip("Per-probe radius multiplier: 1.0 = fixed radius, higher reaches farther "
                              "cells with later probes (12 probes at 1.25 span ~radius x 11.6).");
        ImGui::SliderInt("Search probes##SPMIS", &rs.spmisSearchIters, 4, 32);
        ImGui::SetItemTooltip("WRS cell-search probe count (batched 4-wide). More probes = better cell "
                              "mixing / farther reuse at higher select-pass cost.");
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
        ImGui::SliderFloat("Reconnect x2 min##Rough", &rs.reconnectRoughnessMin, 0.0f, 1.0f);
        ImGui::SetItemTooltip("Hybrid ON: the pin-criteria roughness — BOTH vertices of the reconnection "
                              "pair must be at least this rough for raygen to pin there (glossier pairs "
                              "postpone the pin via random replay). Hybrid OFF: the legacy reuse-time "
                              "reject of glossy reconnection vertices. Sky / light (NEE) ends are exempt.");
    }
    if (ImGui::CollapsingHeader("Hybrid Shift (ReSTIR PT)", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Enable##Hybrid", &rs.hybridShift);
        ImGui::SetItemTooltip("ReSTIR PT hybrid shift: reconnection + random replay in primary sample "
                              "space. Raygen pins the reconnection vertex at the FIRST vertex pair passing "
                              "the roughness+distance criteria; glossy prefixes are random-replayed at "
                              "reuse (temporal: compacted Pass_temp_replay indirect dispatch; spatial: "
                              "replay roles inside the unified Pass_spmis_shift) instead of being "
                              "rejected. OFF = legacy pin-at-x2 reconnection shift, zero replay (the "
                              "temporal queue stays empty and its dispatch is skipped).");
        ImGui::SliderFloat("Reconnect dist min (xCamDist)##Hybrid", &rs.reconnectDistMin,
                           0.0f, 0.25f, "%.4f", ImGuiSliderFlags_Logarithmic);
        ImGui::SetItemTooltip("Pin distance criterion: the reconnection segment must be at least this "
                              "FRACTION of the primary camera distance, or the pin postpones via replay. "
                              "Short segments make the area-measure geometric factor singular (corner "
                              "noise). 0 = off.");
        ImGui::SliderInt("Max pin depth k##Hybrid", &rs.rcMaxK, 2, 10);
        ImGui::SetItemTooltip("Deepest reconnection-vertex index raygen may pin. Replay cost per reuse "
                              "eval is k-2 bounces (TraceRay each), so this is the main perf lever for "
                              "glossy-prefix scenes. 2 = first-vertex pins only (zero replay even when "
                              "hybrid is on). Host-capped at 8 while lobe-indexed PSS is on.");
        ImGui::Checkbox("Lobe-indexed PSS (Enhanced supp 1)##Hybrid", &rs.lobeIndexedPss);
        ImGui::SetItemTooltip("Lobe-indexed paths: the extension estimator splits per sampled BSDF lobe "
                              "(single-lobe rho over conditional pdf; the lobe pmf peels out of the "
                              "stored F exactly like RR survival) and shifts PRESERVE the lobe sequence. "
                              "Replay forces the recorded lobe per bounce - no strategy re-roll flips at "
                              "offset pixels - which fixes the multilobe (plastic) variance regression "
                              "and makes replay cheaper (one lobe eval instead of the 4-lobe walk). "
                              "Jacobian bundles go conditional; MIS and footprint criteria stay marginal "
                              "(supp 3). Samples self-describe (RC_F_LOBES), so toggling is safe without "
                              "a reservoir reset.");
        ImGui::Separator();
        ImGui::Checkbox("Footprint pin criteria (Enhanced 4)##Hybrid", &rs.rcFootprint);
        ImGui::SetItemTooltip("ReSTIR PT Enhanced 4: dual footprint threshold + pdf-proxy glossiness "
                              "guard choose the pin instead of the material-roughness + distance pair "
                              "test. Lobe-aware (diffuse bounces on plastics pin cleanly) and scale-"
                              "adaptive (thresholds derive from the primary ray footprint). The distance "
                              "slider above is inert while this is on; 'Reconnect x2 min' acts as the "
                              "proxy sigma_min. Generation-side only.");
        ImGui::SliderFloat("Footprint c (Eq.5)##Hybrid", &rs.rcFpKappa, 0.001f, 2.56f, "%.4f",
                           ImGuiSliderFlags_Logarithmic);
        ImGui::SetItemTooltip("The paper's threshold constant c, implemented literally (Eq. 5). Larger = "
                              "stricter footprints = rarer but more robust reconnections (more replay). "
                              "Paper optimum 0.02 across scenes; diffuse scenes are insensitive, calibrate "
                              "on glossy content (their ablation: 0.005-0.64).");
        ImGui::Checkbox("Dual motion vectors (Enhanced 6.4)##Hybrid", &rs.dualMotionVectors);
        ImGui::SetItemTooltip("On a temporal geometry reject (moving-occluder disocclusion), retry the "
                              "history fetch at launchIndex - occluder screen motion (prev surface at the "
                              "reprojected pixel, projected with the current camera). Dual candidates use "
                              "the geometric REJECT band instead of the force-to-1 clamp (clamping their "
                              "far-field Jgeo is an energy bias at disocclusions).");
        ImGui::Checkbox("RGB shading weights (Enhanced 6.3)##Hybrid", &rs.rgbShadeWeights);
        ImGui::SetItemTooltip("Color-noise reduction: the spatial resolve accumulates vectorized "
                              "resampling weights and blends contributor chroma at the scalar chain's "
                              "luminance, instead of outputting the lone winner's color. Nearly free.");
    }
    ImGui::EndDisabled();
    ImGui::End();
}

// ─────────────────────────────────────────────────────────────────
void Editor::DrawInitialSamplingPanel(ReSTIRSettings& rs) {
    ImGui::SetNextWindowSize(ImVec2(340, 170), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Initial Sampling", &m_showInitialSampling)) { ImGui::End(); return; }

    ImGui::SliderInt("Initial samples", &rs.initialSamples, 1, 8);
    ImGui::SetItemTooltip("Independent initial path samples per pixel, combined via 1-pass RIS-over-N "
                          "into the ReSTIR reservoir (M=N). 1 = legacy single sample (byte-identical). "
                          "Higher = less initial-sample variance at ~linear raygen cost. Capped at 8.");

    ImGui::SliderInt("Max diffuse bounces", &rs.maxDiffuseBounces, 1, 32);
    ImGui::SetItemTooltip("Caps how many scattering events a path may take on materials with a diffuse "
                          "component — also metals, and specular/clearcoat lobes layered over diffuse. "
                          "Glass and translucent (SSS) materials bounce past this (refraction needs the "
                          "depth), up to 'Max bounces'. Default 3; lower = cheaper/less noisy diffuse GI.");

    ImGui::SliderInt("Max bounces", &rs.maxBounces, 2, 32);
    ImGui::SetItemTooltip("Hard path-length cap: the raygen loop runs depths [1, N). 2 = primary direct "
                          "only; higher = deeper glass/translucent paths (diffuse GI is bounded by 'Max "
                          "diffuse bounces' above). Capped at 32.");

    ImGui::SliderInt("RR start depth", &rs.rrStartDepth, 1, 32);
    ImGui::SetItemTooltip("Russian roulette begins at depth >= this. Lower = more aggressive termination "
                          "(cheaper, but starves ReSTIR's deep-GI initial samples); set to 32 (>= max "
                          "bounces) to effectively DISABLE RR. Default 3.");


    ImGui::End();
}

// ─────────────────────────────────────────────────────────────────
void Editor::DrawDlssInputsPanel(ReSTIRSettings& rs, DLSSManager& dlss) {
    ImGui::SetNextWindowSize(ImVec2(320, 150), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("DLSS Inputs", &m_showDlssInputs)) { ImGui::End(); return; }

    //order must match DlssInputDebugView's switch in Pass_postprocess_v8.hlsl
    static const char* kLayers[] = {
        "Off (slice stays black)",
        "Color input (g_dlssInput)",
        "Denoised output, raw (g_dlssOutput)",
        "Depth (window)",
        "Motion vectors (0.5 + mv*0.1)",
        "Normals (n*0.5+0.5)",
        "Diffuse albedo",
        "Specular albedo",
        "Roughness",
        "Spec motion vectors (0.5 + mv*0.1)",
        "Spec hit distance (window)",
        "Transparency",
        "Color pre-transparency",
        "Bias hint",
    };
    ImGui::Combo("Layer", &rs.dlssDebugLayer, kLayers, IM_ARRAYSIZE(kLayers));
    ImGui::SetItemTooltip("Renders the selected DLSS-RR input layer RAW (no exposure/AgX/dither, only a "
                          "per-layer range map + display gamma) into output slice 3. Color layers are "
                          "luminance-Reinhard compressed for display; MVs show +-10 px across "
                          "black..white. Guide layers are render-resolution (nearest-scaled when "
                          "upscaling); the raw denoised output is display-res.");
    if (rs.dlssDebugLayer == 3 || rs.dlssDebugLayer == 10) {
        ImGui::DragFloatRange2("Window (m)", &rs.dlssDebugDepthNear, &rs.dlssDebugDepthFar,
                               0.25f, 0.0f, 65000.0f, "near %.2f", "far %.2f");
        ImGui::SetItemTooltip("Depth / hit-dist are mapped LINEARLY from [near, far] onto the 8-bit "
                              "display. The display itself quantizes to 256 steps, so a wide window "
                              "shows contour bands on a perfectly smooth R32F buffer — tighten the "
                              "window around the surface under inspection; banding that SURVIVES a "
                              "tight window is real buffer content.");
    }
    ImGui::TextWrapped("Shown on output slice 3 — press C until the 4th view. "
                       "For edge instability, compare Normals / Depth / MVs / albedos against the "
                       "color input along the artifact edge: the guide that disagrees is the culprit.");

    //====================================
    //GUIDE SENTINEL (live anomaly stats over what RR is being fed)
    //====================================
    if (ImGui::CollapsingHeader("Guide sentinel", ImGuiTreeNodeFlags_DefaultOpen)) {
        const auto& gs = dlss.sentinel;
        ImGui::Text("frame %llu  |  max luma %.3f  max |MV| %.2f px  max |specMV| %.2f px",
                    (unsigned long long)gs.frame, gs.maxLuma, gs.maxMV, gs.maxSpecMV);
        ImGui::Text("pixels at luma cap: %u", gs.capCount);
        if (gs.mask != 0) {
            const uint32_t bx = gs.firstBad & 0xFFFFu, by = gs.firstBad >> 16;
            ImGui::TextColored(ImVec4(1, 0.3f, 0.3f, 1),
                "ANOMALY NOW: mask 0x%03X, %u px, first (%u,%u)",
                gs.mask, gs.badCount, bx ? bx - 1 : 0, by ? by - 1 : 0);
        } else {
            ImGui::TextColored(ImVec4(0.4f, 1, 0.4f, 1), "guides clean");
        }
        if (gs.lastFrame != 0) {
            const uint32_t lx = gs.lastBad & 0xFFFFu, ly = gs.lastBad >> 16;
            ImGui::TextColored(ImVec4(1, 0.8f, 0.3f, 1),
                "last anomaly: frame %llu, mask 0x%03X, first (%u,%u)",
                (unsigned long long)gs.lastFrame, gs.lastMask,
                lx ? lx - 1 : 0, ly ? ly - 1 : 0);
            ImGui::SetItemTooltip(
                "Mask bits: 0x001 color NaN/Inf  0x002 depth bad  0x004 MV NaN/Inf\n"
                "0x008 normal/rough NaN/Inf  0x010 roughness out of [0,1]\n"
                "0x020 specMV NaN/Inf  0x040/0x080 albedo NaN/Inf\n"
                "0x100 |MV|>256px  0x200 |specMV|>256px\n"
                "If an instability onset happens while this stays clean, the\n"
                "poison is NOT in the guide data — it's options/execution side.");
        } else {
            ImGui::TextDisabled("no anomaly seen this session");
        }
    }

    //====================================
    //D3D12 MESSAGES FROM THE EVALUATE WINDOW
    //====================================
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
void Editor::DrawSunPanel(Camera& camera, const FrameStats& stats) {
    ImGui::SetNextWindowSize(ImVec2(320, 300), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Sun / Time of Day")) { ImGui::End(); return; }

    auto& s = camera.sunSettings;

    if (ImGui::CollapsingHeader("Cumulus", ImGuiTreeNodeFlags_DefaultOpen)) {
        auto& c = camera.cumulusSettings;
        bool enabled = c.enabled > 0.5f;
        if (ImGui::Checkbox("Enable cumulus", &enabled)) c.enabled = enabled ? 1.0f : 0.0f;
        ImGui::Checkbox("Cache cloud density (experimental)", &camera.cumulusDensityCache);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Reuses nearby cloud density in world-space blocks. Builds gradually over 27 frames.\nFiltered detail, missing blocks and animated wind use procedural density.\nTurn off to compare with the original cloud shape; normals retain the procedural gradient.");
        if ((stats.cacheTimingMask & 0x1C0u) != 0u) {
            ImGui::Text("Cloud cache %.2f ms | Cloud rays %.2f ms | Sum %.2f ms",
                stats.cachePassMs[6], stats.cachePassMs[7]+stats.cachePassMs[8], stats.cachePassMs[6] + stats.cachePassMs[7]+stats.cachePassMs[8]);
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("GPU time from the previous frame. Includes the one-time noise bake on startup.\nCloud rays includes primary sky, secondary misses and clear atmosphere. Final shading and RR are outside these timers.");
        }
        ImGui::Text("Sky / air %.2f ms | Reflections / diffuse %.2f ms", stats.cachePassMs[7], stats.cachePassMs[8]);
        if (ImGui::Button("Daylight cumulus scene")) {
            c = CumulusSettings{};
            const SunSettings daylight{};
            s.latitude = daylight.latitude; s.longitude = daylight.longitude; s.dayOfYear = daylight.dayOfYear;
            s.sunIntensity = daylight.sunIntensity; s.skyIntensity = daylight.skyIntensity; s.turbidity = daylight.turbidity;
            s.simSpeed = 0.0f; s.startUTCHours = 11.0f;
        }
        ImGui::SameLine();
        if (ImGui::Button("Tall towers")) {
            c = CumulusSettings{}; c.coverage = 0.48f; c.thicknessKm = 4.5f; c.scale = 1.0f; c.extinction = 12.0f;
        }
        ImGui::SliderFloat("Cloud coverage", &c.coverage, 0.0f, 1.0f, "%.2f");
        ImGui::SliderFloat("Cloud base", &c.baseKm, 0.2f, 8.0f, "%.2f km");
        ImGui::SliderFloat("Cloud height", &c.thicknessKm, 0.3f, 8.0f, "%.2f km");
        ImGui::SliderFloat("Cloud size", &c.scale, 0.25f, 3.0f, "%.2fx");
        ImGui::SliderFloat("Cloud density", &c.extinction, 1.0f, 40.0f, "%.1f");
        ImGui::SliderFloat("Lobe and edge detail", &c.detail, 0.0f, 1.5f, "%.2f");
        ImGui::SliderFloat("Billow distortion", &c.fineDetail, 0.0f, 2.0f, "%.2f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Wind shear and bending of the whole cloud outline and its billows.\nKeeps billow sizes; reflections and shadows follow the distorted shape.\nUses Wind X/Z for direction, with a fixed direction when wind is zero.\n0 disables distortion and uses the original local shadow sampling.");
        ImGui::SliderFloat("Internal scattering", &c.multipleScattering, 0.0f, 3.0f, "%.2f");
        ImGui::SliderFloat("Cloud ambient light", &c.ambient, 0.0f, 3.0f, "%.2f");
        ImGui::SliderFloat("Wind X", &c.windX, -40.0f, 40.0f, "%.1f m/s");
        ImGui::SliderFloat("Wind Z", &c.windZ, -40.0f, 40.0f, "%.1f m/s");
        ImGui::SliderFloat("Cloud seed", &c.seed, 0.0f, 100.0f, "%.0f");
        int steps = (int)c.viewSteps, reflectionSteps = (int)c.reflectionSteps;
        if (ImGui::SliderInt("Sky samples", &steps, 16, 160)) c.viewSteps = (float)steps;
        int lightingSamples = (int)c.lightingSamples;
        if (ImGui::SliderInt("Cloud lighting samples", &lightingSamples, 0, 4)) c.lightingSamples = (float)lightingSamples;
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Lighting, shadowing and internal scattering samples per cloud crossing.\n0 evaluates every occupied sky sample for comparison.\n1-4 use weighted stochastic samples for RR; density and guides keep their full detail.");
        if (ImGui::SliderInt("Reflection samples", &reflectionSteps, 4, 32)) c.reflectionSteps = (float)reflectionSteps;
        int debug = (int)c.debugView;
        if (ImGui::Combo("Cloud view", &debug, "Rendered\0Opacity\0Cloud normals\0Cloud depth\0Cloud motion\0Depth spread\0")) c.debugView = (float)debug;
        ImGui::SliderFloat("Cloud guide threshold", &c.guideThreshold, 0.15f, 0.9f, "%.2f");
        ImGui::TextWrapped("RR reconstructs the fresh cloud samples. Cloud depth is an extinction-weighted distance; normals follow the density gradient. PT misses trace clouds from their bounce positions; reused diffuse lighting uses a coarser cache.");
    }

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
        int v;

        v = (int)s.atmosViewSteps;
        if (ImGui::SliderInt("View Steps", &v, 4, 32))
            s.atmosViewSteps = (float)v;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Per ray atmosphere sample count. Dominant cost of\n"
                              "the atmosphere march. 12 = Bruneton\n"
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
            s.atmosEarthShadowSoftness    = 0.005f;
        }
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
