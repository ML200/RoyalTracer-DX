#include "Raytracing/PassSystem.h"

static void Require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

int main() {
    try {
        using namespace pass_feature;
        PassSystem passes;
        passes.Build({L"Pass_camera_v8.hlsl|rg", L"barrier",
            L"Pass_pt_skybake_v8.hlsl|fx:512", L"barrier",
            L"Pass_light_learning_v8.hlsl|fx:256", L"barrier",
            L"Pass_sharc_prepare_v8.hlsl|fx:4096", L"barrier",
            L"Pass_sharc_update_v8.hlsl|rg", L"barrier",
            L"Pass_sharc_resolve_v8.hlsl|fx:4096", L"barrier",
            L"Pass_sharc_debug_v8.hlsl|cs:16x16", L"barrier",
            L"loop:pt_samples",
            L"Pass_pt_trace_v8.hlsl|rg", L"barrier",
            L"Pass_pt_light_v8.hlsl|cs:16x16", L"barrier",
            L"Pass_pt_shade_v8.hlsl|cs:16x16", L"barrier",
            L"endloop",
            L"Pass_lite_shift_v8.hlsl|cs:16x16", L"barrier",
            L"Pass_lite_merge_v8.hlsl|cs:16x16", L"barrier",
            L"Pass_atmosphere_primary_v8.hlsl|cs:8x8", L"barrier",
            L"Pass_shading_v8.hlsl|cs:16x16", L"barrier",
            L"dlss", L"barrier",
            L"Pass_postprocess_v8.hlsl|cs:8x4"});
        const auto& p = passes.Passes();
        const uint32_t standard = Sharc | DiffuseReuse | SpatialReuse | MeshLights;
        for (int i : {0, 2, 15, 17, 19, 26, 28, 32})
            Require(p[i].IsEnabled(0), "Unconditional pass depends on a feature");
        Require(p[15].stage == Stage::RayGen && p[17].stage == Stage::Compute && p[17].groupX == 16 && p[17].groupY == 16,
            "Path tracing stages parsed wrong");
        Require(p[2].stage == Stage::FixedCompute && p[2].groupX == 512 && p[2].groupY == 1, "Fixed dispatch parsed wrong");
        Require(p[8].IsEnabled(standard) && !p[8].IsEnabled(standard & ~Sharc), "Cache training must follow the cache");
        Require(p[6].IsEnabled(Sharc) && p[10].IsEnabled(Sharc) && !p[6].IsEnabled(0), "Cache maintenance gating wrong");
        Require(p[12].IsEnabled(Sharc | SharcDebug) && !p[12].IsEnabled(Sharc), "Cache inspection runs without inspection");
        Require(p[4].IsEnabled(MeshLights | LightLearning) && !p[4].IsEnabled(MeshLights) && !p[4].IsEnabled(LightLearning),
            "Learning runs while disabled or without lights");
        Require(!p[22].IsEnabled(standard & ~SpatialReuse) && p[24].IsEnabled(standard & ~SpatialReuse),
            "Diffuse merge depends on spatial reuse");
        Require(!p[24].IsEnabled(0), "Disabled diffuse reuse still runs");
        Require(p[14].stage == Stage::LoopStart && p[14].loopTag == L"pt_samples" && p[14].loopCount == 0,
            "Sample loop count not deferred to the integrator settings");
        Require(p[21].stage == Stage::LoopEnd && p[21].targetIdx == 14, "Loop end not linked to its tagged start");
        Require(p[30].stage == Stage::DLSS, "Reconstruction stage lost");
        for (const auto& pass : p) Require(!pass.executedLastFrame, "Unexecuted passes shown as active");

        IntegratorSettings base;
        Require(!base.compactLightTree, "Light tree compaction must default off");
        Require(base.sharcEnabled, "Default is not PT with SHARC");
        Require(base.liteEnabled && base.liteSpatial, "Default diffuse spatial reuse is disabled");
        auto changed = base;
        changed.sharcDebugMode = 2;
        changed.sharcDebugCoarse = true;
        changed.lightTreeDebug = true;
        Require(changed.ReconstructionKey() == base.ReconstructionKey(), "Inspection invalidates underlying reconstruction");
        for (int control = 0; control < 10; ++control) {
            changed = base;
            if (control == 0) changed.sharcEnabled = false;
            if (control == 1) changed.forceDiffuseMats = true;
            if (control == 2) changed.texturePointFilter = 1;
            if (control == 3) changed.maxDiffuseBounces += 1;
            if (control == 4) changed.liteEnabled = false;
            if (control == 5) changed.liteUnshadowedTargets = true;
            if (control == 6) changed.lightTreeLearning = false;
            if (control == 7) changed.lightTreeCellExponent += 1;
            if (control == 8) changed.lightTreeLodScale *= 2;
            if (control == 9) changed.compactLightTree = true;
            Require(changed.ReconstructionKey() != base.ReconstructionKey(), "Transport edit retains incompatible reconstruction");
        }
        std::cout << "PASS: render pass selection, default PT/SHARC and history invalidation.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
