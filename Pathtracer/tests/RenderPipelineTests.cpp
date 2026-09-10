#include "Raytracing/PassSystem.h"

static void Require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

int main() {
    try {
        using namespace pass_feature;
        PassSystem passes;
        passes.Build({L"Pass_camera_v8.hlsl|rg", L"barrier", L"Pass_pt_v8.hlsl|rg",
            L"Pass_pt_nee_v8.hlsl|cs:16x16", L"Pass_sharc_update_v8.hlsl|rg",
            L"Pass_raygen_v8.hlsl|rg", L"Pass_spmis_reset_v8.hlsl|cs:16x16",
            L"Pass_shift_v8.hlsl|rg:temporal_shift", L"Pass_lite_shift_v8.hlsl|cs:16x16",
            L"Pass_lite_merge_v8.hlsl|cs:16x16", L"Pass_cumulus_secondary_v8.hlsl|cs:8x8",
            L"Pass_cumulus_noise_v8.hlsl|fx:32768", L"Pass_cumulus_density_v8.hlsl|fx:2048",
            L"Pass_cumulus_ambient_v8.hlsl|fx:32", L"Pass_cumulus_environment_v8.hlsl|fx:2048",
            L"Pass_atmosphere_primary_v8.hlsl|cs:8x8", L"dlss", L"Pass_light_learning_v8.hlsl|fx:256"});
        const auto& p = passes.Passes();
        const uint32_t standard = PathTracer | Sharc | DiffuseReuse | SpatialReuse | MeshLights;
        Require(p[0].IsEnabled(standard) && p[2].IsEnabled(standard) && p[3].IsEnabled(standard) && p[4].IsEnabled(standard),
            "Standard PT/SHARC passes missing");
        for (int i : {5, 6, 7}) Require(!p[i].IsEnabled(standard) && p[i].IsEnabled(LegacyReSTIR), "Legacy pass filtering failed");
        Require(p[7].dispatchTag == L"temporal_shift", "Shared shift shader lost its role");
        Require(!p[3].IsEnabled(standard & ~MeshLights), "Empty scene still prefetches mesh lights");
        Require(!p[4].IsEnabled(PathTracer), "Uncached reference trains SHARC");
        Require(!p[8].IsEnabled(standard & ~SpatialReuse) && p[9].IsEnabled(standard & ~SpatialReuse), "Diffuse merge depends on spatial reuse");
        Require(!p[9].IsEnabled(PathTracer), "Disabled diffuse reuse still runs");
        Require(!p[10].IsEnabled(LegacyReSTIR | Clouds) && p[10].IsEnabled(PathTracer | Clouds), "Secondary cloud integration mismatch");
        Require(!p[11].IsEnabled(Clouds) && p[11].IsEnabled(Clouds | CloudNoise), "Noise bake is not conditional");
        Require(!p[12].IsEnabled(Clouds) && p[12].IsEnabled(Clouds | CloudDensity), "Density bake is not conditional");
        Require(!p[13].IsEnabled(Clouds) && p[13].IsEnabled(Clouds | CloudAmbient), "Ambient bake is not conditional");
        Require(p[14].IsEnabled(Clouds) && !p[14].IsEnabled(PathTracer), "Cloud environment filtering failed");
        Require(p[15].IsEnabled(PathTracer) && p[15].IsEnabled(LegacyReSTIR), "Clear atmosphere was disabled with clouds");
        Require(p[16].IsEnabled(PathTracer), "Reconstruction stage lost");
        Require(p[17].IsEnabled(PathTracer|MeshLights|LightLearning),"Learned-light pass missing");
        Require(!p[17].IsEnabled(PathTracer|MeshLights) && !p[17].IsEnabled(PathTracer|LightLearning) &&
            !p[17].IsEnabled(LegacyReSTIR|MeshLights|LightLearning),"Learning runs while disabled, empty or in replay");
        for (const auto& pass : p) Require(!pass.executedLastFrame, "Unexecuted passes shown as active");

        IntegratorSettings base;
        Require(base.integratorMode == 0 && base.sharcEnabled, "Default is not PT with SHARC");
        Require(base.liteEnabled && base.liteSpatial, "Default diffuse spatial reuse is disabled");
        auto changed = base;
        changed.sharcDebugMode = 2;
        changed.sharcDebugCoarse = true;
        Require(changed.ReconstructionKey() == base.ReconstructionKey(), "Inspection invalidates underlying reconstruction");
        for (int control = 0; control < 10; ++control) {
            changed = base;
            if (control == 0) changed.integratorMode = 1;
            if (control == 1) changed.sharcEnabled = false;
            if (control == 2) changed.forceDiffuseMats = true;
            if (control == 3) changed.texturePointFilter = 1;
            if (control == 4) changed.maxDiffuseBounces += 1;
            if (control == 5) changed.liteEnabled = false;
            if (control == 6) changed.liteUnshadowedTargets = true;
            if (control == 7) changed.lightTreeLearning = false;
            if (control == 8) changed.lightTreeSG = false;
            if (control == 9) changed.lightTreeCellExponent += 1;
            Require(changed.ReconstructionKey() != base.ReconstructionKey(), "Transport edit retains incompatible reconstruction");
        }
        CumulusSettings cloud, inspected = cloud;
        inspected.debugView = 3; inspected.guideThreshold = 0.8f;
        Require(inspected.LightingKey() == cloud.LightingKey(), "Cloud inspection discards cached lighting");
        inspected.coverage += 0.1f;
        Require(inspected.LightingKey() != cloud.LightingKey(), "Cloud shape change retains cached lighting");
        std::cout << "PASS: render pass selection, default PT/SHARC, empty lights and history invalidation.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
