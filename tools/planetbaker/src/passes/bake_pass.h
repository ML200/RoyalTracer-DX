#pragma once

#include <filesystem>
#include <utility>

#include "core/field_set.h"
#include "core/pass.h"

namespace pb {

//====================================
//BakePass: writes the final surface heightmap as a high-resolution cubemap
//to disk. PLAN sections 7.11 + 8.5, minimal flavour: six raw float32 files
//per face plus a manifest.json describing the layout. Bicubic upsample
//(Catmull-Rom) from the working CubedSphereGrid resolution to the bake
//resolution. Faces are halo-exchanged first so the 4x4 sample window can
//cross face boundaries without seams. Output value = bedrock_elevation +
//sediment_thickness * 0.001 in km.
//
//Gated by `bake.enabled` (default false). When disabled the run is a
//no-op; the user flips the flag, hits Run, and the pipeline writes the
//cubemap to ./out/. The flag is opt-in because a 16k bake produces about
//6 GB of disk traffic and takes ~30 s on a modern desktop GPU.
//
//Reads:  bedrock_elevation, sediment_thickness.
//Writes: nothing in PlanetState. Output is side-effected onto the file
//        system; the pipeline cache layer treats the empty FieldSet
//        correctly (no cache entries, but Clean status after a successful
//        run so the pass doesn't re-fire on every pipeline iteration).
//
//Params declared under "bake.*".
//====================================

class BakePass : public Pass {
public:
    //Default constructor writes to ./out/ relative to the process cwd.
    //Tests pass a scratch directory so they don't collide with the live bake.
    BakePass() = default;
    explicit BakePass(std::filesystem::path output_dir)
        : output_dir_(std::move(output_dir)) {}

    const char*   name()    const override { return "bake"; }
    //v3: full-pipeline bake. For bake.elevation_resolution <= 8192 the bake
    //allocates a fresh PlanetState at the bake resolution, runs BedrockNoise
    //+ Impacts + Thermal on it, then writes the result. Above the cap it
    //falls back to bedrock-only direct synthesis per face (v2 behaviour) so
    //16 k bedrock-only stays available for max-detail bakes.
    //v2: bedrock noise re-synthesised at the output resolution per face
    //(replaced v1's bicubic-upsample of the working grid). v2 still skipped
    //impacts + thermal; v3 brings them back at the cap.
    //v4: 16k bakes (> kFullPipelineMaxN) now also include impact craters.
    //bake_impacts_face stamps the same crater population each face's
    //bedrock-noise-only direct-synth path produced, folding ejecta into
    //the bedrock buffer since there is no separate sediment field in the
    //direct path. Thermal erosion still skipped above the cap.
    //v5: full-pipeline bake (dst_n <= kFullPipelineMaxN) now runs
    //HydraulicErosionPass after thermal. With hydraulic.iterations = 0
    //(the default) it's a no-op so existing bakes are unchanged byte-for-
    //byte; turn iterations up to bake in carved canyons. Above the cap,
    //hydraulic is skipped along with thermal because the bedrock-only
    //direct path doesn't carry sediment / water columns.
    //v6: full-pipeline bake also runs SurfaceColorPass and writes 6
    //surface_color_face<N>.png alongside the existing elevation_face<N>.r32.
    //Manifest bumped to version 2 with a "surface_color" layer entry. The
    //>8192 direct-synthesis path still writes elevation only (no color)
    //since it doesn't carry sediment/crust_type/water.
    //v7: writes a 6-face tangent-space normal map (normal_face<N>.png) at
    //the same resolution by central-differencing bedrock+sediment through
    //the cross-face halo. Manifest gains a "normal" layer entry alongside
    //"elevation" and "surface_color". Still full-pipeline path only.
    //v8: writes a 6-face block-averaged cloud-offset map
    //(cloud_offset_face<N>.r32) at a fixed 256x256 per face. Renderer
    //samples it bilinearly to set the cloud-base altitude, so peaks above
    //the smoothed elevation poke through clouds naturally. Adds a
    //"cloud_offset" layer entry to the manifest.
    std::uint64_t version() const override { return 8; }

    FieldSet reads()  const override {
        return FieldSet{}
            .set(FieldId::BedrockElevation)
            .set(FieldId::SedimentThickness);
    }
    FieldSet writes() const override { return FieldSet{}; }

    void declare_params(ParamRegistry& reg) const override;
    void run(PlanetState& state, const ParamRegistry& reg, ProgressSink& progress) override;

private:
    std::filesystem::path output_dir_ = "./out";
};

}
