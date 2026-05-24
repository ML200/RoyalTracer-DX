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
    std::uint64_t version() const override { return 1; }

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
