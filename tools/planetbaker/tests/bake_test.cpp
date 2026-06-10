#include <doctest/doctest.h>

#include "core/cubed_sphere.h"
#include "core/field_set.h"
#include "core/param_registry.h"
#include "core/pass.h"
#include "core/pipeline.h"
#include "core/planet_state.h"
#include "passes/bake_pass.h"
#include "passes/bedrock_noise.h"
#include "passes/impacts_pass.h"
#include "passes/thermal_erosion_pass.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

using namespace pb;

namespace {

std::filesystem::path scratch_dir(const char* tag) {
    auto p = std::filesystem::temp_directory_path() / (std::string("pb_bake_test_") + tag);
    std::filesystem::remove_all(p);
    std::filesystem::create_directories(p);
    return p;
}

void fill(Field<float>& f, float v) {
    std::vector<float> host(static_cast<std::size_t>(f.total_cells()), v);
    f.upload(host);
}

}

TEST_CASE("BakePass disabled by default → no files written") {
    const auto out = scratch_dir("disabled") / "out";

    PlanetState   state(16, PlanetFieldSet::BedrockOnly);
    ParamRegistry reg;
    Pipeline      pipeline(scratch_dir("disabled_cache"));
    pipeline.add_pass(std::make_unique<BakePass>(out));
    pipeline.declare_all(reg);
    pipeline.wire_dirty_tracking(reg);

    //Default bake.enabled is false; the run must be a no-op.
    REQUIRE_FALSE(reg.get_bool("bake.enabled"));

    NullProgressSink null;
    pipeline.run(state, reg, null);

    //No elevation files should appear.
    for (int f = 0; f < 6; ++f) {
        std::string fname = "elevation_face" + std::to_string(f) + ".r32";
        CHECK_FALSE(std::filesystem::exists(out / fname));
    }
    CHECK_FALSE(std::filesystem::exists(out / "manifest.json"));
}

TEST_CASE("BakePass writes 6 cube faces + manifest with correct sizes") {
    const int kSrcN = 16;
    const int kDstN = 128;
    const auto out  = scratch_dir("write") / "out";

    PlanetState   state(kSrcN, PlanetFieldSet::BedrockOnly);
    ParamRegistry reg;
    Pipeline      pipeline(scratch_dir("write_cache"));
    //BakePass's full-pipeline path constructs BedrockNoise / Impacts /
    //Thermal locally and runs them at the bake resolution. Their params
    //(bedrock.*, impacts.*, thermal.*) must already be declared in the
    //registry, so we add the passes here too - their declare_params is
    //the source of truth.
    pipeline.add_pass(std::make_unique<BedrockNoisePass>());
    pipeline.add_pass(std::make_unique<ImpactsPass>());
    pipeline.add_pass(std::make_unique<ThermalErosionPass>());
    pipeline.add_pass(std::make_unique<BakePass>(out));
    pipeline.declare_all(reg);
    pipeline.wire_dirty_tracking(reg);

    reg.set_bool("bake.enabled", true);
    reg.set_int ("bake.elevation_resolution", kDstN);

    NullProgressSink null;
    pipeline.run(state, reg, null);

    REQUIRE(pipeline.entries().size() == 4);
    CHECK(pipeline.entries()[3].status == PassStatus::Clean);   // bake clean

    //Each face file must exist and be exactly N*N*sizeof(float) bytes.
    const std::uintmax_t expected_bytes =
        static_cast<std::uintmax_t>(kDstN) * static_cast<std::uintmax_t>(kDstN) * sizeof(float);

    for (int f = 0; f < 6; ++f) {
        std::string fname = "elevation_face" + std::to_string(f) + ".r32";
        auto path = out / fname;
        REQUIRE_MESSAGE(std::filesystem::exists(path), fname);
        CHECK(std::filesystem::file_size(path) == expected_bytes);
    }

    //Manifest must be valid JSON with the expected layer entry.
    auto manifest_path = out / "manifest.json";
    REQUIRE(std::filesystem::exists(manifest_path));
    std::ifstream is(manifest_path);
    nlohmann::json m;
    is >> m;
    CHECK(m["projection"] == "cubed_sphere_equiangular");
    REQUIRE(m["layers"].contains("elevation"));
    CHECK(m["layers"]["elevation"]["resolution"]  == kDstN);
    CHECK(m["layers"]["elevation"]["channels"]    == 1);
    CHECK(m["layers"]["elevation"]["element_type"] == "float32");
}

TEST_CASE("BakePass synthesises non-degenerate noise at bake resolution") {
    const int kSrcN = 16;
    const int kDstN = 64;
    const auto out  = scratch_dir("values") / "out";

    PlanetState   state(kSrcN, PlanetFieldSet::BedrockOnly);
    ParamRegistry reg;
    Pipeline      pipeline(scratch_dir("values_cache"));
    //All four passes registered so their declare_params populate the
    //registry (BakePass's full-pipeline path needs impacts.* / thermal.*
    //in addition to bedrock.*).
    pipeline.add_pass(std::make_unique<BedrockNoisePass>());
    pipeline.add_pass(std::make_unique<ImpactsPass>());
    pipeline.add_pass(std::make_unique<ThermalErosionPass>());
    pipeline.add_pass(std::make_unique<BakePass>(out));
    pipeline.declare_all(reg);
    pipeline.wire_dirty_tracking(reg);

    reg.set_bool("bake.enabled", true);
    reg.set_int ("bake.elevation_resolution", kDstN);

    NullProgressSink null;
    pipeline.run(state, reg, null);

    //Read back face 0 and verify the synthesised values: finite, in a
    //reasonable km range, AND varying (per-pixel noise should produce
    //distinct values across the face).
    auto path = out / "elevation_face0.r32";
    REQUIRE(std::filesystem::exists(path));
    std::ifstream is(path, std::ios::binary);
    std::vector<float> data(static_cast<std::size_t>(kDstN) * static_cast<std::size_t>(kDstN));
    is.read(reinterpret_cast<char*>(data.data()),
            static_cast<std::streamsize>(data.size() * sizeof(float)));
    REQUIRE(static_cast<std::streamsize>(is.gcount()) ==
            static_cast<std::streamsize>(data.size() * sizeof(float)));

    float vmin = +1e30f, vmax = -1e30f;
    double  sum = 0.0;
    for (float v : data) {
        CHECK(std::isfinite(v));
        CHECK(v >= -20.0f);
        CHECK(v <= +20.0f);
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
        sum += v;
    }
    INFO("face0 min=" << vmin << ", max=" << vmax << ", mean=" << (sum / data.size()));
    //A constant face would have vmax - vmin == 0. Bedrock noise across a
    //whole cube face should span at least a few hundred metres.
    CHECK(vmax - vmin > 0.2f);   // km
}
