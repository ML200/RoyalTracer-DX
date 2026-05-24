#include <doctest/doctest.h>

#include "core/field_set.h"
#include "core/param_registry.h"
#include "core/pass.h"
#include "core/pipeline.h"
#include "core/planet_state.h"
#include "passes/impacts_pass.h"

#include <cmath>
#include <filesystem>
#include <memory>
#include <vector>

using namespace pb;

namespace {

std::filesystem::path scratch_dir(const char* tag) {
    auto p = std::filesystem::temp_directory_path() / (std::string("pb_impacts_test_") + tag);
    std::filesystem::remove_all(p);
    return p;
}

}

TEST_CASE("ImpactsPass runs on a blank planet without crashing") {
    PlanetState   state(16);
    ParamRegistry reg;
    Pipeline      pipeline(scratch_dir("smoke"));
    pipeline.add_pass(std::make_unique<ImpactsPass>());
    pipeline.declare_all(reg);
    pipeline.wire_dirty_tracking(reg);

    //Keep it cheap: low flux so we get ~50 craters on the test mesh.
    reg.set_float("impacts.flux_multiplier", 0.05f);

    NullProgressSink null;
    pipeline.run(state, reg, null);

    REQUIRE(pipeline.entries().size() == 1);
    CHECK(pipeline.entries()[0].status      == PassStatus::Clean);
    CHECK(pipeline.entries()[0].last_source == LastSource::Computed);

    //Bedrock should now contain some negative cells (crater cavities).
    std::vector<float> bed;
    state.bedrock_elevation.download(bed);
    float min_b = bed[0];
    for (float v : bed) if (v < min_b) min_b = v;
    CHECK(min_b < -0.1f);

    //At least some sediment from ejecta.
    std::vector<float> sed;
    state.sediment_thickness.download(sed);
    float max_s = sed[0];
    for (float v : sed) if (v > max_s) max_s = v;
    CHECK(max_s > 0.0f);
}

TEST_CASE("ImpactsPass with flux=0 leaves state unchanged") {
    PlanetState   state(16);
    ParamRegistry reg;
    Pipeline      pipeline(scratch_dir("zero_flux"));
    pipeline.add_pass(std::make_unique<ImpactsPass>());
    pipeline.declare_all(reg);
    pipeline.wire_dirty_tracking(reg);

    reg.set_float("impacts.flux_multiplier", 0.0f);

    NullProgressSink null;
    pipeline.run(state, reg, null);

    std::vector<float> bed;
    state.bedrock_elevation.download(bed);
    for (float v : bed) CHECK(v == 0.0f);
}

TEST_CASE("ImpactsPass: different seeds produce different crater fields") {
    NullProgressSink null;
    auto root = scratch_dir("seeds");

    auto run_seed = [&](int seed) {
        PlanetState   state(16);
        ParamRegistry reg;
        Pipeline      pipeline(root);
        pipeline.add_pass(std::make_unique<ImpactsPass>());
        pipeline.declare_all(reg);
        pipeline.wire_dirty_tracking(reg);
        reg.set_float("impacts.flux_multiplier", 0.05f);
        reg.set_int  ("impacts.seed", seed);
        pipeline.run(state, reg, null);
        std::vector<float> bed;
        state.bedrock_elevation.download(bed);
        return bed;
    };

    auto a = run_seed(101);
    auto b = run_seed(202);

    float diff = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) diff += std::fabs(a[i] - b[i]);
    CHECK(diff > 1.0f);
}
