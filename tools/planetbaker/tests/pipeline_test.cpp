#include <doctest/doctest.h>

#include "core/field_set.h"
#include "core/param_registry.h"
#include "core/pass.h"
#include "core/pipeline.h"
#include "core/planet_state.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

using namespace pb;

namespace {

//A trivial pass that fills bedrock_elevation interior with a constant on
//the host side. Avoids depending on the NoisePass kernel here so the test
//is pure CPU + cache I/O.
class FillConstantPass : public Pass {
public:
    const char*   name()    const override { return "fillconst"; }
    std::uint64_t version() const override { return 1; }

    FieldSet reads()  const override { return {}; }
    FieldSet writes() const override { return FieldSet{}.set(FieldId::BedrockElevation); }

    void declare_params(ParamRegistry& reg) const override {
        reg.declare_float("fillconst.value", 1.0f, -10.0f, 10.0f,
                          "value", "Constant elevation to write", "km", "fillconst");
    }

    void run(PlanetState& state, const ParamRegistry& reg, ProgressSink& p) override {
        float v = reg.get_float("fillconst.value");
        auto& field = state.bedrock_elevation;
        std::vector<float> host(static_cast<std::size_t>(field.total_cells()), v);
        field.upload(host);
        p.fraction(1.0f);
    }
};

//Reads bedrock_elevation, writes crust_thickness = 2 * bedrock. Lets the
//pipeline test detect upstream-output-changes-propagate cascading via the
//reads/writes intersection.
class ReadBedrockPass : public Pass {
public:
    const char*   name()    const override { return "readbed"; }
    std::uint64_t version() const override { return 1; }
    FieldSet reads()  const override { return FieldSet{}.set(FieldId::BedrockElevation); }
    FieldSet writes() const override { return FieldSet{}.set(FieldId::CrustThickness); }
    void declare_params(ParamRegistry&) const override {}
    void run(PlanetState& state, const ParamRegistry&, ProgressSink& p) override {
        std::vector<float> bed;
        state.bedrock_elevation.download(bed);
        std::vector<float> thk(bed.size());
        for (std::size_t i = 0; i < bed.size(); ++i) thk[i] = 2.0f * bed[i];
        state.crust_thickness.upload(thk);
        p.fraction(1.0f);
    }
};

std::filesystem::path scratch_dir(const char* tag) {
    auto p = std::filesystem::temp_directory_path() / (std::string("pb_pipeline_test_") + tag);
    std::filesystem::remove_all(p);
    return p;
}

float sample_interior(const Field<float>& f, int face, int i, int j) {
    std::vector<float> host;
    f.download(host);
    return host[static_cast<std::size_t>(f.grid().global_index(face, i, j))];
}

}

TEST_CASE("Pipeline: first run misses cache, second hits") {
    auto cache_root = scratch_dir("hit_miss");
    NullProgressSink null;

    //First run: cold cache.
    {
        PlanetState   state(8);
        ParamRegistry reg;
        Pipeline      pipeline(cache_root);
        pipeline.add_pass(std::make_unique<FillConstantPass>());
        pipeline.declare_all(reg);
        pipeline.wire_dirty_tracking(reg);

        reg.set_float("fillconst.value", 2.5f);
        pipeline.run(state, reg, null);

        auto entries = pipeline.entries();
        REQUIRE(entries.size() == 1);
        CHECK(entries[0].status      == PassStatus::Clean);
        CHECK(entries[0].last_source == LastSource::Computed);

        CHECK(sample_interior(state.bedrock_elevation, 0, 0, 0) == doctest::Approx(2.5f));
    }

    //Second run with a fresh state: should pull from cache, not run kernel.
    {
        PlanetState   state(8);
        ParamRegistry reg;
        Pipeline      pipeline(cache_root);
        pipeline.add_pass(std::make_unique<FillConstantPass>());
        pipeline.declare_all(reg);
        pipeline.wire_dirty_tracking(reg);

        reg.set_float("fillconst.value", 2.5f);
        pipeline.run(state, reg, null);

        auto entries = pipeline.entries();
        REQUIRE(entries.size() == 1);
        CHECK(entries[0].status      == PassStatus::Clean);
        CHECK(entries[0].last_source == LastSource::Cache);

        CHECK(sample_interior(state.bedrock_elevation, 0, 0, 0) == doctest::Approx(2.5f));
    }
}

TEST_CASE("Pipeline: param change flips pass to Dirty") {
    auto cache_root = scratch_dir("dirty");
    NullProgressSink null;

    PlanetState   state(8);
    ParamRegistry reg;
    Pipeline      pipeline(cache_root);
    pipeline.add_pass(std::make_unique<FillConstantPass>());
    pipeline.declare_all(reg);
    pipeline.wire_dirty_tracking(reg);

    pipeline.run(state, reg, null);
    CHECK(pipeline.entries()[0].status == PassStatus::Clean);

    reg.set_float("fillconst.value", 7.0f);
    CHECK(pipeline.entries()[0].status == PassStatus::Dirty);

    pipeline.run(state, reg, null);
    CHECK(pipeline.entries()[0].status      == PassStatus::Clean);
    CHECK(pipeline.entries()[0].last_source == LastSource::Computed);
    CHECK(sample_interior(state.bedrock_elevation, 0, 0, 0) == doctest::Approx(7.0f));
}

TEST_CASE("Pipeline: different param values produce different cache keys") {
    auto cache_root = scratch_dir("keys");
    NullProgressSink null;

    auto run_with_value = [&](float v) -> std::uint64_t {
        PlanetState   state(8);
        ParamRegistry reg;
        Pipeline      pipeline(cache_root);
        pipeline.add_pass(std::make_unique<FillConstantPass>());
        pipeline.declare_all(reg);
        pipeline.wire_dirty_tracking(reg);
        reg.set_float("fillconst.value", v);
        pipeline.run(state, reg, null);
        return pipeline.entries()[0].last_key;
    };

    auto k1 = run_with_value(1.0f);
    auto k2 = run_with_value(2.0f);
    CHECK(k1 != k2);

    //Same value should reproduce the same key and load from cache.
    auto k3 = run_with_value(1.0f);
    CHECK(k1 == k3);
}

TEST_CASE("Pipeline cascade: changing upstream params flips downstream to Dirty") {
    auto cache_root = scratch_dir("cascade");
    NullProgressSink null;

    PlanetState   state(8);
    ParamRegistry reg;
    Pipeline      pipeline(cache_root);
    pipeline.add_pass(std::make_unique<FillConstantPass>());
    pipeline.add_pass(std::make_unique<ReadBedrockPass>());
    pipeline.declare_all(reg);
    pipeline.wire_dirty_tracking(reg);

    reg.set_float("fillconst.value", 1.0f);
    pipeline.run(state, reg, null);
    CHECK(pipeline.entries()[0].status == PassStatus::Clean);
    CHECK(pipeline.entries()[1].status == PassStatus::Clean);
    CHECK(sample_interior(state.crust_thickness, 0, 0, 0) == doctest::Approx(2.0f));

    //Mutating the upstream param should cascade.
    reg.set_float("fillconst.value", 5.0f);
    CHECK(pipeline.entries()[0].status == PassStatus::Dirty);
    CHECK(pipeline.entries()[1].status == PassStatus::Dirty);

    pipeline.run(state, reg, null);
    CHECK(pipeline.entries()[0].status == PassStatus::Clean);
    CHECK(pipeline.entries()[1].status == PassStatus::Clean);
    CHECK(sample_interior(state.crust_thickness, 0, 0, 0) == doctest::Approx(10.0f));
}

TEST_CASE("Pipeline cascade: field content hash bakes into downstream cache key") {
    auto cache_root = scratch_dir("contenthash");
    NullProgressSink null;

    auto run_once = [&](float upstream_value) -> std::uint64_t {
        PlanetState   state(8);
        ParamRegistry reg;
        Pipeline      pipeline(cache_root);
        pipeline.add_pass(std::make_unique<FillConstantPass>());
        pipeline.add_pass(std::make_unique<ReadBedrockPass>());
        pipeline.declare_all(reg);
        pipeline.wire_dirty_tracking(reg);
        reg.set_float("fillconst.value", upstream_value);
        pipeline.run(state, reg, null);
        return pipeline.entries()[1].last_key;
    };

    auto k1 = run_once(1.0f);
    auto k2 = run_once(2.0f);
    CHECK(k1 != k2);   //downstream key reflects upstream's output content
}
