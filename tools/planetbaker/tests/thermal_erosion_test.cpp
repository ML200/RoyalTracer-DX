#include <doctest/doctest.h>

#include "core/cubed_sphere.h"
#include "core/field_set.h"
#include "core/param_registry.h"
#include "core/pass.h"
#include "core/pipeline.h"
#include "core/planet_state.h"
#include "passes/thermal_erosion_pass.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <memory>
#include <vector>

using namespace pb;

namespace {

std::filesystem::path scratch_dir(const char* tag) {
    auto p = std::filesystem::temp_directory_path() / (std::string("pb_thermal_test_") + tag);
    std::filesystem::remove_all(p);
    return p;
}

float total_interior_sediment(const PlanetState& state) {
    std::vector<float> host;
    state.sediment_thickness.download(host);
    const auto& g = state.grid();
    float total = 0.0f;
    for (int face = 0; face < 6; ++face) {
        for (int j = 0; j < g.n; ++j) {
            for (int i = 0; i < g.n; ++i) {
                total += host[g.global_index(face, i, j)];
            }
        }
    }
    return total;
}

}

TEST_CASE("ThermalErosionPass smoke run leaves no NaNs and no negatives") {
    PlanetState   state(16);
    ParamRegistry reg;
    Pipeline      pipeline(scratch_dir("smoke"));
    pipeline.add_pass(std::make_unique<ThermalErosionPass>());
    pipeline.declare_all(reg);
    pipeline.wire_dirty_tracking(reg);

    //Seed some sediment so there's something to relax.
    std::vector<float> host(static_cast<std::size_t>(state.sediment_thickness.total_cells()), 0.5f);
    state.sediment_thickness.upload(host);

    reg.set_int("thermal.iterations", 50);

    NullProgressSink null;
    pipeline.run(state, reg, null);

    std::vector<float> back;
    state.sediment_thickness.download(back);
    for (float s : back) {
        CHECK(std::isfinite(s));
        CHECK(s >= 0.0f);
    }
}

TEST_CASE("ThermalErosionPass smoothes a single tall sediment spike") {
    const int N = 16;
    PlanetState   state(N);
    ParamRegistry reg;
    Pipeline      pipeline(scratch_dir("spike"));
    pipeline.add_pass(std::make_unique<ThermalErosionPass>());
    pipeline.declare_all(reg);
    pipeline.wire_dirty_tracking(reg);

    //Sediment spike: 100 m at one cell, 0 elsewhere. Way above any repose
    //angle, so erosion should redistribute it to neighbours.
    std::vector<float> host(static_cast<std::size_t>(state.sediment_thickness.total_cells()), 0.0f);
    const auto& g = state.grid();
    int peak_idx = g.global_index(0, N / 2, N / 2);
    host[peak_idx] = 100.0f;
    state.sediment_thickness.upload(host);
    float total_before = 100.0f;

    reg.set_int  ("thermal.iterations",       100);
    reg.set_float("thermal.transfer_rate",    0.3f);

    NullProgressSink null;
    pipeline.run(state, reg, null);

    std::vector<float> back;
    state.sediment_thickness.download(back);
    float peak_after = back[peak_idx];
    INFO("spike after = " << peak_after << " m");
    CHECK(peak_after < 100.0f);   //should have dropped from initial spike

    //Some sediment must have moved to neighbours.
    bool any_neighbour = false;
    for (int dj = -2; dj <= 2; ++dj) {
        for (int di = -2; di <= 2; ++di) {
            if (di == 0 && dj == 0) continue;
            int ni = N / 2 + di;
            int nj = N / 2 + dj;
            if (ni < 0 || ni >= N || nj < 0 || nj >= N) continue;
            float v = back[g.global_index(0, ni, nj)];
            if (v > 0.1f) { any_neighbour = true; break; }
        }
        if (any_neighbour) break;
    }
    CHECK(any_neighbour);

    //Sediment shouldn't grow without bound; total should be in the same ballpark
    //as before. With approximate inflow caps we tolerate ~30% slop.
    float total_after = total_interior_sediment(state);
    INFO("total before = " << total_before << ", after = " << total_after);
    CHECK(total_after < total_before * 2.0f);
    CHECK(total_after > total_before * 0.3f);
}

TEST_CASE("ThermalErosionPass: zero iterations leaves sediment untouched") {
    PlanetState   state(8);
    ParamRegistry reg;
    Pipeline      pipeline(scratch_dir("noop"));
    pipeline.add_pass(std::make_unique<ThermalErosionPass>());
    pipeline.declare_all(reg);
    pipeline.wire_dirty_tracking(reg);

    std::vector<float> host(static_cast<std::size_t>(state.sediment_thickness.total_cells()));
    for (std::size_t i = 0; i < host.size(); ++i) host[i] = static_cast<float>(i % 7);
    state.sediment_thickness.upload(host);

    reg.set_int("thermal.iterations", 0);

    NullProgressSink null;
    pipeline.run(state, reg, null);

    std::vector<float> back;
    state.sediment_thickness.download(back);
    REQUIRE(back.size() == host.size());
    for (std::size_t i = 0; i < host.size(); ++i) {
        CHECK(back[i] == host[i]);
    }
}
