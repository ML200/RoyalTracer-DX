#include <doctest/doctest.h>

#include "core/cubed_sphere.h"
#include "core/field.h"
#include "core/neighbor_table.h"

#include <cmath>
#include <vector>

using namespace pb;

namespace {

float cell_value(const CubedSphereGrid& grid, int face, int i, int j) {
    //Use the y-component of the cell-center direction. It is C0 on the
    //sphere, bounded by [-1, +1], and its derivative with respect to angle
    //is at most 1 everywhere, so a 1-cell discretization error translates
    //to at most ~pi/(2N) absolute error. Avoids the asin steepness near
    //the poles that a literal latitude field would suffer from.
    float u = grid.u_of_i(i);
    float v = grid.v_of_j(j);
    Vec3f p = face_uv_to_sphere(face, u, v);
    return p.y;
}

}

TEST_CASE("halo_exchange fills ghost cells with values matching the seam neighbor") {
    const int n    = 32;
    const int halo = CubedSphereGrid::HALO;
    CubedSphereGrid grid(n);
    NeighborTable   table(grid);

    Field<float> field(grid);
    const std::size_t total = static_cast<std::size_t>(grid.total_cells());

    std::vector<float> host(total, 0.0f);
    for (int face = 0; face < 6; ++face) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                host[grid.global_index(face, i, j)] = cell_value(grid, face, i, j);
            }
        }
    }
    field.upload(host);
    field.halo_exchange(table);

    std::vector<float> result;
    field.download(result);

    //Interior cells must not change.
    for (int face = 0; face < 6; ++face) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                int idx = grid.global_index(face, i, j);
                CHECK(result[idx] == host[idx]);
            }
        }
    }

    //Each ghost cell should hold a value close to what its own direction
    //would produce. The worst case is one cell width of angular error
    //(approx pi/(2N)), so 0.1 is comfortably above the floor for N=32.
    const float tolerance = 0.1f;
    int   checked = 0;
    int   failures = 0;
    float max_err  = 0.0f;

    for (int face = 0; face < 6; ++face) {
        for (int j = -halo; j < n + halo; ++j) {
            for (int i = -halo; i < n + halo; ++i) {
                if (i >= 0 && i < n && j >= 0 && j < n) continue;

                float expected = cell_value(grid, face, i, j);
                float actual   = result[grid.global_index(face, i, j)];
                float err      = std::fabs(actual - expected);
                if (err > max_err) max_err = err;
                if (err > tolerance) ++failures;
                ++checked;
            }
        }
    }

    INFO("checked=" << checked << " failures=" << failures << " max_err=" << max_err);
    CHECK(failures == 0);
    CHECK(max_err < tolerance);
}

TEST_CASE("halo_exchange handles a constant field exactly") {
    const int n = 16;
    CubedSphereGrid grid(n);
    NeighborTable   table(grid);

    Field<float> field(grid);
    std::vector<float> host(static_cast<std::size_t>(grid.total_cells()), 0.0f);
    for (int face = 0; face < 6; ++face) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                host[grid.global_index(face, i, j)] = 3.5f;
            }
        }
    }
    field.upload(host);
    field.halo_exchange(table);

    std::vector<float> result;
    field.download(result);

    //Every cell (interior or halo) must be exactly 3.5f. Halo cells gather
    //from interior cells which are all 3.5, so the copy must be bit-exact.
    for (float v : result) {
        CHECK(v == 3.5f);
    }
}
