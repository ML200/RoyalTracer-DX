#include <doctest/doctest.h>

#include "core/cubed_sphere.h"
#include "core/neighbor_table.h"

#include <cmath>
#include <set>
#include <utility>

using namespace pb;

static float dot3(Vec3f a, Vec3f b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

TEST_CASE("face_uv_to_sphere returns unit vectors") {
    const int samples = 8;
    for (int face = 0; face < 6; ++face) {
        for (int s = 0; s <= samples; ++s) {
            for (int t = 0; t <= samples; ++t) {
                float u = -1.0f + 2.0f * static_cast<float>(s) / samples;
                float v = -1.0f + 2.0f * static_cast<float>(t) / samples;
                Vec3f p = face_uv_to_sphere(face, u, v);
                float len2 = dot3(p, p);
                CHECK(len2 == doctest::Approx(1.0f).epsilon(1e-5));
            }
        }
    }
}

TEST_CASE("sphere_to_face_uv inverts face_uv_to_sphere") {
    const int samples = 7;
    for (int face = 0; face < 6; ++face) {
        for (int s = 0; s <= samples; ++s) {
            for (int t = 0; t <= samples; ++t) {
                //Step in slightly from the edges so the back-face assignment is
                //unambiguous; corners are on the seam by construction.
                float u = -0.95f + 1.9f * static_cast<float>(s) / samples;
                float v = -0.95f + 1.9f * static_cast<float>(t) / samples;
                Vec3f p = face_uv_to_sphere(face, u, v);

                int   bf = -1;
                float bu = 0.0f, bv = 0.0f;
                sphere_to_face_uv(p, bf, bu, bv);

                CHECK(bf == face);
                CHECK(bu == doctest::Approx(u).epsilon(1e-4));
                CHECK(bv == doctest::Approx(v).epsilon(1e-4));
            }
        }
    }
}

TEST_CASE("CubedSphereGrid cell coords are consistent") {
    CubedSphereGrid grid(64);
    for (int i = 0; i < grid.n; ++i) {
        float u = grid.u_of_i(i);
        int   back = grid.i_of_u(u);
        CHECK(back == i);
        CHECK(u > -1.0f);
        CHECK(u <  1.0f);
    }
}

TEST_CASE("NeighborTable ghost count matches expectation") {
    const int n = 32;
    const int h = CubedSphereGrid::HALO;
    CubedSphereGrid grid(n);
    NeighborTable   table(grid);

    const std::size_t per_face = static_cast<std::size_t>((n + 2 * h) * (n + 2 * h) - n * n);
    CHECK(table.copy_count() == 6 * per_face);
}

TEST_CASE("Ghost copies route to interior cells on a different face") {
    const int n = 16;
    CubedSphereGrid grid(n);
    NeighborTable   table(grid);

    int  same_face = 0;
    int  out_of_range = 0;

    for (const auto& c : table.host_copies()) {
        if (c.src_face == c.dst_face) ++same_face;
        //src_idx should land inside the interior of src_face, not in halo
        int local = c.src_idx - c.src_face * grid.cells_per_face();
        int j = local / grid.stride() - CubedSphereGrid::HALO;
        int i = local % grid.stride() - CubedSphereGrid::HALO;
        if (i < 0 || i >= n || j < 0 || j >= n) ++out_of_range;
    }

    CHECK(same_face == 0);
    CHECK(out_of_range == 0);
}

TEST_CASE("Each ghost cell has exactly one copy entry") {
    const int n = 8;
    const int h = CubedSphereGrid::HALO;
    CubedSphereGrid grid(n);
    NeighborTable   table(grid);

    std::set<std::pair<int, int>> seen;
    for (const auto& c : table.host_copies()) {
        auto key = std::make_pair(c.dst_face, c.dst_idx);
        CHECK(seen.insert(key).second);
    }

    //Every ghost cell on every face appears.
    for (int face = 0; face < 6; ++face) {
        for (int j = -h; j < n + h; ++j) {
            for (int i = -h; i < n + h; ++i) {
                bool interior = (i >= 0 && i < n && j >= 0 && j < n);
                if (interior) continue;
                auto key = std::make_pair(face, grid.global_index(face, i, j));
                CHECK(seen.count(key) == 1);
            }
        }
    }
}

TEST_CASE("Neighbors4 returns interior cell for interior input") {
    const int n = 16;
    CubedSphereGrid grid(n);
    NeighborTable   table(grid);

    Neighbors4 nb = table.neighbors(0, 5, 7);
    CHECK(nb.plus_i.face == 0);
    CHECK(nb.plus_i.i    == 6);
    CHECK(nb.plus_i.j    == 7);
    CHECK(nb.minus_j.j   == 6);
    CHECK(nb.minus_j.face == 0);
}

TEST_CASE("Neighbors4 crosses faces at edges") {
    const int n = 32;
    CubedSphereGrid grid(n);
    NeighborTable   table(grid);

    //Cell at i = n-1 stepping +i must cross to a different face.
    Neighbors4 nb = table.neighbors(0, n - 1, n / 2);
    CHECK(nb.plus_i.face != 0);
    CHECK(nb.plus_i.i >= 0);
    CHECK(nb.plus_i.i <  n);
    CHECK(nb.plus_i.j >= 0);
    CHECK(nb.plus_i.j <  n);
}
