#include <doctest/doctest.h>

#include "core/cache.h"
#include "core/cubed_sphere.h"
#include "core/field.h"

#include <cuda_runtime.h>
#include <vector_types.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>

using namespace pb;

namespace {

std::filesystem::path scratch_dir(const char* tag) {
    auto p = std::filesystem::temp_directory_path() / (std::string("pb_cache_test_") + tag);
    std::filesystem::remove_all(p);
    std::filesystem::create_directories(p);
    return p;
}

template <typename T>
void fill_sequence(std::vector<T>& v, T base) {
    for (std::size_t i = 0; i < v.size(); ++i) {
        v[i] = static_cast<T>(static_cast<int>(base) + static_cast<int>(i));
    }
}

}

TEST_CASE("cache round-trip Field<float> is bit-identical") {
    auto root = scratch_dir("f1");
    CubedSphereGrid g(8);

    Field<float> src(g);
    std::vector<float> host(static_cast<std::size_t>(g.total_cells()));
    for (std::size_t i = 0; i < host.size(); ++i) host[i] = static_cast<float>(i) * 0.5f;
    src.upload(host);

    CHECK(save_field(root, "stub", 0xdeadbeefcafebabeull, "field_a", src));

    Field<float> dst(g);
    CHECK(load_field(root, "stub", 0xdeadbeefcafebabeull, "field_a", dst));

    std::vector<float> back;
    dst.download(back);
    REQUIRE(back.size() == host.size());
    for (std::size_t i = 0; i < host.size(); ++i) {
        CHECK(back[i] == host[i]);
    }
}

TEST_CASE("cache round-trip Field<uint8_t> bit-identical") {
    auto root = scratch_dir("u8");
    CubedSphereGrid g(8);

    Field<std::uint8_t> src(g);
    std::vector<std::uint8_t> host(static_cast<std::size_t>(g.total_cells()));
    for (std::size_t i = 0; i < host.size(); ++i) {
        host[i] = static_cast<std::uint8_t>(i & 0xFF);
    }
    src.upload(host);

    CHECK(save_field(root, "stub", 1ull, "byte_field", src));

    Field<std::uint8_t> dst(g);
    CHECK(load_field(root, "stub", 1ull, "byte_field", dst));

    std::vector<std::uint8_t> back;
    dst.download(back);
    REQUIRE(back.size() == host.size());
    for (std::size_t i = 0; i < host.size(); ++i) CHECK(back[i] == host[i]);
}

TEST_CASE("cache round-trip Field<float2> and Field<float4> bit-identical") {
    auto root = scratch_dir("f2f4");
    CubedSphereGrid g(8);

    {
        Field<float2> src(g);
        std::vector<float2> host(static_cast<std::size_t>(g.total_cells()));
        for (std::size_t i = 0; i < host.size(); ++i) {
            host[i] = make_float2(static_cast<float>(i),
                                  static_cast<float>(i) * 0.25f);
        }
        src.upload(host);
        CHECK(save_field(root, "stub", 2ull, "wind", src));

        Field<float2> dst(g);
        CHECK(load_field(root, "stub", 2ull, "wind", dst));
        std::vector<float2> back;
        dst.download(back);
        REQUIRE(back.size() == host.size());
        for (std::size_t i = 0; i < host.size(); ++i) {
            CHECK(back[i].x == host[i].x);
            CHECK(back[i].y == host[i].y);
        }
    }

    {
        Field<float4> src(g);
        std::vector<float4> host(static_cast<std::size_t>(g.total_cells()));
        for (std::size_t i = 0; i < host.size(); ++i) {
            host[i] = make_float4(static_cast<float>(i),
                                  static_cast<float>(i) * 0.25f,
                                  static_cast<float>(i) * 0.5f,
                                  static_cast<float>(i) * 0.75f);
        }
        src.upload(host);
        CHECK(save_field(root, "stub", 3ull, "biome", src));

        Field<float4> dst(g);
        CHECK(load_field(root, "stub", 3ull, "biome", dst));
        std::vector<float4> back;
        dst.download(back);
        REQUIRE(back.size() == host.size());
        for (std::size_t i = 0; i < host.size(); ++i) {
            CHECK(back[i].x == host[i].x);
            CHECK(back[i].y == host[i].y);
            CHECK(back[i].z == host[i].z);
            CHECK(back[i].w == host[i].w);
        }
    }
}

TEST_CASE("cache load rejects file with bad magic") {
    auto root = scratch_dir("badmagic");
    CubedSphereGrid g(8);
    Field<float> src(g);
    std::vector<float> host(static_cast<std::size_t>(g.total_cells()), 1.0f);
    src.upload(host);
    CHECK(save_field(root, "stub", 0xabcdull, "field", src));

    auto p = file_path(root, "stub", 0xabcdull, "field");
    {
        std::fstream f(p, std::ios::binary | std::ios::in | std::ios::out);
        REQUIRE(f);
        f.seekp(0);
        f.write("XXXX", 4);  //corrupt magic
    }

    Field<float> dst(g);
    CHECK(load_field(root, "stub", 0xabcdull, "field", dst) == false);
}

TEST_CASE("cache load rejects shape mismatch") {
    auto root = scratch_dir("shape");
    CubedSphereGrid g(8);
    Field<float> src(g);
    std::vector<float> host(static_cast<std::size_t>(g.total_cells()), 2.0f);
    src.upload(host);
    CHECK(save_field(root, "stub", 0xfeedull, "field", src));

    CubedSphereGrid g2(16);  //different N
    Field<float> dst(g2);
    CHECK(load_field(root, "stub", 0xfeedull, "field", dst) == false);
}
