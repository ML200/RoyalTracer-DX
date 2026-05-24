#include <doctest/doctest.h>

#include "core/param_registry.h"

#include <filesystem>
#include <string>

using namespace pb;

namespace {

ParamRegistry make_registry_with_sample_params() {
    ParamRegistry r;
    r.declare_int  ("noise.octaves",   6,    1, 12,  "octaves",   "",         "",   "noise");
    r.declare_float("noise.frequency", 1.2f, 0.1f, 6.0f, "frequency", "",     "",   "noise");
    r.declare_bool ("noise.flag",      true,                        "flag",      "",     "",   "noise");
    r.declare_float("other.gain",      0.5f, 0.0f, 1.0f, "gain",      "",     "",   "other");
    return r;
}

}

TEST_CASE("ParamRegistry declare + get + set round-trip") {
    ParamRegistry r = make_registry_with_sample_params();

    CHECK(r.get_int  ("noise.octaves")   == 6);
    CHECK(r.get_float("noise.frequency") == doctest::Approx(1.2f));
    CHECK(r.get_bool ("noise.flag")      == true);

    r.set_int  ("noise.octaves",   9);
    r.set_float("noise.frequency", 2.5f);
    r.set_bool ("noise.flag",      false);

    CHECK(r.get_int  ("noise.octaves")   == 9);
    CHECK(r.get_float("noise.frequency") == doctest::Approx(2.5f));
    CHECK(r.get_bool ("noise.flag")      == false);
}

TEST_CASE("ParamRegistry clamps int and float to [lo, hi]") {
    ParamRegistry r = make_registry_with_sample_params();

    r.set_int  ("noise.octaves",   1000);
    r.set_float("noise.frequency", 99.0f);
    r.set_int  ("noise.octaves",   -5);

    CHECK(r.get_int("noise.octaves") == 1);
    CHECK(r.get_float("noise.frequency") == doctest::Approx(6.0f));
}

TEST_CASE("ParamRegistry change callback fires on set") {
    ParamRegistry r = make_registry_with_sample_params();
    int           fired = 0;
    std::string   last_path;

    r.on_change([&](std::string_view path) {
        ++fired;
        last_path = std::string(path);
    });

    r.set_int  ("noise.octaves", 7);
    CHECK(fired == 1);
    CHECK(last_path == "noise.octaves");

    //Setting to the same value should not fire.
    r.set_int("noise.octaves", 7);
    CHECK(fired == 1);

    r.set_float("noise.frequency", 3.0f);
    CHECK(fired == 2);
    CHECK(last_path == "noise.frequency");
}

TEST_CASE("ParamRegistry prefix_iter only visits matching paths") {
    ParamRegistry r = make_registry_with_sample_params();
    std::vector<std::string> visited;
    r.prefix_iter("noise.", [&](std::string_view path, const ParamSlot&) {
        visited.emplace_back(path);
    });
    CHECK(visited.size() == 3);
    CHECK(visited[0] == "noise.flag");          //lexicographic
    CHECK(visited[1] == "noise.frequency");
    CHECK(visited[2] == "noise.octaves");
}

TEST_CASE("ParamRegistry JSON round-trip restores values") {
    auto tmp = std::filesystem::temp_directory_path() / "pb_param_test_roundtrip.json";
    std::filesystem::remove(tmp);

    {
        ParamRegistry a = make_registry_with_sample_params();
        a.set_int  ("noise.octaves",   8);
        a.set_float("noise.frequency", 3.14f);
        a.set_bool ("noise.flag",      false);
        a.set_float("other.gain",      0.75f);
        CHECK(a.save(tmp));
    }

    ParamRegistry b = make_registry_with_sample_params();
    CHECK(b.load(tmp));
    CHECK(b.get_int  ("noise.octaves")   == 8);
    CHECK(b.get_float("noise.frequency") == doctest::Approx(3.14f));
    CHECK(b.get_bool ("noise.flag")      == false);
    CHECK(b.get_float("other.gain")      == doctest::Approx(0.75f));

    std::filesystem::remove(tmp);
}

TEST_CASE("ParamRegistry hash_prefix_into differs on value change") {
    ParamRegistry r = make_registry_with_sample_params();

    Hasher h1;
    r.hash_prefix_into("noise.", h1);
    auto k1 = h1.finish();

    r.set_float("noise.frequency", 4.0f);
    Hasher h2;
    r.hash_prefix_into("noise.", h2);
    auto k2 = h2.finish();

    CHECK(k1 != k2);

    //Changing a different-prefix slot should not affect the noise hash.
    r.set_float("other.gain", 0.1f);
    Hasher h3;
    r.hash_prefix_into("noise.", h3);
    CHECK(h3.finish() == k2);
}

TEST_CASE("ParamRegistry reset_unaffected leaves listed paths and resets others") {
    ParamRegistry r = make_registry_with_sample_params();
    r.set_int  ("noise.octaves",   8);
    r.set_float("noise.frequency", 3.14f);
    r.set_float("other.gain",      0.99f);

    r.reset_unaffected({"noise.octaves"});

    CHECK(r.get_int  ("noise.octaves")   == 8);          //touched, kept
    CHECK(r.get_float("noise.frequency") == doctest::Approx(1.2f));  //reset
    CHECK(r.get_float("other.gain")      == doctest::Approx(0.5f));  //reset
}
