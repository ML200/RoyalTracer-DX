#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_NO_STDIO
#include "../src/Util/stb_image.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "minecraft/mc_types.h"
#include "minecraft/nbt.h"
#include "minecraft/anvil.h"
#include "minecraft/block_registry.h"
#include "minecraft/mc_zip.h"
#include "minecraft/mc_models.h"
#include "minecraft/voxel_store.h"
#include "minecraft/voxel_mesher.h"
#include "minecraft/lod_tree.h"
#include "minecraft/mc_world.h"
#include "minecraft/mc_bake.h"
#include "minecraft/mc_materials.h"
#include "minecraft/mc_placement.h"
#include "minecraft/mc_omm.h"
#include "LightTree.h"
#include "planet/worker_pool.h"

using namespace mc;

static int g_checks = 0, g_failures = 0;
#define CHECK(cond) do { ++g_checks; if (!(cond)) { ++g_failures; std::printf("  FAIL  %s   (line %d)\n", #cond, __LINE__); } } while (0)
#define CHECK_EQ(a, b) do { ++g_checks; const auto a_ = (a); const auto b_ = (b); if (!(a_ == b_)) { ++g_failures; \
    std::printf("  FAIL  %s == %s  (got %lld want %lld, line %d)\n", #a, #b, (long long)a_, (long long)b_, __LINE__); } } while (0)

static void test_nbt() {
    std::printf("[nbt]\n");
    NbtWriter w;
    w.begin_compound("root");
    w.write_byte("b", -5);
    w.write_short("s", -300);
    w.write_int("i", 123456789);
    w.write_long("l", -1234567890123LL);
    w.write_float("f", 1.5f);
    w.write_double("d", 2.25);
    w.write_string("str", "hello");
    w.write_byte_array("ba", { 1, 2, 3 });
    w.write_int_array("ia", { -1, 7 });
    w.write_long_array("la", { 1LL << 40, -2 });
    w.begin_list("lst", NbtType::Compound, 2);
    w.write_int("x", 1); w.end_compound();
    w.write_int("x", 2); w.end_compound();
    w.begin_list("strs", NbtType::String, 2);
    w.write_string_item("a"); w.write_string_item("bc");
    w.end_compound();

    NbtValue root;
    std::string err;
    CHECK(nbt_parse(w.bytes.data(), w.bytes.size(), root, &err));
    CHECK(root.name == "root");
    CHECK_EQ(root.get_int("b"), -5);
    CHECK_EQ(root.get_int("s"), -300);
    CHECK_EQ(root.get_int("i"), 123456789);
    CHECK_EQ(root.get_int("l"), -1234567890123LL);
    CHECK(root.get_double("f") == 1.5);
    CHECK(root.get_double("d") == 2.25);
    CHECK(root.get_string("str") == "hello");
    CHECK(root.find("ba") && root.find("ba")->str.size() == 3 && root.find("ba")->str[2] == 3);
    CHECK(root.find("ia") && root.find("ia")->ints.size() == 2 && root.find("ia")->ints[0] == -1);
    CHECK(root.find("la") && root.find("la")->longs.size() == 2 && root.find("la")->longs[0] == (1LL << 40));
    const NbtValue* lst = root.get_list("lst");
    CHECK(lst && lst->children.size() == 2 && lst->children[1].get_int("x") == 2);
    const NbtValue* strs = root.get_list("strs");
    CHECK(strs && strs->children.size() == 2 && strs->children[1].str == "bc");
    NbtValue bad;
    CHECK(!nbt_parse(w.bytes.data(), w.bytes.size() / 2, bad, &err));
    CHECK(!err.empty());
}

static void test_block_states() {
    std::printf("[block states]\n");
    {
        std::vector<int64_t> longs((4096 + 11) / 12, 0);
        for (uint32_t i = 0; i < 4096; ++i) {
            const uint32_t v = (i * 7) % 32;
            longs[i / 12] |= (int64_t)((uint64_t)v << ((i % 12) * 5));
        }
        std::vector<uint32_t> out(4096);
        unpack_block_states(longs.data(), longs.size(), 5, true, out.data());
        bool ok = true;
        for (uint32_t i = 0; i < 4096; ++i) if (out[i] != (i * 7) % 32) { ok = false; break; }
        CHECK(ok);
    }
    {
        std::vector<int64_t> longs(4096 * 5 / 64, 0);
        std::vector<uint64_t> u(longs.size(), 0);
        for (uint32_t i = 0; i < 4096; ++i) {
            const uint64_t v = (i * 13) % 32;
            const uint64_t bit = (uint64_t)i * 5;
            u[bit >> 6] |= v << (bit & 63);
            if ((bit & 63) + 5 > 64) u[(bit >> 6) + 1] |= v >> (64 - (bit & 63));
        }
        for (size_t k = 0; k < u.size(); ++k) longs[k] = (int64_t)u[k];
        std::vector<uint32_t> out(4096);
        unpack_block_states(longs.data(), longs.size(), 5, false, out.data());
        bool ok = true;
        for (uint32_t i = 0; i < 4096; ++i) if (out[i] != (i * 13) % 32) { ok = false; break; }
        CHECK(ok);
    }
}

static void test_section() {
    std::printf("[section]\n");
    std::vector<Voxel> vals(SECTION_VOXELS, 7);
    Section u = Section::from_values(vals.data());
    CHECK(u.is_uniform() && u.uniform_value() == 7 && !u.all_air());
    CHECK(Section::uniform(0).all_air());
    vals[section_index(3, 4, 5)] = 9;
    Section s = Section::from_values(vals.data());
    CHECK(!s.is_uniform() && s.bits() == 1);
    CHECK_EQ(s.get(3, 4, 5), 9);
    CHECK_EQ(s.get(0, 0, 0), 7);
    for (uint32_t i = 0; i < 300; ++i) s.set(i, 1000 + i);
    CHECK_EQ(s.bits(), 16);
    CHECK_EQ(s.get(299), 1299);
    CHECK_EQ(s.get(3, 4, 5), 9);
    CHECK_EQ(s.get(4095), 7);
    std::vector<Voxel> back(SECTION_VOXELS);
    s.unpack(back.data());
    CHECK_EQ(back[299], 1299);
    CHECK_EQ(back[4095], 7);
    Section u2 = Section::uniform(3);
    u2.set(10, 3);
    CHECK(u2.is_uniform());
    u2.set(10, 4);
    CHECK(!u2.is_uniform() && u2.get(10) == 4 && u2.get(11) == 3);
}

static void test_registry() {
    std::printf("[registry]\n");
    BlockRegistry reg;
    CHECK_EQ(reg.count(), 1u);
    const BlockId stone = reg.intern("minecraft:stone", {});
    const BlockId stone2 = reg.intern("stone", {});
    CHECK(stone == stone2 && stone == 1);
    const BlockId stairs = reg.intern("minecraft:oak_stairs", { { "half", "bottom" }, { "facing", "east" } });
    CHECK(reg.desc(stairs).canonical == "minecraft:oak_stairs[facing=east,half=bottom]");
    CHECK(reg.desc(stairs).prop("facing") == "east");
    CHECK_EQ(reg.intern("minecraft:cave_air", {}), AIR_ID);
    CHECK_EQ(reg.find("minecraft:oak_stairs[facing=east,half=bottom]"), stairs);
    CHECK_EQ(reg.find("nope"), INVALID_BLOCK);
    NbtWriter w;
    w.begin_compound("");
    w.write_string("half", "bottom");
    w.write_string("facing", "east");
    w.end_compound();
    NbtValue props;
    CHECK(nbt_parse(w.bytes.data(), w.bytes.size(), props));
    InternCache cache(reg);
    CHECK_EQ(cache.intern("minecraft:oak_stairs", &props), stairs);
    CHECK_EQ(cache.intern("minecraft:oak_stairs", &props), stairs);
}

static std::vector<uint8_t> make_chunk_117(int cx, int cz) {
    NbtWriter w;
    w.begin_compound("");
    w.write_int("DataVersion", 2730);
    w.begin_compound("Level");
    w.write_int("xPos", cx);
    w.write_int("zPos", cz);
    w.begin_list("Sections", NbtType::Compound, 2);
    w.write_byte("Y", 0);
    w.begin_list("Palette", NbtType::Compound, 3);
    w.write_string("Name", "minecraft:air"); w.end_compound();
    w.write_string("Name", "minecraft:stone"); w.end_compound();
    w.write_string("Name", "minecraft:glass"); w.end_compound();
    {
        std::vector<int64_t> longs(256, 0);
        for (uint32_t i = 0; i < 4096; ++i) {
            const int y = (int)(i >> 8), z = (int)((i >> 4) & 15), x = (int)(i & 15);
            uint64_t v = (y < 8) ? 1 : 0;
            if (x == 3 && y == 8 && z == 3) v = 2;
            longs[i / 16] |= (int64_t)(v << ((i % 16) * 4));
        }
        w.write_long_array("BlockStates", longs);
    }
    w.end_compound();
    w.write_byte("Y", 1);
    w.end_compound();
    w.end_compound();
    w.end_compound();
    return w.bytes;
}

static std::vector<uint8_t> make_chunk_118(int cx, int cz) {
    NbtWriter w;
    w.begin_compound("");
    w.write_int("DataVersion", 3120);
    w.write_int("xPos", cx);
    w.write_int("zPos", cz);
    w.begin_list("sections", NbtType::Compound, 1);
    w.write_byte("Y", -1);
    w.begin_compound("block_states");
    w.begin_list("palette", NbtType::Compound, 1);
    w.write_string("Name", "minecraft:deepslate");
    w.begin_compound("Properties"); w.write_string("axis", "y"); w.end_compound();
    w.end_compound();
    w.end_compound();
    w.end_compound();
    w.end_compound();
    return w.bytes;
}

static void test_anvil(const std::filesystem::path& outDir) {
    std::printf("[anvil]\n");
    BlockRegistry reg;
    InternCache cache(reg);
    {
        const std::vector<uint8_t> bytes = make_chunk_117(5, -3);
        NbtValue root;
        CHECK(nbt_parse(bytes.data(), bytes.size(), root));
        DecodedChunk dc;
        std::string err;
        CHECK(decode_chunk(root, cache, dc, &err));
        CHECK_EQ(dc.cx, 5); CHECK_EQ(dc.cz, -3);
        CHECK_EQ(dc.sections.size(), 1u);
        const BlockId stone = reg.find("minecraft:stone");
        const BlockId glass = reg.find("minecraft:glass");
        CHECK(stone != INVALID_BLOCK && glass != INVALID_BLOCK);
        const Section& s = dc.sections[0].blocks;
        CHECK_EQ(s.get(0, 0, 0), stone);
        CHECK_EQ(s.get(0, 9, 0), AIR_ID);
        CHECK_EQ(s.get(3, 8, 3), glass);
    }
    {
        const std::vector<uint8_t> bytes = make_chunk_118(1, 1);
        NbtValue root;
        CHECK(nbt_parse(bytes.data(), bytes.size(), root));
        DecodedChunk dc;
        CHECK(decode_chunk(root, cache, dc));
        CHECK_EQ(dc.sections.size(), 1u);
        CHECK_EQ(dc.sections[0].y, -1);
        CHECK(dc.sections[0].blocks.is_uniform());
        CHECK_EQ(dc.sections[0].blocks.uniform_value(), reg.find("minecraft:deepslate[axis=y]"));
    }
    {
        std::filesystem::create_directories(outDir);
        const std::filesystem::path p = outDir / "r.2.-1.mca";
        std::vector<uint8_t> file(8192, 0);
        const std::vector<uint8_t> chunk = make_chunk_117(2 * 32 + 4, -1 * 32 + 6);
        const size_t idx = (4 + 6 * 32) * 4;
        file[idx + 2] = 2; file[idx + 3] = 1;
        std::vector<uint8_t> sector(4096, 0);
        const uint32_t len = (uint32_t)chunk.size() + 1;
        sector[0] = (uint8_t)(len >> 24); sector[1] = (uint8_t)(len >> 16); sector[2] = (uint8_t)(len >> 8); sector[3] = (uint8_t)len;
        sector[4] = 3;
        std::memcpy(&sector[5], chunk.data(), chunk.size());
        file.insert(file.end(), sector.begin(), sector.end());
        std::ofstream(p, std::ios::binary).write((const char*)file.data(), (std::streamsize)file.size());
        RegionFile rf;
        std::string err;
        CHECK(rf.open(p.string(), &err));
        CHECK_EQ(rf.rx, 2); CHECK_EQ(rf.rz, -1);
        CHECK(rf.chunk_present(4, 6));
        CHECK(!rf.chunk_present(0, 0));
        std::vector<uint8_t> nbt;
        CHECK(rf.read_chunk(4, 6, nbt, &err));
        NbtValue root;
        CHECK(nbt_parse(nbt.data(), nbt.size(), root));
        DecodedChunk dc;
        CHECK(decode_chunk(root, cache, dc));
        CHECK_EQ(dc.cx, 68); CHECK_EQ(dc.cz, -26);
        NbtWriter w;
        w.begin_compound("");
        w.begin_compound("Data");
        w.write_string("LevelName", "Fixture");
        w.write_int("DataVersion", 2730);
        w.write_int("SpawnX", 115); w.write_int("SpawnY", 71); w.write_int("SpawnZ", -2);
        w.end_compound(); w.end_compound();
        std::ofstream(outDir / "level.dat", std::ios::binary).write((const char*)w.bytes.data(), (std::streamsize)w.bytes.size());
        LevelInfo li;
        CHECK(read_level_dat((outDir / "level.dat").string(), li, &err));
        CHECK(li.name == "Fixture" && li.spawnX == 115 && li.spawnZ == -2 && li.hasSpawn);
    }
}

static void test_zip() {
    std::printf("[zip]\n");
    const std::vector<uint8_t> z = make_stored_zip({ { "a/b.txt", "hello" }, { "c.json", "{}" } });
    ZipArchive ar;
    std::string err;
    CHECK(ar.open_memory(z, &err));
    CHECK_EQ(ar.entry_count(), 2u);
    std::vector<uint8_t> out;
    CHECK(ar.read("a/b.txt", out) && std::string(out.begin(), out.end()) == "hello");
    CHECK(!ar.read("missing", out));
    std::vector<std::string> names;
    ar.list("a/", names);
    CHECK_EQ(names.size(), 1u);
    MemoryResources over;
    over.add("c.json", "{\"over\":1}");
    ResourceStack stack;
    stack.push(&over);
    stack.push(&ar);
    CHECK(stack.read("c.json", out) && std::string(out.begin(), out.end()) == "{\"over\":1}");
    CHECK(stack.read("a/b.txt", out) && out.size() == 5);
}

static void add_vanilla_like_models(MemoryResources& r) {
    r.add("assets/minecraft/models/block/block.json", "{ \"gui_light\": \"side\" }");
    r.add("assets/minecraft/models/block/cube.json",
          "{ \"parent\": \"block/block\", \"elements\": [ { \"from\": [0,0,0], \"to\": [16,16,16], \"faces\": {"
          "\"down\": {\"texture\": \"#down\", \"cullface\": \"down\"}, \"up\": {\"texture\": \"#up\", \"cullface\": \"up\"},"
          "\"north\": {\"texture\": \"#north\", \"cullface\": \"north\"}, \"south\": {\"texture\": \"#south\", \"cullface\": \"south\"},"
          "\"west\": {\"texture\": \"#west\", \"cullface\": \"west\"}, \"east\": {\"texture\": \"#east\", \"cullface\": \"east\"} } } ] }");
    r.add("assets/minecraft/models/block/cube_all.json",
          "{ \"parent\": \"block/cube\", \"textures\": { \"particle\": \"#all\", \"down\": \"#all\", \"up\": \"#all\","
          "\"north\": \"#all\", \"east\": \"#all\", \"south\": \"#all\", \"west\": \"#all\" } }");
    r.add("assets/minecraft/models/block/stone.json", "{ \"parent\": \"minecraft:block/cube_all\", \"textures\": { \"all\": \"minecraft:block/stone\" } }");
    r.add("assets/minecraft/blockstates/stone.json", "{ \"variants\": { \"\": [ { \"model\": \"minecraft:block/stone\" }, { \"model\": \"minecraft:block/stone\", \"y\": 180 } ] } }");
    r.add("assets/minecraft/models/block/stairs.json",
          "{ \"parent\": \"block/block\", \"textures\": { \"particle\": \"#side\" }, \"elements\": ["
          "{ \"from\": [0,0,0], \"to\": [16,8,16], \"faces\": { \"down\": {\"uv\":[0,0,16,16], \"texture\": \"#bottom\", \"cullface\": \"down\"},"
          " \"up\": {\"uv\":[0,0,16,16], \"texture\": \"#top\"}, \"north\": {\"uv\":[0,8,16,16], \"texture\": \"#side\", \"cullface\": \"north\"},"
          " \"south\": {\"uv\":[0,8,16,16], \"texture\": \"#side\", \"cullface\": \"south\"}, \"west\": {\"uv\":[0,8,16,16], \"texture\": \"#side\", \"cullface\": \"west\"},"
          " \"east\": {\"uv\":[0,8,16,16], \"texture\": \"#side\", \"cullface\": \"east\"} } },"
          "{ \"from\": [8,8,0], \"to\": [16,16,16], \"faces\": { \"up\": {\"uv\":[8,0,16,16], \"texture\": \"#top\", \"cullface\": \"up\"},"
          " \"north\": {\"uv\":[0,0,8,8], \"texture\": \"#side\", \"cullface\": \"north\"}, \"south\": {\"uv\":[8,0,16,8], \"texture\": \"#side\", \"cullface\": \"south\"},"
          " \"west\": {\"uv\":[0,0,16,8], \"texture\": \"#side\"}, \"east\": {\"uv\":[0,0,16,8], \"texture\": \"#side\", \"cullface\": \"east\"} } } ] }");
    r.add("assets/minecraft/models/block/oak_stairs.json",
          "{ \"parent\": \"minecraft:block/stairs\", \"textures\": { \"bottom\": \"minecraft:block/oak_planks\", \"top\": \"minecraft:block/oak_planks\", \"side\": \"minecraft:block/oak_planks\" } }");
    r.add("assets/minecraft/blockstates/oak_stairs.json",
          "{ \"variants\": { \"facing=east,half=bottom,shape=straight\": { \"model\": \"minecraft:block/oak_stairs\" },"
          " \"facing=south,half=bottom,shape=straight\": { \"model\": \"minecraft:block/oak_stairs\", \"y\": 90, \"uvlock\": true } } }");
    r.add("assets/minecraft/models/block/fence_post.json",
          "{ \"textures\": { \"particle\": \"#texture\" }, \"elements\": [ { \"from\": [6,0,6], \"to\": [10,16,10], \"faces\": {"
          " \"down\": {\"uv\":[6,6,10,10], \"texture\": \"#texture\", \"cullface\": \"down\"}, \"up\": {\"uv\":[6,6,10,10], \"texture\": \"#texture\"},"
          " \"north\": {\"uv\":[6,0,10,16], \"texture\": \"#texture\"}, \"south\": {\"uv\":[6,0,10,16], \"texture\": \"#texture\"},"
          " \"west\": {\"uv\":[6,0,10,16], \"texture\": \"#texture\"}, \"east\": {\"uv\":[6,0,10,16], \"texture\": \"#texture\"} } } ] }");
    r.add("assets/minecraft/models/block/fence_side.json",
          "{ \"textures\": { \"particle\": \"#texture\" }, \"elements\": [ { \"from\": [7,12,0], \"to\": [9,15,9], \"faces\": {"
          " \"down\": {\"texture\": \"#texture\"}, \"up\": {\"texture\": \"#texture\"}, \"north\": {\"texture\": \"#texture\", \"cullface\": \"north\"},"
          " \"west\": {\"texture\": \"#texture\"}, \"east\": {\"texture\": \"#texture\"} } } ] }");
    r.add("assets/minecraft/models/block/oak_fence_post.json", "{ \"parent\": \"minecraft:block/fence_post\", \"textures\": { \"texture\": \"minecraft:block/oak_planks\" } }");
    r.add("assets/minecraft/models/block/oak_fence_side.json", "{ \"parent\": \"minecraft:block/fence_side\", \"textures\": { \"texture\": \"minecraft:block/oak_planks\" } }");
    r.add("assets/minecraft/blockstates/oak_fence.json",
          "{ \"multipart\": [ { \"apply\": { \"model\": \"minecraft:block/oak_fence_post\" } },"
          " { \"apply\": { \"model\": \"minecraft:block/oak_fence_side\", \"uvlock\": true }, \"when\": { \"north\": \"true\" } },"
          " { \"apply\": { \"model\": \"minecraft:block/oak_fence_side\", \"uvlock\": true, \"y\": 90 }, \"when\": { \"east\": \"true\" } } ] }");
    r.add("assets/minecraft/models/block/grass_block.json",
          "{ \"parent\": \"block/block\", \"textures\": { \"particle\": \"block/dirt\", \"bottom\": \"block/dirt\", \"top\": \"block/grass_block_top\","
          " \"side\": \"block/grass_block_side\", \"overlay\": \"block/grass_block_side_overlay\" }, \"elements\": ["
          "{ \"from\": [0,0,0], \"to\": [16,16,16], \"faces\": { \"down\": {\"texture\": \"#bottom\", \"cullface\": \"down\"}, \"up\": {\"texture\": \"#top\", \"cullface\": \"up\", \"tintindex\": 0},"
          " \"north\": {\"texture\": \"#side\", \"cullface\": \"north\"}, \"south\": {\"texture\": \"#side\", \"cullface\": \"south\"},"
          " \"west\": {\"texture\": \"#side\", \"cullface\": \"west\"}, \"east\": {\"texture\": \"#side\", \"cullface\": \"east\"} } },"
          "{ \"from\": [0,0,0], \"to\": [16,16,16], \"faces\": { \"north\": {\"texture\": \"#overlay\", \"tintindex\": 0, \"cullface\": \"north\"},"
          " \"south\": {\"texture\": \"#overlay\", \"tintindex\": 0, \"cullface\": \"south\"}, \"west\": {\"texture\": \"#overlay\", \"tintindex\": 0, \"cullface\": \"west\"},"
          " \"east\": {\"texture\": \"#overlay\", \"tintindex\": 0, \"cullface\": \"east\"} } } ] }");
    r.add("assets/minecraft/blockstates/grass_block.json", "{ \"variants\": { \"snowy=false\": { \"model\": \"minecraft:block/grass_block\" }, \"snowy=true\": { \"model\": \"minecraft:block/grass_block_snow\" } } }");
    r.add("assets/minecraft/models/block/cross.json",
          "{ \"textures\": { \"particle\": \"#cross\" }, \"elements\": [ { \"from\": [0.8,0,8], \"to\": [15.2,16,8],"
          " \"rotation\": { \"origin\": [8,8,8], \"axis\": \"y\", \"angle\": 45, \"rescale\": true }, \"shade\": false,"
          " \"faces\": { \"north\": {\"uv\":[0,0,16,16], \"texture\": \"#cross\"}, \"south\": {\"uv\":[0,0,16,16], \"texture\": \"#cross\"} } } ] }");
    r.add("assets/minecraft/models/block/poppy.json", "{ \"parent\": \"minecraft:block/cross\", \"textures\": { \"cross\": \"minecraft:block/poppy\" } }");
    r.add("assets/minecraft/blockstates/poppy.json", "{ \"variants\": { \"\": { \"model\": \"minecraft:block/poppy\" } } }");
    r.add("assets/minecraft/models/block/chest.json", "{ \"textures\": { \"particle\": \"minecraft:block/oak_planks\" } }");
    r.add("assets/minecraft/blockstates/chest.json", "{ \"variants\": { \"\": { \"model\": \"minecraft:block/chest\" } } }");
}

static int face_of(const RawQuad& q) {
    const Vec3f& n = q.normal;
    const float ax = std::fabs(n.x), ay = std::fabs(n.y), az = std::fabs(n.z);
    if (ay >= ax && ay >= az) return n.y > 0 ? FACE_UP : FACE_DOWN;
    if (ax >= az) return n.x > 0 ? FACE_EAST : FACE_WEST;
    return n.z > 0 ? FACE_SOUTH : FACE_NORTH;
}

static void test_models() {
    std::printf("[models]\n");
    MemoryResources res;
    add_vanilla_like_models(res);
    ModelResolver mr(res);
    BlockRegistry reg;
    std::string err;

    {
        ResolvedShape sh;
        CHECK(mr.resolve(reg.desc(reg.intern("minecraft:stone", {})), sh, &err));
        CHECK_EQ(sh.quads.size(), 6u);
        bool okTex = true, okCull = true, okWind = true;
        for (const RawQuad& q : sh.quads) {
            okTex = okTex && q.texture == "minecraft:block/stone";
            okCull = okCull && q.cullFace == face_of(q);
            const Vec3f g = cross(q.pos[1] - q.pos[0], q.pos[2] - q.pos[0]);
            okWind = okWind && dot(g, q.normal) > 0.0f;
        }
        CHECK(okTex); CHECK(okCull); CHECK(okWind);
        CHECK(sh.particle == "minecraft:block/stone");
    }
    {
        ResolvedShape east, south;
        CHECK(mr.resolve(reg.desc(reg.intern("minecraft:oak_stairs", { { "facing", "east" }, { "half", "bottom" }, { "shape", "straight" } })), east, &err));
        CHECK(mr.resolve(reg.desc(reg.intern("minecraft:oak_stairs", { { "facing", "south" }, { "half", "bottom" }, { "shape", "straight" } })), south, &err));
        CHECK_EQ(east.quads.size(), 11u);
        CHECK_EQ(south.quads.size(), 11u);
        int risersEastAtX8 = 0, risersSouthAtZ8 = 0;
        for (const RawQuad& q : east.quads)
            if (face_of(q) == FACE_WEST && std::fabs(q.pos[0].x - 8.0f) < 1e-3f) ++risersEastAtX8;
        for (const RawQuad& q : south.quads)
            if (face_of(q) == FACE_NORTH && std::fabs(q.pos[0].z - 8.0f) < 1e-3f) ++risersSouthAtZ8;
        CHECK_EQ(risersEastAtX8, 1);
        CHECK_EQ(risersSouthAtZ8, 1);
        int cullSouth = 0;
        for (const RawQuad& q : south.quads) if (q.cullFace == FACE_SOUTH) ++cullSouth;
        CHECK(cullSouth >= 2);
    }
    {
        ResolvedShape sh;
        CHECK(mr.resolve(reg.desc(reg.intern("minecraft:oak_fence", { { "north", "true" }, { "east", "true" }, { "south", "false" }, { "west", "false" }, { "waterlogged", "false" } })), sh, &err));
        CHECK_EQ(sh.quads.size(), 16u);
        ResolvedShape post;
        CHECK(mr.resolve(reg.desc(reg.intern("minecraft:oak_fence", { { "north", "false" }, { "east", "false" }, { "south", "false" }, { "west", "false" }, { "waterlogged", "false" } })), post, &err));
        CHECK_EQ(post.quads.size(), 6u);
    }
    {
        ResolvedShape sh;
        CHECK(mr.resolve(reg.desc(reg.intern("minecraft:grass_block", { { "snowy", "false" } })), sh, &err));
        CHECK_EQ(sh.quads.size(), 6u);
        int overlays = 0, tintedTop = 0;
        for (const RawQuad& q : sh.quads) {
            if (!q.overlay.empty()) ++overlays;
            if (face_of(q) == FACE_UP && q.tintIndex == 0) ++tintedTop;
        }
        CHECK_EQ(overlays, 4);
        CHECK_EQ(tintedTop, 1);
    }
    {
        ResolvedShape sh;
        CHECK(mr.resolve(reg.desc(reg.intern("minecraft:poppy", {})), sh, &err));
        CHECK_EQ(sh.quads.size(), 2u);
        float maxExtent = 0.0f;
        for (const RawQuad& q : sh.quads) for (int k = 0; k < 4; ++k) maxExtent = std::max(maxExtent, std::fabs(q.pos[k].x - 8.0f));
        CHECK(maxExtent > 7.0f && maxExtent < 8.01f);
        CHECK(sh.quads[0].cullFace == -1);
    }
    {
        ResolvedShape sh;
        CHECK(mr.resolve(reg.desc(reg.intern("minecraft:chest", { { "facing", "north" } })), sh, &err));
        CHECK(sh.quads.empty());
        CHECK(sh.particle == "minecraft:block/oak_planks");
    }
    {
        ResolvedShape sh;
        CHECK(!mr.resolve(reg.desc(reg.intern("minecraft:nothing", {})), sh, &err));
        CHECK(!sh.found);
    }
    CHECK(ModelResolver::texture_resource_path("block/stone") == "assets/minecraft/textures/block/stone.png");
}

static void test_bake() {
    std::printf("[bake]\n");
    std::vector<uint8_t> red(16 * 16 * 4), half(16 * 16 * 4);
    for (int i = 0; i < 256; ++i) {
        red[i * 4] = 200; red[i * 4 + 1] = 30; red[i * 4 + 2] = 30; red[i * 4 + 3] = 255;
        const int x = i % 16;
        half[i * 4] = 30; half[i * 4 + 1] = 200; half[i * 4 + 2] = 30; half[i * 4 + 3] = x < 8 ? 0 : 255;
    }
    auto lookup = [&](const RawQuad& q) -> BakeTexture {
        if (q.texture == "red")  return BakeTexture{ red.data(), 16, 16, 64 };
        if (q.texture == "half") return BakeTexture{ half.data(), 16, 16, 64 };
        return BakeTexture{};
    };
    auto quad = [](Vec3f a, Vec3f b, Vec3f c, Vec3f d, Vec3f n, const char* tex) {
        RawQuad q;
        q.pos[0] = a; q.pos[1] = b; q.pos[2] = c; q.pos[3] = d; q.normal = n; q.texture = tex;
        const float uv[4][2] = { { 0, 0 }, { 16, 0 }, { 16, 16 }, { 0, 16 } };
        for (int k = 0; k < 4; ++k) { q.uv[k][0] = uv[k][0]; q.uv[k][1] = uv[k][1]; }
        return q;
    };
    std::vector<RawQuad> slab;
    slab.push_back(quad({ 0, 8, 0 }, { 0, 8, 16 }, { 16, 8, 16 }, { 16, 8, 0 }, { 0, 1, 0 }, "red"));
    slab.push_back(quad({ 0, 0, 0 }, { 16, 0, 0 }, { 16, 0, 16 }, { 0, 0, 16 }, { 0, -1, 0 }, "red"));
    slab.push_back(quad({ 0, 0, 0 }, { 0, 8, 0 }, { 16, 8, 0 }, { 16, 0, 0 }, { 0, 0, -1 }, "red"));
    slab.push_back(quad({ 0, 0, 16 }, { 16, 0, 16 }, { 16, 8, 16 }, { 0, 8, 16 }, { 0, 0, 1 }, "red"));
    slab.push_back(quad({ 0, 0, 0 }, { 0, 0, 16 }, { 0, 8, 16 }, { 0, 8, 0 }, { -1, 0, 0 }, "red"));
    slab.push_back(quad({ 16, 0, 0 }, { 16, 8, 0 }, { 16, 8, 16 }, { 16, 0, 16 }, { 1, 0, 0 }, "red"));
    BakedFace faces[6];
    bake_block_faces(slab, 16, lookup, faces);
    CHECK(std::fabs(faces[FACE_UP].coverage - 1.0f) < 1e-3f);
    CHECK(std::fabs(faces[FACE_UP].minDepth - 0.5f) < 1e-3f);
    CHECK(std::fabs(faces[FACE_DOWN].coverage - 1.0f) < 1e-3f);
    CHECK(faces[FACE_DOWN].minDepth < 1e-3f);
    CHECK(std::fabs(faces[FACE_NORTH].coverage - 0.5f) < 1e-3f);
    {
        bool lowerRows = true;
        for (int y = 0; y < 16; ++y) for (int x = 0; x < 16; ++x) {
            const bool drawn = faces[FACE_NORTH].rgba[((size_t)y * 16 + x) * 4 + 3] != 0;
            if (drawn != (y >= 8)) lowerRows = false;
        }
        CHECK(lowerRows);
        CHECK(faces[FACE_NORTH].rgba[(15 * 16 + 0) * 4] == 200);
    }
    std::vector<RawQuad> cross;
    cross.push_back(quad({ 0, 0, 0 }, { 0, 16, 0 }, { 16, 16, 16 }, { 16, 0, 16 }, { 0.707f, 0, -0.707f }, "half"));
    cross.push_back(quad({ 0, 0, 16 }, { 0, 16, 16 }, { 16, 16, 0 }, { 16, 0, 0 }, { 0.707f, 0, 0.707f }, "half"));
    bake_block_faces(cross, 16, lookup, faces);
    CHECK(faces[FACE_UP].empty());
    CHECK(faces[FACE_NORTH].coverage > 0.4f && faces[FACE_NORTH].coverage < 0.6f);
    CHECK(faces[FACE_NORTH].minDepth < 0.1f);
    {
        std::vector<uint8_t> img(4 * 4 * 4, 0);
        for (int y = 0; y < 4; ++y) for (int x = 0; x < 2; ++x) { uint8_t* p = &img[((size_t)y * 4 + x) * 4]; p[0] = 10; p[1] = 200; p[2] = 30; p[3] = 255; }
        const double cov = fill_holes(img, 4, 4);
        CHECK(std::fabs(cov - 0.5) < 1e-9);
        bool filled = true;
        for (int i = 0; i < 16; ++i) if (img[i * 4 + 3] != 255 || img[i * 4 + 1] != 200) filled = false;
        CHECK(filled);
        std::vector<uint8_t> none(4 * 4 * 4, 0);
        CHECK(fill_holes(none, 4, 4) == 0.0);
        CHECK(none[3] == 0);
    }
}

static void test_placement() {
    std::printf("[placement]\n");
    Placement p;
    const float m[16] = { 2, 0, 0, 0,  0, 2, 0, 0,  0, 0, 2, 0,  10, 20, 30, 1 };
    p.set(m);
    CHECK(!p.identity);
    double b[3] = { 1, 1, 1 }, s[3], back[3];
    p.to_scene(b, s);
    CHECK(std::fabs(s[0] - 12) < 1e-9 && std::fabs(s[1] - 22) < 1e-9 && std::fabs(s[2] - 32) < 1e-9);
    p.to_blocks(s, back);
    CHECK(std::fabs(back[0] - 1) < 1e-9 && std::fabs(back[1] - 1) < 1e-9 && std::fabs(back[2] - 1) < 1e-9);
    CHECK(std::fabs(p.area_scale() - 4.0) < 1e-9);
    const float r[16] = { 0, 0, -1, 0,  0, 1, 0, 0,  1, 0, 0, 0,  0, 0, 0, 1 };
    Placement q;
    q.set(r);
    const double ex[3] = { 1, 0, 0 };
    q.dir_to_scene(ex, s);
    CHECK(std::fabs(s[0]) < 1e-9 && std::fabs(s[2] + 1) < 1e-9);
    q.to_blocks(s, back);
    CHECK(std::fabs(back[0] - 1) < 1e-9 && std::fabs(back[2]) < 1e-9);
    CHECK(std::fabs(q.area_scale() - 1.0) < 1e-9);
    Placement id;
    CHECK(id.identity);
}

static void setup_test_registry(BlockRegistry& reg) {
    reg.intern("minecraft:stone", {});
    reg.intern("minecraft:glass", {});
    reg.intern("minecraft:stone_slab", { { "type", "bottom" } });
    reg.intern("minecraft:poppy", {});
    reg.intern("minecraft:glowstone", {});
    reg.intern("minecraft:dirt", {});
    reg.intern("minecraft:grass_block", {});
    reg.intern("minecraft:water", {});
    reg.intern("minecraft:rail", {});
    reg.intern("minecraft:red_carpet", {});
    reg.materialAlpha = { 0, 1, 0, 0, 0, 1, 1, 0, 0 };
    reg.materialCutoutTexture = { -1, -1, -1, -1, -1, -1, 0, -1, -1 };
    reg.textureAlpha.resize(1);
    reg.textureAlpha[0].width = 16; reg.textureAlpha[0].height = 16; reg.textureAlpha[0].alpha.assign(256, 255);
    reg.materialEmission = { Vec3f{}, Vec3f{}, Vec3f{}, Vec3f{}, Vec3f{ 1.0f, 0.8f, 0.5f }, Vec3f{}, Vec3f{}, Vec3f{}, Vec3f{} };
    {
        BlockInfo& c = reg.info(10);
        c.isCube = false; c.hasQuads = true; c.sig = Significance::None;
        c.quadBegin = (uint32_t)reg.quads.size();
        BlockQuad top;
        top.pos[0] = { 0, 0.06f, 0 }; top.pos[1] = { 0, 0.06f, 1 }; top.pos[2] = { 1, 0.06f, 1 }; top.pos[3] = { 1, 0.06f, 0 };
        for (int k = 0; k < 4; ++k) { top.uv[k][0] = top.pos[k].x; top.uv[k][1] = top.pos[k].z; }
        top.normal = { 0, 1, 0 }; top.material = 8; top.cullFace = FACE_NONE; top.alphaGeom = 0;
        reg.quads.push_back(top);
        c.quadCount = 1;
        for (int f = 0; f < 6; ++f) { c.lodFaceMaterial[f] = NO_MATERIAL; c.flatFaceMaterial[f] = NO_MATERIAL; }
        c.lodFaceMaterial[FACE_UP] = 8; c.flatFaceMaterial[FACE_UP] = 8;
        c.lodFaceSolid = 1u << FACE_UP;
    }
    {
        BlockInfo& r = reg.info(9);
        r.isCube = false; r.hasQuads = true; r.sig = Significance::None;
        r.quadBegin = (uint32_t)reg.quads.size();
        BlockQuad top;
        top.pos[0] = { 0, 0.06f, 0 }; top.pos[1] = { 0, 0.06f, 1 }; top.pos[2] = { 1, 0.06f, 1 }; top.pos[3] = { 1, 0.06f, 0 };
        for (int k = 0; k < 4; ++k) { top.uv[k][0] = top.pos[k].x; top.uv[k][1] = top.pos[k].z; }
        top.normal = { 0, 1, 0 }; top.material = 6; top.cullFace = FACE_NONE; top.alphaGeom = 1;
        reg.quads.push_back(top);
        r.quadCount = 1;
        for (int f = 0; f < 6; ++f) { r.lodFaceMaterial[f] = NO_MATERIAL; r.flatFaceMaterial[f] = NO_MATERIAL; }
        r.lodFaceMaterial[FACE_UP] = 6; r.flatFaceMaterial[FACE_UP] = 6;
    }
    {
        BlockInfo& w = reg.info(8);
        w.isCube = true; w.fullOpaque = false; w.cullSameId = true; w.sig = Significance::Full; w.volume = 1; w.water = 1;
        for (int f = 0; f < 6; ++f) { w.faceMaterial[f] = 5; w.lodFaceMaterial[f] = 5; w.flatFaceMaterial[f] = 5; }
    }
    for (BlockId id : { (BlockId)5, (BlockId)6, (BlockId)7 }) {
        BlockInfo& b = reg.info(id);
        b.isCube = true; b.fullOpaque = true; b.sig = Significance::Full;
        b.emissive = id == 5 ? 1 : 0;
        const uint16_t m = id == 5 ? 4 : 0;
        for (int f = 0; f < 6; ++f) { b.faceMaterial[f] = m; b.lodFaceMaterial[f] = m; b.flatFaceMaterial[f] = id == 5 ? 4 : 3; }
    }
    {
        BlockInfo& gr = reg.info(7);
        gr.faceMaterial[FACE_UP] = 7; gr.lodFaceMaterial[FACE_UP] = 7; gr.flatFaceMaterial[FACE_UP] = 7;
    }
    {
        BlockInfo& s = reg.info(1);
        s.isCube = true; s.fullOpaque = true; s.sig = Significance::Full;
        for (int f = 0; f < 6; ++f) { s.faceMaterial[f] = 0; s.lodFaceMaterial[f] = 0; s.flatFaceMaterial[f] = 3; }
    }
    {
        BlockInfo& g = reg.info(2);
        g.isCube = true; g.fullOpaque = false; g.cullSameId = true; g.sig = Significance::Full;
        for (int f = 0; f < 6; ++f) { g.faceMaterial[f] = 1; g.lodFaceMaterial[f] = 1; g.flatFaceMaterial[f] = 1; }
    }
    {
        BlockInfo& sl = reg.info(3);
        sl.isCube = false; sl.hasQuads = true; sl.sig = Significance::Partial;
        for (int f = 0; f < 6; ++f) { sl.lodFaceMaterial[f] = 2; sl.flatFaceMaterial[f] = 2; }
        sl.quadBegin = (uint32_t)reg.quads.size();
        BlockQuad top;
        top.pos[0] = { 0, 0.5f, 0 }; top.pos[1] = { 0, 0.5f, 1 }; top.pos[2] = { 1, 0.5f, 1 }; top.pos[3] = { 1, 0.5f, 0 };
        for (int k = 0; k < 4; ++k) { top.uv[k][0] = top.pos[k].x; top.uv[k][1] = top.pos[k].z; }
        top.normal = { 0, 1, 0 }; top.material = 2; top.cullFace = FACE_NONE;
        BlockQuad bottom = top;
        for (int k = 0; k < 4; ++k) bottom.pos[k].y = 0.0f;
        bottom.normal = { 0, -1, 0 }; bottom.cullFace = FACE_DOWN;
        reg.quads.push_back(top); reg.quads.push_back(bottom);
        sl.quadCount = 2;
    }
    {
        BlockInfo& p = reg.info(4);
        p.sig = Significance::None; p.hasQuads = false;
    }
}

static void test_store_and_mesher() {
    std::printf("[store / lod / mesher]\n");
    BlockRegistry reg;
    setup_test_registry(reg);
    VoxelStore store;
    store.configure(0, 15);
    std::vector<Voxel> vals(SECTION_VOXELS, 0);
    for (int y = 0; y < 8; ++y) for (int z = 0; z < 16; ++z) for (int x = 0; x < 16; ++x) vals[section_index(x, y, z)] = 1;
    for (int z = 4; z <= 5; ++z) for (int x = 4; x <= 5; ++x) vals[section_index(x, 8, z)] = 2;
    vals[section_index(10, 8, 10)] = 3;
    vals[section_index(12, 8, 12)] = 4;
    vals[section_index(2, 12, 2)] = 5;
    vals[section_index(14, 8, 2)] = 8;
    vals[section_index(14, 9, 2)] = 8;
    store.put_section(0, 0, 0, 0, Section::from_values(vals.data()));
    store.put_section(0, 1, 0, 0, Section::uniform(1));

    CHECK_EQ(store.get(0, 3, 3, 3), 1u);
    CHECK_EQ(store.get(0, 4, 8, 4), 2u);
    CHECK_EQ(store.get(0, 40, 8, 4), 0u);
    CHECK(store.chunk_occupied(NodeKey{ 0, 0, 0, 0 }));
    CHECK(!store.chunk_occupied(NodeKey{ 0, 3, 0, 0 }));

    store.build_lod(reg, 3, nullptr);
    CHECK_EQ(store.levels(), 3);
    {
        const Voxel v = store.get(1, 2, 4, 2);
        CHECK_EQ(voxel_id(v), 2u);
        CHECK((v & VOX_ANY) != 0);
        CHECK((v & VOX_ALL) == 0);
        const Voxel s = store.get(1, 1, 1, 1);
        CHECK_EQ(voxel_id(s), 1u);
        CHECK((s & VOX_ALL) != 0);
        const Voxel f = store.get(1, 6, 4, 6);
        CHECK_EQ(voxel_id(f), 0u);
        CHECK((f & VOX_OCC) != 0);
        CHECK((f & VOX_ANY) == 0);
        const Voxel sl = store.get(1, 5, 4, 5);
        CHECK_EQ(voxel_id(sl), 3u);
        CHECK((sl & VOX_ANY) != 0 && (sl & VOX_ALL) == 0);
        const Voxel s2 = store.get(2, 0, 0, 0);
        CHECK_EQ(voxel_id(s2), 1u);
        CHECK((s2 & VOX_ALL) != 0);
        const Voxel s3 = store.get(2, 1, 2, 1);
        CHECK((s3 & VOX_ANY) != 0 && (s3 & VOX_ALL) == 0);
    }

    ChunkMesher mesher(reg, store);
    ChunkMesh mesh;
    mesher.mesh(NodeKey{ 0, 0, 0, 0 }, MeshParams{}, mesh);
    CHECK(!mesh.empty());
    CHECK_EQ(mesh.indices.size(), mesh.triangle_count() * 3);
    CHECK_EQ(mesh.materials.size(), mesh.triangle_count());
    CHECK_EQ(mesh.opaqueTriCount + mesh.alphaTriCount, mesh.triangle_count());
    bool ordered = true;
    for (uint32_t t = 0; t < mesh.triangle_count(); ++t) {
        const bool alpha = mesh.materials[t] == 1 || mesh.materials[t] == 5;
        if (t < mesh.opaqueTriCount && alpha) ordered = false;
        if (t >= mesh.opaqueTriCount && !alpha) ordered = false;
    }
    CHECK(ordered);
    CHECK_EQ(mesh.opaqueLightTriCount, 12u);
    CHECK_EQ(mesh.alphaLightTriCount, 0u);
    CHECK_EQ(mesh.light_tri_count(), 12u);
    {
        bool litFirst = true;
        for (uint32_t t = 0; t < mesh.opaqueTriCount; ++t)
            if ((mesh.materials[t] == 4) != (t < mesh.opaqueLightTriCount)) litFirst = false;
        CHECK(litFirst);
        CHECK_EQ(mesh.light_tri_index(3), 3u);
        const uint32_t i0 = mesh.indices[0], i1 = mesh.indices[1], i2 = mesh.indices[2];
        (void)i1;
        const uint16_t one = float_to_half(1.0f), zero = 0u, minusOne = float_to_half(-1.0f);
        const MeshVertex& a = mesh.vertices[i0];
        const MeshVertex& c = mesh.vertices[i2];
        auto unit = [&](uint16_t h) { return h == one || h == zero || h == minusOne || (h & 0x7FFFu) == 0u || h == float_to_half(2.0f) || h == float_to_half(3.0f) || h == float_to_half(-2.0f) || h == float_to_half(-3.0f); };
        CHECK(unit(a.u) && unit(a.v) && unit(c.u) && unit(c.v));
    }
    {
        std::vector<LightTriangle> records;
        for (uint32_t k = 0; k < mesh.light_tri_count(); ++k) {
            const uint32_t t = mesh.light_tri_index(k);
            const MeshVertex* v[3] = { &mesh.vertices[mesh.indices[3 * t]], &mesh.vertices[mesh.indices[3 * t + 1]], &mesh.vertices[mesh.indices[3 * t + 2]] };
            LightTriangle lt{};
            lt.x = { v[0]->px, v[0]->py, v[0]->pz }; lt.y = { v[1]->px, v[1]->py, v[1]->pz }; lt.z = { v[2]->px, v[2]->py, v[2]->pz };
            lt.weight = 0.5f; lt.emission = { 1, 1, 1 };
            records.push_back(lt);
        }
        lt::LightTreeBuilder::SingleBLAS blas;
        lt::LightTreeBuilder::BuildSingleBLAS(records, blas, 16u);
        CHECK_EQ(blas.leafTriLocal.size(), records.size());
        CHECK_EQ(blas.trails.size(), records.size());
        CHECK(blas.nodes.size() >= records.size());
        uint32_t leaves = 0;
        for (const auto& n : blas.nodes) if (n.childCount == 0) { ++leaves; CHECK_EQ(n.triCount, 1u); }
        CHECK_EQ(leaves, (uint32_t)records.size());
        std::vector<bool> seen(records.size(), false);
        for (uint32_t r : blas.leafTriLocal) { CHECK(r < records.size() && !seen[r]); seen[r] = true; }
        bool trailsOk = true;
        for (uint32_t k = 0; k < records.size(); ++k) {
            uint32_t node = 0, depth = 0;
            while (blas.nodes[node].childCount != 0 && depth < 32) {
                const uint32_t child = (uint32_t)((blas.trails[k] >> (2u * depth)) & 3u);
                if (child >= blas.nodes[node].childCount) { trailsOk = false; break; }
                node = blas.nodes[node].firstChild + child;
                ++depth;
            }
            if (blas.nodes[node].childCount != 0 || blas.leafTriLocal[blas.nodes[node].triFirst] != k) trailsOk = false;
        }
        CHECK(trailsOk);
        CHECK(blas.nodes[0].power > 0.0f);
    }
    {
        uint32_t glassTris = 0;
        for (uint32_t t = 0; t < mesh.triangle_count(); ++t) if (mesh.materials[t] == 1) ++glassTris;
        CHECK_EQ(glassTris, 5u * 2u);
    }
    {
        uint32_t stoneTopQuads = 0;
        for (uint32_t t = 0; t + 1 < mesh.triangle_count(); t += 2) {
            if (mesh.materials[t] != 0) continue;
            const MeshVertex& v0 = mesh.vertices[mesh.indices[t * 3]];
            const MeshVertex& v2 = mesh.vertices[mesh.indices[t * 3 + 2]];
            if (std::fabs(v0.py - 8.0f) < 1e-4f && std::fabs(v2.py - 8.0f) < 1e-4f) ++stoneTopQuads;
        }
        CHECK_EQ(stoneTopQuads, 1u);
    }
    {
        uint32_t eastFacesAt16 = 0, westFacesAt16 = 0;
        const uint32_t east = pack_normal_oct16(Vec3f{ 1, 0, 0 }), west = pack_normal_oct16(Vec3f{ -1, 0, 0 });
        for (uint32_t t = 0; t < mesh.triangle_count(); ++t) {
            const MeshVertex& v0 = mesh.vertices[mesh.indices[t * 3]];
            const MeshVertex& v1 = mesh.vertices[mesh.indices[t * 3 + 1]];
            const MeshVertex& v2 = mesh.vertices[mesh.indices[t * 3 + 2]];
            if (std::fabs(v0.px - 16.0f) < 1e-4f && std::fabs(v1.px - 16.0f) < 1e-4f && std::fabs(v2.px - 16.0f) < 1e-4f) {
                if (v0.packedNormal == east) ++eastFacesAt16;
                if (v0.packedNormal == west) ++westFacesAt16;
            }
        }
        CHECK_EQ(eastFacesAt16, 0u);
        CHECK_EQ(westFacesAt16, 2u);
    }
    {
        uint32_t slabTris = 0;
        for (uint32_t t = 0; t < mesh.triangle_count(); ++t) if (mesh.materials[t] == 2) ++slabTris;
        CHECK_EQ(slabTris, 2u);
    }
    {
        uint32_t waterTris = 0, bottomTris = 0;
        bool bottomInset = true;
        const uint32_t down = pack_normal_oct16(Vec3f{ 0, -1, 0 });
        for (uint32_t t = 0; t < mesh.triangle_count(); ++t) {
            if (mesh.materials[t] != 5) continue;
            ++waterTris;
            const MeshVertex& v0 = mesh.vertices[mesh.indices[3 * t]];
            if (v0.packedNormal == down) {
                ++bottomTris;
                if (std::fabs(v0.py - 8.02f) > 1e-4f) bottomInset = false;
            }
        }
        CHECK_EQ(waterTris, 12u);
        CHECK_EQ(bottomTris, 2u);
        CHECK(bottomInset);
    }
    {
        const uint32_t up = pack_normal_oct16(Vec3f{ 0, 1, 0 });
        bool anyUp = false;
        for (const MeshVertex& v : mesh.vertices) if (v.packedNormal == up) { anyUp = true; break; }
        CHECK(anyUp);
        CHECK_EQ(float_to_half(1.0f), 0x3C00u);
        CHECK_EQ(float_to_half(-2.0f), 0xC000u);
        CHECK_EQ(float_to_half(0.0f), 0u);
    }

    ChunkMesh mesh1;
    MeshParams flat; flat.flatMaterials = true;
    mesher.mesh(NodeKey{ 1, 0, 0, 0 }, flat, mesh1);
    CHECK(!mesh1.empty());
    {
        bool allFlat = true;
        for (uint32_t t = 0; t < mesh1.triangle_count(); ++t) if (mesh1.materials[t] != 3 && mesh1.materials[t] != 1 && mesh1.materials[t] != 2 && mesh1.materials[t] != 4 && mesh1.materials[t] != 5) allFlat = false;
        CHECK(allFlat);
        bool anyFloor = false;
        for (const MeshVertex& v : mesh1.vertices) if (std::fabs(v.py - 8.0f) < 1e-4f) anyFloor = true;
        CHECK(anyFloor);
        auto half_to_float = [](uint16_t h) {
            const int e = (h >> 10) & 31, m = h & 1023;
            float v = e == 0 ? (float)m / 1024.0f * std::ldexp(1.0f, -14) : std::ldexp(1.0f + (float)m / 1024.0f, e - 15);
            return (h & 0x8000u) ? -v : v;
        };
        bool tiled = false;
        for (uint32_t t = 0; t < mesh1.triangle_count() && !tiled; ++t) {
            if (mesh1.materials[t] != 3) continue;
            float lo = 1e9f, hi = -1e9f;
            for (int k = 0; k < 3; ++k) {
                const float u = half_to_float(mesh1.vertices[mesh1.indices[3 * t + k]].u);
                lo = std::min(lo, u); hi = std::max(hi, u);
            }
            if (std::fabs(hi - lo - 16.0f) < 1e-4f) tiled = true;
        }
        CHECK(tiled);
        CHECK_EQ(mesh1.opaqueLightTriCount, 12u);
        {
            float lo[3] = { 1e9f, 1e9f, 1e9f }, hi[3] = { -1e9f, -1e9f, -1e9f };
            for (uint32_t t = 0; t < mesh1.triangle_count(); ++t) {
                if (mesh1.materials[t] != 4) continue;
                for (int k = 0; k < 3; ++k) {
                    const MeshVertex& vt = mesh1.vertices[mesh1.indices[3 * t + k]];
                    lo[0] = std::min(lo[0], vt.px); hi[0] = std::max(hi[0], vt.px);
                    lo[1] = std::min(lo[1], vt.py); hi[1] = std::max(hi[1], vt.py);
                    lo[2] = std::min(lo[2], vt.pz); hi[2] = std::max(hi[2], vt.pz);
                }
            }
            CHECK(std::fabs(hi[0] - lo[0] - 1.0f) < 1e-4f && std::fabs(hi[1] - lo[1] - 1.0f) < 1e-4f && std::fabs(hi[2] - lo[2] - 1.0f) < 1e-4f);
            CHECK(std::fabs(lo[0] - 2.5f) < 1e-4f && std::fabs(lo[1] - 12.5f) < 1e-4f && std::fabs(lo[2] - 2.5f) < 1e-4f);
            CHECK_EQ(voxel_emissive(store.get(1, 1, 6, 1)), 1u);
        }
    }

    {
        VoxelStore deco;
        deco.configure(0, 15);
        std::vector<Voxel> g(SECTION_VOXELS, 0);
        g[section_index(5, 5, 5)] = 9;
        g[section_index(5, 9, 5)] = 4;
        deco.put_section(0, 0, 0, 0, Section::from_values(g.data()));
        deco.build_lod(reg, 3, nullptr, 1);
        const Voxel r1 = deco.get(1, 2, 2, 2);
        CHECK_EQ(voxel_id(r1), 9u);
        CHECK((r1 & VOX_ANY) != 0 && (r1 & VOX_ALL) == 0);
        CHECK_EQ(voxel_id(deco.get(2, 1, 1, 1)), 0u);
        CHECK((deco.get(2, 1, 1, 1) & VOX_OCC) != 0);
        CHECK_EQ(voxel_id(deco.get(1, 2, 4, 2)), 0u);
        ChunkMesher dm(reg, deco);
        ChunkMesh rm;
        dm.mesh(NodeKey{ 1, 0, 0, 0 }, MeshParams{}, rm);
        uint32_t railTris = 0;
        for (uint32_t t = 0; t < rm.triangle_count(); ++t) if (rm.materials[t] == 6) ++railTris;
        CHECK_EQ(railTris, 2u);
        CHECK_EQ(rm.alphaTriCount, 2u);
        CHECK_EQ(rm.ommKeys.size(), rm.triangle_count());
        uint32_t keyed = 0;
        for (uint32_t t = 0; t < rm.triangle_count(); ++t)
            if (rm.materials[t] == 6 && rm.ommKeys[t] == omm_key_face(0u, FACE_UP, 1, 1, 1, t == rm.opaqueTriCount ? 0 : 1)) ++keyed;
        CHECK_EQ(keyed, 2u);
        ChunkMesh r0;
        dm.mesh(NodeKey{ 0, 0, 0, 0 }, MeshParams{}, r0);
        uint32_t quadKeyed = 0;
        for (uint32_t t = 0; t < r0.triangle_count(); ++t)
            if (r0.ommKeys[t] == omm_key_quad(reg.info(9).quadBegin, 0) || r0.ommKeys[t] == omm_key_quad(reg.info(9).quadBegin, 1)) ++quadKeyed;
        CHECK_EQ(quadKeyed, 2u);
        std::vector<OmmBakeTri> omm;
        enumerate_omm_triangles(reg, omm);
        CHECK_EQ(omm.size(), 26u);
        float uv[4][2];
        ChunkMesher::face_quad_uvs(FACE_UP, 1, 1, 2.0f, uv);
        float lo = 1e9f, hi = -1e9f;
        for (int k = 0; k < 4; ++k) { lo = std::min(lo, uv[k][0]); hi = std::max(hi, uv[k][0]); }
        CHECK(std::fabs(hi - lo - 2.0f) < 1e-6f);
    }

    {
        VoxelStore surf;
        surf.configure(0, 15);
        std::vector<Voxel> g(SECTION_VOXELS, 0);
        for (int z = 0; z < 2; ++z) for (int x = 0; x < 2; ++x) g[section_index(x, 0, z)] = 1;
        g[section_index(0, 1, 0)] = 3; g[section_index(1, 1, 0)] = 3; g[section_index(0, 1, 1)] = 3;
        g[section_index(1, 1, 1)] = 5;
        for (int z = 8; z < 10; ++z) { g[section_index(8, 0, z)] = 6; g[section_index(9, 0, z)] = 6; g[section_index(8, 1, z)] = 7; }
        for (int z = 12; z < 14; ++z) for (int x = 12; x < 14; ++x) { g[section_index(x, 0, z)] = 7; g[section_index(x, 1, z)] = 4; }
        surf.put_section(0, 0, 0, 0, Section::from_values(g.data()));
        surf.build_lod(reg, 2, nullptr);
        const Voxel stand = surf.get(1, 0, 0, 0);
        CHECK_EQ(voxel_id(stand), 3u);
        CHECK_EQ(voxel_emissive(stand), 1u);
        CHECK_EQ(voxel_id(surf.get(1, 4, 0, 4)), 7u);
        CHECK_EQ(voxel_id(surf.get(1, 6, 0, 6)), 7u);
    }

    {
        VoxelStore field;
        field.configure(0, 15);
        std::vector<Voxel> g(SECTION_VOXELS, 0);
        for (int z = 4; z < 6; ++z) for (int x = 4; x < 6; ++x) { g[section_index(x, 4, z)] = 5; g[section_index(x, 5, z)] = 10; }
        field.put_section(0, 0, 0, 0, Section::from_values(g.data()));
        field.build_lod(reg, 2, nullptr);
        const Voxel v = field.get(1, 2, 2, 2);
        CHECK_EQ(voxel_id(v), 10u);
        CHECK_EQ(voxel_emissive(v), 4u);
        ChunkMesher fm(reg, field);
        ChunkMesh m;
        fm.mesh(NodeKey{ 1, 0, 0, 0 }, MeshParams{}, m);
        uint32_t top = 0, lamp = 0;
        for (uint32_t t = 0; t < m.triangle_count(); ++t) { if (m.materials[t] == 8u) ++top; if (m.materials[t] == 4u) ++lamp; }
        CHECK_EQ(m.triangle_count(), 12u);
        CHECK_EQ(top, 2u);
        CHECK_EQ(lamp, 10u);
        CHECK_EQ(m.opaqueLightTriCount, 10u);
    }

    {
        VoxelStore hill;
        hill.configure(0, 15);
        std::vector<Voxel> g(SECTION_VOXELS, 0);
        for (int z = 4; z < 6; ++z) for (int x = 4; x < 6; ++x) { g[section_index(x, 4, z)] = 6; g[section_index(x, 5, z)] = 7; }
        hill.put_section(0, 0, 0, 0, Section::from_values(g.data()));
        hill.build_lod(reg, 2, nullptr);
        CHECK_EQ(voxel_id(hill.get(1, 2, 2, 2)), 7u);
        ChunkMesher hm(reg, hill);
        for (int pass = 0; pass < 2; ++pass) {
            MeshParams mp; mp.flatMaterials = pass == 1;
            ChunkMesh m;
            hm.mesh(NodeKey{ 1, 0, 0, 0 }, mp, m);
            uint32_t top = 0, rest = 0;
            const uint32_t restMat = pass == 1 ? 3u : 0u;
            for (uint32_t t = 0; t < m.triangle_count(); ++t) {
                if (m.materials[t] == 7u) ++top;
                else if (m.materials[t] == restMat) ++rest;
            }
            CHECK_EQ(m.triangle_count(), 12u);
            CHECK_EQ(top, 2u);
            CHECK_EQ(rest, 10u);
        }
    }

    {
        VoxelStore ground;
        ground.configure(0, 15);
        std::vector<Voxel> g(SECTION_VOXELS, 0);
        for (int y = 0; y < 7; ++y) for (int z = 0; z < 16; ++z) for (int x = 0; x < 16; ++x) g[section_index(x, y, z)] = 6;
        for (int z = 0; z < 16; ++z) for (int x = 0; x < 16; ++x) g[section_index(x, 7, z)] = 7;
        ground.put_section(0, 0, 0, 0, Section::from_values(g.data()));
        ground.build_lod(reg, 2, nullptr);
        const Voxel v = ground.get(1, 4, 3, 4);
        CHECK_EQ(voxel_id(v), 7u);
        CHECK((v & VOX_ALL) != 0);
        CHECK_EQ(voxel_id(ground.get(1, 4, 1, 4)), 6u);
        std::vector<uint64_t> st;
        ground.set_block(8, 7, 8, 6, reg, st);
        CHECK_EQ(voxel_id(ground.get(1, 4, 3, 4)), 6u);
        ground.set_block(8, 7, 8, 7, reg, st);
        CHECK_EQ(voxel_id(ground.get(1, 4, 3, 4)), 7u);
    }

    std::vector<uint64_t> stale;
    store.set_block(4, 8, 4, AIR_ID, reg, stale);
    CHECK_EQ(store.get(0, 4, 8, 4), 0u);
    CHECK(stale.size() >= 3);
    bool hasL0 = false, hasL2 = false;
    for (uint64_t k : stale) { const NodeKey nk = unpack_node(k); if (nk.level == 0) hasL0 = true; if (nk.level == 2) hasL2 = true; }
    CHECK(hasL0 && hasL2);
    ChunkMesh after;
    mesher.mesh(NodeKey{ 0, 0, 0, 0 }, MeshParams{}, after);
    CHECK_EQ(after.alphaTriCount, 28u);

    LodTree tree;
    tree.configure(store);
    CHECK(tree.root_level() >= 1);
    LodCut cut;
    const double camNear[3] = { 8.0, 10.0, 8.0 };
    tree.select(camNear, 48.0f, cut);
    CHECK(!cut.leaves.empty());
    bool anyL0 = false;
    for (uint64_t k : cut.leafList) if (unpack_node(k).level == 0) anyL0 = true;
    CHECK(anyL0);
    const double camFar[3] = { 100000.0, 10.0, 8.0 };
    LodCut farCut;
    tree.select(camFar, 48.0f, farCut);
    CHECK_EQ(farCut.leaves.size(), 1u);
    CHECK_EQ((int)unpack_node(farCut.leafList[0]).level, tree.root_level());
    {
        CHECK(cut.rootCount >= 1 && cut.rootCount <= cut.nodes.size());
        size_t leafNodes = 0;
        bool consistent = true;
        for (const LodCut::Node& n : cut.nodes) {
            if (n.leaf()) { ++leafNodes; if (!cut.leaves.count(n.key)) consistent = false; }
            else {
                if (!cut.interior.count(n.key)) consistent = false;
                if (n.firstChild + n.childCount > cut.nodes.size()) consistent = false;
                for (uint32_t j = 0; j < n.childCount && consistent; ++j)
                    if (parent_key(unpack_node(cut.nodes[n.firstChild + j].key)) != unpack_node(n.key)) consistent = false;
            }
        }
        CHECK(consistent);
        CHECK_EQ(leafNodes, cut.leaves.size());
    }
    {
        planet::WorkerPool pool(4);
        LodCut par;
        tree.select(camNear, 48.0f, par, &pool);
        CHECK_EQ(par.leaves.size(), cut.leaves.size());
        CHECK_EQ(par.interior.size(), cut.interior.size());
        CHECK_EQ(par.nodes.size(), cut.nodes.size());
        bool sameLeaves = true;
        for (uint64_t k : cut.leafList) if (!par.leaves.count(k)) sameLeaves = false;
        CHECK(sameLeaves);
        bool linked = true;
        size_t parLeaves = 0;
        for (const LodCut::Node& n : par.nodes) {
            if (n.leaf()) { ++parLeaves; if (!par.leaves.count(n.key)) linked = false; continue; }
            if (n.firstChild + n.childCount > par.nodes.size()) { linked = false; continue; }
            for (uint32_t j = 0; j < n.childCount; ++j)
                if (parent_key(unpack_node(par.nodes[n.firstChild + j].key)) != unpack_node(n.key)) linked = false;
        }
        CHECK(linked);
        CHECK_EQ(parLeaves, par.leaves.size());
        const uint64_t missing = cut.leafList[cut.leafList.size() / 2];
        const uint64_t parentKey = pack_node(parent_key(unpack_node(missing)));
        auto readyFn = [&](uint64_t k, void*&) { return k != missing && (k == parentKey || cut.leaves.count(k) != 0); };
        std::vector<RenderItem> serial, parallel;
        tree.render_list(cut, readyFn, serial);
        tree.render_list(par, readyFn, parallel, {}, &pool);
        std::set<uint64_t> a, b;
        for (const RenderItem& it : serial) a.insert(it.key);
        for (const RenderItem& it : parallel) b.insert(it.key);
        CHECK(a == b && a.size() == serial.size() && b.size() == parallel.size());
    }
    std::vector<RenderItem> rl;
    tree.render_list(cut, [](uint64_t, void*&) { return false; }, rl);
    CHECK(rl.empty());
    tree.render_list(cut, [](uint64_t, void*&) { return true; }, rl);
    CHECK_EQ(rl.size(), cut.leaves.size());
    {
        const uint64_t missing = cut.leafList[0];
        const NodeKey parent = parent_key(unpack_node(missing));
        const uint64_t parentKey = pack_node(parent);
        tree.render_list(cut, [&](uint64_t k, void*&) { return k != missing && (k == parentKey || cut.leaves.count(k) != 0); }, rl);
        bool hasParent = false, hasSibling = false;
        for (const RenderItem& item : rl) {
            if (item.key == parentKey) hasParent = true;
            if (item.key != parentKey && is_ancestor(parent, unpack_node(item.key))) hasSibling = true;
        }
        CHECK(hasParent);
        CHECK(!hasSibling);
        int marker = 0;
        tree.render_list(cut, [&](uint64_t, void*& chunk) { chunk = &marker; return true; }, rl);
        bool carried = !rl.empty();
        for (const RenderItem& item : rl) if (item.chunk != &marker) carried = false;
        CHECK(carried);
    }
    {
        const NodeKey k{ 7, -12345, -3, 67890 };
        CHECK(unpack_node(pack_node(k)) == k);
        CHECK(is_ancestor(NodeKey{ 1, -1, 0, 0 }, NodeKey{ 0, -2, 1, 1 }));
        CHECK(!is_ancestor(NodeKey{ 1, 0, 0, 0 }, NodeKey{ 0, -1, 0, 0 }));
        CHECK(parent_key(NodeKey{ 0, -1, -1, -1 }) == (NodeKey{ 1, -1, -1, -1 }));
    }
}


static void test_water_lod() {
    std::printf("[water height / lod boundaries / edits]\n");
    BlockRegistry reg;
    setup_test_registry(reg);
    const uint32_t up = pack_normal_oct16(Vec3f{ 0, 1, 0 });
    auto check_surface = [&](ChunkMesher& mesher, const NodeKey& key, float worldHeight) {
        for (int pass = 0; pass < 2; ++pass) {
            MeshParams params; params.flatMaterials = pass != 0;
            ChunkMesh mesh;
            mesher.mesh(key, params, mesh);
            const float originY = (float)(key.y * CHUNK_SIZE * (1 << key.level));
            uint32_t topTris = 0;
            bool flat = true, sidesClosed = true;
            for (uint32_t t = 0; t < mesh.triangle_count(); ++t) {
                if (mesh.materials[t] != 5u) continue;
                const bool top = mesh.vertices[mesh.indices[3 * t]].packedNormal == up;
                if (top) ++topTris;
                for (int k = 0; k < 3; ++k) {
                    const float y = originY + mesh.vertices[mesh.indices[3 * t + k]].py;
                    if (top && std::fabs(y - worldHeight) > 1e-4f) flat = false;
                    if (y > worldHeight + 1e-4f) sidesClosed = false;
                }
            }
            CHECK_EQ(topTris, 2u);
            CHECK(flat);
            CHECK(sidesClosed);
            CHECK_EQ(mesh.light_tri_count(), 0u);
        }
    };
    for (int sign : { 1, -1 }) {
        reg.info(8).fullOpaque = sign < 0; // Exercise both volume and imported-water occupancy flags.
        VoxelStore store;
        store.configure(-1, 0);
        std::vector<Voxel> values(SECTION_VOXELS, AIR_ID);
        for (int y = 0; y < 13; ++y)
        for (int z = 0; z < SECTION_SIZE; ++z)
        for (int x = 0; x < SECTION_SIZE; ++x) values[section_index(x, y, z)] = 8;
        const int sx0 = sign > 0 ? 0 : -8, sz0 = sign > 0 ? 0 : -4, sy = sign > 0 ? 0 : -1;
        for (int sz = sz0; sz < sz0 + 4; ++sz)
        for (int sx = sx0; sx < sx0 + 8; ++sx)
            store.put_section(0, sx, sy, sz, Section::from_values(values.data()));
        store.build_lod(reg, MAX_LOD_LEVELS, nullptr);
        ChunkMesher mesher(reg, store);
        const float height = sign > 0 ? 13.0f : -3.0f;
        for (int level = 0; level < MAX_LOD_LEVELS; ++level) {
            const NodeKey key{ (uint8_t)level, sign > 0 ? 0 : -1, sign > 0 ? 0 : -1, sign > 0 ? 0 : -1 };
            check_surface(mesher, key, height);
            const int vy = floor_shift((int)height - 1, level);
            const Voxel v = store.get(level, sign > 0 ? 0 : -1, vy, sign > 0 ? 0 : -1);
            CHECK_EQ(voxel_id(v), 8u);
            CHECK_EQ(voxel_water_height(v, level), (uint32_t)((int)height - vy * (1 << level)));
            CHECK_EQ(voxel_emissive(v, reg.info(voxel_id(v)).water != 0), 0u);
        }
        // These two meshes meet at x=64 (or x=-64) with different LODs.
        check_surface(mesher, NodeKey{ 0, sign > 0 ? 1 : -2, sign > 0 ? 0 : -1, sign > 0 ? 0 : -1 }, height);
        check_surface(mesher, NodeKey{ 1, sign > 0 ? 1 : -2, sign > 0 ? 0 : -1, sign > 0 ? 0 : -1 }, height);
    }
    reg.info(8).fullOpaque = false;
    {
        VoxelStore store;
        store.configure(0, 0);
        std::vector<Voxel> values(SECTION_VOXELS, AIR_ID);
        const BlockId otherWater = reg.intern("minecraft:water", { { "level", "1" } });
        reg.info(otherWater) = reg.info(8);
        // Adjacent coarse voxels have tops at y=5 and y=6 within the same slice.
        for (int z = 0; z < 4; ++z) for (int x = 0; x < 4; ++x)
        for (int y = 0; y < (x < 2 ? 5 : 6); ++y) values[section_index(x, y, z)] = x < 2 ? 8 : otherWater;
        store.put_section(0, 0, 0, 0, Section::from_values(values.data()));
        store.build_lod(reg, 3, nullptr);
        ChunkMesher mesher(reg, store);
        ChunkMesh mesh;
        mesher.mesh(NodeKey{ 1, 0, 0, 0 }, MeshParams{}, mesh);
        uint32_t low = 0, high = 0, step = 0;
        const uint32_t west = pack_normal_oct16(Vec3f{ -1, 0, 0 });
        bool coplanar = true;
        for (uint32_t t = 0; t < mesh.triangle_count(); ++t) {
            const MeshVertex& a = mesh.vertices[mesh.indices[3 * t]];
            if (mesh.materials[t] == 5u && a.packedNormal == west && a.px == 2.0f) {
                ++step;
                for (int k = 0; k < 3; ++k) {
                    const float y = mesh.vertices[mesh.indices[3 * t + k]].py;
                    CHECK(y == 5.0f || y == 6.0f);
                }
            }
            if (mesh.materials[t] != 5u || a.packedNormal != up) continue;
            if (a.py == 5.0f) ++low;
            if (a.py == 6.0f) ++high;
            for (int k = 1; k < 3; ++k) if (mesh.vertices[mesh.indices[3 * t + k]].py != a.py) coplanar = false;
        }
        CHECK_EQ(low, 2u); CHECK_EQ(high, 2u); CHECK(coplanar);
        CHECK_EQ(step, 2u);
        std::vector<uint64_t> stale;
        for (int z = 0; z < 4; ++z) for (int x = 0; x < 2; ++x) store.set_block(x, 5, z, 8, reg, stale);
        check_surface(mesher, NodeKey{ 1, 0, 0, 0 }, 6.0f);
        for (int z = 0; z < 4; ++z) for (int x = 0; x < 4; ++x) store.set_block(x, 5, z, AIR_ID, reg, stale);
        check_surface(mesher, NodeKey{ 1, 0, 0, 0 }, 5.0f);
        check_surface(mesher, NodeKey{ 2, 0, 0, 0 }, 5.0f);
    }
    {
        VoxelStore store;
        Voxel children[8];
        for (Voxel& v : children) v = make_voxel(8, true, false, true, 13u);
        children[0] = make_voxel(5, true, true, true, 4u);
        const Voxel parent = store.downsample(reg, 4, children, 1u, 1u);
        CHECK_EQ(voxel_id(parent), 5u);
        CHECK_EQ(voxel_emissive(parent), 4u);
    }
    {
        MemoryResources res;
        BlockRegistry actual;
        const BlockId water = actual.intern("minecraft:water", {});
        const BlockId bubbles = actual.intern("minecraft:bubble_column", {});
        const BlockId lava = actual.intern("minecraft:lava", {});
        MaterialSoA materials;
        std::vector<std::string> names;
        std::vector<TextureData> textures;
        MaterialBuildStats stats;
        CHECK(MaterialBuilder().build(actual, res, materials, names, textures, 0, stats));
        CHECK(actual.info(water).water && actual.info(bubbles).water);
        CHECK(!actual.info(lava).water && actual.info(lava).emissive);
        CHECK_EQ(voxel_emissive(make_voxel(lava, true, true, true, VOX_EMIT_MAX)), VOX_EMIT_MAX);
    }
}

static void test_foliage_and_glass_materials() {
    std::printf("[foliage subsurface / thin plants / clear glass / shared textures / lod]\n");
    MemoryResources res;
    add_vanilla_like_models(res);
    const unsigned char leafPng[] = { 137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 2, 0, 0, 0, 2, 8, 6, 0, 0, 0, 114, 182, 13, 36, 0, 0, 0, 19, 73, 68, 65, 84, 120, 156, 99, 112, 216, 162, 241, 31, 136, 25, 24, 160, 140, 255, 0, 62, 134, 7, 110, 145, 51, 211, 154, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130 };
    const unsigned char glassPng[] = { 137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 2, 0, 0, 0, 2, 8, 6, 0, 0, 0, 114, 182, 13, 36, 0, 0, 0, 17, 73, 68, 65, 84, 120, 156, 99, 80, 208, 80, 248, 15, 194, 12, 48, 6, 0, 45, 100, 5, 157, 156, 219, 89, 56, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130 };
    const unsigned char flowerPng[] = { 137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 2, 0, 0, 0, 2, 8, 6, 0, 0, 0, 114, 182, 13, 36, 0, 0, 0, 21, 73, 68, 65, 84, 120, 156, 99, 84, 104, 248, 240, 159, 129, 129, 129, 129, 9, 68, 128, 48, 0, 38, 88, 2, 147, 161, 96, 194, 88, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130 };
    res.add("assets/minecraft/textures/block/leaf_fixture.png", std::string((const char*)leafPng, sizeof(leafPng)));
    res.add("assets/minecraft/textures/block/glass_fixture.png", std::string((const char*)glassPng, sizeof(glassPng)));
    res.add("assets/minecraft/textures/block/flower_fixture.png", std::string((const char*)flowerPng, sizeof(flowerPng)));
    res.add("assets/minecraft/textures/block/flower_pot.png", std::string((const char*)leafPng, sizeof(leafPng)));
    res.add("assets/minecraft/textures/block/dirt.png", std::string((const char*)leafPng, sizeof(leafPng)));
    for (const char* block : { "stone", "oak_leaves", "glass", "glass_pane", "red_stained_glass", "tinted_glass" }) {
        const std::string name = block;
        const bool glass = name.find("glass") != std::string::npos;
        res.add("assets/minecraft/blockstates/" + name + ".json",
                "{\"variants\":{\"\":{\"model\":\"minecraft:block/" + name + "\"}}}");
        res.add("assets/minecraft/models/block/" + name + ".json",
                "{\"parent\":\"block/cube_all\",\"textures\":{\"all\":\"block/" +
                std::string(glass ? "glass_fixture" : "leaf_fixture") + "\"}}");
    }
    const char* thinPlants[] = {
        "short_grass", "tall_grass", "fern", "large_fern", "vine", "cave_vines_plant",
        "sugar_cane", "bamboo", "oak_sapling", "poppy", "dandelion", "wither_rose",
        "red_tulip", "lily_of_the_valley", "spore_blossom", "closed_eyeblossom", "pink_petals",
        "wheat", "carrots", "potatoes", "beetroots", "attached_pumpkin_stem", "pitcher_crop",
        "crimson_roots", "warped_fungus", "brown_mushroom", "dead_bush", "moss_carpet",
        "glow_lichen", "lily_pad", "big_dripleaf", "small_dripleaf", "big_dripleaf_stem",
        "tube_coral_fan", "sea_pickle", "seagrass", "kelp_plant",
        "modded_flower"
    };
    const char* ordinaryBlocks[] = { "grass_block", "moss_block", "red_mushroom_block", "bamboo_block", "torch", "cobweb", "rail" };
    for (const char* block : thinPlants) {
        const std::string name = block;
        res.add("assets/minecraft/blockstates/" + name + ".json",
                "{\"variants\":{\"\":{\"model\":\"minecraft:block/" + name + "\"}}}");
        res.add("assets/minecraft/models/block/" + name + ".json",
                "{\"parent\":\"block/cross\",\"textures\":{\"cross\":\"block/" +
                std::string(name == "poppy" ? "flower_fixture" : "leaf_fixture") + "\"}}");
    }
    for (const char* block : ordinaryBlocks) {
        const std::string name = block;
        res.add("assets/minecraft/blockstates/" + name + ".json",
                "{\"variants\":{\"\":{\"model\":\"minecraft:block/" + name + "\"}}}");
        res.add("assets/minecraft/models/block/" + name + ".json",
                "{\"parent\":\"block/cube_all\",\"textures\":{\"all\":\"block/leaf_fixture\"}}");
    }
    res.add("assets/minecraft/blockstates/potted_poppy.json", "{\"variants\":{\"\":{\"model\":\"block/potted_poppy\"}}}");
    res.add("assets/minecraft/models/block/potted_poppy.json",
            "{\"textures\":{\"pot\":\"block/flower_pot\",\"soil\":\"block/dirt\",\"plant\":\"block/flower_fixture\"},"
            "\"elements\":[{\"from\":[5,0,5],\"to\":[11,6,11],\"faces\":{\"north\":{\"texture\":\"#pot\"},\"up\":{\"texture\":\"#soil\"}}},"
            "{\"from\":[1,6,8],\"to\":[15,16,8],\"faces\":{\"north\":{\"texture\":\"#plant\"},\"south\":{\"texture\":\"#plant\"}}}]}");
    for (bool opaqueLeaves : { false, true }) {
        BlockRegistry reg;
        const BlockId stone = reg.intern("minecraft:stone", {});
        const BlockId leaves = reg.intern("minecraft:oak_leaves", {});
        // Load tinted glass first to exercise clear/tinted material cache separation.
        const BlockId stained = reg.intern("minecraft:red_stained_glass", {});
        const BlockId tinted = reg.intern("minecraft:tinted_glass", {});
        const BlockId glass = reg.intern("minecraft:glass", {});
        const BlockId pane = reg.intern("minecraft:glass_pane", {});
        std::vector<BlockId> plants, ordinary;
        for (const char* name : ordinaryBlocks) ordinary.push_back(reg.intern(std::string("minecraft:") + name, {}));
        for (const char* name : thinPlants) plants.push_back(reg.intern(std::string("minecraft:") + name, {}));
        const BlockId potted = reg.intern("minecraft:potted_poppy", {});
        MaterialSoA materials;
        std::vector<std::string> names;
        std::vector<TextureData> textures;
        MaterialBuildStats stats;
        MaterialBuilder builder;
        builder.opaqueLeaves = opaqueLeaves;
        CHECK(builder.build(reg, res, materials, names, textures, 0, stats));
        CHECK_EQ(stats.statesMissing, 0u);
        CHECK_EQ(stats.texturesMissing, 0u);
        for (BlockId id : plants) {
            const BlockInfo& plant = reg.info(id);
            CHECK(plant.hasQuads && plant.quadCount > 0);
            for (uint32_t q = plant.quadBegin; q < plant.quadBegin + plant.quadCount; ++q) {
                const Material m = materials.Get(reg.quads[q].material);
                CHECK(m.sssEnable && !m.thinGlass && m.sssWeight > 0.0f && m.sssRadius > 0.0f);
                if (reg.desc(id).path() == "poppy") CHECK(m.sssAlbedo.z > m.sssAlbedo.y && m.sssAlbedo.z > m.sssAlbedo.x);
            }
            for (int f = 0; f < 6; ++f) for (uint16_t material : { plant.lodFaceMaterial[f], plant.flatFaceMaterial[f] })
                if (material != NO_MATERIAL) CHECK(materials.Get(material).sssEnable);
        }
        for (BlockId id : ordinary) for (uint16_t material : reg.info(id).faceMaterial)
            CHECK(!materials.Get(material).sssEnable);
        const BlockInfo& pot = reg.info(potted);
        uint32_t potSurfaces = 0, plantSurfaces = 0;
        for (uint32_t q = pot.quadBegin; q < pot.quadBegin + pot.quadCount; ++q) {
            const Material m = materials.Get(reg.quads[q].material);
            const bool plant = m.sssEnable != 0;
            if (plant) { ++plantSurfaces; CHECK(m.sssAlbedo.z > m.sssAlbedo.y); }
            else ++potSurfaces;
        }
        CHECK(plantSurfaces > 0 && potSurfaces > 0);
        const BlockInfo& foliage = reg.info(leaves);
        for (int f = 0; f < 6; ++f) {
            for (uint16_t id : { foliage.faceMaterial[f], foliage.lodFaceMaterial[f], foliage.flatFaceMaterial[f] }) {
                const Material m = materials.Get(id);
                CHECK(m.sssEnable && !m.thinGlass);
                CHECK(m.sssRadius >= 0.5f && m.sssRadius <= 2.0f);
                CHECK(m.sssWeight > 0.4f && m.sssWeight < 0.9f);
                CHECK(m.sssAlbedo.y > m.sssAlbedo.x && m.sssAlbedo.y > m.sssAlbedo.z);
                CHECK(m.sssAlbedo.x >= 0.5f && m.sssAlbedo.y <= 1.0f && m.sssAlbedo.z >= 0.5f);
                uint32_t packed[MaterialPack::kMatPackedU32];
                MaterialPack::PackOne(m, packed);
                CHECK((packed[6] & (1u << 17)) != 0u);
            }
            CHECK(!materials.Get(reg.info(stone).faceMaterial[f]).sssEnable);
            CHECK(!materials.Get(reg.info(stone).flatFaceMaterial[f]).sssEnable);
            CHECK_EQ(reg.materialAlpha[foliage.faceMaterial[f]], opaqueLeaves ? 0u : 1u);
            for (BlockId id : { glass, pane }) {
                const Material m = materials.Get(reg.info(id).faceMaterial[f]);
                CHECK(m.thinGlass && !m.sssEnable && m.Kd.w == 0.0f);
                CHECK(m.Tf.x == 1.0f && m.Tf.y == 1.0f && m.Tf.z == 1.0f);
                CHECK_EQ(reg.materialAlpha[reg.info(id).faceMaterial[f]], 1u);
            }
            for (BlockId id : { stained, tinted }) {
                const Material m = materials.Get(reg.info(id).faceMaterial[f]);
                CHECK(m.thinGlass && m.Tf.x < 0.5f && m.Tf.y < 0.5f && m.Tf.z < 0.5f);
            }
        }
    }
}

static void test_imported_emission() {
    std::printf("[imported emission / material colour / meshed lights / lod]\n");
    MemoryResources res;
    add_vanilla_like_models(res);
    const unsigned char png[] = { 137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 2, 0, 0, 0, 2, 8, 6, 0, 0, 0, 114, 182, 13, 36, 0, 0, 0, 21, 73, 68, 65, 84, 120, 156, 99, 84, 104, 248, 240, 159, 129, 129, 129, 129, 9, 68, 128, 48, 0, 38, 88, 2, 147, 161, 96, 194, 88, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130 };
    res.add("assets/minecraft/textures/block/stone.png", std::string((const char*)png, sizeof(png)));
    BlockRegistry reg;
    const BlockId plain = reg.intern("minecraft:stone", {});
    const BlockId weak = reg.intern("minecraft:stone", {{"nightcity_emission", "2"}});
    const BlockId lit = reg.intern("minecraft:stone", {{"nightcity_emission", "6"}});
    const BlockId bright = reg.intern("minecraft:stone", {{"nightcity_emission", "12"}});
    const BlockId invalid = reg.intern("minecraft:stone", {{"nightcity_emission", "bad"}});
    MaterialSoA materials;
    std::vector<std::string> names;
    std::vector<TextureData> textures;
    MaterialBuildStats stats;
    std::string err;
    CHECK(MaterialBuilder().build(reg, res, materials, names, textures, 0, stats, &err));
    CHECK_EQ(stats.statesMissing, 0u);
    CHECK_EQ(stats.texturesMissing, 0u);
    CHECK(!reg.info(plain).emissive && !reg.info(invalid).emissive);
    const auto emission = [&](BlockId id) { return reg.materialEmission[reg.info(id).faceMaterial[0]]; };
    const Vec3f e2 = emission(weak), e6 = emission(lit), e12 = emission(bright);
    CHECK(e2.x > 0 && e2.z > e2.y && e2.y > e2.x);
    CHECK(std::abs(e6.z - 3.0f * e2.z) < 0.0001f);
    CHECK(std::abs(e12.z - 2.0f * e6.z) < 0.0001f);
    CHECK_EQ(emission(plain).z, 0.0f);
    for (const BlockId id : {plain, weak, lit, bright, invalid}) {
        CHECK(reg.info(id).isCube);
        const bool expectedLit = id == weak || id == lit || id == bright;
        VoxelStore store;
        store.configure(0, 15);
        std::vector<Voxel> values(SECTION_VOXELS, 0);
        for (int y = 4; y < 6; ++y) for (int z = 4; z < 6; ++z) for (int x = 4; x < 6; ++x)
            values[section_index(x, y, z)] = id;
        store.put_section(0, 0, 0, 0, Section::from_values(values.data()));
        store.build_lod(reg, 2, nullptr);
        ChunkMesher mesher(reg, store);
        for (int level = 0; level <= 1; ++level) {
            ChunkMesh mesh;
            mesher.mesh(NodeKey{(uint8_t)level, 0, 0, 0}, MeshParams{}, mesh);
            CHECK(mesh.triangle_count() > 0);
            CHECK_EQ(mesh.light_tri_count(), expectedLit ? mesh.triangle_count() : 0u);
        }
    }
}

static void test_real_world(const std::string& worldDir, bool full, const std::vector<std::string>& packs) {
    std::printf("[world] %s%s\n", worldDir.c_str(), full ? " (full)" : " (spawn window)");
    planet::WorkerPool pool;
    World world;
    WorldLoadConfig cfg;
    cfg.worldDir = worldDir;
    if (!full) {
        LevelInfo li;
        read_level_dat((std::filesystem::path(worldDir) / "level.dat").string(), li);
        const int cx = floor_shift(li.spawnX, 4), cz = floor_shift(li.spawnZ, 4);
        cfg.chunkMinX = cx - 12; cfg.chunkMaxX = cx + 11;
        cfg.chunkMinZ = cz - 12; cfg.chunkMaxZ = cz + 11;
    }
    std::string err;
    const bool ok = world.load(cfg, &pool, &err);
    CHECK(ok);
    if (!ok) { std::printf("  %s\n", err.c_str()); return; }
    BlockRegistry& reg = world.registry();
    bool builtMaterials = false;
    MaterialSoA materials;
    std::vector<std::string> names;
    std::vector<TextureData> textures;
    if (!packs.empty()) {
        std::vector<std::unique_ptr<ZipArchive>> archives;
        ResourceStack stack;
        for (const std::string& p : packs) {
            auto z = std::make_unique<ZipArchive>();
            if (z->open(p, &err)) { stack.push(z.get()); archives.push_back(std::move(z)); }
            else std::printf("  pack skipped: %s\n", err.c_str());
        }
        const std::string jar = find_minecraft_jar();
        auto z = std::make_unique<ZipArchive>();
        if (!jar.empty() && z->open(jar, &err)) { stack.push(z.get()); archives.push_back(std::move(z)); }
        else std::printf("  no vanilla jar (%s)\n", err.c_str());
        MaterialBuildStats ms;
        builtMaterials = MaterialBuilder().build(reg, stack, materials, names, textures, 0, ms, &err);
        CHECK(builtMaterials);
        if (builtMaterials) {
            std::map<std::string, uint32_t> lit;
            for (size_t i = 1; i < reg.count(); ++i)
                if (reg.info((BlockId)i).emissive) ++lit[std::string(reg.desc((BlockId)i).path())];
            std::printf("  emissive blocks (%zu kinds, states in brackets):", lit.size());
            for (const auto& kv : lit) std::printf(" %s[%u]", kv.first.c_str(), kv.second);
            std::printf("\n");
        } else {
            std::printf("  materials failed: %s\n", err.c_str());
        }
    }
    if (!builtMaterials) {
        reg.materialAlpha.assign(1, 0);
        for (size_t i = 1; i < reg.count(); ++i) {
            BlockInfo& bi = reg.info((BlockId)i);
            bi.isCube = true; bi.fullOpaque = true; bi.sig = Significance::Full;
            for (int f = 0; f < 6; ++f) bi.faceMaterial[f] = bi.lodFaceMaterial[f] = bi.flatFaceMaterial[f] = 0;
        }
    }
    world.build_lod(&pool);
    const auto& st = world.stats();
    std::printf("  chunks %u sections %u states %u load %.2fs lod %.2fs store %.1f MB\n",
                st.chunks, st.sections, st.blockStates, st.loadSeconds, st.lodSeconds, st.storeBytes / 1048576.0);
    LodCut cut;
    const double cam[3] = { (double)world.level().spawnX, (double)world.level().spawnY + 40.0, (double)world.level().spawnZ };
    world.lod_tree().select(cam, 512.0f, cut);
    std::printf("  cut at detail distance 512: %zu leaves (instances), %zu interior\n", cut.leaves.size(), cut.interior.size());
    struct LevelStat { uint32_t leaves = 0, seen = 0, meshed = 0; uint64_t quads = 0, tris = 0, alpha = 0, lit = 0; double ms = 0; };
    LevelStat lv[MAX_LOD_LEVELS];
    for (uint64_t k : cut.leafList) ++lv[unpack_node(k).level].leaves;
    ChunkMesher mesher(reg, world.store());
    ChunkMesh mesh;
    uint64_t tris = 0;
    std::map<uint32_t, uint64_t> alphaByMat[2];
    for (uint64_t k : cut.leafList) {
        const NodeKey key = unpack_node(k);
        LevelStat& L = lv[key.level];
        if ((L.seen++ % std::max(1u, L.leaves / 120u)) != 0u || L.meshed >= 120) continue;
        const auto t0 = std::chrono::steady_clock::now();
        mesher.mesh(key, MeshParams{}, mesh);
        L.ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        L.tris += mesh.triangle_count(); L.quads += mesh.quadCount;
        L.alpha += mesh.alphaTriCount; L.lit += mesh.light_tri_count();
        tris += mesh.triangle_count();
        if (key.level < 2)
            for (uint32_t t = mesh.opaqueTriCount; t < mesh.triangle_count(); ++t) ++alphaByMat[key.level][mesh.materials[t]];
        ++L.meshed;
    }
    std::printf("  level  leaves  sampled   tris/chunk   alpha%%   lit%%   ms/chunk   est. tris in cut\n");
    double estTotal = 0.0, estAlpha = 0.0;
    for (int L = 0; L < MAX_LOD_LEVELS; ++L) {
        if (!lv[L].leaves) continue;
        const LevelStat& x = lv[L];
        const double perChunk = x.meshed ? (double)x.tris / x.meshed : 0.0;
        const double alphaPct = x.tris ? 100.0 * (double)x.alpha / (double)x.tris : 0.0;
        const double litPct   = x.tris ? 100.0 * (double)x.lit / (double)x.tris : 0.0;
        estTotal += perChunk * x.leaves;
        estAlpha += perChunk * x.leaves * alphaPct / 100.0;
        std::printf("  %5d  %6u  %7u  %11.0f  %6.1f  %5.1f  %9.2f  %14.1fM\n",
                    L, x.leaves, x.meshed, perChunk, alphaPct, litPct, x.meshed ? x.ms / x.meshed : 0.0, perChunk * x.leaves / 1e6);
    }
    std::printf("  estimated cut: %.1fM triangles, %.1fM alpha-tested (%.0f%%)\n", estTotal / 1e6, estAlpha / 1e6, estTotal ? 100.0 * estAlpha / estTotal : 0.0);
    if (builtMaterials) {
        std::vector<OmmBakeTri> omm;
        enumerate_omm_triangles(reg, omm);
        double bytes = 0.0;
        uint32_t perLevel[13] = {};
        for (const OmmBakeTri& t : omm) { bytes += (double)(1ull << (2 * t.level)) / 4.0; ++perLevel[std::min<int>(t.level, 12)]; }
        std::printf("  opacity micromaps: %zu face triangles to bake, %.1f MB before deduplication; subdivision levels:", omm.size(), bytes / 1048576.0);
        for (int L = 0; L < 13; ++L) if (perLevel[L]) std::printf(" %d:%u", L, perLevel[L]);
        std::printf("\n");
    }
    if (builtMaterials) {
        auto cutoutFraction = [&](uint32_t mat) -> double {
            const int tex = mat < materials.albedoTexID.size() ? materials.albedoTexID[mat] : -1;
            if (tex < 0 || (size_t)tex >= textures.size()) return -1.0;
            const DirectX::Image* img = textures[(size_t)tex].image.GetImage(0, 0, 0);
            if (!img) return -1.0;
            size_t n = 0, cut = 0;
            for (size_t y = 0; y < img->height; ++y)
                for (size_t x = 0; x < img->width; ++x) { ++n; if (img->pixels[y * img->rowPitch + x * 4 + 3] < 128) ++cut; }
            return n ? (double)cut / (double)n : -1.0;
        };
        for (int L = 0; L < 2; ++L) {
            uint64_t total = 0;
            for (const auto& kv : alphaByMat[L]) total += kv.second;
            std::vector<std::pair<uint64_t, uint32_t>> top;
            for (const auto& kv : alphaByMat[L]) top.push_back({ kv.second, kv.first });
            std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
            std::printf("  level %d alpha triangles by material (%zu materials, %llu triangles in the sample):\n", L, alphaByMat[L].size(), (unsigned long long)total);
            for (size_t i = 0; i < top.size() && i < 14; ++i) {
                const uint32_t mat = top[i].second;
                const double cf = cutoutFraction(mat);
                std::printf("    %5.1f%%  %-48s  cutout texels %s%.1f%%\n", total ? 100.0 * (double)top[i].first / (double)total : 0.0,
                            mat < names.size() ? names[mat].c_str() : "?", cf < 0 ? "n/a " : "", cf < 0 ? 0.0 : 100.0 * cf);
            }
        }
    }
    CHECK(tris > 0);
}

int main(int argc, char** argv) {
    const std::filesystem::path outDir = std::filesystem::temp_directory_path() / "royaltracer-minecraft-tests";
    test_nbt();
    test_block_states();
    test_section();
    test_registry();
    test_anvil(outDir);
    test_zip();
    test_models();
    test_bake();
    test_placement();
    test_store_and_mesher();
    test_water_lod();
    test_foliage_and_glass_materials();
    test_imported_emission();
    if (argc > 1) {
        std::vector<std::string> packs;
        for (int i = 3; i < argc; ++i) packs.push_back(argv[i]);
        test_real_world(argv[1], argc > 2 && std::strcmp(argv[2], "full") == 0, packs);
    }
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
