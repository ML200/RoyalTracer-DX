#include "mc_world.h"
#include "nbt.h"
#include "../planet/worker_pool.h"
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <mutex>

namespace mc {

namespace fs = std::filesystem;

namespace {
struct RegionResult {
    std::vector<DecodedChunk> chunks;
    uint32_t chunksFailed = 0;
    bool     ok = false;
};
}

// Loads regions in parallel, then updates bounds and occupancy metadata.
bool World::load(const WorldLoadConfig& cfg, planet::WorkerPool* pool, std::string* err) {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    m_stats = WorldLoadStats{};

    const fs::path worldDir(cfg.worldDir);
    std::string lerr;
    if (!read_level_dat((worldDir / "level.dat").string(), m_level, &lerr)) {
        if (cfg.logProgress) std::printf("[mc] level.dat: %s (continuing without spawn/name)\n", lerr.c_str());
    }

    std::vector<std::string> regionPaths;
    std::error_code ec;
    const fs::path regionDir = worldDir / "region";
    for (const auto& entry : fs::directory_iterator(regionDir, ec)) {
        if (!entry.is_regular_file()) continue;
        int32_t rx, rz;
        const std::string name = entry.path().filename().string();
        if (!RegionFile::parse_name(name, rx, rz)) continue;
        if (cfg.chunkMinX <= cfg.chunkMaxX) {
            if (rx * 32 + 31 < cfg.chunkMinX || rx * 32 > cfg.chunkMaxX) continue;
            if (rz * 32 + 31 < cfg.chunkMinZ || rz * 32 > cfg.chunkMaxZ) continue;
        }
        regionPaths.push_back(entry.path().string());
    }
    if (regionPaths.empty()) {
        if (err) *err = "no region files under " + regionDir.string();
        return false;
    }
    std::sort(regionPaths.begin(), regionPaths.end());
    m_stats.regionFiles = (uint32_t)regionPaths.size();

    std::vector<RegionResult> results(regionPaths.size());
    std::atomic<uint32_t> done{ 0 };
    auto job = [&](uint32_t i) {
        RegionResult& r = results[i];
        RegionFile rf;
        std::string e;
        if (!rf.open(regionPaths[i], &e)) {
            if (cfg.logProgress) std::printf("[mc] %s\n", e.c_str());
            return;
        }
        r.ok = true;
        InternCache cache(m_registry);
        std::vector<uint8_t> nbt;
        NbtValue root;
        for (int lz = 0; lz < 32; ++lz)
        for (int lx = 0; lx < 32; ++lx) {
            if (!rf.chunk_present(lx, lz)) continue;
            const int32_t cx = rf.rx * 32 + lx, cz = rf.rz * 32 + lz;
            if (cfg.chunkMinX <= cfg.chunkMaxX &&
                (cx < cfg.chunkMinX || cx > cfg.chunkMaxX || cz < cfg.chunkMinZ || cz > cfg.chunkMaxZ)) continue;
            if (!rf.read_chunk(lx, lz, nbt, &e)) { ++r.chunksFailed; continue; }
            if (!nbt_parse(nbt.data(), nbt.size(), root, &e)) { ++r.chunksFailed; continue; }
            DecodedChunk dc;
            if (!decode_chunk(root, cache, dc, &e)) { ++r.chunksFailed; continue; }
            if (dc.sections.empty()) continue;
            r.chunks.push_back(std::move(dc));
        }
        const uint32_t n = ++done;
        if (cfg.logProgress && (n % 32 == 0 || n == regionPaths.size()))
            std::printf("[mc] regions %u/%u\n", n, (uint32_t)regionPaths.size());
    };
    if (pool) pool->parallel_for((uint32_t)regionPaths.size(), job);
    else for (uint32_t i = 0; i < (uint32_t)regionPaths.size(); ++i) job(i);

    int32_t minSy = INT32_MAX, maxSy = INT32_MIN;
    int32_t minCx = INT32_MAX, maxCx = INT32_MIN, minCz = INT32_MAX, maxCz = INT32_MIN;
    uint32_t chunkCount = 0, sectionCount = 0;
    for (const RegionResult& r : results) {
        if (!r.ok) { ++m_stats.regionsFailed; continue; }
        m_stats.chunksFailed += r.chunksFailed;
        for (const DecodedChunk& c : r.chunks) {
            ++chunkCount;
            minCx = std::min(minCx, c.cx); maxCx = std::max(maxCx, c.cx);
            minCz = std::min(minCz, c.cz); maxCz = std::max(maxCz, c.cz);
            for (const DecodedSection& s : c.sections) {
                ++sectionCount;
                minSy = std::min(minSy, s.y); maxSy = std::max(maxSy, s.y);
            }
        }
    }
    if (chunkCount == 0) {
        if (err) *err = "no chunk with block data was decoded";
        return false;
    }
    minSy = std::min(minSy, 0);
    maxSy = std::max(maxSy, 15);
    m_store.configure(minSy, maxSy);
    for (RegionResult& r : results) {
        for (DecodedChunk& c : r.chunks)
            for (DecodedSection& s : c.sections)
                m_store.put_section(0, c.cx, s.y, c.cz, std::move(s.blocks));
        r.chunks.clear();
        r.chunks.shrink_to_fit();
    }

    m_stats.chunks      = chunkCount;
    m_stats.sections    = sectionCount;
    m_stats.blockStates = (uint32_t)m_registry.count();
    m_stats.minChunkX = minCx; m_stats.maxChunkX = maxCx;
    m_stats.minChunkZ = minCz; m_stats.maxChunkZ = maxCz;
    m_stats.minSectionY = minSy; m_stats.maxSectionY = maxSy;
    m_stats.loadSeconds = std::chrono::duration<double>(clock::now() - t0).count();
    m_stats.storeBytes  = m_store.stats().bytes;

    {
        const int extent = std::max({ maxCx - minCx + 1, maxCz - minCz + 1, (maxSy - minSy + 1) }) ;
        int level = 0;
        while ((CHUNK_SECTIONS << level) < extent && level < MAX_LOD_LEVELS - 1) ++level;
        m_lodLevels = level + 1;
    }
    if (cfg.logProgress)
        std::printf("[mc] world '%s': %u regions, %u chunks, %u sections, %u block states, %.1f MB store, %.2f s\n",
                    m_level.name.c_str(), m_stats.regionFiles, chunkCount, sectionCount,
                    m_stats.blockStates, (double)m_stats.storeBytes / (1024.0 * 1024.0), m_stats.loadSeconds);
    return true;
}

// Builds derived voxel levels before configuring the render tree.
void World::build_lod(planet::WorkerPool* pool, int decorMaxLevel) {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    m_store.build_lod(m_registry, m_lodLevels, pool, decorMaxLevel);
    m_tree.configure(m_store);
    m_stats.lodSeconds = std::chrono::duration<double>(clock::now() - t0).count();
    m_stats.storeBytes = m_store.stats().bytes;
    std::printf("[mc] LOD pyramid: %d levels, root level %d, %zu roots, %.1f MB total, %.2f s\n",
                m_lodLevels, m_tree.root_level(), m_tree.roots().size(),
                (double)m_stats.storeBytes / (1024.0 * 1024.0), m_stats.lodSeconds);
}

std::string find_minecraft_jar() {
    const char* appdata = std::getenv("APPDATA");
    if (!appdata) return {};
    const fs::path versions = fs::path(appdata) / ".minecraft" / "versions";
    std::error_code ec;
    std::string best;
    fs::file_time_type bestTime{};
    for (const auto& dir : fs::directory_iterator(versions, ec)) {
        if (!dir.is_directory()) continue;
        const fs::path jar = dir.path() / (dir.path().filename().string() + ".jar");
        if (!fs::exists(jar, ec)) continue;
        const std::string name = dir.path().filename().string();
        bool release = true;
        for (char c : name) if (!(std::isdigit((unsigned char)c) || c == '.')) { release = false; break; }
        const auto t = fs::last_write_time(jar, ec);
        if (best.empty() || (release && (t > bestTime))) {
            if (best.empty() || release) { best = jar.string(); bestTime = t; }
        }
    }
    return best;
}

}
