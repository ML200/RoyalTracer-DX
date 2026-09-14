#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/pass.h"

namespace pb {

struct PlanetState;
class  ParamRegistry;

//====================================
//Pipeline orchestrates ordered passes plus the on-disk cache. Each pass has
//a PassEntry mirroring its current state so the viewer can render status
//badges and timing without poking the pass list directly.
//
//Status flow: Dirty -> Running -> Clean (or Error on throw). last_source
//tracks whether the last successful run loaded from cache or recomputed.
//====================================

enum class PassStatus : std::uint8_t {
    Dirty,
    Running,
    Clean,
    Error,
};

enum class LastSource : std::uint8_t {
    None,
    Cache,
    Computed,
};

struct PassEntry {
    std::string  name;
    std::string  param_prefix;     //matches Pass::param_prefix() for this entry
    PassStatus   status      = PassStatus::Dirty;
    LastSource   last_source = LastSource::None;
    double       last_ms     = 0.0;
    std::uint64_t last_key   = 0;
    std::string  last_error;
};

class Pipeline {
public:
    explicit Pipeline(std::filesystem::path cache_dir);

    void add_pass(std::unique_ptr<Pass> pass);
    void declare_all(ParamRegistry& reg) const;

    //Subscribes a callback so any registry set_* call on a path owned by a
    //pass flips that pass (and any downstream pass) to Dirty.
    void wire_dirty_tracking(ParamRegistry& reg);

    std::span<const PassEntry> entries() const { return entries_; }

    //Looks up by Pass::name(). Returns nullptr if not present.
    const Pass* find_pass(std::string_view name) const;

    void invalidate_pass(std::string_view name);
    void invalidate_from(std::string_view name);

    //Executes every Dirty pass in order, consulting the cache. Updates
    //entries() in place. Safe to call repeatedly: Clean passes are skipped.
    void run(PlanetState& state, const ParamRegistry& reg, ProgressSink& progress);

private:
    std::uint64_t compute_key(const Pass& p, const PlanetState& state,
                              const ParamRegistry& reg) const;

    bool try_load_writes(const Pass& p, std::uint64_t key, PlanetState& state) const;
    bool save_writes    (const Pass& p, std::uint64_t key, const PlanetState& state) const;

    void cascade_dirty(std::size_t from_index);

    std::filesystem::path              cache_dir_;
    std::vector<std::unique_ptr<Pass>> passes_;
    std::vector<PassEntry>             entries_;
};

}
