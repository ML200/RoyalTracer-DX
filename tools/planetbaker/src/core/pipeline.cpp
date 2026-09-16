#include "core/pipeline.h"
#include "core/cache.h"
#include "core/cubed_sphere.h"
#include "core/field.h"
#include "core/log.h"
#include "core/param_registry.h"
#include "core/planet_state.h"

#include <cuda_runtime.h>
#include <vector_types.h>

#include <chrono>
#include <cstring>

namespace pb {

namespace {

bool save_field_for_kind(const std::filesystem::path& root,
                         std::string_view pass_name,
                         std::uint64_t key,
                         const FieldDescriptor& desc,
                         const PlanetState& state) {
    void* ptr = desc.field_ptr(const_cast<PlanetState&>(state));
    switch (desc.kind) {
        case FieldKind::F1: return save_field(root, pass_name, key, desc.name,
                                              *static_cast<const Field<float>*>(ptr));
        case FieldKind::F2: return save_field(root, pass_name, key, desc.name,
                                              *static_cast<const Field<float2>*>(ptr));
        case FieldKind::F4: return save_field(root, pass_name, key, desc.name,
                                              *static_cast<const Field<float4>*>(ptr));
        case FieldKind::U8: return save_field(root, pass_name, key, desc.name,
                                              *static_cast<const Field<std::uint8_t>*>(ptr));
    }
    return false;
}

bool load_field_for_kind(const std::filesystem::path& root,
                         std::string_view pass_name,
                         std::uint64_t key,
                         const FieldDescriptor& desc,
                         PlanetState& state) {
    void* ptr = desc.field_ptr(state);
    switch (desc.kind) {
        case FieldKind::F1: return load_field(root, pass_name, key, desc.name,
                                              *static_cast<Field<float>*>(ptr));
        case FieldKind::F2: return load_field(root, pass_name, key, desc.name,
                                              *static_cast<Field<float2>*>(ptr));
        case FieldKind::F4: return load_field(root, pass_name, key, desc.name,
                                              *static_cast<Field<float4>*>(ptr));
        case FieldKind::U8: return load_field(root, pass_name, key, desc.name,
                                              *static_cast<Field<std::uint8_t>*>(ptr));
    }
    return false;
}

bool all_writes_cached(const std::filesystem::path& root,
                       std::string_view pass_name,
                       std::uint64_t key,
                       FieldSet writes) {
    auto descs = PlanetState::descriptors();
    bool all_present = true;
    writes.for_each([&](FieldId fid) {
        const auto& desc = descs[static_cast<int>(fid)];
        if (!entry_exists(root, pass_name, key, desc.name)) all_present = false;
    });
    return all_present && writes.any();
}

template <typename T>
std::uint64_t hash_field_bytes(const Field<T>& f) {
    std::vector<T> host;
    f.download(host);
    return fnv1a64(host.data(), host.size() * sizeof(T));
}

std::uint64_t hash_field_for_kind(const FieldDescriptor& desc, const PlanetState& state) {
    void* ptr = desc.field_ptr(const_cast<PlanetState&>(state));
    switch (desc.kind) {
        case FieldKind::F1: return hash_field_bytes(*static_cast<const Field<float>*>(ptr));
        case FieldKind::F2: return hash_field_bytes(*static_cast<const Field<float2>*>(ptr));
        case FieldKind::F4: return hash_field_bytes(*static_cast<const Field<float4>*>(ptr));
        case FieldKind::U8: return hash_field_bytes(*static_cast<const Field<std::uint8_t>*>(ptr));
    }
    return 0;
}

}

Pipeline::Pipeline(std::filesystem::path cache_dir)
    : cache_dir_(std::move(cache_dir)) {
    std::error_code ec;
    std::filesystem::create_directories(cache_dir_, ec);
    if (ec) {
        PB_LOG_WARN("pipeline", "could not create cache dir %s: %s",
                    cache_dir_.string().c_str(), ec.message().c_str());
    }
}

void Pipeline::add_pass(std::unique_ptr<Pass> pass) {
    PassEntry entry;
    entry.name         = pass->name();
    entry.param_prefix = pass->param_prefix();
    entries_.push_back(std::move(entry));
    passes_.push_back(std::move(pass));
}

void Pipeline::declare_all(ParamRegistry& reg) const {
    for (const auto& p : passes_) {
        p->declare_params(reg);
    }
}

void Pipeline::wire_dirty_tracking(ParamRegistry& reg) {
    reg.on_change([this](std::string_view path) {
        for (const auto& p : passes_) {
            std::string prefix = p->param_prefix();
            if (path.size() >= prefix.size()
             && path.substr(0, prefix.size()) == prefix) {
                invalidate_pass(p->name());
                return;
            }
        }
    });
}

const Pass* Pipeline::find_pass(std::string_view name) const {
    for (const auto& p : passes_) {
        if (name == p->name()) return p.get();
    }
    return nullptr;
}

void Pipeline::cascade_dirty(std::size_t from_index) {
    if (from_index >= passes_.size()) return;
    FieldSet rolling = passes_[from_index]->writes();
    for (std::size_t j = from_index + 1; j < passes_.size(); ++j) {
        if ((passes_[j]->reads() & rolling).any()) {
            entries_[j].status      = PassStatus::Dirty;
            entries_[j].last_source = LastSource::None;
            rolling |= passes_[j]->writes();
        }
    }
}

void Pipeline::invalidate_pass(std::string_view name) {
    for (std::size_t i = 0; i < passes_.size(); ++i) {
        if (name == passes_[i]->name()) {
            entries_[i].status      = PassStatus::Dirty;
            entries_[i].last_source = LastSource::None;
            cascade_dirty(i);
            return;
        }
    }
}

void Pipeline::invalidate_from(std::string_view name) {
    for (std::size_t i = 0; i < passes_.size(); ++i) {
        if (name == passes_[i]->name()) {
            for (std::size_t j = i; j < passes_.size(); ++j) {
                entries_[j].status      = PassStatus::Dirty;
                entries_[j].last_source = LastSource::None;
            }
            return;
        }
    }
}

std::uint64_t Pipeline::compute_key(const Pass& p,
                                    const PlanetState& state,
                                    const ParamRegistry& reg) const {
    Hasher h;
    h.feed_string(p.name());
    std::uint64_t v = p.version();
    h.feed_pod(v);
    //grid signature
    h.feed_pod(static_cast<std::uint32_t>(state.grid().n));
    h.feed_pod(static_cast<std::uint32_t>(CubedSphereGrid::HALO));
    h.feed_pod(static_cast<std::uint32_t>(6));
    //params under the pass's declared param-tree prefix (usually "<name>.",
    //but BedrockNoisePass overrides to "bedrock." - see Pass::param_prefix).
    std::string prefix = p.param_prefix();
    reg.hash_prefix_into(prefix, h);
    //Input field content hashes, in stable FieldId order. Downloads each
    //input to host and FNV-1a's the bytes. Adds a few ms per input field
    //at preview resolution; cheap relative to most kernels.
    auto descs = PlanetState::descriptors();
    p.reads().for_each([&](FieldId fid) {
        const auto& desc = descs[static_cast<int>(fid)];
        std::uint64_t h_field = hash_field_for_kind(desc, state);
        h.feed_pod(h_field);
    });
    return h.finish();
}

bool Pipeline::try_load_writes(const Pass& p, std::uint64_t key, PlanetState& state) const {
    FieldSet writes = p.writes();
    if (!all_writes_cached(cache_dir_, p.name(), key, writes)) return false;

    auto descs = PlanetState::descriptors();
    bool ok = true;
    writes.for_each([&](FieldId fid) {
        if (!ok) return;
        const auto& desc = descs[static_cast<int>(fid)];
        if (!load_field_for_kind(cache_dir_, p.name(), key, desc, state)) ok = false;
    });
    return ok;
}

bool Pipeline::save_writes(const Pass& p, std::uint64_t key, const PlanetState& state) const {
    FieldSet writes = p.writes();
    if (writes.empty()) return true;

    auto descs = PlanetState::descriptors();
    bool ok = true;
    writes.for_each([&](FieldId fid) {
        const auto& desc = descs[static_cast<int>(fid)];
        if (!save_field_for_kind(cache_dir_, p.name(), key, desc, state)) ok = false;
    });
    return ok;
}

void Pipeline::run(PlanetState& state, const ParamRegistry& reg, ProgressSink& progress) {
    for (std::size_t i = 0; i < passes_.size(); ++i) {
        auto& entry = entries_[i];
        if (entry.status == PassStatus::Clean) continue;

        Pass& p = *passes_[i];
        entry.status      = PassStatus::Running;
        entry.last_error.clear();
        progress.stage(p.name());
        progress.fraction(0.0f);

        std::uint64_t key = compute_key(p, state, reg);
        entry.last_key    = key;

        auto t0 = std::chrono::steady_clock::now();

        bool ok = try_load_writes(p, key, state);
        if (ok) {
            entry.last_source = LastSource::Cache;
            PB_LOG_INFO("pipeline", "%s cache hit %s",
                        p.name(), format_key(key).c_str());
        } else {
            PB_LOG_INFO("pipeline", "%s compute %s",
                        p.name(), format_key(key).c_str());
            try {
                p.run(state, reg, progress);
                if (!save_writes(p, key, state)) {
                    PB_LOG_WARN("pipeline", "%s save_writes failed", p.name());
                }
                entry.last_source = LastSource::Computed;
                ok = true;
            } catch (const std::exception& e) {
                entry.status      = PassStatus::Error;
                entry.last_error  = e.what();
                entry.last_source = LastSource::None;
                PB_LOG_ERROR("pipeline", "%s threw: %s", p.name(), e.what());
                ok = false;
            }
        }

        auto t1 = std::chrono::steady_clock::now();
        entry.last_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (ok) {
            entry.status = PassStatus::Clean;
        }
        progress.fraction(1.0f);
    }
    progress.stage("");
    progress.fraction(0.0f);
}

}
