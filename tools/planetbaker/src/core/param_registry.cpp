#include "core/param_registry.h"
#include "core/log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cassert>
#include <fstream>

namespace pb {

namespace {

template <typename T>
T clamp_to(T v, T lo, T hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

}

void ParamRegistry::declare_int(std::string path, int def, int lo, int hi,
                                std::string label, std::string tooltip,
                                std::string units, std::string owner) {
    if (slots_.find(path) != slots_.end()) return;
    ParamSlot s;
    s.v       = IntSlot{def, def, lo, hi};
    s.label   = std::move(label);
    s.tooltip = std::move(tooltip);
    s.units   = std::move(units);
    s.owner   = std::move(owner);
    slots_.emplace(std::move(path), std::move(s));
}

void ParamRegistry::declare_float(std::string path, float def, float lo, float hi,
                                  std::string label, std::string tooltip,
                                  std::string units, std::string owner) {
    if (slots_.find(path) != slots_.end()) return;
    ParamSlot s;
    s.v       = FloatSlot{def, def, lo, hi};
    s.label   = std::move(label);
    s.tooltip = std::move(tooltip);
    s.units   = std::move(units);
    s.owner   = std::move(owner);
    slots_.emplace(std::move(path), std::move(s));
}

void ParamRegistry::declare_bool(std::string path, bool def,
                                 std::string label, std::string tooltip,
                                 std::string units, std::string owner) {
    if (slots_.find(path) != slots_.end()) return;
    ParamSlot s;
    s.v       = BoolSlot{def, def};
    s.label   = std::move(label);
    s.tooltip = std::move(tooltip);
    s.units   = std::move(units);
    s.owner   = std::move(owner);
    slots_.emplace(std::move(path), std::move(s));
}

int ParamRegistry::get_int(std::string_view path) const {
    auto it = slots_.find(path);
    assert(it != slots_.end() && "unknown param path");
    auto* p = std::get_if<IntSlot>(&it->second.v);
    assert(p && "wrong type for get_int");
    return p->value;
}

float ParamRegistry::get_float(std::string_view path) const {
    auto it = slots_.find(path);
    assert(it != slots_.end() && "unknown param path");
    auto* p = std::get_if<FloatSlot>(&it->second.v);
    assert(p && "wrong type for get_float");
    return p->value;
}

bool ParamRegistry::get_bool(std::string_view path) const {
    auto it = slots_.find(path);
    assert(it != slots_.end() && "unknown param path");
    auto* p = std::get_if<BoolSlot>(&it->second.v);
    assert(p && "wrong type for get_bool");
    return p->value;
}

void ParamRegistry::set_int(std::string_view path, int value) {
    auto it = slots_.find(path);
    if (it == slots_.end()) { PB_LOG_WARN("params", "set_int unknown path '%.*s'", (int)path.size(), path.data()); return; }
    auto* p = std::get_if<IntSlot>(&it->second.v);
    if (!p) { PB_LOG_WARN("params", "set_int type mismatch on '%.*s'", (int)path.size(), path.data()); return; }
    int clamped = clamp_to(value, p->lo, p->hi);
    if (clamped == p->value) return;
    p->value = clamped;
    fire_change(it->first);
}

void ParamRegistry::set_float(std::string_view path, float value) {
    auto it = slots_.find(path);
    if (it == slots_.end()) { PB_LOG_WARN("params", "set_float unknown path '%.*s'", (int)path.size(), path.data()); return; }
    auto* p = std::get_if<FloatSlot>(&it->second.v);
    if (!p) { PB_LOG_WARN("params", "set_float type mismatch on '%.*s'", (int)path.size(), path.data()); return; }
    float clamped = clamp_to(value, p->lo, p->hi);
    if (clamped == p->value) return;
    p->value = clamped;
    fire_change(it->first);
}

void ParamRegistry::set_bool(std::string_view path, bool value) {
    auto it = slots_.find(path);
    if (it == slots_.end()) { PB_LOG_WARN("params", "set_bool unknown path '%.*s'", (int)path.size(), path.data()); return; }
    auto* p = std::get_if<BoolSlot>(&it->second.v);
    if (!p) { PB_LOG_WARN("params", "set_bool type mismatch on '%.*s'", (int)path.size(), path.data()); return; }
    if (value == p->value) return;
    p->value = value;
    fire_change(it->first);
}

bool ParamRegistry::has(std::string_view path) const {
    return slots_.find(path) != slots_.end();
}

void ParamRegistry::prefix_iter(std::string_view prefix,
                                const std::function<void(std::string_view, const ParamSlot&)>& visit) const {
    auto it = slots_.lower_bound(prefix);
    for (; it != slots_.end(); ++it) {
        const std::string& key = it->first;
        if (key.size() < prefix.size() || std::string_view(key).substr(0, prefix.size()) != prefix) break;
        visit(key, it->second);
    }
}

void ParamRegistry::reset_unaffected(const std::vector<std::string>& touched) {
    for (auto& [path, slot] : slots_) {
        if (std::find(touched.begin(), touched.end(), path) != touched.end()) continue;
        std::visit([&](auto&& s) {
            using T = std::decay_t<decltype(s)>;
            if constexpr (std::is_same_v<T, IntSlot>)   s.value = s.default_value;
            else if constexpr (std::is_same_v<T, FloatSlot>) s.value = s.default_value;
            else if constexpr (std::is_same_v<T, BoolSlot>)  s.value = s.default_value;
        }, slot.v);
        fire_change(path);
    }
}

bool ParamRegistry::save(const std::filesystem::path& file) const {
    nlohmann::json out = nlohmann::json::object();
    for (const auto& [path, slot] : slots_) {
        nlohmann::json entry;
        std::visit([&](auto&& s) {
            using T = std::decay_t<decltype(s)>;
            if constexpr (std::is_same_v<T, IntSlot>)   { entry["type"] = "int";   entry["value"] = s.value; }
            else if constexpr (std::is_same_v<T, FloatSlot>) { entry["type"] = "float"; entry["value"] = s.value; }
            else if constexpr (std::is_same_v<T, BoolSlot>)  { entry["type"] = "bool";  entry["value"] = s.value; }
        }, slot.v);
        out[path] = std::move(entry);
    }
    std::ofstream os(file);
    if (!os) { PB_LOG_ERROR("params", "save failed: cannot open %s", file.string().c_str()); return false; }
    os << out.dump(2);
    return static_cast<bool>(os);
}

bool ParamRegistry::load(const std::filesystem::path& file) {
    std::ifstream is(file);
    if (!is) {
        PB_LOG_INFO("params", "no config at %s; using defaults", file.string().c_str());
        return false;
    }
    nlohmann::json j;
    try {
        is >> j;
    } catch (const std::exception& e) {
        PB_LOG_ERROR("params", "load parse error in %s: %s", file.string().c_str(), e.what());
        return false;
    }
    if (!j.is_object()) { PB_LOG_ERROR("params", "load: top level is not an object"); return false; }

    for (auto& [key, val] : j.items()) {
        if (!val.is_object() || !val.contains("type") || !val.contains("value")) {
            PB_LOG_WARN("params", "load: skipping malformed entry '%s'", key.c_str());
            continue;
        }
        auto it = slots_.find(key);
        if (it == slots_.end()) {
            PB_LOG_WARN("params", "load: skipping undeclared param '%s'", key.c_str());
            continue;
        }
        const std::string type = val["type"].get<std::string>();
        try {
            if (type == "int"   && std::holds_alternative<IntSlot>  (it->second.v)) set_int  (key, val["value"].get<int>());
            else if (type == "float" && std::holds_alternative<FloatSlot>(it->second.v)) set_float(key, val["value"].get<float>());
            else if (type == "bool"  && std::holds_alternative<BoolSlot> (it->second.v)) set_bool (key, val["value"].get<bool>());
            else PB_LOG_WARN("params", "load: type mismatch for '%s' (file says %s)", key.c_str(), type.c_str());
        } catch (const std::exception& e) {
            PB_LOG_WARN("params", "load: bad value for '%s': %s", key.c_str(), e.what());
        }
    }
    PB_LOG_INFO("params", "loaded %s", file.string().c_str());
    return true;
}

void ParamRegistry::on_change(ChangeCallback cb) {
    callbacks_.push_back(std::move(cb));
}

void ParamRegistry::hash_prefix_into(std::string_view prefix, Hasher& h) const {
    auto it = slots_.lower_bound(prefix);
    for (; it != slots_.end(); ++it) {
        const std::string& key = it->first;
        if (key.size() < prefix.size() || std::string_view(key).substr(0, prefix.size()) != prefix) break;
        h.feed_string(key);
        std::visit([&](auto&& s) {
            using T = std::decay_t<decltype(s)>;
            if constexpr (std::is_same_v<T, IntSlot>) {
                std::uint8_t tag = 0; h.feed_pod(tag); h.feed_pod(s.value);
            } else if constexpr (std::is_same_v<T, FloatSlot>) {
                std::uint8_t tag = 1; h.feed_pod(tag); h.feed_pod(s.value);
            } else if constexpr (std::is_same_v<T, BoolSlot>) {
                std::uint8_t tag = 2; h.feed_pod(tag);
                std::uint8_t b = s.value ? 1 : 0; h.feed_pod(b);
            }
        }, it->second.v);
    }
}

void ParamRegistry::fire_change(std::string_view path) {
    for (auto& cb : callbacks_) cb(path);
}

}
