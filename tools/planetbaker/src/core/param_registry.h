#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "core/hash.h"

namespace pb {

//====================================
//ParamSlot variants. The discriminator is which alternative is held in the
//variant. ParamRegistry indirects through ParamSlot to avoid spilling
//implementation details into headers that just read params.
//====================================

struct IntSlot   { int   value; int   default_value; int   lo; int   hi; };
struct FloatSlot { float value; float default_value; float lo; float hi; };
struct BoolSlot  { bool  value; bool  default_value; };

struct ParamSlot {
    std::variant<IntSlot, FloatSlot, BoolSlot> v;
    std::string label;
    std::string tooltip;
    std::string units;
    std::string owner;   //pass name (used for prefix grouping in the UI)
};

enum class ParamType : std::uint8_t { Int = 0, Float = 1, Bool = 2 };

inline ParamType param_type_of(const ParamSlot& s) {
    return static_cast<ParamType>(s.v.index());
}

//====================================
//ParamRegistry is the source of truth for every tunable value in the tool.
//Passes register their params via declare_*; the viewer renders them by
//walking prefix_iter; the Pipeline subscribes to on_change to flip
//pass status to Dirty when a value changes.
//====================================

class ParamRegistry {
public:
    using ChangeCallback = std::function<void(std::string_view path)>;

    //Declarations. Re-declaring an existing path is a no-op (lets the
    //pipeline call declare_all on every startup without losing values).
    void declare_int  (std::string path, int   default_value, int lo, int hi,
                       std::string label, std::string tooltip, std::string units, std::string owner);
    void declare_float(std::string path, float default_value, float lo, float hi,
                       std::string label, std::string tooltip, std::string units, std::string owner);
    void declare_bool (std::string path, bool  default_value,
                       std::string label, std::string tooltip, std::string units, std::string owner);

    //Typed getters. Asserts on missing path or type mismatch (caller error).
    int   get_int  (std::string_view path) const;
    float get_float(std::string_view path) const;
    bool  get_bool (std::string_view path) const;

    //Setters fire on_change callbacks. Out-of-range values are clamped to
    //[lo, hi] for int/float; bool is unconditionally accepted.
    void set_int  (std::string_view path, int   value);
    void set_float(std::string_view path, float value);
    void set_bool (std::string_view path, bool  value);

    bool has(std::string_view path) const;

    //Iteration. Visitor: (path, slot). Slots are iterated in lexicographic
    //order by path within the chosen prefix (std::map gives this for free).
    void prefix_iter(std::string_view prefix,
                     const std::function<void(std::string_view, const ParamSlot&)>& visit) const;

    //M14 presets: leaves any slot whose path is in touched untouched; resets
    //every other slot to its default. touched is matched by exact equality.
    void reset_unaffected(const std::vector<std::string>& touched);

    //Persistence. JSON schema is { "<path>": { "type": "<int|float|bool>", "value": <v> } }.
    //load() skips paths not previously declared and logs a warning, so call
    //declare_all() before load().
    bool save(const std::filesystem::path& file) const;
    bool load(const std::filesystem::path& file);

    //Pipeline hooks in.
    void on_change(ChangeCallback cb);

    //Compute a deterministic hash contribution covering only slots whose
    //path begins with the given prefix. Used to build per-pass cache keys.
    void hash_prefix_into(std::string_view prefix, Hasher& h) const;

private:
    void fire_change(std::string_view path);

    std::map<std::string, ParamSlot, std::less<>> slots_;
    std::vector<ChangeCallback>                   callbacks_;
};

}
