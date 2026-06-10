#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include "core/field_set.h"

namespace pb {

struct PlanetState;
class  ParamRegistry;

//====================================
//Reports stage label + fractional progress (0..1) to the viewer or CLI.
//Pass.run() owns the cadence: call stage() when entering a logical phase,
//fraction() periodically. The default sink is a no-op (CLI / cache hits).
//====================================

class ProgressSink {
public:
    virtual ~ProgressSink() = default;
    virtual void stage(std::string_view label) = 0;
    virtual void fraction(float f) = 0;
};

class NullProgressSink final : public ProgressSink {
public:
    void stage(std::string_view) override {}
    void fraction(float) override {}
};

//====================================
//Pass is the unit of computation in the pipeline. Subclasses declare their
//inputs (reads), outputs (writes), tunable params, and the kernel logic
//(run). The Pipeline uses name + version + params + input field hashes to
//compute a deterministic cache key per pass.
//
//Bump version() when the algorithm changes in a way that invalidates
//existing cache entries. Param changes do NOT need a version bump (they
//flow into the key directly).
//====================================

class Pass {
public:
    virtual ~Pass() = default;

    virtual const char*   name()    const = 0;
    virtual std::uint64_t version() const = 0;

    //Param-tree prefix this pass owns. Default is `<name>.`, so most passes
    //don't need to override (impacts -> "impacts.", thermal -> "thermal.",
    //hydraulic -> "hydraulic.", etc). Override when the pass-name and the
    //declare_params prefix intentionally diverge - BedrockNoisePass is the
    //only existing case (name = "bedrock_noise", prefix = "bedrock."). The
    //Pipeline uses this for dirty-tracking and cache-key hashing; the UI
    //uses it to iterate params for display. Returning the wrong prefix
    //silently breaks both.
    virtual std::string param_prefix() const { return std::string(name()) + "."; }

    virtual FieldSet reads()  const = 0;
    virtual FieldSet writes() const = 0;

    virtual void declare_params(ParamRegistry& reg) const = 0;
    virtual void run(PlanetState& state,
                     const ParamRegistry& reg,
                     ProgressSink& progress) = 0;
};

}
