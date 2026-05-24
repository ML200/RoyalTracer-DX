#pragma once

#include "core/field_set.h"
#include "core/pass.h"

namespace pb {

//====================================
//ImpactsPass: stamps a population of craters onto bedrock. Power-law SFD,
//morphology-dependent profile (simple bowl / complex bowl + central peak /
//basin / multi-ring), ejecta with cubic falloff. Per PLAN section 7.2.
//
//Reads:  bedrock_elevation (modified in place).
//Writes: bedrock_elevation, sediment_thickness, crust_type.
//
//Params declared under "impacts.*".
//====================================

class ImpactsPass : public Pass {
public:
    const char*   name()    const override { return "impacts"; }
    //v2: crater depth/diameter ratios re-tuned to Pike's law and observed
    //planetary depths. v1 used 0.20 / 0.10 / 0.04 / 0.03 (Simple /
    //Complex / Basin / Multiring) which gave 100 km craters 10 km deep
    //- taller than the tallest mountains. v2 uses 0.10 / 0.025 / 0.008 /
    //0.004, matching real Hellas-class basins (~9 km deep at 2300 km).
    //Rim heights scaled down proportionally.
    std::uint64_t version() const override { return 2; }

    FieldSet reads()  const override { return FieldSet{}.set(FieldId::BedrockElevation); }
    FieldSet writes() const override {
        return FieldSet{}
            .set(FieldId::BedrockElevation)
            .set(FieldId::SedimentThickness)
            .set(FieldId::CrustType);
    }

    void declare_params(ParamRegistry& reg) const override;
    void run(PlanetState& state, const ParamRegistry& reg, ProgressSink& progress) override;
};

}
