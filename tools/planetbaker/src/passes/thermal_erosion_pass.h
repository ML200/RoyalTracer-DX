#pragma once

#include "core/field_set.h"
#include "core/pass.h"

namespace pb {

//====================================
//ThermalErosionPass: relaxes sediment toward the local angle of repose.
//PLAN section 7.4. Per iteration: 4-neighbour gather. Where slope exceeds
//the cell's repose (sand-like vs rock-like by local sediment), transfer
//sediment downhill, capped by what's actually available. Bedrock is held
//fixed in M7 — only sediment relaxes. Crater ejecta apron + mountain talus
//emerge after a few hundred iterations.
//
//Reads:  bedrock_elevation, sediment_thickness.
//Writes: bedrock_elevation (no-op in M7; declared for forward compat),
//        sediment_thickness.
//
//Params declared under "thermal.*".
//====================================

class ThermalErosionPass : public Pass {
public:
    const char*   name()    const override { return "thermal"; }
    //v2: biome_weights dropped from reads. The kernel never read it - the
    //sand-vs-rock repose split is driven by sediment depth, not biome.
    //Removing the dependency lets the pass run without BiomeClassifier
    //having to be in the pipeline.
    std::uint64_t version() const override { return 2; }

    FieldSet reads()  const override {
        return FieldSet{}
            .set(FieldId::BedrockElevation)
            .set(FieldId::SedimentThickness);
    }
    FieldSet writes() const override {
        return FieldSet{}
            .set(FieldId::BedrockElevation)
            .set(FieldId::SedimentThickness);
    }

    void declare_params(ParamRegistry& reg) const override;
    void run(PlanetState& state, const ParamRegistry& reg, ProgressSink& progress) override;
};

}
