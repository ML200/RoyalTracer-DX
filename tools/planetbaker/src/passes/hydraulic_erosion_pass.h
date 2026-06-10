#pragma once

#include "core/field_set.h"
#include "core/pass.h"

namespace pb {

//====================================
//HydraulicErosionPass: Mei-et-al. virtual pipes hydraulic erosion adapted
//to the equiangular cubed sphere. Carves canyons / drainage networks into
//bedrock by simulating rainfall, water flow, suspended-sediment transport
//and deposition. PLAN section 7.7, "wet era override" variant: a forced
//rainfall amount is applied for a configurable number of iterations on an
//otherwise dry planet, leaving the carved valleys frozen in place.
//
//Per iteration, the kernels do:
//  1. Add `rain_per_iter_m` to the water column at every cell.
//  2. Compute pipe outflow flux from height differences with each 4-
//     neighbour (stateless: flux is recomputed each iter, no momentum).
//     Outflow is scaled so it never exceeds the available water column.
//  3. Update the water column: w' = w - sum(outflows) + sum(inflows from
//     neighbours, gathered by reading neighbours' outflow towards us).
//  4. Derive a per-cell speed proxy from net outflow; compute sediment
//     carrying capacity = K * slope * speed * water_depth. Compare to
//     suspended sediment; erode bedrock + sediment_thickness or deposit
//     into sediment_thickness as appropriate.
//  5. Transport suspended sediment by the same outflow fractions as the
//     water (scatter via halo + gather).
//  6. Evaporate: w' = w * (1 - evaporation_rate).
//
//Bedrock is sourced from BedrockNoisePass (+ impacts + thermal); the
//hydraulic pass eats away at it where flow concentrates. Sediment moved
//downstream is deposited as `sediment_thickness`, which the BakePass adds
//to the final surface elevation (bedrock_km + sediment_m * 0.001).
//
//Default `iterations = 0` -> no-op, so the existing dry-rocky pipeline is
//unchanged. Bump iterations to ~500 to carve visible canyons; ~2000 for
//a fully developed drainage network.
//
//Reads:  bedrock_elevation, sediment_thickness.
//Writes: bedrock_elevation, sediment_thickness, water_column.
//
//Params declared under "hydraulic.*".
//====================================

class HydraulicErosionPass : public Pass {
public:
    const char*   name()    const override { return "hydraulic"; }
    std::uint64_t version() const override { return 1; }

    FieldSet reads()  const override {
        return FieldSet{}
            .set(FieldId::BedrockElevation)
            .set(FieldId::SedimentThickness);
    }
    FieldSet writes() const override {
        return FieldSet{}
            .set(FieldId::BedrockElevation)
            .set(FieldId::SedimentThickness)
            .set(FieldId::WaterColumn);
    }

    void declare_params(ParamRegistry& reg) const override;
    void run(PlanetState& state, const ParamRegistry& reg, ProgressSink& progress) override;
};

}
