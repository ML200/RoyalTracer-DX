#pragma once

#include "core/field_set.h"
#include "core/pass.h"

namespace pb {

//====================================
//SurfaceColorPass: per-cell Mars-like RGB tint computed from the heightmap
//pipeline outputs. "Far detail only" - no per-pixel rock variation; just
//big-picture colouring so the planet reads as a rocky body from orbit:
//
//  * Base palette interpolates a deep mare-red (low elevation), a Mars-
//    standard red-brown (mid), and a dusty highland tan (high).
//  * Steep slopes blend toward an exposed-rock grey (less weathered).
//  * Heavy sediment cover (impact ejecta blankets, hydraulic deposits)
//    blends toward a lighter dust tan so crater rims and fans read.
//  * Impact-melt cells (crust_type == 1) darken a touch to suggest
//    chilled basaltic floors.
//  * Polar latitudes blend toward ice white past a configurable band.
//  * High-altitude peaks above an altitude threshold blend to ice too,
//    giving Mars-style snow caps on the largest mountains.
//
//Reads:  bedrock_elevation, sediment_thickness, crust_type.
//Writes: surface_color (RGBA, .w is reserved / unused for now).
//
//Params declared under "color.*".
//====================================

class SurfaceColorPass : public Pass {
public:
    const char*   name()    const override { return "surface_color"; }
    std::string   param_prefix() const override { return "color."; }
    //v2: polar + mountain ice borders perturbed by fBm noise on the cell
    //direction so cap edges scallop instead of running along a clean
    //latitude / elevation contour.
    //v3: large-scale albedo provinces (dust-mantled bright vs basaltic dark) -
    //a low-freq field decorrelated from elevation, which is Mars's dominant
    //orbital albedo signal and which the elevation-only palette cannot make.
    //Palette anchors also desaturated toward muted true-colour brown.
    std::uint64_t version() const override { return 3; }

    FieldSet reads()  const override {
        return FieldSet{}
            .set(FieldId::BedrockElevation)
            .set(FieldId::SedimentThickness)
            .set(FieldId::CrustType);
    }
    FieldSet writes() const override {
        return FieldSet{}.set(FieldId::SurfaceColor);
    }

    void declare_params(ParamRegistry& reg) const override;
    void run(PlanetState& state, const ParamRegistry& reg, ProgressSink& progress) override;
};

}
