#pragma once

#include <cstdint>
#include <vector>

#include "core/field_set.h"
#include "core/pass.h"

namespace pb {

class ParamRegistry;

//====================================
//Impact crater public API
//====================================
//Crater + ImpactsParams + sample_craters were originally private to the
//.cu's anonymous namespace. They're exposed here so BakePass can stamp the
//same population of craters directly into the per-face buffer it produces in
//the bedrock-only direct-synthesis path (used when bake.elevation_resolution
//exceeds the full-pipeline VRAM cap, i.e. 16k bakes). The Pass interface
//still runs them via the PlanetState fields when the full pipeline fits.

//Crater morphology values stored in Crater::morphology. Public because
//callers (bake_pass) need to count basins / multi-ring craters for logging
//and may want to filter by morphology.
enum CraterMorphology : std::uint8_t {
    MorphSimple    = 0,
    MorphComplex   = 1,
    MorphBasin     = 2,
    MorphMultiring = 3,
};

struct Crater {
    float pos_x;          // unit-sphere centre direction
    float pos_y;
    float pos_z;
    float diameter_km;
    float age_myr;
    float ejecta_extent;  // ejecta reach in crater radii
    std::uint32_t morphology;  // CraterMorphology cast to uint32_t
};

struct ImpactsParams {
    float         flux_multiplier;
    float         size_distribution_exponent;
    float         min_diameter_km;
    float         max_diameter_km;
    float         epoch_start_myr;
    float         epoch_end_myr;
    float         simple_complex_threshold_km;
    float         basin_threshold_km;
    float         multiring_threshold_km;
    float         ejecta_extent_radii;
    std::uint64_t seed;
};

//Reads impacts.* from the registry into an ImpactsParams struct.
ImpactsParams load_impacts_params(const ParamRegistry& reg);

//Host-side SFD sampler. Returns N craters sorted oldest-first so newer
//craters splat on top during the splat pass.
std::vector<Crater> sample_craters(const ImpactsParams& p);

//Stamps craters into a tight n*n single-face bedrock buffer (KM, no halo,
//no padding) - matches the layout bake_bedrock_face writes. Ejecta thickness
//and rim height are added to the bedrock buffer directly since this path
//does not maintain a separate sediment field (the >8192 bake path skips the
//full PlanetState to stay within VRAM). craters_device is a device-side
//array of `crater_count` craters (use a DeviceBuffer<Crater>).
void bake_impacts_face(int face, int n,
                       const Crater* craters_device,
                       int crater_count,
                       float planet_radius_km,
                       float* dst_device);

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
