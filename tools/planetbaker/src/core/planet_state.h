#pragma once

#include <cstdint>
#include <span>

#include "core/cubed_sphere.h"
#include "core/field.h"
#include "core/neighbor_table.h"

namespace pb {

//====================================
//Element type of a field, used for runtime dispatch (display, bake, etc).
//Channels gives the meaningful scalar count per cell for UI purposes
//(F1 = 1, F2 = 2, F4 = 4, U8 = 1).
//====================================

enum class FieldKind : std::uint8_t {
    F1,
    F2,
    F4,
    U8,
};

//Which subset of fields to allocate up front. Higher-res previews / bakes
//only need the bedrock-pipeline fields; allocating climate / cloud / biome
//fields at 8k+ exceeds typical desktop VRAM.
enum class PlanetFieldSet : std::uint8_t {
    //Every field listed in PLAN section 6.3 (full pipeline).
    All,
    //Fields written by BedrockNoisePass + ImpactsPass: bedrock_elevation,
    //crust_thickness, crust_density, crust_age, crust_type, and
    //sediment_thickness (the impacts ejecta layer). Other Field<T>
    //members default-construct empty (grid.n == 0); any pass that reads
    //them would crash, so caller must register only the bedrock-pipeline
    //passes (BedrockNoisePass + ImpactsPass).
    BedrockOnly,
};

struct PlanetState;

//====================================
//Estimate the device memory required for a PlanetState allocation at a
//given resolution and PlanetFieldSet. Sum of every Field<T>'s bytes. Useful for
//memory-budget messages at startup before any allocation happens.
//====================================
std::size_t estimate_planet_state_bytes(int face_resolution, PlanetFieldSet which);

struct FieldDescriptor {
    const char* name;
    FieldKind   kind;
    int         channels;
    float       default_min;
    float       default_max;
    //Returns a pointer to the Field<T> object inside the supplied state, where
    //T matches kind. Cast with static_cast<Field<T>*>(...).
    void*     (*field_ptr)(PlanetState&);
};

//====================================
//PlanetState owns one CubedSphereGrid and the matching NeighborTable, plus
//every field listed in PLAN section 6.3. All fields are allocated up front
//and zero-initialized. Fields not exercised by a given world type stay at
//zero. Cost at N=512 is roughly 155 MB total.
//
//The static descriptors() table lets the viewer and future per-field
//tooling iterate fields by name and dispatch based on FieldKind without
//naming individual members.
//====================================

struct PlanetState {
    explicit PlanetState(int face_resolution, PlanetFieldSet which = PlanetFieldSet::All);

    const CubedSphereGrid& grid()      const { return grid_; }
    const NeighborTable&   neighbors() const { return neighbors_; }

    //Elevation and crust
    Field<float>         bedrock_elevation;
    Field<float>         crust_thickness;
    Field<float>         crust_density;
    Field<float>         crust_age;
    Field<std::uint8_t>  crust_type;

    //Surface deposits
    Field<float>         sediment_thickness;
    Field<float>         soil_thickness;
    Field<float>         ice_thickness;
    Field<float>         water_column;
    Field<float>         vegetation_density;

    //Climate
    Field<float>         surface_temperature_mean;
    Field<float>         surface_temperature_range;
    Field<float>         precipitation_mean;
    Field<float>         atmospheric_moisture;
    Field<float2>        wind_surface;
    Field<float>         dust_loading;

    //Cloud state
    Field<float>         cloud_dust_od;
    Field<float>         cloud_ice_od;
    Field<float>         cloud_water_od;
    Field<float>         cloud_top_altitude;

    //Derived
    Field<float4>        biome_weights;
    Field<float>         surface_elevation;

    static std::span<const FieldDescriptor> descriptors();

private:
    CubedSphereGrid grid_;
    NeighborTable   neighbors_;
};

}
