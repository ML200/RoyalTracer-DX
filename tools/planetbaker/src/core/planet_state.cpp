#include "core/planet_state.h"

namespace pb {

PlanetState::PlanetState(int face_resolution, PlanetFieldSet which)
    : bedrock_elevation         (CubedSphereGrid(face_resolution)),
      crust_thickness           (CubedSphereGrid(face_resolution)),
      crust_density             (CubedSphereGrid(face_resolution)),
      crust_age                 (CubedSphereGrid(face_resolution)),
      crust_type                (CubedSphereGrid(face_resolution)),
      sediment_thickness        (CubedSphereGrid(face_resolution)),
      //water_column is allocated in BedrockOnly because HydraulicErosionPass
      //runs in the bedrock-only pipeline; the rest of the climate / cloud /
      //biome fields stay default-constructed (grid.n == 0) and are filled in
      //below only when PlanetFieldSet::All is requested.
      water_column              (CubedSphereGrid(face_resolution)),
      //surface_color is allocated in BedrockOnly too because SurfaceColorPass
      //runs as part of the heightmap pipeline; the viewer + BakePass both
      //consume the result. ~16 bytes/cell adds ~6.4 GB at 8 k and ~100 MB at
      //2 k preview.
      surface_color             (CubedSphereGrid(face_resolution)),
      grid_                     (face_resolution),
      neighbors_                (grid_)
{
    bedrock_elevation.zero();
    crust_thickness.zero();
    crust_density.zero();
    crust_age.zero();
    crust_type.zero();
    sediment_thickness.zero();
    water_column.zero();
    surface_color.zero();

    if (which == PlanetFieldSet::All) {
        soil_thickness            = Field<float>       (grid_);
        ice_thickness             = Field<float>       (grid_);
        vegetation_density        = Field<float>       (grid_);
        surface_temperature_mean  = Field<float>       (grid_);
        surface_temperature_range = Field<float>       (grid_);
        precipitation_mean        = Field<float>       (grid_);
        atmospheric_moisture      = Field<float>       (grid_);
        wind_surface              = Field<float2>      (grid_);
        dust_loading              = Field<float>       (grid_);
        cloud_dust_od             = Field<float>       (grid_);
        cloud_ice_od              = Field<float>       (grid_);
        cloud_water_od            = Field<float>       (grid_);
        cloud_top_altitude        = Field<float>       (grid_);
        biome_weights             = Field<float4>      (grid_);
        surface_elevation         = Field<float>       (grid_);

        soil_thickness.zero();
        ice_thickness.zero();
        vegetation_density.zero();
        surface_temperature_mean.zero();
        surface_temperature_range.zero();
        precipitation_mean.zero();
        atmospheric_moisture.zero();
        wind_surface.zero();
        dust_loading.zero();
        cloud_dust_od.zero();
        cloud_ice_od.zero();
        cloud_water_od.zero();
        cloud_top_altitude.zero();
        biome_weights.zero();
        surface_elevation.zero();
    }
}

std::size_t estimate_planet_state_bytes(int face_resolution, PlanetFieldSet which) {
    const std::size_t cells_per_face = static_cast<std::size_t>(face_resolution + 2 * CubedSphereGrid::HALO);
    const std::size_t total_cells    = 6 * cells_per_face * cells_per_face;

    //Bedrock-pipeline fields always allocated: 6 float (bedrock_elevation,
    //crust_thickness/density/age + sediment_thickness for impacts ejecta +
    //water_column for hydraulic erosion) + 1 uint8 (crust_type) + 1 float4
    //(surface_color RGBA for SurfaceColorPass) per cell.
    const std::size_t bedrock_bytes_per_cell =
          6 * sizeof(float)
        +  sizeof(std::uint8_t)
        +  4 * sizeof(float);   //surface_color RGBA
    std::size_t bytes = total_cells * bedrock_bytes_per_cell;

    if (which == PlanetFieldSet::All) {
        //13 additional float fields + 1 float2 + 1 float4 (water_column +
        //surface_color moved out of the All-only block into the always-
        //allocated set).
        const std::size_t extra_bytes_per_cell =
              13 * sizeof(float)
            +  1 * 2 * sizeof(float)
            +  1 * 4 * sizeof(float);
        bytes += total_cells * extra_bytes_per_cell;
    }
    return bytes;
}

//====================================
//Descriptor table. Default min/max are display defaults for the viewer; they
//are not authoritative bounds on the underlying value. Lambdas are
//captureless so they decay to function pointers.
//====================================

#define PB_F(member, kind_v, channels_v, dmin, dmax)                       \
    FieldDescriptor{                                                       \
        #member, FieldKind::kind_v, (channels_v), (dmin), (dmax),          \
        +[](PlanetState& s) -> void* { return &s.member; }                 \
    }

std::span<const FieldDescriptor> PlanetState::descriptors() {
    static const FieldDescriptor table[] = {
        PB_F(bedrock_elevation,         F1, 1,  -10.0f,   10.0f),
        PB_F(crust_thickness,           F1, 1,    0.0f,   80.0f),
        PB_F(crust_density,             F1, 1, 2700.0f, 3100.0f),
        PB_F(crust_age,                 F1, 1,    0.0f, 2000.0f),
        PB_F(crust_type,                U8, 1,    0.0f,    3.0f),
        PB_F(sediment_thickness,        F1, 1,    0.0f,   10.0f),
        PB_F(soil_thickness,            F1, 1,    0.0f,    5.0f),
        PB_F(ice_thickness,             F1, 1,    0.0f, 1000.0f),
        PB_F(water_column,              F1, 1,    0.0f, 4000.0f),
        PB_F(vegetation_density,        F1, 1,    0.0f,    1.0f),
        PB_F(surface_temperature_mean,  F1, 1,  220.0f,  320.0f),
        PB_F(surface_temperature_range, F1, 1,    0.0f,   60.0f),
        PB_F(precipitation_mean,        F1, 1,    0.0f,    5.0f),
        PB_F(atmospheric_moisture,      F1, 1,    0.0f,   50.0f),
        PB_F(wind_surface,              F2, 2,  -30.0f,   30.0f),
        PB_F(dust_loading,              F1, 1,    0.0f,    1.0f),
        PB_F(cloud_dust_od,             F1, 1,    0.0f,    1.0f),
        PB_F(cloud_ice_od,              F1, 1,    0.0f,    1.0f),
        PB_F(cloud_water_od,            F1, 1,    0.0f,    1.0f),
        PB_F(cloud_top_altitude,        F1, 1,    0.0f,   20.0f),
        PB_F(biome_weights,             F4, 4,    0.0f,    1.0f),
        PB_F(surface_elevation,         F1, 1,   -3.0f,    3.0f),
        PB_F(surface_color,             F4, 4,    0.0f,    1.0f),
    };
    return std::span<const FieldDescriptor>(table, sizeof(table) / sizeof(table[0]));
}

#undef PB_F

}
