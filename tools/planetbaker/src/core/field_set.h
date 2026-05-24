#pragma once

#include <cstdint>

namespace pb {

//====================================
//Stable identifiers for every field in PlanetState. Order MUST match the
//declaration order in planet_state.h (and the descriptor table in
//planet_state.cpp). FieldId is the bit position in a FieldSet bitset. Bump
//Count when fields are added; FieldSet is sized for up to 32 fields.
//====================================

enum class FieldId : std::uint8_t {
    BedrockElevation = 0,
    CrustThickness,
    CrustDensity,
    CrustAge,
    CrustType,

    SedimentThickness,
    SoilThickness,
    IceThickness,
    WaterColumn,
    VegetationDensity,

    SurfaceTemperatureMean,
    SurfaceTemperatureRange,
    PrecipitationMean,
    AtmosphericMoisture,
    WindSurface,
    DustLoading,

    CloudDustOD,
    CloudIceOD,
    CloudWaterOD,
    CloudTopAltitude,

    BiomeWeights,
    SurfaceElevation,

    Count
};

constexpr int kFieldIdCount = static_cast<int>(FieldId::Count);
static_assert(kFieldIdCount <= 32, "FieldSet bitset only holds 32 fields");

//====================================
//Small bitset over the field roster. Used by Pass::reads() and Pass::writes()
//and by the Pipeline to compute the dirty cascade. operator| / operator&
//work the way they read.
//====================================

struct FieldSet {
    std::uint32_t bits = 0;

    constexpr FieldSet() = default;
    constexpr explicit FieldSet(std::uint32_t b) : bits(b) {}

    constexpr FieldSet& set(FieldId f) {
        bits |= (1u << static_cast<std::uint32_t>(f));
        return *this;
    }
    constexpr FieldSet& clear(FieldId f) {
        bits &= ~(1u << static_cast<std::uint32_t>(f));
        return *this;
    }
    constexpr bool test(FieldId f) const {
        return (bits & (1u << static_cast<std::uint32_t>(f))) != 0u;
    }
    constexpr bool contains(FieldId f) const { return test(f); }
    constexpr bool any() const { return bits != 0u; }
    constexpr bool empty() const { return bits == 0u; }

    constexpr FieldSet operator|(FieldSet other) const { return FieldSet{bits | other.bits}; }
    constexpr FieldSet operator&(FieldSet other) const { return FieldSet{bits & other.bits}; }
    constexpr FieldSet& operator|=(FieldSet other) { bits |= other.bits; return *this; }
    constexpr bool      operator==(FieldSet other) const { return bits == other.bits; }

    template <typename Fn>
    void for_each(Fn&& fn) const {
        for (int i = 0; i < kFieldIdCount; ++i) {
            if (bits & (1u << static_cast<std::uint32_t>(i))) {
                fn(static_cast<FieldId>(i));
            }
        }
    }
};

inline const char* field_id_name(FieldId f) {
    switch (f) {
        case FieldId::BedrockElevation:         return "bedrock_elevation";
        case FieldId::CrustThickness:           return "crust_thickness";
        case FieldId::CrustDensity:             return "crust_density";
        case FieldId::CrustAge:                 return "crust_age";
        case FieldId::CrustType:                return "crust_type";
        case FieldId::SedimentThickness:        return "sediment_thickness";
        case FieldId::SoilThickness:            return "soil_thickness";
        case FieldId::IceThickness:             return "ice_thickness";
        case FieldId::WaterColumn:              return "water_column";
        case FieldId::VegetationDensity:        return "vegetation_density";
        case FieldId::SurfaceTemperatureMean:   return "surface_temperature_mean";
        case FieldId::SurfaceTemperatureRange:  return "surface_temperature_range";
        case FieldId::PrecipitationMean:        return "precipitation_mean";
        case FieldId::AtmosphericMoisture:      return "atmospheric_moisture";
        case FieldId::WindSurface:              return "wind_surface";
        case FieldId::DustLoading:              return "dust_loading";
        case FieldId::CloudDustOD:              return "cloud_dust_od";
        case FieldId::CloudIceOD:               return "cloud_ice_od";
        case FieldId::CloudWaterOD:             return "cloud_water_od";
        case FieldId::CloudTopAltitude:         return "cloud_top_altitude";
        case FieldId::BiomeWeights:             return "biome_weights";
        case FieldId::SurfaceElevation:         return "surface_elevation";
        case FieldId::Count:                    return "<count>";
    }
    return "<unknown>";
}

}
