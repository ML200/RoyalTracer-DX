#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "heightmap_source.h"

namespace planet {

class HeightmapCubemap : public IHeightmapSource {
public:
    HeightmapCubemap() = default;

    // Loads elevation, surface color, and normal cubemap faces.
    bool load(const std::filesystem::path& terrain_dir);

    bool   loaded()         const { return m_resolution > 0; }
    uint32_t resolution()   const { return m_resolution; }

    // Downsamples one face into elevation values expressed in kilometres.
    bool downsample_face_km(uint8_t face, uint32_t dst_resolution, float* out) const;

    float sample(const DVec3& dir_normalized, uint8_t lod) const override;

    bool     surface_color_loaded() const { return m_surface_color_resolution > 0; }
    uint32_t surface_color_resolution() const { return m_surface_color_resolution; }
    bool surface_color_face(uint8_t face, uint32_t dst_resolution, uint8_t* out) const;

    bool     normal_loaded() const { return m_normal_resolution > 0; }
    uint32_t normal_resolution() const { return m_normal_resolution; }
    bool normal_face(uint8_t face, uint32_t dst_resolution, uint8_t* out) const;

private:
    uint32_t                          m_resolution = 0;
    std::vector<std::vector<float>>   m_faces;

    uint32_t                              m_surface_color_resolution = 0;
    std::vector<std::vector<std::uint8_t>> m_surface_color_faces;

    uint32_t                              m_normal_resolution = 0;
    std::vector<std::vector<std::uint8_t>> m_normal_faces;

};

}
