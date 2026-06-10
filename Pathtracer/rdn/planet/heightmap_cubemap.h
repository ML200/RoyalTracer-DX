#pragma once
//====================================
//PLANET - CUBEMAP HEIGHTMAP
//====================================
//IHeightmapSource backed by the 6 cube-face .r32 files the planet baker
//writes. Loads ./terrain/manifest.json + the 6 elevation_face<N>.r32 files
//on construction. Sampling: equiangular cubed-sphere projection (matches
//the baker's `face_uv_to_sphere`) + bilinear. Returns metres.
//
//The bake stores surface elevation (bedrock + sediment) in KILOMETRES; the
//IHeightmapSource contract is METRES, so sample() multiplies by 1000 on
//the way out.
//
//Thread-safe (read-only data after load).

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "heightmap_source.h"

namespace planet {

class HeightmapCubemap : public IHeightmapSource {
public:
    HeightmapCubemap() = default;

    //Loads manifest.json + 6 face files from a directory. Returns false (and
    //logs to stderr) on any I/O / parse failure - sample() then returns 0.
    //After elevation succeeds, optionally loads the v8 baker's three companion
    //layers (surface_color, normal, cloud_offset). Any missing companion is
    //silently treated as "not present"; the renderer then binds a null SRV
    //and the shader falls back to the legacy uniform-tint / analytic-normal /
    //flat-cloud path.
    bool load(const std::filesystem::path& terrain_dir);

    bool   loaded()         const { return m_resolution > 0; }
    uint32_t resolution()   const { return m_resolution; }

    //Box-downsampled copy of one face for GPU upload. dst_resolution must be
    //a power-of-two divisor of resolution(). out buffer must have
    //dst_resolution * dst_resolution floats. Values are in KM (the bake's
    //native signed units - negative values dip below the analytic surface,
    //positive values rise above) so the shader's amplitude convention is
    //independent of CPU-side metres conversion. Returns false if not loaded
    //or arguments are invalid.
    bool downsample_face_km(uint8_t face, uint32_t dst_resolution, float* out) const;

    //IHeightmapSource. dir must be unit length; the result is the surface
    //offset along that direction in METRES (signed - mesh may dip below the
    //analytic surface in crater basins, but the atmosphere code clamps).
    float sample(const DVec3& dir_normalized, uint8_t lod) const override;

    //====================================
    //BAKER v8 COMPANION LAYERS
    //====================================
    //All three are independent of elevation - any combination may be absent.
    //The renderer queries `*_loaded()` to decide whether to bind real SRVs or
    //null fallbacks. Resolutions and pixel formats match the baker:
    //  surface_color: same resolution as elevation, 4 bytes/cell RGBA8.
    //  normal       : same resolution as elevation, 4 bytes/cell RGBA8.
    //  cloud_offset : fixed 256, 4 bytes/cell float32 km.
    bool     surface_color_loaded() const { return m_surface_color_resolution > 0; }
    uint32_t surface_color_resolution() const { return m_surface_color_resolution; }
    //Copy one face's RGBA8 pixels (downsampled if needed) into `out`, which
    //must have dst_resolution * dst_resolution * 4 bytes. The downsample is
    //a box average over the per-channel uint8 values; the result is still
    //RGBA8 (A is always 255).
    bool surface_color_face(uint8_t face, uint32_t dst_resolution, uint8_t* out) const;

    bool     normal_loaded() const { return m_normal_resolution > 0; }
    uint32_t normal_resolution() const { return m_normal_resolution; }
    bool normal_face(uint8_t face, uint32_t dst_resolution, uint8_t* out) const;

    bool     cloud_offset_loaded() const { return m_cloud_offset_resolution > 0; }
    uint32_t cloud_offset_resolution() const { return m_cloud_offset_resolution; }
    //Copy one face of cloud_offset as float32 km, source resolution (no
    //downsample - the cloud_offset bake is already small).
    bool cloud_offset_face(uint8_t face, float* out) const;

private:
    uint32_t                          m_resolution = 0;  // pixels per face (square)
    std::vector<std::vector<float>>   m_faces;           // 6 entries, each m_resolution^2 floats in km

    //Companion layers (baker v8). Vectors stay empty when not loaded.
    uint32_t                              m_surface_color_resolution = 0;
    std::vector<std::vector<std::uint8_t>> m_surface_color_faces;   // RGBA8 per cell

    uint32_t                              m_normal_resolution = 0;
    std::vector<std::vector<std::uint8_t>> m_normal_faces;          // RGBA8 per cell

    uint32_t                              m_cloud_offset_resolution = 0;
    std::vector<std::vector<float>>        m_cloud_offset_faces;    // km, float
};

} // namespace planet
