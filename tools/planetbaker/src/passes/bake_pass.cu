#include "passes/bake_pass.h"

#include "core/cubed_sphere.h"
#include "core/field.h"
#include "core/log.h"
#include "core/neighbor_table.h"
#include "core/param_registry.h"
#include "core/planet_state.h"
#include "gpu/cuda_check.h"
#include "gpu/device_buffer.h"
#include "passes/bedrock_noise.h"
#include "passes/hydraulic_erosion_pass.h"
#include "passes/impacts_pass.h"
#include "passes/surface_color_pass.h"
#include "passes/thermal_erosion_pass.h"

#include <nlohmann/json.hpp>
#include <stb_image_write.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace pb {

namespace {

//Above this resolution the full-pipeline bake (Bedrock + Impacts + Thermal
//on a temp PlanetState) does not fit a typical desktop GPU's VRAM. A
//BedrockOnly PlanetState at N=8192 with halo=2 is ~8.4 GB; 16384 would be
//~34 GB. Above the cap we drop to bedrock-only direct synthesis (which
//runs face-by-face into a 1 GB destination buffer and fits everywhere).
constexpr int kFullPipelineMaxN = 8192;

//Resolution of the per-face cloud-offset map. Fixed (not tied to bake.elevation_resolution)
//because the renderer just needs a smoothed elevation reference for the
//cloud base - 256 is plenty under bilinear filtering and keeps the on-disk
//set tiny (6 * 256 * 256 * 4 B = 1.5 MB total).
constexpr int kCloudOffsetFaceN = 256;

//====================================
//Compose-face kernel: copy a face's interior cells from the (halo-padded)
//PlanetState bedrock + sediment fields into a tight n*n destination buffer,
//computing elevation = bedrock_km + sediment_m * 0.001 on the fly. Used by
//the full-pipeline bake path to extract one face at a time for disk write.
//====================================
__global__ void compose_face_kernel(
    float* __restrict__ dst,
    const float* __restrict__ bedrock_state,
    const float* __restrict__ sediment_state,
    int n, int stride, int halo,
    int face) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n) return;

    int src_idx = face * stride * stride + (j + halo) * stride + (i + halo);
    float bed = bedrock_state[src_idx];
    float sed = sediment_state[src_idx];
    dst[static_cast<std::size_t>(j) * n + i] = bed + sed * 0.001f;
}

//====================================
//Compose-color-face kernel: copy a face's interior cells from the
//surface_color Field<float4> into a tight n*n*3 uint8 RGB buffer, ready for
//PNG / raw-RGB disk write. Alpha is dropped; we currently only use it as a
//padding channel inside the float4. Clamped + quantised to 0..255.
//====================================
__global__ void compose_color_face_kernel(
    std::uint8_t* __restrict__ dst,
    const float4* __restrict__ color_state,
    int n, int stride, int halo,
    int face) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n) return;

    int src_idx = face * stride * stride + (j + halo) * stride + (i + halo);
    float4 c = color_state[src_idx];
    float r = c.x; if (r < 0.0f) r = 0.0f; if (r > 1.0f) r = 1.0f;
    float g = c.y; if (g < 0.0f) g = 0.0f; if (g > 1.0f) g = 1.0f;
    float b = c.z; if (b < 0.0f) b = 0.0f; if (b > 1.0f) b = 1.0f;
    std::size_t dst_idx = (static_cast<std::size_t>(j) * n + i) * 3;
    dst[dst_idx + 0] = static_cast<std::uint8_t>(r * 255.0f + 0.5f);
    dst[dst_idx + 1] = static_cast<std::uint8_t>(g * 255.0f + 0.5f);
    dst[dst_idx + 2] = static_cast<std::uint8_t>(b * 255.0f + 0.5f);
}

//====================================
//Compose-cloud-offset-face kernel: block-average the elevation (bedrock_km +
//sediment_m * 0.001, in km) over (src_n / dst_n)^2 source cells to produce
//a low-resolution per-face map of "general terrain elevation".
//
//Use case: the renderer composes the cloud base as cloud_offset(p) +
//cloud_height_above_terrain, where p is a sphere direction. The block-
//average naturally drops everything finer than ~(src_n / dst_n) cells, so
//sharp summits don't push the local cloud base up; instead they poke
//through clouds rendered at the smoothed altitude.
//
//Output cell (i, j) in [0, dst_n)^2 covers source cells in
//  [i * block, (i + 1) * block)  x  [j * block, (j + 1) * block)
//where block = src_n / dst_n. Caller is expected to pick dst_n so that
//src_n is an integer multiple; otherwise the right / bottom output edges
//cover a slightly different source area than the rest (we don't crash but
//the block boundaries shift a little).
//====================================
__global__ void compose_cloud_offset_face_kernel(
    float* __restrict__ dst,
    const float* __restrict__ bedrock_state,
    const float* __restrict__ sediment_state,
    int src_n, int src_stride, int src_halo,
    int dst_n,
    int face) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= dst_n || j >= dst_n) return;

    int block  = src_n / dst_n;
    if (block < 1) block = 1;
    int src_i0 = i * block;
    int src_j0 = j * block;
    if (src_i0 + block > src_n) src_i0 = src_n - block;
    if (src_j0 + block > src_n) src_j0 = src_n - block;

    double sum = 0.0;
    int    count = 0;
    for (int dj = 0; dj < block; ++dj) {
        for (int di = 0; di < block; ++di) {
            int si  = src_i0 + di;
            int sj  = src_j0 + dj;
            int idx = face * src_stride * src_stride
                    + (sj + src_halo) * src_stride
                    + (si + src_halo);
            float bed = bedrock_state[idx];
            float sed = sediment_state[idx];
            sum += static_cast<double>(bed) + static_cast<double>(sed) * 0.001;
            ++count;
        }
    }

    dst[static_cast<std::size_t>(j) * dst_n + i] = (count > 0)
        ? static_cast<float>(sum / count)
        : 0.0f;
}

//====================================
//Compose-normal-face kernel: tangent-space normal from central-difference of
//the surface height (bedrock_km*1000 + sediment_m, in metres). The result is
//(-dh/du_m, -dh/dv_m, 1) normalised, then biased to [0,1] and quantised to
//8-bit RGB, OpenGL convention (G points "up" in v).
//
//`inv_2dx_m` is 1 / (2 * cell_arc_m) and is the conversion from "metres of
//height difference between two cells two-apart" to "rise over run". We use
//the cell arc length at face centre as a single per-face approximation; the
//equiangular projection stretches by ~sqrt(2) at face corners, so corners
//see slightly weaker normals than they would in a fully metric mapping.
//Good enough for far-detail use.
//
//Reads bedrock + sediment with halo (cross-face neighbours via the standard
//halo_exchange done at the call site).
//====================================
__global__ void compose_normal_face_kernel(
    std::uint8_t* __restrict__ dst,
    const float* __restrict__ bedrock_state,
    const float* __restrict__ sediment_state,
    int n, int stride, int halo,
    int face,
    float inv_2dx_m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n) return;

    int idx_c  = face * stride * stride + (j + halo) * stride + (i + halo);
    int idx_pi = idx_c + 1;
    int idx_mi = idx_c - 1;
    int idx_pj = idx_c + stride;
    int idx_mj = idx_c - stride;

    float h_pi = bedrock_state[idx_pi] * 1000.0f + sediment_state[idx_pi];
    float h_mi = bedrock_state[idx_mi] * 1000.0f + sediment_state[idx_mi];
    float h_pj = bedrock_state[idx_pj] * 1000.0f + sediment_state[idx_pj];
    float h_mj = bedrock_state[idx_mj] * 1000.0f + sediment_state[idx_mj];

    float gx = (h_pi - h_mi) * inv_2dx_m;   //rise / run, dimensionless
    float gy = (h_pj - h_mj) * inv_2dx_m;

    float nx = -gx;
    float ny = -gy;
    float nz =  1.0f;
    float inv_len = rsqrtf(nx * nx + ny * ny + nz * nz);
    nx *= inv_len;
    ny *= inv_len;
    nz *= inv_len;

    //Bias [-1, 1] -> [0, 1] -> [0, 255] (OpenGL/Blender convention; +G
    //points up the v axis on the face). Most renderers can flip a channel
    //at sample time if a DirectX-style convention is wanted.
    float r = nx * 0.5f + 0.5f;
    float g = ny * 0.5f + 0.5f;
    float b = nz * 0.5f + 0.5f;
    if (r < 0.0f) r = 0.0f; if (r > 1.0f) r = 1.0f;
    if (g < 0.0f) g = 0.0f; if (g > 1.0f) g = 1.0f;
    if (b < 0.0f) b = 0.0f; if (b > 1.0f) b = 1.0f;

    std::size_t dst_idx = (static_cast<std::size_t>(j) * n + i) * 3;
    dst[dst_idx + 0] = static_cast<std::uint8_t>(r * 255.0f + 0.5f);
    dst[dst_idx + 1] = static_cast<std::uint8_t>(g * 255.0f + 0.5f);
    dst[dst_idx + 2] = static_cast<std::uint8_t>(b * 255.0f + 0.5f);
}

//====================================
//Preview PNG. Box-downsample one face's float-KM elevation buffer to
//<= kPreviewSize, normalize to [0, 255] using per-face min/max, and write
//as a 1-channel PNG so the user can eyeball the bake without a custom .r32
//viewer. Pure host code; no CUDA. Returns false on any I/O failure.
constexpr int kPreviewSize = 4096;

bool write_preview_png(const std::filesystem::path& filepath,
                       const float* src, int src_n) {
    if (src == nullptr || src_n <= 0) return false;

    //Never upsample - if the bake is below 4k the preview matches the bake.
    int preview_n = src_n < kPreviewSize ? src_n : kPreviewSize;

    std::vector<float> downsampled(static_cast<std::size_t>(preview_n) * preview_n);
    if (preview_n == src_n) {
        std::memcpy(downsampled.data(), src,
                    downsampled.size() * sizeof(float));
    } else {
        //Box average over the source block covering each preview texel.
        //Integer math when src_n is a multiple of preview_n (the common 16k/4k
        //and 8k/4k cases). The double scale keeps boundaries correct for
        //non-multiple resolutions; we just lose half a pixel at the edge.
        const double scale = static_cast<double>(src_n)
                           / static_cast<double>(preview_n);
        for (int j = 0; j < preview_n; ++j) {
            const int sy0 = static_cast<int>(j * scale);
            const int sy1 = static_cast<int>((j + 1) * scale);
            for (int i = 0; i < preview_n; ++i) {
                const int sx0 = static_cast<int>(i * scale);
                const int sx1 = static_cast<int>((i + 1) * scale);
                double acc = 0.0;
                int    count = 0;
                for (int y = sy0; y < sy1; ++y) {
                    const std::size_t row = static_cast<std::size_t>(y) * src_n;
                    for (int x = sx0; x < sx1; ++x) {
                        acc += src[row + x];
                        ++count;
                    }
                }
                downsampled[static_cast<std::size_t>(j) * preview_n + i] =
                    static_cast<float>(count > 0 ? acc / count : 0.0);
            }
        }
    }

    float min_v = downsampled[0];
    float max_v = downsampled[0];
    for (float v : downsampled) {
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
    }
    const float range = std::max(max_v - min_v, 1.0e-9f);

    std::vector<unsigned char> bytes(downsampled.size());
    for (std::size_t k = 0; k < downsampled.size(); ++k) {
        const float t = (downsampled[k] - min_v) / range;
        const int   x = static_cast<int>(std::lround(t * 255.0f));
        bytes[k] = static_cast<unsigned char>(x < 0 ? 0 : (x > 255 ? 255 : x));
    }

    const int written = stbi_write_png(filepath.string().c_str(),
                                       preview_n, preview_n,
                                       1, bytes.data(), preview_n);
    if (written == 0) return false;
    PB_LOG_INFO("bake",
                "preview PNG: %dx%d, elevation range [%.3f, %.3f] km -> %s",
                preview_n, preview_n, min_v, max_v,
                filepath.string().c_str());
    return true;
}

//====================================
//Manifest. Single elevation layer; we'll grow this object in later passes
//when normals / sediment / etc. join the bake.
//====================================
bool write_manifest(const std::filesystem::path& dir, int dst_n,
                    float planet_radius_km, bool include_color) {
    using json = nlohmann::json;
    json m;
    m["version"]    = 2;
    m["projection"] = "cubed_sphere_equiangular";
    m["face_order"] = json::array({"+X", "-X", "+Y", "-Y", "+Z", "-Z"});
    m["planet_radius_km"] = planet_radius_km;

    json layer;
    layer["filename_pattern"] = "elevation_face{N}.r32";
    layer["resolution"]       = dst_n;
    layer["channels"]         = 1;
    layer["element_type"]     = "float32";
    layer["byte_order"]       = "little_endian";
    layer["row_major"]        = true;
    layer["units"]            = "km";
    layer["description"]      = "Surface elevation (bedrock + sediment) above mean radius";
    m["layers"]["elevation"]  = layer;

    if (include_color) {
        json color_layer;
        color_layer["filename_pattern"] = "surface_color_face{N}.png";
        color_layer["resolution"]       = dst_n;
        color_layer["channels"]         = 3;
        color_layer["element_type"]     = "uint8";
        color_layer["color_space"]      = "linear";
        color_layer["description"]      = "Mars-style surface tint (RGB) from SurfaceColorPass";
        m["layers"]["surface_color"]    = color_layer;

        json normal_layer;
        normal_layer["filename_pattern"] = "normal_face{N}.png";
        normal_layer["resolution"]       = dst_n;
        normal_layer["channels"]         = 3;
        normal_layer["element_type"]     = "uint8";
        normal_layer["color_space"]      = "linear";
        normal_layer["encoding"]         = "tangent_space_unorm";
        normal_layer["convention"]       = "opengl";   // +Y up the v axis
        normal_layer["description"]      = "Tangent-space normal map from heightmap "
                                           "(bedrock + sediment), face-local +u, +v, "
                                           "outward; biased [-1,1] -> [0,1].";
        m["layers"]["normal"]            = normal_layer;

        json cloud_layer;
        cloud_layer["filename_pattern"] = "cloud_offset_face{N}.r32";
        cloud_layer["resolution"]       = kCloudOffsetFaceN;
        cloud_layer["channels"]         = 1;
        cloud_layer["element_type"]     = "float32";
        cloud_layer["byte_order"]       = "little_endian";
        cloud_layer["row_major"]        = true;
        cloud_layer["units"]            = "km";
        cloud_layer["description"]      = "Block-averaged surface elevation (km) for "
                                          "cloud base offset; renderer should sample "
                                          "this with bilinear filtering and add a "
                                          "constant cloud_thickness above it. Sharp "
                                          "peaks above the smoothed value naturally "
                                          "poke through the cloud layer.";
        m["layers"]["cloud_offset"]     = cloud_layer;
    }

    //Serialize FIRST, then write the resulting string as a single block.
    //Doing dump() inside the operator<< had the JSON object live across the
    //write and we observed manifests landing on disk as 1700 bytes of NUL
    //(file size matched what dump would have produced but the content never
    //made it through). Pre-serialising + binary-mode write + explicit
    //flush+close removes any ambiguity.
    const std::string text = m.dump(2);
    std::ofstream out(dir / "manifest.json", std::ios::binary | std::ios::trunc);
    if (!out) {
        PB_LOG_ERROR("bake", "cannot open manifest for writing in %s",
                     dir.string().c_str());
        return false;
    }
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    out.flush();
    out.close();
    if (!out) {
        PB_LOG_ERROR("bake", "manifest write failed in %s (%zu bytes intended)",
                     dir.string().c_str(), text.size());
        return false;
    }
    PB_LOG_INFO("bake", "manifest.json written (%zu bytes) to %s",
                text.size(), dir.string().c_str());
    return true;
}

}

//====================================
//Pass interface
//====================================

void BakePass::declare_params(ParamRegistry& reg) const {
    const char* o = name();
    reg.declare_bool ("bake.enabled",                false,
                      "enabled",
                      "When false (default) the bake is a no-op. Flip on, hit "
                      "Run, and the pipeline writes 6 raw float32 cube faces "
                      "plus manifest.json under ./out/. A 16k bake produces "
                      "about 6 GB of disk and ~30 s of GPU work, hence opt-in.",
                      "", o);
    reg.declare_int  ("bake.elevation_resolution",    8192, 64, 32768,
                      "elevation resolution",
                      "Pixels per cube face for the elevation layer. "
                      "<= 8192: FULL pipeline (bedrock noise + impacts + "
                      "thermal erosion) re-synthesised at the bake resolution "
                      "via a temp PlanetState. > 8192: bedrock-only direct "
                      "synthesis per face (thermal skipped to stay within VRAM; "
                      "impacts run via bake_impacts_face). Default 8192 matches "
                      "the renderer's GPU heightmap (Renderer_Pipeline.cpp's "
                      "TERRAIN_HEIGHTMAP_GPU_RESOLUTION) so the upload is a "
                      "lossless 1:1 transfer rather than a 4x downsample, and "
                      "thermal erosion runs because we stay under the full-"
                      "pipeline cap. 16384 is the absolute-max bedrock detail "
                      "but drops thermal erosion and gets downsampled 4x on "
                      "upload anyway.",
                      "px", o);
    reg.declare_int  ("bake.preview_face",           0, 0, 5,
                      "preview face",
                      "Which cube face (0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z) "
                      "is dumped as a 4kx4k normalized grayscale PNG alongside "
                      "the .r32 binaries. The PNG is box-downsampled from the "
                      "bake resolution and remapped to [0, 255] using the "
                      "face's min/max so you can sanity-check the output "
                      "without a custom .r32 viewer.",
                      "", o);
}

void BakePass::run(PlanetState& state, const ParamRegistry& reg, ProgressSink& progress) {
    const bool enabled = reg.get_bool("bake.enabled");
    if (!enabled) {
        PB_LOG_INFO("bake", "skipped (bake.enabled is false)");
        progress.stage("bake skipped");
        progress.fraction(1.0f);
        return;
    }

    const int dst_n = reg.get_int("bake.elevation_resolution");
    if (dst_n < 64) {
        PB_LOG_ERROR("bake", "elevation_resolution %d below minimum 64", dst_n);
        return;
    }

    //Output directory comes from the BakePass constructor (default ./out
    //relative to the process cwd; tests pass a scratch path). String params
    //in the registry would be the cleaner long-term answer.
    const std::filesystem::path out_dir = output_dir_;
    std::error_code ec;
    std::filesystem::create_directories(out_dir, ec);
    if (ec) {
        PB_LOG_ERROR("bake", "cannot create %s: %s",
                     out_dir.string().c_str(), ec.message().c_str());
        return;
    }

    //Two synthesis paths, gated on whether dst_n fits a full temp
    //PlanetState in VRAM:
    //
    //  * dst_n <= kFullPipelineMaxN (~8 GB at 8192): allocate a temp
    //    PlanetState at dst_n, run BedrockNoisePass + ImpactsPass +
    //    ThermalErosionPass + HydraulicErosionPass on it, then copy the
    //    per-face interior cells out as elevation = bedrock + sediment *
    //    0.001. The bake carries ALL the pipeline's contributions
    //    (craters, talus, ejecta, hydraulic-carved canyons) at the output
    //    resolution.
    //
    //  * dst_n >  kFullPipelineMaxN (would be >8 GB just for bedrock+crust
    //    fields at 16k): drop to bedrock-only direct synthesis via
    //    bake_bedrock_face per face. ~1 GB destination buffer, ~30 s, 16k
    //    of bedrock-noise detail; impacts + thermal are NOT present.
    //
    //The user picks the trade-off by choosing bake.elevation_resolution.
    const std::size_t face_cells = static_cast<std::size_t>(dst_n) * static_cast<std::size_t>(dst_n);
    const float planet_radius_km = reg.has("bedrock.planet_radius_km")
                                     ? reg.get_float("bedrock.planet_radius_km")
                                     : 6371.0f;

    //One face is dumped as a 4kx4k normalized grayscale PNG so the user can
    //sanity-check the bake without a custom .r32 viewer. clamp to [0, 5] in
    //case the registry was hand-edited.
    int preview_face = reg.get_int("bake.preview_face");
    if (preview_face < 0) preview_face = 0;
    if (preview_face > 5) preview_face = 5;

    if (dst_n <= kFullPipelineMaxN) {
        //----- FULL PIPELINE PATH -----
        //Allocate a fresh PlanetState sized to the bake. BedrockOnly is the
        //field set the three passes touch (bedrock_elevation,
        //sediment_thickness, crust fields).
        progress.stage("bake: allocating temp state");
        progress.fraction(0.0f);
        PlanetState temp_state(dst_n, PlanetFieldSet::BedrockOnly);

        BedrockNoisePass        bedrock;
        ImpactsPass             impacts;
        ThermalErosionPass      thermal;
        HydraulicErosionPass    hydraulic;
        SurfaceColorPass        color;

        progress.stage("bake: bedrock noise");
        bedrock.run(temp_state, reg, progress);

        progress.stage("bake: impacts");
        impacts.run(temp_state, reg, progress);

        progress.stage("bake: thermal erosion");
        thermal.run(temp_state, reg, progress);

        progress.stage("bake: hydraulic erosion");
        hydraulic.run(temp_state, reg, progress);

        progress.stage("bake: surface color");
        color.run(temp_state, reg, progress);

        //Normal compose needs the latest bedrock + sediment values readable
        //THROUGH the cross-face halo. Pass-pipeline halos may be stale after
        //the colour pass (it doesn't modify either field but doesn't refresh
        //their halos either), so do an explicit exchange before we sample
        //h(+i, -i, +j, -j) in compose_normal_face_kernel.
        temp_state.bedrock_elevation.halo_exchange(temp_state.neighbors());
        temp_state.sediment_thickness.halo_exchange(temp_state.neighbors());

        //Per-face copy + compose + write. We reuse the device buffers across
        //the six faces: one float buffer for elevation, two uint8 RGB
        //buffers for surface_color and normal.
        DeviceBuffer<float>        dst_buf(face_cells);
        std::vector<float>         host_buf(face_cells);
        DeviceBuffer<std::uint8_t> dst_color_buf(face_cells * 3);
        std::vector<std::uint8_t>  host_color_buf(face_cells * 3);
        DeviceBuffer<std::uint8_t> dst_normal_buf(face_cells * 3);
        std::vector<std::uint8_t>  host_normal_buf(face_cells * 3);

        //Cloud-offset map: low-resolution per-face elevation average. Each
        //cell averages a (dst_n / kCloudOffsetFaceN)^2 patch of the bake (at
        //8 k that's a 32x32 box ~ 38 km wavelength of smoothing on Earth),
        //so individual mountain peaks drop out but continent-scale relief
        //survives. Renderer bilinearly filters for the actual lookup.
        const std::size_t cloud_face_cells = static_cast<std::size_t>(kCloudOffsetFaceN)
                                           * static_cast<std::size_t>(kCloudOffsetFaceN);
        DeviceBuffer<float> dst_cloud_buf(cloud_face_cells);
        std::vector<float>  host_cloud_buf(cloud_face_cells);

        //Single-anchor approximation of cell arc length: equiangular face
        //covers pi/2 rad of arc, so cell_arc_m at face centre is
        //(pi * R_m) / (2 * dst_n). Used by compose_normal_face_kernel to
        //turn finite-difference height (m) into a dimensionless slope.
        const float planet_radius_m = planet_radius_km * 1000.0f;
        const float cell_arc_m      = (3.14159265358979323846f * planet_radius_m)
                                    / (2.0f * static_cast<float>(dst_n));
        const float inv_2dx_m       = 1.0f / (2.0f * cell_arc_m);

        dim3 cloud_block(16, 16, 1);
        dim3 cloud_gd((kCloudOffsetFaceN + cloud_block.x - 1) / cloud_block.x,
                      (kCloudOffsetFaceN + cloud_block.y - 1) / cloud_block.y,
                      1);

        const auto& g      = temp_state.grid();
        const int   n      = g.n;
        const int   stride = g.stride();
        const int   halo   = CubedSphereGrid::HALO;

        dim3 block(16, 16, 1);
        dim3 gd((n + block.x - 1) / block.x,
                (n + block.y - 1) / block.y,
                1);

        for (int face = 0; face < 6; ++face) {
            char stage[64];
            std::snprintf(stage, sizeof(stage), "bake: write face %d / 6", face);
            progress.stage(stage);
            progress.fraction(static_cast<float>(face) / 6.0f);

            compose_face_kernel<<<gd, block>>>(
                dst_buf.data(),
                temp_state.bedrock_elevation.data(),
                temp_state.sediment_thickness.data(),
                n, stride, halo,
                face);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            dst_buf.download(host_buf);

            char fname[64];
            std::snprintf(fname, sizeof(fname), "elevation_face%d.r32", face);
            std::filesystem::path filepath = out_dir / fname;
            std::ofstream out(filepath, std::ios::binary);
            if (!out) {
                PB_LOG_ERROR("bake", "cannot open %s for writing", filepath.string().c_str());
                continue;
            }
            out.write(reinterpret_cast<const char*>(host_buf.data()),
                      static_cast<std::streamsize>(host_buf.size() * sizeof(float)));
            if (!out) {
                PB_LOG_ERROR("bake", "write failed: %s", filepath.string().c_str());
            }

            //Surface-color face: PNG so it's eyeballable directly from disk.
            compose_color_face_kernel<<<gd, block>>>(
                dst_color_buf.data(),
                temp_state.surface_color.data(),
                n, stride, halo,
                face);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            dst_color_buf.download(host_color_buf);

            char color_fname[64];
            std::snprintf(color_fname, sizeof(color_fname),
                          "surface_color_face%d.png", face);
            std::filesystem::path color_path = out_dir / color_fname;
            if (stbi_write_png(color_path.string().c_str(),
                               dst_n, dst_n, 3,
                               host_color_buf.data(), dst_n * 3) == 0) {
                PB_LOG_ERROR("bake", "color PNG write failed: %s",
                             color_path.string().c_str());
            }

            //Normal face: tangent-space RGB normal map computed from the
            //heightmap, sharing the heightmap's halo for clean seams.
            compose_normal_face_kernel<<<gd, block>>>(
                dst_normal_buf.data(),
                temp_state.bedrock_elevation.data(),
                temp_state.sediment_thickness.data(),
                n, stride, halo,
                face,
                inv_2dx_m);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            dst_normal_buf.download(host_normal_buf);

            char normal_fname[64];
            std::snprintf(normal_fname, sizeof(normal_fname),
                          "normal_face%d.png", face);
            std::filesystem::path normal_path = out_dir / normal_fname;
            if (stbi_write_png(normal_path.string().c_str(),
                               dst_n, dst_n, 3,
                               host_normal_buf.data(), dst_n * 3) == 0) {
                PB_LOG_ERROR("bake", "normal PNG write failed: %s",
                             normal_path.string().c_str());
            }

            //Cloud-offset face: block-averaged elevation, written as raw
            //float32 km matching the elevation layer's encoding.
            compose_cloud_offset_face_kernel<<<cloud_gd, cloud_block>>>(
                dst_cloud_buf.data(),
                temp_state.bedrock_elevation.data(),
                temp_state.sediment_thickness.data(),
                n, stride, halo,
                kCloudOffsetFaceN,
                face);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            dst_cloud_buf.download(host_cloud_buf);

            char cloud_fname[64];
            std::snprintf(cloud_fname, sizeof(cloud_fname),
                          "cloud_offset_face%d.r32", face);
            std::filesystem::path cloud_path = out_dir / cloud_fname;
            std::ofstream cloud_out(cloud_path, std::ios::binary);
            if (!cloud_out) {
                PB_LOG_ERROR("bake", "cannot open %s for writing",
                             cloud_path.string().c_str());
            } else {
                cloud_out.write(reinterpret_cast<const char*>(host_cloud_buf.data()),
                                static_cast<std::streamsize>(host_cloud_buf.size() * sizeof(float)));
                if (!cloud_out) {
                    PB_LOG_ERROR("bake", "cloud offset write failed: %s",
                                 cloud_path.string().c_str());
                }
            }

            if (face == preview_face) {
                char png_name[64];
                std::snprintf(png_name, sizeof(png_name), "preview_face%d.png", face);
                write_preview_png(out_dir / png_name, host_buf.data(), dst_n);
            }
        }

        write_manifest(out_dir, dst_n, planet_radius_km, /*include_color=*/true);
        progress.fraction(1.0f);
        PB_LOG_INFO("bake",
                    "FULL pipeline bake: 6 x %dx%d float32 elev + RGB color + RGB normal "
                    "+ 6 x %dx%d cloud_offset (%.1f GB elev) - impacts + thermal + "
                    "hydraulic + color -> %s",
                    dst_n, dst_n, kCloudOffsetFaceN, kCloudOffsetFaceN,
                    6.0 * static_cast<double>(face_cells) * 4.0 / (1024.0 * 1024.0 * 1024.0),
                    out_dir.string().c_str());
        return;
    }

    //----- BEDROCK + IMPACTS DIRECT-SYNTHESIS PATH (v4) -----
    //At dst_n > kFullPipelineMaxN a full PlanetState doesn't fit VRAM, so
    //we synthesise bedrock noise directly per face into a 1 GB destination
    //buffer (~30 s for 16k). v4: impacts (craters) now stamp onto the same
    //per-face buffer via bake_impacts_face, so 16k bakes finally carry
    //crater detail. Thermal erosion is still skipped at this resolution
    //(it needs a neighbour-aware halo PlanetState the direct path doesn't
    //allocate); drop bake.elevation_resolution to 8192 or lower to get it.
    PB_LOG_WARN("bake",
                "elevation_resolution %d > %d - thermal erosion skipped to "
                "stay in VRAM; bedrock noise + impacts only. Lower the "
                "resolution to include thermal.",
                dst_n, kFullPipelineMaxN);

    const BedrockNoiseParams P  = load_bedrock_params(reg);
    const ImpactsParams      IP = load_impacts_params(reg);

    //Sample the crater list ONCE before the face loop - all 6 faces stamp
    //the same population, just sampled on whichever direction lands in the
    //face's cells. Upload to device once too; the per-face kernel reads
    //from a stable device-side array.
    std::vector<Crater> craters = sample_craters(IP);
    int basin_count = 0;
    int multi_count = 0;
    for (const auto& c : craters) {
        if (c.morphology == MorphBasin)     ++basin_count;
        if (c.morphology == MorphMultiring) ++multi_count;
    }
    PB_LOG_INFO("bake", "impacts: %zu craters (basins=%d, multi-ring=%d)",
                craters.size(), basin_count, multi_count);

    DeviceBuffer<Crater> d_craters(craters.empty() ? 1 : craters.size());
    if (!craters.empty()) d_craters.upload(craters);

    DeviceBuffer<float> dst_buf(face_cells);
    std::vector<float>  host_buf(face_cells);

    for (int face = 0; face < 6; ++face) {
        char stage[64];
        std::snprintf(stage, sizeof(stage), "bake face %d / 6 (bedrock+impacts)", face);
        progress.stage(stage);
        progress.fraction(static_cast<float>(face) / 6.0f);

        bake_bedrock_face(face, dst_n, P, dst_buf.data());
        if (!craters.empty()) {
            bake_impacts_face(face, dst_n,
                              d_craters.data(),
                              static_cast<int>(craters.size()),
                              planet_radius_km,
                              dst_buf.data());
        }
        dst_buf.download(host_buf);

        char fname[64];
        std::snprintf(fname, sizeof(fname), "elevation_face%d.r32", face);
        std::filesystem::path filepath = out_dir / fname;
        std::ofstream out(filepath, std::ios::binary);
        if (!out) {
            PB_LOG_ERROR("bake", "cannot open %s for writing", filepath.string().c_str());
            continue;
        }
        out.write(reinterpret_cast<const char*>(host_buf.data()),
                  static_cast<std::streamsize>(host_buf.size() * sizeof(float)));
        if (!out) {
            PB_LOG_ERROR("bake", "write failed: %s", filepath.string().c_str());
        }

        if (face == preview_face) {
            char png_name[64];
            std::snprintf(png_name, sizeof(png_name), "preview_face%d.png", face);
            write_preview_png(out_dir / png_name, host_buf.data(), dst_n);
        }
    }

    write_manifest(out_dir, dst_n, planet_radius_km, /*include_color=*/false);
    progress.fraction(1.0f);
    PB_LOG_INFO("bake",
                "BEDROCK+IMPACTS bake: 6 x %dx%d float32 (%.1f GB) -> %s",
                dst_n, dst_n,
                6.0 * static_cast<double>(face_cells) * 4.0 / (1024.0 * 1024.0 * 1024.0),
                out_dir.string().c_str());
}

}
