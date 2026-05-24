#include "passes/bake_pass.h"

#include "core/cubed_sphere.h"
#include "core/field.h"
#include "core/log.h"
#include "core/neighbor_table.h"
#include "core/param_registry.h"
#include "core/planet_state.h"
#include "gpu/cuda_check.h"
#include "gpu/device_buffer.h"

#include <nlohmann/json.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace pb {

namespace {

//====================================
//Catmull-Rom bicubic. Standard tension=0.5 weights. Sum is 1 for any t in
//[0, 1], so this preserves the source's mean elevation. The kernel calls
//cr_weight with the four offsets {-1, 0, 1, 2} relative to floor(f).
//====================================
__device__ __forceinline__ float cr_weight(float t, int offset) {
    float t2 = t * t;
    float t3 = t2 * t;
    switch (offset) {
        case -1: return -0.5f * t  +        t2 - 0.5f * t3;
        case  0: return  1.0f      - 2.5f * t2 + 1.5f * t3;
        case  1: return  0.5f * t  + 2.0f * t2 - 1.5f * t3;
        case  2: return            -0.5f * t2 + 0.5f * t3;
        default: return 0.0f;
    }
}

//Bicubic sample at (fi, fj) on one face. Reads a 4x4 cell window centered
//at floor((fi, fj)). The caller must have halo-exchanged the source field
//first - cells near a face boundary read into the halo, which only holds
//valid cross-face data after halo_exchange.
__device__ float bicubic_sample(const float* __restrict__ face_buf,
                                int face,
                                float fi, float fj,
                                int src_n, int src_stride, int src_halo) {
    int   i0 = static_cast<int>(floorf(fi));
    int   j0 = static_cast<int>(floorf(fj));
    float ti = fi - static_cast<float>(i0);
    float tj = fj - static_cast<float>(j0);

    int   lo = -src_halo;
    int   hi = src_n + src_halo - 1;
    float sum = 0.0f;

    #pragma unroll
    for (int dj = -1; dj <= 2; ++dj) {
        int j = j0 + dj;
        if (j < lo) j = lo;
        if (j > hi) j = hi;
        float wj = cr_weight(tj, dj);

        #pragma unroll
        for (int di = -1; di <= 2; ++di) {
            int i = i0 + di;
            if (i < lo) i = lo;
            if (i > hi) i = hi;
            float wi = cr_weight(ti, di);

            int idx = face * src_stride * src_stride
                    + (j + src_halo) * src_stride
                    + (i + src_halo);
            sum += face_buf[idx] * wi * wj;
        }
    }
    return sum;
}

//====================================
//Per-face upsample kernel. Output is one face of the cubemap at dst_n x
//dst_n, layout row-major. surface_km = bedrock_km + sediment_m * 0.001.
//Computed independently per face; cross-face seams are handled implicitly
//via the source halo (halo-exchanged before launch).
//====================================
__global__ void bake_face_kernel(
    float* __restrict__ dst,
    const float* __restrict__ bed,
    const float* __restrict__ sed,
    int src_n, int src_stride, int src_halo,
    int dst_n, int face) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_n || y >= dst_n) return;

    //Destination pixel center in face-local (u, v) coords in [-1, +1].
    float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(dst_n) * 2.0f - 1.0f;
    float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(dst_n) * 2.0f - 1.0f;

    //Source fractional cell index (centered at cell middles).
    float fi = (u + 1.0f) * 0.5f * static_cast<float>(src_n) - 0.5f;
    float fj = (v + 1.0f) * 0.5f * static_cast<float>(src_n) - 0.5f;

    float bed_v = bicubic_sample(bed, face, fi, fj, src_n, src_stride, src_halo);
    float sed_v = bicubic_sample(sed, face, fi, fj, src_n, src_stride, src_halo);
    dst[y * dst_n + x] = bed_v + sed_v * 0.001f;
}

//====================================
//Manifest. Single elevation layer; we'll grow this object in later passes
//when normals / sediment / etc. join the bake.
//====================================
bool write_manifest(const std::filesystem::path& dir, int dst_n,
                    float planet_radius_km) {
    using json = nlohmann::json;
    json m;
    m["version"]    = 1;
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

    std::ofstream out(dir / "manifest.json");
    if (!out) {
        PB_LOG_ERROR("bake", "cannot open manifest for writing in %s",
                     dir.string().c_str());
        return false;
    }
    out << m.dump(2);
    return static_cast<bool>(out);
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
    reg.declare_int  ("bake.elevation_resolution",   16384, 64, 32768,
                      "elevation resolution",
                      "Pixels per cube face for the elevation layer. Default "
                      "16384 matches PLAN section 7.11. Lower values are useful "
                      "for quick iteration / smoke tests.",
                      "px", o);
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

    const auto& g = state.grid();
    if (g.n <= 0) {
        PB_LOG_ERROR("bake", "PlanetState grid is empty (N=0)");
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

    //Cross-face bicubic needs valid halo cells (the 4x4 sample window steps
    //one cell outside the face's interior near edges). Without halo_exchange
    //the bake shows seams along every cube-face boundary.
    progress.stage("halo exchange");
    progress.fraction(0.0f);
    state.bedrock_elevation.halo_exchange(state.neighbors());
    state.sediment_thickness.halo_exchange(state.neighbors());
    CUDA_CHECK(cudaDeviceSynchronize());

    //One destination buffer reused across 6 faces. At 16384^2 that's 1 GB,
    //which is comfortable on any 8 GB+ card. Re-allocation across faces
    //would also work but wastes a few cudaMalloc calls.
    const std::size_t face_cells = static_cast<std::size_t>(dst_n) * static_cast<std::size_t>(dst_n);
    DeviceBuffer<float> dst_buf(face_cells);
    std::vector<float>  host_buf(face_cells);

    const int src_n      = g.n;
    const int src_stride = g.stride();
    const int src_halo   = CubedSphereGrid::HALO;

    dim3 block(16, 16, 1);
    dim3 gd((dst_n + block.x - 1) / block.x,
            (dst_n + block.y - 1) / block.y,
            1);

    for (int face = 0; face < 6; ++face) {
        char stage[64];
        std::snprintf(stage, sizeof(stage), "bake face %d / 6", face);
        progress.stage(stage);
        progress.fraction(static_cast<float>(face) / 6.0f);

        bake_face_kernel<<<gd, block>>>(
            dst_buf.data(),
            state.bedrock_elevation.data(),
            state.sediment_thickness.data(),
            src_n, src_stride, src_halo,
            dst_n, face);
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
    }

    //Pick up planet radius from bedrock if available so the manifest carries
    //the body size the elevations were synthesised for. Falls back to 6371
    //(Earth) when bedrock isn't in the pipeline.
    const float planet_radius_km = reg.has("bedrock.planet_radius_km")
                                     ? reg.get_float("bedrock.planet_radius_km")
                                     : 6371.0f;
    write_manifest(out_dir, dst_n, planet_radius_km);

    progress.fraction(1.0f);
    PB_LOG_INFO("bake",
                "wrote 6 x %dx%d float32 cube faces (%.1f GB) + manifest.json to %s",
                dst_n, dst_n,
                6.0 * static_cast<double>(face_cells) * 4.0 / (1024.0 * 1024.0 * 1024.0),
                out_dir.string().c_str());
}

}
